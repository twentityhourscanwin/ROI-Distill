# labeldistill/refine_head/target_assigner/quality_aware_mask.py
# 论文 Quality-Aware Mask Weight Assignment 的实现
# 三级分层权重:
#   高质量匹配: k = 1
#   中等质量匹配: k = w_l + (1 - w_l) * s_roi, 范围 [w_l, 1]
#   未匹配: k = w_u * (1 - r / R_max)^2, 范围 [0, w_u]
#   约束: w_l >= w_u
import torch
import torch.nn as nn
from mmdet3d.models.utils.gaussian import draw_heatmap_gaussian, gaussian_radius


class QualityAwareMaskGenerator(nn.Module):

    SMALL_CLASSES = {5, 6, 8, 9}  # barrier, motorcycle, pedestrian, traffic_cone

    def __init__(self, w_l=0.6, w_u=0.4, r_max=50.0, boost_small_medium=False):
        super().__init__()
        assert w_l >= w_u, f"w_l ({w_l}) must be >= w_u ({w_u})"
        assert 0 < w_u <= w_l < 1, f"Need 0 < w_u <= w_l < 1, got w_l={w_l}, w_u={w_u}"

        self.w_l = w_l
        self.w_u = w_u
        self.r_max = r_max
        self.boost_small_medium = boost_small_medium

        grid_size = [1024, 1024, 40]
        voxel_size = [0.1, 0.1, 0.2]
        out_size_factor = 8
        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.out_factor = out_size_factor
        self.gaussian_overlap = 0.1
        self.min_radius = 2
        self.feature_map_size = (
            grid_size[0] // out_size_factor,
            grid_size[1] // out_size_factor,
        )  # (W, H) = (128, 128)

    def _compute_high_weight(self):
        """高质量匹配: k = 1"""
        return 1.0

    def _compute_medium_weight(self, s_roi):
        """中等质量匹配: k = w_l + (1 - w_l) * s_roi"""
        return self.w_l + (1.0 - self.w_l) * s_roi

    def _compute_unmatched_weight(self, gt_boxes):
        """未匹配: k = w_u * (1 - r / R_max)^2, 当 r >= R_max 时 k = 0"""
        r = torch.sqrt(gt_boxes[:, 0] ** 2 + gt_boxes[:, 1] ** 2)
        ratio = torch.clamp(1.0 - r / self.r_max, min=0.0)
        return self.w_u * ratio ** 2

    def _draw_boxes_to_mask(self, boxes, mask, center_values,
                            pc_range, voxel_size, out_factor,
                            feature_map_size, gaussian_overlap, min_radius):
        num_boxes = len(boxes)
        feat_width, feat_height = feature_map_size
        heatmap = mask[0]

        for k in range(num_boxes):
            width = boxes[k][3]
            length = boxes[k][4]

            width_fm = width / voxel_size[0] / out_factor
            length_fm = length / voxel_size[1] / out_factor

            if width_fm.item() <= 0 or length_fm.item() <= 0:
                continue
            if width_fm.item() > 1000 or length_fm.item() > 1000:
                continue

            radius_tensor = gaussian_radius(
                (length_fm, width_fm), min_overlap=gaussian_overlap)
            radius = max(min_radius, int(radius_tensor.item()))

            x, y = boxes[k][0], boxes[k][1]
            coor_x = (x - pc_range[0]) / voxel_size[0] / out_factor
            coor_y = (y - pc_range[1]) / voxel_size[1] / out_factor

            center_int = torch.stack([coor_x, coor_y], dim=0).to(torch.int32)

            if not (0 <= center_int[0] < feat_width and
                    0 <= center_int[1] < feat_height):
                continue

            cv = center_values[k].item() if torch.is_tensor(center_values[k]) else center_values[k]
            if cv <= 0:
                continue
            heatmap = draw_heatmap_gaussian(heatmap, center_int, radius, k=cv)

        return mask

    def generate_mask(self, matched_results, batch_size):
        feat_W, feat_H = self.feature_map_size

        refined_high_gts = matched_results['refined_high_quality_gt']
        medium_gts = matched_results['medium_quality_gt']
        medium_scores = matched_results['medium_quality_roi_scores']
        unmatch_near = matched_results['unmatched_gt_near']
        unmatch_mid = matched_results['unmatched_gt_medium']
        unmatch_far = matched_results['unmatched_gt_far']

        mask_list = []

        for idx in range(batch_size):
            device = refined_high_gts[idx].device
            pc_range_dev = self.pc_range.to(device)
            voxel_size_dev = self.voxel_size.to(device)

            mask = torch.zeros((1, feat_H, feat_W), device=device, dtype=torch.float32)

            # 1. 高质量匹配 GT: k = 1
            cur_high_gts = refined_high_gts[idx]
            if len(cur_high_gts) > 0:
                high_weights = torch.ones(len(cur_high_gts), device=device)
                mask = self._draw_boxes_to_mask(
                    boxes=cur_high_gts[:, :7], mask=mask,
                    center_values=high_weights,
                    pc_range=pc_range_dev, voxel_size=voxel_size_dev,
                    out_factor=self.out_factor, feature_map_size=self.feature_map_size,
                    gaussian_overlap=self.gaussian_overlap, min_radius=self.min_radius)

            # 2. 中等质量匹配 GT
            cur_med_gts = medium_gts[idx]
            cur_med_scores = medium_scores[idx]
            if len(cur_med_gts) > 0:
                if self.boost_small_medium:
                    gt_classes = cur_med_gts[:, -1].long()
                    is_small = torch.zeros(len(gt_classes), dtype=torch.bool, device=device)
                    for cls_id in self.SMALL_CLASSES:
                        is_small |= (gt_classes == cls_id)

                    med_weights = self._compute_medium_weight(cur_med_scores)
                    if is_small.any():
                        boosted = cur_med_scores[is_small] / 1.0 * (1 - self.w_l)
                        med_weights[is_small] = torch.clamp(
                            med_weights[is_small] + boosted, max=1.0)
                else:
                    med_weights = self._compute_medium_weight(cur_med_scores)

                mask = self._draw_boxes_to_mask(
                    boxes=cur_med_gts[:, :7], mask=mask,
                    center_values=med_weights,
                    pc_range=pc_range_dev, voxel_size=voxel_size_dev,
                    out_factor=self.out_factor, feature_map_size=self.feature_map_size,
                    gaussian_overlap=self.gaussian_overlap, min_radius=self.min_radius)

            # 3. 未匹配 GT: k = w_u * (1 - r / R_max)^2
            for unmatch in (unmatch_near[idx], unmatch_mid[idx], unmatch_far[idx]):
                if len(unmatch) > 0:
                    um_weights = self._compute_unmatched_weight(unmatch)
                    mask = self._draw_boxes_to_mask(
                        boxes=unmatch[:, :7], mask=mask,
                        center_values=um_weights,
                        pc_range=pc_range_dev, voxel_size=voxel_size_dev,
                        out_factor=self.out_factor, feature_map_size=self.feature_map_size,
                        gaussian_overlap=self.gaussian_overlap, min_radius=self.min_radius)

            mask_list.append(mask)

        return torch.stack(mask_list, dim=0)  # (B, 1, H, W)

    def forward(self, matched_results, batch_size):
        return self.generate_mask(matched_results, batch_size)
