import numpy as np
import torch
import torch.nn as nn
from mmdet3d.models.utils.gaussian import draw_heatmap_gaussian, gaussian_radius


class BEVDistillationMaskGeneratorV2(nn.Module):
    """
    论文对应版本：未匹配 GT 使用 k_0·(1-r/R_max)² 连续权重，
    高/中质量部分保持不变。
    """
    def __init__(self, k0=0.5, R_max=50.0):
        super().__init__()
        self.k0 = k0
        self.R_max = R_max

        grid_size = [1024, 1024, 40]
        voxel_size = [0.1, 0.1, 0.2]
        out_size_factor = 8
        point_cloud_range_list = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        gaussian_overlap = 0.1
        min_radius = 2

        self.grid_size = torch.tensor(grid_size)
        self.pc_range = torch.tensor(point_cloud_range_list)
        self.voxel_size = torch.tensor(voxel_size)
        self.out_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius

        self.feature_map_size = (self.grid_size[:2] // self.out_factor).cpu().numpy()

        self.small_classes = [5, 6, 8, 9]

    def _compute_unmatched_weight(self, gt_box):
        """k = k0 · (1 - r/R_max)², returns 0 when r >= R_max."""
        x, y = gt_box[0].item(), gt_box[1].item()
        r = np.sqrt(x ** 2 + y ** 2)
        if r >= self.R_max:
            return 0.0
        return self.k0 * ((self.R_max - r) / self.R_max) ** 2

    def generate_mask(self, matched_results, batch_size):
        pc_range = self.pc_range
        voxel_size = self.voxel_size
        out_factor = self.out_factor
        feature_map_size = self.feature_map_size
        gaussian_overlap = self.gaussian_overlap
        min_radius = self.min_radius

        refined_high_rois = matched_results['refined_high_quality_rois']
        refined_high_scores = matched_results['refined_high_quality_roi_scores']

        medium_rois = matched_results['medium_quality_rois']
        medium_scores = matched_results['medium_quality_roi_scores']
        medium_gts = matched_results['medium_quality_gt']

        near_unmatched_gts = matched_results['unmatched_gt_near']
        medium_unmatched_gts = matched_results['unmatched_gt_medium']
        far_unmatched_gts = matched_results['unmatched_gt_far']

        assert len(refined_high_rois) == batch_size
        assert len(medium_rois) == batch_size

        mask_list = []

        for idx in range(batch_size):
            cur_refined_rois = refined_high_rois[idx]
            cur_refined_scores = refined_high_scores[idx]

            cur_medium_rois = medium_rois[idx]
            cur_medium_scores = medium_scores[idx]
            cur_medium_gts = medium_gts[idx]

            cur_near_unmatched = near_unmatched_gts[idx]
            cur_medium_unmatched = medium_unmatched_gts[idx]

            device = cur_refined_rois.device
            pc_range_dev = pc_range.to(device)
            voxel_size_dev = voxel_size.to(device)

            mask = torch.zeros((1, feature_map_size[1], feature_map_size[0]),
                               device=device, dtype=torch.float32)

            # ===== 1. 高质量匹配 ROI =====
            if len(cur_refined_rois) > 0:
                mask = self._draw_boxes_to_mask(
                    boxes=cur_refined_rois,
                    mask=mask,
                    center_values=cur_refined_scores,
                    pc_range=pc_range_dev,
                    voxel_size=voxel_size_dev,
                    out_factor=out_factor,
                    feature_map_size=feature_map_size,
                    gaussian_overlap=gaussian_overlap,
                    min_radius=min_radius
                )

            # ===== 2. 中等质量匹配 GT =====
            if len(cur_medium_gts) > 0:
                gt_classes = cur_medium_gts[:, -1].long()

                small_class_mask = torch.zeros(len(gt_classes), dtype=torch.bool, device=device)
                for small_class_id in self.small_classes:
                    small_class_mask |= (gt_classes == small_class_id)

                if small_class_mask.any():
                    small_gts = cur_medium_gts[small_class_mask]
                    small_roi_scores = cur_medium_scores[small_class_mask]
                    small_center_values = (small_roi_scores + 1.0) / 2.0

                    mask = self._draw_boxes_to_mask(
                        boxes=small_gts[:, :7],
                        mask=mask,
                        center_values=small_center_values,
                        pc_range=pc_range_dev,
                        voxel_size=voxel_size_dev,
                        out_factor=out_factor,
                        feature_map_size=feature_map_size,
                        gaussian_overlap=gaussian_overlap,
                        min_radius=min_radius
                    )

                other_class_mask = ~small_class_mask
                if other_class_mask.any():
                    other_gts = cur_medium_gts[other_class_mask]
                    other_center_values = torch.full((len(other_gts),), 0.5,
                                                    device=device, dtype=torch.float32)

                    mask = self._draw_boxes_to_mask(
                        boxes=other_gts[:, :7],
                        mask=mask,
                        center_values=other_center_values,
                        pc_range=pc_range_dev,
                        voxel_size=voxel_size_dev,
                        out_factor=out_factor,
                        feature_map_size=feature_map_size,
                        gaussian_overlap=gaussian_overlap,
                        min_radius=min_radius
                    )

            # ===== 3. 未匹配 GT: 统一使用 k0·(1-r/R_max)² =====
            all_unmatched = []
            if len(cur_near_unmatched) > 0:
                all_unmatched.append(cur_near_unmatched)
            if len(cur_medium_unmatched) > 0:
                all_unmatched.append(cur_medium_unmatched)

            if len(all_unmatched) > 0:
                unmatched_cat = torch.cat(all_unmatched, dim=0)
                weights = torch.tensor(
                    [self._compute_unmatched_weight(unmatched_cat[j]) for j in range(len(unmatched_cat))],
                    device=device, dtype=torch.float32
                )
                valid = weights > 0
                if valid.any():
                    mask = self._draw_boxes_to_mask(
                        boxes=unmatched_cat[valid, :7],
                        mask=mask,
                        center_values=weights[valid],
                        pc_range=pc_range_dev,
                        voxel_size=voxel_size_dev,
                        out_factor=out_factor,
                        feature_map_size=feature_map_size,
                        gaussian_overlap=gaussian_overlap,
                        min_radius=min_radius
                    )

            mask_list.append(mask)

        weight_mask = torch.stack(mask_list, dim=0)
        return weight_mask

    def _draw_boxes_to_mask(self, boxes, mask, center_values, pc_range, voxel_size,
                            out_factor, feature_map_size, gaussian_overlap, min_radius):
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
                (length_fm, width_fm),
                min_overlap=gaussian_overlap
            )
            radius = max(min_radius, int(radius_tensor.item()))

            x, y = boxes[k][0], boxes[k][1]
            coor_x = (x - pc_range[0]) / voxel_size[0] / out_factor
            coor_y = (y - pc_range[1]) / voxel_size[1] / out_factor

            center_int = torch.stack([coor_x, coor_y], dim=0).to(torch.int32)

            if not (0 <= center_int[0] < feat_width and
                    0 <= center_int[1] < feat_height):
                continue

            heatmap = draw_heatmap_gaussian(
                heatmap,
                center_int,
                radius,
                k=center_values[k].item()
            )

        return mask

    def forward(self, matched_results, batch_size):
        return self.generate_mask(matched_results, batch_size)
