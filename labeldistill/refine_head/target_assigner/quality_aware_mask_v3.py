import torch
import torch.nn as nn
from mmdet3d.models.utils.gaussian import draw_heatmap_gaussian, gaussian_radius


class QualityAwareMaskGeneratorV3(nn.Module):
    """
    三级分层权重掩码生成器 v3:
        高质量匹配:  k = w_h + (1 - w_h) * s_roi   ∈ [w_h, 1]
        中等质量匹配: k = w_l + (w_h - w_l) * s_roi  ∈ [w_l, w_h]
        未匹配:      k = w_l * (1 - r / R_max)^2      ∈ [0,  w_l]
    约束: 0 < w_l <= w_h < 1
    """

    SMALL_CLASSES = {5, 6, 8, 9}  # barrier, motorcycle, pedestrian, traffic_cone

    def __init__(self, w_l=0.4, w_h=0.7, r_max=50.0,
                 boost_small_medium=False, *,
                 point_cloud_range=None, voxel_size=None,
                 out_size_factor=8, feature_map_size=None,
                 gaussian_overlap=0.1, min_radius=2,
                 small_class_ids=None):
        super().__init__()
        assert 0 < w_l <= w_h < 1, (
            f"Need 0 < w_l <= w_h < 1, got w_l={w_l}, w_h={w_h}")
        self.w_l = w_l
        self.w_h = w_h
        self.r_max = r_max
        self.boost_small_medium = boost_small_medium
        self.small_class_ids = set(
            self.SMALL_CLASSES if small_class_ids is None else small_class_ids)

        voxel_size = list(voxel_size or [0.1, 0.1, 0.2])
        point_cloud_range = list(
            point_cloud_range
            or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        if len(voxel_size) != 3 or len(point_cloud_range) != 6:
            raise ValueError('voxel_size and point_cloud_range must have 3 and 6 values')
        if out_size_factor <= 0 or gaussian_overlap <= 0 or min_radius < 0:
            raise ValueError(
                'out_size_factor/gaussian_overlap must be positive and '
                'min_radius must be non-negative')
        if feature_map_size is None:
            extents = [
                point_cloud_range[3] - point_cloud_range[0],
                point_cloud_range[4] - point_cloud_range[1],
            ]
            feature_map_size = [
                int(round(extent / voxel_size[idx] / out_size_factor))
                for idx, extent in enumerate(extents)
            ]
        if len(feature_map_size) != 2:
            raise ValueError('feature_map_size must contain [width, height]')

        self.register_buffer('pc_range', torch.tensor(point_cloud_range))
        self.register_buffer('voxel_size', torch.tensor(voxel_size))
        self.out_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.feature_map_size = tuple(int(value) for value in feature_map_size)

    def _compute_high_weight(self, s_roi):
        """k = w_h + (1 - w_h) * s_roi  ∈ [w_h, 1]"""
        return self.w_h + (1.0 - self.w_h) * s_roi

    def _compute_medium_weight(self, s_roi):
        """k = w_l + (w_h - w_l) * s_roi  ∈ [w_l, w_h]"""
        return self.w_l + (self.w_h - self.w_l) * s_roi

    def _compute_unmatched_weight(self, gt_boxes):
        """k = w_l * (1 - r / R_max)^2  ∈ [0, w_l]"""
        r = torch.sqrt(gt_boxes[:, 0] ** 2 + gt_boxes[:, 1] ** 2)
        ratio = torch.clamp(1.0 - r / self.r_max, min=0.0)
        return self.w_l * ratio ** 2

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

            if not (0 <= center_int[0] < feat_width
                    and 0 <= center_int[1] < feat_height):
                continue

            cv = center_values[k].item() if torch.is_tensor(
                center_values[k]) else center_values[k]
            if cv <= 0:
                continue
            heatmap = draw_heatmap_gaussian(heatmap, center_int, radius, k=cv)

        return mask

    def generate_mask(self, matched_results, batch_size):
        feat_W, feat_H = self.feature_map_size

        refined_high_gts = matched_results['refined_high_quality_gt']
        high_scores = matched_results['refined_high_quality_roi_scores']
        medium_gts = matched_results['medium_quality_gt']
        medium_scores = matched_results['medium_quality_roi_scores']
        unmatched_gts = matched_results['unmatched_gt']

        mask_list = []

        for idx in range(batch_size):
            device = refined_high_gts[idx].device
            pc_range_dev = self.pc_range.to(device)
            voxel_size_dev = self.voxel_size.to(device)

            mask = torch.zeros(
                (1, feat_H, feat_W), device=device, dtype=torch.float32)

            # 1. 高质量匹配 GT: k ∈ [w_h, 1]
            cur_high_gts = refined_high_gts[idx]
            cur_high_scores = high_scores[idx]
            if len(cur_high_gts) > 0:
                high_weights = self._compute_high_weight(cur_high_scores)
                mask = self._draw_boxes_to_mask(
                    boxes=cur_high_gts[:, :7], mask=mask,
                    center_values=high_weights,
                    pc_range=pc_range_dev, voxel_size=voxel_size_dev,
                    out_factor=self.out_factor,
                    feature_map_size=self.feature_map_size,
                    gaussian_overlap=self.gaussian_overlap,
                    min_radius=self.min_radius)

            # 2. 中等质量匹配 GT: k ∈ [w_l, w_h]
            cur_med_gts = medium_gts[idx]
            cur_med_scores = medium_scores[idx]
            if len(cur_med_gts) > 0:
                med_weights = self._compute_medium_weight(cur_med_scores)

                if self.boost_small_medium:
                    gt_classes = cur_med_gts[:, -1].long()
                    is_small = torch.zeros(
                        len(gt_classes), dtype=torch.bool, device=device)
                    for cls_id in self.small_class_ids:
                        is_small |= (gt_classes == cls_id)
                    if is_small.any():
                        boost = cur_med_scores[is_small] * (1.0 - self.w_l)
                        med_weights[is_small] = torch.clamp(
                            med_weights[is_small] + boost, max=1.0)

                mask = self._draw_boxes_to_mask(
                    boxes=cur_med_gts[:, :7], mask=mask,
                    center_values=med_weights,
                    pc_range=pc_range_dev, voxel_size=voxel_size_dev,
                    out_factor=self.out_factor,
                    feature_map_size=self.feature_map_size,
                    gaussian_overlap=self.gaussian_overlap,
                    min_radius=self.min_radius)

            # 3. 未匹配 GT: k ∈ [0, w_l]
            cur_unmatched = unmatched_gts[idx]
            if len(cur_unmatched) > 0:
                um_weights = self._compute_unmatched_weight(cur_unmatched)
                mask = self._draw_boxes_to_mask(
                    boxes=cur_unmatched[:, :7], mask=mask,
                    center_values=um_weights,
                    pc_range=pc_range_dev, voxel_size=voxel_size_dev,
                    out_factor=self.out_factor,
                    feature_map_size=self.feature_map_size,
                    gaussian_overlap=self.gaussian_overlap,
                    min_radius=self.min_radius)

            mask_list.append(mask)

        return torch.stack(mask_list, dim=0)

    def forward(self, matched_results, batch_size):
        return self.generate_mask(matched_results, batch_size)
