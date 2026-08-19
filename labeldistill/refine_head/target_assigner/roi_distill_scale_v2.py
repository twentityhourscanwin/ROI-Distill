import torch
import numpy as np
from copy import deepcopy


class AdaptiveGTScalerV2:
    """
    论文对应版本的 Adaptive GT Scaler。

    高质量: Δl = min(|Δx_local|, τ) + f(v)  （单侧）
    中等:   d_x = (τ_high/τ_med) · d · |n_x|, Δl = min(d_x, τ) + f(v)  （双侧）
    未匹配: Δl = (1 - r/R_max)² · f(v)  （距离自适应）
    """

    def __init__(self, tau=2.0, s_vel=0.2, R_max=50.0):
        self.tau = tau
        self.s_vel = s_vel
        self.R_max = R_max

        self.distance_thresholds = {
            0: {'high': 2.0, 'medium': 4.0},
            1: {'high': 2.5, 'medium': 4.0},
            2: {'high': 2.5, 'medium': 4.0},
            3: {'high': 3.5, 'medium': 4.0},
            4: {'high': 2.5, 'medium': 4.0},
            5: {'high': 0.5, 'medium': 2.5},
            6: {'high': 1.0, 'medium': 2.5},
            7: {'high': 1.0, 'medium': 2.5},
            8: {'high': 0.5, 'medium': 2.0},
            9: {'high': 0.5, 'medium': 2.0},
        }

    def _f_vel(self, v, l_gt):
        """CRKD velocity compensation: returns absolute extension in meters."""
        v_abs = abs(v)
        if v_abs < 0.3:
            return 0.0
        elif v_abs < 0.8:
            return 0.5 * self.s_vel * l_gt
        else:
            return self.s_vel * l_gt

    def scale_high_quality(self, gt_boxes, roi_centers, roi_scores, gt_original_sizes):
        """Δl = min(|Δx_local|, τ) + f(v), single-side expansion."""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled_boxes = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            roi_center = roi_centers[i]
            original_l, original_w = gt_original_sizes[i]

            offset_x = roi_center[0] - gt_box[0]
            offset_y = roi_center[1] - gt_box[1]
            offset_dist = torch.sqrt(offset_x ** 2 + offset_y ** 2).item()

            if offset_dist < 0.1:
                continue

            heading = gt_box[6].item()
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)

            local_offset_x = offset_x.item() * cos_h + offset_y.item() * sin_h
            local_offset_y = -offset_x.item() * sin_h + offset_y.item() * cos_h

            velocity = gt_box[7:9] if gt_box.shape[0] > 8 else torch.zeros(2, device=gt_box.device)
            vx, vy = velocity[0].item(), velocity[1].item()

            extend_l = 0.0
            extend_w = 0.0

            if abs(local_offset_x) > 0.1:
                extend_l = min(abs(local_offset_x), self.tau) + self._f_vel(vx, original_l.item())

            if abs(local_offset_y) > 0.1:
                extend_w = min(abs(local_offset_y), self.tau) + self._f_vel(vy, original_w.item())

            new_l = original_l.item() + extend_l
            new_w = original_w.item() + extend_w

            local_center_shift_x = extend_l / 2.0 if local_offset_x > 0 else -extend_l / 2.0 if local_offset_x < -0.1 else 0
            local_center_shift_y = extend_w / 2.0 if local_offset_y > 0 else -extend_w / 2.0 if local_offset_y < -0.1 else 0

            global_shift_x = local_center_shift_x * cos_h - local_center_shift_y * sin_h
            global_shift_y = local_center_shift_x * sin_h + local_center_shift_y * cos_h

            scaled_boxes[i, 0] = gt_box[0] + global_shift_x
            scaled_boxes[i, 1] = gt_box[1] + global_shift_y
            scaled_boxes[i, 3] = new_l
            scaled_boxes[i, 4] = new_w

        return scaled_boxes

    def scale_medium_quality(self, gt_boxes, roi_centers, gt_original_sizes):
        """d_x = (τ_high/τ_med)·d·|n_x|, Δl = min(d_x, τ) + f(v), bilateral."""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled_boxes = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            roi_center = roi_centers[i]
            original_l, original_w = gt_original_sizes[i]

            diff = roi_center[:2] - gt_box[:2]
            d_offset = torch.sqrt((diff ** 2).sum()).item()

            if d_offset < 0.2:
                scaled_boxes[i, 3] = original_l * (1 + 0.2)
                scaled_boxes[i, 4] = original_w * (1 + 0.2)
                continue

            n_x = abs(diff[0].item()) / d_offset
            n_y = abs(diff[1].item()) / d_offset

            gt_class_id = int(gt_box[-1].item())
            thresholds = self.distance_thresholds.get(gt_class_id, {'high': 2.0, 'medium': 4.0})
            tau_ratio = thresholds['high'] / thresholds['medium']

            d_x = tau_ratio * d_offset * n_x
            d_y = tau_ratio * d_offset * n_y

            velocity = gt_box[7:9] if gt_box.shape[0] > 8 else torch.zeros(2, device=gt_box.device)
            vx, vy = velocity[0].item(), velocity[1].item()

            extend_l = min(d_x, self.tau) + self._f_vel(vx, original_l.item())
            extend_w = min(d_y, self.tau) + self._f_vel(vy, original_w.item())

            scaled_boxes[i, 3] = original_l.item() + extend_l
            scaled_boxes[i, 4] = original_w.item() + extend_w

        return scaled_boxes

    def scale_unmatched_gt(self, gt_boxes):
        """Δl = (1 - r/R_max)² · f(v), distance-adaptive velocity-only."""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled_boxes = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt_box = gt_boxes[i]
            original_l = gt_box[3].item()
            original_w = gt_box[4].item()

            x, y = gt_box[0].item(), gt_box[1].item()
            r = np.sqrt(x ** 2 + y ** 2)

            if r >= self.R_max:
                continue

            dist_weight = ((self.R_max - r) / self.R_max) ** 2

            velocity = gt_box[7:9] if gt_box.shape[0] > 8 else torch.zeros(2, device=gt_box.device)
            vx, vy = velocity[0].item(), velocity[1].item()

            extend_l = dist_weight * self._f_vel(vx, original_l)
            extend_w = dist_weight * self._f_vel(vy, original_w)

            scaled_boxes[i, 3] = original_l + extend_l
            scaled_boxes[i, 4] = original_w + extend_w

        return scaled_boxes

    def forward(self, matched_results):
        batch_size = len(matched_results['refined_high_quality_gt'])

        scaled_high_gt_list = []
        scaled_medium_gt_list = []
        scaled_unmatched_near_list = []
        scaled_unmatched_medium_list = []
        scaled_unmatched_far_list = []

        for idx in range(batch_size):
            high_gt = matched_results['refined_high_quality_gt'][idx]
            high_rois = matched_results['refined_high_quality_rois'][idx]
            high_scores = matched_results['refined_high_quality_roi_scores'][idx]

            if len(high_gt) > 0:
                original_sizes = high_gt[:, 3:5].clone()
                scaled_high_gt = self.scale_high_quality(
                    high_gt, high_rois[:, :3], high_scores, original_sizes
                )
            else:
                scaled_high_gt = high_gt
            scaled_high_gt_list.append(scaled_high_gt)

            medium_gt = matched_results['medium_quality_gt'][idx]
            medium_rois = matched_results['medium_quality_rois'][idx]

            if len(medium_gt) > 0:
                original_sizes = medium_gt[:, 3:5].clone()
                scaled_medium_gt = self.scale_medium_quality(
                    medium_gt, medium_rois[:, :3], original_sizes
                )
            else:
                scaled_medium_gt = medium_gt
            scaled_medium_gt_list.append(scaled_medium_gt)

            unmatched_near = matched_results['unmatched_gt_near'][idx]
            unmatched_medium = matched_results['unmatched_gt_medium'][idx]
            unmatched_far = matched_results['unmatched_gt_far'][idx]

            scaled_unmatched_near_list.append(self.scale_unmatched_gt(unmatched_near))
            scaled_unmatched_medium_list.append(self.scale_unmatched_gt(unmatched_medium))
            scaled_unmatched_far_list.append(self.scale_unmatched_gt(unmatched_far))

        scaled_results = deepcopy(matched_results)
        scaled_results['refined_high_quality_gt'] = scaled_high_gt_list
        scaled_results['medium_quality_gt'] = scaled_medium_gt_list
        scaled_results['unmatched_gt_near'] = scaled_unmatched_near_list
        scaled_results['unmatched_gt_medium'] = scaled_unmatched_medium_list
        scaled_results['unmatched_gt_far'] = scaled_unmatched_far_list

        return scaled_results
