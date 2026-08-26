import torch
import numpy as np
from copy import deepcopy

from labeldistill.refine_head.target_assigner.roi_distill import (
    MatchQuality,
    MatchResult,
)


class AdaptiveGTScalerV3:
    """
    Adaptive GT Box Scaling v3: ?????? ?

    ? GT ????????????????????

    ????? (??????):
        ?l = min(|?x_local|, ? * l_gt) + f(v_long)
    ?????? (??????):
        ?l = min(d_long, ? * l_gt) + f(v_long)
    ??? GT (?????????):
        ?l = (1 - r / R_max)^2 * f(v_long)

    """

    DEFAULT_CLASS_NAMES = (
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    )
    DEFAULT_DISTANCE_THRESHOLDS = {
        'car': {'high': 2.0, 'medium': 4.0},
        'truck': {'high': 2.5, 'medium': 4.0},
        'construction_vehicle': {'high': 2.5, 'medium': 4.0},
        'bus': {'high': 3.5, 'medium': 4.0},
        'trailer': {'high': 2.5, 'medium': 4.0},
        'barrier': {'high': 0.5, 'medium': 2.5},
        'motorcycle': {'high': 1.0, 'medium': 2.5},
        'bicycle': {'high': 1.0, 'medium': 2.5},
        'pedestrian': {'high': 0.5, 'medium': 2.0},
        'traffic_cone': {'high': 0.5, 'medium': 2.0},
    }

    def __init__(self, mu=0.15, s_vel=0.2, r_max=50.0, *,
                 class_names=None, distance_thresholds=None):
        self.mu = mu
        self.s_vel = s_vel
        self.r_max = r_max

        class_names = tuple(class_names or self.DEFAULT_CLASS_NAMES)
        distance_thresholds = distance_thresholds or self.DEFAULT_DISTANCE_THRESHOLDS
        if len(set(class_names)) != len(class_names):
            raise ValueError('class_names must be unique')
        if set(distance_thresholds) != set(class_names):
            raise ValueError('distance_thresholds must cover class_names exactly once')
        self.distance_thresholds = {
            class_id: {
                'high': float(distance_thresholds[name]['high']),
                'medium': float(distance_thresholds[name]['medium']),
            }
            for class_id, name in enumerate(class_names)
        }

    @staticmethod
    def _global_to_local(x_global, y_global, heading):
        """Rotate an ego/global BEV vector into the GT box local frame."""
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        x_local = x_global * cos_h + y_global * sin_h
        y_local = -x_global * sin_h + y_global * cos_h
        return x_local, y_local

    def _velocity_compensation(self, v_component, gt_size):
        """f(v) = { 0; 0.5 * s_vel * l_gt; s_vel * l_gt }"""
        abs_v = abs(v_component)
        if abs_v < 0.3:
            return 0.0
        elif abs_v < 0.8:
            return 0.5 * self.s_vel * gt_size
        else:
            return self.s_vel * gt_size

    def scale_high_quality(self, gt_boxes, roi_boxes):
        """??????: ?l = min(|?x_local|, ? * l_gt) + f(v_long)"""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt = gt_boxes[i]
            roi = roi_boxes[i]

            heading = gt[6].item()
            cos_h, sin_h = np.cos(heading), np.sin(heading)

            dx_global = roi[0].item() - gt[0].item()
            dy_global = roi[1].item() - gt[1].item()
            dx_local, dy_local = self._global_to_local(
                dx_global, dy_global, heading)

            original_l, original_w = gt[3].item(), gt[4].item()

            vx = gt[7].item() if gt.shape[0] > 8 else 0.0
            vy = gt[8].item() if gt.shape[0] > 8 else 0.0
            v_long, v_lat = self._global_to_local(vx, vy, heading)
            f_v_long = self._velocity_compensation(v_long, original_l)
            f_v_lat = self._velocity_compensation(v_lat, original_w)

            tau_l = self.mu * original_l
            tau_w = self.mu * original_w

            delta_l = min(abs(dx_local), tau_l) + f_v_long
            delta_w = min(abs(dy_local), tau_w) + f_v_lat

            scaled[i, 3] = original_l + delta_l
            scaled[i, 4] = original_w + delta_w

            sgn_x = (1.0 if dx_local > 0
                      else (-1.0 if dx_local < -0.1 else 0.0))
            sgn_y = (1.0 if dy_local > 0
                      else (-1.0 if dy_local < -0.1 else 0.0))
            local_shift_x = sgn_x * delta_l / 2.0
            local_shift_y = sgn_y * delta_w / 2.0

            global_shift_x = local_shift_x * cos_h - local_shift_y * sin_h
            global_shift_y = local_shift_x * sin_h + local_shift_y * cos_h

            scaled[i, 0] = gt[0] + global_shift_x
            scaled[i, 1] = gt[1] + global_shift_y

        return scaled

    def scale_medium_quality(self, gt_boxes, roi_boxes):
        """??????: ?l = min(d_long, ? * l_gt) + f(v_long)"""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt = gt_boxes[i]
            roi = roi_boxes[i]

            class_id = int(gt[-1].item())
            thresh = self.distance_thresholds.get(
                class_id, {'high': 2.0, 'medium': 4.0})
            tau_high_c = thresh['high']
            tau_med_c = thresh['medium']
            norm_ratio = tau_high_c / tau_med_c

            heading = gt[6].item()
            dx_global = roi[0].item() - gt[0].item()
            dy_global = roi[1].item() - gt[1].item()
            dx_local, dy_local = self._global_to_local(
                dx_global, dy_global, heading)
            d_offset = np.sqrt(dx_local ** 2 + dy_local ** 2)

            if d_offset < 0.1:
                continue

            d_long = norm_ratio * abs(dx_local)
            d_lat = norm_ratio * abs(dy_local)

            original_l, original_w = gt[3].item(), gt[4].item()

            vx = gt[7].item() if gt.shape[0] > 8 else 0.0
            vy = gt[8].item() if gt.shape[0] > 8 else 0.0
            v_long, v_lat = self._global_to_local(vx, vy, heading)
            f_v_long = self._velocity_compensation(v_long, original_l)
            f_v_lat = self._velocity_compensation(v_lat, original_w)

            tau_l = self.mu * original_l
            tau_w = self.mu * original_w

            delta_l = min(d_long, tau_l) + f_v_long
            delta_w = min(d_lat, tau_w) + f_v_lat

            scaled[i, 3] = original_l + delta_l
            scaled[i, 4] = original_w + delta_w

        return scaled

    def scale_unmatched_gt(self, gt_boxes):
        """?????????: ?l = (1 - r/R_max)^2 * f(v_long)"""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt = gt_boxes[i]
            x, y = gt[0].item(), gt[1].item()
            r = np.sqrt(x ** 2 + y ** 2)

            decay = max(0.0, 1.0 - r / self.r_max) ** 2

            original_l, original_w = gt[3].item(), gt[4].item()
            heading = gt[6].item()

            vx = gt[7].item() if gt.shape[0] > 8 else 0.0
            vy = gt[8].item() if gt.shape[0] > 8 else 0.0
            v_long, v_lat = self._global_to_local(vx, vy, heading)
            f_v_long = self._velocity_compensation(v_long, original_l)
            f_v_lat = self._velocity_compensation(v_lat, original_w)

            delta_l = decay * f_v_long
            delta_w = decay * f_v_lat

            scaled[i, 3] = original_l + delta_l
            scaled[i, 4] = original_w + delta_w

        return scaled

    def forward(self, matched_results):
        if isinstance(matched_results, MatchResult):
            scaled_gt_list = []
            for batch_idx, (gt_boxes, quality, matched_rois) in enumerate(zip(
                    matched_results.gt_boxes,
                    matched_results.quality,
                    matched_results.matched_rois)):
                scaled_gt = gt_boxes.clone()
                if matched_results.matched_mask is not None:
                    # The continuous-value baseline has no High/Medium geometry
                    # buckets. Every matched instance uses the same continuous
                    # proposal-offset expansion; q controls only loss strength.
                    matched_mask = matched_results.matched_mask[batch_idx]
                    scaled_gt[matched_mask] = self.scale_high_quality(
                        gt_boxes[matched_mask], matched_rois[matched_mask])
                    scaled_gt_list.append(scaled_gt)
                    continue

                high_mask = quality == int(MatchQuality.HIGH)
                medium_mask = quality == int(MatchQuality.MEDIUM)
                unmatched_mask = quality == int(MatchQuality.UNMATCHED)

                scaled_gt[high_mask] = self.scale_high_quality(
                    gt_boxes[high_mask], matched_rois[high_mask])
                scaled_gt[medium_mask] = self.scale_medium_quality(
                    gt_boxes[medium_mask], matched_rois[medium_mask])
                scaled_gt[unmatched_mask] = self.scale_unmatched_gt(
                    gt_boxes[unmatched_mask])
                scaled_gt_list.append(scaled_gt)

            return matched_results.with_gt_boxes(scaled_gt_list)

        # Compatibility path for experiments still using the legacy dict.
        batch_size = len(matched_results['refined_high_quality_gt'])

        scaled_high_gt_list = []
        scaled_medium_gt_list = []
        scaled_unmatched_list = []

        for idx in range(batch_size):
            high_gt = matched_results['refined_high_quality_gt'][idx]
            high_rois = matched_results['refined_high_quality_rois'][idx]

            if len(high_gt) > 0:
                scaled_high_gt = self.scale_high_quality(high_gt, high_rois)
            else:
                scaled_high_gt = high_gt
            scaled_high_gt_list.append(scaled_high_gt)

            medium_gt = matched_results['medium_quality_gt'][idx]
            medium_rois = matched_results['medium_quality_rois'][idx]

            if len(medium_gt) > 0:
                scaled_medium_gt = self.scale_medium_quality(
                    medium_gt, medium_rois)
            else:
                scaled_medium_gt = medium_gt
            scaled_medium_gt_list.append(scaled_medium_gt)

            unmatched_gt = matched_results['unmatched_gt'][idx]
            scaled_unmatched_list.append(self.scale_unmatched_gt(unmatched_gt))

        scaled_results = deepcopy(matched_results)
        scaled_results['refined_high_quality_gt'] = scaled_high_gt_list
        scaled_results['medium_quality_gt'] = scaled_medium_gt_list
        scaled_results['unmatched_gt'] = scaled_unmatched_list

        return scaled_results
