import torch
import numpy as np
from copy import deepcopy


class AdaptiveGTScalerV2:
    """
    Adaptive GT Box Scaling 实现，支持两种 tau 模式:

    模式1 (tau): 所有类别统一的扩展上限  — 原始 v2 用法
    模式2 (tau_ratio): τ_c = high_threshold_c × tau_ratio — 类别相关

    高质量匹配: 单侧定向扩展
        Δl = min(|Δx_local|, τ) + f(v^x)

    中等质量匹配: 双侧对称扩展
        Δl = min(d_x, τ) + f(v^x)

    未匹配 GT: 距离自适应 + 基础扩展下界
        Δl = max(base_expand_ratio * l_gt, decay * f(v^x))
        decay = (1 - r/R_max)^2

    速度函数 (来自 CRKD):
        f(v) = { 0                    if |v| < 0.3
               { 0.5 * s_vel * l_gt   if 0.3 <= |v| < 0.8
               { s_vel * l_gt         if |v| >= 0.8
    """

    def __init__(self, tau=None, tau_ratio=None, s_vel=0.2, r_max=50.0,
                 base_expand_ratio=0.0, medium_expand_ratio=0.0):
        self.s_vel = s_vel
        self.r_max = r_max
        self.base_expand_ratio = base_expand_ratio
        self.medium_expand_ratio = medium_expand_ratio

        self.distance_thresholds = {
            0: {'high': 2.0, 'medium': 4.0},   # car
            1: {'high': 2.5, 'medium': 4.0},   # truck
            2: {'high': 2.5, 'medium': 4.0},   # construction_vehicle
            3: {'high': 3.5, 'medium': 4.0},   # bus
            4: {'high': 2.5, 'medium': 4.0},   # trailer
            5: {'high': 0.5, 'medium': 2.5},   # barrier
            6: {'high': 1.0, 'medium': 2.5},   # motorcycle
            7: {'high': 1.0, 'medium': 2.5},   # bicycle
            8: {'high': 0.5, 'medium': 2.0},   # pedestrian
            9: {'high': 0.5, 'medium': 2.0},   # traffic_cone
        }

        if tau is not None and tau_ratio is None:
            self._tau_fixed = tau
            self.tau_per_class = None
        else:
            ratio = tau_ratio if tau_ratio is not None else 0.3
            self._tau_fixed = None
            self.tau_per_class = {
                cls_id: thresh['high'] * ratio
                for cls_id, thresh in self.distance_thresholds.items()
            }

    def _velocity_compensation(self, v_component, gt_size):
        """f(v) = { 0; 0.5 * s_vel * l_gt; s_vel * l_gt }"""
        abs_v = abs(v_component)
        if abs_v < 0.3:
            return 0.0
        elif abs_v < 0.8:
            return 0.5 * self.s_vel * gt_size
        else:
            return self.s_vel * gt_size

    def _get_tau(self, class_id):
        if self._tau_fixed is not None:
            return self._tau_fixed
        return self.tau_per_class.get(int(class_id), 0.6)

    def scale_high_quality(self, gt_boxes, roi_boxes):
        """单侧定向扩展，严格对应论文公式 (8)(9)(10)，τ_c 为类别相关"""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt = gt_boxes[i]
            roi = roi_boxes[i]

            class_id = int(gt[-1].item())
            tau_c = self._get_tau(class_id)

            heading = gt[6].item()
            cos_h, sin_h = np.cos(heading), np.sin(heading)

            dx_global = roi[0].item() - gt[0].item()
            dy_global = roi[1].item() - gt[1].item()

            dx_local = dx_global * cos_h + dy_global * sin_h
            dy_local = -dx_global * sin_h + dy_global * cos_h

            original_l, original_w = gt[3].item(), gt[4].item()

            vx = gt[7].item() if gt.shape[0] > 8 else 0.0
            vy = gt[8].item() if gt.shape[0] > 8 else 0.0
            f_vx = self._velocity_compensation(vx, original_l)
            f_vy = self._velocity_compensation(vy, original_w)

            delta_l = min(abs(dx_local), tau_c) + f_vx
            delta_w = min(abs(dy_local), tau_c) + f_vy

            scaled[i, 3] = original_l + delta_l
            scaled[i, 4] = original_w + delta_w

            sgn_x = 1.0 if dx_local > 0 else (-1.0 if dx_local < -0.1 else 0.0)
            sgn_y = 1.0 if dy_local > 0 else (-1.0 if dy_local < -0.1 else 0.0)
            local_shift_x = sgn_x * delta_l / 2.0
            local_shift_y = sgn_y * delta_w / 2.0

            global_shift_x = local_shift_x * cos_h - local_shift_y * sin_h
            global_shift_y = local_shift_x * sin_h + local_shift_y * cos_h

            scaled[i, 0] = gt[0] + global_shift_x
            scaled[i, 1] = gt[1] + global_shift_y

        return scaled

    def scale_medium_quality(self, gt_boxes, roi_boxes):
        """双侧对称扩展，严格对应论文公式 (11)(12)，τ_c 为类别相关"""
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt = gt_boxes[i]
            roi = roi_boxes[i]

            class_id = int(gt[-1].item())
            thresh = self.distance_thresholds.get(class_id, {'high': 2.0, 'medium': 4.0})
            tau_high_c = thresh['high']
            tau_med_c = thresh['medium']
            norm_ratio = tau_high_c / tau_med_c
            tau_c = self._get_tau(class_id)

            dx = roi[0].item() - gt[0].item()
            dy = roi[1].item() - gt[1].item()
            d_offset = np.sqrt(dx ** 2 + dy ** 2)

            if d_offset < 0.1:
                continue

            n_x = dx / d_offset
            n_y = dy / d_offset

            d_x = norm_ratio * d_offset * abs(n_x)
            d_y = norm_ratio * d_offset * abs(n_y)

            original_l, original_w = gt[3].item(), gt[4].item()

            vx = gt[7].item() if gt.shape[0] > 8 else 0.0
            vy = gt[8].item() if gt.shape[0] > 8 else 0.0
            f_vx = self._velocity_compensation(vx, original_l)
            f_vy = self._velocity_compensation(vy, original_w)

            delta_l = min(d_x, tau_c) + f_vx
            delta_w = min(d_y, tau_c) + f_vy

            if self.medium_expand_ratio > 0:
                floor_l = self.medium_expand_ratio * original_l
                floor_w = self.medium_expand_ratio * original_w
                delta_l = max(delta_l, floor_l)
                delta_w = max(delta_w, floor_w)

            scaled[i, 3] = original_l + delta_l
            scaled[i, 4] = original_w + delta_w

        return scaled

    def scale_unmatched_gt(self, gt_boxes):
        """距离自适应速度补偿 + 基础扩展下界
        Δl = max(base_expand_ratio * l_gt, (1-r/R_max)^2 * f(v^x))
        """
        if len(gt_boxes) == 0:
            return gt_boxes

        scaled = gt_boxes.clone()

        for i in range(len(gt_boxes)):
            gt = gt_boxes[i]
            x, y = gt[0].item(), gt[1].item()
            r = np.sqrt(x ** 2 + y ** 2)

            decay = max(0.0, 1.0 - r / self.r_max) ** 2

            original_l, original_w = gt[3].item(), gt[4].item()

            vx = gt[7].item() if gt.shape[0] > 8 else 0.0
            vy = gt[8].item() if gt.shape[0] > 8 else 0.0
            f_vx = self._velocity_compensation(vx, original_l)
            f_vy = self._velocity_compensation(vy, original_w)

            floor_l = self.base_expand_ratio * original_l
            floor_w = self.base_expand_ratio * original_w

            delta_l = max(floor_l, decay * f_vx)
            delta_w = max(floor_w, decay * f_vy)

            scaled[i, 3] = original_l + delta_l
            scaled[i, 4] = original_w + delta_w

        return scaled

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

            if len(high_gt) > 0:
                scaled_high_gt = self.scale_high_quality(high_gt, high_rois)
            else:
                scaled_high_gt = high_gt
            scaled_high_gt_list.append(scaled_high_gt)

            medium_gt = matched_results['medium_quality_gt'][idx]
            medium_rois = matched_results['medium_quality_rois'][idx]

            if len(medium_gt) > 0:
                scaled_medium_gt = self.scale_medium_quality(medium_gt, medium_rois)
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
