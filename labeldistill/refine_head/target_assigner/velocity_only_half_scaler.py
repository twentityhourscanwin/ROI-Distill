"""Velocity-only GT geometry for multi-sweep feature distillation."""

import torch

from labeldistill.refine_head.target_assigner.roi_distill import MatchResult


class VelocityOnlyHalfScaler:
    """Expand GT dimensions by a fraction of the multi-sweep displacement.

    Velocity is projected into the GT box frame.  The total length and width
    increments are ``fraction * abs(v_axis) * (T_past + T_future)``.  The
    ``fixed`` mode keeps the current GT center unchanged.  The
    ``temporal_midpoint`` mode shifts it to the midpoint of the retained past
    and future trajectory interval.
    """

    CENTER_MODES = frozenset({'fixed', 'temporal_midpoint'})

    def __init__(self, *, displacement_fraction=0.5,
                 past_time_seconds=0.25, future_time_seconds=0.20,
                 center_mode='fixed'):
        if not 0 < displacement_fraction <= 1:
            raise ValueError('displacement_fraction must be in (0, 1]')
        if past_time_seconds < 0 or future_time_seconds < 0:
            raise ValueError('past/future time spans must be non-negative')
        if past_time_seconds + future_time_seconds <= 0:
            raise ValueError('multi-sweep time span must be positive')
        if center_mode not in self.CENTER_MODES:
            raise ValueError(f'Unsupported center_mode={center_mode!r}')
        self.displacement_fraction = float(displacement_fraction)
        self.past_time_seconds = float(past_time_seconds)
        self.future_time_seconds = float(future_time_seconds)
        self.center_mode = center_mode

    def scale_boxes(self, gt_boxes):
        """Return scaled boxes without modifying the input tensor."""
        if gt_boxes.ndim != 2 or gt_boxes.shape[-1] < 9:
            raise ValueError(
                'Expected GT boxes with shape [N, >=9], got '
                f'{tuple(gt_boxes.shape)}')
        scaled = gt_boxes.clone()
        if len(gt_boxes) == 0:
            return scaled

        yaw = gt_boxes[:, 6]
        vx = gt_boxes[:, 7]
        vy = gt_boxes[:, 8]
        valid_motion = (
            torch.isfinite(yaw) & torch.isfinite(vx) & torch.isfinite(vy))

        cos_yaw = torch.cos(torch.where(valid_motion, yaw, torch.zeros_like(yaw)))
        sin_yaw = torch.sin(torch.where(valid_motion, yaw, torch.zeros_like(yaw)))
        safe_vx = torch.where(valid_motion, vx, torch.zeros_like(vx))
        safe_vy = torch.where(valid_motion, vy, torch.zeros_like(vy))
        v_long = safe_vx * cos_yaw + safe_vy * sin_yaw
        v_lat = -safe_vx * sin_yaw + safe_vy * cos_yaw

        time_span = self.past_time_seconds + self.future_time_seconds
        delta_length = (
            self.displacement_fraction * v_long.abs() * time_span)
        delta_width = self.displacement_fraction * v_lat.abs() * time_span
        scaled[:, 3] = gt_boxes[:, 3] + delta_length
        scaled[:, 4] = gt_boxes[:, 4] + delta_width

        if self.center_mode == 'temporal_midpoint':
            midpoint_time = 0.5 * (
                self.future_time_seconds - self.past_time_seconds)
            local_shift_long = (
                self.displacement_fraction * v_long * midpoint_time)
            local_shift_lat = (
                self.displacement_fraction * v_lat * midpoint_time)
            global_shift_x = (
                local_shift_long * cos_yaw - local_shift_lat * sin_yaw)
            global_shift_y = (
                local_shift_long * sin_yaw + local_shift_lat * cos_yaw)
            scaled[:, 0] = gt_boxes[:, 0] + global_shift_x
            scaled[:, 1] = gt_boxes[:, 1] + global_shift_y

        return scaled

    def forward(self, matched_results):
        if not isinstance(matched_results, MatchResult):
            raise TypeError('VelocityOnlyHalfScaler requires MatchResult input')
        if matched_results.effective_gt_mask is None:
            raise ValueError(
                'VelocityOnlyHalfScaler requires effective_gt_mask')

        scaled_gt_list = []
        for gt_boxes, effective_mask in zip(
                matched_results.gt_boxes,
                matched_results.effective_gt_mask):
            if effective_mask.dtype != torch.bool or len(effective_mask) != len(gt_boxes):
                raise ValueError(
                    'effective_gt_mask must be a boolean tensor aligned to GTs')
            scaled = gt_boxes.clone()
            scaled[effective_mask] = self.scale_boxes(gt_boxes[effective_mask])
            scaled_gt_list.append(scaled)
        return matched_results.with_gt_boxes(scaled_gt_list)
