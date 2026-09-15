"""Proposal-centered motion expansion for feature-distillation regions."""

import math

import torch

from labeldistill.refine_head.target_assigner.proposal_center_shift import (
    ProposalCenterShift,
)
from labeldistill.refine_head.target_assigner.roi_distill import MatchResult


class ProposalCenterMotionExpansion:
    """Use matched Proposal centers and expand box dimensions by motion span."""

    requires_lidar_time_span = True

    def __init__(self, *, motion_alpha):
        if not math.isfinite(motion_alpha) or not 0 < motion_alpha <= 1:
            raise ValueError("motion_alpha must be finite and in (0, 1]")
        self.motion_alpha = float(motion_alpha)

    @staticmethod
    def _validate_time_spans(time_spans_s, batch_size):
        spans = torch.as_tensor(time_spans_s, dtype=torch.float64)
        if spans.ndim != 1 or len(spans) != batch_size:
            raise ValueError("time span count must equal MatchResult batch size")
        if not torch.isfinite(spans).all() or (spans < 0).any():
            raise ValueError("time spans must be finite and non-negative")
        return spans.tolist()

    @torch.no_grad()
    def forward(self, result: MatchResult, *, time_spans_s) -> MatchResult:
        if not isinstance(result, MatchResult):
            raise TypeError("ProposalCenterMotionExpansion requires MatchResult input")
        if result.matched_mask is None or result.effective_gt_mask is None:
            raise ValueError(
                "Proposal motion expansion requires matched_mask and "
                "effective_gt_mask"
            )
        spans = self._validate_time_spans(time_spans_s, len(result.gt_boxes))
        centered = ProposalCenterShift().forward(result)

        expanded_boxes = []
        for boxes, centered_boxes, matched, effective, time_span_s in zip(
            result.gt_boxes,
            centered.gt_boxes,
            result.matched_mask,
            result.effective_gt_mask,
            spans,
        ):
            if boxes.ndim != 2 or boxes.shape[-1] < 9:
                raise ValueError("GT boxes must have shape [N, >=9]")
            if matched.dtype != torch.bool or effective.dtype != torch.bool:
                raise ValueError("matched/effective masks must be boolean")
            if len(matched) != len(boxes) or len(effective) != len(boxes):
                raise ValueError("matched/effective masks must align with GT boxes")

            expanded = centered_boxes.clone()
            selected = matched & effective
            if selected.any():
                motion = boxes[selected, 6:9]
                valid_motion = torch.isfinite(motion).all(dim=1)
                safe_motion = torch.where(
                    valid_motion[:, None], motion, torch.zeros_like(motion)
                )
                yaw, vx, vy = safe_motion.unbind(dim=1)
                cosine = torch.cos(yaw)
                sine = torch.sin(yaw)
                v_long = vx * cosine + vy * sine
                v_lat = -vx * sine + vy * cosine
                scale = self.motion_alpha * float(time_span_s)
                expanded[selected, 3] = boxes[selected, 3] + scale * v_long.abs()
                expanded[selected, 4] = boxes[selected, 4] + scale * v_lat.abs()

            expanded_boxes.append(expanded)
        return centered.with_gt_boxes(expanded_boxes)
