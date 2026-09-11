"""Blend GT/proposal centers by proposal score without resizing GT boxes."""

import torch

from labeldistill.refine_head.target_assigner.roi_distill import MatchResult


class ScoreCenterShift:
    """Replace only drawable matched GT x/y after matching and q are fixed."""

    @torch.no_grad()
    def forward(self, result: MatchResult) -> MatchResult:
        if result.matched_mask is None or result.effective_gt_mask is None:
            raise ValueError(
                'Center-only geometry requires matched_mask and effective_gt_mask')

        shifted_boxes = []
        for boxes, proposals, matched, effective, scores in zip(
                result.gt_boxes, result.matched_rois,
                result.matched_mask, result.effective_gt_mask, result.roi_scores):
            shifted = boxes.clone()
            selected = matched & effective
            if selected.any():
                gt_xy = boxes[selected, :2]
                proposal_xy = proposals[selected, :2]
                if not (torch.isfinite(gt_xy).all()
                        and torch.isfinite(proposal_xy).all()):
                    raise ValueError('Selected GT and proposal centers must be finite')
                score = scores[selected]
                if not (torch.isfinite(score).all()
                        and ((score >= 0) & (score <= 1)).all()):
                    raise ValueError('Selected proposal score must be finite and in [0, 1]')
                weight = score[:, None]
                shifted[selected, :2] = weight * proposal_xy + (1 - weight) * gt_xy
            shifted_boxes.append(shifted)

        return result.with_gt_boxes(shifted_boxes)
