"""Proposal-offset center ablation of AdaptiveGTScalerV3."""

from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import (
    AdaptiveGTScalerV3,
)


class ProposalCenterOnlyScaler(AdaptiveGTScalerV3):
    """Keep the legacy offset-only center displacement and original box size.

    Reuse the exact Adaptive implementation, including its asymmetric sign
    deadzone, to avoid adding a direction-policy change to this ablation.
    Velocity contributes zero for every input, including non-finite velocity.
    The inherited structured forward preserves matching and teacher values.
    """

    def __init__(self, *, mu=0.15, r_max=50.0, class_names=None,
                 distance_thresholds=None):
        super().__init__(mu=mu, s_vel=0.0, r_max=r_max,
                         class_names=class_names,
                         distance_thresholds=distance_thresholds)

    def _velocity_compensation(self, v_component, gt_size):
        return 0.0

    def scale_high_quality(self, gt_boxes, roi_boxes):
        offset_boxes = gt_boxes.clone()
        offset_boxes[:, 7:9] = 0.0
        shifted = super().scale_high_quality(offset_boxes, roi_boxes)
        # Restore every field except x/y, including dimensions and velocity.
        shifted[:, 2:] = gt_boxes[:, 2:]
        return shifted
