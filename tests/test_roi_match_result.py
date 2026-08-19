import torch

from labeldistill.models.distill_outputs import ProposalBatch
from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import (
    AdaptiveGTScalerV3,
)
from labeldistill.refine_head.target_assigner.roi_distill import (
    MatchQuality,
    MatchResult,
    ProposalTargetLayer,
)
from labeldistill.refine_head.target_assigner.quality_aware_mask_v3 import (
    QualityAwareMaskGeneratorV3,
)


def _make_batch():
    rois = torch.zeros((1, 4, 9), dtype=torch.float32)
    scores = torch.zeros((1, 4), dtype=torch.float32)
    labels = torch.zeros((1, 4), dtype=torch.long)
    gt = torch.zeros((1, 4, 10), dtype=torch.float32)

    # GT 0: two high-quality car proposals; the higher score must win.
    rois[0, 0, :2] = torch.tensor([0.5, 0.0])
    rois[0, 1, :2] = torch.tensor([1.0, 0.0])
    rois[0, :2, 3:6] = 1.0
    scores[0, :2] = torch.tensor([0.6, 0.9])
    labels[0, :2] = 0

    # GT 1: a medium-quality truck proposal.
    rois[0, 2, :2] = torch.tensor([13.0, 0.0])
    rois[0, 2, 3:6] = 1.0
    scores[0, 2] = 0.8
    labels[0, 2] = 1

    gt[0, 0, :7] = torch.tensor([0, 0, 0, 4, 2, 1, 0])
    gt[0, 0, 9] = 0
    gt[0, 1, :7] = torch.tensor([10, 0, 0, 5, 2, 2, 0])
    gt[0, 1, 9] = 1
    gt[0, 2, :7] = torch.tensor([30, 0, 0, 1, 1, 1, 0])
    gt[0, 2, 9] = 8

    return {
        'batch_size': 1,
        'rois': rois,
        'roi_scores': scores,
        'roi_labels': labels,
        'gt_boxes_and_cls': gt,
    }


def test_structured_match_result_is_aligned_to_gt_order():
    result = ProposalTargetLayer(structured=True)(_make_batch())

    assert isinstance(result, MatchResult)
    assert result.quality[0].tolist() == [
        int(MatchQuality.HIGH),
        int(MatchQuality.MEDIUM),
        int(MatchQuality.UNMATCHED),
    ]
    assert torch.allclose(result.roi_scores[0], torch.tensor([0.9, 0.8, 0.0]))
    assert result.roi_labels[0].tolist() == [0, 1, -1]
    assert torch.allclose(
        result.center_distances[0][:2], torch.tensor([1.0, 3.0]))
    assert torch.isinf(result.center_distances[0][2])

    # Existing mask consumers can still read the legacy split views.
    assert len(result['refined_high_quality_gt'][0]) == 1
    assert len(result['medium_quality_gt'][0]) == 1
    assert len(result['unmatched_gt'][0]) == 1


def test_scaler_preserves_match_metadata_and_gt_alignment():
    result = ProposalTargetLayer(structured=True)(_make_batch())
    scaled = AdaptiveGTScalerV3(
        mu=0.15, s_vel=0.2, r_max=50.0).forward(result)

    assert isinstance(scaled, MatchResult)
    assert scaled.quality is result.quality
    assert scaled.matched_rois is result.matched_rois
    assert scaled.roi_scores is result.roi_scores
    assert scaled.gt_boxes[0].shape == result.gt_boxes[0].shape
    assert torch.equal(scaled.gt_boxes[0][:, -1], result.gt_boxes[0][:, -1])


def test_default_mode_keeps_legacy_return_type():
    result = ProposalTargetLayer()(_make_batch())
    assert isinstance(result, dict)
    assert len(result['refined_high_quality_gt'][0]) == 1


def test_ragged_proposals_are_matched_without_padding_or_truncation():
    boxes = torch.zeros((300, 9), dtype=torch.float32)
    boxes[:, 0] = 40.0
    boxes[:, 3:6] = 1.0
    boxes[-1, 0] = 0.5
    scores = torch.linspace(0.1, 0.99, 300)
    labels = torch.zeros(300, dtype=torch.long)
    proposals = ProposalBatch(
        boxes=[boxes], scores=[scores], labels=[labels])
    gt_boxes = [torch.tensor(
        [[0, 0, 0, 4, 2, 1, 0, 0, 0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    result = ProposalTargetLayer(structured=True)(
        proposals, gt_boxes, gt_labels)

    assert isinstance(result, MatchResult)
    assert result.quality[0].tolist() == [int(MatchQuality.HIGH)]
    assert torch.allclose(result.matched_rois[0][0], boxes[-1])
    assert torch.allclose(result.roi_scores[0], scores[-1:])


def test_structured_result_flows_through_scaler_and_mask_generator():
    result = ProposalTargetLayer(structured=True)(_make_batch())
    scaled = AdaptiveGTScalerV3().forward(result)
    mask = QualityAwareMaskGeneratorV3(
        w_l=0.5, w_h=0.7, r_max=50.0,
        boost_small_medium=True)(scaled, batch_size=1)

    assert mask.shape == (1, 1, 128, 128)
    assert torch.isfinite(mask).all()
    assert mask.max() > 0
