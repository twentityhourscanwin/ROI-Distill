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


SMALL_CLASSES = [
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
LARGE_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer']
OFFICIAL_RANGES = {
    'car': 50.0, 'truck': 50.0, 'construction_vehicle': 50.0,
    'bus': 50.0, 'trailer': 50.0, 'barrier': 30.0,
    'motorcycle': 40.0, 'bicycle': 40.0, 'pedestrian': 40.0,
    'traffic_cone': 30.0,
}


def _center_value_matcher(
        *, value_type='normalized_squared_margin',
        selection='score_first_nearest_unmatched_gt', one_to_one=True):
    return ProposalTargetLayer(
        structured=True,
        matching_type='scale_conditioned_center_distance',
        class_policy='exact_class',
        selection=selection,
        one_to_one=one_to_one,
        strict_less_than=True,
        trust_radius={'small': 1.0, 'large': 2.0},
        small_classes=SMALL_CLASSES,
        large_classes=LARGE_CLASSES,
        official_class_ranges=OFFICIAL_RANGES,
        point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        value_type=value_type,
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


def test_center_value_matcher_is_exact_class_score_first_and_one_to_one():
    boxes = torch.zeros((3, 9), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    boxes[0, 0] = 0.30  # Highest score: nearest GT 0.
    boxes[1, 0] = 0.10  # Then forced to the still-unmatched GT 1.
    boxes[2, 0] = 0.70  # Wrong class despite being centered on GT 1.
    proposals = ProposalBatch(
        boxes=[boxes],
        scores=[torch.tensor([0.9, 0.8, 0.99])],
        labels=[torch.tensor([0, 0, 1])],
    )
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 4, 2, 1, 0, 0, 0],
        [0.7, 0, 0, 4, 2, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([0, 0])]

    result = _center_value_matcher()(proposals, gt_boxes, gt_labels)

    assert result.matched_mask[0].tolist() == [True, True]
    assert result.matched_proposal_indices[0].tolist() == [0, 1]
    assert result.roi_labels[0].tolist() == [0, 0]
    assert result.teacher_values[0][0] == torch.tensor(1 - (0.3 / 2) ** 2)
    assert result.teacher_values[0][1] == torch.tensor(1 - (0.6 / 2) ** 2)


def test_gt_nearest_prefers_geometry_over_teacher_score():
    boxes = torch.zeros((2, 9), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    boxes[:, 0] = torch.tensor([0.1, 0.5])
    proposals = ProposalBatch(
        boxes=[boxes],
        scores=[torch.tensor([0.5, 0.9])],
        labels=[torch.tensor([0, 0])],
    )
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 4, 2, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([0])]

    result = _center_value_matcher(
        selection='gt_nearest', one_to_one=False)(
            proposals, gt_boxes, gt_labels)

    assert result.matched_proposal_indices[0].tolist() == [0]
    assert result.roi_scores[0].tolist() == [0.5]
    assert result.center_distances[0][0] == torch.tensor(0.1)
    assert result.teacher_values[0][0] == torch.tensor(1 - (0.1 / 2) ** 2)


def test_gt_nearest_allows_one_proposal_to_match_multiple_gt():
    boxes = torch.zeros((1, 9), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    boxes[0, 0] = 0.4
    proposals = ProposalBatch(
        boxes=[boxes], scores=[torch.tensor([0.8])],
        labels=[torch.tensor([0])])
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 4, 2, 1, 0, 0, 0],
        [0.7, 0, 0, 4, 2, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([0, 0])]

    result = _center_value_matcher(
        selection='gt_nearest', one_to_one=False)(
            proposals, gt_boxes, gt_labels)

    assert result.matched_mask[0].tolist() == [True, True]
    assert result.matched_proposal_indices[0].tolist() == [0, 0]
    assert torch.allclose(
        result.center_distances[0], torch.tensor([0.4, 0.3]))
    assert torch.allclose(
        result.teacher_values[0],
        torch.tensor([1 - (0.4 / 2) ** 2, 1 - (0.3 / 2) ** 2]))


def test_center_value_matcher_uses_strict_radius_and_rejects_wrong_class():
    boxes = torch.zeros((2, 9), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    boxes[0, 0] = 1.0
    boxes[1, 0] = 0.0
    proposals = ProposalBatch(
        boxes=[boxes],
        scores=[torch.tensor([0.9, 0.99])],
        # bicycle and motorcycle belong to the same CenterPoint task, but the
        # second proposal must not be a candidate for a bicycle GT.
        labels=[torch.tensor([7, 6])],
    )
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 1, 1, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([7])]

    result = _center_value_matcher()(proposals, gt_boxes, gt_labels)

    assert result.effective_gt_mask[0].tolist() == [True]
    assert result.matched_mask[0].tolist() == [False]
    assert result.teacher_values[0].tolist() == [0.0]
    assert torch.isinf(result.center_distances[0][0])


def test_center_value_and_official_range_are_invariant_to_bda_scale():
    scale = 1.05
    boxes = torch.zeros((2, 9), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    boxes[0, 0] = 0.5 * scale
    boxes[1, 0] = 30.0 * scale
    proposals = ProposalBatch(
        boxes=[boxes],
        scores=[torch.tensor([0.8, 0.9])],
        labels=[torch.tensor([7, 9])],
    )
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 1, 1, 1, 0, 0, 0],
        [30.0 * scale, 0, 0, 1, 1, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([7, 9])]
    bda = torch.eye(4).unsqueeze(0)
    bda[0, 0, 0] = scale
    bda[0, 1, 1] = scale

    result = _center_value_matcher()(
        proposals, gt_boxes, gt_labels, bda_mats=bda)

    assert result.effective_gt_mask[0].tolist() == [True, True]
    assert result.matched_mask[0].tolist() == [True, True]
    assert result.center_distances[0][0] == torch.tensor(0.5)
    assert result.teacher_values[0][0] == torch.tensor(0.75)
    assert result.teacher_values[0][1] == torch.tensor(1.0)


def test_continuous_match_result_scaler_uses_one_geometry_path_only():
    boxes = torch.zeros((1, 9), dtype=torch.float32)
    boxes[:, 0] = 0.5
    boxes[:, 3:6] = 1.0
    proposals = ProposalBatch(
        boxes=[boxes], scores=[torch.tensor([0.8])],
        labels=[torch.tensor([0])])
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 4, 2, 1, 0, 0, 0],
        [10.0, 0, 0, 4, 2, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([0, 0])]
    result = _center_value_matcher()(proposals, gt_boxes, gt_labels)

    scaled = AdaptiveGTScalerV3().forward(result)

    assert scaled.gt_boxes[0][0, 3] > result.gt_boxes[0][0, 3]
    assert torch.equal(scaled.gt_boxes[0][1], result.gt_boxes[0][1])
    assert scaled.teacher_values is result.teacher_values


def test_uniform_gt_values_do_not_depend_on_teacher_proposals():
    matcher = _center_value_matcher(value_type='uniform_gt')
    empty = ProposalBatch(
        boxes=[torch.zeros((0, 9))],
        scores=[torch.zeros(0)],
        labels=[torch.zeros(0, dtype=torch.long)],
    )
    noisy = ProposalBatch(
        boxes=[torch.tensor([[20.0, 20.0, 0, 1, 1, 1, 0, 0, 0]])],
        scores=[torch.tensor([0.99])],
        labels=[torch.tensor([9])],
    )
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 4, 2, 1, 0, 0, 0],
        [51.0, 0, 0, 4, 2, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([0, 0])]

    first = matcher(empty, gt_boxes, gt_labels)
    second = matcher(noisy, gt_boxes, gt_labels)

    assert first.effective_gt_mask[0].tolist() == [True, False]
    assert first.teacher_values[0].tolist() == [1.0, 0.0]
    assert torch.equal(
        first.teacher_values[0], second.teacher_values[0])
    assert not first.matched_mask[0].any()


def test_uniform_gt_keeps_matching_metadata_for_response_bbox_gating():
    matcher = _center_value_matcher(value_type='uniform_gt')
    proposals = ProposalBatch(
        boxes=[torch.tensor([[0.5, 0.0, 0, 1, 1, 1, 0, 0, 0]])],
        scores=[torch.tensor([0.9])],
        labels=[torch.tensor([0])],
    )
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 4, 2, 1, 0, 0, 0],
        [10.0, 0, 0, 4, 2, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([0, 0])]

    result = matcher(proposals, gt_boxes, gt_labels)

    assert result.teacher_values[0].tolist() == [1.0, 1.0]
    assert result.matched_mask[0].tolist() == [True, False]


def test_non_positive_gt_size_is_not_effective():
    matcher = _center_value_matcher(value_type='uniform_gt')
    proposals = ProposalBatch(
        boxes=[torch.zeros((0, 9))],
        scores=[torch.zeros(0)],
        labels=[torch.zeros(0, dtype=torch.long)],
    )
    gt_boxes = [torch.tensor([
        [0.0, 0, 0, 4, 2, 1, 0, 0, 0],
        [0.0, 0, 0, 0, 2, 1, 0, 0, 0],
        [0.0, 0, 0, 4, -1, 1, 0, 0, 0],
    ])]
    gt_labels = [torch.tensor([0, 0, 0])]

    result = matcher(proposals, gt_boxes, gt_labels)

    assert result.effective_gt_mask[0].tolist() == [True, False, False]
    assert result.teacher_values[0].tolist() == [1.0, 0.0, 0.0]
