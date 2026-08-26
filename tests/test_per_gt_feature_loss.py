import math

import pytest
import torch

from labeldistill.refine_head.target_assigner.per_gt_feature_loss import (
    PerGTFeatureDistillationLoss,
)
from labeldistill.refine_head.target_assigner.roi_distill import (
    MatchQuality,
    MatchResult,
)


def _reducer(**overrides):
    kwargs = dict(
        point_cloud_range=[-4, -4, -2, 4, 4, 2],
        feature_map_size=[8, 8],
        gaussian_overlap=0.1,
        min_radius=0,
        small_class_ids=[5, 6, 7, 8, 9],
        overlap_merge='max',
    )
    kwargs.update(overrides)
    return PerGTFeatureDistillationLoss(
        **kwargs,
    )


def _match_result(values, *, sizes=None, effective=None, matched=None):
    count = len(values)
    boxes = torch.zeros((count, 10), dtype=torch.float32)
    boxes[:, 3:6] = 1.0
    if sizes is not None:
        boxes[:, 3] = torch.tensor(sizes, dtype=torch.float32)
        boxes[:, 4] = torch.tensor(sizes, dtype=torch.float32)
    boxes[:, -1] = 0
    value_tensor = torch.tensor(values, dtype=torch.float32)
    if matched is None:
        matched = value_tensor > 0
    else:
        matched = torch.tensor(matched, dtype=torch.bool)
    if effective is None:
        effective = torch.ones(count, dtype=torch.bool)
    return MatchResult(
        gt_boxes=[boxes],
        quality=[torch.where(
            matched,
            torch.full((count,), int(MatchQuality.HIGH)),
            torch.full((count,), int(MatchQuality.UNMATCHED)))],
        matched_rois=[torch.zeros((count, 9))],
        roi_scores=[torch.zeros(count)],
        roi_labels=[torch.where(
            matched, torch.zeros(count, dtype=torch.long),
            torch.full((count,), -1, dtype=torch.long))],
        center_distances=[torch.where(
            matched, torch.zeros(count),
            torch.full((count,), float('inf')))],
        matched_proposal_indices=[torch.where(
            matched, torch.arange(count),
            torch.full((count,), -1, dtype=torch.long))],
        matched_mask=[matched],
        effective_gt_mask=[effective],
        teacher_values=[value_tensor],
        trust_radii=[torch.full((count,), 2.0)],
    )


def _gradient_norm(value):
    student = torch.ones((1, 1, 8, 8), requires_grad=True)
    teacher = torch.zeros_like(student)
    output = _reducer()([teacher], [student], _match_result([value]))
    output.loss.backward()
    return output.loss.detach(), student.grad.norm()


def test_teacher_value_controls_absolute_feature_gradient():
    full_loss, full_gradient = _gradient_norm(1.0)
    partial_loss, partial_gradient = _gradient_norm(0.4)

    assert partial_loss == pytest.approx(0.4 * full_loss)
    assert partial_gradient == pytest.approx(0.4 * full_gradient)


def test_zero_value_gt_has_graph_connected_zero_gradient():
    loss, gradient = _gradient_norm(0.0)

    assert loss == 0
    assert gradient == 0


def test_per_instance_normalization_removes_gaussian_area_bias():
    student = torch.ones((1, 1, 8, 8), requires_grad=True)
    teacher = torch.zeros_like(student)

    small = _reducer()(
        [teacher], [student], _match_result([1.0], sizes=[1.0]))
    large = _reducer()(
        [teacher], [student], _match_result([1.0], sizes=[4.0]))

    assert small.loss.detach() == pytest.approx(1.0)
    assert large.loss.detach() == pytest.approx(1.0)


def test_every_feature_level_is_normalized_independently():
    student_8 = torch.ones((1, 1, 8, 8), requires_grad=True)
    student_4 = torch.ones((1, 1, 4, 4), requires_grad=True)
    output = _reducer()(
        [torch.zeros_like(student_8), torch.zeros_like(student_4)],
        [student_8, student_4],
        _match_result([1.0]),
    )

    assert output.level_losses[0].detach() == pytest.approx(1.0)
    assert output.level_losses[1].detach() == pytest.approx(1.0)
    assert output.loss.detach() == pytest.approx(2.0)


def test_empty_effective_gt_returns_connected_zero():
    student = torch.ones((1, 1, 8, 8), requires_grad=True)
    output = _reducer()(
        [torch.zeros_like(student)], [student], _match_result([]))

    output.loss.backward()
    assert output.loss == 0
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0


def test_undrawable_effective_gt_is_skipped_from_loss_and_statistics():
    student = torch.ones((1, 1, 8, 8), requires_grad=True)
    match_result = _match_result([1.0, 0.5], sizes=[1.0, 0.0])

    output = _reducer()(
        [torch.zeros_like(student)], [student], match_result)

    assert output.loss.detach() == pytest.approx(1.0)
    assert output.effective_gt_count == 1
    assert output.matched_gt_count == 1
    assert output.teacher_value_sum == 1
    assert output.skipped_gt_count == 1


def test_all_undrawable_effective_gts_return_connected_zero():
    student = torch.ones((1, 1, 8, 8), requires_grad=True)
    match_result = _match_result([1.0], sizes=[0.0])

    output = _reducer()(
        [torch.zeros_like(student)], [student], match_result)
    output.loss.backward()

    assert output.loss == 0
    assert output.effective_gt_count == 0
    assert output.skipped_gt_count == 1
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) == 0


def test_uniform_value_is_reduced_even_without_teacher_match():
    student = torch.ones((1, 1, 8, 8), requires_grad=True)
    output = _reducer()(
        [torch.zeros_like(student)], [student],
        _match_result([1.0], matched=[False]),
    )

    assert output.loss.detach() == pytest.approx(1.0)
    assert output.matched_gt_count == 0
    assert output.teacher_value_sum == 1


def test_identical_overlap_uses_pixelwise_max():
    student = torch.ones((1, 1, 8, 8), requires_grad=True)
    output = _reducer()(
        [torch.zeros_like(student)], [student], _match_result([1.0, 1.0]))

    # One unit-mass mask survives max merge, then fixed-count reduction / 2.
    assert output.loss.detach() == pytest.approx(0.5)


def test_elliptical_mask_tracks_box_aspect_and_yaw():
    reducer = _reducer(
        mask_type='per_gt_elliptical_gaussian', min_radius=2)
    box = torch.tensor([0, 0, 0, 4, 1, 1, 0, 0, 0, 0], dtype=torch.float32)

    horizontal = reducer._draw_base_mask(box, device=torch.device('cpu'))
    vertical_box = box.clone()
    vertical_box[6] = math.pi / 2
    vertical = reducer._draw_base_mask(
        vertical_box, device=torch.device('cpu'))

    hy, hx = torch.nonzero(horizontal, as_tuple=True)
    vy, vx = torch.nonzero(vertical, as_tuple=True)
    assert hx.max() - hx.min() > hy.max() - hy.min()
    assert vy.max() - vy.min() > vx.max() - vx.min()
