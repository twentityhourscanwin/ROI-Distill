import math

import pytest
import torch

from labeldistill.refine_head.target_assigner.roi_distill import MatchResult
from labeldistill.refine_head.target_assigner.velocity_only_half_scaler import (
    VelocityOnlyHalfScaler,
)


def make_gt(*, yaw=0.0, vx=0.0, vy=0.0, x=0.0, y=0.0):
    return torch.tensor(
        [[x, y, 0.0, 4.0, 2.0, 1.5, yaw, vx, vy, 0.0]],
        dtype=torch.float32,
    )


def make_match_result(gt_boxes, effective_mask):
    count = len(gt_boxes)
    return MatchResult(
        gt_boxes=[gt_boxes],
        quality=[torch.zeros(count, dtype=torch.long)],
        matched_rois=[torch.zeros((count, 9))],
        roi_scores=[torch.zeros(count)],
        roi_labels=[torch.full((count,), -1, dtype=torch.long)],
        center_distances=[torch.full((count,), float('inf'))],
        effective_gt_mask=[effective_mask],
    )


def test_fixed_mode_expands_longitudinal_size_without_center_shift():
    scaler = VelocityOnlyHalfScaler(center_mode='fixed')
    gt = make_gt(vx=10.0)

    scaled = scaler.scale_boxes(gt)

    assert scaled[0, 3].item() == pytest.approx(6.25)
    assert scaled[0, 4].item() == pytest.approx(2.0)
    assert scaled[0, :3].tolist() == pytest.approx(gt[0, :3].tolist())


def test_velocity_sign_changes_no_fixed_mode_geometry():
    scaler = VelocityOnlyHalfScaler(center_mode='fixed')
    positive = scaler.scale_boxes(make_gt(vx=10.0))
    negative = scaler.scale_boxes(make_gt(vx=-10.0))

    assert positive[:, :7] == pytest.approx(negative[:, :7])


def test_velocity_is_projected_into_rotated_box_width():
    scaler = VelocityOnlyHalfScaler(center_mode='fixed')
    scaled = scaler.scale_boxes(make_gt(yaw=math.pi / 2, vx=10.0))

    assert scaled[0, 3].item() == pytest.approx(4.0)
    assert scaled[0, 4].item() == pytest.approx(4.25)


def test_temporal_midpoint_mode_preserves_asymmetric_time_direction():
    scaler = VelocityOnlyHalfScaler(center_mode='temporal_midpoint')
    positive = scaler.scale_boxes(make_gt(vx=10.0))
    negative = scaler.scale_boxes(make_gt(vx=-10.0))

    assert positive[0, 3].item() == pytest.approx(6.25)
    assert positive[0, 0].item() == pytest.approx(-0.125)
    assert negative[0, 0].item() == pytest.approx(0.125)


def test_nonfinite_motion_keeps_geometry_unchanged():
    scaler = VelocityOnlyHalfScaler(center_mode='temporal_midpoint')
    gt = make_gt(vx=float('nan'), vy=2.0)

    scaled = scaler.scale_boxes(gt)

    assert torch.equal(scaled[:, :7], gt[:, :7])


def test_forward_scales_only_effective_gt_rows():
    scaler = VelocityOnlyHalfScaler(center_mode='fixed')
    gt_boxes = torch.cat([make_gt(vx=10.0), make_gt(vx=10.0, x=5.0)])
    result = make_match_result(
        gt_boxes, torch.tensor([True, False], dtype=torch.bool))

    scaled = scaler.forward(result)

    assert scaled.gt_boxes[0][0, 3].item() == pytest.approx(6.25)
    assert torch.equal(scaled.gt_boxes[0][1], gt_boxes[1])
    assert torch.equal(result.gt_boxes[0], gt_boxes)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'displacement_fraction': 0.0},
        {'displacement_fraction': 1.1},
        {'past_time_seconds': -0.1},
        {'future_time_seconds': -0.1},
        {'past_time_seconds': 0.0, 'future_time_seconds': 0.0},
        {'center_mode': 'proposal'},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        VelocityOnlyHalfScaler(**kwargs)
