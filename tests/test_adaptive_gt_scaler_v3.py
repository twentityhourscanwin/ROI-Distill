import math

import pytest
import torch

from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import (
    AdaptiveGTScalerV3,
)


def make_gt(*, yaw=0.0, vx=0.0, vy=0.0, x=0.0, y=0.0):
    # [x, y, z, length, width, height, yaw, vx, vy, class_id]
    return torch.tensor(
        [[x, y, 0.0, 4.0, 2.0, 1.5, yaw, vx, vy, 0.0]],
        dtype=torch.float32,
    )


def make_roi(x, y):
    return torch.tensor(
        [[x, y, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )


def make_scaler(*, mu=1.0, s_vel=0.0):
    return AdaptiveGTScalerV3(
        mu=mu,
        s_vel=s_vel,
        r_max=50.0,
    )


def test_medium_yaw_zero_global_x_expands_length():
    scaler = make_scaler()
    scaled = scaler.scale_medium_quality(
        make_gt(yaw=0.0),
        make_roi(1.0, 0.0),
    )
    assert scaled[0, 3].item() == pytest.approx(4.5)
    assert scaled[0, 4].item() == pytest.approx(2.0)


def test_medium_yaw_90_global_y_expands_local_length():
    scaler = make_scaler()
    scaled = scaler.scale_medium_quality(
        make_gt(yaw=math.pi / 2),
        make_roi(0.0, 1.0),
    )
    assert scaled[0, 3].item() == pytest.approx(4.5)
    assert scaled[0, 4].item() == pytest.approx(2.0)


def test_medium_yaw_90_global_x_expands_local_width():
    scaler = make_scaler()
    scaled = scaler.scale_medium_quality(
        make_gt(yaw=math.pi / 2),
        make_roi(1.0, 0.0),
    )
    assert scaled[0, 3].item() == pytest.approx(4.0)
    assert scaled[0, 4].item() == pytest.approx(2.5)


def test_high_yaw_90_global_x_velocity_compensates_local_width():
    scaler = make_scaler(mu=0.0, s_vel=0.2)
    scaled = scaler.scale_high_quality(
        make_gt(yaw=math.pi / 2, vx=2.0, vy=0.0),
        make_roi(0.0, 0.0),
    )
    assert scaled[0, 3].item() == pytest.approx(4.0)
    assert scaled[0, 4].item() == pytest.approx(2.4)


def test_high_yaw_90_global_y_velocity_compensates_local_length():
    scaler = make_scaler(mu=0.0, s_vel=0.2)
    scaled = scaler.scale_high_quality(
        make_gt(yaw=math.pi / 2, vx=0.0, vy=2.0),
        make_roi(0.0, 0.0),
    )
    assert scaled[0, 3].item() == pytest.approx(4.8)
    assert scaled[0, 4].item() == pytest.approx(2.0)


def test_unmatched_uses_local_velocity_before_distance_decay():
    scaler = make_scaler(mu=0.0, s_vel=0.2)
    scaled = scaler.scale_unmatched_gt(
        make_gt(yaw=math.pi / 2, vx=0.0, vy=2.0, x=0.0, y=0.0)
    )
    assert scaled[0, 3].item() == pytest.approx(4.8)
    assert scaled[0, 4].item() == pytest.approx(2.0)


def test_global_to_local_preserves_vector_norm_at_arbitrary_yaw():
    x_global, y_global = 2.0, -3.0
    x_local, y_local = AdaptiveGTScalerV3._global_to_local(
        x_global,
        y_global,
        math.pi / 4,
    )
    assert math.hypot(x_local, y_local) == pytest.approx(
        math.hypot(x_global, y_global)
    )

ARBITRARY_YAWS = [
    math.radians(angle)
    for angle in (-179, -135, -45, -30, 27, 45, 120, 179)
]


def local_to_global(x_local, y_local, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x_global = x_local * cos_yaw - y_local * sin_yaw
    y_global = x_local * sin_yaw + y_local * cos_yaw
    return x_global, y_global


@pytest.mark.parametrize('yaw', ARBITRARY_YAWS)
def test_medium_arbitrary_yaw_local_x_expands_only_length(yaw):
    scaler = make_scaler()
    roi_x, roi_y = local_to_global(1.0, 0.0, yaw)

    scaled = scaler.scale_medium_quality(
        make_gt(yaw=yaw),
        make_roi(roi_x, roi_y),
    )

    assert scaled[0, 3].item() == pytest.approx(4.5)
    assert scaled[0, 4].item() == pytest.approx(2.0)


@pytest.mark.parametrize('yaw', ARBITRARY_YAWS)
def test_medium_arbitrary_yaw_local_y_expands_only_width(yaw):
    scaler = make_scaler()
    roi_x, roi_y = local_to_global(0.0, 1.0, yaw)

    scaled = scaler.scale_medium_quality(
        make_gt(yaw=yaw),
        make_roi(roi_x, roi_y),
    )

    assert scaled[0, 3].item() == pytest.approx(4.0)
    assert scaled[0, 4].item() == pytest.approx(2.5)


@pytest.mark.parametrize('yaw', ARBITRARY_YAWS)
def test_high_arbitrary_yaw_local_longitudinal_velocity_expands_length(yaw):
    scaler = make_scaler(mu=0.0, s_vel=0.2)
    vx, vy = local_to_global(2.0, 0.0, yaw)

    scaled = scaler.scale_high_quality(
        make_gt(yaw=yaw, vx=vx, vy=vy),
        make_roi(0.0, 0.0),
    )

    assert scaled[0, 3].item() == pytest.approx(4.8)
    assert scaled[0, 4].item() == pytest.approx(2.0)


@pytest.mark.parametrize('yaw', ARBITRARY_YAWS)
def test_high_arbitrary_yaw_local_lateral_velocity_expands_width(yaw):
    scaler = make_scaler(mu=0.0, s_vel=0.2)
    vx, vy = local_to_global(0.0, 2.0, yaw)

    scaled = scaler.scale_high_quality(
        make_gt(yaw=yaw, vx=vx, vy=vy),
        make_roi(0.0, 0.0),
    )

    assert scaled[0, 3].item() == pytest.approx(4.0)
    assert scaled[0, 4].item() == pytest.approx(2.4)
