from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config
from labeldistill.refine_head.target_assigner.raw_gaussian_feature_loss import RawGaussianUnionFeatureLoss
from test_raw_gaussian_feature_loss import _match_result
from test_b_experiment_configs import _scientific_diff

ROOT = Path(__file__).resolve().parents[1]


def reducer(**kwargs):
    params = dict(point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                  feature_map_size=[128, 128], min_radius=1, max_radius=2)
    params.update(kwargs)
    return RawGaussianUnionFeatureLoss(**params)


@pytest.mark.parametrize('size,nonzero', [(0.01, 9), (0.7, 9), (4.0, 25), (30.0, 25)])
def test_floor_and_cap_produce_nonempty_gaussians(size, nonzero):
    box = _match_result([1.0], sizes=[size]).gt_boxes[0][0]
    mask = reducer()._draw_base_mask(box, device='cpu')
    assert torch.count_nonzero(mask) == nonzero
    assert mask.max() == 1
    assert torch.isfinite(mask).all()


def test_radius_zero_disappears_at_cell_one_but_radius_one_survives():
    box = _match_result([1.0], sizes=[0.1], centers=[[-50.3, -50.3]]).gt_boxes[0][0]
    zero = reducer(min_radius=0)._draw_base_mask(box, device='cpu')
    one = reducer()._draw_base_mask(box, device='cpu')
    down = lambda m: F.interpolate(m[None, None], (64, 64), mode='bilinear', align_corners=True)
    assert zero.sum() > 0
    assert down(zero).sum() == 0
    assert down(one).sum() > 0


def test_radius_one_survives_all_center_phases_in_both_axes():
    # A radius-one Gaussian is separable; test each 1-D center including edges.
    impulses = torch.eye(128).reshape(128, 1, 128)
    kernel = torch.tensor([0.1353352832, 1.0, 0.1353352832]).reshape(1, 1, 3)
    masks = F.conv1d(impulses, kernel, padding=1)
    down = F.interpolate(masks, size=64, mode='linear', align_corners=True)
    assert torch.all(down.sum(dim=(1, 2)) > 0)


def test_two_level_loss_and_gradients_remain_valid_near_boundary():
    match = _match_result([0.7], sizes=[0.1], centers=[[-50.3, -50.3]])
    students = [torch.ones((1, 2, n, n), requires_grad=True) for n in (128, 64)]
    teachers = [torch.zeros_like(s) for s in students]
    output = reducer()(teachers, students, match)
    assert output.loss.item() == pytest.approx(4.0)
    output.loss.backward()
    for s in students:
        assert torch.isfinite(s.grad).all()
        assert s.grad.abs().sum() > 0


@pytest.mark.parametrize('kwargs', [dict(max_radius=0), dict(max_radius=1.5),
                                    dict(mask_type='per_gt_elliptical_gaussian')])
def test_invalid_caps_rejected(kwargs):
    with pytest.raises(ValueError):
        reducer(**kwargs)


def test_new_configs_preserve_baseline_and_reach_real_builder():
    def load(name, overrides=()):
        return load_and_resolve_config(ROOT / 'configs/experiments' / name,
                                      list(overrides), project_root=ROOT,
                                      require_checkpoint=False)
    baseline = load('b1_teacher_value_no_scale.yaml')
    r1 = load('r1_teacher_value_radius_cap.yaml')
    r2 = load('r2_adaptive_radius_cap.yaml')
    assert baseline.config.region.mask.max_radius is None
    assert baseline.config.region.mask.min_radius == 2
    assert _scientific_diff(baseline.config, r1.config) == {
        'region.mask.min_radius', 'region.mask.max_radius'}
    assert _scientific_diff(r1.config, r2.config) == {'region.scaler.enabled'}
    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    captured = {}
    class DummyExperiment:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    build_experiment(r1, model_cls=DummyModel, experiment_cls=DummyExperiment)
    assert captured['feature_loss_reducer'].min_radius == 1
    assert captured['feature_loss_reducer'].max_radius == 2
    with pytest.raises(ValueError, match='max_radius'):
        load('r1_teacher_value_radius_cap.yaml', ['region.mask.max_radius=0'])
    with pytest.raises(ValueError, match='max_radius'):
        load('r1_teacher_value_radius_cap.yaml', ['region.mask.type=per_gt_elliptical_gaussian'])
