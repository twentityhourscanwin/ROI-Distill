from pathlib import Path

import torch

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config
from labeldistill.refine_head.target_assigner.velocity_only_half_scaler import VelocityOnlyHalfScaler
from labeldistill.refine_head.target_assigner.raw_gaussian_feature_loss import RawGaussianUnionFeatureLoss
from test_b_experiment_configs import _scientific_diff
from test_raw_gaussian_feature_loss import _match_result

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    'A': 'b1_teacher_value_no_scale.yaml',
    'B': 'b2_speed_half_centered_circular.yaml',
    'C': 'r1_teacher_value_radius_cap.yaml',
    'D': 's1_speed_half_radius_cap.yaml',
}


def test_factorial_configs_change_only_the_two_intended_factors():
    bundles = {k: load_and_resolve_config(
        ROOT / 'configs/experiments' / name, project_root=ROOT,
        require_checkpoint=False) for k, name in CONFIGS.items()}
    cfg = {k: v.config for k, v in bundles.items()}
    speed = {'region.scaler.type', 'region.scaler.enabled'}
    radius = {'region.mask.min_radius', 'region.mask.max_radius'}
    assert _scientific_diff(cfg['A'], cfg['B']) == speed
    assert _scientific_diff(cfg['C'], cfg['D']) == speed
    assert _scientific_diff(cfg['A'], cfg['C']) == radius
    assert _scientific_diff(cfg['B'], cfg['D']) == radius
    assert _scientific_diff(cfg['A'], cfg['D']) == speed | radius
    for config in cfg.values():
        assert config.region.scaler.center_mode == 'fixed'
        assert config.region.scaler.displacement_fraction == 0.5
        assert config.loss.feature_weight == 0.6
        assert config.loss.response_bbox_scope == 'matched_gt'
        assert config.derived.global_batch_size == 256
        assert config.experiment.seed == 0
    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    captured = {}
    class DummyExperiment:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    build_experiment(bundles['D'], model_cls=DummyModel, experiment_cls=DummyExperiment)
    assert isinstance(captured['scaler'], VelocityOnlyHalfScaler)
    assert captured['scaler'].center_mode == 'fixed'
    assert captured['feature_loss_reducer'].min_radius == 1
    assert captured['feature_loss_reducer'].max_radius == 2


def test_speed_can_cross_new_radius_threshold_while_old_mask_stays_unchanged():
    match = _match_result([0.8], sizes=[3.2])
    match.gt_boxes[0][0, 7] = 10.0
    expanded = VelocityOnlyHalfScaler().forward(match)
    assert torch.equal(expanded.gt_boxes[0][:, :3], match.gt_boxes[0][:, :3])
    assert expanded.teacher_values is match.teacher_values
    assert expanded.matched_mask is match.matched_mask
    kwargs = dict(point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
                  feature_map_size=[128, 128])
    old = RawGaussianUnionFeatureLoss(**kwargs, min_radius=2)
    cap = RawGaussianUnionFeatureLoss(**kwargs, min_radius=1, max_radius=2)
    a = old._draw_base_mask(match.gt_boxes[0][0], device='cpu')
    b = old._draw_base_mask(expanded.gt_boxes[0][0], device='cpu')
    c = cap._draw_base_mask(match.gt_boxes[0][0], device='cpu')
    d = cap._draw_base_mask(expanded.gt_boxes[0][0], device='cpu')
    assert torch.equal(a, b)
    assert not torch.equal(c, d)
    assert torch.count_nonzero(c) == 9
    assert torch.count_nonzero(d) == 25
    assert torch.all(d >= c)
    students = [torch.linspace(0, 1, n*n).reshape(1, 1, n, n).requires_grad_()
                for n in (128, 64)]
    teachers = [torch.zeros_like(s) for s in students]
    before = cap(teachers, students, match).loss
    after = cap(teachers, students, expanded).loss
    g0 = torch.autograd.grad(before, students, retain_graph=True)
    g1 = torch.autograd.grad(after, students)
    for initial, changed in zip(g0, g1):
        assert torch.isfinite(changed).all()
        assert not torch.equal(initial, changed)
