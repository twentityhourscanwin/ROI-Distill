import math
from pathlib import Path

import pytest
import torch

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config
from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import AdaptiveGTScalerV3
from labeldistill.refine_head.target_assigner.proposal_center_only_scaler import ProposalCenterOnlyScaler
from test_raw_gaussian_feature_loss import _match_result, _reducer
from test_b_experiment_configs import _scientific_diff

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2, -0.73])
@pytest.mark.parametrize('offset', [-0.4, -0.05, 0.0, 0.05, 0.4])
def test_matches_legacy_offset_only_center_without_changing_other_fields(yaw, offset):
    gt = torch.tensor([[0.7, 0.0, 1.0, 4.0, 2.0, 1.5, yaw, 12.0, -3.0, 0.0]])
    roi = gt[:, :9].clone()
    roi[:, 0] += offset
    roi[:, 1] += 0.2
    saved = gt.clone()
    expected = AdaptiveGTScalerV3(mu=0.15, s_vel=0.0).scale_high_quality(gt, roi)
    actual = ProposalCenterOnlyScaler().scale_high_quality(gt, roi)
    assert torch.equal(actual[:, :2], expected[:, :2])
    assert torch.equal(actual[:, 2:], gt[:, 2:])
    assert torch.equal(gt, saved)


def test_velocity_independence_and_matching_contract():
    result = _match_result([0.8, 0.0], centers=[[0.7, 0.0], [1.0, 1.0]])
    result.gt_boxes[0][:, 3:5] = torch.tensor([4.0, 2.0])
    result.matched_rois[0][0, :2] = torch.tensor([1.5, 0.0])
    scaler = ProposalCenterOnlyScaler()
    stationary = scaler.forward(result)
    result.gt_boxes[0][:, 7:9] = torch.tensor([float('nan'), float('inf')])
    moving = scaler.forward(result)
    assert torch.equal(stationary.gt_boxes[0][:, :7], moving.gt_boxes[0][:, :7])
    assert torch.equal(moving.gt_boxes[0][1, :7], result.gt_boxes[0][1, :7])
    assert moving.teacher_values is result.teacher_values
    assert moving.matched_mask is result.matched_mask
    assert moving.matched_proposal_indices is result.matched_proposal_indices
    assert moving.effective_gt_mask is result.effective_gt_mask
    assert moving.gt_boxes[0][0, 0].item() == pytest.approx(1.0)


def test_center_crossing_changes_actual_mask_and_feature_gradient():
    result = _match_result([0.8], centers=[[0.9, 0.0]])
    result.gt_boxes[0][:, 3:5] = torch.tensor([4.0, 2.0])
    result.matched_rois[0][0, :2] = torch.tensor([1.7, 0.0])
    shifted = ProposalCenterOnlyScaler().forward(result)
    reducer = _reducer(min_radius=2)
    old = reducer._draw_base_mask(result.gt_boxes[0][0], device='cpu')
    new = reducer._draw_base_mask(shifted.gt_boxes[0][0], device='cpu')
    assert not torch.equal(old, new)
    student = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8).requires_grad_()
    teacher = torch.zeros_like(student)
    old_loss = reducer([teacher], [student], result).loss
    new_loss = reducer([teacher], [student], shifted).loss
    old_grad = torch.autograd.grad(old_loss, student, retain_graph=True)[0]
    new_grad = torch.autograd.grad(new_loss, student)[0]
    assert torch.isfinite(new_grad).all()
    assert not torch.equal(old_grad, new_grad)


def test_empty_gt():
    result = _match_result([])
    actual = ProposalCenterOnlyScaler().forward(result)
    assert actual.gt_boxes[0].shape == (0, 10)


def test_config_reaches_builder_and_has_only_intended_changes():
    def load(name, overrides=()):
        return load_and_resolve_config(ROOT / 'configs/experiments' / name,
                                      list(overrides), project_root=ROOT,
                                      require_checkpoint=False)
    bundle = load('c1_proposal_center_only.yaml')
    baseline = load('b1_teacher_value_no_scale.yaml')
    assert _scientific_diff(baseline.config, bundle.config) == {
        'region.scaler.type', 'region.scaler.enabled', 'region.scaler.velocity_scale'}
    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    captured = {}
    class DummyExperiment:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    build_experiment(bundle, model_cls=DummyModel, experiment_cls=DummyExperiment)
    assert isinstance(captured['scaler'], ProposalCenterOnlyScaler)
    assert captured['feature_loss_reducer'].min_radius == 2
    assert bundle.config.loss.response_bbox_scope == 'matched_gt'
    with pytest.raises(ValueError, match='velocity_scale=0'):
        load('c1_proposal_center_only.yaml', ['region.scaler.velocity_scale=0.2'])
