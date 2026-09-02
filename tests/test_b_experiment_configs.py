from pathlib import Path

import torch
from omegaconf import OmegaConf

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'configs/experiments'
TEACHER = (
    'ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth'
)
IDENTITY_PATHS = {
    'experiment.name', 'experiment.description', 'runtime.output_dir',
}


def _load(name):
    return load_and_resolve_config(
        CONFIG_DIR / name, project_root=PROJECT_ROOT,
        require_checkpoint=False).config


def _flatten(value, prefix=''):
    plain = OmegaConf.to_container(value, resolve=True)
    output = {}

    def visit(node, path):
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, f'{path}.{key}' if path else key)
        else:
            output[path] = node

    visit(plain, prefix)
    return output


def _scientific_diff(left, right):
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    return {
        path for path in left_flat
        if path not in IDENTITY_PATHS and left_flat[path] != right_flat[path]
    }


def test_b0_b1_b2_b1t_and_m1_are_controlled_relatives_of_b1():
    b0 = _load('b0_full_gt_uniform_no_scale.yaml')
    b1 = _load('b1_teacher_value_no_scale.yaml')
    b2 = _load('b2_teacher_value_adaptive_scale.yaml')
    b1t = _load('b1t_teacher_value_elliptical_mask.yaml')
    m1 = _load('m1_gt_nearest_reuse.yaml')

    for config in (b0, b1, b2, b1t, m1):
        assert config.teacher.checkpoint == TEACHER
        assert config.teacher.checkpoint_prefix == 'model.centerpoint.'
        assert config.data.train_info == 'nuscenes_infos_train.pkl'
        assert config.data.val_info == 'nuscenes_infos_val.pkl'
        assert config.region.mask.overlap_merge == 'max'
        assert config.region.mask.normalize_per_instance is False
        assert config.loss.feature_weight == 0.6
        assert config.runtime.gpus == 16
        assert config.runtime.batch_size_per_device == 16
        assert config.derived.global_batch_size == 256
        assert config.derived.effective_learning_rate == 0.0004
        assert config.optimizer.backbone_lr_mult == 1.0
        assert config.scheduler.warmup_steps == 200
        assert config.scheduler.warmup_ratio == 0.001

    assert _scientific_diff(b0, b1) == {
        'loss.response_bbox_scope',
        'region.value.type', 'region.value.unmatched_value'}
    assert b0.loss.response_bbox_scope == 'all_gt'
    assert b1.loss.response_bbox_scope == 'matched_gt'
    assert _scientific_diff(b1, b2) == {'region.scaler.enabled'}
    assert _scientific_diff(b1, b1t) == {'region.mask.type'}
    assert _scientific_diff(b1, m1) == {
        'matching.one_to_one', 'matching.selection'}


def test_b_experiment_configs_reach_real_builder_consumers():
    expected = {
        'b0_full_gt_uniform_no_scale.yaml': (
            'uniform_gt', False, 'per_gt_gaussian'),
        'b1_teacher_value_no_scale.yaml': (
            'normalized_squared_margin', False, 'per_gt_gaussian'),
        'b2_teacher_value_adaptive_scale.yaml': (
            'normalized_squared_margin', True, 'per_gt_gaussian'),
        'b1t_teacher_value_elliptical_mask.yaml': (
            'normalized_squared_margin', False,
            'per_gt_elliptical_gaussian'),
        'm1_gt_nearest_reuse.yaml': (
            'normalized_squared_margin', False, 'per_gt_gaussian'),
    }

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    for name, (value_type, scaler_enabled, mask_type) in expected.items():
        captured = {}

        class DummyExperiment:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        bundle = load_and_resolve_config(
            CONFIG_DIR / name, project_root=PROJECT_ROOT,
            require_checkpoint=False)
        build_experiment(
            bundle, model_cls=DummyModel,
            experiment_cls=DummyExperiment)

        assert captured['matcher'].value_type == value_type
        assert bundle.config.region.scaler.enabled is scaler_enabled
        assert captured['feature_loss_reducer'].mask_type == mask_type
        assert captured['feature_loss_reducer'].overlap_merge == 'max'
        if name == 'm1_gt_nearest_reuse.yaml':
            assert captured['matcher'].selection == 'gt_nearest'
            assert captured['matcher'].one_to_one is False
