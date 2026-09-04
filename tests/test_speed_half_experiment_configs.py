from pathlib import Path

import torch
from omegaconf import OmegaConf

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / 'configs/experiments'
IDENTITY_PATHS = {
    'experiment.name', 'experiment.description', 'runtime.output_dir',
}


def load(name):
    return load_and_resolve_config(
        CONFIG_DIR / name, project_root=PROJECT_ROOT,
        require_checkpoint=False)


def scientific_diff(left, right):
    def flatten(value):
        value = OmegaConf.to_container(value, resolve=True)
        output = {}

        def visit(node, path=''):
            if isinstance(node, dict):
                for key, child in node.items():
                    visit(child, f'{path}.{key}' if path else key)
            else:
                output[path] = node

        visit(value)
        return output

    left_flat = flatten(left)
    right_flat = flatten(right)
    return {
        key for key in left_flat
        if key not in IDENTITY_PATHS and left_flat[key] != right_flat[key]
    }


def test_b3_and_b3t_speed_half_configs_only_change_mask_shape():
    b1 = load('b1_teacher_value_no_scale.yaml').config
    b1t = load('b1t_teacher_value_elliptical_mask.yaml').config
    b3 = load('b3_speed_half_asymmetric_circular.yaml').config
    b3t = load('b3t_speed_half_asymmetric_elliptical.yaml').config

    assert b3.region.scaler.type == 'velocity_only_half'
    assert b3.region.scaler.enabled is True
    assert b3.region.scaler.displacement_fraction == 0.5
    assert b3.region.scaler.past_time_seconds == 0.25
    assert b3.region.scaler.future_time_seconds == 0.20
    assert b3.region.scaler.center_mode == 'temporal_midpoint'
    assert b3.region.mask.type == 'per_gt_gaussian'
    assert b3t.region.mask.type == 'per_gt_elliptical_gaussian'
    assert scientific_diff(b1, b3) == {
        'region.scaler.center_mode', 'region.scaler.enabled',
        'region.scaler.type'}
    assert scientific_diff(b1t, b3t) == {
        'region.scaler.center_mode', 'region.scaler.enabled',
        'region.scaler.type'}
    assert scientific_diff(b3, b3t) == {'region.mask.type'}


def test_speed_half_config_reaches_builder_scaler():
    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class DummyExperiment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    bundle = load('b3_speed_half_asymmetric_circular.yaml')
    build_experiment(
        bundle, model_cls=DummyModel, experiment_cls=DummyExperiment)

    scaler = captured['scaler']
    assert scaler.displacement_fraction == 0.5
    assert scaler.past_time_seconds == 0.25
    assert scaler.future_time_seconds == 0.20
    assert scaler.center_mode == 'temporal_midpoint'
