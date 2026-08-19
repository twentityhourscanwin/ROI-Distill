from pathlib import Path

import pytest
import torch

from labeldistill.builders import build_experiment
from labeldistill.builders import trainer_builder
from labeldistill.config import load_and_resolve_config
from labeldistill.exps.nuscenes.ablation_param import param_J4_wl05_wh08


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_builder_wires_every_j4_policy_without_legacy_entry():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml',
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured['model_args'] = args
            captured['model_kwargs'] = kwargs

    class DummyExperiment:
        def __init__(self, **kwargs):
            captured['experiment_kwargs'] = kwargs

    experiment = build_experiment(
        bundle, model_cls=DummyModel, experiment_cls=DummyExperiment)
    assert isinstance(experiment, DummyExperiment)

    config = bundle.config
    model_kwargs = captured['model_kwargs']
    assert model_kwargs['lidar_checkpoint_prefix'] == config.teacher.checkpoint_prefix
    assert model_kwargs['temporal_kd_selection'] == 'legacy_half'
    assert model_kwargs['distill_feature_channels'] == [128, 256]
    assert model_kwargs['structured_output'] is True
    backbone, head, teacher, checkpoint = captured['model_args']
    assert checkpoint.endswith('ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth')
    assert backbone['output_channels'] == 150
    assert head['bev_backbone_conf']['in_channels'] == 750
    assert teacher['voxel_layer']['point_cloud_range'] == list(
        config.geometry.point_cloud_range)

    experiment_kwargs = captured['experiment_kwargs']
    matcher = experiment_kwargs['matcher']
    scaler = experiment_kwargs['scaler']
    mask = experiment_kwargs['mask_generator']
    assert matcher.distance_thresholds[0] == {'high': 2.0, 'medium': 4.0}
    assert matcher.class_policy == 'same_task_group'
    assert matcher.selection == 'highest_score'
    assert scaler.distance_thresholds == matcher.distance_thresholds
    assert mask.w_l == pytest.approx(config.region.mask.w_low)
    assert mask.w_h == pytest.approx(config.region.mask.w_high)
    assert mask.feature_map_size == (128, 128)
    assert mask.gaussian_overlap == pytest.approx(0.1)
    assert mask.min_radius == 2


def test_gradient_accumulation_is_part_of_effective_batch_and_lr():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml',
        ['runtime.accumulate_grad_batches=2'],
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    assert bundle.config.derived.global_batch_size == 64
    assert bundle.config.derived.effective_learning_rate == pytest.approx(4e-4)


def test_builder_model_preset_exactly_matches_legacy_j4(monkeypatch, tmp_path):
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml',
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    legacy_calls = []
    builder_calls = []

    class LegacyDummy(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            legacy_calls.append((args, kwargs))

    class BuilderDummy(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            builder_calls.append((args, kwargs))

    class DummyExperiment:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(param_J4_wl05_wh08, 'LabelDistill', LegacyDummy)
    param_J4_wl05_wh08.LabelDistillModel(
        gpus=2,
        batch_size_per_device=16,
        default_root_dir=str(tmp_path),
    )
    build_experiment(
        bundle, model_cls=BuilderDummy, experiment_cls=DummyExperiment)

    legacy_args, legacy_kwargs = legacy_calls[0]
    builder_args, builder_kwargs = builder_calls[0]
    assert builder_args[:3] == legacy_args[:3]
    assert builder_kwargs['teacher_proposal_cfg'] == legacy_kwargs[
        'teacher_proposal_cfg']
    assert builder_kwargs['is_train_depth'] is True
    assert builder_kwargs['structured_output'] is True


def test_trainer_builder_consumes_runtime_policy(monkeypatch):
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml',
        ['runtime.gpus=1', 'runtime.accumulate_grad_batches=3',
         'runtime.limit_train_batches=2', 'ema.enabled=false'],
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    captured = {}

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured['trainer'] = kwargs

    monkeypatch.setattr(trainer_builder.pl, 'Trainer', DummyTrainer)
    monkeypatch.setattr(
        trainer_builder, 'ModelCheckpoint',
        lambda **kwargs: ('checkpoint', kwargs))
    trainer = trainer_builder.build_trainer(bundle.config, object())
    assert isinstance(trainer, DummyTrainer)
    kwargs = captured['trainer']
    assert kwargs['devices'] == 1
    assert kwargs['strategy'] == 'auto'
    assert kwargs['accumulate_grad_batches'] == 3
    assert kwargs['limit_train_batches'] == 2
    assert kwargs['deterministic'] is True
    assert kwargs['precision'] == '16-mixed'
    assert kwargs['callbacks'][0][0] == 'checkpoint'


def test_real_j4_experiment_embeds_config_without_serializing_components(tmp_path):
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml',
        [f'runtime.output_dir={tmp_path.as_posix()}'],
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    experiment = build_experiment(bundle, model_cls=DummyModel)
    assert experiment.hparams['resolved_config']['experiment']['name'] == (
        'j4_wl05_wh07')
    for component_name in ('model', 'matcher', 'scaler', 'mask_generator'):
        assert component_name not in experiment.hparams

    checkpoint = {}
    experiment.on_save_checkpoint(checkpoint)
    assert checkpoint['labeldistill_schema_version'] == 1
    assert checkpoint['labeldistill_resolved_config']['derived'][
        'feature_map_size'] == [128, 128]
