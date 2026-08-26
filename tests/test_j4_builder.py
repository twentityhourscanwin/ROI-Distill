from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from labeldistill.builders import build_experiment
from labeldistill.builders import trainer_builder
from labeldistill.config import load_and_resolve_config
from labeldistill.experiments.j4 import J4Experiment


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
    assert checkpoint.endswith(
        'ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_'
        'cyclic_20e_nus_20220810_030004-9061688e.pth')
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


def test_builder_wires_center_value_matcher_and_per_gt_reducer():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/b1_teacher_value_no_scale.yaml',
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class DummyExperiment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    build_experiment(
        bundle, model_cls=DummyModel, experiment_cls=DummyExperiment)

    matcher = captured['matcher']
    reducer = captured['feature_loss_reducer']
    assert matcher.matching_type == 'scale_conditioned_center_distance'
    assert matcher.class_policy == 'exact_class'
    assert matcher.one_to_one is True
    assert matcher.strict_less_than is True
    assert matcher.value_type == 'normalized_squared_margin'
    assert matcher._trust_radius_lookup.tolist() == [
        2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    assert captured['mask_generator'] is None
    assert reducer.feature_map_size == (128, 128)
    assert reducer.mask_type == 'per_gt_gaussian'
    assert reducer.overlap_merge == 'max'
    assert 7 in reducer.small_class_ids


def test_gradient_accumulation_is_part_of_effective_batch_and_lr():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml',
        ['runtime.accumulate_grad_batches=2'],
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    assert bundle.config.derived.global_batch_size == 64
    assert bundle.config.derived.effective_learning_rate == pytest.approx(4e-4)


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
    assert kwargs['callbacks'][0][1]['dirpath'] == (
        '/mnt/nas_data/guqiupeng/checkpoint_nes/j4_wl05_wh07')


def test_trainer_builder_evaluation_disables_training_callbacks(monkeypatch):
    bundle = load_and_resolve_config(
        PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml',
        ['runtime.gpus=1', 'runtime.limit_test_batches=2'],
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    captured = {}

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(trainer_builder.pl, 'Trainer', DummyTrainer)
    trainer_builder.build_trainer(bundle.config, object(), for_evaluation=True)
    assert captured['enable_checkpointing'] is False
    assert captured['callbacks'] == []
    assert captured['limit_test_batches'] == 2


def test_prediction_only_test_epoch_exports_without_metric_evaluator(tmp_path):
    class Dummy:
        config = OmegaConf.create({
            'evaluation': {'run_metrics': False},
            'runtime': {'output_dir': str(tmp_path)},
        })
        _test_step_outputs = [[['prediction']]]

    dummy = Dummy()
    J4Experiment.on_test_epoch_end(dummy)
    exported = torch.load(tmp_path / 'predictions_rank_0.pt')
    assert exported == [[['prediction']]]
    assert dummy._test_step_outputs == []


def test_legacy_checkpoint_drops_only_exact_duplicate_teacher_keys():
    shared = torch.tensor([1.0, 2.0])
    checkpoint = {'state_dict': {
        'model.student.weight': torch.tensor([3.0]),
        'model.centerpoint.layer.weight': shared.clone(),
        'centerpoint.layer.weight': shared.clone(),
    }}
    J4Experiment.on_load_checkpoint(object(), checkpoint)
    assert set(checkpoint['state_dict']) == {
        'model.student.weight', 'model.centerpoint.layer.weight'}


def test_legacy_checkpoint_rejects_conflicting_duplicate_teacher_keys():
    checkpoint = {'state_dict': {
        'model.centerpoint.layer.weight': torch.tensor([1.0]),
        'centerpoint.layer.weight': torch.tensor([2.0]),
    }}
    with pytest.raises(RuntimeError, match='conflicting duplicated teacher'):
        J4Experiment.on_load_checkpoint(object(), checkpoint)


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
