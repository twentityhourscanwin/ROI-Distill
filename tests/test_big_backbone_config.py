from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import LightningModule

from labeldistill.builders import build_experiment
from labeldistill.builders import trainer_builder
from labeldistill.config import ConfigValidationError, load_and_resolve_config
from labeldistill.experiments.j4 import J4Experiment
from labeldistill.presets import build_j4_model_configs
from torch.optim.lr_scheduler import LambdaLR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / 'configs/experiments/convnextb_896x1600_j4_cbgs.yaml'


def _bundle(overrides=()):
    return load_and_resolve_config(
        CONFIG,
        list(overrides),
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )


def test_convnextb_leaderboard_yaml_owns_its_training_policy():
    config = _bundle().config
    assert config.student.type == 'camera_bevdepth_convnextb'
    assert list(config.student.image.final_size) == [896, 1600]
    assert config.student.image.gradient_checkpointing is True
    assert config.student.image.drop_path_rate == pytest.approx(0.3)
    assert config.data.split == 'trainval'
    assert config.data.use_cbgs is True
    assert config.runtime.max_epochs == 20
    assert config.runtime.gradient_clip_val == pytest.approx(5.0)
    assert config.runtime.ddp_static_graph is True
    assert config.runtime.find_unused_parameters is False
    assert config.ema.enabled is False
    assert config.optimizer.base_lr_at_global_batch_64 == pytest.approx(2e-4)
    assert config.optimizer.backbone_lr_mult == pytest.approx(0.1)
    assert config.scheduler.type == 'LinearWarmupCosine'
    assert config.scheduler.warmup_steps == 1000
    assert config.loss.response_weight == pytest.approx(0.6)
    assert config.teacher.proposal.score_threshold == pytest.approx(0.15)
    assert config.derived.global_batch_size == 16
    assert config.derived.effective_learning_rate == pytest.approx(5e-5)


def test_convnextb_preset_is_selected_by_yaml_and_keeps_common_bev_contracts():
    bundle = _bundle()
    configs = build_j4_model_configs(bundle.config)
    image_backbone = configs.backbone['img_backbone_conf']
    assert image_backbone['type'] == 'TorchvisionConvNeXt'
    assert image_backbone['arch'] == 'base'
    assert image_backbone['with_cp'] is True
    assert image_backbone['pretrained'] is True
    assert configs.backbone['final_dim'] == (896, 1600)
    assert configs.ida_aug['H'] == 900
    assert configs.ida_aug['W'] == 1600
    assert configs.ida_aug['resize_lim'] == (1.0, 1.1)
    assert configs.head['bev_backbone_conf']['in_channels'] == 750
    assert configs.teacher['voxel_layer']['point_cloud_range'] == list(
        bundle.config.geometry.point_cloud_range)


def test_resume_skips_redundant_convnext_pretrained_download():
    bundle = _bundle(['runtime.resume_from=/tmp/convnext.ckpt'])
    configs = build_j4_model_configs(bundle.config)
    assert configs.backbone['img_backbone_conf']['pretrained'] is False


def test_builder_passes_convnext_data_view_without_importing_legacy_entry():
    bundle = _bundle()
    captured = {}

    class DummyModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class DummyExperiment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    build_experiment(
        bundle, model_cls=DummyModel, experiment_cls=DummyExperiment)
    assert captured['ida_aug_conf']['final_dim'] == (896, 1600)
    assert captured['img_conf']['to_rgb'] is True
    assert captured['config'].student.type == 'camera_bevdepth_convnextb'


def test_convnextb_ddp_policy_reaches_trainer(monkeypatch):
    config = _bundle().config
    captured = {}

    monkeypatch.setattr(
        trainer_builder,
        'DDPStrategy',
        lambda **kwargs: ('ddp', kwargs),
    )
    monkeypatch.setattr(
        trainer_builder,
        'ModelCheckpoint',
        lambda **kwargs: ('checkpoint', kwargs),
    )
    monkeypatch.setattr(
        trainer_builder.pl,
        'Trainer',
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    trainer_builder.build_trainer(config, object())
    assert captured['strategy'][1]['static_graph'] is True
    assert captured['strategy'][1]['find_unused_parameters'] is False


def test_linear_warmup_cosine_scheduler_is_step_based():
    experiment = J4Experiment.__new__(J4Experiment)
    LightningModule.__init__(experiment)
    experiment.model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    experiment.effective_learning_rate = 5e-5
    experiment.optimizer_weight_decay = 0.01
    experiment.backbone_lr_mult = 0.1
    experiment.scheduler_type = 'LinearWarmupCosine'
    experiment.scheduler_milestones = []
    experiment.scheduler_warmup_steps = 1000
    experiment.scheduler_warmup_ratio = 0.001
    experiment.scheduler_min_lr_ratio = 0.0
    experiment._trainer = SimpleNamespace(estimated_stepping_batches=2000)

    configured = experiment.configure_optimizers()
    scheduler_config = configured['lr_scheduler']
    assert isinstance(scheduler_config['scheduler'], LambdaLR)
    assert scheduler_config['interval'] == 'step'
    assert scheduler_config['scheduler'].lr_lambdas[0](0) == pytest.approx(0.001)
    assert scheduler_config['scheduler'].lr_lambdas[0](1000) == pytest.approx(1.0)
    assert scheduler_config['scheduler'].lr_lambdas[0](2000) == pytest.approx(0.0)


def test_convnextb_rejects_non_aligned_final_size():
    with pytest.raises(ConfigValidationError, match='divisible by 32'):
        _bundle(['student.image.final_size=[900,1600]'])
