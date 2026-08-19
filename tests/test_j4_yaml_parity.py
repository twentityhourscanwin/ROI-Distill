"""Lock critical YAML values to the current legacy J4 implementation."""

from inspect import signature
from pathlib import Path

import pytest
import torch

from labeldistill.config import load_and_resolve_config
from labeldistill.exps.nuscenes import base_exp
from labeldistill.exps.nuscenes.ablation_param import param_J4_wl05_wh08
from labeldistill.models.lidardistill import LabelDistill


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_j4_yaml_matches_legacy_constructor(monkeypatch, tmp_path):
    model_calls = []

    class DummyLabelDistill(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            model_calls.append((args, kwargs))

    monkeypatch.setattr(base_exp, "BaseBEVDepth", lambda *args, **kwargs: None)
    monkeypatch.setattr(param_J4_wl05_wh08, "LabelDistill", DummyLabelDistill)
    legacy = param_J4_wl05_wh08.LabelDistillModel(
        gpus=2,
        batch_size_per_device=16,
        default_root_dir=str(tmp_path),
    )
    config = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    ).config

    assert len(model_calls) == 1
    args, kwargs = model_calls[0]
    backbone_conf, head_conf, lidar_conf, checkpoint = args
    proposal = kwargs["teacher_proposal_cfg"]
    bbox_coder = proposal["bbox_coder"]
    teacher_test_cfg = lidar_conf["pts_bbox_head"]["test_cfg"]["pts"]

    assert config.data.key_idxes == legacy.key_idxes
    assert config.student.output_channels == backbone_conf["output_channels"]
    assert config.derived.student_bev_input_channels == head_conf[
        "bev_backbone_conf"
    ]["in_channels"]
    assert config.student.temporal_kd_selection == signature(LabelDistill).parameters[
        "temporal_kd_selection"
    ].default
    assert config.teacher.checkpoint == checkpoint.removeprefix("./")
    assert config.teacher.proposal.score_threshold == bbox_coder["score_threshold"]
    assert config.teacher.proposal.max_num == bbox_coder["max_num"]
    assert teacher_test_cfg["pre_max_size"] == 1000
    assert config.teacher.proposal.post_max_size == proposal["post_max_size"]
    assert config.teacher.proposal.min_radius == proposal["min_radius"]
    assert bbox_coder["code_size"] == 9
    assert config.geometry.point_cloud_range == lidar_conf["voxel_layer"][
        "point_cloud_range"
    ]
    assert config.geometry.lidar_voxel_size == lidar_conf["voxel_layer"]["voxel_size"]
    assert config.classes.tasks == [
        task["class_names"] for task in lidar_conf["pts_bbox_head"]["tasks"]
    ]

    assert config.region.mask.w_low == legacy.generate_bev_mask.w_l
    assert config.region.mask.w_high == legacy.generate_bev_mask.w_h
    assert config.region.mask.max_distance == legacy.generate_bev_mask.r_max
    assert config.region.mask.gaussian_overlap == legacy.generate_bev_mask.gaussian_overlap
    assert config.region.mask.min_radius == legacy.generate_bev_mask.min_radius
    assert config.region.scaler.mu == legacy.change_gt.mu
    assert config.region.scaler.velocity_scale == legacy.change_gt.s_vel
    assert config.region.scaler.max_distance == legacy.change_gt.r_max


def test_j4_yaml_matches_legacy_optimizer_and_scheduler(monkeypatch, tmp_path):
    captured = {}

    class DummyLabelDistill(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1)

    def fake_build_optim_wrapper(model, wrapper_config):
        captured.update(wrapper_config)
        return torch.optim.AdamW(
            model.parameters(),
            lr=wrapper_config["optimizer"]["lr"],
            weight_decay=wrapper_config["optimizer"]["weight_decay"],
        )

    monkeypatch.setattr(param_J4_wl05_wh08, "LabelDistill", DummyLabelDistill)
    monkeypatch.setattr(
        param_J4_wl05_wh08, "build_optim_wrapper", fake_build_optim_wrapper
    )
    legacy = param_J4_wl05_wh08.LabelDistillModel(
        gpus=2,
        batch_size_per_device=16,
        default_root_dir=str(tmp_path),
    )
    config = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    ).config

    optimizers, schedulers = legacy.configure_optimizers()
    optimizer = optimizers[0]
    scheduler = schedulers[0]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        config.derived.effective_learning_rate
    )
    assert captured["optimizer"]["weight_decay"] == config.optimizer.weight_decay
    assert captured["paramwise_cfg"]["custom_keys"]["backbone"][
        "lr_mult"
    ] == config.optimizer.backbone_lr_mult
    assert sorted(scheduler.milestones.elements()) == list(config.scheduler.milestones)
