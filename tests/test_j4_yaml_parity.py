"""Lock the migrated J4 YAML to its frozen executable contract."""

from inspect import signature
from pathlib import Path

import pytest

from labeldistill.config import load_and_resolve_config
from labeldistill.models.lidardistill import LabelDistill
from labeldistill.presets.j4 import BOX_CODE_SIZE, build_j4_model_configs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _j4():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    return bundle.config, build_j4_model_configs(bundle.config)


def test_j4_yaml_expands_to_frozen_model_contract():
    config, model = _j4()
    proposal = model.teacher_proposal
    bbox_coder = proposal["bbox_coder"]
    teacher_test_cfg = model.teacher["pts_bbox_head"]["test_cfg"]["pts"]

    assert config.data.key_idxes == [-2, -4, -6, -8]
    assert model.backbone["output_channels"] == config.student.output_channels
    assert model.head["bev_backbone_conf"]["in_channels"] == (
        config.derived.student_bev_input_channels)
    assert config.student.temporal_kd_selection == signature(
        LabelDistill).parameters["temporal_kd_selection"].default
    assert config.teacher.proposal.score_threshold == bbox_coder["score_threshold"]
    assert config.teacher.proposal.max_num == bbox_coder["max_num"]
    assert teacher_test_cfg["pre_max_size"] == 1000
    assert config.teacher.proposal.post_max_size == proposal["post_max_size"]
    assert config.teacher.proposal.min_radius == proposal["min_radius"]
    assert bbox_coder["code_size"] == BOX_CODE_SIZE == 9
    assert model.teacher["voxel_layer"]["point_cloud_range"] == list(
        config.geometry.point_cloud_range)
    assert model.teacher["voxel_layer"]["voxel_size"] == list(
        config.geometry.lidar_voxel_size)
    assert [task["class_names"] for task in model.teacher[
        "pts_bbox_head"]["tasks"]] == list(config.classes.tasks)


def test_j4_yaml_locks_optimizer_scheduler_and_region_values():
    config, _ = _j4()
    assert config.runtime.gpus == 16
    assert config.runtime.batch_size_per_device == 16
    assert config.derived.global_batch_size == 256
    assert config.derived.effective_learning_rate == pytest.approx(4e-4)
    assert config.optimizer.weight_decay == pytest.approx(1e-2)
    assert config.optimizer.backbone_lr_mult == pytest.approx(1.0)
    assert config.scheduler.milestones == [19, 23]
    assert config.scheduler.warmup_steps == 200
    assert config.scheduler.warmup_ratio == pytest.approx(0.001)
    assert config.scheduler.gamma == pytest.approx(0.1)
    assert config.region.mask.w_low == pytest.approx(0.5)
    assert config.region.mask.w_high == pytest.approx(0.7)
    assert config.region.scaler.mu == pytest.approx(0.15)
    assert config.region.scaler.velocity_scale == pytest.approx(0.2)
