from pathlib import Path
from datetime import datetime

import pytest
from omegaconf import OmegaConf
import torch

from labeldistill.config import (
    RUN_ID_ENV,
    ConfigLoadError,
    ConfigValidationError,
    checkpoint_config_file,
    checkpoint_directory,
    dated_directory,
    generate_run_id,
    load_training_bundle,
    load_and_resolve_config,
    save_config_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_run_directories_share_a_sortable_timestamp():
    run_id = generate_run_id(datetime(2026, 8, 28, 15, 30, 12))
    assert run_id == "20260828_153012"
    assert dated_directory("outputs/b1", run_id) == (
        "outputs/b1_20260828_153012")
    assert dated_directory(
        "outputs/b1_20260827_101500",
        run_id,
        previous_run_id="20260827_101500",
    ) == "outputs/b1_20260828_153012"


def test_checkpoint_directory_uses_the_resolved_run_id():
    config = OmegaConf.create({
        "experiment": {"name": "b1"},
        "runtime": {"run_id": "20260828_153012"},
        "checkpoint": {"root_dir": "/checkpoints"},
    })
    assert checkpoint_directory(config) == Path(
        "/checkpoints/b1_20260828_153012")


def test_new_training_bundle_assigns_one_run_id_to_output_and_checkpoint(
        monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, "20260828_153012")
    bundle = load_training_bundle(
        PROJECT_ROOT / "configs/experiments/b1_teacher_value_no_scale.yaml",
        [],
        PROJECT_ROOT,
        require_checkpoint=False,
    )

    assert bundle.config.runtime.run_id == "20260828_153012"
    assert bundle.config.runtime.output_dir == (
        "outputs/b1_teacher_value_no_scale_20260828_153012")
    assert checkpoint_directory(bundle.config) == Path(
        "/mnt/nas_data/guqiupeng/checkpoint_nes/"
        "b1_teacher_value_no_scale_20260828_153012")


def test_resume_keeps_the_existing_run_identity_and_directories(
        tmp_path, monkeypatch):
    monkeypatch.setenv(RUN_ID_ENV, "20260828_235959")
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    existing_output = "outputs/b1_teacher_value_no_scale_20260828_153012"
    bundle = load_training_bundle(
        PROJECT_ROOT / "configs/experiments/b1_teacher_value_no_scale.yaml",
        [
            "runtime.run_id=20260828_153012",
            f"runtime.output_dir={existing_output}",
            f"runtime.resume_from={checkpoint}",
        ],
        PROJECT_ROOT,
        require_checkpoint=False,
    )

    assert bundle.config.runtime.run_id == "20260828_153012"
    assert bundle.config.runtime.output_dir == existing_output
    assert checkpoint_directory(bundle.config).name == (
        "b1_teacher_value_no_scale_20260828_153012")


def test_checkpoint_embedded_resolved_config_is_a_supported_input(tmp_path):
    bundle = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    checkpoint = tmp_path / "last.ckpt"
    torch.save({
        "labeldistill_schema_version": 1,
        "labeldistill_resolved_config": OmegaConf.to_container(
            bundle.config, resolve=True),
    }, checkpoint)
    with checkpoint_config_file(checkpoint) as config_path:
        resumed = load_and_resolve_config(
            config_path,
            ["runtime.gpus=1"],
            project_root=PROJECT_ROOT,
            require_checkpoint=False,
        )
    assert resumed.config.experiment.name == "j4_wl05_wh07"
    assert resumed.config.runtime.gpus == 1


def test_checkpoint_without_embedded_config_fails(tmp_path):
    checkpoint = tmp_path / "old.ckpt"
    torch.save({"state_dict": {}}, checkpoint)
    with pytest.raises(ConfigLoadError, match="predates embedded resolved config"):
        with checkpoint_config_file(checkpoint):
            pass


def test_j4_baseline_resolves_real_derived_values():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )

    config = bundle.config
    assert config.experiment.name == "j4_wl05_wh07"
    assert config.region.mask.w_low == 0.5
    assert config.region.mask.w_high == 0.7
    assert config.student.temporal_kd_selection == "legacy_half"
    assert config.teacher.proposal.max_num == 500
    assert config.derived.global_batch_size == 256
    assert config.derived.effective_learning_rate == pytest.approx(4e-4)
    assert config.optimizer.backbone_lr_mult == pytest.approx(1.0)
    assert config.scheduler.warmup_steps == 200
    assert config.scheduler.warmup_ratio == pytest.approx(0.001)
    assert config.scheduler.gamma == pytest.approx(0.1)
    assert list(config.derived.feature_map_size) == [128, 128]
    assert list(config.derived.bev_cell_size) == [0.8, 0.8]
    assert config.derived.key_frame_count == 5
    assert config.derived.student_bev_input_channels == 750
    assert config.checkpoint.root_dir == (
        "/mnt/nas_data/guqiupeng/checkpoint_nes")


def test_center_value_baseline_resolves_strict_low_parameter_policy():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/b1_teacher_value_no_scale.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )

    config = bundle.config
    assert config.matching.type == "scale_conditioned_center_distance"
    assert config.matching.class_policy == "exact_class"
    assert config.matching.one_to_one is True
    assert config.matching.strict_less_than is True
    assert config.matching.trust_radius.small == 1.0
    assert config.matching.trust_radius.large == 2.0
    assert "bicycle" in config.matching.small_classes
    assert config.region.value.type == "normalized_squared_margin"
    assert config.region.value.use_teacher_score is False
    assert config.region.mask.normalize_per_instance is False
    assert config.region.mask.overlap_merge == "max"
    assert config.region.scaler.enabled is False
    assert config.loss.response_bbox_scope == "matched_gt"
    assert config.teacher.checkpoint.endswith(
        "ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth")
    assert config.teacher.checkpoint_prefix == "model.centerpoint."
    assert config.runtime.gpus == 16
    assert config.runtime.batch_size_per_device == 16
    assert config.derived.global_batch_size == 256
    assert config.derived.effective_learning_rate == pytest.approx(4e-4)
    assert config.optimizer.backbone_lr_mult == pytest.approx(1.0)
    assert config.scheduler.warmup_steps == 200


@pytest.mark.parametrize(
    "override, message",
    [
        ("matching.one_to_one=false", "requires one_to_one=true"),
        ("matching.strict_less_than=false", "strict_less_than=true"),
        ("matching.trust_radius.small=0", "trust_radius.small must be positive"),
        ("region.value.use_teacher_score=true", "use_teacher_score=false"),
        ("region.mask.normalize_per_instance=true", "raw per-GT"),
        ("region.mask.overlap_merge=sum", "max overlap"),
        ("loss.response_bbox_scope=all_gt",
         "response_bbox_scope=matched_gt"),
    ],
)
def test_center_value_baseline_rejects_semantic_drift(override, message):
    with pytest.raises(ConfigValidationError, match=message):
        load_and_resolve_config(
            PROJECT_ROOT / "configs/experiments/b1_teacher_value_no_scale.yaml",
            [override],
            project_root=PROJECT_ROOT,
            require_checkpoint=False,
        )


def test_inherited_override_and_cli_override_have_leaf_diff(tmp_path):
    source_path = tmp_path / "legacy_override.yaml"
    source_path.write_text(
        f"_base_: {(PROJECT_ROOT / 'configs/experiments/j4_wl05_wh07.yaml').as_posix()}\n"
        "experiment:\n"
        "  name: temporary_legacy_override\n"
        "region:\n"
        "  mask:\n"
        "    w_high: 0.8\n",
        encoding="utf-8",
    )
    bundle = load_and_resolve_config(
        source_path,
        ["runtime.gpus=4", "runtime.batch_size_per_device=4"],
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )

    assert bundle.config.region.mask.w_high == 0.8
    assert bundle.config.derived.global_batch_size == 16
    assert bundle.config.derived.effective_learning_rate == pytest.approx(2.5e-5)
    assert bundle.config_diff["region.mask.w_high"] == {
        "base": 0.7,
        "resolved": 0.8,
    }
    assert bundle.config_diff["runtime.gpus"] == {"base": 16, "resolved": 4}
    assert bundle.config_diff["runtime.batch_size_per_device"] == {
        "base": 16,
        "resolved": 4,
    }


@pytest.mark.parametrize(
    "override, message",
    [
        ("region.mask.w_high=1.0", "0 < w_low <= w_high < 1"),
        ("geometry.bev_output_stride=0", "bev_output_stride must be positive"),
        ("distillation.feature_channels=[]", "at least one feature level"),
        ("matching.one_to_one=true", "one_to_one=true is not implemented"),
        ("matching.selection=nearest_distance", "is not supported"),
        ("scheduler.warmup_steps=-1", "warmup_steps must be non-negative"),
        ("scheduler.warmup_ratio=0", "warmup_ratio must be in"),
        ("scheduler.gamma=0", "scheduler.gamma must be in"),
    ],
)
def test_invalid_cross_field_config_fails_before_training(override, message):
    with pytest.raises(ConfigValidationError, match=message):
        load_and_resolve_config(
            PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
            [override],
            project_root=PROJECT_ROOT,
            require_checkpoint=False,
        )


def test_unknown_schema_key_is_rejected():
    with pytest.raises(ConfigLoadError, match="not in 'LabelDistillConfig'"):
        load_and_resolve_config(
            PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
            ["typo.value=1"],
            project_root=PROJECT_ROOT,
            require_checkpoint=False,
        )


def test_derived_values_cannot_be_overridden():
    with pytest.raises(ConfigLoadError, match="derived values cannot be overridden"):
        load_and_resolve_config(
            PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
            ["derived.global_batch_size=999"],
            project_root=PROJECT_ROOT,
            require_checkpoint=False,
        )


def test_inheritance_cycle_is_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("_base_: second.yaml\n", encoding="utf-8")
    second.write_text("_base_: first.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="inheritance cycle"):
        load_and_resolve_config(first, project_root=tmp_path)


def test_audit_artifacts_include_exact_source_and_resolved_config(tmp_path):
    source_path = (
        PROJECT_ROOT
        / "configs/experiments/b1_teacher_value_no_scale.yaml"
    )
    bundle = load_and_resolve_config(
        source_path,
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    output = save_config_artifacts(
        bundle,
        str(tmp_path / "audit"),
        source_files=[PROJECT_ROOT / "labeldistill/config/loader.py"],
    )

    assert (output / "source_config.yaml").read_bytes() == source_path.read_bytes()
    resolved = OmegaConf.load(output / "resolved_config.yaml")
    diff = OmegaConf.load(output / "config_diff.yaml")
    environment = (output / "environment.txt").read_text(encoding="utf-8")
    assert resolved.derived.feature_map_size == [128, 128]
    assert resolved.teacher.checkpoint.endswith(
        "centerpoint_vox01_128x128_20e_10sweeps.pth")
    assert "source_sha256[" in environment


def test_saved_resolved_config_is_a_valid_recomputed_input(tmp_path):
    bundle = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    resolved_path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(bundle.config, resolved_path, resolve=True)

    reloaded = load_and_resolve_config(
        resolved_path,
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    assert OmegaConf.to_container(reloaded.config, resolve=True) == OmegaConf.to_container(
        bundle.config, resolve=True)


def test_tampered_resolved_derived_values_are_rejected(tmp_path):
    bundle = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh07.yaml",
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )
    tampered = OmegaConf.create(OmegaConf.to_container(bundle.config, resolve=True))
    tampered.derived.global_batch_size = 999
    resolved_path = tmp_path / "resolved_config.yaml"
    OmegaConf.save(tampered, resolved_path, resolve=True)

    with pytest.raises(ConfigLoadError, match="do not match recomputed values"):
        load_and_resolve_config(
            resolved_path,
            project_root=PROJECT_ROOT,
            require_checkpoint=False,
        )
