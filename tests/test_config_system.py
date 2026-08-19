from pathlib import Path

import pytest
from omegaconf import OmegaConf

from labeldistill.config import (
    ConfigLoadError,
    ConfigValidationError,
    load_and_resolve_config,
    save_config_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    assert config.teacher.proposal.pre_max_size == 1000
    assert config.derived.global_batch_size == 32
    assert config.derived.effective_learning_rate == pytest.approx(2e-4)
    assert list(config.derived.feature_map_size) == [128, 128]
    assert list(config.derived.bev_cell_size) == [0.8, 0.8]
    assert config.derived.student_bev_input_channels == 750


def test_inherited_ablation_and_cli_override_have_leaf_diff():
    bundle = load_and_resolve_config(
        PROJECT_ROOT / "configs/experiments/j4_wl05_wh08.yaml",
        ["runtime.gpus=4", "runtime.batch_size_per_device=4"],
        project_root=PROJECT_ROOT,
        require_checkpoint=False,
    )

    assert bundle.config.region.mask.w_high == 0.8
    assert bundle.config.derived.global_batch_size == 16
    assert bundle.config.derived.effective_learning_rate == pytest.approx(1e-4)
    assert bundle.config_diff["region.mask.w_high"] == {
        "base": 0.7,
        "resolved": 0.8,
    }
    assert bundle.config_diff["runtime.gpus"] == {"base": 2, "resolved": 4}
    assert bundle.config_diff["runtime.batch_size_per_device"] == {
        "base": 16,
        "resolved": 4,
    }


@pytest.mark.parametrize(
    "override, message",
    [
        ("region.mask.w_high=1.0", "0 < w_low <= w_high < 1"),
        ("teacher.proposal.box_code_size=10", "box_code_size must be 9"),
        ("student.distill_channels=[128]", "distill channels must match"),
        ("data.split_purpose=trainval", "use_train_val and data.split_purpose"),
        ("experiment.name=j4_wl05_wh08", "name encodes w_high=0.8"),
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


def test_inheritance_cycle_is_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("_base_: second.yaml\n", encoding="utf-8")
    second.write_text("_base_: first.yaml\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="inheritance cycle"):
        load_and_resolve_config(first, project_root=tmp_path)


def test_audit_artifacts_include_exact_source_and_resolved_config(tmp_path):
    source_path = PROJECT_ROOT / "configs/experiments/j4_wl05_wh08.yaml"
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
    assert diff.changes["region.mask.w_high"].base == 0.7
    assert "source_sha256[" in environment
