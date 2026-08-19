"""Cross-field validation and deterministic derived values."""

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf


class ConfigValidationError(ValueError):
    """Raised when an experiment config is unsafe or internally inconsistent."""


def _require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def _resolve_path(path: str, project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_experiment_name(config: DictConfig, errors: List[str]) -> None:
    name = config.experiment.name
    match = re.fullmatch(r"j4_wl(\d+)_wh(\d+)", name)
    if match is None:
        return
    # Historical names encode 0.5 as ``05`` and 0.35 as ``035``.
    expected_low = int(match.group(1)) / (10 ** (len(match.group(1)) - 1))
    expected_high = int(match.group(2)) / (10 ** (len(match.group(2)) - 1))
    _require(
        math.isclose(expected_low, config.region.mask.w_low),
        f"experiment.name encodes w_low={expected_low}, but config has "
        f"{config.region.mask.w_low}",
        errors,
    )
    _require(
        math.isclose(expected_high, config.region.mask.w_high),
        f"experiment.name encodes w_high={expected_high}, but config has "
        f"{config.region.mask.w_high}",
        errors,
    )


def validate_config(
    config: DictConfig,
    project_root: Path,
    *,
    require_checkpoint: bool = True,
) -> DictConfig:
    """Validate in place, populate ``derived``, and return ``config``.

    Validation is intentionally strict: an invalid experiment must fail before
    model construction or any trainer side effects.
    """
    project_root = Path(project_root).resolve()
    errors: List[str] = []

    _require(config.schema_version == 1, "schema_version must be 1", errors)
    _require(config.runtime.gpus > 0, "runtime.gpus must be positive", errors)
    _require(
        config.runtime.num_nodes > 0, "runtime.num_nodes must be positive", errors
    )
    _require(
        config.runtime.batch_size_per_device > 0,
        "runtime.batch_size_per_device must be positive",
        errors,
    )
    _require(config.runtime.max_epochs > 0, "runtime.max_epochs must be positive", errors)
    _require(config.data.num_workers >= 0, "data.num_workers must be non-negative", errors)
    _require(
        config.data.split_purpose in {"train", "trainval"},
        "data.split_purpose must be 'train' or 'trainval'",
        errors,
    )
    _require(
        config.data.use_train_val == (config.data.split_purpose == "trainval"),
        "data.use_train_val and data.split_purpose disagree",
        errors,
    )

    w_low = config.region.mask.w_low
    w_high = config.region.mask.w_high
    _require(0 < w_low <= w_high < 1, "need 0 < w_low <= w_high < 1", errors)
    _require(config.region.scaler.mu >= 0, "region.scaler.mu must be non-negative", errors)
    _require(
        config.region.scaler.velocity_scale >= 0,
        "region.scaler.velocity_scale must be non-negative",
        errors,
    )
    _require(
        config.region.mask.max_distance > 0,
        "region.mask.max_distance must be positive",
        errors,
    )
    _require(
        config.region.scaler.max_distance > 0,
        "region.scaler.max_distance must be positive",
        errors,
    )

    class_names = list(config.classes.names)
    flattened_tasks = [name for task in config.classes.tasks for name in task]
    _require(len(class_names) == 10, "classes.names must contain 10 classes", errors)
    _require(
        len(flattened_tasks) == len(set(flattened_tasks)),
        "classes.tasks contains duplicate classes",
        errors,
    )
    _require(
        set(flattened_tasks) == set(class_names),
        "classes.tasks must cover exactly classes.names",
        errors,
    )
    _require(
        len(config.teacher.proposal.min_radius) == len(config.classes.tasks),
        "teacher proposal min_radius count must equal task count",
        errors,
    )
    _require(
        set(config.matching.distance_thresholds.keys()) == set(class_names),
        "matching.distance_thresholds must cover every class exactly once",
        errors,
    )
    for class_name, thresholds in config.matching.distance_thresholds.items():
        _require(
            0 < thresholds.high <= thresholds.medium,
            f"matching thresholds for {class_name} need 0 < high <= medium",
            errors,
        )

    _require(
        config.geometry.gt_box_code_size == 9,
        "geometry.gt_box_code_size must be 9",
        errors,
    )
    _require(
        config.teacher.proposal.box_code_size == 9,
        "teacher.proposal.box_code_size must be 9",
        errors,
    )
    _require(
        list(config.student.distill_channels) == list(config.teacher.distill_channels),
        "teacher/student distill channels must match level by level",
        errors,
    )
    _require(
        len(config.student.distill_channels) > 0,
        "distillation needs at least one feature level",
        errors,
    )
    _require(
        config.student.key_frame_count == len(config.data.key_idxes) + 1,
        "student.key_frame_count must equal len(data.key_idxes) + 1",
        errors,
    )

    point_range = list(config.geometry.point_cloud_range)
    voxel_size = list(config.geometry.lidar_voxel_size)
    _require(len(point_range) == 6, "geometry.point_cloud_range must have 6 values", errors)
    _require(len(voxel_size) == 3, "geometry.lidar_voxel_size must have 3 values", errors)
    if len(point_range) == 6 and len(voxel_size) == 3:
        extents = [point_range[3] - point_range[0], point_range[4] - point_range[1]]
        _require(all(extent > 0 for extent in extents), "BEV extents must be positive", errors)
        _require(all(value > 0 for value in voxel_size), "voxel sizes must be positive", errors)
        cell_size = [
            voxel_size[0] * config.geometry.bev_output_stride,
            voxel_size[1] * config.geometry.bev_output_stride,
        ]
        _require(
            all(
                math.isclose(value, config.geometry.student_bev_resolution)
                for value in cell_size
            ),
            "teacher and student BEV physical cell sizes disagree",
            errors,
        )
        feature_map_float = [extent / cell for extent, cell in zip(extents, cell_size)]
        _require(
            all(math.isclose(value, round(value)) for value in feature_map_float),
            "BEV range must divide evenly by voxel size × stride",
            errors,
        )
        feature_map_size = [int(round(value)) for value in feature_map_float]
    else:
        cell_size = []
        feature_map_size = []

    _validate_experiment_name(config, errors)

    checkpoint_path = _resolve_path(config.teacher.checkpoint, project_root)
    if require_checkpoint:
        _require(
            checkpoint_path.is_file(),
            f"teacher checkpoint does not exist: {checkpoint_path}",
            errors,
        )

    if errors:
        formatted = "\n".join(f"  - {message}" for message in errors)
        raise ConfigValidationError(f"Invalid LabelDistill config:\n{formatted}")

    global_batch_size = (
        config.runtime.gpus
        * config.runtime.num_nodes
        * config.runtime.batch_size_per_device
    )
    config.derived.global_batch_size = global_batch_size
    config.derived.effective_learning_rate = (
        config.optimizer.base_lr_at_global_batch_64 * global_batch_size / 64
    )
    config.derived.feature_map_size = feature_map_size
    config.derived.bev_cell_size = cell_size
    config.derived.student_bev_input_channels = (
        config.student.output_channels * config.student.key_frame_count
    )
    config.derived.teacher_checkpoint_sha256 = (
        sha256_file(checkpoint_path) if checkpoint_path.is_file() else "unavailable"
    )

    # Resolve all interpolations and mandatory values before returning.
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    OmegaConf.set_readonly(config, True)
    return config
