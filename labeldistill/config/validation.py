"""Cross-field validation and deterministic derived values."""

import hashlib
import math
import re
from pathlib import Path
from typing import List

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
    _require(
        config.runtime.accumulate_grad_batches > 0,
        "runtime.accumulate_grad_batches must be positive",
        errors,
    )
    if config.runtime.limit_train_batches is not None:
        _require(
            config.runtime.limit_train_batches > 0,
            "runtime.limit_train_batches must be positive when set",
            errors,
        )
    _require(config.runtime.max_epochs > 0, "runtime.max_epochs must be positive", errors)
    _require(
        config.runtime.gradient_clip_val >= 0,
        "runtime.gradient_clip_val must be non-negative",
        errors,
    )
    _require(config.checkpoint.save_top_k >= -1, "checkpoint.save_top_k must be >= -1", errors)
    _require(
        config.checkpoint.every_n_epochs > 0,
        "checkpoint.every_n_epochs must be positive",
        errors,
    )
    _require(config.data.num_workers >= 0, "data.num_workers must be non-negative", errors)
    _require(
        config.data.split in {"train", "trainval"},
        "data.split must be 'train' or 'trainval'",
        errors,
    )

    supported_values = {
        "data.dataset": (config.data.dataset, {"nuscenes"}),
        "student.type": (config.student.type, {"camera_bevdepth_r50"}),
        "student.temporal_kd_selection": (
            config.student.temporal_kd_selection,
            {"legacy_half", "current_t2_half_t6"},
        ),
        "teacher.type": (config.teacher.type, {"frozen_centerpoint"}),
        "teacher.proposal.nms_type": (config.teacher.proposal.nms_type, {"circle"}),
        "matching.type": (config.matching.type, {"center_distance"}),
        "matching.class_policy": (config.matching.class_policy, {"same_task_group"}),
        "matching.selection": (config.matching.selection, {"highest_score"}),
        "region.scaler.type": (config.region.scaler.type, {"adaptive_gt_scaler_v3"}),
        "region.mask.type": (config.region.mask.type, {"quality_aware_mask_v3"}),
        "optimizer.type": (config.optimizer.type, {"AdamW"}),
        "scheduler.type": (config.scheduler.type, {"MultiStepLR"}),
        "runtime.precision": (
            str(config.runtime.precision),
            {"16", "16-mixed", "bf16", "bf16-mixed", "32", "64"},
        ),
        "runtime.distributed_backend": (
            config.runtime.distributed_backend,
            {"nccl", "gloo", "mpi", "ucc"},
        ),
    }
    for path, (value, supported) in supported_values.items():
        _require(value in supported, f"{path}={value!r} is not supported", errors)
    _require(
        not config.matching.one_to_one,
        "matching.one_to_one=true is not implemented",
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
        0 < config.region.mask.gaussian_overlap <= 1,
        "region.mask.gaussian_overlap must be in (0, 1]",
        errors,
    )
    _require(
        config.region.mask.min_radius >= 0,
        "region.mask.min_radius must be non-negative",
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
        0 <= config.teacher.proposal.score_threshold < 1,
        "teacher.proposal.score_threshold must be in [0, 1)",
        errors,
    )
    _require(config.teacher.proposal.max_num > 0, "teacher.proposal.max_num must be positive", errors)
    _require(
        config.teacher.proposal.post_max_size > 0,
        "teacher.proposal.post_max_size must be positive",
        errors,
    )
    _require(
        len(config.teacher.proposal.post_center_range) == 6,
        "teacher.proposal.post_center_range must have 6 values",
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
        len(config.distillation.feature_channels) > 0,
        "distillation needs at least one feature level",
        errors,
    )
    _require(
        list(config.distillation.feature_channels) == [128, 256],
        "camera_bevdepth_r50/frozen_centerpoint requires "
        "distillation.feature_channels=[128, 256]",
        errors,
    )
    _require(
        all(channel > 0 for channel in config.distillation.feature_channels),
        "distillation.feature_channels must be positive",
        errors,
    )
    _require(config.student.output_channels > 0, "student.output_channels must be positive", errors)
    _require(
        config.student.output_channels % 2 == 0,
        "student.output_channels must be even for temporal KD selection",
        errors,
    )
    if config.student.temporal_kd_selection == "current_t2_half_t6":
        _require(
            len(config.data.key_idxes) == 4,
            "current_t2_half_t6 requires exactly five frames",
            errors,
        )

    point_range = list(config.geometry.point_cloud_range)
    voxel_size = list(config.geometry.lidar_voxel_size)
    _require(len(point_range) == 6, "geometry.point_cloud_range must have 6 values", errors)
    _require(len(voxel_size) == 3, "geometry.lidar_voxel_size must have 3 values", errors)
    _require(
        config.geometry.bev_output_stride > 0,
        "geometry.bev_output_stride must be positive",
        errors,
    )
    if (
        len(point_range) == 6
        and len(voxel_size) == 3
        and all(value > 0 for value in voxel_size)
        and config.geometry.bev_output_stride > 0
    ):
        extents = [point_range[3] - point_range[0], point_range[4] - point_range[1]]
        _require(all(extent > 0 for extent in extents), "BEV extents must be positive", errors)
        _require(all(value > 0 for value in voxel_size), "voxel sizes must be positive", errors)
        cell_size = [
            voxel_size[0] * config.geometry.bev_output_stride,
            voxel_size[1] * config.geometry.bev_output_stride,
        ]
        feature_map_float = (
            [extent / cell for extent, cell in zip(extents, cell_size)]
            if all(extent > 0 for extent in extents)
            else []
        )
        _require(
            bool(feature_map_float)
            and all(math.isclose(value, round(value)) for value in feature_map_float),
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
        * config.runtime.accumulate_grad_batches
    )
    config.derived.global_batch_size = global_batch_size
    config.derived.effective_learning_rate = (
        config.optimizer.base_lr_at_global_batch_64 * global_batch_size / 64
    )
    config.derived.key_frame_count = len(config.data.key_idxes) + 1
    config.derived.feature_map_size = feature_map_size
    config.derived.bev_cell_size = cell_size
    config.derived.student_bev_input_channels = (
        config.student.output_channels * config.derived.key_frame_count
    )
    config.derived.teacher_checkpoint_sha256 = (
        sha256_file(checkpoint_path)
        if require_checkpoint and checkpoint_path.is_file()
        else "unavailable"
    )

    # Resolve all interpolations and mandatory values before returning.
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    OmegaConf.set_readonly(config, True)
    return config
