"""Typed, data-only schema for LabelDistill experiment configuration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from omegaconf import MISSING


@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = 0
    description: str = ""
    legacy_aliases: List[str] = field(default_factory=list)


@dataclass
class RuntimeConfig:
    gpus: int = 1
    num_nodes: int = 1
    batch_size_per_device: int = 8
    max_epochs: int = 24
    precision: str = "16"
    gradient_clip_val: float = 35.0
    output_dir: str = MISSING
    resume_from: Optional[str] = None


@dataclass
class DataConfig:
    dataset: str = "nuscenes"
    root: str = MISSING
    train_info: str = "nuscenes_infos_train.pkl"
    val_info: str = "nuscenes_infos_val.pkl"
    use_train_val: bool = False
    split_purpose: str = "train"
    num_workers: int = 4
    use_cbgs: bool = False
    key_idxes: List[int] = field(default_factory=list)
    return_depth: bool = True
    return_lidar: bool = True


@dataclass
class GeometryConfig:
    point_cloud_range: List[float] = MISSING
    lidar_voxel_size: List[float] = MISSING
    bev_output_stride: int = MISSING
    student_bev_resolution: float = MISSING
    gt_box_code_size: int = 9


@dataclass
class ClassesConfig:
    names: List[str] = MISSING
    tasks: List[List[str]] = MISSING


@dataclass
class StudentConfig:
    type: str = MISSING
    output_channels: int = MISSING
    key_frame_count: int = MISSING
    temporal_kd_selection: str = MISSING
    distill_channels: List[int] = MISSING
    structured_output: bool = True


@dataclass
class TeacherProposalConfig:
    score_threshold: float = 0.1
    max_num: int = 500
    pre_max_size: int = 1000
    nms_type: str = "circle"
    min_radius: List[float] = MISSING
    post_max_size: int = 83
    post_center_range: List[float] = MISSING
    norm_bbox: bool = True
    box_code_size: int = 9


@dataclass
class TeacherConfig:
    type: str = MISSING
    checkpoint: str = MISSING
    checkpoint_prefix: str = "model.centerpoint."
    distill_channels: List[int] = MISSING
    proposal: TeacherProposalConfig = field(default_factory=TeacherProposalConfig)


@dataclass
class DistanceThresholdConfig:
    high: float = MISSING
    medium: float = MISSING


@dataclass
class MatchingConfig:
    type: str = "center_distance"
    class_policy: str = "same_task_group"
    structured: bool = True
    selection: str = "highest_score"
    one_to_one: bool = False
    distance_thresholds: Dict[str, DistanceThresholdConfig] = field(
        default_factory=dict)


@dataclass
class ScalerConfig:
    type: str = "adaptive_gt_scaler_v3"
    enabled: bool = True
    mu: float = 0.15
    velocity_scale: float = 0.2
    max_distance: float = 50.0


@dataclass
class MaskConfig:
    type: str = "quality_aware_mask_v3"
    w_low: float = MISSING
    w_high: float = MISSING
    max_distance: float = 50.0
    boost_small_medium: bool = True
    gaussian_overlap: float = 0.1
    min_radius: int = 2


@dataclass
class RegionConfig:
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)


@dataclass
class LossConfig:
    detection_weight: float = 1.0
    depth_weight: float = 1.0
    feature_weight: float = MISSING
    response_weight: float = 1.0


@dataclass
class OptimizerConfig:
    type: str = "AdamW"
    base_lr_at_global_batch_64: float = MISSING
    weight_decay: float = 0.01
    backbone_lr_mult: float = 0.5


@dataclass
class SchedulerConfig:
    type: str = "MultiStepLR"
    milestones: List[int] = MISSING


@dataclass
class DerivedConfig:
    global_batch_size: int = MISSING
    effective_learning_rate: float = MISSING
    feature_map_size: List[int] = MISSING
    bev_cell_size: List[float] = MISSING
    student_bev_input_channels: int = MISSING
    teacher_checkpoint_sha256: str = MISSING


@dataclass
class LabelDistillConfig:
    schema_version: int = MISSING
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    classes: ClassesConfig = field(default_factory=ClassesConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    derived: DerivedConfig = field(default_factory=DerivedConfig)
