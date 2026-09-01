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
    distributed_backend: str = "nccl"
    batch_size_per_device: int = 8
    accumulate_grad_batches: int = 1
    limit_train_batches: Optional[int] = None
    limit_test_batches: Optional[int] = None
    max_epochs: int = 24
    precision: str = "16"
    gradient_clip_val: float = 35.0
    deterministic: bool = True
    find_unused_parameters: bool = True
    ddp_static_graph: bool = False
    output_dir: str = MISSING
    run_id: Optional[str] = None
    resume_from: Optional[str] = None


@dataclass
class EMAConfig:
    enabled: bool = True


@dataclass
class CheckpointConfig:
    root_dir: str = "/mnt/nas_data/guqiupeng/checkpoint_nes"
    save_top_k: int = 3
    save_last: bool = True
    every_n_epochs: int = 1


@dataclass
class EvaluationConfig:
    run_metrics: bool = True


@dataclass
class DataConfig:
    dataset: str = "nuscenes"
    root: str = MISSING
    train_info: str = "nuscenes_infos_train.pkl"
    val_info: str = "nuscenes_infos_val.pkl"
    split: str = "train"
    num_workers: int = 4
    use_cbgs: bool = False
    key_idxes: List[int] = field(default_factory=list)


@dataclass
class GeometryConfig:
    point_cloud_range: List[float] = MISSING
    lidar_voxel_size: List[float] = MISSING
    bev_output_stride: int = MISSING


@dataclass
class ClassesConfig:
    names: List[str] = MISSING
    tasks: List[List[str]] = MISSING


@dataclass
class StudentImageConfig:
    source_size: List[int] = field(default_factory=lambda: [900, 1600])
    final_size: List[int] = field(default_factory=lambda: [256, 704])
    resize_limit: List[float] = field(default_factory=lambda: [0.386, 0.55])
    gradient_checkpointing: bool = False
    pretrained: bool = True
    drop_path_rate: float = 0.0


@dataclass
class StudentConfig:
    type: str = MISSING
    output_channels: int = MISSING
    temporal_kd_selection: str = MISSING
    image: StudentImageConfig = field(default_factory=StudentImageConfig)


@dataclass
class TeacherProposalConfig:
    score_threshold: float = 0.1
    max_num: int = 500
    nms_type: str = "circle"
    min_radius: List[float] = MISSING
    post_max_size: int = 83
    post_center_range: List[float] = MISSING
    norm_bbox: bool = True


@dataclass
class TeacherConfig:
    type: str = MISSING
    checkpoint: str = MISSING
    checkpoint_prefix: str = "model.centerpoint."
    proposal: TeacherProposalConfig = field(default_factory=TeacherProposalConfig)


@dataclass
class DistillationConfig:
    feature_channels: List[int] = MISSING


@dataclass
class DistanceThresholdConfig:
    high: float = MISSING
    medium: float = MISSING


@dataclass
class TrustRadiusConfig:
    small: Optional[float] = None
    large: Optional[float] = None


@dataclass
class MatchingConfig:
    type: str = "center_distance"
    class_policy: str = "same_task_group"
    selection: str = "highest_score"
    one_to_one: bool = False
    strict_less_than: bool = False
    distance_thresholds: Dict[str, DistanceThresholdConfig] = field(
        default_factory=dict)
    trust_radius: TrustRadiusConfig = field(default_factory=TrustRadiusConfig)
    small_classes: List[str] = field(default_factory=list)
    large_classes: List[str] = field(default_factory=list)
    official_class_ranges: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalerConfig:
    type: str = "adaptive_gt_scaler_v3"
    enabled: bool = True
    mu: float = 0.15
    velocity_scale: float = 0.2
    max_distance: float = 50.0


@dataclass
class ValueConfig:
    type: str = "legacy_discrete"
    use_teacher_score: bool = True
    unmatched_value: float = 0.0


@dataclass
class MaskConfig:
    type: str = "quality_aware_mask_v3"
    w_low: Optional[float] = None
    w_high: Optional[float] = None
    max_distance: float = 50.0
    boost_small_medium: bool = True
    gaussian_overlap: float = 0.1
    min_radius: int = 2
    normalize_per_instance: bool = False
    overlap_merge: str = "max"


@dataclass
class RegionConfig:
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)


@dataclass
class LossConfig:
    detection_weight: float = 1.0
    depth_weight: float = 1.0
    feature_weight: float = MISSING
    response_weight: float = 1.0
    response_bbox_scope: str = "all_gt"


@dataclass
class OptimizerConfig:
    type: str = "AdamW"
    base_lr_at_global_batch_64: float = MISSING
    weight_decay: float = 0.01
    backbone_lr_mult: float = 0.5


@dataclass
class SchedulerConfig:
    type: str = "MultiStepLR"
    milestones: List[int] = field(default_factory=list)
    warmup_steps: int = 0
    warmup_ratio: float = 0.001
    gamma: float = 0.1
    min_lr_ratio: float = 0.0


@dataclass
class DerivedConfig:
    global_batch_size: int = MISSING
    effective_learning_rate: float = MISSING
    key_frame_count: int = MISSING
    feature_map_size: List[int] = MISSING
    bev_cell_size: List[float] = MISSING
    student_bev_input_channels: int = MISSING
    teacher_checkpoint_sha256: str = MISSING


@dataclass
class LabelDistillConfig:
    schema_version: int = MISSING
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    classes: ClassesConfig = field(default_factory=ClassesConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    derived: DerivedConfig = field(default_factory=DerivedConfig)
