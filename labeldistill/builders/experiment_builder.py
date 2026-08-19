"""Construct the J4 experiment from one validated config bundle."""

from pathlib import Path

from omegaconf import OmegaConf

from labeldistill.experiments import J4Experiment
from labeldistill.models.lidardistill import LabelDistill
from labeldistill.presets import build_j4_model_configs
from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import (
    AdaptiveGTScalerV3,
)
from labeldistill.refine_head.target_assigner.quality_aware_mask_v3 import (
    QualityAwareMaskGeneratorV3,
)
from labeldistill.refine_head.target_assigner.roi_distill import ProposalTargetLayer


SMALL_CLASS_NAMES = {
    'barrier', 'motorcycle', 'pedestrian', 'traffic_cone',
}


def _plain_thresholds(config):
    return OmegaConf.to_container(
        config.matching.distance_thresholds, resolve=True)


def build_experiment(bundle, *, model_cls=LabelDistill,
                     experiment_cls=J4Experiment):
    """Build a complete J4 LightningModule without importing legacy entries."""
    config = bundle.config
    model_configs = build_j4_model_configs(config)
    checkpoint_path = Path(config.teacher.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = bundle.project_root / checkpoint_path

    model = model_cls(
        model_configs.backbone,
        model_configs.head,
        model_configs.teacher,
        str(checkpoint_path),
        lidar_checkpoint_prefix=config.teacher.checkpoint_prefix,
        is_train_depth=True,
        temporal_kd_selection=config.student.temporal_kd_selection,
        distill_feature_channels=list(config.distillation.feature_channels),
        teacher_proposal_cfg=model_configs.teacher_proposal,
        structured_output=True,
    )
    class_names = list(config.classes.names)
    class_groups = [list(group) for group in config.classes.tasks]
    thresholds = _plain_thresholds(config)
    matcher = ProposalTargetLayer(
        structured=True,
        class_names=class_names,
        class_groups=class_groups,
        distance_thresholds=thresholds,
        class_policy=config.matching.class_policy,
        selection=config.matching.selection,
        one_to_one=config.matching.one_to_one,
    )
    scaler = AdaptiveGTScalerV3(
        mu=config.region.scaler.mu,
        s_vel=config.region.scaler.velocity_scale,
        r_max=config.region.scaler.max_distance,
        class_names=class_names,
        distance_thresholds=thresholds,
    )
    small_class_ids = [
        idx for idx, name in enumerate(class_names)
        if name in SMALL_CLASS_NAMES
    ]
    mask_generator = QualityAwareMaskGeneratorV3(
        w_l=config.region.mask.w_low,
        w_h=config.region.mask.w_high,
        r_max=config.region.mask.max_distance,
        boost_small_medium=config.region.mask.boost_small_medium,
        point_cloud_range=list(config.geometry.point_cloud_range),
        voxel_size=list(config.geometry.lidar_voxel_size),
        out_size_factor=config.geometry.bev_output_stride,
        feature_map_size=list(config.derived.feature_map_size),
        gaussian_overlap=config.region.mask.gaussian_overlap,
        min_radius=config.region.mask.min_radius,
        small_class_ids=small_class_ids,
    )
    return experiment_cls(
        config=config,
        model=model,
        matcher=matcher,
        scaler=scaler,
        mask_generator=mask_generator,
        backbone_conf=model_configs.backbone,
        head_conf=model_configs.head,
    )
