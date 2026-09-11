"""Construct the J4 experiment from one validated config bundle."""

from pathlib import Path

from omegaconf import OmegaConf

from labeldistill.experiments import J4Experiment
from labeldistill.models.lidardistill import LabelDistill
from labeldistill.presets import build_j4_model_configs
from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import (
    AdaptiveGTScalerV3,
)
from labeldistill.refine_head.target_assigner.proposal_center_shift import (
    ProposalCenterShift,
)
from labeldistill.refine_head.target_assigner.quality_aware_mask_v3 import (
    QualityAwareMaskGeneratorV3,
)
from labeldistill.refine_head.target_assigner.raw_gaussian_feature_loss import (
    RawGaussianUnionFeatureLoss,
)
from labeldistill.refine_head.target_assigner.roi_distill import ProposalTargetLayer


SMALL_CLASS_NAMES = {
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
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
    thresholds = (
        _plain_thresholds(config)
        if config.matching.type == 'center_distance' else None)
    trust_radius = OmegaConf.to_container(
        config.matching.trust_radius, resolve=True)
    official_class_ranges = OmegaConf.to_container(
        config.matching.official_class_ranges, resolve=True)
    matcher = ProposalTargetLayer(
        structured=True,
        class_names=class_names,
        class_groups=class_groups,
        distance_thresholds=thresholds,
        class_policy=config.matching.class_policy,
        selection=config.matching.selection,
        one_to_one=config.matching.one_to_one,
        matching_type=config.matching.type,
        strict_less_than=config.matching.strict_less_than,
        trust_radius=trust_radius,
        small_classes=list(config.matching.small_classes),
        large_classes=list(config.matching.large_classes),
        official_class_ranges=official_class_ranges,
        point_cloud_range=list(config.geometry.point_cloud_range),
        value_type=config.region.value.type,
    )
    if config.region.scaler.type == 'proposal_center_only':
        scaler = ProposalCenterShift()
    else:
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
    mask_generator = None
    if config.region.mask.type == 'quality_aware_mask_v3':
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
    feature_loss_reducer = None
    if config.region.mask.type in {
            'per_gt_gaussian', 'per_gt_elliptical_gaussian'}:
        feature_loss_reducer = RawGaussianUnionFeatureLoss(
            point_cloud_range=list(config.geometry.point_cloud_range),
            feature_map_size=list(config.derived.feature_map_size),
            gaussian_overlap=config.region.mask.gaussian_overlap,
            min_radius=config.region.mask.min_radius,
            small_class_ids=small_class_ids,
            mask_type=config.region.mask.type,
            overlap_merge=config.region.mask.overlap_merge,
        )
    return experiment_cls(
        config=config,
        model=model,
        matcher=matcher,
        scaler=scaler,
        mask_generator=mask_generator,
        feature_loss_reducer=feature_loss_reducer,
        backbone_conf=model_configs.backbone,
        head_conf=model_configs.head,
        ida_aug_conf=model_configs.ida_aug,
        bda_aug_conf=model_configs.bda_aug,
        img_conf=model_configs.img,
    )
