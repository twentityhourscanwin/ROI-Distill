"""J4 model presets.

YAML owns experiment semantics.  This module owns low-level network topology
that should change only through an explicit preset revision.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

from labeldistill.exps.nuscenes import base_exp


BOX_CODE_SIZE = 9
STUDENT_HEAD_OUT_SIZE_FACTOR = 4
TEACHER_PRE_NMS_SIZE = 1000


@dataclass(frozen=True)
class J4ModelConfigs:
    backbone: Dict[str, Any]
    head: Dict[str, Any]
    teacher: Dict[str, Any]
    teacher_proposal: Dict[str, Any]
    ida_aug: Dict[str, Any]
    bda_aug: Dict[str, Any]
    img: Dict[str, Any]


def _task_dicts(config):
    return [
        dict(num_class=len(class_names), class_names=list(class_names))
        for class_names in config.classes.tasks
    ]


def _student_base(config):
    if config.student.type == 'camera_bevdepth_convnextb':
        from .convnextb import build_convnextb_student_base
        return build_convnextb_student_base(config)
    if config.student.type == 'camera_bevdepth_r50':
        return (
            deepcopy(base_exp.backbone_conf),
            deepcopy(base_exp.head_conf),
            deepcopy(base_exp.ida_aug_conf),
            deepcopy(base_exp.bda_aug_conf),
            deepcopy(base_exp.img_conf),
        )
    raise ValueError(f'Unsupported student preset: {config.student.type!r}')


def _student_configs(config):
    backbone, head, ida_aug, bda_aug, img = _student_base(config)
    point_range = list(config.geometry.point_cloud_range)
    bev_cell = list(config.derived.bev_cell_size)
    key_frames = int(config.derived.key_frame_count)
    output_channels = int(config.student.output_channels)

    backbone['x_bound'] = [point_range[0], point_range[3], bev_cell[0]]
    backbone['y_bound'] = [point_range[1], point_range[4], bev_cell[1]]
    backbone['z_bound'] = [point_range[2], point_range[5], point_range[5] - point_range[2]]
    backbone['output_channels'] = output_channels

    head['bev_backbone_conf']['in_channels'] = output_channels * key_frames
    head['bev_backbone_conf']['base_channels'] = output_channels * 2
    head['bev_neck_conf']['in_channels'] = [
        output_channels * key_frames,
        output_channels * 2,
        output_channels * 4,
        output_channels * 8,
    ]
    head['tasks'] = _task_dicts(config)
    head['train_cfg']['code_weights'] = [1.0] * 10

    # The camera detection head uses a fixed stride-4 CenterPoint grid.  Its
    # voxel values are a representation detail derived from the shared BEV
    # cell size, not an independent experiment parameter.
    student_voxel = [
        bev_cell[0] / STUDENT_HEAD_OUT_SIZE_FACTOR,
        bev_cell[1] / STUDENT_HEAD_OUT_SIZE_FACTOR,
        point_range[5] - point_range[2],
    ]
    grid_size = [
        int(round((point_range[3] - point_range[0]) / student_voxel[0])),
        int(round((point_range[4] - point_range[1]) / student_voxel[1])),
        1,
    ]
    head['bbox_coder'].update(
        pc_range=point_range,
        voxel_size=student_voxel,
        out_size_factor=STUDENT_HEAD_OUT_SIZE_FACTOR,
        code_size=BOX_CODE_SIZE,
    )
    head['train_cfg'].update(
        point_cloud_range=point_range,
        grid_size=grid_size,
        voxel_size=student_voxel,
        out_size_factor=STUDENT_HEAD_OUT_SIZE_FACTOR,
    )
    head['test_cfg'].update(
        voxel_size=student_voxel,
        out_size_factor=STUDENT_HEAD_OUT_SIZE_FACTOR,
    )
    return backbone, head, ida_aug, bda_aug, img


def _teacher_configs(config):
    point_range = list(config.geometry.point_cloud_range)
    voxel_size = list(config.geometry.lidar_voxel_size)
    out_size_factor = int(config.geometry.bev_output_stride)
    proposal = config.teacher.proposal
    tasks = _task_dicts(config)
    grid_size = [
        int(round((point_range[3] - point_range[0]) / voxel_size[0])),
        int(round((point_range[4] - point_range[1]) / voxel_size[1])),
        int(round((point_range[5] - point_range[2]) / voxel_size[2])),
    ]

    bbox_coder = dict(
        type='CenterPointBBoxCoder',
        post_center_range=list(proposal.post_center_range),
        max_num=int(proposal.max_num),
        score_threshold=float(proposal.score_threshold),
        out_size_factor=out_size_factor,
        voxel_size=voxel_size[:2],
        pc_range=point_range,
        code_size=BOX_CODE_SIZE,
    )
    train_cfg = dict(pts=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        out_size_factor=out_size_factor,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0] * 8 + [0.2, 0.2],
    ))
    test_cfg = dict(pts=dict(
        post_center_limit_range=list(proposal.post_center_range),
        max_per_img=500,
        max_pool_nms=False,
        min_radius=list(proposal.min_radius),
        score_threshold=float(proposal.score_threshold),
        out_size_factor=out_size_factor,
        voxel_size=voxel_size[:2],
        nms_type=proposal.nms_type,
        pre_max_size=TEACHER_PRE_NMS_SIZE,
        post_max_size=int(proposal.post_max_size),
        nms_thr=0.2,
    ))
    teacher = dict(
        type='CenterPoint',
        voxel_layer=dict(
            point_cloud_range=point_range,
            max_num_points=10,
            voxel_size=voxel_size,
            max_voxels=(90000, 120000),
        ),
        pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
        pts_middle_encoder=dict(
            type='SparseEncoder',
            in_channels=5,
            sparse_shape=[grid_size[2] + 1, grid_size[1], grid_size[0]],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64),
                              (64, 64, 128), (128, 128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1),
                              (0, 0, [0, 1, 1]), (0, 0)),
            block_type='basicblock',
        ),
        pts_backbone=dict(
            type='SECOND', in_channels=256, out_channels=[128, 256],
            layer_nums=[5, 5], layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False),
        ),
        pts_neck=dict(
            type='SECONDFPN', in_channels=[128, 256],
            out_channels=[256, 256], upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True,
        ),
        pts_bbox_head=dict(
            type='CenterHead', in_channels=512, tasks=tasks,
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2),
                rot=(2, 2), vel=(2, 2),
            ),
            share_conv_channel=64,
            bbox_coder=bbox_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(
                type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
            norm_bbox=True,
        ),
    )
    teacher_proposal = dict(
        bbox_coder=deepcopy(bbox_coder),
        num_classes=[len(task) for task in config.classes.tasks],
        nms_type=proposal.nms_type,
        min_radius=list(proposal.min_radius),
        post_max_size=int(proposal.post_max_size),
        norm_bbox=bool(proposal.norm_bbox),
    )
    return teacher, teacher_proposal


def build_j4_model_configs(config):
    """Expand a validated J4 config into concrete model constructor kwargs."""
    backbone, head, ida_aug, bda_aug, img = _student_configs(config)
    teacher, teacher_proposal = _teacher_configs(config)
    return J4ModelConfigs(
        backbone=backbone,
        head=head,
        teacher=teacher,
        teacher_proposal=teacher_proposal,
        ida_aug=ida_aug,
        bda_aug=bda_aug,
        img=img,
    )
