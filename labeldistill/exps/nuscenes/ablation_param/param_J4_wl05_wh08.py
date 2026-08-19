# Joint: w_l=0.5, w_h=0.7 (gap=0.2, 测试更高的 w_l)
# (w_l, w_h) 联合消融, mu=0.15, s_vel=0.2, lambda_feat=0.6 固定
from copy import deepcopy

from labeldistill.exps.base_cli import run_cli
from labeldistill.exps.nuscenes.base_exp import \
    LabelDistillModel as BaseLabelDistillModel
from labeldistill.models.lidardistill import LabelDistill
from torch.optim.lr_scheduler import MultiStepLR
from mmengine.optim import build_optim_wrapper
from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from functools import partial
import torch
import torch.nn.functional as F
import os

from labeldistill.refine_head.target_assigner.roi_distill import ProposalTargetLayer
from labeldistill.refine_head.target_assigner.quality_aware_mask_v3 import QualityAwareMaskGeneratorV3
from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import AdaptiveGTScalerV3


class LabelDistillModel(BaseLabelDistillModel):

    def __init__(self, **kwargs):
        # The base class initializes data/evaluation state only. Building its
        # default BaseBEVDepth here would allocate a model that is immediately
        # replaced by LabelDistill below.
        kwargs.pop('build_model', None)
        super().__init__(build_model=False, **kwargs)

        self.key_idxes = [-2, -4, -6, -8]

        self.generate_bev_mask = QualityAwareMaskGeneratorV3(
            w_l=0.5, w_h=0.7, r_max=50.0, boost_small_medium=True)
        self.proposal_target_layer = ProposalTargetLayer(structured=True)
        self.change_gt = AdaptiveGTScalerV3(
            mu=0.15, s_vel=0.2, r_max=50.0)

        self.backbone_conf['output_channels'] = 150
        self.head_conf['bev_backbone_conf']['in_channels'] = 150 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_backbone_conf']['base_channels'] = 150 * 2
        self.head_conf['bev_neck_conf']['in_channels'] = [
            150 * (len(self.key_idxes) + 1), 150 * 2, 150 * 4, 150 * 8
        ]
        self.head_conf['train_cfg']['code_weights'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]

        self.data_return_lidar = True

        self.optimizer_config = dict(
            type='AdamW',
            lr=4e-4,
            paramwise_cfg=dict(
                custom_keys={
                    'backbone': dict(lr_mult=0.5),
                }),
            weight_decay=1e-2)

        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size = [0.1, 0.1, 0.2]
        teacher_score_threshold = 0.1

        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=teacher_score_threshold,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            code_size=9)

        # Teacher proposal selection is an explicit distillation policy.  It
        # is independent of the student KDHead inference decoder.
        teacher_proposal_cfg = dict(
            bbox_coder=bbox_coder,
            num_classes=[1, 2, 2, 1, 2, 2],
            nms_type='circle',
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            post_max_size=83,
            norm_bbox=True)

        train_cfg = dict(
            pts=dict(
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
        test_cfg = dict(
            pts=dict(
                post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=teacher_score_threshold,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                nms_type='circle',
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2))

        self.lidar_conf = dict(type='CenterPoint',
            voxel_layer=dict(
                point_cloud_range=point_cloud_range, max_num_points=10, voxel_size=voxel_size,
                max_voxels=(90000, 120000)),
            pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
            pts_middle_encoder=dict(
                type='SparseEncoder',
                in_channels=5,
                sparse_shape=[41, 1024, 1024],
                output_channels=128,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
                block_type='basicblock'),
            pts_backbone=dict(
                type='SECOND',
                in_channels=256,
                out_channels=[128, 256],
                layer_nums=[5, 5],
                layer_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                conv_cfg=dict(type='Conv2d', bias=False)),
            pts_neck=dict(
                type='SECONDFPN',
                in_channels=[128, 256],
                out_channels=[256, 256],
                upsample_strides=[1, 2],
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                upsample_cfg=dict(type='deconv', bias=False),
                use_conv_for_no_stride=True),
             pts_bbox_head=dict(
                 type='CenterHead',
                 in_channels=sum([256, 256]),
                 tasks=[
                     dict(num_class=1, class_names=['car']),
                     dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                     dict(num_class=2, class_names=['bus', 'trailer']),
                     dict(num_class=1, class_names=['barrier']),
                     dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                     dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
                 ],
                 common_heads=dict(
                     reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
                 share_conv_channel=64,
                 bbox_coder=bbox_coder,
                 train_cfg=train_cfg,
                 test_cfg=test_cfg,
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
                 norm_bbox=True),
        )
        #############################################################################################
        lidar_ckpt_path = './ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth'
        #############################################################################################

        self.model = LabelDistill(self.backbone_conf,
                                  self.head_conf,
                                  self.lidar_conf,
                                  lidar_ckpt_path,
                                  is_train_depth=True,
                                  teacher_proposal_cfg=teacher_proposal_cfg,
                                  structured_output=True)
        self.save_resolved_configs()
        self.hparams.update({
            'lidar_conf': deepcopy(self.lidar_conf),
            'teacher_proposal_cfg': deepcopy(teacher_proposal_cfg),
        })

    def training_step(self, batch):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, lidar_pts, depth_labels) = batch

        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]
            self.model = self.model.cuda()

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            bev_mask, bev_box, bev_label, targets = self.model.module.get_targets(gt_boxes, gt_labels)
        else:
            bev_mask, bev_box, bev_label, targets = self.model.get_targets(gt_boxes, gt_labels)

        outputs = self.model(
            bev_mask, bev_box, bev_label, sweep_imgs, mats, lidar_pts)
        student = outputs.student
        teacher = outputs.teacher

        batch_result = self.proposal_target_layer(
            teacher.proposals, gt_boxes, gt_labels)

        B = len(gt_boxes)
        batch_result = self.change_gt.forward(batch_result)
        bev_roi_mask = self.generate_bev_mask(batch_result, B)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            detection_loss, response_loss = self.model.module.response_loss(
                targets, student.raw_preds, teacher.raw_preds)
        else:
            detection_loss, response_loss = self.model.response_loss(
                targets, student.raw_preds, teacher.raw_preds)

        if len(depth_labels.shape) == 5:
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), student.depth)
        lidar_distill_loss = self.get_feature_distill_loss1(
            teacher.backbone_features,
            student.distill_features,
            bev_roi_mask,
            binary_mask=False) * 0.6

        total_loss = detection_loss + depth_loss + lidar_distill_loss + response_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f'Non-finite total loss at global_step={self.global_step}: '
                f'detection={detection_loss.detach().float().item():.6g}, '
                f'response={response_loss.detach().float().item():.6g}, '
                f'depth={depth_loss.detach().float().item():.6g}, '
                f'distill={lidar_distill_loss.detach().float().item():.6g}')

        self.log('train/total_loss', total_loss, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('train/detection_loss', detection_loss, on_step=True,
                 prog_bar=True, sync_dist=True)
        self.log('train/response_loss', response_loss, on_step=True,
                 sync_dist=True)
        self.log('train/depth_loss', depth_loss, on_step=True, sync_dist=True)
        self.log('train/distill_loss', lidar_distill_loss, on_step=True,
                 sync_dist=True)

        return total_loss

    def get_feature_distill_loss1(self, lidar_feat, distill_feats, bev_mask=None, binary_mask=False):
        if len(lidar_feat) != len(distill_feats):
            raise ValueError(
                f'Feature level mismatch: teacher={len(lidar_feat)}, '
                f'student={len(distill_feats)}')

        label_losses = distill_feats[0].new_zeros((), dtype=torch.float32)

        if bev_mask is not None:
            bev_mask = bev_mask.float()
            bev_mask = [
                F.interpolate(bev_mask, size=feat.shape[-2:], mode='bilinear',
                              align_corners=False)
                for feat in lidar_feat
            ]

            if binary_mask:
                bev_mask = [(mask > 0).float() for mask in bev_mask]

        for i in range(len(lidar_feat)):
            if lidar_feat[i].shape != distill_feats[i].shape:
                raise ValueError(
                    f'Feature shape mismatch at level {i}: '
                    f'teacher={tuple(lidar_feat[i].shape)}, '
                    f'student={tuple(distill_feats[i].shape)}')

            label_loss = (
                lidar_feat[i].detach().float() - distill_feats[i].float()
            ).square().mean(dim=1)

            if bev_mask is not None:
                spatial_mask = bev_mask[i][:, 0]
                label_loss = (
                    (label_loss * spatial_mask).sum()
                    / spatial_mask.sum().clamp_min(1.0)
                )
            else:
                label_loss = label_loss.mean()
            label_losses += label_loss

        return label_losses

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(x=sweep_imgs, mats_dict=mats)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            results = self.model.module.get_bboxes(preds, img_metas)
        else:
            results = self.model.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def train_dataloader(self):
        if self.use_train_val:
            info_paths = [self.train_info_paths, self.val_info_paths]
        else:
            info_paths = self.train_info_paths

        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=info_paths,
                                       is_train=True,
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=self.data_return_depth,
                                       return_lidar=self.data_return_lidar,
                                       use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion,
                               is_return_lidar=self.data_return_lidar),
            sampler=None,
        )
        return train_loader

    def configure_optimizers(self):
        # Preserve the successful reference run's 4e-4 at global batch 64.
        # The requested 2 GPUs x 16 images therefore uses 2e-4.
        lr = (4e-4 / 64) * self.batch_size_per_device * self.gpus
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='AdamW',
                lr=lr,
                weight_decay=1e-2
            ),
            paramwise_cfg=dict(
                custom_keys={
                    'backbone': dict(lr_mult=0.5),
                }
            )
        )
        optimizer = build_optim_wrapper(self.model, optim_wrapper_cfg)
        if hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]


if __name__ == '__main__':
    run_cli(LabelDistillModel,
            'param_J4_wl05_wh08',
            extra_trainer_config_args={'epochs': 24},
            use_ema=True)
