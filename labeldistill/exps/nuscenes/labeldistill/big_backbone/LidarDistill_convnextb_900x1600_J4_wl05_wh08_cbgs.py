# ConvNeXt-B 900x1600 + ROI-V3 (param_J4) + CBGS
# 蒸馏架构沿用主实验 param_J4_wl05_wh08.py:
#   - QualityAwareMaskGeneratorV3(w_l=0.5, w_h=0.7, boost_small_medium)
#   - AdaptiveGTScalerV3(mu=0.15, s_vel=0.2)
#   - response_loss * 0.6, lidar_distill_loss * 0.6
# 训练设置参考 LidarDistill_convnextb_900x1600_e24_roi_v2_exp7.py:
#   - ConvNeXt-B backbone (base_exp_convnextb), with_cp 梯度检查点
#   - lr 按 2e-4 / global_batch_64 线性缩放, backbone lr_mult=0.1
#   - warmup + 余弦退火调度（分母自动取 trainer.max_epochs，避免半截余弦）
# 额外开启 CBGS 类别平衡采样 -> epoch 采用 CBGS 经验值 20
#   （CBGS 对稀有类过采样，单 epoch 有效迭代数增加，故比无 CBGS 的 24 减少）
# 训练数据：默认合并 train + val（与 train_teacher_centerpoint.py 一致，use_train_val=True）
from labeldistill.exps.base_cli import run_cli
from labeldistill.exps.nuscenes.base_exp_convnextb import \
    LabelDistillModel as BaseLabelDistillModel
from labeldistill.models.lidardistill import LabelDistill
from mmengine.optim import build_optim_wrapper
from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from labeldistill.refine_head.target_assigner.roi_distill import ProposalTargetLayer
from labeldistill.refine_head.target_assigner.quality_aware_mask_v3 import QualityAwareMaskGeneratorV3
from labeldistill.refine_head.target_assigner.adaptive_gt_scaler_v3 import AdaptiveGTScalerV3


class LabelDistillModel(BaseLabelDistillModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.key_idxes = [-2, -4, -6, -8]

        self.generate_bev_mask = QualityAwareMaskGeneratorV3(
            w_l=0.5, w_h=0.7, r_max=50.0, boost_small_medium=True)
        self.proposal_target_layer = ProposalTargetLayer()
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
        self.data_use_cbgs = True
        # 与 train_teacher_centerpoint 一致：train_dataloader 合并 val pkl 参与训练
        self.use_train_val = kwargs.get('use_train_val', True)

        self.optimizer_config = dict(
            type='AdamW',
            paramwise_cfg=dict(
                custom_keys={
                    'backbone': dict(lr_mult=0.1),
                }),
            weight_decay=1e-2)

        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size = [0.1, 0.1, 0.2]

        bbox_coder = dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
            code_size=9)

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
                score_threshold=0.15,
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
        lidar_ckpt_path = '/mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL/ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth'
        #############################################################################################

        self.model = LabelDistill(self.backbone_conf,
                                  self.head_conf,
                                  self.lidar_conf,
                                  lidar_ckpt_path,
                                  is_train_depth=True,
                                  temporal_kd_selection='current_t2_half_t6')

    def on_train_start(self):
        current_strategy = type(self.trainer.strategy).__name__
        is_ddp_wrapped = isinstance(self.trainer.model, nn.parallel.DistributedDataParallel)

        if is_ddp_wrapped:
            print("✅ 模型已被 torch.nn.parallel.DistributedDataParallel 封装。")
            if self.trainer.strategy.parallel_devices:
                print(f"   DDP Device IDs: {self.trainer.strategy.parallel_devices}")

            if self.backbone_conf.get('img_backbone_conf', {}).get('with_cp', False):
                if hasattr(self.trainer.model, '_set_static_graph'):
                    print(">>> ⚙️ 启用 DDP 静态图：解决梯度检查点冲突...")
                    self.trainer.model._set_static_graph()
                    print(">>> 启用成功。")
                else:
                    print(">>> ⚠️ 警告: DDP 实例不支持 _set_static_graph。请检查 PyTorch 版本。")
        else:
            print("❌ 模型未被直接 DDP 封装。")

        print("-" * 50)

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

        preds, lidar_preds, depth_preds, distill_feats_lidar, lidar_feats, neck_feats, neck_output, lidar_pred_box, image_pred_box = self.model(
            bev_mask, bev_box, bev_label, sweep_imgs, mats, lidar_pts)

        batch_dict_pre = self.prepare_batch_dict_from_pred_box(lidar_pred_box, gt_boxes, gt_labels)
        batch_result = self.proposal_target_layer(batch_dict_pre)

        B = len(gt_boxes)
        batch_result = self.change_gt.forward(batch_result)
        bev_roi_mask = self.generate_bev_mask(batch_result, B)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            detection_loss, response_loss = self.model.module.response_loss(targets, preds, lidar_preds)
        else:
            detection_loss, response_loss = self.model.response_loss(targets, preds, lidar_preds)

        if len(depth_labels.shape) == 5:
            depth_labels = depth_labels[:, 0, ...]

        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds, sigma=1.5)
        lidar_distill_loss = self.get_feature_distill_loss1(
            lidar_feats, distill_feats_lidar, bev_roi_mask, binary_mask=False) * 0.6
        response_loss = response_loss * 0.6

        total_loss = detection_loss + depth_loss + lidar_distill_loss + response_loss

        # ---- 损失分量诊断：定位最先出现 NaN/Inf 的分量 ----
        self._diagnose_loss_terms(
            dict(detection=detection_loss, response=response_loss,
                 distill=lidar_distill_loss, depth=depth_loss),
            preds=preds, lidar_preds=lidar_preds, depth_preds=depth_preds,
            bev_roi_mask=bev_roi_mask,
            fwd_feats=dict(distill_feats_lidar=distill_feats_lidar,
                           neck_output=neck_output, neck_feats=neck_feats))

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f'Non-finite total loss at global_step={self.global_step}; '
                'see the NaN-DEBUG statistics above')

        # 四个分量全部上进度条，方便实时观察哪一项异常
        self.log('train/total_loss',        total_loss,         on_step=True, prog_bar=True,  sync_dist=True)
        self.log('train/detection_loss',    detection_loss,     on_step=True, prog_bar=True,  sync_dist=True)
        self.log('train/response_loss',     response_loss,      on_step=True, prog_bar=True,  sync_dist=True)
        self.log('train/distill_loss',      lidar_distill_loss, on_step=True, prog_bar=True,  sync_dist=True)
        self.log('train/depth_loss',        depth_loss,         on_step=True, prog_bar=False, sync_dist=True)

        return total_loss

    def _diagnose_loss_terms(self, loss_terms, preds=None, lidar_preds=None,
                             depth_preds=None, bev_roi_mask=None, fwd_feats=None):
        """检测各损失分量是否出现 NaN/Inf，定位最先损坏的分量。

        - 每个 step 都检查（开销极小，仅 isfinite + item）。
        - 出现非有限值时打印：哪个分量坏、各分量数值、关键前向张量的
          min/max/finite，用于区分「前向爆炸」还是「损失爆炸」。
        - 详细前向张量统计只在首次触发时打印，避免刷屏。
        """
        bad = [k for k, v in loss_terms.items()
               if not torch.isfinite(v).all()]

        step = int(self.global_step)
        rank = int(getattr(self, 'global_rank', 0))

        # 每步都打一行紧凑的分量数值（仅 rank0），方便对照 NaN 出现的时刻
        if rank == 0 and (step % 50 == 0 or bad):
            compact = ' '.join(
                f"{k}={v.detach().float().item():.4g}"
                for k, v in loss_terms.items())
            print(f"[loss][step {step}] {compact}")

        # 前向激活幅度监控：趁还没溢出时抓住「finite 但在增长」的现场，
        # 定位是 BEV 特征 / 头颈特征 / 学生预测里哪一处先变大。
        if rank == 0 and (step % 50 == 0 or bad):
            def _maxabs(t):
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    tf = t.detach().float().abs()
                    finite = torch.isfinite(tf)
                    return tf[finite].max().item() if finite.any() else float('nan')
                if isinstance(t, (list, tuple)):
                    vals = [_maxabs(x) for x in t]
                    vals = [v for v in vals if v == v]  # 去掉 nan
                    return max(vals) if vals else float('nan')
                return float('nan')

            mags = []
            if depth_preds is not None:
                mags.append(f"depth={_maxabs(depth_preds):.4g}")
            if fwd_feats:
                for fname, ft in fwd_feats.items():
                    mags.append(f"{fname}={_maxabs(ft):.4g}")
            # 学生预测各分支的最大绝对值（取第一个 task 头）
            try:
                pd0 = preds[0][0]
                for key in ('heatmap', 'reg', 'height', 'dim', 'rot', 'vel'):
                    if key in pd0:
                        mags.append(f"stu.{key}={_maxabs(pd0[key]):.4g}")
            except Exception:
                pass
            print(f"[maxabs][step {step}] " + ' '.join(mags))

        if not bad:
            return

        print(f"[NaN-DEBUG][step {step}][rank {rank}] non-finite in {bad}")

        if getattr(self, '_nan_reported', False):
            return
        self._nan_reported = True

        def _stat(name, t):
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                tf = t.detach().float()
                finite = torch.isfinite(tf)
                fmin = tf[finite].min().item() if finite.any() else float('nan')
                fmax = tf[finite].max().item() if finite.any() else float('nan')
                print(f"    {name}: finite_ratio={finite.float().mean().item():.4f} "
                      f"min={fmin:.4g} max={fmax:.4g}")

        _stat('depth_preds', depth_preds)
        _stat('bev_roi_mask', bev_roi_mask)
        try:
            for ti, pd in enumerate(preds):
                for key in ('heatmap', 'reg', 'height', 'dim', 'rot', 'vel'):
                    if key in pd[0]:
                        _stat(f'student[{ti}].{key}', pd[0][key])
        except Exception as e:
            print(f"    (student stat skip: {e})")
        try:
            for ti, td in enumerate(lidar_preds):
                for key in ('heatmap', 'reg', 'height', 'dim', 'rot', 'vel'):
                    if key in td[0]:
                        _stat(f'teacher[{ti}].{key}', td[0][key])
        except Exception as e:
            print(f"    (teacher stat skip: {e})")

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

            # Compute in FP32 and average over channels.  The previous channel
            # sum made this term scale with C (128/256 here), reaching hundreds
            # before its 0.6 coefficient was applied.
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

    def prepare_batch_dict_from_pred_box(self, pred_box, gt_boxes, gt_labels, max_rois=256, max_gts=128):
        batch_size = len(pred_box)

        device = None
        for batch_item in pred_box:
            if isinstance(batch_item, list) and len(batch_item) > 0:
                if isinstance(batch_item[0], torch.Tensor):
                    device = batch_item[0].device
                    break
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        rois_padded = torch.zeros((batch_size, max_rois, 9), device=device, dtype=torch.float32)
        roi_scores_padded = torch.zeros((batch_size, max_rois), device=device, dtype=torch.float32)
        roi_labels_padded = torch.zeros((batch_size, max_rois), device=device, dtype=torch.long)
        gt_boxes_padded = torch.zeros((batch_size, max_gts, 10), device=device, dtype=torch.float32)

        for i in range(batch_size):
            if not isinstance(pred_box[i], list) or len(pred_box[i]) != 3:
                print(f"[WARNING] pred_box[{i}] format unexpected: {type(pred_box[i])}")
                continue

            cur_rois, cur_scores, cur_labels = pred_box[i]
            cur_rois = cur_rois.float()
            cur_scores = cur_scores.float()
            cur_labels = cur_labels.long()

            num_rois = min(cur_rois.shape[0], max_rois)
            if num_rois > 0:
                rois_padded[i, :num_rois] = cur_rois[:num_rois]
                roi_scores_padded[i, :num_rois] = cur_scores[:num_rois]
                roi_labels_padded[i, :num_rois] = cur_labels[:num_rois]

            cur_gts = gt_boxes[i].float()
            cur_gt_labels = gt_labels[i].long()
            num_gts = min(cur_gts.shape[0], max_gts)
            if num_gts > 0:
                gt_boxes_padded[i, :num_gts, :9] = cur_gts[:num_gts]
                gt_boxes_padded[i, :num_gts, 9] = cur_gt_labels[:num_gts]

        batch_dict = {
            'batch_size': batch_size,
            'rois': rois_padded,
            'roi_scores': roi_scores_padded,
            'roi_labels': roi_labels_padded,
            'gt_boxes_and_cls': gt_boxes_padded,
        }

        return batch_dict

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
        info_paths = (self.train_info_paths + [self.val_info_paths]
                      if self.use_train_val else self.train_info_paths)
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
        import math
        from torch.optim.lr_scheduler import LambdaLR

        # Keep the same linear scaling rule as the base experiment.  With the
        # requested 2 GPUs x 4 images this is 2.5e-5, whereas the previous
        # fixed 2e-4 was calibrated for a global batch of 64.
        lr = self.basic_lr_per_img * self.batch_size_per_device * self.gpus
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=self.optimizer_config.get('optimizer', dict(
                type='AdamW',
                lr=lr,
                weight_decay=self.optimizer_config.get('weight_decay', 1e-2)
            )),
            paramwise_cfg=self.optimizer_config.get('paramwise_cfg', None)
        )
        optimizer = build_optim_wrapper(self.model, optim_wrapper_cfg)
        if hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer

        warmup_iters = 1000
        warmup_ratio = 0.001
        # 余弦退火分母用 PL 估算的「整个训练实际优化器 step 总数」。
        # 它已把 DDP 多卡切分(world_size)、梯度累积、max_epochs 全部考虑在内，
        # 避免手动 len(dataloader)*max_epochs 在多卡下高估 world_size 倍
        # （例：16 卡会把 total_steps 放大 16 倍，导致余弦几乎不退火）。
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            # 回退：trainer 不可用时按单卡估算（仅用于离线构造/调试）
            max_epochs = getattr(self.trainer, 'max_epochs', None) or 20
            total_steps = max_epochs * len(self.train_dataloader())

        def lr_lambda(current_step):
            if current_step < warmup_iters:
                alpha = current_step / warmup_iters
                return warmup_ratio * (1 - alpha) + alpha
            else:
                progress = (current_step - warmup_iters) / max(1, total_steps - warmup_iters)
                progress = min(progress, 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


if __name__ == '__main__':
    run_cli(LabelDistillModel,
            'LidarDistill_convnextb_900x1600_J4_wl05_wh08_cbgs',
            extra_trainer_config_args={'epochs': 20, 'gradient_clip_val': 5.0},
            # Keep only the standard Lightning checkpoints under checkpoints/.
            # Enabling EMA also writes one large <epoch>.pth file per epoch to
            # lightning_logs/version_*, which this experiment does not use.
            use_ema=False)
