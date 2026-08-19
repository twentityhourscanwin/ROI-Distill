#labeldistill/exps/nuscenes/bevdepth/BEVDepth_convnextb_900x1600_e30.py
# BEVDepth + ConvNeXt-B 900x1600 纯检测训练（无蒸馏）
# 用途：作为蒸馏实验的 baseline / student 预训练
#
# 训练策略：
#   1. Warmup(1000步) + 余弦退火 学习率调度
#   2. backbone lr_mult=0.1，保护 ConvNeXt-B 预训练权重
#   3. drop_path_rate=0.3（随机深度正则化）
#   4. with_cp=True（梯度检查点节省显存）
#   5. KL散度软标签深度损失（高斯软化 sigma=1.5）
#   6. 时序融合：key_idxes=[-2,-4]（2个历史帧 + 当前帧）
#   7. 30 epoch，效果优先
import math
import os
from functools import partial

# 导入自定义 ConvNeXt wrapper，触发注册到 mmdet::model registry
import labeldistill.layers.backbones.convnext_backbone  # noqa: F401

import torch
import torch.nn as nn
from mmengine.optim import build_optim_wrapper

from labeldistill.exps.base_cli import run_cli
from labeldistill.exps.nuscenes.base_exp_convnextb import \
    LabelDistillModel as BaseBEVDepthModel
from labeldistill.datasets.nusc_det_dataset import NuscDetDataset, collate_fn


class BEVDepthConvNeXtModel(BaseBEVDepthModel):
    """
    BEVDepth + ConvNeXt-B（纯检测，无蒸馏）

    复用 base_exp_convnextb.py 的全部配置：
      - ConvNeXt-B backbone（900x1600）
      - BaseBEVDepth 模型（detection_loss + depth_loss）
      - KL 散度软标签深度损失

    本脚本新增/覆盖：
      - 时序融合（key_idxes=[-2,-4]）
      - Warmup + 余弦退火 学习率调度
      - backbone lr_mult=0.1
      - DDP 静态图初始化（配合 with_cp=True）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ==================== 时序融合 ====================
        self.key_idxes = [-2, -4]  # 2个历史帧 + 当前帧 = 3帧时序融合

        # 调整通道配置以适配时序拼接
        # BEV 特征拼接后：output_channels * (num_key_frames + 1)
        output_channels = self.backbone_conf['output_channels']  # 80
        num_frames = len(self.key_idxes) + 1  # 3

        # BEV backbone 输入通道 = 时序拼接后的通道数
        self.head_conf['bev_backbone_conf']['in_channels'] = output_channels * num_frames  # 240

        # BEV neck 输入通道：[时序拼接特征, stage0, stage1, stage2]
        base_channels = self.head_conf['bev_backbone_conf']['base_channels']  # 160
        self.head_conf['bev_neck_conf']['in_channels'] = [
            output_channels * num_frames,   # 240（时序拼接跳跃连接）
            base_channels,                  # 160（stage 0）
            base_channels * 2,              # 320（stage 1）
            base_channels * 4,              # 640（stage 2）
        ]

        # ==================== 优化器配置 ====================
        self.optimizer_config = dict(
            type='AdamW',
            lr=2e-4,
            paramwise_cfg=dict(
                custom_keys={
                    'backbone': dict(lr_mult=0.1),  # ConvNeXt-B 预训练权重学习率降10倍
                }),
            weight_decay=1e-2)

        # 重新构建模型（使用更新后的 head_conf）
        from labeldistill.models.base_labeldistill import BaseBEVDepth
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)

    def on_train_start(self):
        """
        DDP 初始化钩子：
        ConvNeXt with_cp=True 使用梯度检查点，会触发 DDP "unused parameters" 问题，
        需要启用 _set_static_graph() 提升效率。
        """
        is_ddp_wrapped = isinstance(self.trainer.model, nn.parallel.DistributedDataParallel)

        if is_ddp_wrapped:
            print("✅ 模型已被 DDP 封装。")
            if self.backbone_conf.get('img_backbone_conf', {}).get('with_cp', False):
                if hasattr(self.trainer.model, '_set_static_graph'):
                    print(">>> ⚙️ 启用 DDP 静态图：解决梯度检查点冲突...")
                    self.trainer.model._set_static_graph()
                    print(">>> 启用成功。")
                else:
                    print(">>> ⚠️ 警告: DDP 实例不支持 _set_static_graph。")
        else:
            print("ℹ️ 模型未使用 DDP（单卡模式）。")

    def training_step(self, batch):
        """
        纯 BEVDepth 训练步骤：detection_loss + depth_loss
        继承自 base_exp_convnextb.py，增加更详细的日志记录
        """
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
            gt_boxes = [gt_box.cuda() for gt_box in gt_boxes]
            gt_labels = [gt_label.cuda() for gt_label in gt_labels]

        preds, depth_preds = self(sweep_imgs, mats)

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            targets = self.model.module.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.module.loss(targets, preds)
        else:
            targets = self.model.get_targets(gt_boxes, gt_labels)
            detection_loss = self.model.loss(targets, preds)

        if len(depth_labels.shape) == 5:
            depth_labels = depth_labels[:, 0, ...]

        # KL 散度软标签深度损失（继承自 base_exp_convnextb.py）
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds, sigma=1.5)

        total_loss = detection_loss + depth_loss

        self.log('train/total_loss',     total_loss,     on_step=True, prog_bar=True,  sync_dist=True)
        self.log('train/detection_loss', detection_loss, on_step=True, prog_bar=True,  sync_dist=True)
        self.log('train/depth_loss',     depth_loss,     on_step=True, prog_bar=True,  sync_dist=True)

        return total_loss

    def train_dataloader(self):
        train_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                       bda_aug_conf=self.bda_aug_conf,
                                       classes=self.class_names,
                                       data_root=self.data_root,
                                       info_paths=self.train_info_paths,
                                       is_train=True,
                                       use_cbgs=self.data_use_cbgs,
                                       img_conf=self.img_conf,
                                       num_sweeps=self.num_sweeps,
                                       sweep_idxes=self.sweep_idxes,
                                       key_idxes=self.key_idxes,
                                       return_depth=self.data_return_depth,
                                       use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=4,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion),
            sampler=None,
        )
        return train_loader

    def configure_optimizers(self):
        from torch.optim.lr_scheduler import LambdaLR

        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='AdamW',
                lr=self.optimizer_config.get('lr', 2e-4),
                weight_decay=self.optimizer_config.get('weight_decay', 1e-2)
            ),
            paramwise_cfg=self.optimizer_config.get('paramwise_cfg', None)
        )
        optimizer = build_optim_wrapper(self.model, optim_wrapper_cfg)
        if hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer

        # ===== Warmup(1000步) + 余弦退火 =====
        warmup_iters = 1000
        warmup_ratio = 0.001
        max_epochs = 30
        steps_per_epoch = len(self.train_dataloader())
        total_steps = max_epochs * steps_per_epoch

        def lr_lambda(current_step):
            """
            Phase 1 (0~999步): 线性 Warmup
            Phase 2 (1000步~结束): 余弦退火到 0
            """
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
    run_cli(BEVDepthConvNeXtModel,
            'BEVDepth_convnextb_900x1600_e30',
            extra_trainer_config_args={'epochs': 30},
            use_ema=True)

