"""Config-driven J4 Lightning experiment."""

from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.optim import build_optim_wrapper
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import MultiStepLR

from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from labeldistill.exps.nuscenes.base_exp import LabelDistillModel as BaseExperiment


class J4Experiment(BaseExperiment):
    """The shared J4 training loop with all experiment policies injected."""

    def __init__(self, *, config, model, matcher, scaler, mask_generator,
                 backbone_conf, head_conf):
        self.config = config
        super().__init__(
            gpus=config.runtime.gpus,
            data_root=config.data.root,
            batch_size_per_device=config.runtime.batch_size_per_device,
            num_workers=config.data.num_workers,
            class_names=list(config.classes.names),
            backbone_conf=backbone_conf,
            head_conf=head_conf,
            default_root_dir=config.runtime.output_dir,
            use_train_val=config.data.split == 'trainval',
            build_model=False,
        )
        self.model = model
        self.proposal_target_layer = matcher
        self.change_gt = scaler
        self.generate_bev_mask = mask_generator
        self.scaler_enabled = bool(config.region.scaler.enabled)
        self.key_idxes = list(config.data.key_idxes)
        self.data_use_cbgs = bool(config.data.use_cbgs)
        self.data_return_depth = True
        self.data_return_lidar = True
        self.train_info_paths = self._resolve_info_path(config.data.train_info)
        self.val_info_paths = self._resolve_info_path(config.data.val_info)
        self.loss_weights = {
            'detection': float(config.loss.detection_weight),
            'depth': float(config.loss.depth_weight),
            'feature': float(config.loss.feature_weight),
            'response': float(config.loss.response_weight),
        }
        self.effective_learning_rate = float(
            config.derived.effective_learning_rate)
        self.optimizer_weight_decay = float(config.optimizer.weight_decay)
        self.backbone_lr_mult = float(config.optimizer.backbone_lr_mult)
        self.scheduler_milestones = list(config.scheduler.milestones)
        self.resolved_config = OmegaConf.to_container(config, resolve=True)
        self.hparams.update({'resolved_config': self.resolved_config})

    def _resolve_info_path(self, value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.data_root) / path
        return str(path)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['labeldistill_schema_version'] = self.config.schema_version
        checkpoint['labeldistill_resolved_config'] = self.resolved_config

    @staticmethod
    def _call_target(model):
        return model.module if isinstance(
            model, torch.nn.parallel.DistributedDataParallel) else model

    def training_step(self, batch, batch_idx=None):
        (sweep_imgs, mats, _, _, gt_boxes, gt_labels,
         lidar_pts, depth_labels) = batch
        target_model = self._call_target(self.model)
        bev_mask, bev_box, bev_label, targets = target_model.get_targets(
            gt_boxes, gt_labels)
        outputs = self.model(
            bev_mask, bev_box, bev_label, sweep_imgs, mats, lidar_pts)
        student = outputs.student
        teacher = outputs.teacher

        match_result = self.proposal_target_layer(
            teacher.proposals, gt_boxes, gt_labels)
        if self.scaler_enabled:
            match_result = self.change_gt.forward(match_result)
        bev_roi_mask = self.generate_bev_mask(match_result, len(gt_boxes))

        detection_loss, response_loss = target_model.response_loss(
            targets, student.raw_preds, teacher.raw_preds)
        if len(depth_labels.shape) == 5:
            depth_labels = depth_labels[:, 0, ...]
        depth_labels = depth_labels.to(student.depth.device)
        depth_loss = self.get_depth_loss(depth_labels, student.depth)
        feature_loss = self.get_feature_distill_loss(
            teacher.backbone_features,
            student.distill_features,
            bev_roi_mask,
        )

        weighted_detection = detection_loss * self.loss_weights['detection']
        weighted_depth = depth_loss * self.loss_weights['depth']
        weighted_feature = feature_loss * self.loss_weights['feature']
        weighted_response = response_loss * self.loss_weights['response']
        total_loss = (
            weighted_detection + weighted_depth
            + weighted_feature + weighted_response)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f'Non-finite total loss at global_step={self.global_step}: '
                f'detection={weighted_detection.detach().float().item():.6g}, '
                f'response={weighted_response.detach().float().item():.6g}, '
                f'depth={weighted_depth.detach().float().item():.6g}, '
                f'distill={weighted_feature.detach().float().item():.6g}')

        self.log('train/total_loss', total_loss, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('train/detection_loss', weighted_detection, on_step=True,
                 prog_bar=True, sync_dist=True)
        self.log('train/response_loss', weighted_response, on_step=True,
                 sync_dist=True)
        self.log('train/depth_loss', weighted_depth, on_step=True,
                 sync_dist=True)
        self.log('train/distill_loss', weighted_feature, on_step=True,
                 sync_dist=True)
        return total_loss

    @staticmethod
    def get_feature_distill_loss(lidar_feat, distill_feats, bev_mask=None,
                                 binary_mask=False):
        if len(lidar_feat) != len(distill_feats):
            raise ValueError(
                f'Feature level mismatch: teacher={len(lidar_feat)}, '
                f'student={len(distill_feats)}')
        losses = distill_feats[0].new_zeros((), dtype=torch.float32)
        if bev_mask is not None:
            bev_mask = [
                F.interpolate(bev_mask.float(), size=feat.shape[-2:],
                              mode='bilinear', align_corners=False)
                for feat in lidar_feat
            ]
            if binary_mask:
                bev_mask = [(mask > 0).float() for mask in bev_mask]

        for teacher_feature, student_feature, level_mask in zip(
                lidar_feat, distill_feats,
                bev_mask if bev_mask is not None else [None] * len(lidar_feat)):
            if teacher_feature.shape != student_feature.shape:
                raise ValueError(
                    'Feature shape mismatch: '
                    f'teacher={tuple(teacher_feature.shape)}, '
                    f'student={tuple(student_feature.shape)}')
            level_loss = (
                teacher_feature.detach().float() - student_feature.float()
            ).square().mean(dim=1)
            if level_mask is None:
                losses += level_loss.mean()
            else:
                spatial_mask = level_mask[:, 0]
                losses += (
                    (level_loss * spatial_mask).sum()
                    / spatial_mask.sum().clamp_min(1.0))
        return losses

    def eval_step(self, batch, batch_idx, prefix: str):
        sweep_imgs, mats, _, img_metas, _, _ = batch
        preds = self.model(x=sweep_imgs, mats_dict=mats)
        results = self._call_target(self.model).get_bboxes(preds, img_metas)
        for result, meta in zip(results, img_metas):
            result[0] = result[0].detach().cpu().numpy()
            result[1] = result[1].detach().cpu().numpy()
            result[2] = result[2].detach().cpu().numpy()
            result.append(meta)
        return results

    def train_dataloader(self):
        info_paths = (
            [self.train_info_paths, self.val_info_paths]
            if self.use_train_val else self.train_info_paths)
        dataset = NuscDetDataset(
            ida_aug_conf=self.ida_aug_conf,
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
            return_depth=True,
            return_lidar=True,
            use_fusion=self.use_fusion,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(
                collate_fn, is_return_depth=True, is_return_lidar=True),
            sampler=None,
        )

    def configure_optimizers(self):
        wrapper_config = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='AdamW',
                lr=self.effective_learning_rate,
                weight_decay=self.optimizer_weight_decay,
            ),
            paramwise_cfg=dict(custom_keys={
                'backbone': dict(lr_mult=self.backbone_lr_mult),
            }),
        )
        optimizer = build_optim_wrapper(self.model, wrapper_config)
        if hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        scheduler = MultiStepLR(optimizer, self.scheduler_milestones)
        return [[optimizer], [scheduler]]
