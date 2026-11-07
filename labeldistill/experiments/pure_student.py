"""Teacher-free training with the shared student optimizer and evaluator."""
from functools import partial

import torch

from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from labeldistill.experiments.j4 import J4Experiment


class PureStudentExperiment(J4Experiment):
    """Train GT detection and depth supervision without executing KD routes."""

    def __init__(self, *, config, model, backbone_conf, head_conf,
                 ida_aug_conf, bda_aug_conf, img_conf):
        if config.loss.feature_weight != 0 or config.loss.response_weight != 0:
            raise ValueError('Pure student requires both KD weights to be zero')
        super().__init__(
            config=config, model=model, matcher=None, scaler=None,
            mask_generator=None, feature_loss_reducer=None,
            backbone_conf=backbone_conf, head_conf=head_conf,
            ida_aug_conf=ida_aug_conf, bda_aug_conf=bda_aug_conf, img_conf=img_conf)
        self.data_return_lidar = False

    def training_step(self, batch, batch_idx=None):
        sweep_imgs, mats, _, _, gt_boxes, gt_labels, depth_labels = batch
        model = self._call_target(self.model)
        targets = model.get_targets(gt_boxes, gt_labels)
        output = self.model(x=sweep_imgs, mats_dict=mats)
        detection_loss = model.loss(targets, output.raw_preds)
        if depth_labels.ndim == 5:
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(
            depth_labels.to(output.depth.device), output.depth)
        weighted_detection = detection_loss * self.loss_weights['detection']
        weighted_depth = depth_loss * self.loss_weights['depth']
        total_loss = weighted_detection + weighted_depth
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f'Non-finite student loss at global_step={self.global_step}: '
                f'detection={weighted_detection.detach().float().item():.6g}, '
                f'depth={weighted_depth.detach().float().item():.6g}')
        for name, value in (
                ('total_loss', total_loss),
                ('detection_loss', weighted_detection),
                ('depth_loss', weighted_depth)):
            self.log(f'train/{name}', value, on_step=True,
                     prog_bar=True, sync_dist=True)
        if self.config.diagnostics.enabled:
            zero = total_loss.detach().new_zeros(())
            self._diagnostics_batch = ({
                'raw/detection': detection_loss.detach(),
                'raw/depth': depth_loss.detach(),
                'raw/feature': zero, 'raw/response': zero,
                'weighted/detection': weighted_detection.detach(),
                'weighted/depth': weighted_depth.detach(),
                'weighted/feature': zero, 'weighted/response': zero,
                'total': total_loss.detach(),
            }, int(sweep_imgs.shape[0]))
        return total_loss

    def train_dataloader(self):
        info_paths = (
            [self.train_info_paths, self.val_info_paths]
            if self.use_train_val else self.train_info_paths)
        dataset = NuscDetDataset(
            ida_aug_conf=self.ida_aug_conf, bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names, data_root=self.data_root,
            info_paths=info_paths, is_train=True, use_cbgs=self.data_use_cbgs,
            img_conf=self.img_conf, num_sweeps=self.num_sweeps,
            sweep_idxes=self.sweep_idxes, key_idxes=self.key_idxes,
            return_depth=True, return_lidar=False, use_fusion=self.use_fusion,
            history_fallback=self.config.data.train_history_fallback)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size_per_device,
            num_workers=self.num_workers, drop_last=True, shuffle=False,
            collate_fn=partial(
                collate_fn, is_return_depth=True, is_return_lidar=False),
            sampler=None)
