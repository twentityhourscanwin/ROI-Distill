#labeldistill/exps/nuscenes/base_exp.py
from copy import deepcopy
import os
from functools import partial

import mmengine
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from pytorch_lightning.core import LightningModule
from torch.cuda.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import MultiStepLR

from labeldistill.datasets.nusc_det_dataset import NuscDetDataset, collate_fn
from labeldistill.evaluators.det_evaluators import DetNuscEvaluator
from labeldistill.models.base_labeldistill import BaseBEVDepth
from labeldistill.utils.torch_dist import all_gather_object, get_rank, synchronize

H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8],
    'y_bound': [-51.2, 51.2, 0.8],
    'z_bound': [-5, 3, 8],
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}
ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}

bev_backbone = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[80, 160, 320, 640],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    'truck',
    'construction_vehicle',
    'bus',
    'trailer',
    'barrier',
    'motorcycle',
    'bicycle',
    'pedestrian',
    'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}


class LabelDistillModel(LightningModule):
    MODEL_NAMES = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith('__')
                         and callable(models.__dict__[name]))

    def __init__(self,
                 gpus: int = 1,
                 data_root='data/nuScenes',
                 eval_interval=1,
                 batch_size_per_device=8,
                 num_workers=4,
                 class_names=CLASSES,
                 backbone_conf=backbone_conf,
                 head_conf=head_conf,
                 ida_aug_conf=ida_aug_conf,
                 bda_aug_conf=bda_aug_conf,
                 default_root_dir='./outputs/',
                 use_train_val=False,  # 新增参数:是否使用train+val数据集
                 build_model=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'backbone_conf', 'head_conf', 'ida_aug_conf', 'bda_aug_conf',
            'build_model', 'config', 'model', 'matcher', 'scaler',
            'mask_generator'
        ])
        self.gpus = gpus
        self.eval_interval = eval_interval
        self.batch_size_per_device = batch_size_per_device
        self.num_workers = num_workers
        self.data_root = data_root
        self.basic_lr_per_img = 2e-4 / 64
        self.class_names = deepcopy(class_names)
        self.backbone_conf = deepcopy(backbone_conf)
        self.head_conf = deepcopy(head_conf)
        self.ida_aug_conf = deepcopy(ida_aug_conf)
        self.bda_aug_conf = deepcopy(bda_aug_conf)
        self.use_train_val = use_train_val  # 保存参数
        mmengine.mkdir_or_exist(default_root_dir)
        self.default_root_dir = default_root_dir
        self.evaluator = DetNuscEvaluator(class_names=self.class_names,
                                          output_dir=self.default_root_dir)
        self.model = None
        if build_model:
            self.model = BaseBEVDepth(self.backbone_conf,
                                      self.head_conf,
                                      is_train_depth=True)
        self.mode = 'valid'
        self.img_conf = img_conf
        self.data_use_cbgs = False
        self.num_sweeps = 1
        self.sweep_idxes = list()
        self.key_idxes = list()
        self.data_return_depth = True
        self.downsample_factor = self.backbone_conf['downsample_factor']
        self.dbound = self.backbone_conf['d_bound']
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.use_fusion = False
        self.train_info_paths = os.path.join(self.data_root,
                                             'nuscenes_infos_train.pkl')
        self.val_info_paths = os.path.join(self.data_root,
                                           'nuscenes_infos_val.pkl')
        self.predict_info_paths = os.path.join(self.data_root,
                                               'nuscenes_infos_test.pkl')
        self.save_resolved_configs()

    def save_resolved_configs(self):
        """Persist copies of the effective configs used to build the model."""
        self.hparams.update({
            'class_names': deepcopy(self.class_names),
            'backbone_conf': deepcopy(self.backbone_conf),
            'head_conf': deepcopy(self.head_conf),
            'ida_aug_conf': deepcopy(self.ida_aug_conf),
            'bda_aug_conf': deepcopy(self.bda_aug_conf),
        })

    def forward(self, sweep_imgs, mats):
        return self.model(sweep_imgs, mats)

    def training_step(self, batch):
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
            # only key-frame will calculate depth loss
            depth_labels = depth_labels[:, 0, ...]
        depth_loss = self.get_depth_loss(depth_labels.cuda(), depth_preds)
        self.log('detection_loss', detection_loss)
        self.log('depth_loss', depth_loss)
        return detection_loss + depth_loss

    def get_depth_loss(self, depth_labels, depth_preds, sigma=1.5):
        """Compute the shared Gaussian-label depth classification loss.

        DepthNet returns raw logits.  Valid projected LiDAR depths are kept as
        continuous bin indices, converted to Gaussian targets, and supervised
        with KL divergence against the log-softmax depth distribution.
        """
        depth_indices, fg_mask = self.get_downsampled_gt_depth(depth_labels)
        if not fg_mask.any():
            # Keep the zero connected to DepthNet for a valid backward pass.
            return depth_preds.sum() * 0.0

        soft_labels = self.create_soft_labels(
            depth_indices, fg_mask, sigma=sigma)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)

        with autocast(enabled=False):
            depth_loss = F.kl_div(
                F.log_softmax(depth_preds[fg_mask].float(), dim=1),
                soft_labels.float(),
                reduction='batchmean',
            )

        return 3.0 * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        """Downsample LiDAR depth and return continuous bins plus valid mask."""
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        nearest_depth = torch.where(
            gt_depths > 0.0,
            gt_depths,
            torch.full_like(gt_depths, float('inf')),
        ).min(dim=-1).values
        nearest_depth = nearest_depth.view(
            B * N,
            H // self.downsample_factor,
            W // self.downsample_factor,
        )

        fg_mask = (
            torch.isfinite(nearest_depth)
            & (nearest_depth >= self.dbound[0])
            & (nearest_depth < self.dbound[1])
        )
        depth_indices = (nearest_depth - self.dbound[0]) / self.dbound[2]
        depth_indices = torch.where(
            fg_mask, depth_indices, torch.zeros_like(depth_indices))
        depth_indices = torch.clamp(
            depth_indices, 0.0, float(self.depth_channels - 1))

        return depth_indices.view(-1), fg_mask.view(-1)

    def create_soft_labels(self, depth_indices, fg_mask, sigma=1.5):
        """Create normalized Gaussian targets around continuous depth bins."""
        if sigma <= 0:
            raise ValueError(f'depth label sigma must be positive, got {sigma}')

        depth_indices = depth_indices[fg_mask]
        if depth_indices.numel() == 0:
            return depth_indices.new_zeros((0, self.depth_channels))

        bin_indices = torch.arange(
            self.depth_channels,
            device=depth_indices.device,
            dtype=depth_indices.dtype,
        )
        distances = depth_indices[:, None] - bin_indices[None, :]
        weights = torch.exp(-0.5 * (distances / sigma) ** 2)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def eval_step(self, batch, batch_idx, prefix: str):
        (sweep_imgs, mats, _, img_metas, _, _) = batch
        if torch.cuda.is_available():
            for key, value in mats.items():
                mats[key] = value.cuda()
            sweep_imgs = sweep_imgs.cuda()
        preds = self.model(sweep_imgs, mats)
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

    def validation_step(self, batch, batch_idx):
        results = self.eval_step(batch, batch_idx, 'val')
        if not hasattr(self, '_validation_step_outputs'):
            self._validation_step_outputs = []
        self._validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        validation_step_outputs = getattr(self, '_validation_step_outputs', [])
        all_pred_results = list()
        all_img_metas = list()
        for validation_step_output in validation_step_outputs:
            for i in range(len(validation_step_output)):
                all_pred_results.append(validation_step_output[i][:3])
                all_img_metas.append(validation_step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)
        self._validation_step_outputs = []

    def on_test_epoch_end(self):
        test_step_outputs = getattr(self, '_test_step_outputs', [])
        all_pred_results = list()
        all_img_metas = list()
        for test_step_output in test_step_outputs:
            for i in range(len(test_step_output)):
                all_pred_results.append(test_step_output[i][:3])
                all_img_metas.append(test_step_output[i][3])
        synchronize()
        # TODO: Change another way.
        dataset_length = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:dataset_length]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:dataset_length]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)
        self._test_step_outputs = []

    def configure_optimizers(self):
        lr = self.basic_lr_per_img * \
            self.batch_size_per_device * self.gpus
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=lr,
                                      weight_decay=1e-7)
        scheduler = MultiStepLR(optimizer, [19, 23])
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        # 根据use_train_val参数决定使用哪些数据
        if self.use_train_val:
            # 使用train+val数据集
            info_paths = [self.train_info_paths, self.val_info_paths]
        else:
            # 仅使用train数据集
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
                                       use_fusion=self.use_fusion)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            collate_fn=partial(collate_fn,
                               is_return_depth=self.data_return_depth
                               or self.use_fusion),
            sampler=None,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                     bda_aug_conf=self.bda_aug_conf,
                                     classes=self.class_names,
                                     data_root=self.data_root,
                                     info_paths=self.val_info_paths,
                                     is_train=False,
                                     img_conf=self.img_conf,
                                     num_sweeps=self.num_sweeps,
                                     sweep_idxes=self.sweep_idxes,
                                     key_idxes=self.key_idxes,
                                     return_depth=self.use_fusion,
                                     use_fusion=self.use_fusion)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=self.num_workers,
            sampler=None,
        )
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        predict_dataset = NuscDetDataset(ida_aug_conf=self.ida_aug_conf,
                                         bda_aug_conf=self.bda_aug_conf,
                                         classes=self.class_names,
                                         data_root=self.data_root,
                                         info_paths=self.predict_info_paths,
                                         is_train=False,
                                         img_conf=self.img_conf,
                                         num_sweeps=self.num_sweeps,
                                         sweep_idxes=self.sweep_idxes,
                                         key_idxes=self.key_idxes,
                                         return_depth=self.use_fusion,
                                         use_fusion=self.use_fusion)
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=self.use_fusion),
            num_workers=self.num_workers,
            sampler=None,
        )
        return predict_loader

    def test_step(self, batch, batch_idx):
        results = self.eval_step(batch, batch_idx, 'test')
        if not hasattr(self, '_test_step_outputs'):
            self._test_step_outputs = []
        self._test_step_outputs.append(results)

    def predict_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'predict')

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        return parent_parser
