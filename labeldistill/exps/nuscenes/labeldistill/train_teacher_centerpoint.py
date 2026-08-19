# labeldistill/exps/nuscenes/labeldistill/train_teacher_centerpoint.py
"""单独训练教师网络 (CenterPoint, voxel 0.1 / BEV 128x128) 的脚本。

设计目标
--------
1. 直接复用蒸馏脚本 ``LidarDistill_r50_128x128_e24_roi.py`` 中 *完全相同* 的
   CenterPoint 结构 (``lidar_conf``) 以及 *完全相同* 的激光数据管线
   (``NuscDetDataset``, ``return_lidar=True``)，保证训练出的教师权重在结构与输入
   分布上与蒸馏时一致，可被蒸馏脚本直接当作教师加载。
2. 前向沿用 ``models/lidardistill.py`` 里 mmdet3d 1.4 的写法 (手动体素化 ->
   middle_encoder -> backbone -> neck -> CenterHead)，但 *不* 加 ``no_grad``，
   使梯度能够回传以训练教师。
3. 训练目标 + 检测损失复用仓库现成的 ``BEVDepthHead.get_targets / loss``：
   它接受 ``gt_boxes/gt_labels`` 张量并生成 128x128 的 heatmap/回归目标，与
   CenterPoint 的输出天然对齐 (这也是蒸馏中 ``response_loss`` 能跑通的原因)。

关键点：权重前缀
----------------
本模块把 CenterPoint 放在 ``self.model.centerpoint`` 下，因此 PyTorch-Lightning
保存的 checkpoint 中其参数 key 形如 ``model.centerpoint.*``，正是蒸馏脚本加载教师
时使用的前缀::

    prefix = 'model.centerpoint.'

所以 **无需任何转换**，训练完成后把蒸馏脚本里的 ``lidar_ckpt_path`` 指向本脚本产出
的 ``.ckpt`` 即可。

用法
----
训练 (例如 4 卡, 每卡 batch 4)::

    python labeldistill/exps/nuscenes/labeldistill/train_teacher_centerpoint.py --gpus 4 -b 4

产物::

    ./outputs/train_teacher_centerpoint/checkpoints/last.ckpt   (以及 epoch_xx.ckpt)

在蒸馏脚本中使用::

    lidar_ckpt_path = './outputs/train_teacher_centerpoint/checkpoints/last.ckpt'

备注
----
- 为保持「单文件 + 完全一致的数据/标签处理」，这里复用了同时加载相机图像的
  ``NuscDetDataset``；教师训练并不需要图像，图像加载属于额外开销 (可后续换成
  纯激光数据集来加速)。
- 若 16-mixed 下出现数值不稳定/OOM，可改用 ``--precision 32``。
"""
import os
from functools import partial

import torch
import torch.nn as nn
from pytorch_lightning.core import LightningModule
from torch.optim.lr_scheduler import MultiStepLR

import mmdet.models.losses  # noqa: F401  注册 GaussianFocalLoss / L1Loss
import mmdet3d.models       # noqa: F401  注册 CenterPoint / SparseEncoder / SECOND 等
from mmdet3d.registry import MODELS
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape

from labeldistill.exps.base_cli import run_cli
from labeldistill.exps.nuscenes.base_exp import (CLASSES, ida_aug_conf,
                                                 bda_aug_conf, img_conf)
from labeldistill.layers.heads.bev_depth_head import BEVDepthHead
from labeldistill.datasets.nusc_det_dataset_lidar import (NuscDetDataset,
                                                          collate_fn)
from labeldistill.evaluators.det_evaluators import DetNuscEvaluator
from labeldistill.utils.torch_dist import (all_gather_object, get_rank,
                                           synchronize)

# ----------------------------------------------------------------------------
# CenterPoint 配置 (与 LidarDistill_r50_128x128_e24_roi.py 中 self.lidar_conf 一致)
# ----------------------------------------------------------------------------
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]
common_heads = dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=8,
    voxel_size=voxel_size[:2],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9)

# CenterHead 自身的 train/test_cfg —— 仅供构建检测器；前向不依赖它们，
# 训练目标/损失统一交给下面的 target_head 负责。
lidar_train_cfg = dict(pts=dict(
    grid_size=[1024, 1024, 40],
    voxel_size=voxel_size,
    out_size_factor=8,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]))
lidar_test_cfg = dict(pts=dict(
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

lidar_conf = dict(
    type='CenterPoint',
    voxel_layer=dict(
        point_cloud_range=point_cloud_range, max_num_points=10,
        voxel_size=voxel_size, max_voxels=(90000, 120000)),
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
        tasks=TASKS,
        common_heads=common_heads,
        share_conv_channel=64,
        bbox_coder=bbox_coder,
        train_cfg=lidar_train_cfg,
        test_cfg=lidar_test_cfg,
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
)

# 仅用于「生成训练目标 + 计算检测损失」的 head 的 train_cfg。
# 注意是扁平结构 (含 point_cloud_range)，与 BEVDepthHead.get_targets_single 的取值一致。
head_train_cfg = dict(
    point_cloud_range=point_cloud_range,
    grid_size=[1024, 1024, 40],
    voxel_size=voxel_size,
    out_size_factor=8,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])


class _CenterPointWrapper(nn.Module):
    """把 CenterPoint 包一层，使其参数在 LightningModule 内以 ``model.centerpoint.*``
    命名，与蒸馏脚本加载教师权重时使用的前缀保持一致。"""

    def __init__(self, conf):
        super().__init__()
        self.centerpoint = MODELS.build(conf)


class TeacherCenterPointModel(LightningModule):
    """纯激光 CenterPoint 教师网络的训练 LightningModule。"""

    def __init__(self,
                 gpus: int = 1,
                 data_root: str = 'data/nuScenes',
                 batch_size_per_device: int = 4,
                 default_root_dir: str = './outputs/',
                 use_train_val: bool = True,
                 use_cbgs: bool = True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.gpus = gpus
        self.data_root = data_root
        self.batch_size_per_device = batch_size_per_device
        self.default_root_dir = default_root_dir
        self.use_train_val = use_train_val
        self.use_cbgs = use_cbgs
        # run_cli 会把 max_epochs 透传到 **kwargs，用于推导学习率衰减节点
        self.max_epochs = kwargs.get('max_epochs', 20)

        self.class_names = CLASSES
        self.ida_aug_conf = ida_aug_conf
        self.bda_aug_conf = bda_aug_conf
        self.img_conf = img_conf
        self.train_info_paths = os.path.join(data_root,
                                             'nuscenes_infos_train.pkl')
        self.val_info_paths = os.path.join(data_root, 'nuscenes_infos_val.pkl')

        # mmdet3d 1.4 把 voxel_layer 从检测器里移出，需单独构建 voxelizer
        conf = {k: v for k, v in lidar_conf.items() if k != 'voxel_layer'}
        self.voxelizer = VoxelizationByGridShape(**lidar_conf['voxel_layer'])

        # 待训练的教师检测器（参数 key: model.centerpoint.*）
        self.model = _CenterPointWrapper(conf)

        # 仅用于 get_targets + loss；其内部卷积层 (trunk/neck/task_heads) 既不参与
        # 教师前向，也不加入优化器，仅作为目标生成/损失计算的载体。
        self.target_head = BEVDepthHead(
            in_channels=sum([256, 256]),
            tasks=TASKS,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
            loss_bbox=dict(type='mmdet.L1Loss', reduction='mean',
                           loss_weight=0.25),
            gaussian_overlap=0.1,
            min_radius=2,
            train_cfg=head_train_cfg,
            test_cfg=lidar_test_cfg['pts'],
            separate_head=dict(
                type='SeparateHead', init_bias=-2.19, final_kernel=3))

        # 验证集评估器 (官方 nuScenes mAP/NDS)，与主模型 base_exp 一致
        self.evaluator = DetNuscEvaluator(class_names=self.class_names,
                                          output_dir=self.default_root_dir)

    def _forward_centerpoint(self, lidar_pts):
        """复刻 lidardistill.py 的前向，但允许梯度回传。

        Args:
            lidar_pts (Tensor): [B, 1, N, 5] 激光点 (已在 dataset 内做过 BDA 增强)。

        Returns:
            tuple(list[dict]): CenterHead 的逐 task 预测 (与 BEVDepthHead.loss 对齐)。
        """
        cp = self.model.centerpoint
        voxels_list, coors_list, num_points_list = [], [], []
        for i, pts in enumerate(lidar_pts.squeeze(1)):
            voxels, coors, num_points = self.voxelizer(pts)
            # 在 coors 前面插入 batch_idx 列
            coors = torch.cat(
                [coors.new_full((coors.shape[0], 1), i), coors], dim=1)
            voxels_list.append(voxels)
            coors_list.append(coors)
            num_points_list.append(num_points)
        voxels = torch.cat(voxels_list, dim=0)
        coors = torch.cat(coors_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)

        voxel_features = cp.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = int(coors[-1, 0].item()) + 1
        x = cp.pts_middle_encoder(voxel_features, coors, batch_size)
        x = cp.pts_backbone(x)
        x = cp.pts_neck(x)
        return cp.pts_bbox_head(x)

    def training_step(self, batch, batch_idx=None):
        # return_depth=False, return_lidar=True 时, batch 为 7 元组
        _imgs, _mats, _ts, _metas, gt_boxes, gt_labels, lidar_pts = batch[:7]
        gt_boxes = [g.cuda() for g in gt_boxes]
        gt_labels = [g.cuda() for g in gt_labels]
        lidar_pts = lidar_pts.cuda()

        preds = self._forward_centerpoint(lidar_pts)
        # 与 KDHead 一致：损失在 fp32 下计算，避免 16-mixed 下的数值问题
        for task in preds:
            for key in list(task[0].keys()):
                task[0][key] = task[0][key].float()

        targets = self.target_head.get_targets(gt_boxes, gt_labels)
        loss = self.target_head.loss(targets, preds)
        self.log('det_loss', loss, prog_bar=True)
        return loss

    def eval_step(self, batch, batch_idx, prefix: str):
        """教师网络验证集前向 + 解码。

        与 base_exp.eval_step 对齐：``is_train=False`` 时无 BDA 增强，激光点与输出框
        都在 ego 系，正好满足 DetNuscEvaluator 的 ego->global 转换假设。解码复用
        ``target_head.get_bboxes`` (CenterPointBBoxCoder + circle NMS)，与主模型评估同源。
        """
        _imgs, _mats, _ts, img_metas, _gt_boxes, _gt_labels, lidar_pts = batch[:7]
        lidar_pts = lidar_pts.cuda()

        preds = self._forward_centerpoint(lidar_pts)
        for task in preds:
            for key in list(task[0].keys()):
                task[0][key] = task[0][key].float()

        results = self.target_head.get_bboxes(preds, img_metas)
        for i in range(len(results)):
            results[i][0] = results[i][0].detach().cpu().numpy()
            results[i][1] = results[i][1].detach().cpu().numpy()
            results[i][2] = results[i][2].detach().cpu().numpy()
            results[i].append(img_metas[i])
        return results

    def test_step(self, batch, batch_idx):
        results = self.eval_step(batch, batch_idx, 'test')
        if not hasattr(self, '_test_step_outputs'):
            self._test_step_outputs = []
        self._test_step_outputs.append(results)

    def validation_step(self, batch, batch_idx):
        results = self.eval_step(batch, batch_idx, 'val')
        if not hasattr(self, '_validation_step_outputs'):
            self._validation_step_outputs = []
        self._validation_step_outputs.append(results)

    def _aggregate_and_evaluate(self, step_outputs):
        all_pred_results = list()
        all_img_metas = list()
        for step_output in step_outputs:
            for i in range(len(step_output)):
                all_pred_results.append(step_output[i][:3])
                all_img_metas.append(step_output[i][3])
        synchronize()
        len_dataset = len(self.val_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        if get_rank() == 0:
            self.evaluator.evaluate(all_pred_results, all_img_metas)

    def on_test_epoch_end(self):
        self._aggregate_and_evaluate(getattr(self, '_test_step_outputs', []))
        self._test_step_outputs = []

    def on_validation_epoch_end(self):
        self._aggregate_and_evaluate(
            getattr(self, '_validation_step_outputs', []))
        self._validation_step_outputs = []

    def configure_optimizers(self):
        # 仅优化 CenterPoint (self.model)；target_head 不参与训练
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4,
                                      weight_decay=1e-2)
        milestones = [int(self.max_epochs * 0.8), int(self.max_epochs * 0.95)]
        scheduler = MultiStepLR(optimizer, milestones)
        return [[optimizer], [scheduler]]

    def train_dataloader(self):
        info_paths = ([self.train_info_paths, self.val_info_paths]
                      if self.use_train_val else self.train_info_paths)
        train_dataset = NuscDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_paths=info_paths,
            is_train=True,
            use_cbgs=self.use_cbgs,
            img_conf=self.img_conf,
            num_sweeps=1,
            sweep_idxes=[],
            key_idxes=[],
            return_depth=False,
            return_lidar=True,
            use_fusion=False)
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=8,
            drop_last=True,
            shuffle=True,
            collate_fn=partial(collate_fn, is_return_depth=False,
                               is_return_lidar=True),
            sampler=None)

    def val_dataloader(self):
        val_dataset = NuscDetDataset(
            ida_aug_conf=self.ida_aug_conf,
            bda_aug_conf=self.bda_aug_conf,
            classes=self.class_names,
            data_root=self.data_root,
            info_paths=self.val_info_paths,
            is_train=False,
            use_cbgs=False,
            img_conf=self.img_conf,
            num_sweeps=1,
            sweep_idxes=[],
            key_idxes=[],
            return_depth=False,
            return_lidar=True,
            use_fusion=False)
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=8,
            drop_last=False,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=False,
                               is_return_lidar=True),
            sampler=None)

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_callbacks(self):
        # 为便于快速验证：除 base_cli 里按 epoch 保存外，这里再每 N 步保存一次，
        # 这样不必等满一个 epoch 就能拿到 checkpoint 来测试加载。
        from pytorch_lightning.callbacks import ModelCheckpoint
        return [ModelCheckpoint(
            dirpath=os.path.join(self.default_root_dir, 'checkpoints'),
            filename='step_{step}',
            save_top_k=1,
            save_last=True,
            every_n_train_steps=200,
            save_on_train_epoch_end=False)]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        return parent_parser


if __name__ == '__main__':
    run_cli(TeacherCenterPointModel,
            'train_teacher_centerpoint',
            extra_trainer_config_args={'epochs': 20},
            use_ema=False)
