# labeldistill/models/lidardistill_full.py
# 与 lidardistill.py 的区别：不对学生 backbone 输出做通道切分，全部通道送入 adaptor 蒸馏
from torch import nn
from labeldistill.layers.backbones.base_lss_fpn import BaseLSSFPN
from labeldistill.layers.backbones.adaptor import DistillAdaptor
from labeldistill.layers.heads.kd_head import KDHead
from mmdet3d.registry import MODELS
import mmdet3d.models  # noqa: F401
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
import torch

__all__ = ['LabelDistillFull']


class LabelDistillFull(nn.Module):

    def __init__(self,
                 backbone_conf,
                 head_conf,
                 lidar_conf=None,
                 lidar_ckpt_path=None,
                 is_train_depth=False,
                 ):
        super(LabelDistillFull, self).__init__()
        self.backbone = BaseLSSFPN(**backbone_conf)
        self.head = KDHead(**head_conf)
        self.is_train_depth = is_train_depth

        # 不做 // 2，使用全部通道作为 adaptor 输入
        distill_in_feature = head_conf['bev_neck_conf']['in_channels'][:2]
        self.distill_encoder_lidar = DistillAdaptor(distill_in_feature,
                                                    out_features=[128, 256],
                                                    stride=[1, 1])

        voxel_layer_cfg = lidar_conf.pop('voxel_layer', None)
        if voxel_layer_cfg is not None:
            self.voxelizer = VoxelizationByGridShape(**voxel_layer_cfg)
        else:
            self.voxelizer = None

        self.centerpoint = MODELS.build(lidar_conf)

        lidar_params = torch.load(lidar_ckpt_path, map_location='cpu')
        prefix = 'model.centerpoint.'
        load_keys = [k for k in lidar_params['state_dict'] if k.startswith(prefix)]
        self.centerpoint.load_state_dict({k[len(prefix):]: lidar_params['state_dict'][k] for k in load_keys})
        self.centerpoint.eval()

    def forward(
        self,
        bev_mask=None,
        bev_box=None,
        bev_label=None,
        x=None,
        mats_dict=None,
        lidar_pts=None,
        timestamps=None,
    ):
        if self.is_train_depth and self.training:
            lidar_preds = None
            lidar_pred_box = None
            lidar_feats_out = None
            neck_feats = None

            with torch.no_grad():
                pts_list = [p for p in lidar_pts.squeeze(1)]
                voxels_list, coors_list, num_points_list = [], [], []
                for i, pts in enumerate(pts_list):
                    res = self.voxelizer(pts)
                    v, c, n = res
                    batch_idx = c.new_full((c.shape[0], 1), i)
                    c = torch.cat([batch_idx, c], dim=1)
                    voxels_list.append(v)
                    coors_list.append(c)
                    num_points_list.append(n)
                voxels = torch.cat(voxels_list, dim=0)
                coors = torch.cat(coors_list, dim=0)
                num_points = torch.cat(num_points_list, dim=0)

                voxel_features = self.centerpoint.pts_voxel_encoder(voxels, num_points, coors)
                batch_size = coors[-1, 0] + 1
                lidar_feats = self.centerpoint.pts_middle_encoder(voxel_features, coors, batch_size)
                lidar_feats = self.centerpoint.pts_backbone(lidar_feats)
                lidar_feats_out = lidar_feats

                neck_feats = self.centerpoint.pts_neck(lidar_feats)
                lidar_preds = self.centerpoint.pts_bbox_head(neck_feats)
                lidar_pred_box = self.get_bboxes(lidar_preds)

            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)

            preds, backbone_out, neck_output = self.head(x)
            image_pred_box = self.get_bboxes(preds)

            # 不做通道切分，全部通道送入 adaptor
            distill_feats_lidar = self.distill_encoder_lidar([backbone_out[0],
                                                              backbone_out[1]])

            return preds, lidar_preds, depth_pred, distill_feats_lidar, lidar_feats_out, neck_feats, neck_output, lidar_pred_box, image_pred_box
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds, _, x = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        return self.head.loss(targets, preds_dicts)

    def response_loss(self, targets, preds_dicts, teacher_dicts):
        return self.head.response_loss(targets, preds_dicts, teacher_dicts)

    def distill_loss(self, targets, preds_dicts, teacher_dicts):
        return self.head.distill_loss(targets, preds_dicts, teacher_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
