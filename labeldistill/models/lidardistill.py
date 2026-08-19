# labeldistill/models/lidardistill.py
from copy import deepcopy

from torch import nn
import matplotlib.pyplot as plt
from labeldistill.layers.backbones.base_lss_fpn import BaseLSSFPN
from labeldistill.layers.backbones.adaptor import DistillAdaptor

from labeldistill.layers.heads.kd_head import KDHead
from labeldistill.models.distill_outputs import (
    LabelDistillOutput,
    StudentOutput,
    TeacherOutput,
)
from labeldistill.models.teacher_proposal_decoder import TeacherProposalDecoder
from mmdet3d.registry import MODELS
import mmdet3d.models  # noqa: F401  触发 mmdet3d 所有模型注册
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
import torch

__all__ = ['LabelDistill', 'select_temporal_kd_channels']


def select_temporal_kd_channels(feature, frame_channels):
    """Select current + t-2 + half(t-6) from a five-frame BEV tensor.

    Expected concatenation order: current, t-2, t-4, t-6, t-8.
    The output keeps half of the total channels without changing the
    detection head's chronological input order.
    """
    expected_channels = frame_channels * 5
    if feature.ndim != 4:
        raise ValueError(
            f'Expected a 4D BEV tensor, got shape={tuple(feature.shape)}')
    if feature.shape[1] != expected_channels:
        raise ValueError(
            f'Expected {expected_channels} temporal channels '
            f'(5 x {frame_channels}), got {feature.shape[1]}')
    if frame_channels % 2 != 0:
        raise ValueError(
            f'frame_channels must be even, got {frame_channels}')

    current = feature[:, :frame_channels]
    t_minus_2 = feature[:, frame_channels:2 * frame_channels]
    t_minus_6_half = feature[
        :, 3 * frame_channels:3 * frame_channels + frame_channels // 2
    ]
    return torch.cat([current, t_minus_2, t_minus_6_half], dim=1)

def print_memory_usage(stage_name, device_id=None):
    if not torch.cuda.is_available():
        print(f"[{stage_name} Memory] CUDA not available.")
        return

    if device_id is None:
        device_id = torch.cuda.current_device()
        
    # 获取当前已分配内存 (allocated)
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    # 获取当前缓存/预留内存 (reserved)
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    print(f"[{stage_name} Memory] Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")
# --- 内存打印辅助函数结束 ---

"""
LiDAR Distillation (backbone to backbone)
"""

class LabelDistill(nn.Module):
    
    def __init__(self,
                 backbone_conf,
                 head_conf,
                 lidar_conf=None,
                 lidar_ckpt_path=None,
                 is_train_depth=False,
                 temporal_kd_selection='legacy_half',
                 teacher_proposal_cfg=None,
                 structured_output=False,
                 ):
        super(LabelDistill, self).__init__()
        # Model builders may consume or mutate nested config dictionaries.
        # Own private copies so one experiment cannot alter another instance.
        backbone_conf = deepcopy(backbone_conf)
        head_conf = deepcopy(head_conf)
        lidar_conf = deepcopy(lidar_conf)
        teacher_proposal_cfg = deepcopy(teacher_proposal_cfg)

        self.backbone = BaseLSSFPN(**backbone_conf)
        self.head = KDHead(**head_conf)
        self.teacher_proposal_decoder = (
            TeacherProposalDecoder(**teacher_proposal_cfg)
            if teacher_proposal_cfg is not None else None
        )
        if structured_output and self.teacher_proposal_decoder is None:
            raise ValueError(
                'structured_output requires teacher_proposal_cfg so the '
                'teacher output contains a ProposalBatch')
        self.is_train_depth = is_train_depth
        self.structured_output = structured_output
        self.temporal_kd_selection = temporal_kd_selection
        self.temporal_frame_channels = backbone_conf['output_channels']

        distill_in_feature = head_conf['bev_neck_conf']['in_channels'][:2]
        self.distill_encoder_lidar = DistillAdaptor([x // 2 for x in distill_in_feature],
                                                    out_features=[128, 256],
                                                    stride=[1, 1]
                                                    )
        
        # mmdet3d 1.4 moved voxelization out of the detector.
        voxel_layer_cfg = lidar_conf.pop('voxel_layer', None)
        if voxel_layer_cfg is not None:
            self.voxelizer = VoxelizationByGridShape(**voxel_layer_cfg)
        else:
            self.voxelizer = None

        # build lidar detection model
        self.centerpoint = MODELS.build(lidar_conf)

        # load pretrained parameters for lidar detection model
        lidar_params = torch.load(lidar_ckpt_path, map_location='cpu')

        prefix = 'model.centerpoint.'
        load_keys = [k for k in lidar_params['state_dict'] if k.startswith(prefix)]
        if not load_keys:
            raise RuntimeError(
                f'No CenterPoint weights with prefix {prefix!r} in '
                f'{lidar_ckpt_path}')
        self.centerpoint.load_state_dict({k[len(prefix):]: lidar_params['state_dict'][k] for k in load_keys})
        self.centerpoint.eval()
        for param in self.centerpoint.parameters():
            param.requires_grad_(False)

    def train(self, mode=True):
        """Keep the frozen LiDAR teacher in eval mode during student training."""
        super().train(mode)
        self.centerpoint.eval()
        return self


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
                # mmdet3d 1.4: VoxelizationByGridShape.forward 接收单个 Tensor
                # 需要逐帧 voxelize 后手动拼接，并在 coors 第0列写入 batch_idx
                pts_list = [p for p in lidar_pts.squeeze(1)]
                voxels_list, coors_list, num_points_list = [], [], []
                for i, pts in enumerate(pts_list):
                    res = self.voxelizer(pts)
                    # res = (voxels, coors, num_points_per_voxel)
                    v, c, n = res
                    # 在 coors 前面插入 batch_idx 列
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
                # '''
                # lidar_feats_out 的类型：<class 'tuple'>
                # lidar_feats_out 的长度：2
                # 元素 0 的类型：<class 'torch.Tensor'>
                # 元素 0 的形状：torch.Size([4, 128, 128, 128])
                # 元素 1 的类型：<class 'torch.Tensor'>
                # 元素 1 的形状：torch.Size([4, 256, 64, 64])
                # '''
                neck_feats = self.centerpoint.pts_neck(lidar_feats)
                # '''
                # lidar_feats 的类型：<class 'tuple'>
                # lidar_feats 的长度：2
                # 元素 0 的类型：<class 'torch.Tensor'>
                # 元素 0 的形状：torch.Size([4, 128, 128, 128])
                # 元素 1 的类型：<class 'torch.Tensor'>
                # 元素 1 的形状：torch.Size([4, 256, 64, 64])
                # '''
                
                lidar_preds = self.centerpoint.pts_bbox_head(neck_feats)
                if self.teacher_proposal_decoder is None:
                    # Backward compatibility for experiment configs that have
                    # not yet made teacher proposal selection explicit.
                    lidar_pred_box = self.get_bboxes(lidar_preds)
                else:
                    lidar_pred_box = self.teacher_proposal_decoder(lidar_preds)
          
         
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            
           
            preds, backbone_out, neck_output = self.head(x) #backbone output list of [B, C, W, H]
            # Student decode remains available through get_bboxes() for
            # inference. Legacy experiments retain their previous training
            # output; the structured J4 path avoids unused training-time NMS.
            image_pred_box = (
                None if self.structured_output else self.get_bboxes(preds))
            
            # '''
            # backone_outs
            # [0] Shape: torch.Size([4, 450, 128, 128])
            # [1] Shape: torch.Size([4, 300, 64, 64])
            # [2] Shape: torch.Size([4, 600, 32, 32])
            # '''
            if self.temporal_kd_selection == 'current_t2_half_t6':
                first_level_kd = select_temporal_kd_channels(
                    backbone_out[0], self.temporal_frame_channels)
            elif self.temporal_kd_selection == 'legacy_half':
                c1 = backbone_out[0].shape[1] // 2
                first_level_kd = backbone_out[0][:, :c1]
            else:
                raise ValueError(
                    'Unknown temporal_kd_selection: '
                    f'{self.temporal_kd_selection!r}')

            c2 = backbone_out[1].shape[1] // 2

            # adaptor
            distill_feats_lidar = self.distill_encoder_lidar([
                first_level_kd,
                backbone_out[1][:, :c2],
            ])
                
            # '''
            # 经过适配器之后的用于作用lidar的特征：
            # 特征 0: torch.Size([4, 128, 128, 128])
            # 特征 1: torch.Size([4, 256, 64, 64])
            # '''
            
            if self.structured_output:
                return LabelDistillOutput(
                    student=StudentOutput(
                        raw_preds=preds,
                        depth=depth_pred,
                        distill_features=distill_feats_lidar,
                        neck_features=neck_output,
                        decoded_boxes=image_pred_box,
                    ),
                    teacher=TeacherOutput(
                        raw_preds=lidar_preds,
                        backbone_features=lidar_feats_out,
                        neck_features=neck_feats,
                        proposals=lidar_pred_box,
                    ),
                )

            return preds, lidar_preds, depth_pred, distill_feats_lidar, lidar_feats_out, neck_feats, neck_output, lidar_pred_box, image_pred_box
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds, _, x = self.head(x)
            return preds
        

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def response_loss(self, targets, preds_dicts, teacher_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.response_loss(targets, preds_dicts, teacher_dicts)
    
    def distill_loss(self, targets, preds_dicts, teacher_dicts):

        return self.head.distill_loss(targets, preds_dicts, teacher_dicts)


    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)
