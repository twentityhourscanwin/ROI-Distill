"""Camera detector without teacher models or distillation adaptors."""
from copy import deepcopy

from torch import nn

from labeldistill.layers.backbones.base_lss_fpn import BaseLSSFPN
from labeldistill.layers.heads.kd_head import KDHead
from labeldistill.models.distill_outputs import StudentOutput


class PureStudent(nn.Module):
    """Keep the student detection and depth paths independent of a teacher."""

    def __init__(self, backbone_conf, head_conf):
        super().__init__()
        self.backbone = BaseLSSFPN(**deepcopy(backbone_conf))
        self.head = KDHead(**deepcopy(head_conf))

    def forward(self, x, mats_dict, timestamps=None):
        if self.training:
            features, depth = self.backbone(
                x, mats_dict, timestamps, is_return_depth=True)
            predictions, _, neck = self.head(features)
            return StudentOutput(
                raw_preds=predictions, depth=depth,
                distill_features=(), neck_features=neck)
        features = self.backbone(x, mats_dict, timestamps)
        predictions, _, _ = self.head(features)
        return predictions

    def get_targets(self, gt_boxes, gt_labels):
        # The shared head also returns GT-only ROI helpers; detection uses
        # the same heatmaps/boxes/indices/masks as the distillation model.
        return self.head.get_targets(gt_boxes, gt_labels)[-1]

    def loss(self, targets, predictions):
        return self.head.loss(targets, predictions)

    def get_bboxes(self, predictions, img_metas=None, img=None, rescale=False):
        return self.head.get_bboxes(predictions, img_metas, img, rescale)
