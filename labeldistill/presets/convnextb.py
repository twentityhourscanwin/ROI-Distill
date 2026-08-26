"""ConvNeXt-B camera preset controlled by the experiment schema."""

from copy import deepcopy

from labeldistill.exps.nuscenes import base_exp

# Importing the wrapper registers ``TorchvisionConvNeXt`` with mmdet.
import labeldistill.layers.backbones.convnext_backbone  # noqa: F401


def build_convnextb_student_base(config):
    """Return topology and data-view settings for the ConvNeXt-B student."""
    image = config.student.image
    source_height, source_width = map(int, image.source_size)
    final_dim = tuple(map(int, image.final_size))

    backbone = deepcopy(base_exp.backbone_conf)
    backbone.update(
        final_dim=final_dim,
        downsample_factor=16,
        img_backbone_conf=dict(
            type='TorchvisionConvNeXt',
            arch='base',
            out_indices=[1, 2, 3],
            with_cp=bool(image.gradient_checkpointing),
            # A resumed/evaluated checkpoint will immediately replace these
            # weights. Avoid a redundant network download in that path.
            pretrained=(
                bool(image.pretrained)
                and config.runtime.resume_from is None
            ),
            frozen_stages=-1,
            drop_path_rate=float(image.drop_path_rate),
        ),
        img_neck_conf=dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024],
            upsample_strides=[0.5, 1, 2],
            out_channels=[128, 128, 128],
        ),
        depth_net_conf=dict(in_channels=384, mid_channels=384),
    )
    head = deepcopy(base_exp.head_conf)
    ida_aug = deepcopy(base_exp.ida_aug_conf)
    ida_aug.update(
        resize_lim=tuple(map(float, image.resize_limit)),
        final_dim=final_dim,
        H=source_height,
        W=source_width,
    )
    return (
        backbone,
        head,
        ida_aug,
        deepcopy(base_exp.bda_aug_conf),
        deepcopy(base_exp.img_conf),
    )
