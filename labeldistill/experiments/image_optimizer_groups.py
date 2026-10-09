"""Identity-based student parameter groups for pretrained image encoders."""

import math
import re

import torch


class ImageOptimizerGroups:
    """Keep image initialization policy separate from the BEVDepth container."""

    @staticmethod
    def convnext_layer_id(name):
        """Map the ConvNeXt-B stages to the fixed layer-wise LR hierarchy."""
        match = re.fullmatch(r"stages\.([0-3])\.([01])\.(.+)", name)
        if match is None:
            raise ValueError(f"Unmapped ConvNeXt-B image parameter: {name}")
        stage, section = int(match[1]), int(match[2])
        if section == 0:
            return (0, 2, 3, 12)[stage]
        block_match = re.fullmatch(r"(\d+)\.(.+)", match[3])
        if block_match is None:
            raise ValueError(f"Invalid ConvNeXt-B block parameter: {name}")
        block = int(block_match[1])
        if block >= (3, 3, 27, 3)[stage]:
            raise ValueError(f"ConvNeXt-B block index out of range: {name}")
        return (1, 2, 3 + block // 3, 12)[stage]

    @staticmethod
    def _layer_ids(image):
        stages = getattr(image, "stages", None)
        if (
            not isinstance(stages, (torch.nn.ModuleList, torch.nn.Sequential))
            or len(stages) != 4
            or any(not isinstance(stage, torch.nn.Sequential) or len(stage) != 2
                   or not isinstance(stage[1], torch.nn.Sequential) for stage in stages)
            or tuple(len(stage[1]) for stage in stages) != (3, 3, 27, 3)
        ):
            raise ValueError("Layer decay requires ConvNeXt-B stages with depths (3, 3, 27, 3)")
        return {
            id(parameter): ImageOptimizerGroups.convnext_layer_id(name)
            for name, parameter in image.named_parameters()
        }

    @staticmethod
    def build(model, *, lr, image_lr_mult, weight_decay, image_weight_decay,
              layer_decay=1.0):
        if not math.isfinite(layer_decay) or not 0 < layer_decay <= 1:
            raise ValueError("layer_decay must be finite and in (0, 1]")
        image = model.backbone.img_backbone
        image_ids = {id(parameter) for parameter in image.parameters()}
        teacher_ids = {id(parameter) for parameter in model.centerpoint.parameters()}
        if image_ids & teacher_ids:
            raise ValueError("Image encoder and frozen teacher cannot share parameters")
        layer_ids = ImageOptimizerGroups._layer_ids(image) if layer_decay != 1.0 else {}

        no_decay_ids = set()
        for module in image.modules():
            for local_name, parameter in module.named_parameters(recurse=False):
                # Torchvision LayerScale stores one scale per channel as C x 1 x 1.
                if (
                    local_name == "bias"
                    or parameter.ndim <= 1
                    or local_name == "layer_scale"
                    or isinstance(module, torch.nn.LayerNorm)
                ):
                    no_decay_ids.add(id(parameter))

        grouped = {}
        seen = set()
        for name, parameter in model.named_parameters():
            parameter_id = id(parameter)
            if not parameter.requires_grad or parameter_id in teacher_ids:
                continue
            if parameter_id in seen:
                raise ValueError(f"Duplicate optimizer parameter: {name}")
            seen.add(parameter_id)
            is_image = parameter_id in image_ids
            zero_decay = is_image and parameter_id in no_decay_ids
            group_name = (
                "image_no_decay" if zero_decay else
                "image_decay" if is_image else "student_other"
            )
            learning_rate = lr * image_lr_mult if is_image else lr
            if is_image and layer_decay != 1.0:
                layer_id = layer_ids[parameter_id]
                suffix = "no_decay" if zero_decay else "decay"
                group_name = f"image_layer{layer_id}_{suffix}"
                learning_rate *= layer_decay ** (12 - layer_id)
            if group_name not in grouped:
                grouped[group_name] = dict(
                    params=[], param_names=[], group_name=group_name,
                    lr=learning_rate,
                    weight_decay=(
                        0.0 if zero_decay else
                        image_weight_decay if is_image else weight_decay
                    ),
                )
            grouped[group_name]["params"].append(parameter)
            grouped[group_name]["param_names"].append(name)

        expected = {
            id(parameter) for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in teacher_ids
        }
        if seen != expected:
            raise ValueError("Optimizer groups do not cover every student parameter")
        if not grouped:
            raise ValueError("Student has no trainable parameters")
        return list(grouped.values())

    @staticmethod
    def summarize(groups):
        return [
            dict(
                name=group["group_name"],
                parameter_tensors=len(group["params"]),
                parameter_elements=sum(p.numel() for p in group["params"]),
                peak_lr=group["lr"],
                weight_decay=group["weight_decay"],
            )
            for group in groups
        ]
