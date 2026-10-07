"""Identity-based student parameter groups for pretrained image encoders."""

import torch


class ImageOptimizerGroups:
    """Keep image initialization policy separate from the BEVDepth container."""

    @staticmethod
    def build(model, *, lr, image_lr_mult, weight_decay, image_weight_decay):
        image = model.backbone.img_backbone
        image_ids = {id(parameter) for parameter in image.parameters()}
        teacher_ids = {id(parameter) for parameter in model.centerpoint.parameters()}
        if image_ids & teacher_ids:
            raise ValueError("Image encoder and frozen teacher cannot share parameters")

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
            if group_name not in grouped:
                grouped[group_name] = dict(
                    params=[], param_names=[], group_name=group_name,
                    lr=lr * image_lr_mult if is_image else lr,
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
