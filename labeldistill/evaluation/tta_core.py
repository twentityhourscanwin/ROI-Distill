"""确定性 TTA 视图、几何还原及与原生 KDHead 一致的框级融合。"""

from dataclasses import dataclass
import math
from typing import Mapping

import torch


@dataclass(frozen=True)
class View:
    name: str
    bev_rotation: float = 0.0
    bev_scale: float = 1.0
    flip_x: bool = False
    flip_y: bool = False
    image_flip: bool = False
    image_scale: float = 1.0

    def __post_init__(self):
        values = (self.bev_rotation, self.bev_scale, self.image_scale)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("View parameters must be finite")
        if self.bev_scale <= 0 or self.image_scale <= 0:
            raise ValueError("View scales must be positive")


VIEW_SPECS = {
    view.name: view for view in (
        View("identity"),
        View("bev_flip_x", flip_x=True),
        View("bev_flip_y", flip_y=True),
        View("bev_flip_xy", flip_x=True, flip_y=True),
        View("bev_rot_m12p5", bev_rotation=-12.5),
        View("bev_rot_p12p5", bev_rotation=12.5),
        View("bev_scale_095", bev_scale=0.95),
        View("bev_scale_105", bev_scale=1.05),
        View("image_flip", image_flip=True),
        View("image_scale_105", image_scale=1.05),
        View("image_scale_110", image_scale=1.1),
    )
}
GROUPS = {
    "B0": ("identity",),
    "F4": ("identity", "bev_flip_x", "bev_flip_y", "bev_flip_xy"),
    "R3": ("identity", "bev_rot_m12p5", "bev_rot_p12p5"),
    "S3": ("identity", "bev_scale_095", "bev_scale_105"),
    "I2": ("identity", "image_flip"),
    "Z3": ("identity", "image_scale_105", "image_scale_110"),
    "A11": tuple(VIEW_SPECS),
}


def bev_matrix(view: View, *, device=None, dtype=torch.float32):
    # 与训练的 float32 构造和 F @ (S @ R) 次序一致。
    angle = torch.tensor(view.bev_rotation / 180 * math.pi)
    sine, cosine = torch.sin(angle), torch.cos(angle)
    rotation = torch.tensor([
        [cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1],
    ])
    scale = torch.eye(3) * view.bev_scale
    reflection = torch.diag(torch.tensor([
        -1.0 if view.flip_x else 1.0,
        -1.0 if view.flip_y else 1.0,
        1.0,
    ]))
    matrix = torch.eye(4)
    matrix[:3, :3] = reflection @ (scale @ rotation)
    return matrix.to(device=device, dtype=dtype)


def inverse_boxes(boxes: torch.Tensor, view: View):
    """恢复增强系的中心、尺寸、朝向及速度；输入保持不变。"""
    if boxes.ndim != 2 or boxes.shape[1] != 9:
        raise ValueError("Expected boxes shaped (N, 9)")
    if not boxes.is_floating_point() or not torch.isfinite(boxes).all():
        raise ValueError("Boxes must contain finite floating-point values")
    restored = boxes.clone()
    if not (view.flip_x or view.flip_y or view.bev_rotation
            or view.bev_scale != 1.0):
        # 不经三角函数可保留原生 identity 框的逐位数值。
        return restored
    work = boxes.float()
    linear = bev_matrix(view, device=boxes.device)[:3, :3]
    inverse = linear.T / (view.bev_scale * view.bev_scale)
    restored[:, :3] = work[:, :3] @ inverse.T
    restored[:, 3:6] = work[:, 3:6] / view.bev_scale
    heading = torch.stack((work[:, 6].cos(), work[:, 6].sin()), dim=-1)
    heading = heading @ inverse[:2, :2].T
    restored[:, 6] = torch.atan2(heading[:, 1], heading[:, 0])
    restored[:, 7:9] = work[:, 7:9] @ inverse[:2, :2].T
    return restored


def decode_candidates(head, predictions):
    """返回 [batch][task] 的增强系候选，沿用原生 coder，尚未执行 NMS。"""
    if len(predictions) != len(head.num_classes):
        raise ValueError("Prediction task count differs from the head")
    decoded = []
    batch_size = None
    for task_id, task_predictions in enumerate(predictions):
        prediction = task_predictions[0]
        heatmap = prediction["heatmap"].sigmoid()
        if batch_size is None:
            batch_size = heatmap.shape[0]
        elif batch_size != heatmap.shape[0]:
            raise ValueError("Task batch sizes differ")
        dimensions = (prediction["dim"].exp() if head.norm_bbox
                      else prediction["dim"])
        decoded.append(head.bbox_coder.decode(
            heatmap,
            prediction["rot"][:, 0].unsqueeze(1),
            prediction["rot"][:, 1].unsqueeze(1),
            prediction["height"],
            dimensions,
            prediction.get("vel"),
            reg=prediction["reg"], task_id=task_id,
        ))
    if batch_size is None:
        raise ValueError("No prediction tasks")
    return [[task[index] for task in decoded] for index in range(batch_size)]


def merge_candidates(head, per_view_tasks: Mapping, img_meta=None):
    """合并一个样本；缓存候选留在增强系，只有这里执行逆变换。

    每个 task 沿用 KDHead 的 circle NMS。相同 task 内不同类别也可能相互
    抑制，这是原生推理的契约；这里不引入类别级 NMS 或 z 中心偏移。
    """
    from mmdet3d.models.dense_heads.centerpoint_head import circle_nms

    if head.test_cfg["nms_type"] != "circle":
        raise ValueError("TTA currently requires the native circle NMS")
    if not per_view_tasks:
        raise ValueError("At least one view is required")
    for name, tasks in per_view_tasks.items():
        if name not in VIEW_SPECS:
            raise ValueError(f"Unknown view: {name}")
        if len(tasks) != len(head.num_classes):
            raise ValueError("Candidate task count differs from the head")

    merged_boxes, merged_scores, merged_labels = [], [], []
    class_offset = 0
    for task_id, class_count in enumerate(head.num_classes):
        boxes_per_view, scores_per_view, labels_per_view = [], [], []
        for name, tasks in per_view_tasks.items():
            task = tasks[task_id]
            boxes = inverse_boxes(task["bboxes"], VIEW_SPECS[name])
            scores, labels = task["scores"], task["labels"]
            count = len(boxes)
            if scores.shape != (count,) or labels.shape != (count,):
                raise ValueError("Candidate scores/labels must match boxes")
            if not torch.isfinite(scores).all() or not torch.isfinite(labels).all():
                raise ValueError("Candidate scores/labels must be finite")
            if not ((labels >= 0) & (labels < class_count)
                    & (labels == labels.long())).all():
                raise ValueError("Candidate labels must be local task class IDs")
            boxes_per_view.append(boxes)
            scores_per_view.append(scores)
            labels_per_view.append(labels)
        boxes = torch.cat(boxes_per_view)
        scores = torch.cat(scores_per_view)
        labels = torch.cat(labels_per_view)
        centers_scores = torch.cat([boxes[:, :2], scores[:, None]], dim=1)
        keep = torch.tensor(
            circle_nms(
                centers_scores.detach().cpu().numpy(),
                head.test_cfg["min_radius"][task_id],
                post_max_size=head.test_cfg["post_max_size"],
            ),
            dtype=torch.long, device=boxes.device,
        )
        merged_boxes.append(boxes[keep])
        merged_scores.append(scores[keep])
        merged_labels.append((labels[keep] + class_offset).int())
        class_offset += class_count
    return [torch.cat(merged_boxes), torch.cat(merged_scores),
            torch.cat(merged_labels)]
