from copy import deepcopy

import torch
from torch import nn
from mmdet3d.models.dense_heads.centerpoint_head import circle_nms
from mmdet3d.registry import TASK_UTILS

from labeldistill.models.distill_outputs import ProposalBatch


class TeacherProposalDecoder(nn.Module):
    """Decode frozen CenterPoint outputs into teacher proposals.

    This module deliberately owns its proposal-selection configuration.  It
    must not inherit the student head's bbox coder, score threshold or NMS
    settings.  Circle NMS remains CPU-based here to preserve the baseline's
    numerical behavior; it can be replaced in a separately validated change.
    """

    def __init__(self, bbox_coder, num_classes, min_radius,
                 post_max_size=83, nms_type='circle', norm_bbox=True):
        super().__init__()
        if nms_type != 'circle':
            raise ValueError(
                'TeacherProposalDecoder currently supports circle NMS only, '
                f'got {nms_type!r}')
        if len(num_classes) != len(min_radius):
            raise ValueError(
                'num_classes and min_radius must contain one value per task')

        self.bbox_coder = TASK_UTILS.build(deepcopy(bbox_coder))
        self.num_classes = tuple(num_classes)
        self.min_radius = tuple(min_radius)
        self.post_max_size = int(post_max_size)
        self.nms_type = nms_type
        self.norm_bbox = norm_bbox

    def forward(self, preds_dicts):
        if len(preds_dicts) != len(self.num_classes):
            raise ValueError(
                f'Expected {len(self.num_classes)} teacher tasks, '
                f'got {len(preds_dicts)}')

        task_results = []
        for task_id, task_preds in enumerate(preds_dicts):
            pred = task_preds[0]
            heatmap = pred['heatmap'].sigmoid()
            dimensions = (
                torch.exp(pred['dim']) if self.norm_bbox else pred['dim'])
            rot_sine = pred['rot'][:, 0].unsqueeze(1)
            rot_cosine = pred['rot'][:, 1].unsqueeze(1)

            decoded = self.bbox_coder.decode(
                heatmap,
                rot_sine,
                rot_cosine,
                pred['height'],
                dimensions,
                pred.get('vel'),
                reg=pred['reg'],
                task_id=task_id,
            )

            current_task = []
            for sample in decoded:
                boxes = sample['bboxes']
                scores = sample['scores']
                labels = sample['labels']
                centers_with_scores = torch.cat(
                    [boxes[:, :2], scores[:, None]], dim=1)
                keep = circle_nms(
                    centers_with_scores.detach().cpu().numpy(),
                    self.min_radius[task_id],
                    post_max_size=self.post_max_size,
                )
                keep = torch.as_tensor(
                    keep, dtype=torch.long, device=boxes.device)
                current_task.append({
                    'bboxes': boxes[keep],
                    'scores': scores[keep],
                    'labels': labels[keep],
                })
            task_results.append(current_task)

        num_samples = len(task_results[0])
        merged_boxes = []
        merged_scores = []
        merged_labels = []
        label_offsets = []
        offset = 0
        for num_classes in self.num_classes:
            label_offsets.append(offset)
            offset += num_classes

        for sample_idx in range(num_samples):
            boxes = torch.cat([
                result[sample_idx]['bboxes'] for result in task_results
            ])
            scores = torch.cat([
                result[sample_idx]['scores'] for result in task_results
            ])
            labels = torch.cat([
                result[sample_idx]['labels'].int() + label_offsets[task_id]
                for task_id, result in enumerate(task_results)
            ])
            merged_boxes.append(boxes)
            merged_scores.append(scores)
            merged_labels.append(labels)
        return ProposalBatch(
            boxes=merged_boxes,
            scores=merged_scores,
            labels=merged_labels,
        )
