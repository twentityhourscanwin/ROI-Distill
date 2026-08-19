from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import List

import torch
import torch.nn as nn

from labeldistill.models.distill_outputs import ProposalBatch


class MatchQuality(IntEnum):
    """Quality assigned to one GT after matching teacher proposals."""

    UNMATCHED = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(frozen=True)
class MatchResult(Mapping):
    """Batch match result with every tensor aligned to the original GT order.

    Each list contains one tensor per batch item.  For a batch item with N
    valid GT boxes, every field has N rows/elements.  Unmatched GT entries use
    a zero proposal, score 0, label -1 and infinite center distance.

    The Mapping interface exposes the old split-list keys so downstream mask
    generators remain compatible while callers migrate to the aligned fields.
    """

    gt_boxes: List[torch.Tensor]
    quality: List[torch.Tensor]
    matched_rois: List[torch.Tensor]
    roi_scores: List[torch.Tensor]
    roi_labels: List[torch.Tensor]
    center_distances: List[torch.Tensor]

    _LEGACY_KEYS = (
        'refined_high_quality_rois',
        'refined_high_quality_roi_scores',
        'refined_high_quality_roi_labels',
        'refined_high_quality_gt',
        'medium_quality_rois',
        'medium_quality_roi_scores',
        'medium_quality_roi_labels',
        'medium_quality_gt',
        'unmatched_gt',
    )

    def __post_init__(self):
        batch_size = len(self.gt_boxes)
        fields = (
            self.quality,
            self.matched_rois,
            self.roi_scores,
            self.roi_labels,
            self.center_distances,
        )
        if any(len(field) != batch_size for field in fields):
            raise ValueError('All MatchResult fields must have equal batch size')
        for idx in range(batch_size):
            num_gt = len(self.gt_boxes[idx])
            if any(len(field[idx]) != num_gt for field in fields):
                raise ValueError(
                    f'MatchResult batch {idx} is not aligned to its GT count')

    def with_gt_boxes(self, gt_boxes: List[torch.Tensor]):
        """Return a copy containing transformed GT boxes."""
        return replace(self, gt_boxes=gt_boxes)

    def _select(self, values, quality):
        return [
            value[cur_quality == int(quality)]
            for value, cur_quality in zip(values, self.quality)
        ]

    def __getitem__(self, key):
        high = MatchQuality.HIGH
        medium = MatchQuality.MEDIUM
        if key == 'refined_high_quality_rois':
            return self._select(self.matched_rois, high)
        if key == 'refined_high_quality_roi_scores':
            return self._select(self.roi_scores, high)
        if key == 'refined_high_quality_roi_labels':
            return self._select(self.roi_labels, high)
        if key == 'refined_high_quality_gt':
            return self._select(self.gt_boxes, high)
        if key == 'medium_quality_rois':
            return self._select(self.matched_rois, medium)
        if key == 'medium_quality_roi_scores':
            return self._select(self.roi_scores, medium)
        if key == 'medium_quality_roi_labels':
            return self._select(self.roi_labels, medium)
        if key == 'medium_quality_gt':
            return self._select(self.gt_boxes, medium)
        if key == 'unmatched_gt':
            return self._select(self.gt_boxes, MatchQuality.UNMATCHED)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._LEGACY_KEYS)

    def __len__(self) -> int:
        return len(self._LEGACY_KEYS)

    def to_legacy_dict(self):
        """Materialize the pre-refactor dictionary representation."""
        return {key: self[key] for key in self._LEGACY_KEYS}


class ProposalTargetLayer(nn.Module):
    """Match each GT to the best teacher proposal from its CenterPoint task."""

    def __init__(self, roi_sampler_cfg=None, structured=False):
        super().__init__()
        # Kept only so old experiment configs can still construct this module.
        self.roi_sampler_cfg = roi_sampler_cfg
        self.structured = structured

        self.class_groups = [
            {'class_ids': [0]},
            {'class_ids': [1, 2]},
            {'class_ids': [3, 4]},
            {'class_ids': [5]},
            {'class_ids': [6, 7]},
            {'class_ids': [8, 9]},
        ]
        self.class_to_group = {}
        for group_idx, group in enumerate(self.class_groups):
            for class_id in group['class_ids']:
                self.class_to_group[class_id] = group_idx

        self.distance_thresholds = {
            0: {'high': 2.0, 'medium': 4.0},
            1: {'high': 2.5, 'medium': 4.0},
            2: {'high': 2.5, 'medium': 4.0},
            3: {'high': 3.5, 'medium': 4.0},
            4: {'high': 2.5, 'medium': 4.0},
            5: {'high': 0.5, 'medium': 2.5},
            6: {'high': 1.0, 'medium': 2.5},
            7: {'high': 1.0, 'medium': 2.5},
            8: {'high': 0.5, 'medium': 2.0},
            9: {'high': 0.5, 'medium': 2.0},
        }
        self.register_buffer(
            '_class_group_lookup',
            torch.tensor([
                self.class_to_group[class_id] for class_id in range(10)
            ], dtype=torch.long),
            persistent=False)
        self.register_buffer(
            '_high_threshold_lookup',
            torch.tensor([
                self.distance_thresholds[class_id]['high']
                for class_id in range(10)
            ]),
            persistent=False)
        self.register_buffer(
            '_medium_threshold_lookup',
            torch.tensor([
                self.distance_thresholds[class_id]['medium']
                for class_id in range(10)
            ]),
            persistent=False)

    def get_distance_thresholds_for_class(self, class_id):
        thresholds = self.distance_thresholds.get(
            int(class_id), {'high': 2.0, 'medium': 4.0})
        return thresholds['high'], thresholds['medium']

    def same_task_group(self, class_id1, class_id2):
        group1 = self.class_to_group.get(int(class_id1), -1)
        group2 = self.class_to_group.get(int(class_id2), -1)
        return group1 == group2 and group1 != -1

    @staticmethod
    def compute_bev_center_distance(roi_centers, gt_centers):
        return torch.cdist(roi_centers[:, :2], gt_centers[:, :2], p=2)

    def _match_single(self, cur_roi, cur_scores, cur_labels, cur_gt):
        num_gt = len(cur_gt)
        roi_dim = cur_roi.shape[-1]
        device = cur_gt.device

        quality = torch.full(
            (num_gt,), int(MatchQuality.UNMATCHED),
            dtype=torch.long, device=device)
        matched_rois = cur_roi.new_zeros((num_gt, roi_dim))
        matched_scores = cur_scores.new_zeros((num_gt,))
        matched_labels = cur_labels.new_full((num_gt,), -1)
        matched_distances = cur_gt.new_full((num_gt,), float('inf'))

        if num_gt == 0 or len(cur_roi) == 0:
            return (quality, matched_rois, matched_scores, matched_labels,
                    matched_distances)

        center_distances = self.compute_bev_center_distance(
            cur_roi[:, :3], cur_gt[:, :3])

        # Keep class/task lookup, thresholds and matching masks on GPU.  The
        # GT loop now controls only ragged candidate selection and performs no
        # per-box .item() synchronization.
        class_groups = self._class_group_lookup.to(cur_labels.device)
        valid_roi_class = (cur_labels >= 0) & (cur_labels < len(class_groups))
        roi_groups = cur_labels.new_full(cur_labels.shape, -1)
        roi_groups[valid_roi_class] = class_groups[cur_labels[valid_roi_class]]

        gt_classes = cur_gt[:, -1].long()
        valid_gt_class = (gt_classes >= 0) & (gt_classes < len(class_groups))
        gt_groups = cur_labels.new_full((num_gt,), -1)
        gt_groups[valid_gt_class] = class_groups[gt_classes[valid_gt_class]]
        high_thresholds = cur_gt.new_full((num_gt,), 2.0)
        medium_thresholds = cur_gt.new_full((num_gt,), 4.0)
        high_lookup = self._high_threshold_lookup.to(cur_gt.device)
        medium_lookup = self._medium_threshold_lookup.to(cur_gt.device)
        high_thresholds[valid_gt_class] = high_lookup[
            gt_classes[valid_gt_class]]
        medium_thresholds[valid_gt_class] = medium_lookup[
            gt_classes[valid_gt_class]]

        for gt_idx in range(num_gt):
            high_thresh = high_thresholds[gt_idx]
            medium_thresh = medium_thresholds[gt_idx]
            gt_group = gt_groups[gt_idx]
            same_group = (roi_groups == gt_group) & (gt_group >= 0)
            distances = center_distances[:, gt_idx]

            high_indices = torch.nonzero(
                (distances <= high_thresh) & same_group,
                as_tuple=False).flatten()
            medium_indices = torch.nonzero(
                (distances > high_thresh)
                & (distances <= medium_thresh)
                & same_group,
                as_tuple=False).flatten()

            if high_indices.numel() > 0:
                candidates = high_indices
                match_quality = MatchQuality.HIGH
            elif medium_indices.numel() > 0:
                candidates = medium_indices
                match_quality = MatchQuality.MEDIUM
            else:
                continue

            best = candidates[cur_scores[candidates].argmax()]
            quality[gt_idx] = int(match_quality)
            matched_rois[gt_idx] = cur_roi[best]
            matched_scores[gt_idx] = cur_scores[best]
            matched_labels[gt_idx] = cur_labels[best]
            matched_distances[gt_idx] = distances[best]

        return (quality, matched_rois, matched_scores, matched_labels,
                matched_distances)

    def _forward_ragged(self, proposals, gt_boxes, gt_labels):
        if gt_boxes is None or gt_labels is None:
            raise ValueError(
                'gt_boxes and gt_labels are required with ProposalBatch')
        if len(proposals) != len(gt_boxes) or len(proposals) != len(gt_labels):
            raise ValueError(
                'ProposalBatch and GT lists must have equal batch size')

        gt_list = []
        quality_list = []
        matched_rois_list = []
        matched_scores_list = []
        matched_labels_list = []
        matched_distances_list = []

        for cur_roi, cur_scores, cur_labels, cur_gt, cur_gt_labels in zip(
                proposals.boxes,
                proposals.scores,
                proposals.labels,
                gt_boxes,
                gt_labels):
            cur_roi = cur_roi.float()
            cur_scores = cur_scores.float()
            cur_labels = cur_labels.long()
            cur_gt = cur_gt.float()
            cur_gt_labels = cur_gt_labels.long()

            if cur_gt.ndim != 2 or cur_gt.shape[-1] != 9:
                raise ValueError(
                    'Expected GT boxes with shape [N, 9], got '
                    f'{tuple(cur_gt.shape)}')
            if len(cur_gt) != len(cur_gt_labels):
                raise ValueError('GT boxes and labels must have equal length')

            cur_gt_with_cls = torch.cat([
                cur_gt,
                cur_gt_labels.to(dtype=cur_gt.dtype).unsqueeze(-1),
            ], dim=-1)
            result = self._match_single(
                cur_roi, cur_scores, cur_labels, cur_gt_with_cls)
            gt_list.append(cur_gt_with_cls)
            quality_list.append(result[0])
            matched_rois_list.append(result[1])
            matched_scores_list.append(result[2])
            matched_labels_list.append(result[3])
            matched_distances_list.append(result[4])

        matched = MatchResult(
            gt_boxes=gt_list,
            quality=quality_list,
            matched_rois=matched_rois_list,
            roi_scores=matched_scores_list,
            roi_labels=matched_labels_list,
            center_distances=matched_distances_list,
        )
        return matched if self.structured else matched.to_legacy_dict()

    def forward(self, batch_dict, gt_boxes=None, gt_labels=None):
        if isinstance(batch_dict, ProposalBatch):
            return self._forward_ragged(batch_dict, gt_boxes, gt_labels)

        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes_and_cls']

        gt_list = []
        quality_list = []
        matched_rois_list = []
        matched_scores_list = []
        matched_labels_list = []
        matched_distances_list = []

        for idx in range(batch_size):
            valid_gt = gt_boxes[idx].sum(dim=-1) != 0
            valid_roi = rois[idx].sum(dim=-1) != 0
            cur_gt = gt_boxes[idx][valid_gt]
            cur_roi = rois[idx][valid_roi]
            cur_scores = roi_scores[idx][valid_roi]
            cur_labels = roi_labels[idx][valid_roi]

            result = self._match_single(
                cur_roi, cur_scores, cur_labels, cur_gt)
            gt_list.append(cur_gt)
            quality_list.append(result[0])
            matched_rois_list.append(result[1])
            matched_scores_list.append(result[2])
            matched_labels_list.append(result[3])
            matched_distances_list.append(result[4])

        matched = MatchResult(
            gt_boxes=gt_list,
            quality=quality_list,
            matched_rois=matched_rois_list,
            roi_scores=matched_scores_list,
            roi_labels=matched_labels_list,
            center_distances=matched_distances_list,
        )
        return matched if self.structured else matched.to_legacy_dict()
