from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import List, Optional

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
    matched_proposal_indices: Optional[List[torch.Tensor]] = None
    matched_mask: Optional[List[torch.Tensor]] = None
    effective_gt_mask: Optional[List[torch.Tensor]] = None
    teacher_values: Optional[List[torch.Tensor]] = None
    trust_radii: Optional[List[torch.Tensor]] = None

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
        optional_fields = (
            self.matched_proposal_indices,
            self.matched_mask,
            self.effective_gt_mask,
            self.teacher_values,
            self.trust_radii,
        )
        if any(len(field) != batch_size for field in fields):
            raise ValueError('All MatchResult fields must have equal batch size')
        if any(
                field is not None and len(field) != batch_size
                for field in optional_fields):
            raise ValueError(
                'All populated MatchResult fields must have equal batch size')
        for idx in range(batch_size):
            num_gt = len(self.gt_boxes[idx])
            if any(len(field[idx]) != num_gt for field in fields):
                raise ValueError(
                    f'MatchResult batch {idx} is not aligned to its GT count')
            if any(
                    field is not None and len(field[idx]) != num_gt
                    for field in optional_fields):
                raise ValueError(
                    f'MatchResult batch {idx} has an optional field that is '
                    'not aligned to its GT count')

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

    DEFAULT_CLASS_NAMES = (
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
    )
    DEFAULT_CLASS_GROUPS = (
        ('car',),
        ('truck', 'construction_vehicle'),
        ('bus', 'trailer'),
        ('barrier',),
        ('motorcycle', 'bicycle'),
        ('pedestrian', 'traffic_cone'),
    )
    DEFAULT_DISTANCE_THRESHOLDS = {
        'car': {'high': 2.0, 'medium': 4.0},
        'truck': {'high': 2.5, 'medium': 4.0},
        'construction_vehicle': {'high': 2.5, 'medium': 4.0},
        'bus': {'high': 3.5, 'medium': 4.0},
        'trailer': {'high': 2.5, 'medium': 4.0},
        'barrier': {'high': 0.5, 'medium': 2.5},
        'motorcycle': {'high': 1.0, 'medium': 2.5},
        'bicycle': {'high': 1.0, 'medium': 2.5},
        'pedestrian': {'high': 0.5, 'medium': 2.0},
        'traffic_cone': {'high': 0.5, 'medium': 2.0},
    }

    def __init__(self, roi_sampler_cfg=None, structured=False, *,
                 class_names=None, class_groups=None,
                 distance_thresholds=None, class_policy='same_task_group',
                 selection='highest_score', one_to_one=False,
                 matching_type='center_distance', strict_less_than=False,
                 trust_radius=None, small_classes=None, large_classes=None,
                 official_class_ranges=None, point_cloud_range=None,
                 value_type='normalized_squared_margin'):
        super().__init__()
        # Kept only so old experiment configs can still construct this module.
        self.roi_sampler_cfg = roi_sampler_cfg
        self.structured = structured
        if matching_type not in {
                'center_distance', 'scale_conditioned_center_distance'}:
            raise ValueError(f'Unsupported matching_type={matching_type!r}')
        if matching_type == 'center_distance':
            if class_policy != 'same_task_group':
                raise ValueError(
                    f'Unsupported class_policy={class_policy!r} for legacy '
                    'center_distance matching')
            if selection != 'highest_score':
                raise ValueError(
                    f'Unsupported selection={selection!r} for legacy '
                    'center_distance matching')
            if one_to_one:
                raise ValueError(
                    'one_to_one is only supported by scale-conditioned matching')
        else:
            if class_policy != 'exact_class':
                raise ValueError(
                    'scale-conditioned matching requires exact_class policy')
            if selection not in {
                    'score_first_nearest_unmatched_gt', 'gt_nearest'}:
                raise ValueError(
                    f'Unsupported selection={selection!r} for '
                    'scale-conditioned matching')
            if (selection == 'score_first_nearest_unmatched_gt'
                    and not one_to_one):
                raise ValueError(
                    'score-first nearest unmatched GT selection requires '
                    'one_to_one')
            if selection == 'gt_nearest' and one_to_one:
                raise ValueError(
                    'gt_nearest selection requires proposal reuse '
                    '(one_to_one=False)')
            if not strict_less_than:
                raise ValueError(
                    'scale-conditioned matching requires strict less-than '
                    'distance gates')
        self.matching_type = matching_type
        self.class_policy = class_policy
        self.selection = selection
        self.one_to_one = one_to_one
        self.strict_less_than = strict_less_than
        if value_type not in {
                'legacy_discrete', 'normalized_squared_margin', 'uniform_gt'}:
            raise ValueError(f'Unsupported value_type={value_type!r}')
        if (matching_type != 'scale_conditioned_center_distance'
                and value_type == 'uniform_gt'):
            raise ValueError(
                'uniform_gt is only supported by scale-conditioned matching')
        self.value_type = value_type

        class_names = tuple(class_names or self.DEFAULT_CLASS_NAMES)
        class_groups = tuple(class_groups or self.DEFAULT_CLASS_GROUPS)
        distance_thresholds = distance_thresholds or self.DEFAULT_DISTANCE_THRESHOLDS
        name_to_id = {name: idx for idx, name in enumerate(class_names)}
        if len(name_to_id) != len(class_names):
            raise ValueError('class_names must be unique')
        flattened_groups = [name for group in class_groups for name in group]
        if set(flattened_groups) != set(class_names) or len(flattened_groups) != len(class_names):
            raise ValueError('class_groups must cover class_names exactly once')
        if (matching_type == 'center_distance'
                and set(distance_thresholds) != set(class_names)):
            raise ValueError('distance_thresholds must cover class_names exactly once')

        self.class_names = class_names
        self.class_groups = [
            {'class_ids': [name_to_id[name] for name in group]}
            for group in class_groups
        ]
        self.class_to_group = {}
        for group_idx, group in enumerate(self.class_groups):
            for class_id in group['class_ids']:
                self.class_to_group[class_id] = group_idx

        self.distance_thresholds = {
            name_to_id[name]: {
                'high': float(values['high']),
                'medium': float(values['medium']),
            }
            for name, values in distance_thresholds.items()
            if name in name_to_id
        }
        self.register_buffer(
            '_class_group_lookup',
            torch.tensor([
                self.class_to_group[class_id] for class_id in range(len(class_names))
            ], dtype=torch.long),
            persistent=False)
        self.register_buffer(
            '_high_threshold_lookup',
            torch.tensor([
                self.distance_thresholds[class_id]['high']
                for class_id in range(len(class_names))
            ]),
            persistent=False)
        self.register_buffer(
            '_medium_threshold_lookup',
            torch.tensor([
                self.distance_thresholds[class_id]['medium']
                for class_id in range(len(class_names))
            ]),
            persistent=False)

        if matching_type == 'scale_conditioned_center_distance':
            trust_radius = trust_radius or {}
            small_classes = tuple(small_classes or ())
            large_classes = tuple(large_classes or ())
            grouped = small_classes + large_classes
            if len(grouped) != len(set(grouped)) or set(grouped) != set(class_names):
                raise ValueError(
                    'small_classes and large_classes must partition class_names')
            if set(trust_radius) != {'small', 'large'}:
                raise ValueError('trust_radius must contain small and large')
            if set(official_class_ranges or {}) != set(class_names):
                raise ValueError(
                    'official_class_ranges must cover class_names exactly once')
            radii = [
                float(trust_radius[
                    'small' if name in small_classes else 'large'])
                for name in class_names
            ]
            official_ranges = [
                float(official_class_ranges[name]) for name in class_names
            ]
            if any(value <= 0 for value in radii + official_ranges):
                raise ValueError('trust radii and official ranges must be positive')
            point_cloud_range = list(point_cloud_range or [])
            if len(point_cloud_range) != 6:
                raise ValueError('point_cloud_range must contain six values')
        else:
            radii = [0.0] * len(class_names)
            official_ranges = [float('inf')] * len(class_names)
            point_cloud_range = list(
                point_cloud_range
                or [-float('inf'), -float('inf'), -float('inf'),
                    float('inf'), float('inf'), float('inf')])
        self.register_buffer(
            '_trust_radius_lookup', torch.tensor(radii, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            '_official_range_lookup',
            torch.tensor(official_ranges, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            '_point_cloud_range',
            torch.tensor(point_cloud_range, dtype=torch.float32),
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

    @staticmethod
    def _extract_bda_scale(bda_mat, reference):
        """Return the uniform BEV scale encoded by rotation/flip/scale BDA."""
        if bda_mat is None:
            return reference.new_tensor(1.0)
        linear = bda_mat[:2, :2].to(
            device=reference.device, dtype=reference.dtype)
        scale = torch.linalg.vector_norm(linear[:, 0])
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError('BDA matrix contains an invalid BEV scale')
        return scale

    def _effective_gt_mask(self, cur_gt, bda_scale):
        """Select canonical-range GTs that can be represented on the BEV map."""
        num_gt = len(cur_gt)
        if num_gt == 0:
            return torch.zeros(0, dtype=torch.bool, device=cur_gt.device)
        gt_classes = cur_gt[:, -1].long()
        valid_class = (gt_classes >= 0) & (gt_classes < len(self.class_names))
        official_ranges = cur_gt.new_zeros((num_gt,))
        range_lookup = self._official_range_lookup.to(cur_gt.device)
        official_ranges[valid_class] = range_lookup[gt_classes[valid_class]]
        canonical_radius = torch.linalg.vector_norm(
            cur_gt[:, :2], dim=1) / bda_scale
        # Official range is inclusive. Undoing BDA scale can introduce a few
        # ulps at exact 30/40/50m boundaries, so tolerate only numeric noise;
        # the learned trust-radius gate remains strictly ``distance < tau``.
        within_official_range = canonical_radius <= official_ranges + 1e-4

        pc_range = self._point_cloud_range.to(
            device=cur_gt.device, dtype=cur_gt.dtype)
        finite_geometry = torch.isfinite(
            cur_gt[:, [0, 1, 3, 4]]).all(dim=1)
        positive_size = (cur_gt[:, 3] > 0) & (cur_gt[:, 4] > 0)
        drawable = finite_geometry & positive_size & (
            (cur_gt[:, 0] >= pc_range[0])
            & (cur_gt[:, 0] < pc_range[3])
            & (cur_gt[:, 1] >= pc_range[1])
            & (cur_gt[:, 1] < pc_range[4])
        )
        return valid_class & within_official_range & drawable

    def _match_single_scale_conditioned(
            self, cur_roi, cur_scores, cur_labels, cur_gt, bda_mat=None):
        """Exact-class matching with class-specific center-distance gates."""
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
        matched_indices = torch.full(
            (num_gt,), -1, dtype=torch.long, device=device)
        matched_mask = torch.zeros(num_gt, dtype=torch.bool, device=device)
        teacher_values = cur_gt.new_zeros((num_gt,))
        bda_scale = self._extract_bda_scale(bda_mat, cur_gt)
        effective_gt = self._effective_gt_mask(cur_gt, bda_scale)

        gt_classes = cur_gt[:, -1].long()
        trust_radii = cur_gt.new_zeros((num_gt,))
        valid_class = (gt_classes >= 0) & (gt_classes < len(self.class_names))
        radius_lookup = self._trust_radius_lookup.to(cur_gt.device)
        trust_radii[valid_class] = radius_lookup[gt_classes[valid_class]]

        if num_gt == 0 or len(cur_roi) == 0 or not effective_gt.any():
            return (
                quality, matched_rois, matched_scores, matched_labels,
                matched_distances, matched_indices, matched_mask, effective_gt,
                teacher_values, trust_radii,
            )

        center_distances = self.compute_bev_center_distance(
            cur_roi[:, :3], cur_gt[:, :3])
        valid_roi_class = (
            (cur_labels >= 0) & (cur_labels < len(self.class_names)))

        for class_id in range(len(self.class_names)):
            gt_indices = torch.nonzero(
                effective_gt & (gt_classes == class_id),
                as_tuple=False).flatten()
            pred_indices = torch.nonzero(
                valid_roi_class & (cur_labels == class_id),
                as_tuple=False).flatten()
            if gt_indices.numel() == 0 or pred_indices.numel() == 0:
                continue

            class_radius = trust_radii[gt_indices[0]]
            augmented_class_radius = class_radius * bda_scale

            def assign(gt_idx, pred_idx, distance):
                quality[gt_idx] = int(MatchQuality.HIGH)
                matched_rois[gt_idx] = cur_roi[pred_idx]
                matched_scores[gt_idx] = cur_scores[pred_idx]
                matched_labels[gt_idx] = cur_labels[pred_idx]
                canonical_distance = distance / bda_scale
                matched_distances[gt_idx] = canonical_distance
                matched_indices[gt_idx] = pred_idx
                matched_mask[gt_idx] = True
                normalized_distance = canonical_distance / class_radius
                teacher_values[gt_idx] = torch.clamp(
                    1.0 - normalized_distance.square(), min=0.0)

            if self.selection == 'gt_nearest':
                # M1: every GT independently selects its closest same-class
                # proposal. Proposal indices are intentionally not consumed,
                # so one teacher prediction may supervise multiple nearby GTs.
                for gt_idx in gt_indices:
                    distances = center_distances[pred_indices, gt_idx]
                    within_radius = distances < augmented_class_radius
                    candidates = torch.nonzero(
                        within_radius, as_tuple=False).flatten()
                    if candidates.numel() == 0:
                        continue
                    candidate_distances = distances[candidates]
                    local_choice = candidates[candidate_distances.argmin()]
                    pred_idx = pred_indices[local_choice]
                    assign(gt_idx, pred_idx, distances[local_choice])
                continue

            # P0: stable sorting preserves decoder proposal order for equal
            # scores, then each proposal claims its nearest unmatched GT.
            order = torch.argsort(
                cur_scores[pred_indices], descending=True, stable=True)
            pred_indices = pred_indices[order]
            available_gt = torch.ones(
                gt_indices.numel(), dtype=torch.bool, device=device)
            for pred_idx in pred_indices:
                distances = center_distances[pred_idx, gt_indices]
                within_radius = distances < augmented_class_radius
                candidates = torch.nonzero(
                    available_gt & within_radius,
                    as_tuple=False).flatten()
                if candidates.numel() == 0:
                    continue
                candidate_distances = distances[candidates]
                local_choice = candidates[candidate_distances.argmin()]
                gt_idx = gt_indices[local_choice]
                available_gt[local_choice] = False
                assign(gt_idx, pred_idx, distances[local_choice])

        return (
            quality, matched_rois, matched_scores, matched_labels,
            matched_distances, matched_indices, matched_mask, effective_gt,
            teacher_values, trust_radii,
        )

    def _forward_uniform_gt(self, gt_boxes, gt_labels, bda_mats=None):
        """Build B0 values from effective GTs without consulting proposals."""
        if gt_boxes is None or gt_labels is None:
            raise ValueError('gt_boxes and gt_labels are required for uniform_gt')
        if len(gt_boxes) != len(gt_labels):
            raise ValueError('GT boxes and labels must have equal batch size')
        if bda_mats is not None and len(bda_mats) != len(gt_boxes):
            raise ValueError('BDA matrices and GTs must have equal batch size')

        fields = {
            'gt_boxes': [],
            'quality': [],
            'matched_rois': [],
            'roi_scores': [],
            'roi_labels': [],
            'center_distances': [],
            'matched_proposal_indices': [],
            'matched_mask': [],
            'effective_gt_mask': [],
            'teacher_values': [],
            'trust_radii': [],
        }
        radius_lookup = self._trust_radius_lookup
        for batch_idx, (cur_gt, cur_gt_labels) in enumerate(
                zip(gt_boxes, gt_labels)):
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
            bda_mat = None if bda_mats is None else bda_mats[batch_idx]
            bda_scale = self._extract_bda_scale(bda_mat, cur_gt_with_cls)
            effective = self._effective_gt_mask(cur_gt_with_cls, bda_scale)
            num_gt = len(cur_gt_with_cls)
            valid_class = (
                (cur_gt_labels >= 0)
                & (cur_gt_labels < len(self.class_names)))
            trust_radii = cur_gt.new_zeros((num_gt,))
            lookup = radius_lookup.to(cur_gt.device)
            trust_radii[valid_class] = lookup[cur_gt_labels[valid_class]]

            fields['gt_boxes'].append(cur_gt_with_cls)
            fields['quality'].append(torch.full(
                (num_gt,), int(MatchQuality.UNMATCHED),
                dtype=torch.long, device=cur_gt.device))
            fields['matched_rois'].append(cur_gt.new_zeros((num_gt, 9)))
            fields['roi_scores'].append(cur_gt.new_zeros((num_gt,)))
            fields['roi_labels'].append(torch.full(
                (num_gt,), -1, dtype=torch.long, device=cur_gt.device))
            fields['center_distances'].append(cur_gt.new_full(
                (num_gt,), float('inf')))
            fields['matched_proposal_indices'].append(torch.full(
                (num_gt,), -1, dtype=torch.long, device=cur_gt.device))
            fields['matched_mask'].append(torch.zeros(
                num_gt, dtype=torch.bool, device=cur_gt.device))
            fields['effective_gt_mask'].append(effective)
            fields['teacher_values'].append(effective.to(cur_gt.dtype))
            fields['trust_radii'].append(trust_radii)

        matched = MatchResult(**fields)
        return matched if self.structured else matched.to_legacy_dict()

    def _forward_ragged(self, proposals, gt_boxes, gt_labels, bda_mats=None):
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
        matched_indices_list = []
        matched_mask_list = []
        effective_gt_list = []
        teacher_values_list = []
        trust_radii_list = []

        if bda_mats is not None and len(bda_mats) != len(proposals):
            raise ValueError('BDA matrices and proposals must have equal batch size')

        for batch_idx, (
                cur_roi, cur_scores, cur_labels, cur_gt, cur_gt_labels) in enumerate(zip(
                proposals.boxes,
                proposals.scores,
                proposals.labels,
                gt_boxes,
                gt_labels)):
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
            if self.matching_type == 'scale_conditioned_center_distance':
                bda_mat = None if bda_mats is None else bda_mats[batch_idx]
                result = self._match_single_scale_conditioned(
                    cur_roi, cur_scores, cur_labels, cur_gt_with_cls, bda_mat)
                if self.value_type == 'uniform_gt':
                    # B0 keeps q=1 for every effective GT, while retaining the
                    # same teacher-match metadata used to gate response bbox KD.
                    result = (*result[:8], result[7].to(cur_gt.dtype), result[9])
            else:
                result = self._match_single(
                    cur_roi, cur_scores, cur_labels, cur_gt_with_cls)
            gt_list.append(cur_gt_with_cls)
            quality_list.append(result[0])
            matched_rois_list.append(result[1])
            matched_scores_list.append(result[2])
            matched_labels_list.append(result[3])
            matched_distances_list.append(result[4])
            if self.matching_type == 'scale_conditioned_center_distance':
                matched_indices_list.append(result[5])
                matched_mask_list.append(result[6])
                effective_gt_list.append(result[7])
                teacher_values_list.append(result[8])
                trust_radii_list.append(result[9])

        matched = MatchResult(
            gt_boxes=gt_list,
            quality=quality_list,
            matched_rois=matched_rois_list,
            roi_scores=matched_scores_list,
            roi_labels=matched_labels_list,
            center_distances=matched_distances_list,
            matched_proposal_indices=(
                matched_indices_list or None),
            matched_mask=matched_mask_list or None,
            effective_gt_mask=effective_gt_list or None,
            teacher_values=teacher_values_list or None,
            trust_radii=trust_radii_list or None,
        )
        return matched if self.structured else matched.to_legacy_dict()

    def forward(self, batch_dict, gt_boxes=None, gt_labels=None, bda_mats=None):
        if isinstance(batch_dict, ProposalBatch):
            return self._forward_ragged(
                batch_dict, gt_boxes, gt_labels, bda_mats=bda_mats)

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
