"""Raw-Gaussian union-mask reduction for GT-guided BEV feature KD."""

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.utils.gaussian import (
    draw_heatmap_gaussian,
    gaussian_radius,
)

from labeldistill.refine_head.target_assigner.roi_distill import MatchResult


@dataclass(frozen=True)
class RawGaussianFeatureLossOutput:
    loss: torch.Tensor
    level_losses: Tuple[torch.Tensor, ...]
    group_losses: Dict[str, torch.Tensor]
    effective_gt_count: torch.Tensor
    matched_gt_count: torch.Tensor
    teacher_value_sum: torch.Tensor
    skipped_gt_count: torch.Tensor


class RawGaussianUnionFeatureLoss(nn.Module):
    """Reduce BEV feature error with q-weighted raw Gaussian union masks.

    Every drawable GT produces a peak-one Gaussian at base resolution. Its
    teacher value ``q`` scales the complete mask, overlapping masks are merged,
    and every feature level is normalized by its final mask mass.
    """

    def __init__(self, *, point_cloud_range, feature_map_size,
                 gaussian_overlap=0.1, min_radius=2,
                 small_class_ids=(), mask_type='per_gt_gaussian',
                 overlap_merge='max', max_radius=None):
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError('point_cloud_range must contain six values')
        if len(feature_map_size) != 2:
            raise ValueError('feature_map_size must contain [width, height]')
        if gaussian_overlap <= 0 or min_radius < 0:
            raise ValueError(
                'gaussian_overlap must be positive and min_radius non-negative')
        if mask_type not in {
                'per_gt_gaussian', 'per_gt_elliptical_gaussian'}:
            raise ValueError(f'Unsupported mask_type={mask_type!r}')
        if overlap_merge not in {'max', 'sum'}:
            raise ValueError(f'Unsupported overlap_merge={overlap_merge!r}')
        if max_radius is not None:
            if (not isinstance(max_radius, int)
                    or max_radius < min_radius):
                raise ValueError(
                    'max_radius must be an integer >= min_radius')
            if mask_type != 'per_gt_gaussian':
                raise ValueError(
                    'max_radius is supported only for per_gt_gaussian')
        self.max_radius = max_radius
        self.register_buffer(
            'point_cloud_range', torch.tensor(point_cloud_range),
            persistent=False)
        self.feature_map_size = tuple(int(value) for value in feature_map_size)
        self.gaussian_overlap = float(gaussian_overlap)
        self.min_radius = int(min_radius)
        self.small_class_ids = frozenset(int(value) for value in small_class_ids)
        self.mask_type = mask_type
        self.overlap_merge = overlap_merge

    def _draw_base_mask(self, box, *, device):
        if self.mask_type == 'per_gt_elliptical_gaussian':
            return self._draw_elliptical_mask(box, device=device)
        return self._draw_circular_mask(box, device=device)

    def _box_geometry(self, box, *, device):
        feat_width, feat_height = self.feature_map_size
        pc_range = self.point_cloud_range.to(
            device=device, dtype=torch.float32)
        cell_x = (pc_range[3] - pc_range[0]) / feat_width
        cell_y = (pc_range[4] - pc_range[1]) / feat_height
        size_x = box[3].float() / cell_x
        size_y = box[4].float() / cell_y
        geometry_values = torch.stack([
            box[0].float(), box[1].float(), size_x, size_y,
        ])
        if not torch.isfinite(geometry_values).all():
            return None
        if size_x <= 0 or size_y <= 0:
            return None

        center = torch.stack([
            (box[0].float() - pc_range[0]) / cell_x,
            (box[1].float() - pc_range[1]) / cell_y,
        ]).to(torch.int32)
        if not (
                0 <= center[0] < feat_width
                and 0 <= center[1] < feat_height):
            return None
        return (
            feat_width, feat_height, size_x, size_y, center,
            float(cell_x), float(cell_y),
        )

    def _draw_circular_mask(self, box, *, device):
        geometry = self._box_geometry(box, device=device)
        if geometry is None:
            return None
        feat_width, feat_height, size_x, size_y, center, _, _ = geometry

        radius_tensor = gaussian_radius(
            (size_y, size_x), min_overlap=self.gaussian_overlap)
        radius = int(radius_tensor.item())
        if self.max_radius is not None:
            radius = min(self.max_radius, radius)
        radius = max(self.min_radius, radius)

        mask = torch.zeros(
            (feat_height, feat_width), device=device, dtype=torch.float32)
        return draw_heatmap_gaussian(mask, center, radius, k=1.0)

    def _draw_elliptical_mask(self, box, *, device):
        """Draw an oriented BEV ellipse with the circular mask's mass scale."""
        geometry = self._box_geometry(box, device=device)
        if geometry is None:
            return None
        (feat_width, feat_height, size_x, size_y, center,
         cell_x, cell_y) = geometry
        radius = max(
            float(self.min_radius),
            float(gaussian_radius(
                (size_y, size_x), min_overlap=self.gaussian_overlap).item()),
        )
        aspect = math.sqrt(max(float(size_x / size_y), 1e-6))
        radius_x_px = max(0.5, radius * aspect)
        radius_y_px = max(0.5, radius / aspect)
        radius_x_m = radius_x_px * cell_x
        radius_y_m = radius_y_px * cell_y

        # A rotated ellipse fits inside this conservative square window.
        window_px = int(math.ceil(max(
            radius_x_m / min(cell_x, cell_y),
            radius_y_m / min(cell_x, cell_y),
        )))
        center_x = int(center[0].item())
        center_y = int(center[1].item())
        x0, x1 = max(0, center_x - window_px), min(
            feat_width, center_x + window_px + 1)
        y0, y1 = max(0, center_y - window_px), min(
            feat_height, center_y + window_px + 1)
        if x0 >= x1 or y0 >= y1:
            return None

        grid_y, grid_x = torch.meshgrid(
            torch.arange(y0, y1, device=device, dtype=torch.float32),
            torch.arange(x0, x1, device=device, dtype=torch.float32),
            indexing='ij',
        )
        dx = (grid_x - float(center_x)) * cell_x
        dy = (grid_y - float(center_y)) * cell_y
        yaw = box[6].float()
        if not torch.isfinite(yaw):
            return None
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        normalized = (
            local_x.square() / max(radius_x_m ** 2, 1e-6)
            + local_y.square() / max(radius_y_m ** 2, 1e-6)
        )
        # Match CenterPoint's approximately three-sigma truncation at radius.
        patch = torch.exp(-4.5 * normalized) * (normalized <= 1.0)
        mask = torch.zeros(
            (feat_height, feat_width), device=device, dtype=torch.float32)
        mask[y0:y1, x0:x1] = patch
        return mask

    def _merge(self, current, contribution):
        if self.overlap_merge == 'max':
            return torch.maximum(current, contribution)
        return current + contribution

    @staticmethod
    def _global_sum(local_value):
        global_value = local_value.detach().clone()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(global_value, op=dist.ReduceOp.SUM)
        return global_value, world_size

    def forward(self, teacher_features: Sequence[torch.Tensor],
                student_features: Sequence[torch.Tensor],
                match_result: MatchResult):
        if len(teacher_features) != len(student_features):
            raise ValueError(
                f'Feature level mismatch: teacher={len(teacher_features)}, '
                f'student={len(student_features)}')
        if not teacher_features:
            raise ValueError('At least one feature level is required')
        if (
                match_result.teacher_values is None
                or match_result.matched_mask is None
                or match_result.effective_gt_mask is None):
            raise ValueError(
                'Raw-Gaussian feature loss requires continuous-value '
                'MatchResult fields')

        error_maps = []
        level_sums = []
        level_mask_masses = []
        for teacher_feature, student_feature in zip(
                teacher_features, student_features):
            if teacher_feature.shape != student_feature.shape:
                raise ValueError(
                    'Feature shape mismatch: '
                    f'teacher={tuple(teacher_feature.shape)}, '
                    f'student={tuple(student_feature.shape)}')
            student_float = student_feature.float()
            squared_error = (
                teacher_feature.detach().float() - student_float
            ).square()
            error_maps.append(squared_error.sum(dim=1))
            # Graph-connected zero for empty-GT batches on every rank.
            level_sums.append(student_float.sum() * 0.0)
            level_mask_masses.append(torch.zeros(
                (), device=student_float.device, dtype=torch.float32))

        device = student_features[0].device
        local_effective = torch.zeros((), device=device, dtype=torch.float32)
        local_matched = torch.zeros((), device=device, dtype=torch.float32)
        local_value_sum = torch.zeros((), device=device, dtype=torch.float32)
        local_skipped = torch.zeros((), device=device, dtype=torch.float32)
        group_names = ('small', 'large')
        group_level_sums = {
            name: [value * 0.0 for value in level_sums]
            for name in group_names
        }
        group_level_masses = {
            name: [value * 0.0 for value in level_mask_masses]
            for name in group_names
        }

        for batch_idx, gt_boxes in enumerate(match_result.gt_boxes):
            effective = match_result.effective_gt_mask[batch_idx]
            matched = match_result.matched_mask[batch_idx]
            values = match_result.teacher_values[batch_idx].float()
            feat_width, feat_height = self.feature_map_size
            merged_base_mask = torch.zeros(
                (feat_height, feat_width), device=device,
                dtype=torch.float32)
            group_base_masks = {
                name: torch.zeros_like(merged_base_mask)
                for name in group_names
            }
            for gt_idx in torch.nonzero(effective, as_tuple=False).flatten():
                class_id = int(gt_boxes[gt_idx, -1].item())
                group_name = (
                    'small' if class_id in self.small_class_ids else 'large')
                value = values[gt_idx]
                base_mask = self._draw_base_mask(
                    gt_boxes[gt_idx], device=device)
                if base_mask is None:
                    local_skipped += 1.0
                    continue

                # Only drawable GTs participate in the mask and diagnostics.
                local_effective += 1.0
                local_matched += matched[gt_idx].float()
                local_value_sum += value

                weighted_mask = value * base_mask
                merged_base_mask = self._merge(
                    merged_base_mask, weighted_mask)
                group_base_masks[group_name] = self._merge(
                    group_base_masks[group_name], weighted_mask)

            for level_idx, error_map in enumerate(error_maps):
                target_size = error_map.shape[-2:]
                if merged_base_mask.shape != target_size:
                    level_mask = F.interpolate(
                        merged_base_mask[None, None], size=target_size,
                        mode='bilinear', align_corners=True)[0, 0]
                else:
                    level_mask = merged_base_mask
                level_mask_masses[level_idx] = (
                    level_mask_masses[level_idx] + level_mask.sum())
                level_sums[level_idx] = level_sums[level_idx] + (
                    error_map[batch_idx] * level_mask).sum()
                for group_name in group_names:
                    group_mask = group_base_masks[group_name]
                    if group_mask.shape != error_map.shape[-2:]:
                        group_mask = F.interpolate(
                            group_mask[None, None],
                            size=error_map.shape[-2:], mode='bilinear',
                            align_corners=True)[0, 0]
                    group_level_masses[group_name][level_idx] = (
                        group_level_masses[group_name][level_idx]
                        + group_mask.sum())
                    group_level_sums[group_name][level_idx] = (
                        group_level_sums[group_name][level_idx]
                        + (error_map[batch_idx] * group_mask).sum()
                    )

        level_losses_list = []
        for level_sum, local_mass in zip(level_sums, level_mask_masses):
            global_mass, world_size = self._global_sum(local_mass)
            if global_mass > 0:
                level_losses_list.append(
                    level_sum * float(world_size) / global_mass)
            else:
                level_losses_list.append(level_sum * 0.0)
        level_losses = tuple(level_losses_list)

        group_losses = {}
        for name in ('small', 'large'):
            group_loss = group_level_sums[name][0] * 0.0
            for level_sum, local_mass in zip(
                    group_level_sums[name], group_level_masses[name]):
                global_mass, group_world_size = self._global_sum(local_mass)
                if global_mass > 0:
                    group_loss = (
                        group_loss
                        + level_sum * float(group_world_size) / global_mass
                    )
            group_losses[name] = group_loss

        return RawGaussianFeatureLossOutput(
            loss=sum(level_losses),
            level_losses=level_losses,
            group_losses=group_losses,
            effective_gt_count=local_effective.detach(),
            matched_gt_count=local_matched.detach(),
            teacher_value_sum=local_value_sum.detach(),
            skipped_gt_count=local_skipped.detach(),
        )
