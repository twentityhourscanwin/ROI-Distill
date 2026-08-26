"""Analyze frozen teacher proposals against nuScenes train/val GT.

The extractor deliberately keeps the study narrow: class identity, BEV center
distance and teacher score.  It runs the frozen CenterPoint teacher only and
stores enough candidate information to recompute matching diagnostics without
another model forward.

Distributed usage (two PPU devices, batch 16 per process):

    torchrun --standalone --nproc_per_node=2 \
      tools/analyze_teacher_gt_matching.py \
      --config configs/experiments/center_value_baseline_cp50200.yaml \
      --batch-size 16 \
      --output-dir outputs/teacher_gt_matching_r50_train
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import mmengine
import numpy as np
import torch
import torch.distributed as dist
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from mmdet3d.registry import MODELS
from torch.utils.data import DataLoader, Dataset, Sampler

import mmdet3d.models  # noqa: F401 - register mmdet3d modules

from labeldistill.config import load_and_resolve_config
from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset
from labeldistill.models.teacher_proposal_decoder import TeacherProposalDecoder
from labeldistill.presets import build_j4_model_configs


GT_COLUMNS = (
    "sample_index",
    "gt_index",
    "gt_label",
    "is_small",
    "num_task_candidates_4m",
    "num_exact_candidates_4m",
    "num_current_selection_candidates",
    "current_matched",
    "selected_proposal_index",
    "selected_label",
    "selected_distance",
    "selected_score",
    "selected_exact_class",
    "nearest_task_distance",
    "nearest_task_score",
    "nearest_exact_distance",
    "nearest_exact_score",
    "current_top1_score",
    "current_top2_score",
    "current_top1_top2_gap",
    "current_top1_distance",
    "current_highest_score_is_nearest",
    "selected_proposal_reused",
)

CANDIDATE_COLUMNS = (
    "sample_index",
    "gt_index",
    "gt_label",
    "is_small",
    "proposal_index",
    "proposal_label",
    "center_distance",
    "teacher_score",
    "exact_class",
    "selected_by_current_matcher",
)

SMALL_CLASS_NAMES = {
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
}

CLASS_MAX_RANGES = {
    "car": 50.0,
    "truck": 50.0,
    "construction_vehicle": 50.0,
    "bus": 50.0,
    "trailer": 50.0,
    "barrier": 30.0,
    "motorcycle": 40.0,
    "bicycle": 40.0,
    "pedestrian": 40.0,
    "traffic_cone": 30.0,
}

DISTANCE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
DISTANCE_BIN_NAMES = ("<0.5", "0.5-1", "1-2", "2-4")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="nuScenes split to analyze.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override config.teacher.checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--flush-every-batches", type=int, default=100)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Restrict the global train-set prefix for smoke testing.",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Run teacher forward in float32 instead of float16 autocast.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Do not aggregate parts on rank zero after extraction.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Reuse existing parts without running the teacher.",
    )
    parser.add_argument(
        "--official-ranges",
        action="store_true",
        help="Filter GT by nuScenes class-specific 30/40/50m ranges.",
    )
    return parser.parse_args()


class RankStrideSampler(Sampler[int]):
    """Shard indices by rank without DistributedSampler padding duplicates."""

    def __init__(self, size: int, rank: int, world_size: int):
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class TeacherTrainDataset(Dataset):
    """LiDAR-and-GT-only view of a nuScenes labeled split."""

    def __init__(self, config, model_configs, info_path: str, split: str):
        self.base = NuscDetDataset(
            ida_aug_conf=model_configs.ida_aug,
            bda_aug_conf=model_configs.bda_aug,
            classes=list(config.classes.names),
            data_root=config.data.root,
            info_paths=info_path,
            is_train=split == "train",
            use_cbgs=False,
            img_conf=model_configs.img,
            num_sweeps=1,
            sweep_idxes=[],
            key_idxes=list(config.data.key_idxes),
            return_depth=False,
            return_lidar=True,
            use_fusion=False,
        )
        self.cams = list(model_configs.ida_aug["cams"])
        self.identity_bda = torch.eye(3, dtype=torch.float32)
        self.lidar_indices = np.arange(9)

    def __len__(self):
        return len(self.base.infos)

    def __getitem__(self, index: int):
        info = self.base.infos[index]
        gt_boxes, gt_labels = self.base.get_gt(info, self.cams)
        lidar_infos = [[info["lidar_infos"]] + info["lidar_sweeps"]]
        lidar_points = self.base.get_lidar_points(
            lidar_infos, self.identity_bda, self.lidar_indices)
        return {
            "sample_index": index,
            "sample_token": info["sample_token"],
            "lidar_points": lidar_points,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        }


def collate_teacher_batch(samples):
    return {
        "sample_indices": torch.tensor(
            [sample["sample_index"] for sample in samples], dtype=torch.long),
        "sample_tokens": [sample["sample_token"] for sample in samples],
        "lidar_points": torch.stack(
            [sample["lidar_points"] for sample in samples]),
        "gt_boxes": [sample["gt_boxes"] for sample in samples],
        "gt_labels": [sample["gt_labels"] for sample in samples],
    }


class FrozenTeacher(torch.nn.Module):
    """CenterPoint, voxelizer and decoder without the camera student."""

    def __init__(self, config, model_configs, checkpoint_path: Path):
        super().__init__()
        teacher_config = deepcopy(model_configs.teacher)
        voxel_config = teacher_config.pop("voxel_layer")
        self.voxelizer = VoxelizationByGridShape(**voxel_config)
        self.centerpoint = MODELS.build(teacher_config)
        self.decoder = TeacherProposalDecoder(**model_configs.teacher_proposal)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        prefix = config.teacher.checkpoint_prefix
        state_dict = {
            key[len(prefix):]: value
            for key, value in checkpoint["state_dict"].items()
            if key.startswith(prefix)
        }
        if not state_dict:
            raise RuntimeError(
                f"No teacher parameters with prefix {prefix!r} in "
                f"{checkpoint_path}")
        self.centerpoint.load_state_dict(state_dict)
        self.centerpoint.eval()
        for parameter in self.centerpoint.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.centerpoint.eval()
        return self

    def forward(self, lidar_points: torch.Tensor):
        points = lidar_points[:, 0]
        voxels_list = []
        coordinates_list = []
        num_points_list = []
        for batch_index, sample_points in enumerate(points):
            voxels, coordinates, num_points = self.voxelizer(sample_points)
            batch_column = coordinates.new_full(
                (coordinates.shape[0], 1), batch_index)
            coordinates = torch.cat([batch_column, coordinates], dim=1)
            voxels_list.append(voxels)
            coordinates_list.append(coordinates)
            num_points_list.append(num_points)

        voxels = torch.cat(voxels_list, dim=0)
        coordinates = torch.cat(coordinates_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)
        voxel_features = self.centerpoint.pts_voxel_encoder(
            voxels, num_points, coordinates)
        features = self.centerpoint.pts_middle_encoder(
            voxel_features, coordinates, len(points))
        features = self.centerpoint.pts_backbone(features)
        neck_features = self.centerpoint.pts_neck(features)
        predictions = self.centerpoint.pts_bbox_head(neck_features)
        return self.decoder(predictions)


@dataclass(frozen=True)
class MatchingPolicy:
    group_lookup: torch.Tensor
    high_thresholds: torch.Tensor
    medium_thresholds: torch.Tensor
    small_lookup: torch.Tensor


def build_matching_policy(config) -> MatchingPolicy:
    names = list(config.classes.names)
    name_to_id = {name: index for index, name in enumerate(names)}
    group_ids = [-1] * len(names)
    for group_index, group in enumerate(config.classes.tasks):
        for name in group:
            group_ids[name_to_id[name]] = group_index
    high = [
        float(config.matching.distance_thresholds[name].high)
        for name in names
    ]
    medium = [
        float(config.matching.distance_thresholds[name].medium)
        for name in names
    ]
    small = [name in SMALL_CLASS_NAMES for name in names]
    return MatchingPolicy(
        group_lookup=torch.tensor(group_ids, dtype=torch.long),
        high_thresholds=torch.tensor(high, dtype=torch.float32),
        medium_thresholds=torch.tensor(medium, dtype=torch.float32),
        small_lookup=torch.tensor(small, dtype=torch.bool),
    )


def _nearest_index(distances: torch.Tensor, indices: torch.Tensor):
    if indices.numel() == 0:
        return -1, float("inf"), 0.0
    local = distances[indices].argmin()
    proposal_index = int(indices[local])
    return proposal_index, float(distances[proposal_index]), local


def analyze_sample(
    sample_index: int,
    proposal_boxes: torch.Tensor,
    proposal_scores: torch.Tensor,
    proposal_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    policy: MatchingPolicy,
):
    """Return numeric GT summaries and all same-task candidates within 4m."""
    proposal_boxes = proposal_boxes.detach().float().cpu()
    proposal_scores = proposal_scores.detach().float().cpu()
    proposal_labels = proposal_labels.detach().long().cpu()
    gt_boxes = gt_boxes.detach().float().cpu()
    gt_labels = gt_labels.detach().long().cpu()

    num_gt = len(gt_boxes)
    if num_gt == 0:
        return [], []
    if len(proposal_boxes):
        distance_matrix = torch.cdist(
            proposal_boxes[:, :2], gt_boxes[:, :2], p=2)
        valid_proposal_labels = (
            (proposal_labels >= 0)
            & (proposal_labels < len(policy.group_lookup)))
        proposal_groups = torch.full_like(proposal_labels, -1)
        proposal_groups[valid_proposal_labels] = policy.group_lookup[
            proposal_labels[valid_proposal_labels]]
    else:
        distance_matrix = torch.empty((0, num_gt), dtype=torch.float32)
        proposal_groups = torch.empty((0,), dtype=torch.long)

    selected_indices = [-1] * num_gt
    gt_rows = []
    candidate_rows = []

    for gt_index in range(num_gt):
        gt_label = int(gt_labels[gt_index])
        gt_group = int(policy.group_lookup[gt_label])
        is_small = int(policy.small_lookup[gt_label])
        distances = distance_matrix[:, gt_index]
        same_task = proposal_groups == gt_group
        exact_class = proposal_labels == gt_label
        task_4m = torch.nonzero(
            same_task & (distances < 4.0), as_tuple=False).flatten()
        exact_4m = torch.nonzero(
            exact_class & (distances < 4.0), as_tuple=False).flatten()

        high_threshold = float(policy.high_thresholds[gt_label])
        medium_threshold = float(policy.medium_thresholds[gt_label])
        high_candidates = torch.nonzero(
            same_task & (distances <= high_threshold),
            as_tuple=False,
        ).flatten()
        medium_candidates = torch.nonzero(
            same_task
            & (distances > high_threshold)
            & (distances <= medium_threshold),
            as_tuple=False,
        ).flatten()
        current_candidates = (
            high_candidates if high_candidates.numel() else medium_candidates)
        selected = -1
        if current_candidates.numel():
            selected = int(
                current_candidates[proposal_scores[current_candidates].argmax()])
        selected_indices[gt_index] = selected

        def nearest_for(indices):
            if indices.numel() == 0:
                return -1, float("inf"), 0.0
            local_index = distances[indices].argmin()
            proposal_index = int(indices[local_index])
            return (
                proposal_index,
                float(distances[proposal_index]),
                float(proposal_scores[proposal_index]),
            )

        nearest_task_index, nearest_task_distance, nearest_task_score = (
            nearest_for(task_4m))
        nearest_exact_index, nearest_exact_distance, nearest_exact_score = (
            nearest_for(exact_4m))

        top1_score = 0.0
        top2_score = 0.0
        score_gap = 0.0
        top1_distance = float("inf")
        highest_is_nearest = 0
        if current_candidates.numel():
            sorted_indices = current_candidates[
                torch.argsort(
                    proposal_scores[current_candidates], descending=True)]
            top1_index = int(sorted_indices[0])
            top1_score = float(proposal_scores[top1_index])
            top1_distance = float(distances[top1_index])
            nearest_current = int(
                current_candidates[distances[current_candidates].argmin()])
            highest_is_nearest = int(top1_index == nearest_current)
            if len(sorted_indices) > 1:
                top2_score = float(proposal_scores[int(sorted_indices[1])])
                score_gap = top1_score - top2_score

        selected_label = int(proposal_labels[selected]) if selected >= 0 else -1
        selected_distance = (
            float(distances[selected]) if selected >= 0 else float("inf"))
        selected_score = (
            float(proposal_scores[selected]) if selected >= 0 else 0.0)
        selected_exact = int(selected_label == gt_label) if selected >= 0 else 0

        gt_rows.append([
            sample_index,
            gt_index,
            gt_label,
            is_small,
            len(task_4m),
            len(exact_4m),
            len(current_candidates),
            int(selected >= 0),
            selected,
            selected_label,
            selected_distance,
            selected_score,
            selected_exact,
            nearest_task_distance,
            nearest_task_score,
            nearest_exact_distance,
            nearest_exact_score,
            top1_score,
            top2_score,
            score_gap,
            top1_distance,
            highest_is_nearest,
            0,
        ])

        for proposal_index_tensor in task_4m:
            proposal_index = int(proposal_index_tensor)
            candidate_rows.append([
                sample_index,
                gt_index,
                gt_label,
                is_small,
                proposal_index,
                int(proposal_labels[proposal_index]),
                float(distances[proposal_index]),
                float(proposal_scores[proposal_index]),
                int(proposal_labels[proposal_index] == gt_label),
                int(proposal_index == selected),
            ])

    selected_counts = defaultdict(int)
    for selected in selected_indices:
        if selected >= 0:
            selected_counts[selected] += 1
    for gt_index, selected in enumerate(selected_indices):
        gt_rows[gt_index][-1] = int(
            selected >= 0 and selected_counts[selected] > 1)
    return gt_rows, candidate_rows


def rows_to_tensor(rows, width: int):
    if not rows:
        return torch.empty((0, width), dtype=torch.float64)
    return torch.tensor(rows, dtype=torch.float64)


def save_part(
    output_dir: Path,
    rank: int,
    part_index: int,
    gt_rows,
    candidate_rows,
    sample_tokens,
):
    part_path = output_dir / f"part_rank{rank:02d}_{part_index:05d}.pt"
    payload = {
        "gt_columns": GT_COLUMNS,
        "candidate_columns": CANDIDATE_COLUMNS,
        "gt": rows_to_tensor(gt_rows, len(GT_COLUMNS)),
        "candidates": rows_to_tensor(candidate_rows, len(CANDIDATE_COLUMNS)),
        "sample_tokens": dict(sample_tokens),
    }
    torch.save(payload, part_path)
    return part_path


def quantiles(values: torch.Tensor):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {name: None for name in ("mean", "p10", "p25", "p50", "p75", "p90")}
    points = torch.tensor(
        [0.10, 0.25, 0.50, 0.75, 0.90], dtype=finite.dtype)
    result = torch.quantile(finite, points)
    return {
        "mean": float(finite.mean()),
        "p10": float(result[0]),
        "p25": float(result[1]),
        "p50": float(result[2]),
        "p75": float(result[3]),
        "p90": float(result[4]),
    }


def ratio(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


def distance_bin_mask(values: torch.Tensor, bin_index: int):
    lower = (0.0, 0.5, 1.0, 2.0)[bin_index]
    upper = DISTANCE_THRESHOLDS[bin_index]
    return (values >= lower) & (values < upper)


def build_gt_range_index(
    config,
    model_configs,
    info_path: str,
    output_path: Path,
    split: str,
):
    """Recover canonical ego distance for every saved (sample, GT) key."""
    dataset = TeacherTrainDataset(config, model_configs, info_path, split)
    rows = []
    for sample_index, info in enumerate(dataset.base.infos):
        gt_boxes, gt_labels = dataset.base.get_gt(info, dataset.cams)
        distances = torch.linalg.vector_norm(gt_boxes[:, :2], dim=1)
        for gt_index, (label, distance) in enumerate(zip(gt_labels, distances)):
            rows.append([
                sample_index,
                gt_index,
                int(label),
                float(distance),
            ])
    columns = ("sample_index", "gt_index", "gt_label", "ego_distance")
    torch.save({
        "columns": columns,
        "rows": rows_to_tensor(rows, len(columns)),
        "class_max_ranges": CLASS_MAX_RANGES,
    }, output_path)
    return output_path


def _filter_official_ranges(
    gt: torch.Tensor,
    candidates: torch.Tensor,
    range_index_path: Path,
    class_names: Sequence[str],
):
    payload = torch.load(range_index_path, map_location="cpu")
    columns = tuple(payload["columns"])
    expected = ("sample_index", "gt_index", "gt_label", "ego_distance")
    if columns != expected:
        raise RuntimeError(
            f"GT range schema mismatch: expected={expected}, got={columns}")
    ranges = payload["rows"]
    range_keys = (
        ranges[:, 0].long() * 1000 + ranges[:, 1].long())
    order = torch.argsort(range_keys)
    sorted_keys = range_keys[order]
    sorted_ranges = ranges[order]
    max_ranges = torch.tensor(
        [CLASS_MAX_RANGES[name] for name in class_names],
        dtype=torch.float64,
    )

    def keep_rows(rows, sample_column, gt_column, label_column):
        keys = (
            rows[:, sample_column].long() * 1000
            + rows[:, gt_column].long())
        positions = torch.searchsorted(sorted_keys, keys)
        if len(positions) and (
            int(positions.max()) >= len(sorted_keys)
            or not torch.equal(sorted_keys[positions], keys)
        ):
            raise RuntimeError("Saved GT keys do not align with range index")
        ego_distance = sorted_ranges[positions, 3]
        labels = rows[:, label_column].long()
        return ego_distance <= max_ranges[labels]

    gt_keep = keep_rows(gt, 0, 1, 2)
    candidate_keep = keep_rows(candidates, 0, 1, 2)
    return gt[gt_keep], candidates[candidate_keep]


def summarize_parts(
    output_dir: Path,
    class_names: Sequence[str],
    *,
    range_index_path: Path | None = None,
    output_suffix: str = "",
):
    part_paths = sorted(output_dir.glob("part_rank*_*.pt"))
    if not part_paths:
        raise RuntimeError(f"No extraction parts found in {output_dir}")
    gt_parts = []
    candidate_parts = []
    token_map = {}
    for path in part_paths:
        payload = torch.load(path, map_location="cpu")
        if tuple(payload["gt_columns"]) != GT_COLUMNS:
            raise RuntimeError(f"GT schema mismatch in {path}")
        if tuple(payload["candidate_columns"]) != CANDIDATE_COLUMNS:
            raise RuntimeError(f"Candidate schema mismatch in {path}")
        gt_parts.append(payload["gt"])
        candidate_parts.append(payload["candidates"])
        token_map.update(payload["sample_tokens"])
    gt = torch.cat(gt_parts, dim=0)
    candidates = torch.cat(candidate_parts, dim=0)
    source_num_gt = len(gt)
    source_num_candidates = len(candidates)
    if range_index_path is not None:
        gt, candidates = _filter_official_ranges(
            gt, candidates, range_index_path, class_names)
    gi = {name: index for index, name in enumerate(GT_COLUMNS)}
    ci = {name: index for index, name in enumerate(CANDIDATE_COLUMNS)}

    report = {
        "num_samples": len(token_map),
        "num_gt": len(gt),
        "num_candidates_within_4m_same_task": len(candidates),
        "source_num_gt_before_range_filter": source_num_gt,
        "source_num_candidates_before_range_filter": source_num_candidates,
        "official_class_ranges": (
            CLASS_MAX_RANGES if range_index_path is not None else None),
        "groups": {},
        "classes": {},
        "distance_hits": {},
        "nearest_exact_score_by_distance": {},
        "selected_score_by_distance": {},
        "candidate_score_by_distance": {},
    }

    def summarize_scope(mask: torch.Tensor):
        scoped = gt[mask]
        total = len(scoped)
        matched = int(scoped[:, gi["current_matched"]].sum()) if total else 0
        exact = int(
            scoped[:, gi["selected_exact_class"]].sum()) if total else 0
        reused = int(
            scoped[:, gi["selected_proposal_reused"]].sum()) if total else 0
        multi = int(
            (scoped[:, gi["num_current_selection_candidates"]] >= 2).sum()
        ) if total else 0
        candidate_gt = int(
            (scoped[:, gi["num_task_candidates_4m"]] >= 1).sum()) if total else 0
        return {
            "gt": total,
            "matched": matched,
            "unmatched": total - matched,
            "matched_rate": ratio(matched, total),
            "unmatched_rate": ratio(total - matched, total),
            "selected_exact_class": exact,
            "selected_exact_rate_among_matched": ratio(exact, matched),
            "selected_reused": reused,
            "selected_reused_rate_among_matched": ratio(reused, matched),
            "gt_with_candidate_4m": candidate_gt,
            "gt_with_multiple_current_candidates": multi,
            "multiple_candidate_rate_among_matched": ratio(multi, matched),
        }

    group_masks = {
        "small": gt[:, gi["is_small"]] == 1,
        "large": gt[:, gi["is_small"]] == 0,
    }
    for group_name, group_mask in group_masks.items():
        report["groups"][group_name] = summarize_scope(group_mask)

    for label, class_name in enumerate(class_names):
        report["classes"][class_name] = summarize_scope(
            gt[:, gi["gt_label"]] == label)

    for distance_type, column in (
        ("nearest_exact", "nearest_exact_distance"),
        ("nearest_same_task", "nearest_task_distance"),
    ):
        report["distance_hits"][distance_type] = {}
        for group_name, group_mask in group_masks.items():
            values = gt[group_mask, gi[column]]
            total = len(values)
            counts = {
                f"lt_{threshold:g}m": int((values < threshold).sum())
                for threshold in DISTANCE_THRESHOLDS
            }
            report["distance_hits"][distance_type][group_name] = {
                "gt": total,
                **counts,
                **{
                    f"rate_lt_{threshold:g}m": ratio(
                        counts[f"lt_{threshold:g}m"], total)
                    for threshold in DISTANCE_THRESHOLDS
                },
                "unmatched_4m": int((values >= 4.0).sum()),
                "unmatched_4m_rate": ratio(int((values >= 4.0).sum()), total),
            }

    selected = gt[:, gi["current_matched"]] == 1
    selected_distances = gt[:, gi["selected_distance"]]
    selected_scores = gt[:, gi["selected_score"]]
    for group_name, group_mask in group_masks.items():
        report["nearest_exact_score_by_distance"][group_name] = {}
        nearest_exact_distances = gt[:, gi["nearest_exact_distance"]]
        nearest_exact_scores = gt[:, gi["nearest_exact_score"]]
        for bin_index, bin_name in enumerate(DISTANCE_BIN_NAMES):
            mask = group_mask & distance_bin_mask(
                nearest_exact_distances, bin_index)
            report["nearest_exact_score_by_distance"][group_name][bin_name] = {
                "matched": int(mask.sum()),
                "score": quantiles(nearest_exact_scores[mask]),
            }
        report["selected_score_by_distance"][group_name] = {}
        for bin_index, bin_name in enumerate(DISTANCE_BIN_NAMES):
            mask = selected & group_mask & distance_bin_mask(
                selected_distances, bin_index)
            scoped = gt[mask]
            multi = scoped[:, gi["num_current_selection_candidates"]] >= 2
            gaps = scoped[multi, gi["current_top1_top2_gap"]]
            highest_nearest = scoped[
                :, gi["current_highest_score_is_nearest"]]
            report["selected_score_by_distance"][group_name][bin_name] = {
                "matched": len(scoped),
                "score": quantiles(selected_scores[mask]),
                "multiple_candidates": int(multi.sum()),
                "multiple_candidate_rate": ratio(int(multi.sum()), len(scoped)),
                "top1_top2_gap": quantiles(gaps),
                "highest_score_is_nearest_rate": (
                    float(highest_nearest.mean()) if len(scoped) else None),
            }

    if len(candidates):
        candidate_group_masks = {
            "small": candidates[:, ci["is_small"]] == 1,
            "large": candidates[:, ci["is_small"]] == 0,
        }
        candidate_distances = candidates[:, ci["center_distance"]]
        candidate_scores = candidates[:, ci["teacher_score"]]
        for group_name, group_mask in candidate_group_masks.items():
            report["candidate_score_by_distance"][group_name] = {}
            for bin_index, bin_name in enumerate(DISTANCE_BIN_NAMES):
                mask = group_mask & distance_bin_mask(
                    candidate_distances, bin_index)
                report["candidate_score_by_distance"][group_name][bin_name] = {
                    "candidates": int(mask.sum()),
                    "score": quantiles(candidate_scores[mask]),
                    "exact_class_rate": (
                        float(candidates[mask, ci["exact_class"]].mean())
                        if int(mask.sum()) else None),
                }

    sample_rows = []
    for sample_index_value in torch.unique(gt[:, gi["sample_index"]]).tolist():
        sample_index = int(sample_index_value)
        sample_mask = gt[:, gi["sample_index"]] == sample_index
        row = {
            "sample_index": sample_index,
            "sample_token": token_map.get(sample_index, ""),
        }
        for group_name, group_mask in group_masks.items():
            scoped = sample_mask & group_mask
            distances = gt[scoped, gi["nearest_exact_distance"]]
            row[f"{group_name}_gt"] = len(distances)
            for threshold in DISTANCE_THRESHOLDS:
                row[f"{group_name}_lt_{threshold:g}m"] = int(
                    (distances < threshold).sum())
        sample_rows.append(row)

    with (output_dir / f"summary{output_suffix}.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    if sample_rows:
        with (output_dir / f"sample_hits{output_suffix}.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)
    write_markdown_report(output_dir / f"report{output_suffix}.md", report)
    return report


def _percent(value):
    return "-" if value is None else f"{100.0 * value:.2f}%"


def _number(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def write_markdown_report(path: Path, report):
    range_note = (
        "Official class ranges: barrier/traffic_cone 30m; "
        "pedestrian/motorcycle/bicycle 40m; vehicles 50m."
        if report["official_class_ranges"] is not None
        else "Official class ranges: not applied."
    )
    lines = [
        "# R50 teacher–GT train-set matching statistics",
        "",
        range_note,
        "",
        f"Samples: {report['num_samples']}",
        f"GT: {report['num_gt']}",
        (
            "Same-task candidates within 4m: "
            f"{report['num_candidates_within_4m_same_task']}"
        ),
        "",
        "## Matching reliability and unmatched GT",
        "",
        "| Group | GT | Matched | Unmatched | Unmatched rate | Exact-class rate among matched | Reused rate among matched | Multi-candidate rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name in ("small", "large"):
        item = report["groups"][group_name]
        lines.append(
            f"| {group_name} | {item['gt']} | {item['matched']} | "
            f"{item['unmatched']} | {_percent(item['unmatched_rate'])} | "
            f"{_percent(item['selected_exact_rate_among_matched'])} | "
            f"{_percent(item['selected_reused_rate_among_matched'])} | "
            f"{_percent(item['multiple_candidate_rate_among_matched'])} |"
        )

    lines.extend([
        "",
        "## Exact-class nearest-proposal distance hits",
        "",
        "| Group | GT | <0.5m | <1m | <2m | <4m | Unmatched@4m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    exact_hits = report["distance_hits"]["nearest_exact"]
    for group_name in ("small", "large"):
        item = exact_hits[group_name]
        lines.append(
            f"| {group_name} | {item['gt']} | "
            f"{item['lt_0.5m']} ({_percent(item['rate_lt_0.5m'])}) | "
            f"{item['lt_1m']} ({_percent(item['rate_lt_1m'])}) | "
            f"{item['lt_2m']} ({_percent(item['rate_lt_2m'])}) | "
            f"{item['lt_4m']} ({_percent(item['rate_lt_4m'])}) | "
            f"{item['unmatched_4m']} ({_percent(item['unmatched_4m_rate'])}) |"
        )

    lines.extend([
        "",
        "## Nearest exact-class proposal score",
        "",
        "| Group | Distance | Matches | Score mean | Score p25 | Score p50 | Score p75 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for group_name in ("small", "large"):
        for bin_name in DISTANCE_BIN_NAMES:
            item = report["nearest_exact_score_by_distance"][group_name][bin_name]
            lines.append(
                f"| {group_name} | {bin_name} | {item['matched']} | "
                f"{_number(item['score']['mean'])} | "
                f"{_number(item['score']['p25'])} | "
                f"{_number(item['score']['p50'])} | "
                f"{_number(item['score']['p75'])} |"
            )

    lines.extend([
        "",
        "## Selected score and multi-candidate ambiguity",
        "",
        "| Group | Distance | Matches | Score mean | Score p50 | Multi-candidate rate | Gap mean | Gap p50 | Highest-score is nearest |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for group_name in ("small", "large"):
        for bin_name in DISTANCE_BIN_NAMES:
            item = report["selected_score_by_distance"][group_name][bin_name]
            lines.append(
                f"| {group_name} | {bin_name} | {item['matched']} | "
                f"{_number(item['score']['mean'])} | "
                f"{_number(item['score']['p50'])} | "
                f"{_percent(item['multiple_candidate_rate'])} | "
                f"{_number(item['top1_top2_gap']['mean'])} | "
                f"{_number(item['top1_top2_gap']['p50'])} | "
                f"{_percent(item['highest_score_is_nearest_rate'])} |"
            )

    lines.extend([
        "",
        "## Per-class matching summary",
        "",
        "| Class | GT | Matched rate | Unmatched rate | Exact-class rate | Reused rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for class_name, item in report["classes"].items():
        lines.append(
            f"| {class_name} | {item['gt']} | "
            f"{_percent(item['matched_rate'])} | "
            f"{_percent(item['unmatched_rate'])} | "
            f"{_percent(item['selected_exact_rate_among_matched'])} | "
            f"{_percent(item['selected_reused_rate_among_matched'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_path(project_root: Path, value: str):
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bundle = load_and_resolve_config(
        args.config, [], project_root=str(project_root))
    config = bundle.config
    model_configs = build_j4_model_configs(config)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    info_value = (
        config.data.train_info if args.split == "train"
        else config.data.val_info
    )
    info_path = resolve_path(Path(config.data.root), info_value)
    if args.summarize_only:
        range_path = None
        suffix = ""
        if args.official_ranges:
            range_path = output_dir / "gt_ego_ranges.pt"
            if not range_path.exists():
                build_gt_range_index(
                    config,
                    model_configs,
                    str(info_path),
                    range_path,
                    args.split,
                )
            suffix = "_official_ranges"
        report = summarize_parts(
            output_dir,
            list(config.classes.names),
            range_index_path=range_path,
            output_suffix=suffix,
        )
        print(json.dumps(report["groups"], ensure_ascii=False, indent=2))
        return

    dataset = TeacherTrainDataset(
        config, model_configs, str(info_path), args.split)
    global_size = len(dataset)
    if args.max_samples is not None:
        global_size = min(global_size, args.max_samples)
    sampler = RankStrideSampler(global_size, rank, world_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_teacher_batch,
    )

    checkpoint_value = args.checkpoint or config.teacher.checkpoint
    checkpoint_path = resolve_path(project_root, checkpoint_value)
    teacher = FrozenTeacher(config, model_configs, checkpoint_path).to(device)
    teacher.eval()
    policy = build_matching_policy(config)

    if rank == 0:
        metadata = {
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "info_path": str(info_path),
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "num_workers_per_rank": args.num_workers,
            "global_samples": global_size,
            "autocast": not args.no_autocast,
            "gt_columns": GT_COLUMNS,
            "candidate_columns": CANDIDATE_COLUMNS,
            "class_names": list(config.classes.names),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    gt_buffer = []
    candidate_buffer = []
    token_buffer = []
    part_index = 0
    processed = 0
    autocast_enabled = not args.no_autocast
    for batch_index, batch in enumerate(dataloader):
        lidar_points = batch["lidar_points"].to(
            device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                proposals = teacher(lidar_points)
        del lidar_points

        for local_index, sample_index_tensor in enumerate(
                batch["sample_indices"]):
            sample_index = int(sample_index_tensor)
            gt_rows, candidate_rows = analyze_sample(
                sample_index=sample_index,
                proposal_boxes=proposals.boxes[local_index],
                proposal_scores=proposals.scores[local_index],
                proposal_labels=proposals.labels[local_index],
                gt_boxes=batch["gt_boxes"][local_index],
                gt_labels=batch["gt_labels"][local_index],
                policy=policy,
            )
            gt_buffer.extend(gt_rows)
            candidate_buffer.extend(candidate_rows)
            token_buffer.append(
                (sample_index, batch["sample_tokens"][local_index]))
        processed += len(batch["sample_indices"])

        should_flush = (
            (batch_index + 1) % args.flush_every_batches == 0
            or batch_index + 1 == len(dataloader)
        )
        if should_flush:
            path = save_part(
                output_dir,
                rank,
                part_index,
                gt_buffer,
                candidate_buffer,
                token_buffer,
            )
            print(
                f"rank={rank} batches={batch_index + 1}/{len(dataloader)} "
                f"samples={processed} saved={path.name}",
                flush=True,
            )
            gt_buffer.clear()
            candidate_buffer.clear()
            token_buffer.clear()
            part_index += 1

    (output_dir / f"done_rank{rank:02d}.json").write_text(
        json.dumps({"rank": rank, "samples": processed, "parts": part_index}),
        encoding="utf-8",
    )
    if world_size > 1:
        dist.barrier()
    if rank == 0 and not args.extract_only:
        report = summarize_parts(output_dir, list(config.classes.names))
        print(json.dumps(report["groups"], ensure_ascii=False, indent=2))
        print(f"report={output_dir / 'report.md'}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
