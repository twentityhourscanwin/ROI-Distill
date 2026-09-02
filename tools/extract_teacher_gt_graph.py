"""Extract a deterministic teacher/GT candidate graph for matching studies.

The cache contains all exact-class proposal/GT edges with BEV center distance
below ``--edge-radius`` plus enough geometry to compare assignment policies
without another CenterPoint forward.  This script intentionally mirrors the
current training teacher input: key sweep + five past + four future sweeps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from mmcv.ops import box_iou_rotated
from torch.utils.data import DataLoader

from analyze_teacher_gt_matching import (
    FrozenTeacher,
    RankStrideSampler,
    TeacherTrainDataset,
    collate_teacher_batch,
    resolve_path,
)
from labeldistill.config import load_and_resolve_config
from labeldistill.presets import build_j4_model_configs


CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
SMALL_CLASSES = {
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
}
OFFICIAL_RANGES = {
    "car": 50.0, "truck": 50.0, "construction_vehicle": 50.0,
    "bus": 50.0, "trailer": 50.0, "barrier": 30.0,
    "motorcycle": 40.0, "bicycle": 40.0, "pedestrian": 40.0,
    "traffic_cone": 30.0,
}

GT_COLUMNS = (
    "sample_index", "gt_index", "label", "effective",
    "x", "y", "z", "dx", "dy", "dz", "yaw", "vx", "vy",
    "ego_distance", "speed",
)
EDGE_COLUMNS = (
    "sample_index", "gt_index", "proposal_index", "label",
    "distance", "score", "bev_iou", "size_log_l1", "yaw_error",
    "yaw_axis_error", "velocity_error", "proposal_x", "proposal_y", "proposal_z",
    "proposal_dx", "proposal_dy", "proposal_dz", "proposal_yaw",
    "proposal_vx", "proposal_vy",
)
PROPOSAL_DEGREE_COLUMNS = (
    "sample_index", "label", "total", "degree_0", "degree_1",
    "degree_ge_2",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--flush-every-batches", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--edge-radius", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-autocast", action="store_true")
    return parser.parse_args()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(project_root: Path, *args):
    return subprocess.check_output(
        ["git", *args], cwd=project_root, text=True).strip()


class DeterministicTeacherDataset(TeacherTrainDataset):
    """Identity-BDA view with the exact current 10-sweep teacher input."""

    def __init__(self, *args, seed: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = int(seed)
        # 0=key, 1..5=past, 6..9=first four future sweeps.
        self.lidar_indices = np.arange(10)

    def __getitem__(self, index: int):
        # get_lidar_points randomly caps very dense point clouds.  Seed by
        # sample so worker count/order cannot change the cached proposals.
        np.random.seed(self.seed + index)
        return super().__getitem__(index)


def rows_to_tensor(rows, width):
    if not rows:
        return torch.empty((0, width), dtype=torch.float64)
    return torch.tensor(rows, dtype=torch.float64)


def wrap_angle(delta: torch.Tensor):
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def effective_gt_mask(gt_boxes, gt_labels):
    names = CLASS_NAMES
    finite = torch.isfinite(gt_boxes[:, [0, 1, 3, 4]]).all(dim=1)
    drawable = (
        finite & (gt_boxes[:, 3] > 0) & (gt_boxes[:, 4] > 0)
        & (gt_boxes[:, 0] >= -51.2) & (gt_boxes[:, 0] < 51.2)
        & (gt_boxes[:, 1] >= -51.2) & (gt_boxes[:, 1] < 51.2)
    )
    ranges = torch.tensor(
        [OFFICIAL_RANGES[name] for name in names], dtype=gt_boxes.dtype)
    valid = (gt_labels >= 0) & (gt_labels < len(names))
    ego_distance = torch.linalg.vector_norm(gt_boxes[:, :2], dim=1)
    in_range = torch.zeros_like(valid)
    in_range[valid] = ego_distance[valid] <= ranges[gt_labels[valid]] + 1e-4
    return valid & drawable & in_range


def analyze_sample(sample_index, proposals, gt_boxes, gt_labels, edge_radius):
    boxes = proposals.boxes.detach().float().cpu()
    scores = proposals.scores.detach().float().cpu()
    labels = proposals.labels.detach().long().cpu()
    gt_boxes = gt_boxes.detach().float().cpu()
    gt_labels = gt_labels.detach().long().cpu()
    effective = effective_gt_mask(gt_boxes, gt_labels)

    gt_rows = []
    for gt_index, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        ego_distance = float(torch.linalg.vector_norm(box[:2]))
        speed = float(torch.linalg.vector_norm(box[7:9]))
        gt_rows.append([
            sample_index, gt_index, int(label), int(effective[gt_index]),
            *box.tolist(), ego_distance, speed,
        ])

    edge_rows = []
    proposal_degrees = torch.zeros(len(boxes), dtype=torch.long)
    if len(boxes) and len(gt_boxes):
        distances = torch.cdist(boxes[:, :2], gt_boxes[:, :2])
        for label in range(len(CLASS_NAMES)):
            pred_indices = torch.nonzero(labels == label).flatten()
            gt_indices = torch.nonzero(gt_labels == label).flatten()
            if not len(pred_indices) or not len(gt_indices):
                continue
            pair_mask = distances[pred_indices][:, gt_indices] < edge_radius
            pred_local, gt_local = torch.nonzero(pair_mask, as_tuple=True)
            if not len(pred_local):
                continue
            pair_pred = pred_indices[pred_local]
            pair_gt = gt_indices[gt_local]
            pred_box = boxes[pair_pred]
            target_box = gt_boxes[pair_gt]
            iou = box_iou_rotated(
                pred_box[:, [0, 1, 3, 4, 6]],
                target_box[:, [0, 1, 3, 4, 6]],
                aligned=True,
            )
            size_error = torch.abs(torch.log(
                pred_box[:, 3:6].clamp_min(1e-5)
                / target_box[:, 3:6].clamp_min(1e-5)
            )).mean(dim=1)
            yaw_error = wrap_angle(pred_box[:, 6] - target_box[:, 6])
            yaw_axis_error = torch.minimum(yaw_error, math.pi - yaw_error)
            velocity_error = torch.linalg.vector_norm(
                pred_box[:, 7:9] - target_box[:, 7:9], dim=1)
            for row_index in range(len(pair_pred)):
                pred_index = int(pair_pred[row_index])
                gt_index = int(pair_gt[row_index])
                edge_rows.append([
                    sample_index, gt_index, pred_index, label,
                    float(distances[pred_index, gt_index]),
                    float(scores[pred_index]), float(iou[row_index]),
                    float(size_error[row_index]), float(yaw_error[row_index]),
                    float(yaw_axis_error[row_index]), float(velocity_error[row_index]),
                    *boxes[pred_index].tolist(),
                ])

        # Degree is defined only against effective exact-class GT inside the
        # current class-specific trust radius.
        for pred_index in range(len(boxes)):
            label = int(labels[pred_index])
            if label < 0 or label >= len(CLASS_NAMES):
                continue
            radius = 1.0 if CLASS_NAMES[label] in SMALL_CLASSES else 2.0
            proposal_degrees[pred_index] = int((
                effective & (gt_labels == label)
                & (distances[pred_index] < radius)
            ).sum())

    degree_rows = []
    for label in range(len(CLASS_NAMES)):
        mask = labels == label
        values = proposal_degrees[mask]
        degree_rows.append([
            sample_index, label, int(mask.sum()), int((values == 0).sum()),
            int((values == 1).sum()), int((values >= 2).sum()),
        ])
    return gt_rows, edge_rows, degree_rows


def save_part(output_dir, rank, part_index, gt_rows, edge_rows, degree_rows,
              sample_tokens):
    path = output_dir / f"graph_rank{rank:02d}_{part_index:05d}.pt"
    torch.save({
        "gt_columns": GT_COLUMNS,
        "edge_columns": EDGE_COLUMNS,
        "proposal_degree_columns": PROPOSAL_DEGREE_COLUMNS,
        "gt": rows_to_tensor(gt_rows, len(GT_COLUMNS)),
        "edges": rows_to_tensor(edge_rows, len(EDGE_COLUMNS)),
        "proposal_degrees": rows_to_tensor(
            degree_rows, len(PROPOSAL_DEGREE_COLUMNS)),
        "sample_tokens": dict(sample_tokens),
    }, path)
    return path


def sweep_audit(dataset, sample_indices):
    rows = []
    for sample_index in sample_indices:
        info = dataset.base.infos[sample_index]
        sweeps = [info["lidar_infos"]] + info["lidar_sweeps"]
        selected = sweeps[:10]
        key_ts = selected[0]["LIDAR_TOP"]["timestamp"] / 1e6
        time_lags = [
            key_ts - item["LIDAR_TOP"]["timestamp"] / 1e6
            for item in selected
        ]
        rows.append({
            "sample_index": sample_index,
            "sample_token": info["sample_token"],
            "available_entries": len(sweeps),
            "selected_entries": len(selected),
            "time_lags_seconds": time_lags,
            "padded_key_sweeps": 10 - len(selected),
            "effective_time_lags_seconds": time_lags + [0.0] * (10 - len(selected)),
        })
    return rows


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
    info_value = config.data.train_info if args.split == "train" else config.data.val_info
    info_path = resolve_path(Path(config.data.root), info_value)
    dataset = DeterministicTeacherDataset(
        config, model_configs, str(info_path), args.split, seed=args.seed)
    global_size = len(dataset)
    if args.max_samples is not None:
        global_size = min(global_size, args.max_samples)
    sampler = RankStrideSampler(global_size, rank, world_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_teacher_batch,
    )

    checkpoint_value = args.checkpoint or config.teacher.checkpoint
    checkpoint_path = resolve_path(project_root, checkpoint_value)
    teacher = FrozenTeacher(config, model_configs, checkpoint_path).to(device)
    teacher.eval()

    if rank == 0:
        audit_indices = sorted(set([0, min(1, global_size - 1), global_size - 1]))
        metadata = {
            "schema_version": 2,
            "created_by": str(Path(__file__).resolve()),
            "git_commit": git_value(project_root, "rev-parse", "HEAD"),
            "git_branch": git_value(project_root, "branch", "--show-current"),
            "git_status_short": git_value(project_root, "status", "--short"),
            "config": str(resolve_path(project_root, args.config)),
            "split": args.split,
            "info_path": str(info_path),
            "info_sha256": sha256(info_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "num_workers_per_rank": args.num_workers,
            "global_samples": global_size,
            "autocast": not args.no_autocast,
            "seed": args.seed,
            "point_cap_seed": "seed + sample_index",
            "identity_bda": True,
            "edge_radius_m": args.edge_radius,
            "teacher_sweep_indices": list(range(10)),
            "teacher_sweep_semantics": "current + 5 past + 4 future",
            "sweep_audit": sweep_audit(dataset, audit_indices),
            "gt_columns": GT_COLUMNS,
            "edge_columns": EDGE_COLUMNS,
            "proposal_degree_columns": PROPOSAL_DEGREE_COLUMNS,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    gt_buffer, edge_buffer, degree_buffer, token_buffer = [], [], [], []
    part_index = processed = 0
    for batch_index, batch in enumerate(dataloader):
        lidar_points = batch["lidar_points"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16,
            enabled=not args.no_autocast,
        ):
            proposals = teacher(lidar_points)
        del lidar_points
        for local_index, sample_tensor in enumerate(batch["sample_indices"]):
            sample_index = int(sample_tensor)
            sample_proposals = SimpleNamespace(
                boxes=proposals.boxes[local_index],
                scores=proposals.scores[local_index],
                labels=proposals.labels[local_index],
            )
            gt_rows, edge_rows, degree_rows = analyze_sample(
                sample_index, sample_proposals,
                batch["gt_boxes"][local_index], batch["gt_labels"][local_index],
                args.edge_radius,
            )
            gt_buffer.extend(gt_rows)
            edge_buffer.extend(edge_rows)
            degree_buffer.extend(degree_rows)
            token_buffer.append((sample_index, batch["sample_tokens"][local_index]))
        processed += len(batch["sample_indices"])
        if ((batch_index + 1) % args.flush_every_batches == 0
                or batch_index + 1 == len(dataloader)):
            path = save_part(
                output_dir, rank, part_index, gt_buffer, edge_buffer,
                degree_buffer, token_buffer)
            print(
                f"rank={rank} batches={batch_index + 1}/{len(dataloader)} "
                f"samples={processed} edges={len(edge_buffer)} saved={path.name}",
                flush=True,
            )
            gt_buffer.clear()
            edge_buffer.clear()
            degree_buffer.clear()
            token_buffer.clear()
            part_index += 1

    (output_dir / f"done_rank{rank:02d}.json").write_text(
        json.dumps({"rank": rank, "samples": processed, "parts": part_index}),
        encoding="utf-8")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
