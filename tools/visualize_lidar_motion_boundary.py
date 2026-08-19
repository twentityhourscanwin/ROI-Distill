#!/usr/bin/env python3
"""Visualize moving-object LiDAR support outside static GT boxes.

The script scans nuScenes samples and saves three images for each selected
moving object:

1. Full BEV scene with all GT boxes and the selected moving GT highlighted.
2. Object crop where LiDAR points inside/outside the original GT are colored.
3. GT-local coordinate view that makes points beyond the box boundary obvious.

Example:
    python tools/visualize_lidar_motion_boundary.py \
        --split val \
        --max-samples 200 \
        --max-candidates 20 \
        --out-dir outputs/vis_lidar_motion_boundary
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn
from labeldistill.exps.nuscenes import base_exp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize moving-object LiDAR points outside GT boxes.")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Dataset split to scan. GT is loaded in both modes.")
    parser.add_argument("--data-root", default="data/nuScenes",
                        help="nuScenes data root.")
    parser.add_argument("--out-dir", default="outputs/vis_lidar_motion_boundary",
                        help="Directory to save visualizations.")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Dataset index to start from.")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum number of dataset samples to scan.")
    parser.add_argument("--indices", default="",
                        help="Comma-separated dataset indices to visualize.")
    parser.add_argument("--max-candidates", type=int, default=20,
                        help="Stop after saving this many object candidates.")
    parser.add_argument("--classes",
                        default="car,truck,construction_vehicle,bus,trailer,motorcycle,bicycle",
                        help="Comma-separated classes to consider.")
    parser.add_argument("--speed-thresh", type=float, default=1.0,
                        help="Minimum GT speed in m/s.")
    parser.add_argument("--outside-ratio-thresh", type=float, default=0.25,
                        help="Minimum near-object point ratio outside GT.")
    parser.add_argument("--motion-outside-ratio-thresh", type=float, default=0.12,
                        help="Minimum outside ratio along the velocity direction.")
    parser.add_argument("--min-near-points", type=int, default=20,
                        help="Minimum nearby LiDAR points around the object.")
    parser.add_argument("--expand-longitudinal", type=float, default=3.0,
                        help="Longitudinal search margin around the GT box in meters.")
    parser.add_argument("--expand-lateral", type=float, default=1.5,
                        help="Lateral search margin around the GT box in meters.")
    parser.add_argument("--z-margin", type=float, default=0.4,
                        help="Vertical margin around the GT box in meters.")
    parser.add_argument("--scene-range", type=float, default=55.0,
                        help="BEV range for the full-scene figure.")
    parser.add_argument("--save-pdf", action="store_true",
                        help="Also save PDF versions.")
    return parser.parse_args()


def deterministic_aug_conf() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ida_aug_conf = copy.deepcopy(base_exp.ida_aug_conf)
    bda_aug_conf = copy.deepcopy(base_exp.bda_aug_conf)

    # Keep GT and LiDAR in the original ego BEV frame.
    ida_aug_conf["resize_lim"] = (max(ida_aug_conf["resize_lim"]),
                                  max(ida_aug_conf["resize_lim"]))
    ida_aug_conf["rot_lim"] = (0.0, 0.0)
    ida_aug_conf["rand_flip"] = False
    bda_aug_conf["rot_lim"] = (0.0, 0.0)
    bda_aug_conf["scale_lim"] = (1.0, 1.0)
    bda_aug_conf["flip_dx_ratio"] = 0.0
    bda_aug_conf["flip_dy_ratio"] = 0.0
    return ida_aug_conf, bda_aug_conf


def build_dataset(args: argparse.Namespace) -> NuscDetDataset:
    ida_aug_conf, bda_aug_conf = deterministic_aug_conf()
    info_path = Path(args.data_root) / (
        "nuscenes_infos_val.pkl"
        if args.split == "val"
        else "nuscenes_infos_train.pkl")

    return NuscDetDataset(
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=base_exp.CLASSES,
        data_root=args.data_root,
        info_paths=str(info_path),
        is_train=True,
        use_cbgs=False,
        img_conf=base_exp.img_conf,
        num_sweeps=1,
        sweep_idxes=[],
        key_idxes=[],
        return_depth=False,
        return_lidar=True,
        use_fusion=False,
    )


def parse_indices(indices_arg: str,
                  start_idx: int,
                  max_samples: int,
                  dataset_len: int) -> List[int]:
    if indices_arg.strip():
        return [int(item) for item in indices_arg.split(",") if item.strip()]
    stop = min(dataset_len, start_idx + max_samples)
    return list(range(start_idx, stop))


def box_corners_bev(box: np.ndarray) -> np.ndarray:
    x, y, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
    local = np.array([
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2, dy / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def draw_box(ax: plt.Axes,
             box: np.ndarray,
             color: str,
             label: Optional[str] = None,
             linestyle: str = "-",
             linewidth: float = 1.8,
             alpha: float = 1.0) -> None:
    corners = box_corners_bev(box)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label)


def set_unique_legend(ax: plt.Axes, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys(), loc=loc, fontsize=8,
                  framealpha=0.92)


def local_xy(points_xy: np.ndarray, box: np.ndarray) -> np.ndarray:
    rel = points_xy - box[:2]
    c, s = np.cos(box[6]), np.sin(box[6])
    rot = np.array([[c, s], [-s, c]])
    return rel @ rot.T


def local_velocity(box: np.ndarray) -> np.ndarray:
    if box.shape[0] < 9:
        return np.zeros(2, dtype=np.float32)
    vx, vy = box[7], box[8]
    c, s = np.cos(box[6]), np.sin(box[6])
    rot = np.array([[c, s], [-s, c]])
    return np.array([vx, vy]) @ rot.T


def in_original_box(local: np.ndarray, box: np.ndarray) -> np.ndarray:
    return ((np.abs(local[:, 0]) <= box[3] / 2)
            & (np.abs(local[:, 1]) <= box[4] / 2))


def in_search_region(local: np.ndarray,
                     box: np.ndarray,
                     expand_longitudinal: float,
                     expand_lateral: float) -> np.ndarray:
    return ((np.abs(local[:, 0]) <= box[3] / 2 + expand_longitudinal)
            & (np.abs(local[:, 1]) <= box[4] / 2 + expand_lateral))


def in_height_region(points_z: np.ndarray,
                     box: np.ndarray,
                     z_margin: float) -> np.ndarray:
    return np.abs(points_z - box[2]) <= box[5] / 2 + z_margin


def outside_along_motion(local: np.ndarray,
                         box: np.ndarray,
                         velocity_local: np.ndarray) -> np.ndarray:
    speed = np.linalg.norm(velocity_local)
    if speed < 1e-6:
        return np.zeros(local.shape[0], dtype=bool)
    direction = velocity_local / speed
    projection = local @ direction
    half_extent = (
        abs(direction[0]) * box[3] / 2
        + abs(direction[1]) * box[4] / 2
    )
    return np.abs(projection) > half_extent


def choose_motion_candidate(points: np.ndarray,
                            gt_boxes: torch.Tensor,
                            gt_labels: torch.Tensor,
                            args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    points_xy = points[:, :2]
    points_z = points[:, 2]
    gt_np = gt_boxes.detach().cpu().numpy()
    labels_np = gt_labels.detach().cpu().numpy()
    allowed_classes = {
        item.strip() for item in args.classes.split(",") if item.strip()
    }
    best: Optional[Dict[str, Any]] = None

    for gt_idx, box in enumerate(gt_np):
        class_name = base_exp.CLASSES[int(labels_np[gt_idx])]
        if allowed_classes and class_name not in allowed_classes:
            continue
        if box.shape[0] < 9:
            continue
        speed = float(np.linalg.norm(box[7:9]))
        if speed < args.speed_thresh:
            continue

        local = local_xy(points_xy, box)
        search_mask = (
            in_search_region(
                local, box, args.expand_longitudinal, args.expand_lateral)
            & in_height_region(points_z, box, args.z_margin)
        )
        near_count = int(search_mask.sum())
        if near_count < args.min_near_points:
            continue

        local_near = local[search_mask]
        inside = in_original_box(local_near, box)
        outside = ~inside
        outside_ratio = float(outside.sum() / max(near_count, 1))

        vel_local = local_velocity(box)
        motion_outside = outside & outside_along_motion(local_near, box, vel_local)
        motion_outside_ratio = float(motion_outside.sum() / max(near_count, 1))

        if outside_ratio < args.outside_ratio_thresh:
            continue
        if motion_outside_ratio < args.motion_outside_ratio_thresh:
            continue

        # Prefer fast objects with many outside points along the motion axis.
        score = (
            outside_ratio
            + 1.5 * motion_outside_ratio
            + 0.04 * speed
            + min(near_count, 300) / 3000.0
        )
        candidate = {
            "gt_index": int(gt_idx),
            "class_name": class_name,
            "speed": speed,
            "near_points": near_count,
            "inside_points": int(inside.sum()),
            "outside_points": int(outside.sum()),
            "outside_ratio": outside_ratio,
            "motion_outside_points": int(motion_outside.sum()),
            "motion_outside_ratio": motion_outside_ratio,
            "score": float(score),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def draw_velocity_arrow(ax: plt.Axes,
                        box: np.ndarray,
                        scale: float = 1.0,
                        label: str = "velocity") -> None:
    if box.shape[0] < 9:
        return
    vx, vy = box[7], box[8]
    if np.linalg.norm([vx, vy]) < 1e-6:
        return
    ax.arrow(box[0], box[1], vx * scale, vy * scale,
             color="yellow", width=0.06, length_includes_head=True,
             label=label)


def crop_limits(box: np.ndarray, pad: float = 8.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    corners = box_corners_bev(box)
    return ((float(corners[:, 0].min() - pad), float(corners[:, 0].max() + pad)),
            (float(corners[:, 1].min() - pad), float(corners[:, 1].max() + pad)))


def save_candidate(idx: int,
                   token: str,
                   points: np.ndarray,
                   gt_boxes: torch.Tensor,
                   candidate: Dict[str, Any],
                   out_dir: Path,
                   save_pdf: bool,
                   scene_range: float,
                   expand_longitudinal: float,
                   expand_lateral: float,
                   z_margin: float) -> Dict[str, Any]:
    gt_np = gt_boxes.detach().cpu().numpy()
    box = gt_np[candidate["gt_index"]]
    local = local_xy(points[:, :2], box)
    search_mask = (
        in_search_region(local, box, expand_longitudinal, expand_lateral)
        & in_height_region(points[:, 2], box, z_margin)
    )
    inside_mask = np.zeros(points.shape[0], dtype=bool)
    inside_mask[search_mask] = in_original_box(local[search_mask], box)
    outside_near_mask = search_mask & ~inside_mask

    stem = f"sample_{idx:06d}_{token[:8]}_gt{candidate['gt_index']:03d}"
    paths: Dict[str, str] = {}

    # 1. Full scene BEV.
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    ax.scatter(points[:, 0], points[:, 1], s=0.12, c="0.25", alpha=0.45,
               linewidths=0, label="LiDAR points")
    for gt_box in gt_np:
        draw_box(ax, gt_box, "red", "GT", linewidth=0.9, alpha=0.55)
    draw_box(ax, box, "cyan", "selected moving GT", linewidth=2.6)
    draw_velocity_arrow(ax, box, scale=1.0)
    ax.set_title("Full BEV: selected moving object")
    ax.set_aspect("equal")
    ax.set_xlim(-scene_range, scene_range)
    ax.set_ylim(-scene_range, scene_range)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    set_unique_legend(ax)
    scene_path = out_dir / f"{stem}_01_scene_bev.png"
    fig.savefig(scene_path, dpi=220)
    if save_pdf:
        fig.savefig(out_dir / f"{stem}_01_scene_bev.pdf")
    plt.close(fig)
    paths["scene_bev"] = str(scene_path)

    # 2. Object crop in ego BEV.
    xlim, ylim = crop_limits(box)
    crop_mask = ((points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1])
                 & (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1]))
    background_mask = crop_mask & ~search_mask
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    ax.scatter(points[background_mask, 0], points[background_mask, 1],
               s=1.0, c="0.75", alpha=0.45, linewidths=0,
               label="nearby background points")
    ax.scatter(points[inside_mask, 0], points[inside_mask, 1],
               s=6.0, c="#1f77b4", alpha=0.9, linewidths=0,
               label="points inside GT")
    ax.scatter(points[outside_near_mask, 0], points[outside_near_mask, 1],
               s=8.0, c="#ff7f0e", alpha=0.95, linewidths=0,
               label="near-object points outside GT")
    draw_box(ax, box, "red", "static GT boundary", linestyle="--",
             linewidth=2.2)
    draw_velocity_arrow(ax, box, scale=1.0)
    ax.text(
        0.02, 0.98,
        f"{candidate['class_name']}\n"
        f"speed={candidate['speed']:.2f} m/s\n"
        f"outside={candidate['outside_ratio']:.2f}\n"
        f"motion-outside={candidate['motion_outside_ratio']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    ax.set_title("Object crop: LiDAR points exceed static GT")
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    set_unique_legend(ax)
    crop_path = out_dir / f"{stem}_02_object_crop.png"
    fig.savefig(crop_path, dpi=220)
    if save_pdf:
        fig.savefig(out_dir / f"{stem}_02_object_crop.pdf")
    plt.close(fig)
    paths["object_crop"] = str(crop_path)

    # 3. GT-local coordinate view.
    local_search = local[search_mask]
    local_inside = in_original_box(local_search, box)
    vel_local = local_velocity(box)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
    ax.scatter(local_search[local_inside, 0], local_search[local_inside, 1],
               s=8.0, c="#1f77b4", alpha=0.9, linewidths=0,
               label="inside GT")
    ax.scatter(local_search[~local_inside, 0], local_search[~local_inside, 1],
               s=10.0, c="#ff7f0e", alpha=0.95, linewidths=0,
               label="outside GT")
    rect_x = [-box[3] / 2, box[3] / 2, box[3] / 2, -box[3] / 2, -box[3] / 2]
    rect_y = [-box[4] / 2, -box[4] / 2, box[4] / 2, box[4] / 2, -box[4] / 2]
    ax.plot(rect_x, rect_y, color="red", linestyle="--", linewidth=2.2,
            label="static GT boundary")
    if np.linalg.norm(vel_local) > 1e-6:
        ax.arrow(0.0, 0.0, vel_local[0], vel_local[1],
                 color="black", width=0.035, length_includes_head=True,
                 label="velocity in GT-local frame")
    ax.axhline(0, color="0.5", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="0.5", linewidth=0.6, alpha=0.5)
    ax.set_title("GT-local view: points outside boundary")
    ax.set_xlabel("local x / length direction (m)")
    ax.set_ylabel("local y / width direction (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-box[3] / 2 - expand_longitudinal,
                box[3] / 2 + expand_longitudinal)
    ax.set_ylim(-box[4] / 2 - expand_lateral,
                box[4] / 2 + expand_lateral)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    set_unique_legend(ax, loc="best")
    local_path = out_dir / f"{stem}_03_gt_local_boundary.png"
    fig.savefig(local_path, dpi=220)
    if save_pdf:
        fig.savefig(out_dir / f"{stem}_03_gt_local_boundary.pdf")
    plt.close(fig)
    paths["gt_local_boundary"] = str(local_path)

    return {
        "idx": idx,
        "token": token,
        "paths": paths,
        "candidate": candidate,
    }


def run_one_sample(dataset: NuscDetDataset,
                   idx: int,
                   args: argparse.Namespace,
                   out_dir: Path) -> Optional[Dict[str, Any]]:
    sample = dataset[idx]
    batch = collate_fn([sample], is_return_lidar=True, is_return_depth=False)
    _imgs, _mats, _timestamps, img_metas, gt_boxes, gt_labels, lidar_pts = batch

    if len(gt_boxes[0]) == 0:
        return None

    points = lidar_pts[0, 0].detach().float().cpu().numpy()
    valid = points[:, 0] > -900
    points = points[valid]
    if len(points) == 0:
        return None

    candidate = choose_motion_candidate(points, gt_boxes[0], gt_labels[0], args)
    if candidate is None:
        return None

    return save_candidate(
        idx=idx,
        token=img_metas[0].get("token", f"idx{idx}"),
        points=points,
        gt_boxes=gt_boxes[0],
        candidate=candidate,
        out_dir=out_dir,
        save_pdf=args.save_pdf,
        scene_range=args.scene_range,
        expand_longitudinal=args.expand_longitudinal,
        expand_lateral=args.expand_lateral,
        z_margin=args.z_margin)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    indices = parse_indices(args.indices, args.start_idx, args.max_samples,
                            len(dataset))

    print(f"[INFO] Scanning {len(indices)} samples from split={args.split}.")
    print(f"[INFO] Saving motion-boundary figures to {out_dir}.")

    summary: List[Dict[str, Any]] = []
    for idx in indices:
        try:
            item = run_one_sample(dataset, idx, args, out_dir)
        except Exception as exc:
            print(f"[WARN] idx={idx} failed: {exc}")
            continue
        if item is None:
            continue
        summary.append(item)
        print(
            "[INFO] saved idx={idx} gt={gt} outside={outside:.2f}: {path}"
            .format(
                idx=idx,
                gt=item["candidate"]["gt_index"],
                outside=item["candidate"]["outside_ratio"],
                path=item["paths"]["gt_local_boundary"]))
        if len(summary) >= args.max_candidates:
            break

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Done. Saved {len(summary)} candidates.")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
