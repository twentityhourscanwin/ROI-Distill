#!/usr/bin/env python3
"""Visualize teacher BEV heatmap mismatch with GT regions.

This script searches nuScenes samples for paper-friendly cases where a GT box
exists but the teacher BEV heatmap has weak activation in that GT region.

For each selected sample it saves exactly three views:

1. GT BEV: LiDAR points with GT boxes.
2. Teacher BEV heatmap.
3. Overlay: GT boxes drawn on the teacher BEV heatmap.

Example:
    python tools/visualize_roi_distill_evidence.py \
        --split val \
        --max-samples 200 \
        --max-figures 20 \
        --out-dir outputs/vis_evidence
"""

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.models.utils.gaussian import draw_heatmap_gaussian, gaussian_radius


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labeldistill.datasets.nusc_det_dataset_lidar import NuscDetDataset, collate_fn


DEFAULT_EXP = (
    "labeldistill.exps.nuscenes.ablation_param.param_P3_lambda_055:"
    "LabelDistillModel"
)

PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
VOXEL_SIZE = (0.1, 0.1, 0.2)
OUT_FACTOR = 8
FEATURE_MAP_SIZE = (128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and visualize GT/teacher heatmap mismatch cases.")
    parser.add_argument("--exp", default=DEFAULT_EXP,
                        help="Experiment class as module.path:ClassName.")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Which info split to scan. GT is still loaded.")
    parser.add_argument("--out-dir", default="outputs/vis_evidence",
                        help="Directory to save figures and summary json.")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Maximum number of dataset samples to scan.")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Dataset index to start scanning from.")
    parser.add_argument("--max-figures", type=int, default=20,
                        help="Stop after saving this many candidate samples.")
    parser.add_argument("--indices", default="",
                        help="Comma-separated dataset indices to visualize.")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for the model forward pass.")
    parser.add_argument("--activation-thresh", type=float, default=0.15,
                        help="Teacher heatmap max below this is a mismatch.")
    parser.add_argument("--save-pdf", action="store_true",
                        help="Also save PDF figures.")
    return parser.parse_args()


def import_exp_class(exp_spec: str):
    module_name, class_name = exp_spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def deterministic_aug_conf(exp: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ida_aug_conf = copy.deepcopy(exp.ida_aug_conf)
    bda_aug_conf = copy.deepcopy(exp.bda_aug_conf)

    # Keep GT and LiDAR in a stable ego frame for visualization.
    ida_aug_conf["resize_lim"] = (max(ida_aug_conf["resize_lim"]),
                                  max(ida_aug_conf["resize_lim"]))
    ida_aug_conf["rot_lim"] = (0.0, 0.0)
    ida_aug_conf["rand_flip"] = False
    bda_aug_conf["rot_lim"] = (0.0, 0.0)
    bda_aug_conf["scale_lim"] = (1.0, 1.0)
    bda_aug_conf["flip_dx_ratio"] = 0.0
    bda_aug_conf["flip_dy_ratio"] = 0.0
    return ida_aug_conf, bda_aug_conf


def build_dataset(exp: Any, split: str) -> NuscDetDataset:
    ida_aug_conf, bda_aug_conf = deterministic_aug_conf(exp)
    info_paths = exp.val_info_paths if split == "val" else exp.train_info_paths
    return NuscDetDataset(
        ida_aug_conf=ida_aug_conf,
        bda_aug_conf=bda_aug_conf,
        classes=exp.class_names,
        data_root=exp.data_root,
        info_paths=info_paths,
        is_train=True,
        use_cbgs=False,
        img_conf=exp.img_conf,
        num_sweeps=exp.num_sweeps,
        sweep_idxes=exp.sweep_idxes,
        key_idxes=exp.key_idxes,
        return_depth=True,
        return_lidar=True,
        use_fusion=exp.use_fusion,
    )


def move_batch_to_device(batch: Sequence[Any],
                         device: torch.device) -> Tuple[Any, ...]:
    sweep_imgs, mats, timestamps, img_metas, gt_boxes, gt_labels, lidar_pts, depth_labels = batch
    sweep_imgs = sweep_imgs.to(device)
    timestamps = timestamps.to(device)
    lidar_pts = lidar_pts.to(device)
    depth_labels = depth_labels.to(device)
    mats = {key: value.to(device) for key, value in mats.items()}
    gt_boxes = [box.to(device) for box in gt_boxes]
    gt_labels = [label.to(device) for label in gt_labels]
    return (sweep_imgs, mats, timestamps, img_metas, gt_boxes, gt_labels,
            lidar_pts, depth_labels)


def extract_teacher_heatmap(lidar_preds: Any, batch_idx: int = 0) -> np.ndarray:
    heatmaps: List[torch.Tensor] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "heatmap" and torch.is_tensor(value) and value.dim() == 4:
                    heatmaps.append(value)
                else:
                    visit(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                visit(item)

    visit(lidar_preds)
    if not heatmaps:
        raise RuntimeError("Could not find teacher heatmap in lidar_preds.")

    target_hw = heatmaps[0].shape[-2:]
    aligned = []
    for heatmap in heatmaps:
        if heatmap.shape[-2:] != target_hw:
            heatmap = F.interpolate(
                heatmap, size=target_hw, mode="bilinear", align_corners=False)
        aligned.append(heatmap.sigmoid()[batch_idx].detach().float().cpu())
    return torch.cat(aligned, dim=0).max(dim=0)[0].numpy()


def draw_plain_gt_mask(gt_boxes: torch.Tensor) -> np.ndarray:
    mask = torch.zeros((1, FEATURE_MAP_SIZE[1], FEATURE_MAP_SIZE[0]),
                       dtype=torch.float32,
                       device=gt_boxes.device)
    pc_range = torch.tensor(PC_RANGE, dtype=torch.float32, device=gt_boxes.device)
    voxel_size = torch.tensor(VOXEL_SIZE, dtype=torch.float32, device=gt_boxes.device)
    heatmap = mask[0]

    for box in gt_boxes:
        width = box[3]
        length = box[4]
        width_fm = width / voxel_size[0] / OUT_FACTOR
        length_fm = length / voxel_size[1] / OUT_FACTOR
        if width_fm.item() <= 0 or length_fm.item() <= 0:
            continue
        radius = gaussian_radius((length_fm, width_fm), min_overlap=0.1)
        radius = max(2, int(radius.item()))
        coor_x = (box[0] - pc_range[0]) / voxel_size[0] / OUT_FACTOR
        coor_y = (box[1] - pc_range[1]) / voxel_size[1] / OUT_FACTOR
        center_int = torch.stack([coor_x, coor_y], dim=0).to(torch.int32)
        if (0 <= center_int[0] < FEATURE_MAP_SIZE[0]
                and 0 <= center_int[1] < FEATURE_MAP_SIZE[1]):
            heatmap = draw_heatmap_gaussian(heatmap, center_int, radius, k=1.0)
    return mask[0].detach().cpu().numpy()


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


def local_xy(points_xy: np.ndarray, box: np.ndarray) -> np.ndarray:
    rel = points_xy - box[:2]
    c, s = np.cos(box[6]), np.sin(box[6])
    rot = np.array([[c, s], [-s, c]])
    return rel @ rot.T


def points_in_box(points_xy: np.ndarray,
                  box: np.ndarray,
                  margin: float = 0.0) -> np.ndarray:
    loc = local_xy(points_xy, box)
    return ((np.abs(loc[:, 0]) <= box[3] / 2 + margin)
            & (np.abs(loc[:, 1]) <= box[4] / 2 + margin))


def activation_in_box(activation: np.ndarray, box: np.ndarray) -> Dict[str, float]:
    h, w = activation.shape
    xs = (np.arange(w) + 0.5) * VOXEL_SIZE[0] * OUT_FACTOR + PC_RANGE[0]
    ys = (np.arange(h) + 0.5) * VOXEL_SIZE[1] * OUT_FACTOR + PC_RANGE[1]
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
    inside = points_in_box(pts, box).reshape(h, w)
    if not inside.any():
        cx = int(np.clip((box[0] - PC_RANGE[0]) / VOXEL_SIZE[0] / OUT_FACTOR,
                         0, w - 1))
        cy = int(np.clip((box[1] - PC_RANGE[1]) / VOXEL_SIZE[1] / OUT_FACTOR,
                         0, h - 1))
        patch = activation[max(0, cy - 1):min(h, cy + 2),
                           max(0, cx - 1):min(w, cx + 2)]
        return {"max": float(patch.max()), "mean": float(patch.mean())}
    values = activation[inside]
    return {"max": float(values.max()), "mean": float(values.mean())}


def choose_mismatch_candidate(activation: np.ndarray,
                              gt_boxes: torch.Tensor,
                              gt_labels: torch.Tensor,
                              class_names: Sequence[str],
                              activation_thresh: float) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    gt_np = gt_boxes.detach().cpu().numpy()
    labels_np = gt_labels.detach().cpu().numpy()
    for i, box in enumerate(gt_np):
        stats = activation_in_box(activation, box)
        if stats["max"] >= activation_thresh:
            continue
        dist = float(np.linalg.norm(box[:2]))
        score = (activation_thresh - stats["max"]) + 0.002 * dist
        cand = {
            "gt_index": int(i),
            "class_name": class_names[int(labels_np[i])],
            "activation_max": stats["max"],
            "activation_mean": stats["mean"],
            "distance": dist,
            "score": float(score),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def choose_motion_candidate(points_xy: np.ndarray,
                            gt_boxes: torch.Tensor,
                            gt_labels: torch.Tensor,
                            scaled_results: Dict[str, Any],
                            class_names: Sequence[str],
                            speed_thresh: float,
                            outside_ratio_thresh: float,
                            min_points: int) -> Optional[Dict[str, Any]]:
    scaled_map = build_scaled_gt_map(scaled_results)
    best: Optional[Dict[str, Any]] = None
    gt_np = gt_boxes.detach().cpu().numpy()
    labels_np = gt_labels.detach().cpu().numpy()

    for i, box in enumerate(gt_np):
        if box.shape[0] < 9:
            continue
        speed = float(np.linalg.norm(box[7:9]))
        if speed < speed_thresh:
            continue
        near_mask = points_in_box(points_xy, box, margin=1.5)
        near_count = int(near_mask.sum())
        if near_count < min_points:
            continue
        inside_count = int(points_in_box(points_xy[near_mask], box).sum())
        outside_ratio = float((near_count - inside_count) / max(near_count, 1))
        if outside_ratio < outside_ratio_thresh:
            continue

        scaled_box = scaled_map.get((round(float(box[0]), 3),
                                     round(float(box[1]), 3),
                                     int(labels_np[i])))
        area_gain = 0.0
        if scaled_box is not None:
            area_gain = float(
                (scaled_box[3] * scaled_box[4]) / max(box[3] * box[4], 1e-6)
                - 1.0)
        score = outside_ratio + 0.05 * speed + 0.1 * area_gain
        cand = {
            "gt_index": int(i),
            "class_name": class_names[int(labels_np[i])],
            "speed": speed,
            "near_points": near_count,
            "outside_ratio": outside_ratio,
            "area_gain": area_gain,
            "score": float(score),
            "scaled_box": scaled_box,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def iter_result_boxes(items: Iterable[torch.Tensor]) -> Iterable[np.ndarray]:
    for tensor in items:
        if tensor is None or len(tensor) == 0:
            continue
        for box in tensor.detach().cpu().numpy():
            yield box


def build_scaled_gt_map(scaled_results: Dict[str, Any]) -> Dict[Tuple[float, float, int], np.ndarray]:
    out: Dict[Tuple[float, float, int], np.ndarray] = {}
    groups = [
        scaled_results["refined_high_quality_gt"][0],
        scaled_results["medium_quality_gt"][0],
        scaled_results["unmatched_gt"][0],
    ]
    for box in iter_result_boxes(groups):
        if box.shape[0] < 10:
            continue
        key = (round(float(box[0]), 3), round(float(box[1]), 3), int(box[-1]))
        out[key] = box
    return out


def draw_scene_points(ax: plt.Axes,
                      points: np.ndarray,
                      title: str,
                      xlim: Tuple[float, float] = (-50, 50),
                      ylim: Tuple[float, float] = (-50, 50)) -> None:
    if points.size:
        color = points[:, 3] if points.shape[1] > 3 else "0.2"
        ax.scatter(points[:, 0], points[:, 1], s=0.15, c=color,
                   cmap="gray", alpha=0.6, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linewidth=0.3, alpha=0.25)


def show_heatmap(ax: plt.Axes,
                 heatmap: np.ndarray,
                 title: str,
                 cmap: str = "magma",
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None) -> None:
    ax.imshow(heatmap,
              extent=[PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4]],
              origin="lower",
              cmap=cmap,
              alpha=0.92,
              vmin=vmin,
              vmax=vmax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.grid(True, linewidth=0.3, alpha=0.2)


def draw_quality_boxes(ax: plt.Axes, matched_results: Dict[str, Any]) -> None:
    for box in iter_result_boxes([matched_results["refined_high_quality_gt"][0]]):
        draw_box(ax, box, "#2ca02c", "high GT", linewidth=1.5)
    for box in iter_result_boxes([matched_results["medium_quality_gt"][0]]):
        draw_box(ax, box, "#ff7f0e", "medium GT", linewidth=1.5)
    for box in iter_result_boxes([matched_results["unmatched_gt"][0]]):
        draw_box(ax, box, "#d62728", "unmatched GT", linewidth=1.5)
    for box in iter_result_boxes([matched_results["refined_high_quality_rois"][0]]):
        draw_box(ax, box, "#17becf", "teacher ROI", linestyle="--", linewidth=1.2)
    for box in iter_result_boxes([matched_results["medium_quality_rois"][0]]):
        draw_box(ax, box, "#9467bd", "medium ROI", linestyle="--", linewidth=1.2)


def set_unique_legend(ax: plt.Axes, loc: str = "upper right") -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys(), loc=loc, fontsize=7,
                  framealpha=0.9)


def crop_limits(box: np.ndarray, pad: float = 6.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    corners = box_corners_bev(box)
    return ((float(corners[:, 0].min() - pad), float(corners[:, 0].max() + pad)),
            (float(corners[:, 1].min() - pad), float(corners[:, 1].max() + pad)))


def save_three_views(idx: int,
                     token: str,
                     points: np.ndarray,
                     gt_boxes: torch.Tensor,
                     activation: np.ndarray,
                     mismatch: Dict[str, Any],
                     out_dir: Path,
                     save_pdf: bool) -> Dict[str, Any]:
    gt_np = gt_boxes.detach().cpu().numpy()
    selected_box = gt_np[mismatch["gt_index"]]
    stem = f"sample_{idx:06d}_{token[:8]}"
    saved_paths: Dict[str, str] = {}

    # 1. GT BEV.
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    draw_scene_points(ax, points, "GT BEV: LiDAR points + GT boxes")
    for box in gt_np:
        draw_box(ax, box, "red", "GT", linewidth=1.0, alpha=0.65)
    draw_box(ax, selected_box, "yellow", "selected weak-response GT",
             linewidth=2.6)
    set_unique_legend(ax)
    gt_path = out_dir / f"{stem}_01_gt_bev.png"
    fig.savefig(gt_path, dpi=220)
    if save_pdf:
        fig.savefig(out_dir / f"{stem}_01_gt_bev.pdf")
    plt.close(fig)
    saved_paths["gt_bev"] = str(gt_path)

    # 2. Teacher BEV heatmap.
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    show_heatmap(ax, activation, "Teacher BEV heatmap", "magma")
    heatmap_path = out_dir / f"{stem}_02_teacher_heatmap.png"
    fig.savefig(heatmap_path, dpi=220)
    if save_pdf:
        fig.savefig(out_dir / f"{stem}_02_teacher_heatmap.pdf")
    plt.close(fig)
    saved_paths["teacher_heatmap"] = str(heatmap_path)

    # 3. GT + teacher BEV heatmap overlay.
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    show_heatmap(ax, activation, "GT over teacher BEV heatmap", "magma")
    for box in gt_np:
        draw_box(ax, box, "white", "GT", linewidth=1.0, alpha=0.75)
    draw_box(ax, selected_box, "cyan", "selected weak-response GT",
             linewidth=2.8)
    ax.text(
        0.02, 0.98,
        f"{mismatch['class_name']}\n"
        f"teacher max={mismatch['activation_max']:.3f}\n"
        f"teacher mean={mismatch['activation_mean']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
    set_unique_legend(ax)
    overlay_path = out_dir / f"{stem}_03_gt_heatmap_overlay.png"
    fig.savefig(overlay_path, dpi=220)
    if save_pdf:
        fig.savefig(out_dir / f"{stem}_03_gt_heatmap_overlay.pdf")
    plt.close(fig)
    saved_paths["gt_heatmap_overlay"] = str(overlay_path)

    return {
        "idx": idx,
        "token": token,
        "paths": saved_paths,
        "mismatch": mismatch,
    }


def run_one_sample(exp: Any,
                   dataset: NuscDetDataset,
                   idx: int,
                   device: torch.device,
                   args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    sample = dataset[idx]
    batch = collate_fn([sample], is_return_lidar=True, is_return_depth=True)
    (sweep_imgs, mats, timestamps, img_metas, gt_boxes, gt_labels, lidar_pts,
     _depth_labels) = move_batch_to_device(batch, device)

    if len(gt_boxes[0]) == 0:
        return None

    with torch.no_grad():
        bev_mask, bev_box, bev_label, _targets = exp.model.get_targets(
            gt_boxes, gt_labels)
        outputs = exp.model(
            bev_mask, bev_box, bev_label, sweep_imgs, mats, lidar_pts,
            timestamps=timestamps)
        lidar_preds = outputs[1]

    activation = extract_teacher_heatmap(lidar_preds)

    points = lidar_pts[0, 0].detach().float().cpu().numpy()
    valid = points[:, 0] > -900
    points = points[valid]

    mismatch = choose_mismatch_candidate(
        activation=activation,
        gt_boxes=gt_boxes[0],
        gt_labels=gt_labels[0],
        class_names=exp.class_names,
        activation_thresh=args.activation_thresh)

    if mismatch is None:
        return None

    return save_three_views(
        idx=idx,
        token=img_metas[0].get("token", f"idx{idx}"),
        points=points,
        gt_boxes=gt_boxes[0],
        activation=activation,
        mismatch=mismatch,
        out_dir=Path(args.out_dir),
        save_pdf=args.save_pdf)


def parse_indices(indices_arg: str,
                  start_idx: int,
                  max_samples: int,
                  dataset_len: int) -> List[int]:
    if indices_arg.strip():
        return [int(item) for item in indices_arg.split(",") if item.strip()]
    stop = min(dataset_len, start_idx + max_samples)
    return list(range(start_idx, stop))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    exp_cls = import_exp_class(args.exp)
    exp = exp_cls(batch_size_per_device=1)
    exp.model.to(device)
    exp.model.train()
    exp.model.centerpoint.eval()
    if hasattr(exp, "centerpoint"):
        exp.centerpoint.to(device)
        exp.centerpoint.eval()

    dataset = build_dataset(exp, args.split)
    indices = parse_indices(args.indices, args.start_idx, args.max_samples,
                            len(dataset))

    print(f"[INFO] Scanning {len(indices)} samples from split={args.split}.")
    print(f"[INFO] Saving figures to {out_dir}.")

    summary: List[Dict[str, Any]] = []
    for idx in indices:
        try:
            item = run_one_sample(exp, dataset, idx, device, args)
        except Exception as exc:
            print(f"[WARN] idx={idx} failed: {exc}")
            continue
        if item is None:
            continue
        summary.append(item)
        print(f"[INFO] saved idx={idx}: {item['paths']['gt_heatmap_overlay']}")
        if len(summary) >= args.max_figures:
            break

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Done. Saved {len(summary)} candidate samples.")
    print(f"[INFO] Summary: {summary_path}")


if __name__ == "__main__":
    main()
