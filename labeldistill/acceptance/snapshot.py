"""Capture and compare the tensors that define one distillation training step."""

from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(child, device) for child in value)
    if isinstance(value, list):
        return [_to_device(child, device) for child in value]
    return value


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return {
            field.name: _to_cpu(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _to_cpu(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_cpu(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _flatten_tensors(value: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    if torch.is_tensor(value):
        result[prefix or "tensor"] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_tensors(child, path))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            result.update(_flatten_tensors(child, path))
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def tensor_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    for path, tensor in sorted(_flatten_tensors(snapshot).items()):
        numeric = tensor.detach().float()
        finite = torch.isfinite(numeric)
        finite_values = numeric[finite]
        entry = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _tensor_sha256(tensor),
            "numel": tensor.numel(),
            "finite": int(finite.sum().item()),
        }
        if finite_values.numel():
            entry.update({
                "min": finite_values.min().item(),
                "max": finite_values.max().item(),
                "mean": finite_values.mean().item(),
            })
        summary[path] = entry
    return summary


def batch_fingerprint(batch: Any) -> Dict[str, Any]:
    cpu_batch = _to_cpu(batch)
    tensors = _flatten_tensors(cpu_batch)
    return {
        "tensor_count": len(tensors),
        "tensors": {
            path: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _tensor_sha256(tensor),
            }
            for path, tensor in sorted(tensors.items())
        },
    }


@torch.no_grad()
def capture_training_snapshot(experiment, batch, *, seed: int) -> Dict[str, Any]:
    """Run one training-mode forward and retain all numerical parity gates."""
    if not torch.cuda.is_available():
        raise RuntimeError("Numerical snapshots require CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    # Construction order differs between legacy and builder paths. Reset the
    # forward RNG here so stochastic layers cannot masquerade as config drift.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    experiment = experiment.to(device)
    experiment.train()
    batch = _to_device(batch, device)
    (sweep_imgs, mats, _, _, gt_boxes, gt_labels,
     lidar_pts, depth_labels) = batch

    target_model = (
        experiment.model.module
        if isinstance(experiment.model, torch.nn.parallel.DistributedDataParallel)
        else experiment.model
    )
    bev_mask, bev_box, bev_label, targets = target_model.get_targets(
        gt_boxes, gt_labels)
    outputs = experiment.model(
        bev_mask, bev_box, bev_label, sweep_imgs, mats, lidar_pts)
    student = outputs.student
    teacher = outputs.teacher

    match_result = None
    scaled_match_result = None
    if experiment.feature_mask_type == "gt_heatmap":
        distill_mask = experiment.build_gt_heatmap_mask(targets[0])
    else:
        match_result = experiment.proposal_target_layer(
            teacher.proposals, gt_boxes, gt_labels)
        scaled_match_result = match_result
        if experiment.scaler_enabled:
            scaled_match_result = experiment.change_gt.forward(match_result)
        distill_mask = experiment.generate_bev_mask(
            scaled_match_result, len(gt_boxes))

    detection_loss, response_loss = target_model.response_loss(
        targets, student.raw_preds, teacher.raw_preds)
    if len(depth_labels.shape) == 5:
        depth_labels = depth_labels[:, 0, ...]
    depth_labels = depth_labels.to(student.depth.device)
    depth_loss = experiment.get_depth_loss(depth_labels, student.depth)
    feature_loss_fn = getattr(
        experiment, "get_feature_distill_loss",
        getattr(experiment, "get_feature_distill_loss1", None))
    if feature_loss_fn is None:
        raise AttributeError("Experiment has no feature distillation loss")
    feature_loss = feature_loss_fn(
        teacher.backbone_features, student.distill_features, distill_mask)

    raw_losses = {
        "detection": detection_loss,
        "depth": depth_loss,
        "feature": feature_loss,
        "response": response_loss,
    }
    weights = getattr(experiment, "loss_weights", {
        "detection": 1.0, "depth": 1.0, "feature": 0.6, "response": 1.0,
    })
    weighted_losses = {
        name: loss * float(weights[name]) for name, loss in raw_losses.items()
    }
    total_loss = sum(weighted_losses.values())
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"Non-finite snapshot total loss: {total_loss}")

    return _to_cpu({
        "metadata": {
            "format_version": 1,
            "seed": int(seed),
            "experiment": type(experiment).__module__ + "." + type(experiment).__name__,
            "feature_mask_type": experiment.feature_mask_type,
            "scaler_enabled": bool(experiment.scaler_enabled),
            "loss_weights": {key: float(value) for key, value in weights.items()},
        },
        "student_raw_predictions": student.raw_preds,
        "teacher_raw_predictions": teacher.raw_preds,
        "teacher_proposals": teacher.proposals,
        "match_result": match_result,
        "scaled_match_result": scaled_match_result,
        "scaled_gt": None if scaled_match_result is None else scaled_match_result.gt_boxes,
        "distill_mask": distill_mask,
        "losses_raw": raw_losses,
        "losses_weighted": weighted_losses,
        "total_loss": total_loss,
    })


def save_snapshot(snapshot: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)
    path.with_suffix(".summary.json").write_text(
        json.dumps(tensor_summary(snapshot), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compare_snapshots(left: Dict[str, Any], right: Dict[str, Any], *,
                      atol: float = 1e-5, rtol: float = 1e-5) -> Dict[str, Any]:
    left_tensors = _flatten_tensors(left)
    right_tensors = _flatten_tensors(right)
    paths = sorted(set(left_tensors) | set(right_tensors))
    entries = {}
    passed = True
    for path in paths:
        if path not in left_tensors or path not in right_tensors:
            entries[path] = {"allclose": False, "reason": "missing"}
            passed = False
            continue
        lhs, rhs = left_tensors[path], right_tensors[path]
        if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
            entries[path] = {
                "allclose": False,
                "reason": "shape_or_dtype",
                "left_shape": list(lhs.shape), "right_shape": list(rhs.shape),
                "left_dtype": str(lhs.dtype), "right_dtype": str(rhs.dtype),
            }
            passed = False
            continue
        if lhs.is_floating_point() or lhs.is_complex():
            close = torch.isclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True)
            finite_pair = torch.isfinite(lhs) & torch.isfinite(rhs)
            delta = (lhs - rhs).abs()[finite_pair]
            max_abs = delta.max().item() if delta.numel() else 0.0
            denom = rhs.abs().clamp_min(atol)
            rel = ((lhs - rhs).abs() / denom)[finite_pair]
            max_rel = rel.max().item() if rel.numel() else 0.0
            is_close = bool(close.all().item())
        else:
            is_close = bool(torch.equal(lhs, rhs))
            max_abs = 0.0 if is_close else None
            max_rel = 0.0 if is_close else None
        entries[path] = {
            "allclose": is_close,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "shape": list(lhs.shape),
            "dtype": str(lhs.dtype),
        }
        passed = passed and is_close
    return {
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "tensor_count": len(paths),
        "failed": [path for path, entry in entries.items() if not entry["allclose"]],
        "tensors": entries,
    }
