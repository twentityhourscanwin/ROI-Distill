"""Summarize exact-class center-value matching from extracted teacher parts.

The input parts are produced by ``analyze_teacher_gt_matching.py``.  This
script applies the current baseline offline without another teacher forward:
official class ranges, small <1m / large <2m, score-first nearest-unmatched
one-to-one matching, and q=max(0, 1-(d/tau)^2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


CLASS_NAMES = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
SMALL_CLASSES = {
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
}
OFFICIAL_RANGES = {
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--output-name", default="center_value_one_to_one_summary")
    return parser.parse_args()


def _quantiles(values):
    if not values:
        return {
            key: None for key in
            ("mean", "min", "p10", "p25", "p50", "p75", "p90", "max")
        }
    tensor = torch.tensor(values, dtype=torch.float64)
    points = torch.tensor(
        [0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float64)
    result = torch.quantile(tensor, points)
    return {
        "mean": float(tensor.mean()),
        "min": float(tensor.min()),
        "p10": float(result[0]),
        "p25": float(result[1]),
        "p50": float(result[2]),
        "p75": float(result[3]),
        "p90": float(result[4]),
        "max": float(tensor.max()),
    }


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parts(input_dir):
    paths = sorted(input_dir.glob("part_rank*_*.pt"))
    if not paths:
        raise RuntimeError(f"No extraction parts found in {input_dir}")
    gt_parts = []
    candidate_parts = []
    gt_columns = None
    candidate_columns = None
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        current_gt_columns = tuple(payload["gt_columns"])
        current_candidate_columns = tuple(payload["candidate_columns"])
        if gt_columns is None:
            gt_columns = current_gt_columns
            candidate_columns = current_candidate_columns
        if current_gt_columns != gt_columns:
            raise RuntimeError(f"GT column mismatch in {path}")
        if current_candidate_columns != candidate_columns:
            raise RuntimeError(f"Candidate column mismatch in {path}")
        gt_parts.append(payload["gt"])
        candidate_parts.append(payload["candidates"])
    return (
        torch.cat(gt_parts),
        torch.cat(candidate_parts),
        {name: index for index, name in enumerate(gt_columns)},
        {name: index for index, name in enumerate(candidate_columns)},
        paths,
    )


def _load_effective_gt(input_dir, gt, gi):
    range_path = input_dir / "gt_ego_ranges.pt"
    if not range_path.exists():
        raise RuntimeError(
            f"Missing {range_path}; run analyze_teacher_gt_matching.py "
            "--summarize-only --official-ranges first")
    payload = torch.load(range_path, map_location="cpu")
    columns = tuple(payload["columns"])
    expected = ("sample_index", "gt_index", "gt_label", "ego_distance")
    if columns != expected:
        raise RuntimeError(
            f"GT range schema mismatch: expected={expected}, got={columns}")
    ego_ranges = {
        (int(row[0]), int(row[1])): (int(row[2]), float(row[3]))
        for row in payload["rows"]
    }
    records = {}
    for row in gt:
        key = (
            int(row[gi["sample_index"]]),
            int(row[gi["gt_index"]]),
        )
        label = int(row[gi["gt_label"]])
        range_label, ego_distance = ego_ranges[key]
        if range_label != label:
            raise RuntimeError(f"GT label mismatch for key={key}")
        class_name = CLASS_NAMES[label]
        if ego_distance <= OFFICIAL_RANGES[class_name]:
            records[key] = {
                "sample_index": key[0],
                "gt_index": key[1],
                "label": label,
                "class_name": class_name,
                "group": "small" if class_name in SMALL_CLASSES else "large",
                "tau": 1.0 if class_name in SMALL_CLASSES else 2.0,
                "ego_distance": ego_distance,
            }
    return records


def _build_edges(candidates, ci, gt_records):
    edges_by_gt = defaultdict(list)
    edges_by_sample_class_pred = defaultdict(lambda: defaultdict(list))
    prediction_scores = {}
    for row in candidates:
        key = (
            int(row[ci["sample_index"]]),
            int(row[ci["gt_index"]]),
        )
        record = gt_records.get(key)
        if record is None:
            continue
        proposal_label = int(row[ci["proposal_label"]])
        if proposal_label != record["label"]:
            continue
        distance = float(row[ci["center_distance"]])
        if not distance < record["tau"]:
            continue
        proposal_index = int(row[ci["proposal_index"]])
        score = float(row[ci["teacher_score"]])
        edge = {
            "gt_key": key,
            "proposal_index": proposal_index,
            "distance": distance,
            "score": score,
        }
        edges_by_gt[key].append(edge)
        scope = (key[0], record["label"])
        edges_by_sample_class_pred[scope][proposal_index].append(edge)
        score_key = (scope, proposal_index)
        previous = prediction_scores.setdefault(score_key, score)
        if abs(previous - score) > 1e-8:
            raise RuntimeError(
                f"Inconsistent score for proposal {score_key}: "
                f"{previous} vs {score}")
    return edges_by_gt, edges_by_sample_class_pred, prediction_scores


def _assign_one_to_one(edges_by_sample_class_pred, prediction_scores):
    selected_by_gt = {}
    for scope, predictions in edges_by_sample_class_pred.items():
        ordered = sorted(
            predictions,
            key=lambda proposal_index: (
                -prediction_scores[(scope, proposal_index)], proposal_index),
        )
        occupied_gt = set()
        for proposal_index in ordered:
            available = [
                edge for edge in predictions[proposal_index]
                if edge["gt_key"] not in occupied_gt
            ]
            if not available:
                continue
            selected = min(
                available,
                key=lambda edge: (edge["distance"], edge["gt_key"][1]),
            )
            occupied_gt.add(selected["gt_key"])
            selected_by_gt[selected["gt_key"]] = selected
    return selected_by_gt


def _pre_assignment_choices(edges_by_gt):
    highest_score = {}
    nearest = {}
    for key, edges in edges_by_gt.items():
        highest_score[key] = min(
            edges,
            key=lambda edge: (-edge["score"], edge["proposal_index"]),
        )
        nearest[key] = min(
            edges,
            key=lambda edge: (edge["distance"], edge["proposal_index"]),
        )
    return highest_score, nearest


def _scope_summary(keys, gt_records, edges_by_gt, selected_by_gt,
                   highest_score, nearest):
    total = len(keys)
    degrees = [len(edges_by_gt.get(key, ())) for key in keys]
    selected = [selected_by_gt[key] for key in keys if key in selected_by_gt]
    matched = len(selected)
    values = []
    matched_values = []
    distances = []
    normalized_distances = []
    scores = []
    selected_is_nearest = 0
    conflict_keys = []
    alternative_matches = 0
    conflict_unmatched = 0
    for key in keys:
        edge = selected_by_gt.get(key)
        if edge is None:
            values.append(0.0)
        else:
            tau = gt_records[key]["tau"]
            normalized = edge["distance"] / tau
            value = max(0.0, 1.0 - normalized * normalized)
            values.append(value)
            matched_values.append(value)
            distances.append(edge["distance"])
            normalized_distances.append(normalized)
            scores.append(edge["score"])
            selected_is_nearest += int(
                edge["proposal_index"] == nearest[key]["proposal_index"])
        top = highest_score.get(key)
        if top is not None and (
                edge is None
                or edge["proposal_index"] != top["proposal_index"]):
            conflict_keys.append(key)
            alternative_matches += int(edge is not None)
            conflict_unmatched += int(edge is None)

    highest_counts = Counter(
        (key[0], highest_score[key]["proposal_index"])
        for key in keys if key in highest_score)
    nearest_counts = Counter(
        (key[0], nearest[key]["proposal_index"])
        for key in keys if key in nearest)
    reused_highest = {pred for pred, count in highest_counts.items() if count > 1}
    reused_nearest = {pred for pred, count in nearest_counts.items() if count > 1}
    return {
        "effective_gt": total,
        "candidate_degree_0": sum(degree == 0 for degree in degrees),
        "candidate_degree_1": sum(degree == 1 for degree in degrees),
        "candidate_degree_gt1": sum(degree > 1 for degree in degrees),
        "candidate_degree_gt1_rate": _ratio(
            sum(degree > 1 for degree in degrees), total),
        "matched": matched,
        "unmatched": total - matched,
        "coverage": _ratio(matched, total),
        "sum_q": sum(values),
        "q_all_effective": _quantiles(values),
        "q_matched": _quantiles(matched_values),
        "matched_distance_m": _quantiles(distances),
        "matched_normalized_distance": _quantiles(normalized_distances),
        "selected_teacher_score": _quantiles(scores),
        "selected_is_gt_nearest": selected_is_nearest,
        "selected_is_gt_nearest_rate": _ratio(selected_is_nearest, matched),
        "one_to_one_conflict_gt": len(conflict_keys),
        "conflict_resolved_by_alternative": alternative_matches,
        "conflict_became_unmatched": conflict_unmatched,
        "gt_centric_highest_score_reused_predictions": len(reused_highest),
        "gt_centric_highest_score_gt_on_reused_predictions": sum(
            highest_counts[pred] for pred in reused_highest),
        "gt_nearest_reused_predictions": len(reused_nearest),
        "gt_nearest_gt_on_reused_predictions": sum(
            nearest_counts[pred] for pred in reused_nearest),
    }


def _write_markdown(path, report):
    def pct(value):
        return "-" if value is None else f"{100.0 * value:.2f}%"

    def num(value, digits=4):
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        "# Center-value one-to-one teacher–GT matching",
        "",
        f"- Checkpoint: `{report['metadata'].get('checkpoint', '')}`",
        f"- Checkpoint SHA256: `{report['metadata'].get('checkpoint_sha256', '')}`",
        f"- Split: `{report['metadata'].get('split', '')}`",
        f"- Samples: {report['metadata'].get('global_samples', '')}",
        f"- PPU / batch: {report['metadata'].get('world_size', '')} × "
        f"{report['metadata'].get('batch_size_per_rank', '')}",
        "- Policy: exact class; small `<1m`; large `<2m`; "
        "score-first nearest-unmatched one-to-one; official ranges",
        "",
        "## Group summary",
        "",
        "| group | effective GT | matched | coverage | mean q | p50 d/tau | conflict GT | alternative | conflict unmatched |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("small", "large", "all"):
        value = report["groups"][name]
        lines.append(
            f"| {name} | {value['effective_gt']} | {value['matched']} | "
            f"{pct(value['coverage'])} | "
            f"{num(value['q_all_effective']['mean'])} | "
            f"{num(value['matched_normalized_distance']['p50'])} | "
            f"{value['one_to_one_conflict_gt']} | "
            f"{value['conflict_resolved_by_alternative']} | "
            f"{value['conflict_became_unmatched']} |"
        )
    lines.extend([
        "",
        "## Per-class summary",
        "",
        "| class | effective GT | degree 0 | degree >1 | matched | coverage | mean q | mean distance (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name in CLASS_NAMES:
        value = report["classes"][name]
        lines.append(
            f"| {name} | {value['effective_gt']} | "
            f"{value['candidate_degree_0']} | {value['candidate_degree_gt1']} | "
            f"{value['matched']} | {pct(value['coverage'])} | "
            f"{num(value['q_all_effective']['mean'])} | "
            f"{num(value['matched_distance_m']['mean'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    gt, candidates, gi, ci, part_paths = _load_parts(input_dir)
    gt_records = _load_effective_gt(input_dir, gt, gi)
    edges_by_gt, edges_by_scope_pred, prediction_scores = _build_edges(
        candidates, ci, gt_records)
    selected_by_gt = _assign_one_to_one(
        edges_by_scope_pred, prediction_scores)
    highest_score, nearest = _pre_assignment_choices(edges_by_gt)

    metadata_path = input_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists() else {})
    checkpoint_path = Path(metadata.get("checkpoint", ""))
    if checkpoint_path.is_file():
        metadata["checkpoint_sha256"] = _sha256(checkpoint_path)
    metadata.update({
        "parts": len(part_paths),
        "policy": {
            "class_policy": "exact_class",
            "selection": "score_first_nearest_unmatched_gt",
            "one_to_one": True,
            "strict_less_than": True,
            "trust_radius": {"small": 1.0, "large": 2.0},
            "official_class_ranges": OFFICIAL_RANGES,
            "teacher_value": "max(0, 1 - (distance / trust_radius)^2)",
        },
    })

    all_keys = sorted(gt_records)
    group_keys = {
        "small": [key for key in all_keys if gt_records[key]["group"] == "small"],
        "large": [key for key in all_keys if gt_records[key]["group"] == "large"],
        "all": all_keys,
    }
    report = {
        "metadata": metadata,
        "groups": {
            name: _scope_summary(
                keys, gt_records, edges_by_gt, selected_by_gt,
                highest_score, nearest)
            for name, keys in group_keys.items()
        },
        "classes": {
            class_name: _scope_summary(
                [key for key in all_keys
                 if gt_records[key]["class_name"] == class_name],
                gt_records, edges_by_gt, selected_by_gt,
                highest_score, nearest)
            for class_name in CLASS_NAMES
        },
    }
    for scope_kind in ("groups", "classes"):
        for scope_name, summary in report[scope_kind].items():
            if summary["matched"] + summary["unmatched"] != summary["effective_gt"]:
                raise RuntimeError(f"Count mismatch in {scope_kind}.{scope_name}")
            if (summary["candidate_degree_0"]
                    + summary["candidate_degree_1"]
                    + summary["candidate_degree_gt1"]
                    != summary["effective_gt"]):
                raise RuntimeError(
                    f"Candidate-degree mismatch in {scope_kind}.{scope_name}")
            if (summary["candidate_degree_0"]
                    + summary["conflict_became_unmatched"]
                    != summary["unmatched"]):
                raise RuntimeError(
                    f"Unmatched decomposition mismatch in "
                    f"{scope_kind}.{scope_name}")
            if (summary["conflict_resolved_by_alternative"]
                    + summary["conflict_became_unmatched"]
                    != summary["one_to_one_conflict_gt"]):
                raise RuntimeError(
                    f"Conflict decomposition mismatch in "
                    f"{scope_kind}.{scope_name}")
    json_path = input_dir / f"{args.output_name}.json"
    markdown_path = input_dir / f"{args.output_name}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(markdown_path, report)
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")


if __name__ == "__main__":
    main()
