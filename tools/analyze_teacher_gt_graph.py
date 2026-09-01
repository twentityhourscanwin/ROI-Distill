"""Compare teacher/GT assignment policies from a cached candidate graph."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from extract_teacher_gt_graph import CLASS_NAMES, SMALL_CLASSES


POLICIES = (
    "P0_current_score_greedy",
    "P1_gt_nearest_reuse",
    "P2_gt_highest_score_reuse",
    "P3_hungarian_distance",
    "P4_gt_max_score_x_q_reuse",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-visual-examples", type=int, default=6)
    return parser.parse_args()


def load_graph(input_dir: Path):
    paths = sorted(input_dir.glob("graph_rank*_*.pt"))
    if not paths:
        raise RuntimeError(f"No graph parts found in {input_dir}")
    import torch
    tables = {"gt": [], "edges": [], "proposal_degrees": []}
    schemas = {}
    tokens = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        for name, schema_key in (
            ("gt", "gt_columns"), ("edges", "edge_columns"),
            ("proposal_degrees", "proposal_degree_columns"),
        ):
            columns = tuple(payload[schema_key])
            if name in schemas and schemas[name] != columns:
                raise RuntimeError(f"{name} schema mismatch in {path}")
            schemas[name] = columns
            tables[name].append(payload[name].numpy())
        tokens.update({int(k): v for k, v in payload["sample_tokens"].items()})
    frames = {
        name: pd.DataFrame(
            np.concatenate(parts, axis=0), columns=schemas[name])
        for name, parts in tables.items()
    }
    for frame in frames.values():
        for column in frame.columns:
            if column in {"sample_index", "gt_index", "proposal_index", "label",
                          "effective", "total", "degree_0", "degree_1",
                          "degree_ge_2"}:
                frame[column] = frame[column].astype(np.int64)
    return frames["gt"], frames["edges"], frames["proposal_degrees"], tokens, paths


def radius_for_label(label: int, small_radius=1.0, large_radius=2.0):
    return small_radius if CLASS_NAMES[label] in SMALL_CLASSES else large_radius


def candidate_edges_for_scope(edges, gt_keys, small_radius=1.0, large_radius=2.0):
    if not gt_keys:
        return edges.iloc[0:0].copy()
    scoped = edges.merge(
        pd.DataFrame(gt_keys, columns=["sample_index", "gt_index"]),
        on=["sample_index", "gt_index"], how="inner")
    radii = scoped["label"].map(
        lambda label: radius_for_label(int(label), small_radius, large_radius))
    return scoped[scoped["distance"] < radii].copy()


def _edge_records(frame):
    return {int(index): row for index, row in frame.iterrows()}


def assign_current(gt, edges):
    """Exact implementation: score-first, nearest-unmatched GT, one-to-one."""
    selected = {}
    for (sample_index, label), scope in edges.groupby(["sample_index", "label"]):
        occupied = set()
        proposals = scope[["proposal_index", "score"]].drop_duplicates()
        proposals = proposals.sort_values(
            ["score", "proposal_index"], ascending=[False, True], kind="stable")
        for proposal_index in proposals["proposal_index"]:
            candidates = scope[
                (scope["proposal_index"] == proposal_index)
                & ~scope["gt_index"].isin(occupied)
            ]
            if candidates.empty:
                continue
            chosen_index = candidates.sort_values(
                ["distance", "gt_index"], kind="stable").index[0]
            row = scope.loc[chosen_index]
            key = (int(sample_index), int(row.gt_index))
            selected[key] = int(chosen_index)
            occupied.add(int(row.gt_index))
    return selected


def assign_independent(edges, criterion):
    selected = {}
    for key_values, scope in edges.groupby(["sample_index", "gt_index"]):
        if criterion == "nearest":
            chosen = scope.sort_values(
                ["distance", "proposal_index"], kind="stable").index[0]
        elif criterion == "highest_score":
            chosen = scope.sort_values(
                ["score", "proposal_index"], ascending=[False, True],
                kind="stable").index[0]
        elif criterion == "score_x_q":
            radii = scope["label"].map(lambda x: radius_for_label(int(x)))
            value = scope["score"] * (1.0 - (scope["distance"] / radii) ** 2)
            chosen = value.sort_values(ascending=False, kind="stable").index[0]
        else:
            raise ValueError(criterion)
        selected[tuple(map(int, key_values))] = int(chosen)
    return selected


def assign_hungarian(edges):
    selected = {}
    for (sample_index, label), scope in edges.groupby(["sample_index", "label"]):
        gt_ids = sorted(scope["gt_index"].unique().astype(int))
        pred_ids = sorted(scope["proposal_index"].unique().astype(int))
        gt_lookup = {value: index for index, value in enumerate(gt_ids)}
        pred_lookup = {value: index for index, value in enumerate(pred_ids)}
        n_gt, n_pred = len(gt_ids), len(pred_ids)
        unmatched_cost = n_gt + 1.0
        cost = np.full((n_gt, n_pred + n_gt), unmatched_cost, dtype=np.float64)
        edge_lookup = {}
        radius = radius_for_label(int(label))
        for edge_index, row in scope.iterrows():
            i = gt_lookup[int(row.gt_index)]
            j = pred_lookup[int(row.proposal_index)]
            normalized = float(row.distance) / radius
            previous = cost[i, j]
            if normalized < previous:
                cost[i, j] = normalized
                edge_lookup[(i, j)] = int(edge_index)
        row_indices, col_indices = linear_sum_assignment(cost)
        for i, j in zip(row_indices, col_indices):
            if j < n_pred and (i, j) in edge_lookup:
                selected[(int(sample_index), gt_ids[i])] = edge_lookup[(i, j)]
    return selected


def gt_order_greedy(edges, reverse=False):
    selected = {}
    for (sample_index, label), scope in edges.groupby(["sample_index", "label"]):
        occupied = set()
        gt_ids = sorted(scope["gt_index"].unique().astype(int), reverse=reverse)
        for gt_index in gt_ids:
            available = scope[
                (scope["gt_index"] == gt_index)
                & ~scope["proposal_index"].isin(occupied)
            ]
            if available.empty:
                continue
            chosen = available.sort_values(
                ["distance", "proposal_index"], kind="stable").index[0]
            selected[(int(sample_index), gt_index)] = int(chosen)
            occupied.add(int(scope.loc[chosen, "proposal_index"]))
    return selected


def build_assignments(gt, edges):
    return {
        "P0_current_score_greedy": assign_current(gt, edges),
        "P1_gt_nearest_reuse": assign_independent(edges, "nearest"),
        "P2_gt_highest_score_reuse": assign_independent(edges, "highest_score"),
        "P3_hungarian_distance": assign_hungarian(edges),
        "P4_gt_max_score_x_q_reuse": assign_independent(edges, "score_x_q"),
    }


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("mean", "p25", "p50", "p75", "p90")}
    q = np.quantile(values, [0.25, 0.5, 0.75, 0.9])
    return {"mean": float(values.mean()), "p25": float(q[0]),
            "p50": float(q[1]), "p75": float(q[2]), "p90": float(q[3])}


def summarize_policy(gt, edges, selected, nearest=None, highest=None):
    total = len(gt)
    chosen = edges.loc[list(selected.values())].copy() if selected else edges.iloc[0:0]
    radii = chosen["label"].map(lambda x: radius_for_label(int(x)))
    q = 1.0 - (chosen["distance"] / radii) ** 2
    q_all = float(q.sum() / total) if total else None
    if nearest is None:
        nearest = assign_independent(edges, "nearest")
    if highest is None:
        highest = assign_independent(edges, "highest_score")
    chosen_by_key = {key: edge_index for key, edge_index in selected.items()}
    return {
        "effective_gt": total,
        "matched": len(selected),
        "unmatched": total - len(selected),
        "coverage": len(selected) / total if total else None,
        "mean_q_all_effective": q_all,
        "q_matched": quantiles(q),
        "distance_m": quantiles(chosen["distance"]),
        "teacher_score": quantiles(chosen["score"]),
        "bev_iou": quantiles(chosen["bev_iou"]),
        "size_log_l1": quantiles(chosen["size_log_l1"]),
        "yaw_error_rad": quantiles(chosen["yaw_error"]),
        "yaw_axis_error_rad": quantiles(chosen["yaw_axis_error"]),
        "velocity_error": quantiles(chosen["velocity_error"]),
        "selected_is_nearest_rate": (
            sum(chosen_by_key.get(k) == v for k, v in nearest.items()) / len(selected)
            if selected else None),
        "selected_is_highest_score_rate": (
            sum(chosen_by_key.get(k) == v for k, v in highest.items()) / len(selected)
            if selected else None),
    }


def bootstrap_ci(gt, assignments, reps, seed):
    sample_ids = np.array(sorted(gt["sample_index"].unique()), dtype=np.int64)
    totals = gt.groupby("sample_index").size().reindex(sample_ids, fill_value=0).to_numpy()
    policy_counts = {}
    edge_table = assignments["_edges"]
    for policy, selected in assignments.items():
        if policy == "_edges":
            continue
        counts = defaultdict(int)
        q_sums = defaultdict(float)
        for key, edge_index in selected.items():
            counts[key[0]] += 1
            row = edge_table.loc[edge_index]
            radius = radius_for_label(int(row.label))
            q_sums[key[0]] += 1.0 - (float(row.distance) / radius) ** 2
        policy_counts[policy] = (
            np.array([counts[s] for s in sample_ids]),
            np.array([q_sums[s] for s in sample_ids]),
        )
    rng = np.random.default_rng(seed)
    coverage_draws = {policy: [] for policy in policy_counts}
    q_draws = {policy: [] for policy in policy_counts}
    difference_draws = []
    for _ in range(reps):
        indices = rng.integers(0, len(sample_ids), len(sample_ids))
        denominator = totals[indices].sum()
        if denominator == 0:
            continue
        for policy, (counts, q_sums) in policy_counts.items():
            coverage_draws[policy].append(float(counts[indices].sum() / denominator))
            q_draws[policy].append(float(q_sums[indices].sum() / denominator))
        difference_draws.append(
            coverage_draws["P3_hungarian_distance"][-1]
            - coverage_draws["P0_current_score_greedy"][-1])
    result = {}
    for policy in policy_counts:
        result[policy] = {
            "coverage_95ci": np.quantile(coverage_draws[policy], [0.025, 0.975]).tolist(),
            "mean_q_all_effective_95ci": np.quantile(q_draws[policy], [0.025, 0.975]).tolist(),
        }
    result["P3_minus_P0_coverage_95ci"] = np.quantile(
        difference_draws, [0.025, 0.975]).tolist()
    return result


def candidate_diagnostics(gt, edges, proposal_degrees, assignments):
    degrees = edges.groupby(["sample_index", "gt_index"]).size()
    gt_keys = list(zip(gt.sample_index.astype(int), gt.gt_index.astype(int)))
    degree_values = np.array([degrees.get(key, 0) for key in gt_keys])
    nearest = assignments["P1_gt_nearest_reuse"]
    highest = assignments["P2_gt_highest_score_reuse"]
    shared_nearest = defaultdict(int)
    for key, edge_index in nearest.items():
        row = edges.loc[edge_index]
        shared_nearest[(key[0], int(row.proposal_index))] += 1
    p0 = assignments["P0_current_score_greedy"]
    p3 = assignments["P3_hungarian_distance"]
    keys_with_candidates = {key for key, degree in zip(gt_keys, degree_values) if degree > 0}
    p0_unmatched_candidates = keys_with_candidates - set(p0)
    nearest_highest_all = sum(nearest.get(k) == highest.get(k) for k in nearest)
    multi_keys = {key for key, degree in zip(gt_keys, degree_values) if degree > 1}

    nearest_score_ranks, highest_distance_ranks = [], []
    for key, scope in edges.groupby(["sample_index", "gt_index"]):
        score_order = list(scope.sort_values(
            ["score", "proposal_index"], ascending=[False, True], kind="stable").index)
        distance_order = list(scope.sort_values(
            ["distance", "proposal_index"], kind="stable").index)
        nearest_score_ranks.append(score_order.index(nearest[tuple(map(int, key))]) + 1)
        highest_distance_ranks.append(distance_order.index(highest[tuple(map(int, key))]) + 1)

    score_distance = spearmanr(edges["score"], edges["distance"])
    score_iou = spearmanr(edges["score"], edges["bev_iou"])
    score_distance_rho = getattr(score_distance, "statistic", score_distance.correlation)
    score_iou_rho = getattr(score_iou, "statistic", score_iou.correlation)
    degree_totals = proposal_degrees[["total", "degree_0", "degree_1", "degree_ge_2"]].sum()
    return {
        "gt_candidate_degree": {
            "degree_0": int((degree_values == 0).sum()),
            "degree_1": int((degree_values == 1).sum()),
            "degree_2": int((degree_values == 2).sum()),
            "degree_ge_3": int((degree_values >= 3).sum()),
        },
        "proposal_candidate_degree": {key: int(value) for key, value in degree_totals.items()},
        "nearest_equals_highest_score_rate_all_candidates": nearest_highest_all / len(nearest) if nearest else None,
        "nearest_equals_highest_score_rate_multi_candidate": (
            sum(nearest.get(k) == highest.get(k) for k in multi_keys) / len(multi_keys)
            if multi_keys else None),
        "nearest_proposal_score_rank": quantiles(nearest_score_ranks),
        "highest_score_proposal_distance_rank": quantiles(highest_distance_ranks),
        "nearest_reused_proposals": sum(value > 1 for value in shared_nearest.values()),
        "gt_on_reused_nearest_proposals": sum(value for value in shared_nearest.values() if value > 1),
        "P0_candidate_but_unmatched": len(p0_unmatched_candidates),
        "P0_candidate_but_unmatched_rate": len(p0_unmatched_candidates) / len(gt) if len(gt) else None,
        "P0_unmatched_recovered_by_P3": len(p0_unmatched_candidates & set(p3)),
        "P3_minus_P0_matches": len(p3) - len(p0),
        "edge_score_distance_spearman": {"rho": float(score_distance_rho), "p": float(score_distance.pvalue)},
        "edge_score_bev_iou_spearman": {"rho": float(score_iou_rho), "p": float(score_iou.pvalue)},
    }


def scope_rows(gt, edges, policies):
    rows = []
    scopes = [("all", "all", np.ones(len(gt), dtype=bool))]
    for label, name in enumerate(CLASS_NAMES):
        scopes.append(("class", name, gt.label.to_numpy() == label))
    ranges = [(0, 20), (20, 30), (30, 40), (40, 50), (50, math.inf)]
    for low, high in ranges:
        scopes.append(("range_m", f"{low}-{high if math.isfinite(high) else 'inf'}",
                       (gt.ego_distance.to_numpy() >= low) & (gt.ego_distance.to_numpy() < high)))
    speeds = [(0, 1), (1, 3), (3, math.inf)]
    for low, high in speeds:
        scopes.append(("speed_mps", f"{low}-{high if math.isfinite(high) else 'inf'}",
                       (gt.speed.to_numpy() >= low) & (gt.speed.to_numpy() < high)))
    for scope_type, scope_name, mask in scopes:
        scoped_gt = gt[mask]
        keys = set(zip(scoped_gt.sample_index.astype(int), scoped_gt.gt_index.astype(int)))
        for policy, selected in policies.items():
            scoped_selected = {key: value for key, value in selected.items() if key in keys}
            summary = summarize_policy(
                scoped_gt, edges, scoped_selected,
                nearest=policies["P1_gt_nearest_reuse"],
                highest=policies["P2_gt_highest_score_reuse"],
            )
            rows.append({"scope_type": scope_type, "scope": scope_name,
                         "policy": policy, **summary})
    return rows


def threshold_sensitivity(gt, edges):
    rows = []
    settings = {
        "small": [0.5, 0.75, 1.0, 1.25],
        "large": [1.0, 1.5, 2.0, 2.5],
    }
    for group, thresholds in settings.items():
        labels = {i for i, name in enumerate(CLASS_NAMES)
                  if (name in SMALL_CLASSES) == (group == "small")}
        scoped_gt = gt[gt.label.isin(labels)]
        keys = list(zip(scoped_gt.sample_index.astype(int), scoped_gt.gt_index.astype(int)))
        for threshold in thresholds:
            small_radius = threshold if group == "small" else 1.0
            large_radius = threshold if group == "large" else 2.0
            scoped_edges = candidate_edges_for_scope(
                edges, keys, small_radius=small_radius, large_radius=large_radius)
            assignments = {
                "P0_current_score_greedy": assign_current(scoped_gt, scoped_edges),
                "P1_gt_nearest_reuse": assign_independent(scoped_edges, "nearest"),
                "P3_hungarian_distance": assign_hungarian(scoped_edges),
            }
            for policy, selected in assignments.items():
                rows.append({
                    "group": group, "radius_m": threshold, "policy": policy,
                    "effective_gt": len(scoped_gt), "matched": len(selected),
                    "coverage": len(selected) / len(scoped_gt) if len(scoped_gt) else None,
                })
    return rows


def rectangle_vertices(row, prefix=""):
    x, y = row[f"{prefix}x"], row[f"{prefix}y"]
    dx, dy, yaw = row[f"{prefix}dx"], row[f"{prefix}dy"], row[f"{prefix}yaw"]
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]]) / 2
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return corners @ rotation.T + np.array([x, y])


def plot_conflicts(path, gt, edges, tokens, assignments, limit):
    p0, p1, p3 = (assignments[name] for name in (
        "P0_current_score_greedy", "P1_gt_nearest_reuse", "P3_hungarian_distance"))
    candidate_keys = set(zip(edges.sample_index.astype(int), edges.gt_index.astype(int)))
    priorities = list((candidate_keys - set(p0)) & set(p3))
    priorities += [key for key in p0 if key in p1 and p0[key] != p1[key]]
    seen_scopes, scopes = set(), []
    for sample_index, gt_index in priorities:
        label = int(gt[(gt.sample_index == sample_index) & (gt.gt_index == gt_index)].iloc[0].label)
        scope = (sample_index, label)
        if scope not in seen_scopes:
            seen_scopes.add(scope)
            scopes.append(scope)
        if len(scopes) >= limit:
            break
    if not scopes:
        return None
    columns = min(3, len(scopes))
    rows = math.ceil(len(scopes) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows), squeeze=False)
    for ax, (sample_index, label) in zip(axes.flat, scopes):
        scope_gt = gt[(gt.sample_index == sample_index) & (gt.label == label)]
        scope_edges = edges[(edges.sample_index == sample_index) & (edges.label == label)]
        for _, row in scope_gt.iterrows():
            ax.add_patch(Polygon(rectangle_vertices(row), fill=False, edgecolor="black", linewidth=1.5))
            ax.text(row.x, row.y, f"G{int(row.gt_index)}", fontsize=8)
        proposals = scope_edges.drop_duplicates("proposal_index")
        for _, row in proposals.iterrows():
            ax.add_patch(Polygon(rectangle_vertices(row, "proposal_"), fill=False,
                                 edgecolor="0.65", linewidth=0.8))
            ax.text(row.proposal_x, row.proposal_y, f"P{int(row.proposal_index)}\n{row.score:.2f}",
                    fontsize=7, color="0.35")
        for name, color, style in (
            ("P0_current_score_greedy", "tab:red", "-"),
            ("P1_gt_nearest_reuse", "tab:blue", "--"),
            ("P3_hungarian_distance", "tab:green", ":"),
        ):
            for key, edge_index in assignments[name].items():
                if key[0] != sample_index:
                    continue
                row = edges.loc[edge_index]
                if int(row.label) != label:
                    continue
                target = scope_gt[scope_gt.gt_index == key[1]].iloc[0]
                ax.plot([target.x, row.proposal_x], [target.y, row.proposal_y],
                        color=color, linestyle=style, linewidth=1.8, label=name[:2])
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"{CLASS_NAMES[label]} | {tokens.get(sample_index, sample_index)}", fontsize=9)
        ax.set_xlabel("ego x (m)")
        ax.set_ylabel("ego y (m)")
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=7)
    for ax in axes.flat[len(scopes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_markdown(path, metadata, report):
    def pct(value):
        return "-" if value is None else f"{100 * value:.3f}%"
    lines = [
        f"# Teacher–GT matching analysis ({metadata['split']})", "",
        f"- Git: `{metadata['git_commit']}` on `{metadata['git_branch']}`",
        f"- Teacher SHA256: `{metadata['checkpoint_sha256']}`",
        f"- Info SHA256: `{metadata['info_sha256']}`",
        f"- Samples: {metadata['global_samples']}",
        "- Input: current + 5 past + 4 future; identity BDA; exact class; official GT ranges",
        "", "## Policy comparison", "",
        "| policy | matched | coverage | mean q (all GT) | mean distance | mean score | mean BEV IoU | nearest | highest score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        item = report["policies"][policy]
        lines.append(
            f"| {policy} | {item['matched']} | {pct(item['coverage'])} | "
            f"{item['mean_q_all_effective']:.4f} | {item['distance_m']['mean']:.4f} | "
            f"{item['teacher_score']['mean']:.4f} | {item['bev_iou']['mean']:.4f} | "
            f"{pct(item['selected_is_nearest_rate'])} | {pct(item['selected_is_highest_score_rate'])} |")
    d = report["candidate_diagnostics"]
    lines.extend([
        "", "## Conflict diagnostics", "",
        f"- GT candidate degree 0/1/2/≥3: {d['gt_candidate_degree']['degree_0']} / "
        f"{d['gt_candidate_degree']['degree_1']} / {d['gt_candidate_degree']['degree_2']} / "
        f"{d['gt_candidate_degree']['degree_ge_3']}",
        f"- Nearest = highest-score: {pct(d['nearest_equals_highest_score_rate_all_candidates'])}; "
        f"among multi-candidate GT: {pct(d['nearest_equals_highest_score_rate_multi_candidate'])}",
        f"- P0 candidate-but-unmatched: {d['P0_candidate_but_unmatched']}; "
        f"recovered by P3: {d['P0_unmatched_recovered_by_P3']}",
        f"- P3 − P0 matches: {d['P3_minus_P0_matches']}",
        f"- Edge score↔distance Spearman ρ: {d['edge_score_distance_spearman']['rho']:.4f}; "
        f"score↔BEV-IoU ρ: {d['edge_score_bev_iou_spearman']['rho']:.4f}",
        "", "## Interpretation guardrails", "",
        "- P1/P2/P4 allow proposal reuse; their coverage is an upper-bound-style diagnostic, not a drop-in one-to-one matcher.",
        "- P3 is the order-independent one-to-one distance baseline and maximizes match cardinality before minimizing normalized distance.",
        "- Matching statistics measure teacher/GT association quality; they do not by themselves prove downstream distillation gains.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    gt, edges, proposal_degrees, tokens, part_paths = load_graph(input_dir)
    gt = gt[gt.effective == 1].copy()
    gt_keys = list(zip(gt.sample_index.astype(int), gt.gt_index.astype(int)))
    edges = candidate_edges_for_scope(edges, gt_keys)
    assignments = build_assignments(gt, edges)
    report = {
        "metadata": {**metadata, "parts": len(part_paths)},
        "policies": {
            policy: summarize_policy(
                gt, edges, assignments[policy],
                nearest=assignments["P1_gt_nearest_reuse"],
                highest=assignments["P2_gt_highest_score_reuse"],
            )
            for policy in POLICIES
        },
    }
    bootstrap_input = {**assignments, "_edges": edges}
    report["bootstrap"] = bootstrap_ci(
        gt, bootstrap_input, args.bootstrap_reps, args.seed)
    report["candidate_diagnostics"] = candidate_diagnostics(
        gt, edges, proposal_degrees, assignments)
    forward = gt_order_greedy(edges, reverse=False)
    reverse = gt_order_greedy(edges, reverse=True)
    report["gt_iteration_order_diagnostic"] = {
        "forward_matches": len(forward), "reverse_matches": len(reverse),
        "different_gt_assignments": sum(forward.get(k) != reverse.get(k)
                                         for k in set(forward) | set(reverse)),
    }

    scope = scope_rows(gt, edges, assignments)
    sensitivity = threshold_sensitivity(gt, edges)
    pd.DataFrame(scope).to_json(
        input_dir / "scope_summary.json", orient="records", indent=2)
    pd.DataFrame(scope).to_csv(input_dir / "scope_summary.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(
        input_dir / "threshold_sensitivity.csv", index=False)
    report["threshold_sensitivity"] = sensitivity
    (input_dir / "matching_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(input_dir / "matching_analysis.md", metadata, report)
    plot_conflicts(
        input_dir / "conflict_examples.png", gt, edges, tokens,
        assignments, args.max_visual_examples)
    print(json.dumps({
        "policies": {
            policy: {
                "matched": report["policies"][policy]["matched"],
                "coverage": report["policies"][policy]["coverage"],
                "mean_q_all_effective": report["policies"][policy]["mean_q_all_effective"],
                "mean_distance_m": report["policies"][policy]["distance_m"]["mean"],
                "mean_bev_iou": report["policies"][policy]["bev_iou"]["mean"],
            }
            for policy in POLICIES
        },
        "candidate_diagnostics": report["candidate_diagnostics"],
        "gt_iteration_order_diagnostic": report["gt_iteration_order_diagnostic"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
