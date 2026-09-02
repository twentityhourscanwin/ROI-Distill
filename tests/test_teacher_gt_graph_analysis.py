import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analyze_teacher_gt_graph import (  # noqa: E402
    assign_current,
    assign_hungarian,
    assign_independent,
    gt_order_greedy,
    threshold_sensitivity,
)


def _edges(rows):
    return pd.DataFrame(rows, columns=[
        "sample_index", "gt_index", "proposal_index", "label",
        "distance", "score",
    ])


def test_current_score_greedy_can_select_farther_proposal_for_gt():
    edges = _edges([
        [0, 0, 0, 0, 0.30, 0.90],
        [0, 0, 1, 0, 0.10, 0.80],
        [0, 1, 0, 0, 0.40, 0.90],
        [0, 1, 1, 0, 0.60, 0.80],
    ])
    selected = assign_current(pd.DataFrame(), edges)
    assert selected == {(0, 0): 0, (0, 1): 3}
    nearest = assign_independent(edges, "nearest")
    assert nearest == {(0, 0): 1, (0, 1): 2}


def test_hungarian_maximizes_cardinality_before_distance():
    edges = _edges([
        [0, 0, 0, 0, 0.10, 0.99],
        [0, 1, 0, 0, 0.20, 0.99],
        [0, 0, 1, 0, 1.90, 0.10],
    ])
    current = assign_current(pd.DataFrame(), edges)
    hungarian = assign_hungarian(edges)
    assert len(current) == 1
    assert len(hungarian) == 2
    assert hungarian[(0, 1)] == 1
    assert hungarian[(0, 0)] == 2


def test_gt_centric_one_to_one_depends_on_iteration_order():
    edges = _edges([
        [0, 0, 0, 0, 0.10, 0.9],
        [0, 1, 0, 0, 0.20, 0.9],
        [0, 0, 1, 0, 0.30, 0.8],
    ])
    forward = gt_order_greedy(edges, reverse=False)
    reverse = gt_order_greedy(edges, reverse=True)
    assert forward != reverse
    assert len(forward) == 1
    assert len(reverse) == 2


def test_threshold_sensitivity_can_expand_beyond_current_radius():
    gt = pd.DataFrame([
        {"sample_index": 0, "gt_index": 0, "label": 5,
         "ego_distance": 10.0, "speed": 0.0},
    ])
    edges = _edges([
        [0, 0, 0, 5, 1.10, 0.9],
    ])
    rows = threshold_sensitivity(gt, edges)
    p3 = {
        row["radius_m"]: row["coverage"] for row in rows
        if row["group"] == "small"
        and row["policy"] == "P3_hungarian_distance"
    }
    assert p3[1.0] == 0.0
    assert p3[1.25] == 1.0
