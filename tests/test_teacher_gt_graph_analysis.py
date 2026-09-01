import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analyze_teacher_gt_graph import (  # noqa: E402
    assign_current,
    assign_hungarian,
    assign_independent,
    gt_order_greedy,
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
