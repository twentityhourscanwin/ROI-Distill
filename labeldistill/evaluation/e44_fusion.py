"""在 nuScenes 全局坐标系中等权融合多份检测结果。"""
from collections import defaultdict
from itertools import combinations
import math

import numpy as np
from nuscenes.eval.detection.constants import ATTRIBUTE_NAMES, DETECTION_NAMES
from scipy.spatial.distance import cdist


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _group_boxes(boxes, token):
    groups = defaultdict(list)
    if not isinstance(boxes, list):
        raise ValueError(f"expected a detection list for {token}")
    for index, box in enumerate(boxes):
        if not isinstance(box, dict) or box.get("sample_token") != token:
            raise ValueError(f"misassigned detection in {token}")
        name = box.get("detection_name")
        attribute = box.get("attribute_name")
        if name not in DETECTION_NAMES or (attribute != "" and attribute not in ATTRIBUTE_NAMES):
            raise ValueError(f"invalid detection class or attribute in {token}")
        score = box.get("detection_score")
        if not _number(score) or not 0 <= score <= 1:
            raise ValueError(f"invalid detection score in {token}")
        for field, length in (("translation", 3), ("size", 3), ("rotation", 4), ("velocity", 2)):
            values = box.get(field)
            if not isinstance(values, list) or len(values) != length or not all(map(_number, values)):
                raise ValueError(f"invalid {field} in {token}")
        if min(box["size"]) <= 0 or not math.isclose(math.hypot(*box["rotation"]), 1.0, abs_tol=1e-4):
            raise ValueError(f"invalid dimensions or non-unit quaternion in {token}")
        groups[name].append(index)
    return groups


def _clusters(frames, groups):
    # 用 (成员序号, 成员内框序号) 明确定义等距和等分时的稳定顺序。
    clusters = {(m, i): [(m, i)] for m, boxes in enumerate(frames) for i in range(len(boxes))}
    owner = {key: key for key in clusters}
    edges = []
    for m, n in combinations(range(len(frames)), 2):
        for name, first in groups[m].items():
            second = groups[n].get(name)
            if not second:
                continue
            distances = cdist(
                [frames[m][i]["translation"][:2] for i in first],
                [frames[n][i]["translation"][:2] for i in second],
                metric="sqeuclidean",
            )
            rows, cols = np.nonzero(distances <= 1.0)
            edges.extend((distances[r, c], (m, first[r]), (n, second[c])) for r, c in zip(rows, cols))

    for _, a, b in sorted(edges):
        left, right = owner[a], owner[b]
        if left == right:
            continue
        joined = clusters[left] + clusters[right]
        if len({m for m, i in joined}) != len(joined):
            continue
        if len(joined) > 2:
            # 两条短边不能保证第三条也在匹配半径内。
            centers = [frames[m][i]["translation"][:2] for m, i in joined]
            if any(math.dist(x, y) > 1.0 for x, y in combinations(centers, 2)):
                continue
        keep, drop = sorted((left, right))
        clusters[keep] = sorted(joined)
        for key in clusters.pop(drop):
            owner[key] = keep
    return clusters.values()


def _merge(boxes, member_count):
    anchor = boxes[0]
    result = max(boxes, key=lambda box: box["detection_score"]).copy()
    result["detection_score"] = sum(box["detection_score"] for box in boxes) / member_count
    if len(boxes) > 1:
        for field in ("translation", "size", "velocity"):
            result[field] = [sum(value / len(boxes) for value in axis)
                             for axis in zip(*(box[field] for box in boxes))]
        # 四元数的两个符号表示同一姿态，先对齐到首位成员的半球。
        quats = [box["rotation"] for box in boxes]
        signs = [-1 if sum(a*b for a, b in zip(anchor["rotation"], q)) < 0 else 1 for q in quats]
        aligned = [sum(s*v for s, v in zip(signs, axis)) for axis in zip(*quats)]
        norm = math.hypot(*aligned)
        result["rotation"] = [value / norm for value in aligned]
    return result


def _fuse(members):
    for payload in members:
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
            raise ValueError("expected a nuScenes detection payload")
        if not isinstance(payload.get("meta"), dict):
            raise ValueError("missing prediction modality metadata")
    first = members[0]
    tokens = first["results"].keys()
    if any(m["meta"] != first["meta"] or m["results"].keys() != tokens for m in members[1:]):
        raise ValueError("prediction modalities or sample tokens differ")
    results = {}
    for token in tokens:
        if not isinstance(token, str) or not token:
            raise ValueError("invalid sample token")
        frames = [m["results"][token] for m in members]
        groups = [_group_boxes(boxes, token) for boxes in frames]
        merged = [_merge([frames[m][i] for m, i in cluster], len(members))
                  for cluster in _clusters(frames, groups)]
        merged.sort(key=lambda box: box["detection_score"], reverse=True)
        results[token] = merged[:500]
    return {"meta": dict(first["meta"]), "results": results}


def fuse(left: dict, right: dict) -> dict:
    """同类框在水平距离 1 m 内按最近优先配对，孤立框以另一成员得分为零计。"""
    return _fuse((left, right))


def fuse_three(first: dict, second: dict, third: dict) -> dict:
    """每簇每个成员最多贡献一个框，任意两框相距至多 1 m，缺席成员得分为零。"""
    return _fuse((first, second, third))
