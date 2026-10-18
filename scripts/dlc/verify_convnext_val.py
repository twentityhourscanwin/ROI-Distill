#!/usr/bin/env python3
"""Reject incomplete or non-finite full nuScenes validation output."""

import argparse
import json
import math
from pathlib import Path
import pickle

import yaml


def _all_finite(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def verify(result_path, metric_path, info_path, *, expected_count):
    result_path = Path(result_path)
    metric_path = Path(metric_path)
    info_path = Path(info_path)
    with info_path.open("rb") as stream:
        infos = pickle.load(stream)
    expected = [info["sample_token"] for info in infos]
    if len(expected) != expected_count or len(set(expected)) != expected_count:
        raise ValueError("val info token count or uniqueness mismatch")
    with result_path.open(encoding="utf-8") as stream:
        predictions = json.load(stream)["results"]
    if set(predictions) != set(expected):
        missing = len(set(expected) - set(predictions))
        extra = len(set(predictions) - set(expected))
        raise ValueError(f"prediction token mismatch: missing={missing} extra={extra}")
    for token, detections in predictions.items():
        if not isinstance(detections, list):
            raise ValueError(f"prediction value for {token} is not a list")
        for detection in detections:
            if detection.get("sample_token") != token or not _all_finite(detection):
                raise ValueError(f"invalid or non-finite prediction for {token}")
    with metric_path.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    mean_ap = metrics["mean_ap"]
    nd_score = metrics["nd_score"]
    if not (0 <= mean_ap <= 1 and 0 <= nd_score <= 1):
        raise ValueError("mAP or NDS outside [0, 1]")
    if not (math.isfinite(mean_ap) and math.isfinite(nd_score)):
        raise ValueError("non-finite mAP or NDS")
    return {"samples": expected_count, "mAP": mean_ap, "NDS": nd_score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=6019)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    data_root = Path(config["data"]["root"])
    if not data_root.is_absolute():
        data_root = root / data_root
    info_path = data_root / config["data"]["val_info"]
    output_dir = Path(args.output_dir)
    summary = verify(
        output_dir / "results_nusc.json",
        output_dir / "metrics_summary.json",
        info_path,
        expected_count=args.expected_count,
    )
    summary["val_info"] = str(info_path.resolve())
    target = output_dir / "val_audit.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
