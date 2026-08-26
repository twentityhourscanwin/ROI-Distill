#!/usr/bin/env python3
"""Capture YAML numerical baselines and compare frozen snapshots."""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from labeldistill.acceptance import (
    capture_training_snapshot,
    compare_snapshots,
    save_snapshot,
)
from labeldistill.acceptance.snapshot import batch_fingerprint
from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_RUNTIME_OVERRIDES = (
    "runtime.gpus=1",
    "runtime.batch_size_per_device=1",
    "runtime.precision=32",
    "data.num_workers=0",
    "ema.enabled=false",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture one YAML baseline")
    capture.add_argument("--config", required=True)
    capture.add_argument("--output", required=True)
    capture.add_argument("--batch-index", type=int, default=0)
    capture.add_argument("overrides", nargs="*")

    compare = subparsers.add_parser(
        "compare", help="compare two previously frozen .pt snapshots")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument(
        "--atol", type=float, default=1e-2,
        help="absolute tolerance (default covers observed PPU TF32 repeat drift)")
    compare.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def _bundle(config_path, extra_overrides):
    override_keys = {item.split("=", 1)[0] for item in extra_overrides}
    fixed = [
        item for item in FIXED_RUNTIME_OVERRIDES
        if item.split("=", 1)[0] not in override_keys
    ]
    return load_and_resolve_config(
        config_path,
        [*fixed, *extra_overrides],
        project_root=str(PROJECT_ROOT),
    )


def _fixed_batch(experiment, batch_index):
    loader = experiment.train_dataloader()
    iterator = iter(loader)
    batch = None
    for _ in range(batch_index + 1):
        batch = next(iterator)
    return batch


def _write_context(output_dir, bundle, batch, batch_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(bundle.config, output_dir / "resolved_config.yaml", resolve=True)
    context = {
        "config_source": str(bundle.source_path),
        "batch_index": batch_index,
        "seed": int(bundle.config.experiment.seed),
        "batch_fingerprint": batch_fingerprint(batch),
        "intentional_framework_changes": [
            "ragged proposals and GT-aligned MatchResult replace padding/truncation",
            "teacher inference executes once per training forward",
            "feature MSE reduces channels before applying the spatial mask",
        ],
    }
    (output_dir / "context.json").write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture_yaml(args):
    bundle = _bundle(args.config, args.overrides)
    seed = int(bundle.config.experiment.seed)
    pl.seed_everything(seed, workers=True)
    experiment = build_experiment(bundle)
    batch = _fixed_batch(experiment, args.batch_index)
    output = Path(args.output).expanduser().resolve()
    _write_context(output.parent, bundle, batch, args.batch_index)
    snapshot = capture_training_snapshot(experiment, batch, seed=seed)
    save_snapshot(snapshot, output)
    print(json.dumps({
        "snapshot": str(output),
        "total_loss": float(snapshot["total_loss"]),
    }, indent=2))


def compare_frozen(args):
    left_path = Path(args.left).expanduser().resolve()
    right_path = Path(args.right).expanduser().resolve()
    left = torch.load(left_path, map_location="cpu")
    right = torch.load(right_path, map_location="cpu")
    report = compare_snapshots(left, right, atol=args.atol, rtol=args.rtol)
    report.update({
        "left": str(left_path),
        "right": str(right_path),
        "tolerance_rationale": (
            "Repeated PPU TF32 convolution forwards drift by up to about 6e-3 "
            "in student head elements; structured teacher/ROI tensors remain exact."
        ),
    })
    report_path = Path(args.output).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "tensor_count": report["tensor_count"],
        "failed": report["failed"],
        "report": str(report_path),
    }, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def main():
    args = parse_args()
    if args.command == "capture":
        capture_yaml(args)
    else:
        compare_frozen(args)


if __name__ == "__main__":
    main()
