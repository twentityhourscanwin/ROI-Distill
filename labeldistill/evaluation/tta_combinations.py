"""从固定的 E38 候选缓存回放四组 R3 组合，所有计算在 CPU 上进行。"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import signal
from statistics import median
import subprocess
import sys
from types import SimpleNamespace

import torch

from labeldistill.config import load_and_resolve_config
from labeldistill.evaluation.tta_core import GROUPS, VIEW_SPECS, merge_candidates
from labeldistill.evaluation.tta_pipeline import (
    CHECKPOINT_SHA256, OUTPUT_ROOT, ROOT, SOURCE_CONFIG_SHA256, now, save_json, sha256,
)
from labeldistill.evaluation.tta_worker import atomic_save, check_prediction, ordered_records
from labeldistill.evaluators.det_evaluators import DetNuscEvaluator
from labeldistill.presets.j4 import build_j4_model_configs


RELEASE_TAG = "exp/cnb-tta-combinations-20261120"
FULL_CACHE = OUTPUT_ROOT / "e38_tta_20261119T123127_484916Z"
SMOKE_CACHE = OUTPUT_ROOT.parent / "diagnostics/e38_delivery_20261119/smoke_batch4"
# 固定已经核验的 manifest、视图、实际配置和全部 rank 的候选哈希清单。
FULL_DIGEST = "e4348712946f0ec318bb4f2fd408e42080bfe6629152f4fd606c86fa18e9b5f6"
SMOKE_DIGEST = "2d683b58da7b7f1dba4572ed3a1bf12e6489f37cfd0a6d2504cb960b2fb23c53"
VAL_INFO_SHA256 = "eb2b63510e2709e0d18bd4a478af0e92f10820c0c7eee6fae618448362aec36f"
COMBINATIONS = {
    f"R3_{family}": GROUPS["R3"] + GROUPS[family][1:]
    for family in ("I2", "F4", "Z3", "S3")
}


def read_json(path):
    return json.loads(path.read_bytes())


def metadata_files(world):
    return ["pipeline.json", "views.json", "config/resolved_config.yaml",
            *(f"rank_{rank:03d}.json" for rank in range(world))]


def metadata_digest(cache, world):
    listing = "".join(f"{sha256(cache / name)}  {name}\n" for name in metadata_files(world))
    return hashlib.sha256(listing.encode()).hexdigest()


def verify_metadata(cache, world, expected):
    if metadata_digest(cache, world) != expected:
        raise ValueError("E38 metadata checksum mismatch")


def load_candidates(path, expected_digest, expected_tokens):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_digest:
        raise ValueError(f"candidate checksum mismatch: {path}")
    records = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    if [r["meta"]["token"] for r in records] != expected_tokens:
        raise ValueError(f"candidate token order mismatch: {path}")
    return records


def new_output(path, cache):
    path = path.resolve()
    if path.is_relative_to(cache.resolve()):
        raise ValueError("output must be outside the source cache")
    path.mkdir(parents=True, exist_ok=False)
    return path


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def merge_shards(cache, output, head, tokens, ranks, groups):
    for rank, timing in enumerate(ranks):
        local = tokens[rank::len(ranks)]
        candidates = {}
        for view in VIEW_SPECS:
            record = timing["views"][view]
            path = cache / "candidates" / view / f"rank_{rank:03d}.pt"
            candidates[view] = load_candidates(path, record["sha256"], local)
        for group, views in groups.items():
            merged = []
            for i in range(len(local)):
                meta = candidates["identity"][i]["meta"]
                tasks = {v: candidates[v][i]["tasks"] for v in views}
                prediction = merge_candidates(head, tasks, meta)
                check_prediction(prediction)
                merged.append({"meta": meta, "prediction": prediction})
            atomic_save(output / "groups" / group / f"rank_{rank:03d}.pt", merged)
        print(f"merged rank {rank + 1}/{len(ranks)}", flush=True)


def evaluate_group(directory, world, tokens, config, full):
    shards = [torch.load(directory / f"rank_{rank:03d}.pt", map_location="cpu",
                         weights_only=False) for rank in range(world)]
    records = ordered_records(shards, tokens)
    predictions = [[tensor.numpy() for tensor in r["prediction"]] for r in records]
    metas = [r["meta"] for r in records]
    evaluator = DetNuscEvaluator(class_names=list(config.classes.names),
                                 data_root=str(ROOT / config.data.root), output_dir=str(directory))
    if full:
        evaluator.evaluate(predictions, metas)
    else:
        evaluator.format_results(predictions, metas)
    result_path = directory / "results_nusc.json"
    saved = read_json(result_path)["results"]
    if set(saved) != set(tokens):
        raise ValueError(f"formatted token coverage mismatch: {directory}")
    audit = {"samples": len(tokens), "results_sha256": sha256(result_path)}
    if full:
        path = directory / "metrics_summary.json"
        metrics = read_json(path)
        audit.update(mAP=metrics["mean_ap"], NDS=metrics["nd_score"],
                     tp_errors=metrics["tp_errors"], class_AP=metrics["mean_dist_aps"],
                     metrics_sha256=sha256(path))
        errors = audit["tp_errors"]
        class_ap = audit["class_AP"]
        if (set(errors) != {"trans_err", "scale_err", "orient_err", "vel_err", "attr_err"}
                or set(class_ap) != set(config.classes.names)
                or not all(math.isfinite(x) and x >= 0 for x in errors.values())
                or not all(0 <= x <= 1 for x in (audit["mAP"], audit["NDS"], *class_ap.values()))):
            raise ValueError(f"invalid official metrics: {directory}")
    return audit


def replay(cache, output, smoke, state):
    world, fingerprint = (2, SMOKE_DIGEST) if smoke else (16, FULL_DIGEST)
    verify_metadata(cache, world, fingerprint)
    manifest = read_json(cache / "pipeline.json")
    summary = manifest["summary"]
    views = read_json(cache / "views.json")
    tokens = views["tokens"]
    if (manifest["status"] != "complete" or manifest["smoke"] != smoke
            or manifest["stages"]["inference_and_merge"]["exitcode"] != 0
            or manifest["checkpoint_sha256"] != CHECKPOINT_SHA256
            or manifest["source_config_sha256"] != SOURCE_CONFIG_SHA256
            or summary["full_val"] == smoke or summary["samples"] != len(tokens)
            or len(set(tokens)) != (8 if smoke else 6019)
            or views["views"] != {v: asdict(spec) for v, spec in VIEW_SPECS.items()}):
        raise ValueError("unexpected source run or view definitions")
    config_path = cache / "config/resolved_config.yaml"
    config = load_and_resolve_config(str(config_path), [], project_root=str(ROOT)).config
    if not smoke and sha256(ROOT / config.data.root / config.data.val_info) != VAL_INFO_SHA256:
        raise ValueError("validation split checksum mismatch")
    head_cfg = build_j4_model_configs(config).head
    head = SimpleNamespace(test_cfg=head_cfg["test_cfg"],
                           num_classes=[task["num_class"] for task in head_cfg["tasks"]])
    ranks = [read_json(cache / f"rank_{i:03d}.json") for i in range(world)]
    groups = dict(COMBINATIONS)
    if smoke:
        groups["R3"] = GROUPS["R3"]
    state.update(cache_metadata_sha256=fingerprint, checkpoint_sha256=CHECKPOINT_SHA256,
                 source_config_sha256=SOURCE_CONFIG_SHA256, samples=len(tokens), phase="merge")
    save_json(output / "pipeline.json", state)
    merge_shards(cache, output, head, tokens, ranks, groups)
    results = {}
    for name, members in groups.items():
        state["phase"] = f"evaluate/{name}"
        save_json(output / "pipeline.json", state)
        directory = output / "groups" / name
        audit = evaluate_group(directory, world, tokens, config, not smoke)
        audit.update(constituent_views=members, output_dir=str(directory),
                     e38_view_minutes=median(sum(rank["views"][v]["seconds"] for v in members)
                                             for rank in ranks) / 60)
        if smoke and name == "R3":
            if audit["results_sha256"] != summary["groups"]["R3"]["results_sha256"]:
                raise ValueError("R3 smoke replay differs from E38 predictions")
        if not smoke:
            audit["delta_pp"] = {
                ref: {metric: 100 * (audit[metric] - summary["groups"][ref][metric])
                      for metric in ("mAP", "NDS")} for ref in ("R3", "B0", "A11")
            }
        save_json(directory / "val_audit.json", audit)
        results[name] = audit
    return {"status": "complete", "full_val": not smoke, "samples": len(tokens),
            "groups": results, "source_cache": str(cache), "finished_utc": now()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="回放已固定的八帧缓存并核对 R3")
    parser.add_argument("--output-dir", type=Path, help="独占新建的输出目录")
    args = parser.parse_args()
    head = git("rev-parse", "HEAD")
    if not args.smoke and (head != git("rev-parse", f"{RELEASE_TAG}^{{commit}}")
                           or git("status", "--porcelain", "--untracked-files=no")):
        parser.error("formal evaluation requires a clean tagged release")
    cache = SMOKE_CACHE if args.smoke else FULL_CACHE
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    prefix = "e39_tta_smoke" if args.smoke else "e39_tta"
    output = new_output(args.output_dir or OUTPUT_ROOT / f"{prefix}_{run_id}", cache)
    state = {"status": "running", "phase": "verify", "commit": head, "argv": sys.argv,
             "cwd": str(ROOT), "started_utc": now(), "cache": str(cache), "output_dir": str(output)}
    print(f"output_dir={output}", flush=True)
    try:
        save_json(output / "pipeline.json", state)
        summary = replay(cache, output, args.smoke, state)
        save_json(output / "summary.json", summary)
        state.update(status="complete", phase="finished", exitcode=0)
    except SystemExit as exc:
        state.update(status="failed", error=repr(exc), exitcode=exc.code)
        raise
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="failed", error=repr(exc),
                     exitcode=130 if isinstance(exc, KeyboardInterrupt) else 1)
        raise
    finally:
        state["finished_utc"] = now()
        save_json(output / "pipeline.json", state)


def terminate(signum, frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    main()
