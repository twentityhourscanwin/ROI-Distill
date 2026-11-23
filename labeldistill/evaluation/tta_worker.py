"""Distributed, padding-free E38 inference and cached-candidate fusion."""
import argparse
from dataclasses import asdict
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.trainer.connectors.accelerator_connector import _set_torch_flags
import torch
from torch.utils.data import DataLoader, Subset

from labeldistill.builders import build_experiment
from labeldistill.config import load_and_resolve_config
from labeldistill.datasets.nusc_det_dataset import collate_fn
from labeldistill.evaluation.tta_core import (
    GROUPS, VIEW_SPECS, bev_matrix, decode_candidates, merge_candidates)
from labeldistill.evaluation.tta_data import make_dataset, partition_indices
from labeldistill.evaluation.tta_pipeline import ROOT, now, save_json, sha256
from labeldistill.evaluators.det_evaluators import DetNuscEvaluator


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_cpu(item) for item in value]
    return value


def atomic_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def check_prediction(prediction):
    boxes, scores, labels = prediction
    if boxes.ndim != 2 or boxes.shape[1] != 9 or len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("invalid prediction shape")
    if not all(torch.isfinite(t).all() for t in prediction):
        raise ValueError("non-finite merged prediction")
    if len(boxes) and not (boxes[:, 3:6] > 0).all():
        raise ValueError("nonpositive box dimensions")


def ordered_records(shards, expected_tokens):
    records = {}
    for shard in shards:
        for rec in shard:
            token = rec["meta"]["token"]
            if token in records:
                raise ValueError(f"duplicate token: {token}")
            records[token] = rec
    if set(records) != set(expected_tokens):
        raise ValueError("prediction token coverage mismatch")
    return [records[token] for token in expected_tokens]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--indices")
    mode.add_argument("--full-val", action="store_true")
    mode.add_argument("--export-test", choices=("full", "smoke"))
    args = parser.parse_args()
    groups = {k: GROUPS[k] for k in ("B0", "A11")} if args.export_test else GROUPS
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        torch.distributed.init_process_group("gloo", timeout=timedelta(hours=2))
    bundle = load_and_resolve_config(args.config, [], project_root=str(ROOT))
    config = bundle.config
    if world != config.runtime.gpus:
        raise ValueError("worker world size differs from configured devices")
    pl.seed_everything(config.experiment.seed, workers=True)
    _set_torch_flags(deterministic=bool(config.runtime.deterministic))
    experiment = build_experiment(bundle)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    experiment.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model = experiment.model.cuda().eval()
    head = model.head
    dataset = make_dataset(experiment, VIEW_SPECS["identity"])
    if args.export_test and len(dataset) != 6008:
        raise ValueError("expected the complete nuScenes test index")
    if args.export_test == "smoke":
        indices = [0, 2, 8, 40, 42, 48, 3000, 6007]
    elif args.indices is not None:
        indices = list(map(int, args.indices.split(",")))
    else:
        indices = list(range(len(dataset)))
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("invalid evaluation indices")
    if args.full_val and len(indices) != 6019:
        raise ValueError("full validation must contain exactly 6019 samples")
    expected_tokens = [dataset.infos[i]["sample_token"] for i in indices]
    if len(set(expected_tokens)) != len(indices):
        raise ValueError("input token duplication")
    owned = partition_indices(indices, rank, world)
    out = args.output_dir
    timing = {"rank": rank, "world_size": world, "samples": len(owned),
              "device": torch.cuda.get_device_name(local_rank), "views": {},
              "native_identity_exact": True}
    if rank == 0:
        save_json(out / "views.json", {
            "views": {k: asdict(v) for k, v in VIEW_SPECS.items()},
            "groups": groups, "indices": indices, "tokens": expected_tokens,
            "history_fallback": "previous_valid"})
    for view_name, view in VIEW_SPECS.items():
        started = time.perf_counter()
        dataset = make_dataset(experiment, view)
        loader = DataLoader(
            Subset(dataset, owned), batch_size=config.runtime.batch_size_per_device,
            shuffle=False, num_workers=config.data.num_workers,
            collate_fn=partial(collate_fn, is_return_depth=False),
            pin_memory=False)
        records = []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                images, mats, _, metas, _, _ = batch
                expected_bda = bev_matrix(view)
                torch.testing.assert_close(
                    mats["bda_mat"], expected_bda.expand_as(mats["bda_mat"]),
                    rtol=0, atol=0)
                images = images.cuda()
                mats = {k: v.cuda() for k, v in mats.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predictions = model(x=images, mats_dict=mats)
                    candidates = decode_candidates(head, predictions)
                    native = model.get_bboxes(predictions, metas) if view_name == "identity" else None
                for index, (tasks, meta) in enumerate(zip(candidates, metas)):
                    tasks_cpu = to_cpu(tasks)
                    if native is not None:
                        single = merge_candidates(head, {"identity": tasks_cpu}, meta)
                        for got, expected in zip(single, native[index]):
                            torch.testing.assert_close(got, expected.detach().cpu(), rtol=0, atol=0)
                    records.append({"meta": meta, "tasks": tasks_cpu})
                if batch_index % 25 == 0:
                    print(f"rank={rank} view={view_name} done={len(records)}/{len(owned)}", flush=True)
        torch.cuda.synchronize()
        target = out / "candidates" / view_name / f"rank_{rank:03d}.pt"
        atomic_save(target, records)
        timing["views"][view_name] = {
            "seconds": time.perf_counter() - started, "samples": len(records),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "sha256": sha256(target), "path": str(target)}
        save_json(out / f"rank_{rank:03d}.json", timing)
        del records, loader
    # Each rank merges its own nonoverlapping partition, retaining all raw views.
    by_view = {name: torch.load(
        out / "candidates" / name / f"rank_{rank:03d}.pt",
        map_location="cpu", weights_only=False) for name in VIEW_SPECS}
    local_tokens = [dataset.infos[i]["sample_token"] for i in owned]
    for name, recs in by_view.items():
        if [rec["meta"]["token"] for rec in recs] != local_tokens:
            raise ValueError(f"cached view alignment mismatch: {name}")
    for group_name, views in groups.items():
        records = []
        for pos, token in enumerate(local_tokens):
            meta = by_view["identity"][pos]["meta"]
            prediction = merge_candidates(
                head, {v: by_view[v][pos]["tasks"] for v in views}, meta)
            check_prediction(prediction)
            records.append({"meta": meta, "prediction": to_cpu(prediction)})
        target = out / "groups" / group_name / f"rank_{rank:03d}.pt"
        atomic_save(target, records)
    save_json(out / f"rank_{rank:03d}.json", timing)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank != 0:
        return
    summary = {"status": "complete", "samples": len(indices), "full_val": args.full_val,
               "groups": {}, "views": len(VIEW_SPECS), "finished_utc": None}
    for group_name in groups:
        group_out = out / "groups" / group_name
        shards = [torch.load(group_out / f"rank_{i:03d}.pt",
                            map_location="cpu", weights_only=False) for i in range(world)]
        records = ordered_records(shards, expected_tokens)
        predictions = [[t.numpy() for t in rec["prediction"]] for rec in records]
        metas = [rec["meta"] for rec in records]
        evaluator = DetNuscEvaluator(
            class_names=list(config.classes.names), data_root=experiment.data_root,
            output_dir=str(group_out))
        if args.full_val:
            evaluator.evaluate(predictions, metas)
        else:
            evaluator.format_results(predictions, metas)
        result_file = group_out / "results_nusc.json"
        with result_file.open() as f:
            saved = json.load(f)["results"]
        if set(saved) != set(expected_tokens):
            raise ValueError("formatted token coverage mismatch")
        result = {"samples": len(records), "output_dir": str(group_out),
                  "results_sha256": sha256(result_file),
                  "constituent_views": GROUPS[group_name]}
        if args.full_val:
            metrics = json.loads((group_out / "metrics_summary.json").read_text())
            result.update(mAP=metrics["mean_ap"], NDS=metrics["nd_score"],
                          tp_errors=metrics["tp_errors"],
                          class_AP=metrics["mean_dist_aps"],
                          metrics_sha256=sha256(group_out / "metrics_summary.json"))
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in [result["mAP"], result["NDS"]]):
                raise ValueError("invalid metrics")
        summary["groups"][group_name] = result
        save_json(group_out / "val_audit.json", result)
    summary["finished_utc"] = now()
    save_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
