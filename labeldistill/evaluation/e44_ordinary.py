"""Export ordinary camera predictions on the test split using the native evaluation entry."""
import json
from pathlib import Path
import sys

import mmengine
import torch
import yaml

from labeldistill.evaluation.detection_submission import INDEX, TEST_ROOT, run_worker
from labeldistill.evaluation.tta_pipeline import ROOT, now, save_json
from labeldistill.evaluators.det_evaluators import DetNuscEvaluator


def read_test_tokens(index: Path, root: Path):
    tokens = [info["sample_token"] for info in mmengine.load(str(index))]
    official = json.loads((root / "v1.0-test/sample.json").read_text())
    if len(tokens) != 6008 or len(official) != 6008 or len(set(tokens)) != 6008:
        raise ValueError("expected 6008 unique test samples")
    if {row["token"] for row in official} != set(tokens):
        raise ValueError("test index differs from official sample tokens")
    return tokens


def restore_records(ranks, tokens, *, limit):
    """Interleave native DDP rank order, checking real samples and sampler padding."""
    world = len(ranks)
    per_rank = (len(tokens) + world - 1) // world
    if limit is not None:
        per_rank = min(per_rank, limit)
    if any(len(rank) != per_rank for rank in ranks):
        raise ValueError("unexpected prediction shard length")
    records = [record for row in zip(*ranks) for record in row]
    for index, record in enumerate(records):
        if record[3]["token"] != tokens[index % len(tokens)]:
            raise ValueError(f"prediction token differs at index {index}")
    return records[:len(tokens)]


def prediction_shards(output, world_size):
    paths = [output / f"predictions_rank_{rank}.pt" for rank in range(world_size)]
    missing = [str(rank) for rank, path in enumerate(paths) if not path.is_file()]
    if missing:
        raise RuntimeError(f"incomplete predictions in {output}; missing ranks: {', '.join(missing)}")
    return paths


def export(config: Path, checkpoint: Path, output: Path, *,
           gpus: int, batch_size: int, workers: int, smoke: bool) -> Path:
    output = output.resolve()
    if any(output.iterdir()):
        raise FileExistsError(f"ordinary stage needs an empty output directory: {output}")
    if gpus < 1 or batch_size < 1 or workers < 0:
        raise ValueError("invalid resource settings")

    tokens = read_test_tokens(INDEX, TEST_ROOT)
    command = [
        sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={gpus}", str(ROOT / "scripts/dlc/evaluate_history.py"),
        "--history-fallback", "previous_valid", "--config", str(config.resolve()),
        "--checkpoint", str(checkpoint.resolve()), "--output-dir", str(output),
        "--prediction-only",
    ]
    if smoke:
        command.extend(["--limit-test-batches", "1"])
    else:
        command.append("runtime.limit_test_batches=null")
    command.extend([
        f"runtime.gpus={gpus}", "runtime.num_nodes=1",
        f"runtime.batch_size_per_device={batch_size}", f"data.num_workers={workers}",
        f"data.root={TEST_ROOT}", f"data.val_info={INDEX}", "evaluation.run_metrics=false",
    ])
    log = output / "worker.log"
    record = {"command": command, "cwd": str(ROOT), "log": str(log), "started_utc": now()}
    audit_path = output / "ordinary_worker.json"
    save_json(audit_path, record)
    try:
        run_worker(command, cwd=ROOT, log=log, record=record)
    finally:
        save_json(audit_path, record)

    ranks = []
    for shard in prediction_shards(output, gpus):
        batches = torch.load(shard,
                             map_location="cpu", weights_only=False)
        ranks.append([record for batch in batches for record in batch])
    records = restore_records(ranks, tokens, limit=batch_size if smoke else None)
    actual = yaml.safe_load((output / "resolved_config.yaml").read_text())
    evaluator = DetNuscEvaluator(actual["classes"]["names"], data_root=str(TEST_ROOT),
                                 version="v1.0-test", output_dir=str(output))
    evaluator.format_results([record[:3] for record in records], [record[3] for record in records])
    result = output / "results_nusc.json"
    if set(json.loads(result.read_bytes())["results"]) != set(tokens[:len(records)]):
        raise ValueError("native formatter returned incorrect test coverage")
    return result
