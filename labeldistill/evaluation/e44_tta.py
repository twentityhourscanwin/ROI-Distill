"""Export a single checkpoint with the eleven-view test worker and per-class box policy."""
import json
from pathlib import Path
import sys

from labeldistill.config import load_and_resolve_config, save_config_artifacts
from labeldistill.evaluation.detection_submission import INDEX, INPUTS, TEST_ROOT, prepare_submission, run_worker
from labeldistill.evaluation.e44_ordinary import read_test_tokens
from labeldistill.evaluation.tta_class_policy import checked_json
from labeldistill.evaluation.tta_core import GROUPS
from labeldistill.evaluation.tta_pipeline import ROOT, now, save_json, sha256


INDEX_DIGEST = INPUTS[INDEX]
SMOKE_INDICES = (0, 2, 8, 40, 42, 48, 3000, 6007)


def route_completed_worker(output, expected):
    selected = json.loads((output / "views.json").read_text())["tokens"]
    summary = json.loads((output / "summary.json").read_text())
    if (selected != expected or summary["samples"] != len(selected)
            or summary["status"] != "complete" or summary["full_val"]
            or summary["views"] != len(GROUPS["A11"])):
        raise ValueError("test worker summary or token order mismatch")

    def group_predictions(name):
        group = summary["groups"][name]
        directory = output / "groups" / name
        if (Path(group["output_dir"]).resolve() != directory
                or group["samples"] != len(selected)
                or tuple(group["constituent_views"]) != GROUPS[name]):
            raise ValueError(f"unexpected group metadata for {name}")
        return checked_json(directory / "results_nusc.json", group["results_sha256"])

    return prepare_submission(group_predictions("A11"), group_predictions("B0"), selected)


def export(config: Path, checkpoint: Path, output: Path, *,
           gpus: int, batch_size: int, workers: int, smoke: bool) -> Path:
    output = output.resolve()
    if any(output.iterdir()):
        raise FileExistsError(f"TTA stage needs an empty output directory: {output}")
    if gpus < 1 or batch_size < 1 or workers < 0:
        raise ValueError("invalid resource settings")
    if sha256(INDEX) != INDEX_DIGEST:
        raise ValueError("test index SHA-256 mismatch")
    tokens = read_test_tokens(INDEX, TEST_ROOT)
    selected = [tokens[i] for i in SMOKE_INDICES] if smoke else tokens
    config_dir = output / "config"
    bundle = load_and_resolve_config(str(config), [
        f"runtime.output_dir={config_dir}", f"runtime.gpus={gpus}",
        f"runtime.batch_size_per_device={batch_size}", f"data.num_workers={workers}",
        f"data.root={TEST_ROOT}", f"data.val_info={INDEX}", "evaluation.run_metrics=false",
    ], project_root=str(ROOT))
    save_config_artifacts(bundle)
    resolved = config_dir / "resolved_config.yaml"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={gpus}", "-m", "labeldistill.evaluation.tta_worker",
        "--export-test", "smoke" if smoke else "full", "--config", str(resolved),
        "--checkpoint", str(checkpoint.resolve()), "--output-dir", str(output),
    ]
    log = output / "worker.log"
    audit = output / "tta_worker.json"
    record = {"status": "running", "command": command, "cwd": str(ROOT), "log": str(log),
              "started_utc": now(), "inputs": {str(p.resolve()): sha256(p)
                                              for p in [checkpoint, config, resolved]},
              "test_index": str(INDEX), "test_index_sha256": INDEX_DIGEST}
    save_json(audit, record)
    try:
        run_worker(command, cwd=ROOT, log=log, record=record)
        result, capped = route_completed_worker(output, selected)
        path = output / "results_nusc.json"
        with path.open("x") as stream:
            json.dump(result, stream, allow_nan=False, separators=(",", ":"))
        record.update(status="complete", samples=len(selected), capped_samples=capped,
                      results_sha256=sha256(path))
        return path
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        save_json(audit, record)
