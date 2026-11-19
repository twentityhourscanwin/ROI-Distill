"""Fixed E38 evaluation orchestrator; never starts training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = "convnext_wide_depth_s0_20261117_012943"
CHECKPOINT = Path("/mnt/nas_data/guqiupeng/checkpoint_nes") / RUN / "epoch_29.ckpt"
CHECKPOINT_SHA256 = "2f23b5a78bb04e1e13ddb8d2757079e4fcc9b6268d6a1054076c6a55c4019930"
SOURCE_CONFIG = Path("/mnt/nas_data/guqiupeng/ROI_LABEL_DISTILL/outputs") / RUN / "resolved_config.yaml"
SOURCE_CONFIG_SHA256 = "eb60209ddd456a8e2ac3d30cad1d4cb1a3589a48d134fddfffe709ccbe16dac6"
OUTPUT_ROOT = Path("/mnt/nas_data/guqiupeng/ROI_LABEL_DISTILL/outputs/evaluation")
SMOKE_INDICES = (0, 1, 2, 100, 500, 1000, 3000, 6000)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(p)


def unique_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_stage(command, *, cwd, log, record):
    record.update(command=command, cwd=str(cwd), log=str(log), started_utc=now())
    with Path(log).open("w") as stream:
        proc = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    record.update(exitcode=proc.returncode, finished_utc=now())
    if proc.returncode:
        raise RuntimeError(f"stage failed with exit {proc.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    import torch
    if args.gpus < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("gpus/batch-size must be positive; workers nonnegative")
    if torch.cuda.device_count() < args.gpus:
        parser.error(f"requested {args.gpus} devices, visible {torch.cuda.device_count()}")
    output = unique_output(args.output_dir or OUTPUT_ROOT / (
        "e38_tta_" + ("smoke_" if args.smoke else "") +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")))
    state = {"status": "preparing", "started_utc": now(), "checkpoint": str(CHECKPOINT),
             "source_config": str(SOURCE_CONFIG), "output_dir": str(output),
             "argv": sys.argv, "commit": subprocess.check_output(
                 ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "smoke": args.smoke, "gpus": args.gpus, "batch_size": args.batch_size,
             "stages": {}}
    manifest = output / "pipeline.json"
    save_json(manifest, state)
    print(f"output_dir={output}", flush=True)
    try:
        print("Checking fixed checkpoint SHA256...", flush=True)
        state["checkpoint_sha256"] = sha256(CHECKPOINT)
        if state["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise ValueError("fixed seed0 epoch30 checkpoint hash mismatch")
        state["source_config_sha256"] = sha256(SOURCE_CONFIG)
        if state["source_config_sha256"] != SOURCE_CONFIG_SHA256:
            raise ValueError("fixed source config hash mismatch")
        from labeldistill.config import load_and_resolve_config, save_config_artifacts
        bundle = load_and_resolve_config(str(SOURCE_CONFIG), [
            f"runtime.output_dir={output / 'config'}",
            f"runtime.gpus={args.gpus}", f"runtime.batch_size_per_device={args.batch_size}",
            f"data.num_workers={args.workers}", "evaluation.run_metrics=false",
        ], project_root=str(ROOT))
        c = bundle.config
        if c.experiment.seed != 0 or c.runtime.max_epochs != 36 or c.student.output_channels != 256:
            raise ValueError("unexpected fixed training configuration")
        if c.data.correct_rgb or not c.data.correct_depth or str(c.runtime.precision) != "16":
            raise ValueError("unexpected preprocessing/precision")
        sources = list((ROOT / "labeldistill/evaluation").glob("*.py"))
        sources.extend([ROOT / "scripts/dlc/evaluate_tta.sh"])
        save_config_artifacts(bundle, source_files=sources)
        state["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
        state["tracked_diff"] = subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=ROOT, text=True)
        state["status"] = "running"
        print(f"worker_log={output / 'worker.log'}", flush=True)
        config = output / "config/resolved_config.yaml"
        worker = [sys.executable]
        if args.gpus > 1:
            worker += ["-m", "torch.distributed.run", "--standalone",
                       f"--nproc_per_node={args.gpus}"]
        worker += ["-m", "labeldistill.evaluation.tta_worker",
                   "--config", str(config), "--checkpoint", str(CHECKPOINT),
                   "--output-dir", str(output)]
        worker += (["--indices", ",".join(map(str, SMOKE_INDICES))]
                   if args.smoke else ["--full-val"])
        state["stages"]["inference_and_merge"] = {}
        save_json(manifest, state)
        try:
            run_stage(worker, cwd=ROOT, log=output / "worker.log",
                      record=state["stages"]["inference_and_merge"])
        finally:
            save_json(manifest, state)
        state["summary"] = json.loads((output / "summary.json").read_text())
        state["status"] = "complete"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        state["finished_utc"] = now()
        save_json(manifest, state)
    print(json.dumps(state["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
