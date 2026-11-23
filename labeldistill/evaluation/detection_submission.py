"""Export the fixed E42 checkpoint on nuScenes test with the E40 box policy."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from labeldistill.evaluation.tta_class_policy import route_boxes
from labeldistill.evaluation.tta_combinations import terminate
from labeldistill.evaluation.tta_pipeline import (
    OUTPUT_ROOT, ROOT, now, save_json, sha256, unique_output,
)

RUN = "convnext_wide_depth_trainval_s0_20261120_210956"
CHECKPOINT = Path("/mnt/nas_data/guqiupeng/checkpoint_nes") / RUN / "epoch_29.ckpt"
CONFIG = OUTPUT_ROOT.parent / RUN / "resolved_config.yaml"
TEST_ROOT = Path("/mnt/nas_data/guqiupeng/test_nus")
INDEX = TEST_ROOT / "nuscenes_infos_test.pkl"
INPUTS = {
    CHECKPOINT: "800f20019fcf2a4a02c508c6308f0b67a37520a1ad317980f194520ff5802418",
    CONFIG: "f4d9746668f7cc26c19b3244334abbfebaa0ec2045e6fa16cb42b5351b4f22f5",
    INDEX: "b6cb86914f3017d30b2d26ab0702ad2c5ef1102fad977cdc9f108148a566e4ca",
}


def prepare_submission(body, base, tokens, limit=500):
    # Route before applying the joint cap, keeping E40 order for ordinary samples.
    result = route_boxes(body, base, tokens, math.inf)
    clipped = 0
    for token, boxes in result["results"].items():
        for box in boxes:
            if box["sample_token"] != token or not 0 <= box["detection_score"] <= 1:
                raise ValueError(f"invalid detection for {token}")
        if len(boxes) > limit:
            result["results"][token] = sorted(
                boxes, key=lambda box: box["detection_score"], reverse=True)[:limit]
            clipped += 1
    return result, clipped


def write_package(out, result, *, smoke):
    json_path = out / "results_nusc.json"
    with json_path.open("x") as stream:
        json.dump(result, stream, allow_nan=False, separators=(",", ":"))
    name = "smoke_subset.zip" if smoke else "submission.zip"
    tmp = out / (name + ".tmp")
    with ZipFile(tmp, "x", compression=ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname=json_path.name)
    with ZipFile(tmp) as archive:
        if archive.testzip() is not None:
            raise ValueError("corrupt submission archive")
    return tmp.replace(out / name)


def run_worker(command, *, cwd, log, record):
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait()
        except BaseException:
            # Give torchrun time to forward termination to each rank.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            record.update(exitcode=process.returncode, finished_utc=now())
        if code:
            raise subprocess.CalledProcessError(code, command)


def main():
    import torch
    from labeldistill.config import load_and_resolve_config, save_config_artifacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.gpus <= torch.cuda.device_count() or args.batch_size < 1 or args.workers < 0:
        parser.error("invalid GPU count, batch size or workers")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    mode = "smoke" if args.smoke else "full"
    output = unique_output(OUTPUT_ROOT / f"e43_test_{mode}_{timestamp}")
    manifest = output / "pipeline.json"
    state = dict(status="running", mode=mode, started_utc=now(), args=vars(args),
                 commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                text=True).strip(), inputs={}, worker={})
    print(f"output_dir={output}", flush=True)
    save_json(manifest, state)
    try:
        for path, expected in INPUTS.items():
            digest = sha256(path)
            if digest != expected:
                raise ValueError(f"unexpected input SHA-256: {path}")
            state["inputs"][str(path)] = digest
        official = json.loads((TEST_ROOT / "v1.0-test/sample.json").read_text())
        tokens = {s["token"] for s in official}
        if len(tokens) != 6008 or len(official) != len(tokens):
            raise ValueError("unexpected official test metadata")
        config_dir = output / "config"
        bundle = load_and_resolve_config(str(CONFIG), [
            f"runtime.output_dir={config_dir}", f"runtime.gpus={args.gpus}",
            f"runtime.batch_size_per_device={args.batch_size}", f"data.num_workers={args.workers}",
            f"data.root={TEST_ROOT}", f"data.val_info={INDEX}", "evaluation.run_metrics=false",
        ], project_root=str(ROOT))
        save_config_artifacts(bundle)
        worker = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                  f"--nproc_per_node={args.gpus}", "-m", "labeldistill.evaluation.tta_worker",
                  "--export-test", mode, "--config", str(config_dir / "resolved_config.yaml"),
                  "--checkpoint", str(CHECKPOINT), "--output-dir", str(output)]
        print(f"worker_log={output / 'worker.log'}", flush=True)
        state["worker"].update(command=worker, log=str(output / "worker.log"),
                               cwd=str(ROOT), started_utc=now())
        save_json(manifest, state)
        run_worker(worker, cwd=ROOT, log=output / "worker.log", record=state["worker"])
        selected = json.loads((output / "views.json").read_text())["tokens"]
        if len(set(selected)) != len(selected) or not set(selected) <= tokens:
            raise ValueError("invalid test tokens")
        if not args.smoke and set(selected) != tokens:
            raise ValueError("incomplete official test split")
        def load_group(name):
            return json.loads((output / "groups" / name / "results_nusc.json").read_bytes())
        result, clipped = prepare_submission(load_group("A11"), load_group("B0"), selected)
        archive = write_package(output, result, smoke=args.smoke)
        state.update(status="complete", samples=len(selected), capped_samples=clipped,
                     archive=str(archive), archive_sha256=sha256(archive))
        print(f"archive={archive}", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_utc"] = now()
        save_json(manifest, state)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    main()
