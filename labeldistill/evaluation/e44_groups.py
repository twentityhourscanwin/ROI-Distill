"""Run a fixed internal nuScenes test comparison and publish its submission archive."""
import argparse
import fcntl
import hashlib
import json
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from labeldistill.evaluation.detection_submission import INDEX, INPUTS, TEST_ROOT, prepare_submission, write_package
from labeldistill.evaluation.tta_combinations import terminate
from labeldistill.evaluation.tta_pipeline import OUTPUT_ROOT, ROOT, now, save_json, sha256

REFERENCE = OUTPUT_ROOT / "e43_test_full_20261123T084533_339750Z"
REFERENCE_ZIP = "f403f25e98f3bdac591ea6292c88cf1dc8c71597a56e7775ff9fcb6894ae8dae"
BODY_SHA = "70d52e4365d7e7db0299b7713ba12f8732c854087525f880c189909d93286c61"
# Each member is (seed, one-based epoch, inference entry).
GROUPS = {
    "A": (),
    "C": ((0, 30, "ordinary"),),
    "B": ((0, 36, "ordinary"),),
    "D": ((0, 30, "ordinary"), (0, 36, "ordinary")),
    "E": ((0, 30, "ordinary"), (1, 30, "ordinary")),
    "F": ((0, 36, "tta"),),
    "G": ((0, 30, "tta"), (0, 36, "tta")),
    "H": ((1, 30, "tta"),),
    "I": ((0, 30, "tta"), (1, 30, "tta")),
    "J": ((0, 30, "tta"), (1, 30, "tta"), (2, 30, "tta")),
    "K": ((0, 30, "tta"), (0, 33, "tta"), (0, 36, "tta")),
}

def read(path):
    return json.loads(path.read_bytes())

def align(prediction, tokens):
    return {"meta": prediction["meta"], "results": {t: prediction["results"][t] for t in tokens}}

def check(payload, official, *, smoke):
    tokens = set(payload["results"])
    if not tokens or not tokens <= official or (not smoke and tokens != official):
        raise ValueError("incorrect test coverage")
    if smoke and tokens == official:
        raise ValueError("smoke unexpectedly covers all test samples")
    for token, boxes in payload["results"].items():
        if len(boxes) > 500 or any(b["sample_token"] != token for b in boxes):
            raise ValueError(f"invalid frame {token}")

def combine(group, predictions):
    if group in ("B", "C"):
        body, plain = predictions
        result, clipped = prepare_submission(body, plain, list(plain["results"]))
        if clipped:
            raise ValueError(f"routing would truncate {clipped} frames")
        return result
    if len(predictions) == 1:
        return predictions[0]
    from labeldistill.evaluation import e44_fusion
    if len(predictions) == 2:
        return e44_fusion.fuse(*predictions)
    return e44_fusion.fuse_three(*predictions)

def implementation_digest():
    # The launcher differs across experiment branches. Hash the inference sources
    # themselves so jobs on those branches can share identical predictions.
    paths = sorted(p for p in (ROOT / "labeldistill").rglob("*.py")
                   if p.name not in {"e44_groups.py", "e44_members.py", "e44_fusion.py"})
    paths += [ROOT / "tools/evaluate.py", ROOT / "scripts/dlc/evaluate_history.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\x00")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()

def cached_prediction(args, cfg, ckpt, method, code_digest, index_digest):
    from labeldistill.evaluation import e44_ordinary, e44_tta
    identity = dict(entry=method, config=sha256(cfg), checkpoint=sha256(ckpt), code=code_digest,
                    index=index_digest, smoke=args.smoke, gpus=args.gpus, batch_size=args.batch_size,
                    workers=args.workers)
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = OUTPUT_ROOT / "e44_member_cache" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "compute.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = cache / "complete.json"
        if done.is_file():
            record = read(done)
            if record["identity"] != identity or sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError(f"invalid prediction cache: {done}")
        else:
            attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=cache))
            exporter = e44_ordinary.export if method == "ordinary" else e44_tta.export
            path = exporter(cfg, ckpt, attempt, gpus=args.gpus, batch_size=args.batch_size,
                            workers=args.workers, smoke=args.smoke)
            record = dict(identity=identity, path=str(path), sha256=sha256(path))
            save_json(done, record)
    return Path(record["path"]), dict(record, config_path=str(cfg), checkpoint_path=str(ckpt))

def run(args, output, state):
    from labeldistill.evaluation.e44_members import member
    meta = read(TEST_ROOT / "v1.0-test/sample.json")
    official = {s["token"] for s in meta}
    if len(meta) != len(official) or len(official) != 6008:
        raise ValueError("unexpected official test metadata")
    state["index_sha256"] = sha256(INDEX)
    if state["index_sha256"] != INPUTS[INDEX]:
        raise ValueError("E43 test index changed")
    archive = REFERENCE / "submission.zip"
    if args.group == "A":
        if sha256(archive) != REFERENCE_ZIP:
            raise ValueError("E43 archive changed")
        state.update(archive=str(archive), archive_sha256=REFERENCE_ZIP, samples=6008)
        return
    from zipfile import ZipFile
    plans = GROUPS[args.group]
    held = {}
    predictions = []
    code_digest = implementation_digest()
    # Start seed 2 first so J and the seed 1 experiment can train in parallel.
    for seed in sorted({s for s, _, _ in plans if s}, reverse=True):
        held[seed] = member(seed, output, smoke=args.smoke)
    for seed, epoch, method in plans:
        if (seed, epoch, method) == (0, 30, "tta"):
            if sha256(archive) != REFERENCE_ZIP:
                raise ValueError("E43 archive changed")
            with ZipFile(archive) as handle, handle.open("results_nusc.json") as fp:
                prediction = json.load(fp)
            check(prediction, official, smoke=False)
            state["members"].append(dict(seed=seed, epoch=epoch, method=method, archive=str(archive),
                                         sha256=REFERENCE_ZIP))
        else:
            if seed not in held:
                held[seed] = member(seed, output, smoke=args.smoke)
            cfg, c30, c36 = held[seed]
            if epoch == 33:
                from labeldistill.evaluation.e44_members import middle_checkpoint
                checkpoint = middle_checkpoint(cfg, c30)
            else:
                checkpoint = {30: c30, 36: c36}[epoch]
            path, receipt = cached_prediction(args, cfg, checkpoint, method, code_digest,
                                              state["index_sha256"])
            prediction = read(path)
            check(prediction, official, smoke=args.smoke)
            state["members"].append(dict(seed=seed, epoch=epoch, method=method, **receipt))
            # Standalone scores give each fusion a directly comparable single-member baseline.
            if args.group in ("D", "E"):
                endpoint = output / f"ordinary_seed{seed}_epoch{epoch}"
                endpoint.mkdir()
                member_zip = write_package(endpoint, prediction, smoke=args.smoke)
                state["members"][-1].update(archive=str(member_zip), archive_sha256=sha256(member_zip))
        predictions.append(prediction)
        save_json(output / "pipeline.json",state)
    if args.smoke and len(plans) > 1 and plans[0] == (0, 30, "tta"):
        predictions[0] = align(predictions[0], predictions[1]["results"])
    if args.group in ("B", "C"):
        body_file = REFERENCE / "groups/A11/results_nusc.json"
        if sha256(body_file) != BODY_SHA:
            raise ValueError("E43 eight-class source changed")
        body = read(body_file)
        check(body, official, smoke=False)
        predictions.insert(0, align(body, predictions[0]["results"]))
        state.update(body=str(body_file), body_sha256=BODY_SHA)
    result = combine(args.group, predictions)
    check(result, official, smoke=args.smoke)
    archive = write_package(output, result, smoke=args.smoke)
    state.update(archive=str(archive), archive_sha256=sha256(archive), samples=len(result["results"]),
                 boxes=sum(map(len,result["results"].values())), max_per_frame=max(map(len,result["results"].values())))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus",type=int,default=16)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if min(args.gpus,args.batch_size)<1 or args.workers<0:
        parser.error("invalid GPU, batch or worker count")
    if not args.smoke and args.gpus != 16 and any(seed for seed,_,_ in GROUPS[args.group]):
        parser.error("train+val seed training uses 16 GPUs")
    mode = "smoke" if args.smoke else "full"
    output = Path(tempfile.mkdtemp(prefix=f"e44_{args.group}_{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_", dir=OUTPUT_ROOT))
    state = dict(status="running", group=args.group, started_utc=now(), args=vars(args), members=[],
                 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    print(f"output_dir={output}",flush=True)
    save_json(output / "pipeline.json", state)
    try:
        run(args,output,state)
        state["status"]="complete"
        print(f"archive={state['archive']} sha256={state['archive_sha256']}",flush=True)
    except BaseException as exc:
        state.update(status="failed",error=repr(exc))
        raise
    finally:
        state["finished_utc"]=now()
        save_json(output / "pipeline.json",state)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    main()
