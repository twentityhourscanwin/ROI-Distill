"""复用 E38 的 A11 基线，评估九视图与行人、交通锥取原图框的三个新格。"""
import hashlib
import json
import signal

from labeldistill.evaluation.tta_combinations import (
    FULL_DIGEST, SMOKE_DIGEST, evaluate_group, load_replay_inputs, merge_shards,
    official_metrics, run_cli, terminate,
)
from labeldistill.evaluation.tta_core import GROUPS
from labeldistill.evaluation.tta_pipeline import (
    CHECKPOINT_SHA256, SOURCE_CONFIG_SHA256, now, save_json, sha256,
)
from labeldistill.evaluators.det_evaluators import DetNuscEvaluator


RELEASE_TAG = "exp/cnb-tta-class-policy-20261120"
NINE = tuple(v for v in GROUPS["A11"] if v not in GROUPS["S3"][1:])
B0_CLASSES = ("pedestrian", "traffic_cone")


def checked_json(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"input checksum mismatch: {path}")
    return json.loads(raw)


def route_boxes(body, base, tokens, limit):
    expected = set(tokens)
    if body["results"].keys() != expected or base["results"].keys() != expected:
        raise ValueError("prediction sample tokens differ from E38")
    if body["meta"] != base["meta"]:
        raise ValueError("prediction modalities differ")
    results = {}
    for token in tokens:
        boxes = [b for b in body["results"][token] if b["detection_name"] not in B0_CLASSES]
        boxes.extend(b for b in base["results"][token] if b["detection_name"] in B0_CLASSES)
        if len(boxes) > limit:
            raise ValueError(f"official per-sample limit exceeded: {token}")
        results[token] = boxes
    return {"meta": body["meta"], "results": results}


def evaluate_mixed(directory, path, digest, base, tokens, config, full):
    from labeldistill.evaluation.tta_pipeline import ROOT

    evaluator = DetNuscEvaluator(class_names=list(config.classes.names),
                                 data_root=str(ROOT / config.data.root), output_dir=str(directory))
    payload = route_boxes(checked_json(path, digest), base, tokens,
                          evaluator.eval_detection_configs.max_boxes_per_sample)
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / "results_nusc.json"
    with result.open("w") as stream:
        json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
    del payload
    audit = {"samples": len(tokens), "results_sha256": sha256(result)}
    if full:
        evaluator._evaluate_single(str(result))
        audit.update(official_metrics(directory, config))
    return audit


def replay(cache, output, smoke, state):
    source, tokens, config, head, ranks = load_replay_inputs(cache, smoke)
    refs = source["groups"]
    groups = output / "groups"
    state.update(samples=len(tokens), cache_metadata_sha256=SMOKE_DIGEST if smoke else FULL_DIGEST,
                 checkpoint_sha256=CHECKPOINT_SHA256, source_config_sha256=SOURCE_CONFIG_SHA256)

    def phase(name):
        state["phase"] = name
        save_json(output / "pipeline.json", state)

    def record(name, audit, views, selected):
        directory = groups / name
        audit.update(output_dir=str(directory), constituent_views=views, b0_classes=selected)
        if not smoke:
            audit["delta_vs_A11_pp"] = {
                k: 100 * (audit[k] - refs["A11"][k]) for k in ("mAP", "NDS")
            }
        save_json(directory / "val_audit.json", audit)
        return audit

    phase("verify/reused")
    if not smoke:
        for name in ("A11", "B0"):
            path = cache / "groups" / name / "metrics_summary.json"
            checked_json(path, refs[name]["metrics_sha256"])
    phase("merge/A9")
    merge_shards(cache, output, head, tokens, ranks, {"A9": NINE})
    phase("evaluate/A9")
    nine = evaluate_group(groups / "A9", len(ranks), tokens, config, not smoke)
    results = {"A9": record("A9", nine, NINE, ())}

    phase("verify/B0_predictions")
    base_dir = cache / "groups" / "B0"
    base = checked_json(base_dir / "results_nusc.json", refs["B0"]["results_sha256"])
    for body, views, directory, audit in (
        ("A11", GROUPS["A11"], cache / "groups" / "A11", refs["A11"]),
        ("A9", NINE, groups / "A9", nine),
    ):
        name = f"{body}_B0_ped_cone"
        phase(f"evaluate/{name}")
        mixed = evaluate_mixed(groups / name, directory / "results_nusc.json",
                               audit["results_sha256"], base, tokens, config, not smoke)
        results[name] = record(name, mixed, views, B0_CLASSES)
    return {"status": "complete", "full_val": not smoke, "samples": len(tokens),
            "source_cache": str(cache), "reused_A11": refs["A11"], "groups": results,
            "finished_utc": now()}


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    run_cli(__doc__, RELEASE_TAG, "e40_tta", replay)
