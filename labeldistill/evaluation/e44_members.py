"""定位内部 test 的训练成员，并在 DLC 上按需补训同配方的 train+val 成员。"""
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

import yaml

from labeldistill.evaluation.tta_pipeline import ROOT, save_json, sha256

OUTPUTS = Path("/mnt/nas_data/guqiupeng/ROI_LABEL_DISTILL/outputs")
WEIGHTS = Path("/mnt/nas_data/guqiupeng/checkpoint_nes")
E42 = "convnext_wide_depth_trainval_s0_20261120_210956"
E37 = {1: "convnext_wide_depth_s1_20261117_012226", 2: "convnext_wide_depth_s2_20261117_012944"}
E42_CONFIG_SHA = "f4d9746668f7cc26c19b3244334abbfebaa0ec2045e6fa16cb42b5351b4f22f5"
_STOP_TIMEOUT = 30.0
E42_WEIGHTS = {
    "32": "cd27e8c30bf21b2896cc71b0ba41d4469ef134d9cc539eaf6e26eed14304d32e",
    "29": "800f20019fcf2a4a02c508c6308f0b67a37520a1ad317980f194520ff5802418",
    "35": "308878d061debfbf92c29263d72aca53868f489e212ad3de7643a0213a8579f0",
}


def middle_checkpoint(config: Path, early: Path) -> Path:
    """核对同一 E42 轨迹的训练配置和普通第 33 轮权重身份。"""
    import torch
    if (config.resolve() != (OUTPUTS / E42 / "resolved_config.yaml").resolve()
            or early.resolve() != (WEIGHTS / E42 / "epoch_29.ckpt").resolve()
            or sha256(config) != E42_CONFIG_SHA):
        raise ValueError("the intermediate checkpoint requires the E42 seed 0 run")
    path = early.with_name("epoch_32.ckpt")
    if sha256(path) != E42_WEIGHTS["32"]:
        raise ValueError(f"unexpected E42 intermediate checkpoint: {path}")
    saved = torch.load(path, mmap=True, map_location="cpu", weights_only=False)
    if saved.get("epoch") != 32 or saved.get("labeldistill_resolved_config") != yaml.safe_load(config.read_text()):
        raise ValueError(f"checkpoint epoch or embedded E42 config differs: {path}")
    return path


def _stop_training(process):
    # train_distill.sh execs the Python supervisor. Its training stage and DDP ranks
    # occupy a separate process group and must be stopped before killing the supervisor.
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        process.terminate()
        try:
            process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            # Only escalate the supervisor's own session-leading stage, never the DLC shell.
            try:
                stage_pids = children.read_text().split()
            except FileNotFoundError:
                stage_pids = []
            for item in stage_pids:
                pid = int(item)
                try:
                    if os.getpgid(pid) == pid:
                        os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.kill()
            process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _train(seed, directory, lock):
    config = ROOT / f"configs/experiments/convnext_wide_depth_trainval_s{seed}.yaml"
    run_id = directory.name.removeprefix(f"convnext_wide_depth_trainval_s{seed}_")
    command = ["bash", "scripts/dlc/train_distill.sh", "--train-only", "--config", str(config),
               "--run-id", run_id, "--checkpoint-root", str(WEIGHTS)]
    print(f"training member seed={seed} run={directory}", flush=True)
    # The training launcher must retain the shared lock if its parent is interrupted.
    process = subprocess.Popen(command, cwd=ROOT, pass_fds=(lock.fileno(),),
                               start_new_session=True, env={**os.environ, "PYTHON": sys.executable})
    try:
        code = process.wait()
    except BaseException:
        _stop_training(process)
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def _completed_or_train(seed, lock):
    name = f"convnext_wide_depth_trainval_s{seed}"
    for directory in sorted(OUTPUTS.glob(name + "_????????_??????")):
        manifest = directory / "pipeline.json"
        if manifest.is_file() and json.loads(manifest.read_text())["status"] == "complete":
            return directory
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    directory = OUTPUTS / f"{name}_{run_id}"
    _train(seed, directory, lock)
    return directory


def _verify(seed, directory, *, proxy):
    reference_dir = OUTPUTS / E42
    recipe_path = reference_dir / "resolved_config.yaml"
    if sha256(recipe_path) != E42_CONFIG_SHA:
        raise ValueError(f"E42 training config hash mismatch: {recipe_path}")
    reference = yaml.safe_load(recipe_path.read_text())
    state = json.loads((directory / "pipeline.json").read_text())
    if (state["status"] != "complete" or (directory / "train.exitcode").read_text().strip() != "0"
            or state["stages"]["train"][-1]["exitcode"] != 0):
        raise ValueError(f"training was not completed: {directory}")
    config_path = directory / "resolved_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    if sha256(config_path) != state["config_sha256"] or config != state["config"]:
        raise ValueError(f"training config disagrees with manifest: {directory}")

    split = "train" if proxy else "trainval"
    name = f"convnext_wide_depth_{'trainval_' if not proxy else ''}s{seed}"
    expected = copy.deepcopy(reference)
    expected["experiment"].update(name=name, seed=seed, description=config["experiment"]["description"])
    expected["data"]["split"] = split
    expected["runtime"].update(run_id=directory.name.removeprefix(name + "_"), output_dir=str(directory))
    if config != expected or directory.name != f"{name}_{config['runtime']['run_id']}":
        raise ValueError(f"member differs from the E42 recipe for seed {seed}, split {split}: {directory}")
    if not proxy:
        baseline = json.loads((reference_dir / "pipeline.json").read_text())
        for key in ("train_info_sha256", "val_info_sha256"):
            if state[key] != baseline[key]:
                raise ValueError(f"training index identity differs for {key}: {directory}")
    weights = WEIGHTS / directory.name
    checkpoints = tuple(weights / f"epoch_{epoch}.ckpt" for epoch in (29, 35))
    hashes = {}
    for epoch, checkpoint in zip((29, 35), checkpoints):
        key = str(epoch)
        digest = sha256(checkpoint)
        hashes[key] = digest
        if key == "29" or proxy:
            if state["checkpoints"][key] != digest:
                raise ValueError(f"checkpoint hash differs from training manifest: {checkpoint}")
        if seed == 0:
            if digest != E42_WEIGHTS[key]:
                raise ValueError(f"unexpected E42 checkpoint: {checkpoint}")
        elif key == "35" and not proxy:
            # The E42 train-only pipeline records epoch 29 only. Check the saved epoch and
            # embedded run identity before publishing the second checkpoint's fingerprint.
            import torch
            saved = torch.load(checkpoint, mmap=True, map_location="cpu", weights_only=False)
            if saved.get("epoch") != epoch or saved.get("labeldistill_resolved_config") != config:
                raise ValueError(f"checkpoint epoch or training config differs: {checkpoint}")
    audit = {"seed": seed, "split": split, "smoke_proxy": proxy, "run": str(directory),
             "config_sha256": state["config_sha256"], "checkpoints": hashes}
    return (config_path, *checkpoints), audit


def member(seed: int, root: Path, *, smoke: bool) -> tuple[Path, Path, Path]:
    """返回训练配置和第 30、36 轮普通权重；正式作业共用同一 seed 的训练。"""
    if seed not in (0, 1, 2):
        raise ValueError(f"unsupported member seed: {seed}")
    if seed == 0 or smoke:
        directory = OUTPUTS / (E42 if seed == 0 else E37[seed])
        result, audit = _verify(seed, directory, proxy=seed != 0)
    else:
        cache = OUTPUTS / ".e44_members"
        cache.mkdir(exist_ok=True)
        with (cache / f"seed{seed}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            directory = _completed_or_train(seed, lock)
            result, audit = _verify(seed, directory, proxy=False)
            identity = cache / (directory.name + ".json")
            if identity.is_file():
                if json.loads(identity.read_text()) != audit:
                    raise ValueError(f"previously published member changed: {directory}")
            else:
                save_json(identity, audit)
    save_json(root / f"member_seed{seed}.json", audit)
    return result
