"""Persist reproducible config and environment artifacts."""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from omegaconf import OmegaConf

from .loader import ConfigBundle
from .validation import sha256_file


_J4_CRITICAL_SOURCES = (
    "labeldistill/exps/nuscenes/base_exp.py",
    "labeldistill/acceptance/snapshot.py",
    "labeldistill/presets/j4.py",
    "labeldistill/presets/convnextb.py",
    "labeldistill/layers/backbones/convnext_backbone.py",
    "labeldistill/builders/experiment_builder.py",
    "labeldistill/builders/trainer_builder.py",
    "labeldistill/experiments/j4.py",
    "labeldistill/datasets/nusc_det_dataset_lidar.py",
    "labeldistill/models/lidardistill.py",
    "labeldistill/models/teacher_proposal_decoder.py",
    "labeldistill/layers/heads/kd_head.py",
    "labeldistill/refine_head/target_assigner/roi_distill.py",
    "labeldistill/refine_head/target_assigner/adaptive_gt_scaler_v3.py",
    "labeldistill/refine_head/target_assigner/proposal_center_only_scaler.py",
    "labeldistill/refine_head/target_assigner/quality_aware_mask_v3.py",
    "labeldistill/refine_head/target_assigner/raw_gaussian_feature_loss.py",
    "labeldistill/config/schema.py",
    "labeldistill/config/loader.py",
    "labeldistill/config/checkpoint.py",
    "labeldistill/config/run.py",
    "labeldistill/config/validation.py",
    "labeldistill/callbacks/ema.py",
    "tools/train.py",
    "tools/evaluate.py",
)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit(project_root: Optional[Path]) -> str:
    if project_root is None:
        return "unavailable"
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def _torch_cuda_version() -> str:
    try:
        import torch

        return str(torch.version.cuda or "not-available")
    except Exception:
        return "not-available"


def collect_environment(
    source_files: Iterable[Path] = (),
    project_root: Optional[Path] = None,
) -> str:
    lines = [
        f"python={platform.python_version()}",
        f"python_executable={sys.executable}",
        f"platform={platform.platform()}",
        f"pytorch={_package_version('torch')}",
        f"pytorch_lightning={_package_version('pytorch-lightning')}",
        f"cuda={_torch_cuda_version()}",
        f"mmengine={_package_version('mmengine')}",
        f"mmdet3d={_package_version('mmdet3d')}",
        f"omegaconf={_package_version('omegaconf')}",
        f"pyyaml={_package_version('pyyaml')}",
        f"git_commit={_git_commit(project_root)}",
    ]
    for source_file in sorted(Path(path).resolve() for path in source_files):
        if source_file.is_file():
            lines.append(f"source_sha256[{source_file}]={sha256_file(source_file)}")
    return "\n".join(lines) + "\n"


def save_config_artifacts(
    bundle: ConfigBundle,
    output_dir: Optional[str] = None,
    *,
    source_files: Iterable[Path] = (),
) -> Path:
    """Write the immutable inputs and resolved values needed for an audit."""
    target = Path(output_dir or bundle.config.runtime.output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    # Keep the user's exact source YAML, including comments and _base_.
    source_target = target / "source_config.yaml"
    if bundle.source_path != source_target:
        shutil.copyfile(bundle.source_path, source_target)
    OmegaConf.save(bundle.config, target / "resolved_config.yaml", resolve=True)
    OmegaConf.save(
        OmegaConf.create({"changes": bundle.config_diff}),
        target / "config_diff.yaml",
        resolve=True,
    )
    audited_sources = set(Path(path).resolve() for path in source_files)
    audited_sources.update(bundle.config_chain)
    audited_sources.update(
        bundle.project_root / relative_path for relative_path in _J4_CRITICAL_SOURCES
    )
    (target / "environment.txt").write_text(
        collect_environment(audited_sources, bundle.project_root), encoding="utf-8"
    )
    return target
