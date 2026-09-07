"""Create collision-free, traceable directories for training runs."""

from datetime import datetime
import os
from pathlib import Path
import re
from typing import Optional, Sequence


RUN_ID_ENV = "LABELDISTILL_RUN_ID"
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}")


def generate_run_id(now: Optional[datetime] = None) -> str:
    """Return a sortable server-local timestamp for a new training run."""
    current = now or datetime.now().astimezone()
    return current.strftime("%Y%m%d_%H%M%S")


def validate_run_id(run_id: str) -> str:
    """Reject values that could escape or ambiguously nest a run directory."""
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must use YYYYMMDD_HHMMSS, for example "
            "20260828_153012")
    return run_id


def dated_directory(
    directory: str,
    run_id: str,
    *,
    previous_run_id: Optional[str] = None,
) -> str:
    """Append ``run_id`` to a directory name, replacing an older run id."""
    validate_run_id(run_id)
    path = Path(directory).expanduser()
    name = path.name
    if not name:
        raise ValueError("training output directory must have a final name")
    if previous_run_id:
        previous_suffix = f"_{previous_run_id}"
        if name.endswith(previous_suffix):
            name = name[:-len(previous_suffix)]
    suffix = f"_{run_id}"
    if not name.endswith(suffix):
        name += suffix
    return str(path.with_name(name))


def checkpoint_directory(config) -> Path:
    """Return the checkpoint directory for the resolved run config."""
    name = str(config.experiment.name)
    run_id = config.runtime.run_id
    if run_id:
        validate_run_id(str(run_id))
        name = f"{name}_{run_id}"
    return Path(config.checkpoint.root_dir).expanduser() / name


def replace_override(
    overrides: Sequence[str], key: str, value: str
) -> list[str]:
    """Replace one OmegaConf dot-list key without duplicating it."""
    prefix = f"{key}="
    return [item for item in overrides if not item.startswith(prefix)] + [
        f"{key}={value}"
    ]


def load_training_bundle(
    source_path,
    overrides,
    project_root,
    *,
    require_checkpoint: bool = True,
):
    """Resolve one stable dated directory for every process in a new run."""
    # Import locally to keep the low-level path helpers independent of the
    # loader and avoid a package-initialisation cycle.
    from .loader import load_and_resolve_config

    explicit_run_id = next(
        (
            item.split("=", 1)[1].strip("\"'")
            for item in overrides
            if item.startswith("runtime.run_id=")
        ),
        None,
    )
    if explicit_run_id is not None:
        validate_run_id(explicit_run_id)
        overrides = replace_override(
            overrides, "runtime.run_id", f'"{explicit_run_id}"')

    preview = load_and_resolve_config(
        source_path,
        overrides,
        project_root=str(project_root),
        require_checkpoint=require_checkpoint,
    )
    if preview.config.runtime.resume_from is not None:
        return preview

    run_id = explicit_run_id or os.environ.get(RUN_ID_ENV)
    if run_id is None:
        run_id = generate_run_id()
    os.environ[RUN_ID_ENV] = run_id

    run_output_dir = dated_directory(
        preview.config.runtime.output_dir,
        run_id,
        previous_run_id=preview.config.runtime.run_id,
    )
    # Quote the numeric-looking value so OmegaConf does not parse the
    # underscores as integer digit separators.
    run_overrides = replace_override(
        overrides, "runtime.run_id", f'"{run_id}"')
    run_overrides = replace_override(
        run_overrides, "runtime.output_dir", run_output_dir)
    return load_and_resolve_config(
        source_path,
        run_overrides,
        project_root=str(project_root),
        require_checkpoint=require_checkpoint,
    )
