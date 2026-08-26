"""Read the immutable resolved experiment config embedded in checkpoints."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

from omegaconf import OmegaConf
import torch

from .loader import ConfigLoadError


def read_checkpoint_config(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ConfigLoadError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("labeldistill_resolved_config")
    schema_version = checkpoint.get("labeldistill_schema_version")
    if config is None or schema_version is None:
        raise ConfigLoadError(
            f"{checkpoint_path}: checkpoint predates embedded resolved config; "
            "supply --config explicitly")
    config_schema = config.get("schema_version")
    if schema_version != config_schema:
        raise ConfigLoadError(
            f"{checkpoint_path}: checkpoint schema metadata {schema_version!r} "
            f"does not match embedded config {config_schema!r}")
    return OmegaConf.create(config)


@contextmanager
def checkpoint_config_file(checkpoint_path):
    """Materialize embedded config under the loader's reserved filename."""
    config = read_checkpoint_config(checkpoint_path)
    with tempfile.TemporaryDirectory(prefix="labeldistill_checkpoint_config_") as tmp:
        path = Path(tmp) / "resolved_config.yaml"
        OmegaConf.save(config, path, resolve=True)
        yield path
