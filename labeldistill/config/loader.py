"""OmegaConf loader with recursive ``_base_`` inheritance and CLI overrides."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf

from .schema import LabelDistillConfig
from .validation import validate_config


class ConfigLoadError(ValueError):
    """Raised for invalid inheritance or config file structure."""


@dataclass(frozen=True)
class ConfigBundle:
    config: DictConfig
    project_root: Path
    source_path: Path
    source_config: DictConfig
    base_config: DictConfig
    config_diff: Dict[str, Dict[str, Any]]
    config_chain: Tuple[Path, ...]


def _normalise_bases(value: Any, source_path: Path) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, ListConfig)) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    raise ConfigLoadError(f"{source_path}: _base_ must be a string or list of strings")


def _load_recursive(
    source_path: Path,
    stack: Tuple[Path, ...],
) -> Tuple[DictConfig, DictConfig, Tuple[Path, ...]]:
    source_path = source_path.resolve()
    if source_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, source_path))
        raise ConfigLoadError(f"Config inheritance cycle: {cycle}")
    if not source_path.is_file():
        raise ConfigLoadError(f"Config file does not exist: {source_path}")

    raw = OmegaConf.load(source_path)
    if not isinstance(raw, DictConfig):
        raise ConfigLoadError(f"{source_path}: top-level YAML value must be a mapping")

    own = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
    base_entries = _normalise_bases(own.pop("_base_", None), source_path)
    merged_bases = OmegaConf.create({})
    chain: List[Path] = []
    for entry in base_entries:
        base_path = Path(entry).expanduser()
        if not base_path.is_absolute():
            base_path = source_path.parent / base_path
        resolved_base, _, base_chain = _load_recursive(
            base_path, (*stack, source_path)
        )
        merged_bases = OmegaConf.merge(merged_bases, resolved_base)
        for path in base_chain:
            if path not in chain:
                chain.append(path)

    merged = OmegaConf.merge(merged_bases, own)
    chain.append(source_path)
    return merged, merged_bases, tuple(chain)


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "derived":
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, child_prefix))
    else:
        result[prefix] = deepcopy(value)
    return result


def config_diff(base: DictConfig, resolved: DictConfig) -> Dict[str, Dict[str, Any]]:
    """Return leaf-level changes between a merged base and final config."""
    if not base:
        return {}
    base_data = OmegaConf.to_container(base, resolve=True)
    resolved_data = OmegaConf.to_container(resolved, resolve=True)
    base_flat = _flatten(base_data)
    resolved_flat = _flatten(resolved_data)
    missing = "<missing>"
    changes = {}
    for path in sorted(set(base_flat) | set(resolved_flat)):
        before = base_flat.get(path, missing)
        after = resolved_flat.get(path, missing)
        if before != after:
            changes[path] = {"base": before, "resolved": after}
    return changes


def load_and_resolve_config(
    path: str,
    overrides: Optional[Sequence[str]] = None,
    *,
    project_root: Optional[str] = None,
    require_checkpoint: bool = True,
) -> ConfigBundle:
    """Load, merge, type-check, validate, and freeze an experiment config."""
    source_path = Path(path).expanduser().resolve()
    source_config = OmegaConf.load(source_path)
    merged, merged_bases, chain = _load_recursive(source_path, ())

    schema = OmegaConf.structured(LabelDistillConfig)
    try:
        resolved = OmegaConf.merge(schema, merged)
        if overrides:
            resolved = OmegaConf.merge(resolved, OmegaConf.from_dotlist(list(overrides)))
        OmegaConf.resolve(resolved)
    except Exception as exc:
        raise ConfigLoadError(f"Failed to compose {source_path}: {exc}") from exc

    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    else:
        root = next(
            (
                ancestor
                for ancestor in source_path.parents
                if (ancestor / "labeldistill").is_dir()
            ),
            source_path.parent,
        )
    validate_config(resolved, root, require_checkpoint=require_checkpoint)
    return ConfigBundle(
        config=resolved,
        project_root=root,
        source_path=source_path,
        source_config=source_config,
        base_config=merged_bases,
        config_diff=config_diff(merged_bases, resolved),
        config_chain=chain,
    )
