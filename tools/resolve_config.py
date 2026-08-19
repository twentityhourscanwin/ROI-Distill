"""Resolve and validate a LabelDistill YAML without constructing a model."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from labeldistill.config import load_and_resolve_config, save_config_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML path")
    parser.add_argument(
        "--output-dir",
        help="optional audit output; defaults to runtime.output_dir",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="validate and print only; do not create audit files",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides such as runtime.gpus=8",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_and_resolve_config(args.config, args.overrides)
    print(OmegaConf.to_yaml(bundle.config, resolve=True))
    if bundle.config_diff:
        print("Config diff:")
        for path, change in bundle.config_diff.items():
            print(f"  {path}: {change['base']!r} -> {change['resolved']!r}")
    if not args.no_write:
        target = save_config_artifacts(bundle, args.output_dir)
        print(f"Config artifacts written to {target}")


if __name__ == "__main__":
    main()
