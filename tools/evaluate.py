#!/usr/bin/env python3
"""Evaluate LabelDistill from YAML or a checkpoint's resolved config."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path

import pytorch_lightning as pl

from labeldistill.builders import build_experiment, build_trainer
from labeldistill.config import (
    checkpoint_config_file,
    load_and_resolve_config,
    save_config_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', help='experiment or saved resolved YAML; optional with --checkpoint')
    parser.add_argument(
        '--checkpoint', help='weights to evaluate and config source when --config is omitted')
    parser.add_argument('--output-dir', help='evaluation artifact directory')
    parser.add_argument('--limit-test-batches', type=int)
    parser.add_argument(
        '--prediction-only', action='store_true',
        help='export predictions without running the full nuScenes metric suite')
    parser.add_argument(
        'overrides', nargs='*',
        help='OmegaConf dot-list overrides such as runtime.gpus=2')
    args = parser.parse_args()
    if not args.config and not args.checkpoint:
        parser.error('one of --config or --checkpoint is required')
    return args


def _has_override(overrides, key):
    return any(item.split('=', 1)[0] == key for item in overrides)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    with ExitStack() as stack:
        source_path = args.config
        if source_path is None:
            source_path = stack.enter_context(
                checkpoint_config_file(args.checkpoint))

        overrides = list(args.overrides)
        preview = load_and_resolve_config(
            source_path, overrides, project_root=str(project_root))
        checkpoint = args.checkpoint or preview.config.runtime.resume_from
        if not checkpoint:
            raise ValueError(
                'No checkpoint selected; pass --checkpoint or set runtime.resume_from')
        checkpoint = str(Path(checkpoint).expanduser().resolve())

        if args.output_dir:
            overrides.append(f'runtime.output_dir={Path(args.output_dir).resolve()}')
        elif not _has_override(overrides, 'runtime.output_dir'):
            evaluation_dir = Path(preview.config.runtime.output_dir) / 'evaluation'
            overrides.append(f'runtime.output_dir={evaluation_dir.resolve()}')
        if args.limit_test_batches is not None:
            overrides.append(
                f'runtime.limit_test_batches={args.limit_test_batches}')
            if not _has_override(overrides, 'evaluation.run_metrics'):
                overrides.append('evaluation.run_metrics=false')
        if args.prediction_only and not _has_override(
                overrides, 'evaluation.run_metrics'):
            overrides.append('evaluation.run_metrics=false')
        if not _has_override(overrides, 'runtime.resume_from'):
            overrides.append(f'runtime.resume_from={checkpoint}')

        bundle = load_and_resolve_config(
            source_path, overrides, project_root=str(project_root))
        config = bundle.config
        pl.seed_everything(config.experiment.seed, workers=True)
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            save_config_artifacts(bundle)
        experiment = build_experiment(bundle)
        trainer = build_trainer(config, experiment, for_evaluation=True)
        trainer.test(experiment, ckpt_path=checkpoint)


if __name__ == '__main__':
    main()
