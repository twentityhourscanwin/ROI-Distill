"""Train LabelDistill from a validated YAML experiment config."""

import argparse
from contextlib import ExitStack
import os
from pathlib import Path

import pytorch_lightning as pl

from labeldistill.builders import build_experiment, build_trainer
from labeldistill.config import (
    checkpoint_config_file,
    checkpoint_directory,
    load_training_bundle,
    save_config_artifacts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help='experiment or saved resolved YAML path')
    source.add_argument(
        '--checkpoint', help='resume from its embedded resolved config')
    parser.add_argument(
        'overrides', nargs='*',
        help='OmegaConf dot-list overrides such as runtime.gpus=8')
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    with ExitStack() as stack:
        source_path = args.config
        overrides = list(args.overrides)
        if args.checkpoint:
            source_path = stack.enter_context(
                checkpoint_config_file(args.checkpoint))
            if not any(item.startswith('runtime.resume_from=') for item in overrides):
                overrides.append(f'runtime.resume_from={Path(args.checkpoint).resolve()}')
        bundle = load_training_bundle(source_path, overrides, project_root)
        config = bundle.config
        pl.seed_everything(config.experiment.seed, workers=True)
        # Lightning's subprocess DDP launcher re-enters this module once per rank.
        # Persist shared audit artifacts only from local rank zero.
        if int(os.environ.get('LOCAL_RANK', '0')) == 0:
            save_config_artifacts(bundle)
            print(f'run_id={config.runtime.run_id}', flush=True)
            print(f'output_dir={Path(config.runtime.output_dir).resolve()}',
                  flush=True)
            print(f'checkpoint_dir={checkpoint_directory(config).resolve()}',
                  flush=True)
        experiment = build_experiment(bundle)
        trainer = build_trainer(config, experiment)
        trainer.fit(experiment, ckpt_path=config.runtime.resume_from)


if __name__ == '__main__':
    main()
