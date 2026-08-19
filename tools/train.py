"""Train LabelDistill from a validated YAML experiment config."""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl

from labeldistill.builders import build_experiment, build_trainer
from labeldistill.config import load_and_resolve_config, save_config_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='experiment YAML path')
    parser.add_argument(
        'overrides', nargs='*',
        help='OmegaConf dot-list overrides such as runtime.gpus=8')
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    bundle = load_and_resolve_config(
        args.config, args.overrides, project_root=str(project_root))
    config = bundle.config
    pl.seed_everything(config.experiment.seed, workers=True)
    # Lightning's subprocess DDP launcher re-enters this module once per rank.
    # Persist shared audit artifacts only from local rank zero.
    if int(os.environ.get('LOCAL_RANK', '0')) == 0:
        save_config_artifacts(bundle)
    experiment = build_experiment(bundle)
    trainer = build_trainer(config, experiment)
    trainer.fit(experiment, ckpt_path=config.runtime.resume_from)


if __name__ == '__main__':
    main()
