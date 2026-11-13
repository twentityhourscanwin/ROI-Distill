"""Train LabelDistill from a validated YAML experiment config."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from labeldistill.builders import build_experiment, build_trainer
from labeldistill.callbacks.resume_ema import ResumeEMA
from labeldistill.config import (
    checkpoint_config_file,
    checkpoint_directory,
    load_and_resolve_config,
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
        '--resume-state',
        help='restore a full epoch checkpoint while preserving the exact saved --config')
    parser.add_argument(
        'overrides', nargs='*',
        help='OmegaConf dot-list overrides such as runtime.gpus=8')
    args = parser.parse_args()
    if args.resume_state and (not args.config or args.overrides):
        parser.error('--resume-state requires --config and accepts no configuration overrides')
    return args


def validate_resume_state(config, checkpoint_path):
    """验证完整断点及原配置，恢复路径不写入科学配置。"""
    if config.runtime.resume_from is not None:
        raise ValueError('--resume-state conflicts with config.runtime.resume_from')
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    resolved = OmegaConf.to_container(config, resolve=True)
    if (checkpoint.get('labeldistill_schema_version') != config.schema_version
            or checkpoint.get('labeldistill_resolved_config') != resolved):
        raise ValueError('Resume checkpoint config does not match the exact supplied config')
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers'):
        if not checkpoint.get(key):
            raise ValueError(f'Resume checkpoint lacks complete {key}')
    if not isinstance(checkpoint.get('callbacks'), dict):
        raise ValueError('Resume checkpoint lacks callback state')
    if str(config.runtime.precision) in ('16', '16-mixed'):
        if not (checkpoint.get('MixedPrecision') or checkpoint.get('native_amp_scaling_state')):
            raise ValueError('Resume checkpoint lacks FP16 AMP scaler state')
    epoch, step = checkpoint.get('epoch'), checkpoint.get('global_step')
    if type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0:
        raise ValueError('Resume checkpoint has invalid epoch/global_step')
    try:
        fit = checkpoint['loops']['fit_loop']
        batches = fit['epoch_loop.batch_progress']
        completed = batches['total']['completed']
        current = batches['current']
        processed_epochs = fit['epoch_progress']['current']['processed']
    except (KeyError, TypeError) as exc:
        raise ValueError('Resume checkpoint lacks full epoch-loop progress') from exc
    if (batches.get('is_last_batch') is not True
            or type(completed) is not int or completed < 1
            or current.get('completed') != current.get('ready')
            or processed_epochs != epoch + 1):
        raise ValueError('Resume requires a completed epoch checkpoint')
    ema_path = None
    if config.ema.enabled:
        ema_path = path.parent / 'ema' / f'epoch_{epoch:02d}.pth'
        if not ema_path.is_file():
            raise FileNotFoundError(f'Matching EMA archive is required: {ema_path}')
    return {
        'checkpoint': str(path), 'epoch': epoch, 'global_step': step,
        'completed_microbatches': completed,
        'ema_path': str(ema_path.resolve()) if ema_path is not None else None,
    }


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    with ExitStack() as stack:
        source_path = args.config
        overrides = list(args.overrides)
        resume = None
        if args.resume_state:
            # 配置已由源断点固定，不能重生run_id或覆盖原运行的审计文件。
            bundle = load_and_resolve_config(
                source_path, [], project_root=str(project_root))
            resume = validate_resume_state(bundle.config, args.resume_state)
        else:
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
            if resume is None:
                save_config_artifacts(bundle)
            else:
                print('resume_state=' + json.dumps(resume, ensure_ascii=False), flush=True)
            print(f'run_id={config.runtime.run_id}', flush=True)
            print(f'output_dir={Path(config.runtime.output_dir).resolve()}',
                  flush=True)
            print(f'checkpoint_dir={checkpoint_directory(config).resolve()}',
                  flush=True)
        experiment = build_experiment(bundle)
        trainer = build_trainer(config, experiment)
        if resume is not None and resume['ema_path'] is not None:
            # 原EMACallback先构建同结构EMA，本回调再恢复其权重及microbatch计数。
            trainer.callbacks.append(ResumeEMA(
                resume['ema_path'], epoch=resume['epoch'],
                global_step=resume['global_step'],
                completed_microbatches=resume['completed_microbatches'],
            ))
        checkpoint = resume['checkpoint'] if resume is not None else config.runtime.resume_from
        trainer.fit(experiment, ckpt_path=checkpoint)


if __name__ == '__main__':
    main()
