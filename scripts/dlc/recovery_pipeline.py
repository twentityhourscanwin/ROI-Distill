"""从原实验身份恢复训练或续评，将恢复证据与原失败记录分开保存。"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import sys

import yaml

from distill_pipeline import DistillPipeline
import student_pipeline as base

STORAGE_ROOT = Path('/mnt/nas_data/guqiupeng/ROI_LABEL_DISTILL/outputs')
RECOVERY_ONLY_FILES = {
    '.gitignore', 'tools/train.py', 'labeldistill/callbacks/resume_ema.py',
    'scripts/dlc/recovery_pipeline.py', 'scripts/dlc/resume_distill.sh',
    'configs/experiments/convnext_distill.yaml',
    *{f'configs/experiments/convnext_distill_{group}.yaml' for group in ('a1','a2','a3','a4','c1','d1')},
}


def checkpoint_metadata(path):
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    return {
        'epoch': value.get('epoch'), 'global_step': value.get('global_step'),
        'labeldistill_resolved_config': value.get('labeldistill_resolved_config'),
        'has_training_state': bool(value.get('optimizer_states') and value.get('lr_schedulers')
                                   and value.get('loops')),
    }


class RecoveryPipeline(DistillPipeline):
    """原pipeline.json和train.exitcode保持原样；新的阶段写recovery.json。"""

    def __init__(self, root, *, storage_root=STORAGE_ROOT, **kwargs):
        super().__init__(root, **kwargs)
        self.storage_root = Path(storage_root).resolve()

    def _save(self):
        base.atomic_json(self.output / 'recovery.json', self.state)

    def _stage(self, name, argv):
        return super()._stage('recovery_' + name, argv)

    def _check_code(self, original_commit):
        current = self._git('rev-parse', 'HEAD')
        changed = set(filter(None, self._git('diff', '--name-only', original_commit, current).splitlines()))
        unexpected = changed - RECOVERY_ONLY_FILES
        if unexpected:
            raise ValueError(f'Recovery changes scientific sources: {sorted(unexpected)}')
        return current

    def _select_checkpoint(self, directory, config):
        candidates = []
        rejected = []
        for path in directory.glob('epoch_*.ckpt'):
            match = re.fullmatch(r'epoch_(\d+)\.ckpt', path.name)
            if match:
                candidates.append((int(match[1]), path))
        for epoch, path in sorted(candidates, reverse=True):
            try:
                meta = checkpoint_metadata(path)
                if (meta['epoch'] != epoch or not 0 <= epoch < config['runtime']['max_epochs']
                        or meta['labeldistill_resolved_config'] != config
                        or not meta.get('has_training_state')):
                    raise ValueError('Checkpoint training identity/state mismatch')
                return dict(path=str(path.resolve()), sha256=base.file_sha(path),
                            epoch=epoch, global_step=meta.get('global_step'),
                            rejected_newer_checkpoints=rejected)
            except Exception as exc:
                rejected.append({'path': str(path), 'error': str(exc)})
        raise ValueError(f'No complete matching training checkpoint: {rejected}')

    def _training_succeeded(self, original):
        original_exit = self.output / 'train.exitcode'
        original_attempts = original.get('stages', {}).get('train', [])
        if (original_exit.is_file() and original_exit.read_text().strip() == '0'
                and original_attempts and original_attempts[-1].get('exitcode') == 0):
            return True
        attempts = self.state.get('stages', {}).get('recovery_train_resume', [])
        exit_file = self.output / 'recovery_train_resume.exitcode'
        return bool(attempts and attempts[-1].get('exitcode') == 0
                    and exit_file.is_file() and exit_file.read_text().strip() == '0')

    def recover(self, run_dir, *, eval_only=False):
        self.output = Path(run_dir).resolve()
        if not self.output.is_relative_to(self.storage_root):
            raise ValueError(f'All recovery artifacts must be on NAS under {self.storage_root}')
        original_path = self.output / 'pipeline.json'
        original = json.loads(original_path.read_text())
        config = original['config']
        self._validate_config(config)
        if Path(config['runtime']['output_dir']).resolve() != self.output:
            raise ValueError('Original run path must resolve to the migrated NAS directory')
        if self._git('diff', '--name-only') or self._git('diff', '--cached', '--name-only'):
            raise ValueError('Recovery requires a clean tracked worktree')
        with (self.output / '.pipeline.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with base.interruption_signals():
                return self._recover_locked(original, config, original_path, eval_only)

    def _recover_locked(self, original, config, original_path, eval_only):
        recovery_path = self.output / 'recovery.json'
        saved = self.output / 'resolved_config.yaml'
        current = self._git('rev-parse', 'HEAD')
        if recovery_path.exists():
            self.state = json.loads(recovery_path.read_text())
        else:
            self.state = {
                'origin_commit': original['commit'], 'config': config,
                'origin_pipeline_sha256': base.file_sha(original_path),
                'config_sha256': original.get('config_sha256', base.file_sha(saved)),
                'original_exitcodes': {p.name: p.read_text().strip()
                                       for p in self.output.glob('*.exitcode')},
                'results': deepcopy(original.get('results', {})),
                'inference_complete': deepcopy(original.get('inference_complete', {})),
                'checkpoints': deepcopy(original.get('checkpoints', {})),
                'stages': {}, 'execution_history': [],
            }
        try:
            self._check_code(original['commit'])
            if (self.state['origin_commit'] != original['commit']
                    or self.state['origin_pipeline_sha256'] != base.file_sha(original_path)
                    or self.state['config'] != config
                    or yaml.safe_load(saved.read_text()) != config
                    or self.state['config_sha256'] != base.file_sha(saved)):
                raise ValueError('Original pipeline/config identity changed')
            self.state.update(status='running', execution_commit=current, eval_gpus=self.eval_gpus)
            self.state['execution_history'].append({
                'commit': current, 'started_utc': datetime.now(timezone.utc).isoformat(),
                'eval_only': eval_only,
            })
            self._save()
            (self.output / 'val_summary.json').unlink(missing_ok=True)
            checkpoint_dir = Path(config['checkpoint']['root_dir']) / (
                config['experiment']['name'] + '_' + config['runtime']['run_id'])
            if not self._training_succeeded(original):
                if eval_only:
                    raise ValueError('Neither original nor recovery training exited successfully')
                identity = self._select_checkpoint(checkpoint_dir, config)
                if config.get('ema', {}).get('enabled'):
                    ema = checkpoint_dir / 'ema' / f"epoch_{identity['epoch']:02d}.pth"
                    identity['ema_path'] = str(ema.resolve())
                    identity['ema_sha256'] = base.file_sha(ema)
                self.state['resume_checkpoint'] = identity
                self.state.setdefault('resume_history', []).append(deepcopy(identity))
                self._save()
                self._stage('train_resume', [
                    self.python, '-u', 'tools/train.py', '--config', str(saved),
                    '--resume-state', identity['path'],
                ])
            if not self._training_succeeded(original):
                raise ValueError('Recovery training has no successful stage record')
            self.state['training_completed'] = True
            info = Path(config['data']['root']) / config['data']['val_info']
            if not info.is_absolute():
                info = self.root / info
            info_sha = base.file_sha(info)
            expected_info = self.state.get('val_info_sha256', original.get('val_info_sha256', info_sha))
            if expected_info != info_sha:
                raise ValueError('Validation sample info changed')
            self.state['val_info_sha256'] = info_sha
            identities = {}
            for epoch in self.epochs:
                path = checkpoint_dir / f'epoch_{epoch}.ckpt'
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError(f'Missing checkpoint: {path}')
                digest = base.file_sha(path)
                if self.state['checkpoints'].get(str(epoch), digest) != digest:
                    raise ValueError(f'Checkpoint changed: {path}')
                meta = checkpoint_metadata(path)
                if meta['epoch'] != epoch or meta['labeldistill_resolved_config'] != config:
                    raise ValueError(f'Checkpoint epoch/config mismatch: {path}')
                identities[str(epoch)] = digest
            self.state['checkpoints'] = identities
            self._save()
            for epoch in self.epochs:
                self._evaluate(epoch, checkpoint_dir / f'epoch_{epoch}.ckpt', saved)
            results = [self.state['results'][str(epoch)] for epoch in self.epochs]
            base.atomic_json(self.output / 'val_summary.json', {
                'status': 'complete', 'commit': current,
                'origin_commit': original['commit'],
                'config_sha256': self.state['config_sha256'], 'results': results,
                'best_mAP': max(results, key=lambda row: row['mAP']),
                'best_NDS': max(results, key=lambda row: row['NDS']), 'last': results[-1],
                'metric_scale': 'fraction_0_to_1',
            })
            self.state.update(status='complete')
            self.state.pop('error', None)
            self._save()
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.state.update(status='failed', error=str(exc))
            self._save()
            print(str(exc), file=sys.stderr, flush=True)
            return exc.code if isinstance(exc, (base.StageFailed, base.PipelineInterrupted)) else (130 if isinstance(exc, KeyboardInterrupt) else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-gpus', type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        return RecoveryPipeline(root, eval_gpus=args.eval_gpus).recover(args.run_dir, eval_only=args.eval_only)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
