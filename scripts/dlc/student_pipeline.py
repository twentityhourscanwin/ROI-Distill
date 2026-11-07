"""Train one pure student, then audit every checkpoint from epochs 25--36."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys

import yaml


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_checkpoint(path):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return {'epoch': checkpoint.get('epoch'),
            'labeldistill_resolved_config': checkpoint.get('labeldistill_resolved_config')}


class PipelineInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__(f'Pipeline interrupted by signal {signum}')
        self.code = 128 + signum


@contextmanager
def interruption_signals():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def interrupt(signum, frame):
        raise PipelineInterrupted(signum)

    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class StageFailed(RuntimeError):
    def __init__(self, stage, code):
        super().__init__(f'{stage} exited {code}')
        self.code = code if code > 0 else 128 - code


class StudentPipeline:
    epochs = tuple(range(24, 36))
    stop_timeout = 5.0

    def __init__(self, root, *, python=sys.executable, eval_gpus=16):
        self.root = Path(root).resolve()
        self.python = python
        self.eval_gpus = eval_gpus
        self.state = {}

    def _git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def _save(self):
        atomic_json(self.output / 'pipeline.json', self.state)

    def _stop_group(self, process):
        # Ignore repeated interrupts only while releasing the owned process group.
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        errors = []
        try:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(str(exc))
            try:
                process.wait(timeout=self.stop_timeout)
            except subprocess.TimeoutExpired:
                pass
            # A leader can exit before its DDP workers. Kill the group even if
            # wait() already reaped that leader; the workers share its pgid.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(str(exc))
            try:
                process.wait(timeout=self.stop_timeout)
            except subprocess.TimeoutExpired as exc:
                errors.append(str(exc))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return errors

    def _finish_stage(self, name, item, code):
        item['exitcode'] = code
        item['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (self.output / f'{name}.exitcode').write_text(str(code) + '\n')
        self._save()

    def _stage(self, name, argv):
        record = self.state.setdefault('stages', {}).setdefault(name, [])
        attempt = len(record) + 1
        log = self.output / 'logs' / f'{name}.{attempt:03d}.log'
        log.parent.mkdir(exist_ok=True)
        item = {'command': argv, 'cwd': str(self.root), 'log': str(log),
                'started_utc': datetime.now(timezone.utc).isoformat(), 'exitcode': None}
        record.append(item)
        self.state['current_stage'] = name
        self._save()
        env = dict(os.environ)
        env['PYTHONPATH'] = str(self.root) + os.pathsep + env.get('PYTHONPATH', '')
        process = None
        try:
            with log.open('w', encoding='utf-8') as stream:
                process = subprocess.Popen(argv, cwd=self.root, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, start_new_session=True)
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                code = process.wait()
        except BaseException as exc:
            if process is not None:
                try:
                    errors = self._stop_group(process)
                except BaseException as cleanup_error:
                    errors = [str(cleanup_error)]
                if errors:
                    item['cleanup_errors'] = errors
                item['child_returncode'] = process.returncode
            code = exc.code if isinstance(exc, PipelineInterrupted) else (130 if isinstance(exc, KeyboardInterrupt) else 1)
            try:
                self._finish_stage(name, item, code)
            except Exception as record_error:
                print(f'Cannot persist interrupted stage: {record_error}', file=sys.stderr)
            raise
        self._finish_stage(name, item, code)
        if code:
            raise StageFailed(name, code)

    def _validate_config(self, config):
        runtime = config['runtime']
        if (runtime['max_epochs'], runtime['gpus'], runtime['num_nodes'],
                runtime['batch_size_per_device'], runtime['accumulate_grad_batches']) != (36, 16, 1, 4, 2):
            raise ValueError('Expected 36 epochs, one node, 16 GPUs x batch 4 x accumulation 2')
        if runtime.get('limit_train_batches') is not None or runtime.get('limit_test_batches') is not None:
            raise ValueError('Formal pipeline does not accept partial train/val batches')
        if config['data'].get('split', 'train') != 'train':
            raise ValueError('Formal validation baseline must train on the train split only')
        if runtime.get('resume_from') is not None:
            raise ValueError('This launcher creates new training runs; use --eval-only to resume val')
        if config['student']['training_mode'] != 'pure_student':
            raise ValueError('Expected a pure-student experiment')
        if config['loss']['feature_weight'] != 0 or config['loss']['response_weight'] != 0:
            raise ValueError('Both distillation weights must be zero')
        if self.eval_gpus < 1:
            raise ValueError('eval GPU count must be positive')

    def run(self, *, config, config_path, eval_only=False):
        self._validate_config(config)
        if self._git('diff', '--name-only') or self._git('diff', '--cached', '--name-only'):
            raise ValueError('Formal pipeline requires a clean tracked worktree')
        self.output = Path(config['runtime']['output_dir']).resolve()
        checkpoint_root = Path(config['checkpoint']['root_dir']).resolve()
        checkpoint_dir = checkpoint_root / (config['experiment']['name'] + '_' + config['runtime']['run_id'])
        if not eval_only:
            if self.output.exists() or checkpoint_dir.exists():
                raise FileExistsError('Run output/checkpoint directory already exists')
            self.output.mkdir(parents=True, exist_ok=False)
        elif not (self.output / 'pipeline.json').is_file():
            raise ValueError('No pipeline record; refuse to infer training success')
        with (self.output / '.pipeline.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with interruption_signals():
                return self._run_locked(config, Path(config_path).resolve(), checkpoint_dir, eval_only)

    def _run_locked(self, config, config_path, checkpoint_dir, eval_only):
        current_commit = self._git('rev-parse', 'HEAD')
        if eval_only:
            self.state = json.loads((self.output / 'pipeline.json').read_text())
        else:
            self.state = {'status': 'running', 'commit': current_commit,
                          'branch': self._git('branch', '--show-current'),
                          'config_source': str(config_path), 'config': config,
                          'epochs': list(self.epochs), 'results': {}, 'stages': {},
                          'eval_gpus': self.eval_gpus, 'history_fallback': 'previous_valid'}
        self.state['status'] = 'running'
        (self.output / 'val_summary.json').unlink(missing_ok=True)
        self._save()
        try:
            if current_commit != self.state['commit']:
                raise ValueError('Code commit changed; use the original training commit for val')
            if config != self.state['config']:
                raise ValueError('Training configuration differs from pipeline identity')
            if eval_only:
                if (self.output / 'train.exitcode').read_text().strip() != '0':
                    raise ValueError('Training has no successful exit record')
            else:
                argv = [self.python, '-u', 'tools/train.py', '--config', str(config_path),
                        f'runtime.run_id={config["runtime"]["run_id"]}',
                        f'runtime.output_dir={self.output}',
                        f'checkpoint.root_dir={config["checkpoint"]["root_dir"]}',
                        'runtime.gpus=16', 'runtime.num_nodes=1',
                        'runtime.batch_size_per_device=4', 'runtime.accumulate_grad_batches=2',
                        'runtime.max_epochs=36']
                self._stage('train', argv)
            saved = self.output / 'resolved_config.yaml'
            saved_config = yaml.safe_load(saved.read_text())
            if saved_config != config:
                raise ValueError('Saved training config differs from intended config')
            saved_sha = file_sha(saved)
            if 'config_sha256' in self.state and self.state['config_sha256'] != saved_sha:
                raise ValueError('Saved config bytes changed')
            self.state['config_sha256'] = saved_sha
            info = Path(config['data']['root']) / config['data']['val_info']
            if not info.is_absolute():
                info = self.root / info
            info_sha = file_sha(info)
            if self.state.get('val_info_sha256', info_sha) != info_sha:
                raise ValueError('Validation sample info changed')
            self.state['val_info_sha256'] = info_sha
            identities = {}
            for epoch in self.epochs:
                path = checkpoint_dir / f'epoch_{epoch}.ckpt'
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f'Missing checkpoint: {path}')
                digest = file_sha(path)
                if self.state.get('checkpoints', {}).get(str(epoch), digest) != digest:
                    raise ValueError(f'Checkpoint changed: {path}')
                metadata = read_checkpoint(path)
                if metadata['epoch'] != epoch or metadata['labeldistill_resolved_config'] != config:
                    raise ValueError(f'Checkpoint epoch/config mismatch: {path}')
                identities[str(epoch)] = digest
            self.state['checkpoints'] = identities
            self._save()
            for epoch in self.epochs:
                self._evaluate(epoch, checkpoint_dir / f'epoch_{epoch}.ckpt', saved)
            results = [self.state['results'][str(epoch)] for epoch in self.epochs]
            summary = {'status': 'complete', 'commit': current_commit,
                       'config_sha256': saved_sha, 'results': results,
                       'best_mAP': max(results, key=lambda row: row['mAP']),
                       'best_NDS': max(results, key=lambda row: row['NDS']),
                       'last': results[-1], 'metric_scale': 'fraction_0_to_1'}
            atomic_json(self.output / 'val_summary.json', summary)
            self.state['status'] = 'complete'
            self.state.pop('error', None)
            self._save()
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.state['status'] = 'failed'
            self.state['error'] = str(exc)
            self._save()
            print(str(exc), file=sys.stderr, flush=True)
            return exc.code if isinstance(exc, (StageFailed, PipelineInterrupted)) else (130 if isinstance(exc, KeyboardInterrupt) else 1)

    def _evaluate(self, epoch, checkpoint, saved):
        prior = (self.state['results'].get(str(epoch))
                 or self.state.get('inference_complete', {}).get(str(epoch)))
        if prior:
            target = Path(prior['output_dir'])
            for filename, digest in prior['artifact_sha256'].items():
                if file_sha(target / filename) != digest:
                    raise ValueError(f'Completed val artifact changed: {target / filename}')
        else:
            base = self.output / 'validation' / f'epoch_{epoch}'
            base.mkdir(parents=True, exist_ok=True)
            attempt = 1
            while (base / f'attempt_{attempt:03d}').exists():
                attempt += 1
            target = base / f'attempt_{attempt:03d}'
            target.mkdir()
            self._stage(f'epoch_{epoch}_full', [
                self.python, '-u', 'scripts/dlc/evaluate_history.py',
                '--history-fallback', 'previous_valid', '--config', str(saved),
                '--checkpoint', str(checkpoint), '--output-dir', str(target),
                f'runtime.gpus={self.eval_gpus}', 'runtime.num_nodes=1',
                'runtime.batch_size_per_device=4', 'runtime.limit_test_batches=null',
                'evaluation.run_metrics=true'])
            self.state.setdefault('inference_complete', {})[str(epoch)] = {
                'output_dir': str(target),
                'artifact_sha256': {name: file_sha(target / name) for name in
                                   ('results_nusc.json', 'metrics_summary.json')}}
            self._save()
        self._stage(f'epoch_{epoch}_audit', [
            self.python, '-u', 'scripts/dlc/verify_convnext_val.py',
            '--config', str(saved), '--output-dir', str(target), '--expected-count', '6019'])
        audit = json.loads((target / 'val_audit.json').read_text())
        metrics = json.loads((target / 'metrics_summary.json').read_text())
        if audit['samples'] != 6019 or audit['mAP'] != metrics['mean_ap'] or audit['NDS'] != metrics['nd_score']:
            raise ValueError('Audit disagrees with metric output')
        values = [metrics['mean_ap'], metrics['nd_score'], *metrics['tp_errors'].values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Non-finite validation metrics')
        self.state['results'][str(epoch)] = {
            'epoch': epoch, 'training_epoch': epoch + 1,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': self.state['checkpoints'][str(epoch)],
            'output_dir': str(target), 'mAP': metrics['mean_ap'], 'NDS': metrics['nd_score'],
            'tp_errors': metrics['tp_errors'], 'class_AP': metrics.get('mean_dist_aps', {}),
            'artifact_sha256': {name: file_sha(target / name) for name in
                               ('results_nusc.json', 'metrics_summary.json', 'val_audit.json')}}
        self._save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-only', type=Path, help='Resume validation of an existing pipeline run')
    parser.add_argument('--run-id', help='New run identifier YYYYMMDD_HHMMSS')
    parser.add_argument('--checkpoint-root')
    parser.add_argument('--eval-gpus', type=int, default=16)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    if args.eval_only:
        if args.run_id or args.checkpoint_root:
            parser.error('--eval-only cannot change run-id or checkpoint-root')
        output = args.eval_only.resolve()
        record = json.loads((output / 'pipeline.json').read_text())
        config = record['config']
        if Path(config['runtime']['output_dir']).resolve() != output:
            raise ValueError('Run directory does not match its identity')
        source = output / 'resolved_config.yaml'
    else:
        from labeldistill.config import load_training_bundle
        from omegaconf import OmegaConf
        source = root / 'configs/experiments/convnext_student.yaml'
        overrides = []
        if args.run_id:
            overrides.append(f'runtime.run_id={args.run_id}')
        checkpoint_root = args.checkpoint_root or os.environ.get('ROI_CHECKPOINT_ROOT')
        if checkpoint_root:
            overrides.append(f'checkpoint.root_dir={checkpoint_root}')
        bundle = load_training_bundle(source, overrides, root)
        config = OmegaConf.to_container(bundle.config, resolve=True)
        config['runtime']['output_dir'] = str(Path(config['runtime']['output_dir']).resolve())
        config['checkpoint']['root_dir'] = str(Path(config['checkpoint']['root_dir']).resolve())
    try:
        return StudentPipeline(root, eval_gpus=args.eval_gpus).run(
            config=config, config_path=source, eval_only=bool(args.eval_only))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
