"""Read-only training measurements; loss means weight batch scalars by sample count."""
from __future__ import annotations

import json
import math
import re
import time
import uuid
from pathlib import Path

import torch
from pytorch_lightning import Callback


class TrainingDiagnostics(Callback):
    """Observe one automatic optimizer without changing its updates or RNG state.

    Gradient snapshots are rank-zero, unscaled and before clipping. Relative
    updates describe a fixed prefix (at most 4096 elements) of one tensor per
    stage, not a full-stage update norm. No additional backward is performed.
    """

    def __init__(self, config, *, output_dir):
        super().__init__()
        self.config = config
        self.output_dir = Path(output_dir)
        self.enabled = config.enabled
        self.attempts = 0
        self.optimizer_calls = 0
        self.amp_skips = 0
        self.unexpected_missing_calls = 0
        self.resumed = False
        self.session_id = uuid.uuid4().hex
        self._handle = None
        self._pending = None
        self._before = {}
        self._representatives = {}
        self._parameters = []
        self._epoch_sums = None
        self._loss_names = None
        self._epoch_samples = 0
        self._epoch_batches = 0
        self._sampled_gradients = 0
        self._sampled_clips = 0
        self._observer_seconds = 0.0

    @staticmethod
    def stage_name(name):
        image = 'img_backbone.'
        if image in name:
            tail = name.split(image, 1)[1]
            match = re.match(r'(stages|downsample_layers)\.(\d+)', tail)
            if match:
                kind = 'stage' if match[1] == 'stages' else 'downsample'
                return f'image_{kind}_{match[2]}'
            return 'image_other'
        return '.'.join(name.split('.')[:2])

    def state_dict(self):
        return {key: getattr(self, key) for key in (
            'attempts', 'optimizer_calls', 'amp_skips',
            'unexpected_missing_calls')}

    def load_state_dict(self, state_dict):
        for key in self.state_dict():
            setattr(self, key, int(state_dict.get(key, 0)))
        self.resumed = True

    @staticmethod
    def _clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: TrainingDiagnostics._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrainingDiagnostics._clean(v) for v in value]
        return value

    def _write(self, trainer, event, **values):
        if not trainer.is_global_zero:
            return
        record = self._clean(dict(event=event, session_id=self.session_id,
                                  resumed=self.resumed, epoch=int(trainer.current_epoch),
                                  global_step=int(trainer.global_step),
                                  attempts=self.attempts, **values))
        with (self.output_dir / 'training_diagnostics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    @staticmethod
    def _scale(trainer):
        scaler = trainer.precision_plugin.scaler if hasattr(trainer.precision_plugin, 'scaler') else None
        return float(scaler.get_scale()) if scaler is not None else None

    def _tb(self, trainer, values):
        if trainer.is_global_zero:
            finite = {f'diagnostics/{key}': value for key, value in values.items()
                      if isinstance(value, (float, int)) and math.isfinite(value)}
            for logger in trainer.loggers:
                logger.log_metrics(finite, step=int(trainer.global_step))

    def on_fit_start(self, trainer, pl_module):
        if not self.enabled:
            return
        if not pl_module.automatic_optimization or len(trainer.optimizers) != 1:
            raise ValueError('TrainingDiagnostics requires one automatic optimizer')
        for key in ('log_every_n_steps', 'gradient_every_n_steps', 'update_every_n_steps'):
            if getattr(self.config, key) < 1:
                raise ValueError(f'{key} must be positive')
        self._trainer = trainer
        optimizer = trainer.optimizers[0]
        if getattr(optimizer, '_step_supports_amp_scaling', False):
            raise ValueError('Diagnostics requires scaler-managed optimizer steps, not fused AMP steps')
        names = {id(param): name for name, param in pl_module.named_parameters()}
        self._parameters = [(names[id(param)], param) for group in optimizer.param_groups
                            for param in group['params'] if param.requires_grad]
        for name, param in self._parameters:
            stage = self.stage_name(name)
            if stage not in self._representatives and param.ndim >= 2:
                self._representatives[stage] = (name, param)
        self._handle = optimizer.register_step_post_hook(self._after_step)
        if trainer.is_global_zero:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        groups = [dict(index=i, name=group.get('group_name', str(i)),
                       lr=float(group['lr']), weight_decay=float(group.get('weight_decay', 0)),
                       parameter_tensors=len(group['params']),
                       parameter_count=sum(p.numel() for p in group['params']),
                       parameter_names=[names[id(p)] for p in group['params']])
                  for i, group in enumerate(optimizer.param_groups)]
        self._write(trainer, 'session', optimizer_groups=groups,
                    world_size=int(trainer.world_size), precision=str(trainer.precision),
                    gradient_interval=self.config.gradient_every_n_steps,
                    update_interval=self.config.update_every_n_steps,
                    gradient_scope='rank_zero_unscaled_before_clip',
                    update_scope='first_4096_elements_of_first_matrix_per_stage',
                    epoch_loss_semantics='sample_weighted_batch_scalar_not_object_mean',
                    environment_report=str(self.output_dir / 'environment.txt'),
                    resumed_counters=self.state_dict(),
                    initialization_reports={name: module.initialization_report
                        for name, module in pl_module.named_modules()
                        if hasattr(module, 'initialization_report')})

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.enabled:
            return
        self._epoch_start = time.perf_counter()
        self._epoch_sums = None
        self._loss_names = None
        self._epoch_samples = 0
        self._epoch_batches = 0
        self._sampled_gradients = self._sampled_clips = 0
        self._observer_seconds = 0.0
        self._epoch_attempt_start = self.attempts
        self._epoch_call_start = self.optimizer_calls
        self._epoch_skip_start = self.amp_skips
        # Do not reset allocator peaks owned by other callbacks/profilers.

    @torch.no_grad()
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.enabled:
            return
        started = time.perf_counter()
        self.attempts += 1
        self._pending = (self.optimizer_calls, self._scale(trainer),
                         [float(group['lr']) for group in optimizer.param_groups])
        if trainer.is_global_zero:
            if self.attempts % self.config.gradient_every_n_steps == 0:
                by_stage = {}
                for name, param in self._parameters:
                    if param.grad is not None:
                        grad = param.grad.detach()
                        norm = torch.linalg.vector_norm(grad.float())
                        by_stage.setdefault(self.stage_name(name), []).append(norm)
                stage_norms = {stage: torch.linalg.vector_norm(torch.stack(norms))
                               for stage, norms in by_stage.items()}
                if stage_norms:
                    global_norm = float(torch.linalg.vector_norm(torch.stack(list(stage_norms.values()))))
                    threshold = float(trainer.gradient_clip_val or 0)
                    algorithm = str(trainer.gradient_clip_algorithm).lower()
                    coefficient = min(1.0, threshold / (global_norm + 1e-6)) if threshold > 0 and 'norm' in algorithm and math.isfinite(global_norm) else None
                    self._sampled_gradients += 1
                    self._sampled_clips += int(coefficient is not None and coefficient < 1)
                    stages = {stage: float(norm) for stage, norm in stage_norms.items()}
                    self._write(trainer, 'gradient', global_norm=global_norm,
                                stage_norms=stages, finite=math.isfinite(global_norm),
                                clip_coefficient=coefficient,
                                clip_algorithm=algorithm, clip_threshold=threshold,
                                sampled_clip_fraction=self._sampled_clips / self._sampled_gradients,
                                sampled_windows=self._sampled_gradients)
                    self._tb(trainer, {'gradient/global_norm': global_norm,
                                      **{f'gradient/{k}': v for k,v in stages.items()}})
            self._before = {}
            if (self.optimizer_calls + 1) % self.config.update_every_n_steps == 0:
                self._before = {stage: param.detach().reshape(-1)[:4096].clone()
                                for stage, (_, param) in self._representatives.items()}
        self._observer_seconds += time.perf_counter() - started

    @torch.no_grad()
    def _after_step(self, optimizer, args, kwargs):
        self.optimizer_calls += 1
        trainer = self._trainer
        started = time.perf_counter()
        if trainer.is_global_zero:
            for stage, before in self._before.items():
                name, param = self._representatives[stage]
                after = param.detach().reshape(-1)[:4096]
                norm = float(torch.linalg.vector_norm(before.float()))
                delta = float(torch.linalg.vector_norm(after.float() - before.float()))
                relative = delta / max(norm, 1e-12)
                self._write(trainer, 'update', stage=stage, parameter=name,
                            sampled_elements=before.numel(), parameter_elements=param.numel(),
                            before_norm=norm, update_norm=delta, relative_update=relative)
                self._tb(trainer, {f'update/{stage}': relative})
        self._before = {}
        self._observer_seconds += time.perf_counter() - started

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.enabled:
            return
        started = time.perf_counter()
        losses, samples = pl_module._diagnostics_batch
        names = sorted(losses)
        if self._loss_names is None:
            self._loss_names = names
        elif self._loss_names != names:
            raise ValueError('Diagnostic loss keys changed within an epoch')
        values = torch.stack([losses[key].detach().double() for key in names]) * samples
        if self._epoch_sums is None:
            self._epoch_sums = values
        else:
            self._epoch_sums.add_(values)
        self._epoch_samples += samples
        self._epoch_batches += 1
        del pl_module._diagnostics_batch
        if self._pending is not None:
            calls_before, scale_before, lr_used = self._pending
            skipped = self.optimizer_calls == calls_before
            if skipped:
                if scale_before is not None:
                    self.amp_skips += 1
                else:
                    self.unexpected_missing_calls += 1
            if skipped or self.attempts % self.config.log_every_n_steps == 0:
                lr = [float(group['lr']) for group in trainer.optimizers[0].param_groups]
                schedulers = [dict(last_epoch=int(conf.scheduler.last_epoch),
                                   interval=conf.interval, frequency=conf.frequency)
                              for conf in trainer.lr_scheduler_configs]
                self._write(trainer, 'optimizer', optimizer_calls=self.optimizer_calls,
                            amp_skips=self.amp_skips,
                            unexpected_missing_calls=self.unexpected_missing_calls,
                            skipped_this_window=skipped, scale_before=scale_before,
                            scale_after=self._scale(trainer), lr_used=lr_used,
                            lr_after_scheduler=lr,
                            schedulers=schedulers)
                self._tb(trainer, {'optimizer_calls': self.optimizer_calls,
                                  'amp_skips': self.amp_skips,
                                  **{f'lr/group_{i}': v for i,v in enumerate(lr)}})
            self._pending = None
            self._before = {}
        self._observer_seconds += time.perf_counter() - started

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        if not self.enabled or self._epoch_sums is None:
            return
        # All ranks execute identical collectives; rank zero alone writes files.
        count = self._epoch_sums.new_tensor([self._epoch_samples, self._epoch_batches])
        reduced = trainer.strategy.reduce(torch.cat([self._epoch_sums, count]), reduce_op='sum')
        duration = time.perf_counter() - self._epoch_start
        timing = self._epoch_sums.new_tensor([duration, self._observer_seconds])
        peak = torch.cuda.max_memory_allocated(pl_module.device) if pl_module.device.type == 'cuda' else 0
        reserved = torch.cuda.max_memory_reserved(pl_module.device) if pl_module.device.type == 'cuda' else 0
        resources = self._epoch_sums.new_tensor([peak, reserved])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(resources, op=torch.distributed.ReduceOp.MAX)
        samples = int(reduced[-2])
        batches = int(reduced[-1])
        sums = {key: float(value) for key,value in zip(self._loss_names, reduced[:-2])}
        means = {key: value / samples for key,value in sums.items()}
        self._write(trainer, 'epoch', samples=samples,
                    observed_microbatches_global=batches, loss_sum=sums, loss_mean=means,
                    duration_seconds=float(timing[0]), samples_per_second=samples / max(float(timing[0]), 1e-12),
                    observer_host_seconds=float(timing[1]),
                    cuda_peak_allocated_bytes=int(resources[0]),
                    cuda_peak_reserved_bytes=int(resources[1]),
                    cuda_peak_scope='process_lifetime_without_reset',
                    loss_scope='batches_observed_in_this_fit_epoch',
                    epoch_attempts=self.attempts-self._epoch_attempt_start,
                    epoch_optimizer_calls=self.optimizer_calls-self._epoch_call_start,
                    epoch_amp_skips=self.amp_skips-self._epoch_skip_start,
                    gradient_sampled_windows=self._sampled_gradients,
                    gradient_sampled_clipped_windows=self._sampled_clips)
        self._tb(trainer, {f'epoch_loss/{key}':value for key,value in means.items()})

    def teardown(self, trainer, pl_module, stage):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
