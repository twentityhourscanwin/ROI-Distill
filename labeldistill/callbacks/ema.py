#!/usr/bin/env python3
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
import os
from pathlib import Path
import re
import tempfile
from copy import deepcopy

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback

__all__ = ['ModelEMA', 'is_parallel']


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


class ModelEMA:
    """
    Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in
    the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema = deepcopy(
            model.module if is_parallel(model) else model).eval()
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            # model here is the inner model (pl_module.model), not wrapped by DDP
            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


class EMACallback(Callback):

    def __init__(self, len_updates, *, dirpath=None, keep_last=3,
                 every_n_epochs=1) -> None:
        super().__init__()
        self.len_updates = len_updates
        if keep_last < 1 or every_n_epochs < 1:
            raise ValueError('EMA retention and save interval must be positive')
        self.dirpath = Path(dirpath) if dirpath is not None else None
        self.keep_last = keep_last
        self.every_n_epochs = every_n_epochs

    def on_fit_start(self, trainer, pl_module):
        # Todo (@lizeming@megvii.com): delete manually specified device
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in trainer.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        # PL 2.x: pl_module is directly the LightningModule, pl_module.model is the inner model
        inner_model = pl_module.model
        trainer.ema_model = ModelEMA(inner_model.cuda(), 0.9990)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        trainer.ema_model.updates = self.len_updates

    def on_train_batch_end(self,
                           trainer,
                           pl_module,
                           outputs,
                           batch,
                           batch_idx,
                           unused=0):
        trainer.ema_model.update(trainer, pl_module.model)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # DDP ranks must never write or prune the same archive concurrently.
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        directory = self.dirpath
        if directory is None:
            checkpoint_dir = getattr(trainer.checkpoint_callback, 'dirpath', None)
            if checkpoint_dir is None:
                raise RuntimeError('EMA requires an explicit checkpoint directory')
            directory = Path(checkpoint_dir) / 'ema'
        directory.mkdir(parents=True, exist_ok=True)
        state_dict = trainer.ema_model.ema.state_dict()
        state_dict_keys = list(state_dict.keys())
        # TODO: Change to more elegant way.
        for state_dict_key in state_dict_keys:
            new_key = 'model.' + state_dict_key
            state_dict[new_key] = state_dict.pop(state_dict_key)
        checkpoint = {
            # the epoch and global step are saved for
            # compatibility but they are not relevant for restoration
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
            'state_dict': state_dict
        }
        destination = directory / f'epoch_{trainer.current_epoch:02d}.pth'
        # Publish only complete archives. A failed write leaves the previous
        # checkpoint and retention set intact; the temporary file is removed.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                    dir=directory, prefix='.ema-', suffix='.tmp',
                    delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

        # Inspect disk on each successful save, so retention also works after
        # restarting training. Never prune unrelated files or symlinks.
        archives = []
        for path in directory.iterdir():
            match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
            if match and path.is_file() and not path.is_symlink():
                archives.append((int(match.group(1)), path))
        for _, path in sorted(archives)[:-self.keep_last]:
            path.unlink()
