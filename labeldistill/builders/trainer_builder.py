"""Build PyTorch Lightning runtime objects from resolved config."""

from datetime import timedelta
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from labeldistill.callbacks.ema import EMACallback


def build_trainer(config, experiment):
    precision_map = {
        '16': '16-mixed',
        '16-mixed': '16-mixed',
        'bf16': 'bf16-mixed',
        'bf16-mixed': 'bf16-mixed',
        '32': 32,
        '64': 64,
    }
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(config.runtime.output_dir) / 'checkpoints'),
        filename='epoch_{epoch:02d}',
        save_top_k=config.checkpoint.save_top_k,
        save_last=config.checkpoint.save_last,
        monitor='epoch',
        mode='max',
        every_n_epochs=config.checkpoint.every_n_epochs,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]
    if config.ema.enabled:
        # Preserve the legacy callback's update-count convention until EMA is
        # independently migrated and numerically frozen.
        total_samples = len(experiment.train_dataloader().dataset)
        callbacks.insert(0, EMACallback(total_samples * config.runtime.max_epochs))

    world_size = config.runtime.gpus * config.runtime.num_nodes
    strategy = (
        DDPStrategy(
            find_unused_parameters=True,
            process_group_backend=config.runtime.distributed_backend,
            timeout=timedelta(hours=2),
        )
        if world_size > 1 else 'auto'
    )
    trainer_kwargs = dict(
        max_epochs=config.runtime.max_epochs,
        devices=config.runtime.gpus,
        num_nodes=config.runtime.num_nodes,
        accelerator='gpu',
        strategy=strategy,
        num_sanity_val_steps=0,
        gradient_clip_val=config.runtime.gradient_clip_val,
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=config.runtime.accumulate_grad_batches,
        deterministic=config.runtime.deterministic,
        limit_val_batches=0,
        enable_checkpointing=True,
        precision=precision_map.get(
            str(config.runtime.precision), config.runtime.precision),
        reload_dataloaders_every_n_epochs=1,
        default_root_dir=config.runtime.output_dir,
        callbacks=callbacks,
    )
    if config.runtime.limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = config.runtime.limit_train_batches
    return pl.Trainer(**trainer_kwargs)
