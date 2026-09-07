#LabelDistill/labeldistill/exps/base_cli.py
import os
from pathlib import Path
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from labeldistill.callbacks.ema import EMACallback
from labeldistill.config.run import (
    RUN_ID_ENV, dated_directory, generate_run_id, validate_run_id,
)
from labeldistill.utils.torch_dist import all_gather_object, get_rank, synchronize
from .nuscenes.base_exp import LabelDistillModel


CHECKPOINT_ROOT = os.environ.get(
    'LABELDISTILL_CHECKPOINT_ROOT',
    '/mnt/nas_data/guqiupeng/checkpoint_nes',
)


def run_cli(model_class=LabelDistillModel,
            exp_name='base_exp',
            use_ema=False,
            extra_trainer_config_args={}):
    parent_parser = ArgumentParser(add_help=False)
    # PL 2.x 移除了 add_argparse_args，改为手动添加 Trainer 常用参数
    parent_parser.add_argument('--gpus', type=int, default=1,
                               help='number of GPUs per node')
    parent_parser.add_argument('--num_nodes', type=int, default=1,
                               help='number of nodes for multi-node training')
    parent_parser.add_argument('--max_epochs', type=int,
                               default=extra_trainer_config_args.get('epochs', 24))
    # PPU 上的 voxel pooling/DCN 自定义 kernel 不支持 BF16，默认使用
    # FP16 mixed precision；高风险 loss 和 DCN 路径在各自实现中显式转 FP32。
    # 可选值: '16'/'16-mixed'(fp16 AMP), 'bf16'/'bf16-mixed', '32'(fp32)
    parent_parser.add_argument('--precision', type=str, default='16')
    parent_parser.add_argument('--default_root_dir', type=str,
                               default=os.path.join('./outputs/', exp_name))
    parent_parser.add_argument(
        '--gradient_clip_val', type=float,
        default=extra_trainer_config_args.get('gradient_clip_val', 35))
    parent_parser.add_argument('--amp_backend', type=str, default='native')
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-p',
                               '--predict',
                               dest='predict',
                               action='store_true',
                               help='predict model on testing set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int, default=8)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parent_parser.add_argument('--run_id', type=str)
    parser = LabelDistillModel.add_model_specific_args(parent_parser)
    args = parser.parse_args()

    if args.seed is not None:
        pl.seed_everything(args.seed)

    resume_directory = (
        Path(args.ckpt_path).expanduser().resolve().parent
        if args.ckpt_path and not (args.evaluate or args.predict) else None
    )
    checkpoint_root = Path(CHECKPOINT_ROOT).expanduser().resolve()
    if resume_directory is not None and resume_directory.is_relative_to(checkpoint_root):
        checkpoint_dir = resume_directory
    else:
        run_id = args.run_id or os.environ.get(RUN_ID_ENV) or generate_run_id()
        validate_run_id(run_id)
        os.environ[RUN_ID_ENV] = run_id
        checkpoint_dir = checkpoint_root / dated_directory(exp_name, run_id)
        if not (args.evaluate or args.predict):
            args.default_root_dir = dated_directory(args.default_root_dir, run_id)

    model = model_class(**vars(args))

    # 配置checkpoint回调，保留最后3个epoch的checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='epoch_{epoch:02d}',
        save_top_k=3,
        save_last='link',
        enable_version_counter=False,
        monitor='epoch',
        mode='max',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )

    # PL 2.x precision mapping. BF16 remains available for models whose full
    # operator chain supports it, but it is not the PPU default.
    _precision_map = {
        '16': '16-mixed', '16-mixed': '16-mixed',
        'bf16': 'bf16-mixed', 'bf16-mixed': 'bf16-mixed',
        '32': 32, '64': 64,
    }
    precision = _precision_map.get(str(args.precision), args.precision)

    # 多机多卡时使用 NCCL 后端（默认），单机时自动退化
    # init_process_group_kwargs 设置超时时间，防止多机通信超时（默认30min，改为2h）
    ddp_strategy = DDPStrategy(
        find_unused_parameters=True,
        process_group_backend='nccl',
        timeout=__import__('datetime').timedelta(hours=2),
    )

    # 构建 Trainer（PL 2.x 直接传参，不再用 from_argparse_args）
    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        num_nodes=args.num_nodes,       # 多机多卡：节点数，单机默认为1
        accelerator='gpu',
        strategy=ddp_strategy,
        num_sanity_val_steps=0,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm='norm',
        limit_val_batches=0,
        enable_checkpointing=True,
        precision=precision,
        reload_dataloaders_every_n_epochs=1,
        default_root_dir=args.default_root_dir,
    )

    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs,
            dirpath=checkpoint_dir / 'ema', keep_last=3,
        )
        trainer = pl.Trainer(
            **trainer_kwargs,
            callbacks=[ema_callback, checkpoint_callback]
        )
    else:
        trainer = pl.Trainer(
            **trainer_kwargs,
            callbacks=[checkpoint_callback]
        )

    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)
    elif args.predict:
        predict_step_outputs = trainer.predict(model, ckpt_path=args.ckpt_path)
        all_pred_results = list()
        all_img_metas = list()
        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])
        synchronize()
        len_dataset = len(model.predict_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]
        all_img_metas = sum(map(list, zip(*all_gather_object(all_img_metas))),
                            [])[:len_dataset]
        model.evaluator._format_bbox(all_pred_results, all_img_metas,
                                     os.path.dirname(args.ckpt_path))
    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)
