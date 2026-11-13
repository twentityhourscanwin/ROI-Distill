"""恢复与普通训练断点对应的EMA状态及更新计数。"""

from pathlib import Path

from pytorch_lightning import Callback
import torch


class ResumeEMA(Callback):
    """在既有EMA初始化后恢复，避免把optimizer步数误当microbatch计数。"""

    def __init__(self, ema_path, *, epoch, global_step, completed_microbatches):
        super().__init__()
        if (type(epoch) is not int or epoch < 0
                or type(global_step) is not int or global_step < 0
                or type(completed_microbatches) is not int or completed_microbatches < 1):
            raise ValueError('Invalid EMA resume counters')
        self.ema_path = Path(ema_path).expanduser().resolve()
        self.epoch = epoch
        self.global_step = global_step
        self.completed_microbatches = completed_microbatches
        self.applied = False

    def on_fit_start(self, trainer, pl_module):
        if self.applied:
            raise RuntimeError('EMA resume has already been applied')
        if not hasattr(trainer, 'ema_model'):
            raise RuntimeError('ResumeEMA must run after the existing EMACallback')
        archive = torch.load(self.ema_path, map_location='cpu', weights_only=False)
        if (type(archive.get('epoch')) is not int
                or type(archive.get('global_step')) is not int
                or archive['epoch'] != self.epoch
                or archive['global_step'] != self.global_step):
            raise ValueError('EMA epoch/global_step does not match the ordinary checkpoint')
        saved = archive.get('state_dict')
        if not isinstance(saved, dict) or not all(key.startswith('model.') for key in saved):
            raise ValueError('EMA archive has invalid state_dict keys')
        state = {key[len('model.'):]: value for key, value in saved.items()}
        expected = trainer.ema_model.ema.state_dict()
        # 先核对全体key/shape/dtype，避免strict加载失败前已复制部分参数。
        if state.keys() != expected.keys():
            raise ValueError('EMA state_dict keys do not match the current model')
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor)
                    or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype):
                raise ValueError(f'EMA tensor shape/dtype mismatch: {key}')
        base_updates = trainer.ema_model.updates
        if type(base_updates) is not int or base_updates < 0:
            raise ValueError('EMA initializer has invalid update count')
        trainer.ema_model.ema.load_state_dict(state, strict=True)
        trainer.ema_model.updates = base_updates + self.completed_microbatches
        self.applied = True
        if trainer.is_global_zero:
            print(
                f'resume_ema={self.ema_path} epoch={self.epoch} '
                f'global_step={self.global_step} updates={trainer.ema_model.updates}',
                flush=True,
            )
