"""Torchvision ConvNeXt image features with strict, single-use initialization."""

from collections.abc import Mapping
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torch.utils import checkpoint as cp
from mmdet.registry import MODELS as MODELS_DET


@MODELS_DET.register_module()
class TorchvisionConvNeXt(nn.Module):
    """Expose four ConvNeXt stages through the mmdet backbone interface."""

    ARCH_SETTINGS = {
        'tiny': 'convnext_tiny',
        'small': 'convnext_small',
        'base': 'convnext_base',
        'large': 'convnext_large',
    }
    WEIGHTS_SETTINGS = {
        'tiny': 'ConvNeXt_Tiny_Weights',
        'small': 'ConvNeXt_Small_Weights',
        'base': 'ConvNeXt_Base_Weights',
        'large': 'ConvNeXt_Large_Weights',
    }
    CHANNELS = {
        'tiny': (96, 192, 384, 768),
        'small': (96, 192, 384, 768),
        'base': (128, 256, 512, 1024),
        'large': (192, 384, 768, 1536),
    }

    def __init__(self, arch='base', out_indices=(1, 2, 3), with_cp=False,
                 pretrained=True, pretrained_weights='IMAGENET1K_V1',
                 frozen_stages=-1, drop_path_rate=0.0,
                 layer_scale_init_value=1e-6, gap_before_final_norm=False,
                 init_cfg=None, **kwargs):
        super().__init__()
        if arch not in self.ARCH_SETTINGS:
            raise ValueError(f'Unsupported ConvNeXt arch: {arch}')
        if (not out_indices or sorted(set(out_indices)) != list(out_indices)
                or any(index not in range(4) for index in out_indices)):
            raise ValueError('out_indices must contain increasing unique stage indices 0..3')
        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be -1..3')
        if pretrained_weights != 'IMAGENET1K_V1':
            raise ValueError('Only explicit IMAGENET1K_V1 weights are supported')
        self.arch = arch
        self.out_indices = list(out_indices)
        self.with_cp = with_cp
        self.frozen_stages = frozen_stages
        self._pretrained = pretrained
        self.pretrained_weights = pretrained_weights
        self._init_cfg = init_cfg
        self._initialized = False
        self.initialization_report = None

        import torchvision.models as tvm
        backbone = getattr(tvm, self.ARCH_SETTINGS[arch])(
            weights=None, stochastic_depth_prob=drop_path_rate,
            layer_scale=layer_scale_init_value)
        self.stages = nn.ModuleList([
            nn.Sequential(backbone.features[2 * index],
                          backbone.features[2 * index + 1])
            for index in range(4)
        ])
        self.out_channels = [self.CHANNELS[arch][index] for index in out_indices]
        self._freeze_stages()

    def _freeze_stages(self):
        for index in range(self.frozen_stages + 1):
            self.stages[index].eval()
            self.stages[index].requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        return self

    @staticmethod
    def _file_sha256(path):
        if path is None or not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _unwrap_checkpoint(checkpoint, prefix=''):
        if not isinstance(checkpoint, Mapping):
            raise TypeError('Checkpoint must be a state mapping or a state_dict/model container')
        containers = [key for key in ('state_dict', 'model') if key in checkpoint]
        if len(containers) > 1:
            raise ValueError('Ambiguous checkpoint: both state_dict and model are present')
        state = checkpoint[containers[0]] if containers else checkpoint
        if not isinstance(state, Mapping) or not state:
            raise ValueError('Checkpoint state must be a non-empty mapping')
        if any(not isinstance(key, str) for key in state):
            raise ValueError('Checkpoint tensor keys must be strings')
        if prefix:
            state = {key[len(prefix):]: value for key, value in state.items()
                     if key.startswith(prefix)}
            if not state:
                raise ValueError(f'Checkpoint prefix matched no tensors: {prefix}')
        return state

    def _load_torchvision_weights(self, state):
        features = {}
        ignored = []
        for key, tensor in state.items():
            if key.startswith('classifier.'):
                ignored.append(key)
            elif key.startswith('features.'):
                features[key[len('features.'):]] = tensor
            else:
                raise ValueError(f'Unexpected torchvision checkpoint key: {key}')
        report = self._load_from_features_state_dict(features)
        report['ignored_classifier_keys'] = sorted(ignored)
        return report

    def _load_from_features_state_dict(self, features_sd):
        mapped = {}
        for key, tensor in features_sd.items():
            index, separator, suffix = key.partition('.')
            if not separator or not index.isdigit() or int(index) not in range(8):
                raise ValueError(f'Invalid torchvision feature key: {key}')
            stage_index, sub_index = divmod(int(index), 2)
            mapped[f'stages.{stage_index}.{sub_index}.{suffix}'] = tensor
        own_state = self.state_dict()
        missing = sorted(set(own_state) - set(mapped))
        unexpected = sorted(set(mapped) - set(own_state))
        if missing or unexpected:
            raise RuntimeError(f'Incomplete ConvNeXt features: missing={missing}, unexpected={unexpected}')
        for key, tensor in mapped.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f'Checkpoint value is not a tensor: {key}')
            if tensor.shape != own_state[key].shape:
                raise RuntimeError(
                    f'ConvNeXt shape mismatch for {key}: '
                    f'{tuple(tensor.shape)} != {tuple(own_state[key].shape)}')
        self.load_state_dict(mapped, strict=True)
        return {
            'loaded_keys': sorted(mapped),
            'loaded_tensor_count': len(mapped),
            'layer_scale_tensor_count': sum('layer_scale' in key for key in mapped),
            'missing_keys': missing,
            'unexpected_keys': unexpected,
        }

    def init_weights(self):
        """Load every feature tensor exactly once; invalid inputs fail explicitly."""
        if self._initialized:
            return
        source_path = None
        if self._init_cfg is not None:
            if not isinstance(self._init_cfg, Mapping):
                raise TypeError('init_cfg must be a mapping with checkpoint')
            checkpoint = self._init_cfg.get('checkpoint')
            if not isinstance(checkpoint, str) or not checkpoint:
                raise ValueError('init_cfg.checkpoint must be a non-empty path or URL')
            prefix = self._init_cfg.get('prefix', '')
            if not isinstance(prefix, str):
                raise TypeError('init_cfg.prefix must be a string')
            if checkpoint.startswith(('https://', 'http://')):
                state = torch.hub.load_state_dict_from_url(
                    checkpoint, map_location='cpu', check_hash=True)
                source_path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(checkpoint).path).name
            else:
                source_path = Path(checkpoint)
                if not source_path.is_file():
                    raise FileNotFoundError(f'ConvNeXt checkpoint not found: {source_path}')
                state = torch.load(source_path, map_location='cpu', weights_only=True)
            state = self._unwrap_checkpoint(state, prefix)
            report = self._load_torchvision_weights(state)
            report['source'] = checkpoint
            report['weights_enum'] = None
        elif self._pretrained:
            import torchvision.models as tvm
            weights = getattr(tvm, self.WEIGHTS_SETTINGS[self.arch])[self.pretrained_weights]
            state = weights.get_state_dict(progress=True, check_hash=True)
            report = self._load_torchvision_weights(self._unwrap_checkpoint(state))
            report['source'] = weights.url
            report['weights_enum'] = f'{self.WEIGHTS_SETTINGS[self.arch]}.{weights.name}'
            source_path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(urlparse(weights.url).path).name
        else:
            report = {
                'source': 'random', 'weights_enum': None,
                'loaded_keys': [], 'loaded_tensor_count': 0,
                'layer_scale_tensor_count': 0,
                'missing_keys': [], 'unexpected_keys': [],
                'ignored_classifier_keys': [],
            }
        report['checkpoint_path'] = str(source_path) if source_path is not None else None
        report['checkpoint_sha256'] = self._file_sha256(source_path)
        self.initialization_report = report
        self._initialized = True
        print(f'[TorchvisionConvNeXt] source={report["source"]}, '
              f'loaded={report["loaded_tensor_count"]}, '
              f'LayerScale={report["layer_scale_tensor_count"]}, missing=0, unexpected=0')

    def forward(self, x):
        outputs = []
        feature = x
        for index, stage in enumerate(self.stages):
            feature = stage[0](feature)
            for block in stage[1]:
                # Non-reentrant checkpoints retain parameter gradients even when
                # camera pixels themselves do not require gradients.
                if self.with_cp and self.training and torch.is_grad_enabled():
                    feature = cp.checkpoint(block, feature, use_reentrant=False)
                else:
                    feature = block(feature)
            if index in self.out_indices:
                outputs.append(feature)
        return tuple(outputs)
