# labeldistill/layers/backbones/convnext_backbone.py
"""
使用 torchvision 自带的 ConvNeXt 作为 backbone，无需安装 mmpretrain/mmcls。
将其包装为与 mmdet backbone 兼容的接口，并注册到 mmdet::model 注册表。

接口约定（与 ResNet 一致）：
  - forward(x) 返回 tuple of tensors，每个对应一个 out_indices 指定的 stage 输出
  - 支持 init_weights() 方法（加载 torchvision 预训练权重）
  - 支持 with_cp（梯度检查点），节省显存

ConvNeXt-B 各 stage 输出通道：
  stage0: 128   (downsample 4x)
  stage1: 256   (downsample 8x)
  stage2: 512   (downsample 16x)
  stage3: 1024  (downsample 32x)
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmdet.registry import MODELS as MODELS_DET


@MODELS_DET.register_module()
class TorchvisionConvNeXt(nn.Module):
    """
    Torchvision ConvNeXt 的 mmdet-compatible 包装器。

    Args:
        arch (str): 模型规模，支持 'tiny'|'small'|'base'|'large'。默认 'base'。
        out_indices (list[int]): 输出哪些 stage 的特征，默认 [1, 2, 3]。
            ConvNeXt 共 4 个 stage（0~3），索引对应：
              0 -> stride 4,   channels: tiny=96,  small=96,  base=128, large=192
              1 -> stride 8,   channels: tiny=192, small=192, base=256, large=384
              2 -> stride 16,  channels: tiny=384, small=384, base=512, large=768
              3 -> stride 32,  channels: tiny=768, small=768, base=1024,large=1536
        with_cp (bool): 是否开启梯度检查点（节省显存，训练变慢）。默认 False。
        pretrained (bool): 是否加载 torchvision ImageNet-1k 预训练权重。默认 True。
        frozen_stages (int): 冻结前 N 个 stage（-1 表示不冻结）。默认 -1。
    """

    ARCH_SETTINGS = {
        'tiny':  'convnext_tiny',
        'small': 'convnext_small',
        'base':  'convnext_base',
        'large': 'convnext_large',
    }

    def __init__(self,
                 arch='base',
                 out_indices=(1, 2, 3),
                 with_cp=False,
                 pretrained=True,
                 frozen_stages=-1,
                 # 以下参数为与 mmdet backbone 配置兼容而保留，暂不生效
                 drop_path_rate=0.0,
                 layer_scale_init_value=1e-6,
                 gap_before_final_norm=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__()

        assert arch in self.ARCH_SETTINGS, \
            f'arch must be one of {list(self.ARCH_SETTINGS.keys())}, got {arch}'
        self.arch = arch
        self.out_indices = list(out_indices)
        self.with_cp = with_cp
        self.frozen_stages = frozen_stages
        self._pretrained = pretrained
        self._init_cfg = init_cfg   # 保留但由 init_weights() 处理

        # 构建 torchvision ConvNeXt
        import torchvision.models as tvm
        model_name = self.ARCH_SETTINGS[arch]
        # 不传 pretrained 参数，在 init_weights() 里手动加载
        # stochastic_depth_prob 对应 drop_path_rate（随机深度正则化）
        backbone = getattr(tvm, model_name)(weights=None,
                                            stochastic_depth_prob=drop_path_rate)

        # torchvision ConvNeXt 结构：
        #   features[0]  : stem (PatchEmbedding)   stride=4
        #   features[1]  : stage0 blocks
        #   features[2]  : downsample stride=2
        #   features[3]  : stage1 blocks
        #   features[4]  : downsample stride=2
        #   features[5]  : stage2 blocks
        #   features[6]  : downsample stride=2
        #   features[7]  : stage3 blocks
        # => stage i 的输出来自 features[i*2+1]（block 部分）
        # 我们把它拆成 4 个阶段，每阶段 = [downsample(if i>0), blocks]
        self.stages = nn.ModuleList()
        # stage0: stem + blocks
        self.stages.append(nn.Sequential(backbone.features[0],
                                         backbone.features[1]))
        # stage1~3: downsample + blocks
        for i in range(1, 4):
            self.stages.append(nn.Sequential(backbone.features[i * 2],
                                             backbone.features[i * 2 + 1]))

        # 记录各 stage 输出通道（用于外部查询）
        _ch = {'tiny': [96, 192, 384, 768],
                'small': [96, 192, 384, 768],
                'base': [128, 256, 512, 1024],
                'large': [192, 384, 768, 1536]}
        self.out_channels = [_ch[arch][i] for i in self.out_indices]

        # 冻结 stages
        self._freeze_stages()

    def _freeze_stages(self):
        for i in range(self.frozen_stages + 1):
            if i < len(self.stages):
                stage = self.stages[i]
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """加载预训练权重（torchvision ImageNet-1k 或自定义 checkpoint）"""
        # 优先使用 init_cfg 指定的 checkpoint
        if self._init_cfg is not None:
            ckpt_path = self._init_cfg.get('checkpoint', None)
            prefix = self._init_cfg.get('prefix', '')
            if ckpt_path and ckpt_path.startswith('http'):
                import torch.hub as hub
                print(f'[TorchvisionConvNeXt] Loading weights from URL: {ckpt_path}')
                state_dict = hub.load_state_dict_from_url(
                    ckpt_path, map_location='cpu', check_hash=False)
                # 如果有 prefix，去掉 prefix
                if prefix:
                    state_dict = {k[len(prefix):]: v
                                  for k, v in state_dict.items()
                                  if k.startswith(prefix)}
                # 如果 checkpoint 含有 'state_dict' key
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self._load_torchvision_weights(state_dict)
                return
            elif ckpt_path and not ckpt_path.startswith('http'):
                import os
                if os.path.exists(ckpt_path):
                    print(f'[TorchvisionConvNeXt] Loading weights from: {ckpt_path}')
                    state_dict = torch.load(ckpt_path, map_location='cpu')
                    if prefix:
                        state_dict = {k[len(prefix):]: v
                                      for k, v in state_dict.items()
                                      if k.startswith(prefix)}
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    self._load_torchvision_weights(state_dict)
                    return

        # 没有指定 checkpoint，使用 torchvision 默认 ImageNet-1k 预训练权重
        if self._pretrained:
            import torchvision.models as tvm
            print(f'[TorchvisionConvNeXt] Loading torchvision ImageNet-1k '
                  f'pretrained weights for ConvNeXt-{self.arch}...')
            # 使用 DEFAULT weights（等效于 IMAGENET1K_V1）
            pretrained_model = getattr(tvm, self.ARCH_SETTINGS[self.arch])(
                weights='DEFAULT')
            # pretrained_model.features 与 self.stages 结构完全对应
            # features.state_dict() 的 key 格式：'0.xxx', '1.xxx', ...
            # 我们的 stages.i.0.xxx / stages.i.1.xxx 对应 features.(i*2) / features.(i*2+1)
            self._load_from_features_state_dict(
                pretrained_model.features.state_dict())
            print('[TorchvisionConvNeXt] Weights loaded successfully.')
        else:
            print('[TorchvisionConvNeXt] No pretrained weights loaded.')

    def _load_from_features_state_dict(self, features_sd):
        """
        将 torchvision model.features 的 state_dict 加载到本模块的 stages。

        torchvision features 结构（8个子模块，索引0~7）：
          features.0  -> stages.0.0  (stem)
          features.1  -> stages.0.1  (stage0 blocks)
          features.2  -> stages.1.0  (downsample)
          features.3  -> stages.1.1  (stage1 blocks)
          features.4  -> stages.2.0  (downsample)
          features.5  -> stages.2.1  (stage2 blocks)
          features.6  -> stages.3.0  (downsample)
          features.7  -> stages.3.1  (stage3 blocks)

        features_sd 的 key 格式：'{feat_idx}.layer.weight' 等
        本模块的 key 格式：'stages.{stage_i}.{sub_i}.layer.weight'
        """
        # 建立 features index -> (stage_i, sub_i) 的映射
        feat_to_stage = {
            0: (0, 0), 1: (0, 1),   # stage0
            2: (1, 0), 3: (1, 1),   # stage1
            4: (2, 0), 5: (2, 1),   # stage2
            6: (3, 0), 7: (3, 1),   # stage3
        }

        new_sd = {}
        for feat_key, val in features_sd.items():
            # feat_key 形如 '0.block.0.weight' 或 '0.weight'
            parts = feat_key.split('.', 1)
            feat_idx = int(parts[0])
            rest = parts[1] if len(parts) > 1 else ''
            if feat_idx in feat_to_stage:
                si, sj = feat_to_stage[feat_idx]
                own_key = f'stages.{si}.{sj}.{rest}' if rest else f'stages.{si}.{sj}'
                new_sd[own_key] = val

        missing = [k for k in self.state_dict() if k not in new_sd]
        unexpected = [k for k in new_sd if k not in self.state_dict()]
        if missing:
            print(f'[TorchvisionConvNeXt] Missing keys ({len(missing)}): '
                  f'{missing[:3]}{"..." if len(missing)>3 else ""}')
        if unexpected:
            print(f'[TorchvisionConvNeXt] Unexpected keys ({len(unexpected)}): '
                  f'{unexpected[:3]}{"..." if len(unexpected)>3 else ""}')
        self.load_state_dict(new_sd, strict=False)

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, 3, H, W]
        Returns:
            tuple[Tensor]: 各 out_indices 对应的 stage 输出特征

        显存优化说明：
            with_cp=True 时，对每个 stage 内的每个子模块（CNBlock）单独做梯度检查点，
            而不是对整个 stage 做检查点。这样反向传播时只需重算单个 block，
            显存占用大幅降低（约减少 40-50%），代价是训练速度略慢。
        """
        outs = []
        feat = x
        for i, stage in enumerate(self.stages):
            if self.with_cp and feat.requires_grad:
                # 对 stage 内每个子模块逐一做细粒度梯度检查点
                # stage 是 nn.Sequential，包含 [downsample(可选), nn.Sequential(blocks)]
                for sub in stage:
                    if isinstance(sub, nn.Sequential):
                        # sub 是 blocks 序列，对每个 block 单独做检查点
                        for block in sub:
                            feat = cp.checkpoint(block, feat, use_reentrant=True)
                    else:
                        # sub 是 downsample 层（LayerNorm2d + Conv2d），直接前向
                        feat = sub(feat)
            else:
                feat = stage(feat)
            if i in self.out_indices:
                outs.append(feat)
        return tuple(outs)

