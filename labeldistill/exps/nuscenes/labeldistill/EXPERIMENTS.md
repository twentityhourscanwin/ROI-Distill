# LabelDistill 消融实验配置总览

## 一、消融实验脚本（R50, key_idxes=[-2,-4,-6,-8]）

逐步叠加的消融链，每步仅引入一个变量：

| 编号 | 脚本文件 | Mask策略 | 通道 | FP蒸馏 | 消融目的 |
|------|----------|----------|------|--------|----------|
| B | `Lidar_gt_label_r50_128x128_e24_full.py` | GT heatmap | 不分割 | 无 | 基线（LabelDistillFull 全通道） |
| A | `Lidar_gt_label_r50_128x128_e_24.py` | GT heatmap | 分割 | 无 | A vs B → 通道分割有效性 |
| C | `LidarDistill_r50_128x128_e24_roi.py` | ROI mask | 分割 | 无 | C vs A → ROI mask 优于 GT mask |
| D | `LidarDistill_r50_128x128_e24_4keyfp.py` | ROI mask | 分割 | 有 | D vs C → FP蒸馏有效性 |

### 关键配置差异

- **B (全通道 GT)**: model=`LabelDistillFull`, mask=`targets[0]`(GT heatmap), 通道不切分
- **A (分割 GT)**: model=`LabelDistill`, mask=`targets[0]`(GT heatmap), backbone输出通道分割后送adaptor
- **C (ROI only)**: model=`LabelDistill`, mask=`BEVDistillationMaskGenerator`(ROI区域), 无FP模块
- **D (ROI+FP)**: model=`LabelDistill`, mask=ROI区域, fp_distill_start_epoch=1, fp_weight=0.3

### 共享配置（A/C/D）

- backbone: ResNet-50, output_channels=150
- key_idxes: [-2, -4, -6, -8] (4帧历史)
- optimizer: AdamW lr=4e-4, backbone lr_mult=0.5
- scheduler: MultiStepLR [19, 21]
- epochs: 24, EMA: True

## 二、不同Backbone的对比实验

| 脚本文件 | Backbone | 分辨率 | 蒸馏策略 | Epochs |
|----------|----------|--------|----------|--------|
| `LidarDistill_r50_128x128_e24_4keyfp.py` | ResNet-50 | 128×128 | ROI+FP | 24 |
| `LidarDistill_r101_128x128_e24_4keyfp.py` | ResNet-101 | 128×128 | ROI+FP | 21 |
| `LidarDistill_swinb_640x1600_e24_4keyfp.py` | Swin-B | 640×1600 | ROI+FP | 24 |
| `LidarDistill_convnextb_900x1600_e24_4keyfp.py` | ConvNeXt-B | 900×1600 | ROI+FP | 30 |
| `Lidar_gt_label_convnextb_900x1600_e30.py` | ConvNeXt-B | 900×1600 | GT(无ROI/FP) | 30 |

## 三、outputs/ 已有结果总结

### 有完整 checkpoint + 评测结果

| 实验目录 | mAP | NDS | 最佳ckpt | 说明 |
|----------|-----|-----|----------|------|
| `BEVDepth_convnextb_900x1600_e30` | 41.67% | 50.36% | epoch 27-29 + last | ConvNeXt-B BEVDepth baseline（无蒸馏） |
| `LidarDistill_convnextb_900x1600_e24_4keyfp` | **47.21%** | **55.78%** | epoch 27-29 + last | ConvNeXt-B ROI+FP蒸馏（最佳） |
| `Lidar_gt_label_convnextb_900x1600_e30` | - | - | epoch 27-29 + last | ConvNeXt-B GT蒸馏（有ckpt，无metrics_summary） |

### 仅有训练日志（无checkpoint/评测）

| 实验目录 | 状态 | 说明 |
|----------|------|------|
| `LidarDistill_r50_128x128_e24_4keyfp` | 仅 lightning_logs (9个version) | R50实验多次尝试，无保存的ckpt和eval结果 |

### ConvNeXt-B 蒸馏效果对比

```
BEVDepth baseline (无蒸馏):  mAP=41.67%  NDS=50.36%
ROI+FP 蒸馏:                 mAP=47.21%  NDS=55.78%
                             ──────────────────────
提升:                        mAP +5.54%  NDS +5.42%
```
