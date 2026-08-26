# 当前基线：B1

最后更新：2026-08-26

状态：**Implemented / Awaiting rerun**

Baseline ID：`B1`

实现快照：`d27eed6`

入口配置：`configs/experiments/b1_teacher_value_no_scale.yaml`

> B1 是当前候选基线，不是已经得到有效最终指标的 benchmark。B0、B1、B1T、B2 均需在本页口径下重新训练与验证。

## 1. 系统概览

```text
6-camera × 5 timestamps
  → BEVDepth R50 student
  → temporal BEV concat [B, 750, 128, 128]
  ├─ full channels → detection trunk/head
  ├─ selected channels → feature adaptor → teacher BEV features
  └─ student predictions → response KD

current LiDAR + 9 past sweeps
  → frozen CenterPoint teacher (eval)
  ├─ BEV feature targets
  └─ decoded proposals → GT matching → q / ROI masks
```

## 2. 学生网络与通道选择

| 项目 | 当前设置 |
| --- | --- |
| 学生类型 | `camera_bevdepth_r50` |
| 图像源尺寸 / 输入尺寸 | `900×1600` / `256×704` |
| 图像时序 | 当前帧 + `[-2, -4, -6, -8]`，共 5 个时刻 |
| 单时刻 BEV 通道 | 150 |
| 时序拼接 | `[B, 750, 128, 128]` |
| 检测路径 | 使用完整 750 通道；没有把一半通道从检测网络中删除 |
| KD 选择 | `temporal_kd_selection: legacy_half` |
| L0 feature KD | 从 750 通道取前 375，经 adaptor 对齐到教师 128 通道 |
| L1 feature KD | trunk 混合后 `[B,300,64,64]` 取前 150，经 adaptor 对齐到教师 256 通道 |

L0 的前 375 通道对应“当前 150 + `t-2` 150 + `t-4` 的前 75”；L1 已经过卷积混合，因此“前一半”不再具有干净的时间语义。当前实现只限制直接承受 feature KD 的张量切片，不能严格保证一半特征只学 LiDAR、另一半完整保留相机私有能力。

## 3. 教师与数据口径

教师固定为冻结、`eval` 状态的 OpenMMLab CenterPoint：

```text
ckpts/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth
SHA256: 9061688e5f81adae87d28241143e2d33075f68908134264f0de8c901acf911d8
```

数据固定为：

```text
data/nuScenes/nuscenes_infos_train_10sweeps_past9.pkl
data/nuScenes/nuscenes_infos_val_10sweeps_past9.pkl
```

- 教师输入为当前 LiDAR sweep + 最多 9 个过去 sweeps；禁止未来 sweep。
- 历史不足时用当前 sweep 补足至 10。
- 历史点变换到当前坐标系，`time_lag = current_timestamp - sweep_timestamp`。
- BDA 对 GT 与教师 LiDAR xyz 使用同一 `3×3` 变换；反射率和时间通道不变。
- 学生与教师使用相同 nuScenes sample，点云范围为 `[-51.2, -51.2, -5, 51.2, 51.2, 3]`。

## 4. B1 匹配、价值与 mask

Proposal–GT 匹配：

```text
exact-class-only
score-first nearest-unmatched GT
one-to-one
strict distance < threshold
small classes: 1 m
large classes: 2 m
```

Small：`barrier, motorcycle, bicycle, pedestrian, traffic_cone`。

Large：`car, truck, construction_vehicle, bus, trailer`。

实例价值：

```text
r_i = d_i / tau_i
q_i = max(0, 1 - r_i^2)
unmatched q_i = 0
```

B1 使用原始 BDA 后 GT 几何生成 circular Gaussian；不启用 AdaptiveGTScalerV3。每个实例 mask 归一化，重叠位置取 `max`。Feature loss 采用 `per_gt_fixed_count`，分母为 effective GT 数，禁止改成 `sum(q)` 或 `mask.sum()` 后仍称为 B1。

## 5. 损失与训练参数

```text
detection_weight = 1.0
depth_weight = 1.0
feature_weight = 1.0
response_weight = 1.0
optimizer = AdamW
base_lr_at_global_batch_64 = 4e-4
backbone_lr_mult = 0.5
weight_decay = 0.01
epochs = 24
scheduler = MultiStepLR, milestones [19, 23]
EMA = enabled
deterministic = true
```

配置默认是 2 GPU × 16 samples/GPU；可以覆盖硬件参数，但改变全局 batch size 时必须记录 resolved learning rate。

## 6. B 系列唯一变量

| 实验 | 相对 B1 的唯一变化 | 配置 |
| --- | --- | --- |
| B0 | `region.value=uniform_gt`，effective GT 的 `q=1` | `b0_full_gt_uniform_no_scale.yaml` |
| B1 | teacher value + circular Gaussian + no scale | `b1_teacher_value_no_scale.yaml` |
| B1T | 仅改成 oriented elliptical Gaussian | `b1t_teacher_value_elliptical_mask.yaml` |
| B2 | 仅启用 AdaptiveGTScalerV3 | `b2_teacher_value_adaptive_scale.yaml` |

四组实验应使用同一实现提交、教师、数据、seed、硬件与训练参数；不需要建立四个 Git 分支。

## 7. 已知限制与进入正式训练前的门禁

- `legacy_half` 是切片式弱约束，不是显式的 KD/private 双分支；后续应以 no-feature-KD、full-KD、legacy-half、explicit-split 做结构消融。
- AdaptiveGTScalerV3 的连续几何变化经过整数 radius 与 `min_radius=2` 后，B2 的最终 mask 可能经常与 B1 完全相同；正式 B2 必须记录 `mask_changed_rate`。
- scaler 的局部 x 方向 deadzone 当前不对称：小正偏移会生效，小负偏移可能不生效；修复前 B2 不得形成最终结论。
- 当前训练 dataloader 为 `shuffle=False`、`sampler=None` 且 `use_cbgs=false`，需要在正式运行前明确接受或修正。
- trainer 当前 `limit_val_batches=0`，fit 阶段不做 validation；指标必须由独立 evaluation 产生并登记。
- KDHead 仍生成未被模型主体消费的旧 `bev_mask/bev_box/bev_label`，属于性能清理项，不应改变数值结果。

只有门禁问题有明确决策、120 项合同测试通过，并完成独立评测后，才能把 B1 状态改为 `Validated`。
