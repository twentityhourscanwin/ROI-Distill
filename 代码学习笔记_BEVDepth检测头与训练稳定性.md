# 代码学习笔记：BEVDepth 检测头与训练稳定性

> 基于 `LidarDistill_r50_128x128_e24_roi.py` 和 `LidarDistill_convnextb_900x1600_e24_roi.py` 的代码分析

---

## 一、BEVDepth 整体结构（R50 配置）

整个模型核心是 `LabelDistill`，训练流程：

```
多视角图像 → BaseLSSFPN(backbone) → KDHead(head) → 检测结果
                ↓                       ↓
         BEV 特征图 (体素池化)    CenterPoint 风格预测
```

**BaseLSSFPN** (`base_lss_fpn.py`) 包含 3 个子模块：

| 子模块 | 作用 |
|--------|------|
| `img_backbone` (ResNet-50) + `img_neck` (SECONDFPN) | 提取多尺度图像特征，FPN 融合为 512 通道 |
| `DepthNet` | 从图像特征预测 **深度概率分布** + **上下文特征**，输出 `[D+C, H, W]` |
| `voxel_pooling_train` | 将深度加权特征 **投射并聚合** 到 BEV 网格 |

---

## 二、体素池化（Voxel Pooling）

BEVDepth 最核心的 2D→3D→BEV 转换步骤，在 `_forward_single_sweep` 中完成。

### Step 1: 创建视锥体 (Frustum)

生成形状为 `[D, fH, fW, 4]` 的视锥体网格。`d_bound = [2.0, 58.0, 0.5]`，深度通道 D = 112。

### Step 2: 视锥体反投影到 ego 坐标系

```python
# 逆 IDA 增强
points = ida_mat.inverse().matmul(points)
# 相机坐标 → ego坐标
combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
points = combine.matmul(points)
```

得到 `geom_xyz: [B, 6, 112, fH, fW, 3]`。

### Step 3: 深度加权外积

```python
img_feat_with_depth = depth.unsqueeze(1) * depth_feat_slice.unsqueeze(2)
# depth: [B*N, D, H, W] — softmax 后的深度概率
# context: [B*N, C, H, W] — 上下文特征 (C=150)
# 结果: [B*N, C, D, H, W]
```

### Step 4: 体素池化 (CUDA scatter-add)

```python
feature_map = voxel_pooling_train(geom_xyz, img_feat_with_depth, self.voxel_num.cuda())
# 输出: [B, 150, 128, 128]
```

体素网格：`x_bound=[-51.2, 51.2, 0.8]`，BEV 分辨率 128×128。

### 多帧融合

`key_idxes = [-2, -4, -6, -8]`，5 帧 channel-wise concat：`150 × 5 = 750` 通道。

---

## 三、KDHead 检测头结构

继承自 mmdet3d 的 `CenterHead`，采用 CenterPoint 风格的 anchor-free 检测。

### 整体拓扑

```
BEV特征 [B, 750, 128, 128]
     │
     ├─ BEV Backbone (ResNet-18, base_channels=300)
     │     输出 trunk_outs:
     │       [0]: [B, 750, 128, 128]  (原始输入直通)
     │       [1]: [B, 300, 64, 64]
     │       [2]: [B, 600, 32, 32]
     │       [3]: [B, 1200, 16, 16]
     │
     ├─ BEV Neck (SECONDFPN)
     │     输出: [B, 256, 128, 128]
     │
     └─ CenterHead
           shared_conv (3×3, 256→64)  ← 唯一共享的层
               │
         ┌─────┼─────┬─────┬─────┬─────┐
       Task0 Task1 Task2 Task3 Task4 Task5
       (独立SeparateHead × 6)
```

### 6 组任务的类别分配

| Task | 类别 | heatmap 通道数 |
|------|------|--------------|
| Task 0 | car | 1 |
| Task 1 | truck, construction_vehicle | 2 |
| Task 2 | bus, trailer | 2 |
| Task 3 | barrier | 1 |
| Task 4 | motorcycle, bicycle | 2 |
| Task 5 | pedestrian, traffic_cone | 2 |

### SeparateHead 内部结构

**每个 Task 拥有完全独立的 SeparateHead，包括自己的 heatmap（不共享）。**

每个 SeparateHead = 6 条并行小卷积链：

```
shared_conv输出 [B, 64, 128, 128]
     │
     ├─ heatmap: Conv3×3(64→64)+BN+ReLU → Conv3×3(64→num_cls) → [B, num_cls, 128, 128]
     ├─ reg:     Conv3×3(64→64)+BN+ReLU → Conv3×3(64→2)       → [B, 2, 128, 128]
     ├─ height:  Conv3×3(64→64)+BN+ReLU → Conv3×3(64→1)       → [B, 1, 128, 128]
     ├─ dim:     Conv3×3(64→64)+BN+ReLU → Conv3×3(64→3)       → [B, 3, 128, 128]
     ├─ rot:     Conv3×3(64→64)+BN+ReLU → Conv3×3(64→2)       → [B, 2, 128, 128]
     └─ vel:     Conv3×3(64→64)+BN+ReLU → Conv3×3(64→2)       → [B, 2, 128, 128]
```

heatmap 分支 bias 初始化为 -2.19（sigmoid(-2.19) ≈ 0.1，让初始预测倾向背景）。

**设计原因**："Separate" 意为把不同物理含义的预测分开用独立卷积做——不同输出的数值分布差异很大（heatmap 要 sigmoid 到 [0,1]，dim 是 log 空间，rot 是 sin/cos），共享参数难以适配。

总计：6 Task × 6 分支 = **36 个独立小卷积链** + 1 个 shared_conv。

---

## 四、训练目标生成 (`get_targets_single`)

### 流程

1. **按任务分组 GT boxes**
2. **计算 BEV 特征图上的中心坐标**：`coor = (world_coord + 51.2) / 0.8`，映射到 [0, 128)
3. **计算高斯半径**：取决于物体 BEV 投影大小，最小半径 = 2
4. **绘制高斯热图**：中心值 1.0，向外高斯衰减到 0
5. **填写回归目标** (10维)：`[cx_frac, cy_frac, z, log(w), log(l), log(h), sin(rot), cos(rot), vx, vy]`
6. **记录索引和掩码**：`ind = y * W + x`（展平索引），`mask = 1`

---

## 五、GaussianFocalLoss 正/负/不确定样本权重

### 核心公式

```python
pos_weights = gaussian_target.eq(1)              # 严格 =1 才是正样本
neg_weights = (1 - gaussian_target).pow(gamma)    # gamma=4
pos_loss = -log(pred) * (1 - pred)^alpha * pos_weights          # alpha=2
neg_loss = -log(1 - pred) * pred^alpha * neg_weights
loss = pos_loss + neg_loss
```

### 三类样本

| 区域 | target 值 | 参与的损失分支 | 权重机制 |
|------|----------|-------------|---------|
| GT 中心（正样本） | = 1.0 | pos_loss | `(1-pred)^2` focal 调制：已预测好的降权 |
| 高斯衰减区 | 0 < t < 1 | neg_loss | `pred^2 × (1-t)^4`：离中心越近权重越低 |
| 纯背景（负样本） | = 0 | neg_loss | `pred^2 × 1`：容易负样本被 pred^2 压低 |

**全图每个像素都参与损失计算**。高斯衰减区作为"柔和负样本"处理，靠近 GT 中心的位置惩罚被 `(1-target)^4` 大幅压低。

`(1-target)^4` 权重示例：target=0.9 → 0.0001, target=0.5 → 0.0625, target=0.1 → 0.66

`avg_factor = 正样本数量`（跨 GPU reduce_mean 后），保证 loss scale 稳定。

### 回归损失

**只在正样本位置计算**（通过 `_gather_feat(pred, ind)` 取 GT 中心点的预测）。

---

## 六、推理全流程

### 关键：6 个 Task 独立处理，最后才合并

```
6个Task各自独立处理：
  Task k 的 heatmap [B, num_cls_k, 128, 128]
    │
    ├─ sigmoid
    ├─ 第一次 top-K：每类通道取500个候选
    ├─ 第二次 top-K：跨类合并取全局500个（Task内部跨类，不是跨Task）
    ├─ gather 回归分支（reg, height, dim, rot, vel）
    ├─ 加上 reg 亚像素偏移
    ├─ 坐标反映射：x_world = xs * 0.8 - 51.2
    ├─ score > 0.1 过滤
    ├─ 空间范围过滤 (post_center_range)
    └─ Circle NMS（半径=min_radius[k]，最多83个）
         │
         └─ 输出 {bboxes, scores, labels}

最后合并：
  6个Task结果 concat + label偏移 → 最终 [bboxes, scores, labels]
  每个样本最多 6 × 83 = 498 个检测框
```

### 两次 Top-K 详解（以 Task 1 为例）

```
Task 1 heatmap: [B, 2, 128, 128]  (truck + construction_vehicle)
  │
  ├─ truck 通道 → top-500 → 500 个候选
  ├─ construction_vehicle 通道 → top-500 → 500 个候选
  │
  └─ 合并 1000 个 → 再取 top-500 → 最终 500 个（带类别标签）
```

### Circle NMS

用**欧氏距离**（非 IoU）判断重叠。不同 Task 半径不同：

| Task | 类别 | min_radius |
|------|------|-----------|
| 0 | car | 4 |
| 1 | truck, construction_vehicle | 12 |
| 2 | bus, trailer | 10 |
| 3 | barrier | 1 |
| 4 | motorcycle, bicycle | 0.85 |
| 5 | pedestrian, traffic_cone | 0.175 |

### 多任务 Label 偏移合并

```
Task 0: labels 0      → +0 → 全局 0 (car)
Task 1: labels 0,1    → +1 → 全局 1,2 (truck, construction_vehicle)
Task 2: labels 0,1    → +3 → 全局 3,4 (bus, trailer)
Task 3: labels 0      → +5 → 全局 5 (barrier)
Task 4: labels 0,1    → +6 → 全局 6,7 (motorcycle, bicycle)
Task 5: labels 0,1    → +8 → 全局 8,9 (pedestrian, traffic_cone)
```

---

## 七、ConvNeXt-B 训练稳定性措施

ConvNeXt-B 使用更大 backbone + 更高分辨率 (896×1600)，需要额外稳定训练。

### 措施 1：KL 散度软标签深度损失（替代 BCE）

R50 用 one-hot + BCE，ConvNeXt-B 改为高斯软标签 + KL 散度：

```python
# 生成高斯软标签 (sigma=1.5)
distances = depth_indices_fg.view(N, 1) - bin_indices.view(1, -1)
weights = torch.exp(-0.5 * (distances / sigma) ** 2)
soft_labels = weights / weights.sum(dim=1, keepdim=True)

# KL散度损失
depth_probs = F.softmax(depth_preds_fg, dim=1)
depth_loss = F.kl_div(torch.log(depth_probs + 1e-8), soft_labels, reduction='batchmean')
```

好处：AMP 混合精度下更稳定，软标签提供更平滑的梯度。

### 措施 2：梯度检查点 (`with_cp=True`)

用时间换显存。配合 DDP 静态图兼容（`_set_static_graph()`）。

### 措施 3：Backbone 差异化学习率

- R50: `backbone lr_mult = 0.5`
- ConvNeXt-B: `backbone lr_mult = 0.1`（更激进衰减，防止预训练参数剧变）

### 措施 4：Warmup + Cosine 学习率

```python
warmup_iters = 1000, warmup_ratio = 0.001
# 前 1000 步：lr × 0.001 线性升温到满 lr
# 之后：cosine 平滑衰减
```

R50 用简单的 `MultiStepLR([19, 23])`。

### 措施 5：随机深度正则化 (`drop_path_rate=0.3`)

防止大模型过拟合。

### 措施 6：Response Loss 缩放

```python
response_loss = 0.6 * response_loss  # 降低蒸馏信号干扰
```

### 两个配置对比总结

| 维度 | R50 (128×128) | ConvNeXt-B (900×1600) |
|------|--------------|----------------------|
| Backbone | ResNet-50 | ConvNeXt-Base |
| 输入分辨率 | 256×704 | 896×1600 |
| 深度损失 | BCE + one-hot | KL散度 + 高斯软标签 |
| 梯度检查点 | 无 | with_cp=True |
| Backbone lr | ×0.5 | ×0.1 |
| LR Schedule | MultiStepLR [19,23] | Warmup 1000 + Cosine |
| Drop Path | 无 | 0.3 |
| DDP 静态图 | 无 | _set_static_graph() |
