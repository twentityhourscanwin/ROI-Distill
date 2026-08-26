# LidarDistill ConvNeXt-B 16 卡训练：网络与问题诊断

> 检查日期：2026-08-17
> 更新日期：2026-08-19  
> 目标配置：`labeldistill/exps/nuscenes/labeldistill/big_backbone/LidarDistill_convnextb_900x1600_J4_wl05_wh08_cbgs.py`  
> 启动方式：`python ...py --gpus 16 -b 4`  
> 检查方式：当前仓库静态追踪、最小数值例验证、已有 TensorBoard/nuScenes 产物核对。没有修改训练代码，也没有连接或干预 DLC 上正在运行的进程。

## 0. 2026-08-19 代码更新状态

本文档前半部分保留对原始训练和 epoch 19 checkpoint 的诊断结论；以下修复只作用于后续重新训练，旧 checkpoint 的参数与行为不会被追溯修改。

- 已修复 medium ROI：先把全局坐标偏移旋转到 box-local frame，再按局部长宽方向扩展。
- 已修复速度补偿：high、medium 和 unmatched 分支统一把全局速度旋转到 box-local frame 后再计算四条边界扩展。
- 已撤销 AdaptiveGTScalerV3 的 base floor；AdaptiveGTScalerV2 保持不变。
- 已保留“时序特征切一半用于蒸馏”的设计，并将第一层 feature KD 调整为 current 全通道 + t-2 全通道 + t-6 半通道，共 375 通道。
- 检测 head 的输入顺序不变，仍为 current、t-2、t-4、t-6、t-8，避免改变时序融合语义。

## 1. 结论摘要

当前配置能完成训练，局部已有一次稳定跑到约 50,200 step 的记录，但 ROI 框放缩链路确实有问题，而且不止一处。

| 优先级 | 结论 | 是否影响当前训练 |
|---|---|---|
| P0（已修复） | AdaptiveGTScalerV3 的 offset 与 velocity 原先未转换到 box-local frame | 修复只作用于后续重新训练；旧 checkpoint 保持原行为 |
| P0 | 自适应“扩框”最终只用于生成无朝向的圆形 Gaussian；`int(radius)` 加 `min_radius=2` 使汽车和全部小目标的常见扩框前后 mask 完全一样 | 是，大部分 medium/unmatched 扩框实际不生效 |
| P0/P1 | 文件名是 `wl05_wh08`，实际构造为 `w_l=0.5, w_h=0.7` | 是；若本轮目标是 `w_h=0.8`，实验标签错误 |
| P1 | 教师和当前学生默认都用 `train + val` 训练，且 Trainer 关闭验证；已有教师 0.864 mAP / 0.828 NDS 是在见过 val 后测 val，不能作为泛化结果 | 是，污染实验评估解释 |
| P1 | 保存的 `hparams.yaml` 是父类修改前的配置，仍记录 `output_channels=80` 等值，而实际网络使用 150/750 通道 | 是，影响复现和 checkpoint 审计 |
| 设计确认 | 第一层 feature KD 显式选择 current + t-2 + half(t-6)，保留 channel-split idea；检测 head 仍按 current、t-2、t-4、t-6、t-8 | 仅影响后续新训练；建议与 full-channel 做消融 |
| P2 | 每步生成未使用的 label-input 栅格、解码未使用的学生框，并在 ROI/scaler 中大量 `.item()`/NumPy/CPU NMS | 是，主要影响吞吐 |
| P2 | 教师 proposal 实际按学生 head 的 `score_threshold=0.1` 解码，而声明的 LiDAR test cfg 是 0.15 | 是，proposal 分布与配置注释不一致 |
| P2 | `gen_labelinput` 里另有一次明确的二次除 2，框栅格边长只有期望的一半；当前 `lidardistill.py` 不消费它，所以本轮只产生开销，不改变前向结果 | 当前结果否；其他 label-input 模型是 |

如果当前 16 卡任务的目的是验证“J4 + `w_h=0.8`”或验证论文式自适应扩框，建议不要把它当作有效正式实验；若只是训练稳定性/吞吐 smoke test，可以继续跑完收集日志。

## 2. 当前网络实际结构

### 2.1 数据与时序

- 每卡 batch 4，16 卡全局 batch 为 64。
- 每个样本使用 6 个相机、5 个时刻：当前帧及 `[-2, -4, -6, -8]`。
- 输入原始尺寸按 900×1600 配置，网络最终输入为 896×1600。
- IDA：resize 0.77～1.1、旋转 ±5.4°、随机水平翻转。
- BDA：旋转 ±22.5°、scale 0.95～1.05、x/y 各 0.5 概率翻转。
- LiDAR 点、GT 框和相机 BEV 几何都应用了同一个 BDA；这里没有发现 teacher/student 的 BDA 空间错位。
- LiDAR teacher 输入当前样本的 9 个 LiDAR sweep，最多保留 320,000 点。

`NuscDetDataset` 在构造时会把实验给出的历史索引变成 `[0, -2, -4, -6, -8]`，所以数据实际包含当前帧；不是只有 4 个历史帧。

### 2.2 相机学生

1. ConvNeXt-B 提取 stride 8/16/32 的 256/512/1024 通道图像特征。
2. SECONDFPN 融合为 384 通道，DepthNet 输出：
   - 112 个深度 bin：`[2.0, 58.0)`，间隔 0.5 m；
   - 每帧 150 通道的 BEV context。
3. 每个时刻形成 `[B, 150, 128, 128]` BEV，5 帧沿通道拼成 `[B, 750, 128, 128]`。
4. KDHead 的 BEV ResNet 以 750 通道输入，base channel 300；多尺度通道约为 750/300/600/1200，经 FPN 合并为 256 通道后进入 6 个 CenterPoint task head。

历史帧的图像到 BEV 前向位于 `torch.no_grad()` 中，只有当前帧图像分支直接接收 backbone 梯度；历史 BEV 仍参与后续时序融合。

### 2.3 LiDAR 教师与蒸馏

- CenterPoint voxel size 为 `[0.1, 0.1, 0.2]`，输出物理 BEV cell 也是 0.8 m。
- teacher backbone 的两层待蒸馏特征分别为：
  - `[B, 128, 128, 128]`
  - `[B, 256, 64, 64]`
- 学生 KDHead trunk 的前两层先各截一半通道，再经 adaptor 对齐到 128/256 通道。
- 总损失：

  ```text
  detection_loss
  + depth_loss
  + 0.6 * feature_distill_loss
  + 0.6 * response_loss
  ```

- 优化器基准 LR：`2e-4 / 64 × global_batch(64) = 2e-4`。
- 名字包含 `backbone` 的参数使用 `lr_mult=0.1`，实际为 `2e-5`。
- warmup 1,000 step，之后按 Trainer 估计的总 optimizer step 做 cosine；grad clip 为 5，FP16 mixed precision。

### 2.4 CBGS 与总 step

本地数据统计：train 28,130，val 6,019，合并后 34,149 个原始样本。CBGS 对每类生成约 16,074 个采样项，总长度 160,740。

全局 batch 64 时每 epoch 约 2,511 step，20 epoch 约 50,220 step，与已有日志最终的约 50,200 step 一致。当前 cosine 总步数估计逻辑是合理的，没有再出现“16 卡把总 step 放大 16 倍”的问题。

## 3. 框放缩链路：已确认的问题

当前实际路径为：

```text
LiDAR teacher boxes
  -> ProposalTargetLayer（按中心距离和 task group 匹配 GT）
  -> AdaptiveGTScalerV3（改 GT 中心/尺寸）
  -> QualityAwareMaskGeneratorV3（把框转成圆形 Gaussian mask）
  -> 两层 feature distillation
```

### 3.1 P0：中质量扩框混用了全局轴和框局部轴

GT 格式为 `[x, y, z, dx, dy, dz, yaw, vx, vy, class]`。其中 `dx/dy` 是随 yaw 旋转的框局部长宽。

高质量分支会把 ROI 中心偏移旋转到框局部坐标：

```python
dx_local = dx_global * cos(yaw) + dy_global * sin(yaw)
dy_local = -dx_global * sin(yaw) + dy_global * cos(yaw)
```

但中质量分支直接用全局 `dx/dy` 计算 `d_x/d_y`，随后却把它们分别加到局部 `dx/dy` 尺寸。这在 yaw 非 0 时方向错误。

最小例：GT yaw=90°，ROI 位于全局 `+y`，它其实是框局部 `+x`（长度）方向。当前 medium 实现却主要扩 `dy`（宽度），而不是 `dx`（长度）。

修复原则：medium 分支也应先把 `(roi_xy - gt_xy)` 用 GT yaw 转到 local frame，再分别计算 local-x/local-y 扩展。

### 3.2 P0：速度补偿也使用了错误坐标系

`vx/vy` 是 ego/global BEV 轴速度，当前 scaler 直接用 `vx` 补偿框长、`vy` 补偿框宽。但长宽是框局部轴，速度也必须先旋转：

```text
v_long =  vx * cos(yaw) + vy * sin(yaw)
v_lat  = -vx * sin(yaw) + vy * cos(yaw)
```

最小数值例中，yaw=90°、global `vx=2 m/s` 的物体，真实速度完全位于框局部横向；当前代码却把 20% 速度扩展加到框长上。

### 3.3 P0：放缩结果大部分没有改变最终 mask

`AdaptiveGTScalerV3` 修改的是有方向的 3D 框，但 `QualityAwareMaskGeneratorV3` 不画旋转矩形，只根据 `dx/dy` 计算一个标量 radius，再画圆形 Gaussian：

- yaw 完全被忽略；
- 长宽的定向信息被压成一个圆半径；
- radius 被 `int()` 向下取整；
- 最后再执行 `max(min_radius=2, int(radius))`。

按当前 0.8 m feature cell、`gaussian_overlap=0.1` 估算：

| 典型目标 | 原始 radius | 扩到 1.5 倍后的 radius | 最终整数 radius |
|---|---:|---:|---:|
| car 4.0×1.8 m | 1.414 | 2.121 | 都是 2 |
| barrier 2.5×0.5 m | 0.544 | 0.815 | 都是 2 |
| motorcycle 2.0×0.8 m | 0.661 | 0.992 | 都是 2 |
| pedestrian 0.8×0.8 m | 0.432 | 0.649 | 都是 2 |
| traffic cone 0.5×0.5 m | 0.270 | 0.405 | 都是 2 |

因此：

- medium 和 unmatched 分支不移动中心；对上述类别，扩框前后的 mask 完全相同；
- high 分支可能因中心平移而变化，但尺寸扩展通常仍不改变 radius；
- 当前“对小目标 boost”主要改变中心 Gaussian 的权重，不会让自适应扩框覆盖更大区域。

这比单纯的参数不合适更严重：当前 mask 表达本身无法保留论文式单侧/双侧定向扩展。

建议优先改成旋转矩形/椭圆 soft mask，或至少使用连续的 anisotropic Gaussian（分别保留 local length/width 和 yaw），不要先量化为一个整数圆半径。

### 3.4 P1：高质量分支正负方向不对称

当前符号逻辑为：

```python
1.0 if dx_local > 0 else (-1.0 if dx_local < -0.1 else 0.0)
```

所以 `+0.05 m` 会移动中心，`-0.05 m` 不移动中心；但两边都会增加尺寸。最小例的输出是：

```text
+0.05 m -> center shift +0.025 m, dx 4.05 m
-0.05 m -> center shift  0.000 m, dx 4.05 m
```

应对正负使用同一个 dead zone，例如 `abs(offset) <= 0.1 -> 0`，否则取 `sign(offset)`。

### 3.5 时序 feature KD 的 375 通道选择

第一层蒸馏保留 375 通道并不是 accidental slicing，而是明确的 channel-split 设计。当前实现从 750 通道时序特征中选择：

- current：150 通道；
- t-2：150 通道；
- t-6：前 75 通道；
- 合计：375 通道。

这样既保留 current 与近邻帧的完整信息，也让远时刻 t-6 以半通道参与蒸馏。检测 head 不跟随该选择重排，仍接收 current、t-2、t-4、t-6、t-8 的完整 750 通道。

共享模型默认仍保留 legacy_half 兼容模式；ConvNeXt-B J4 主实验显式启用 current_t2_half_t6，避免影响其他已有实验配置。

### 3.6 P1：配置名 `wh08`，实际是 0.7

目标文件和源 J4 文件都明确写的是：

```python
QualityAwareMaskGeneratorV3(w_l=0.5, w_h=0.7, ...)
```

文件名 `J4_wl05_wh08` 与实现不一致。源 J4 顶部注释也写着 `w_h=0.7`，说明更像是文件命名错误，而不是代码偶然写错。但如果实验表把它当作 0.8，则结论会被错误标注。

### 3.7 P2：小类别集合漏掉 bicycle

`SMALL_CLASSES = {5, 6, 8, 9}` 对应 barrier、motorcycle、pedestrian、traffic_cone，未包含 bicycle(7)。motorcycle 和 bicycle 属于同一 CenterPoint task，尺寸也都属于小目标；除非这是明确的消融设计，否则应补充或在文档中解释。

## 4. ROI 匹配与 proposal 的其他问题

### 4.1 proposal 阈值实际为 0.1，不是 LiDAR 配置里的 0.15

虽然实验里为 `lidar_conf.pts_bbox_head.test_cfg` 声明了 `score_threshold=0.15`，teacher 前向后调用的是：

```python
lidar_pred_box = self.get_bboxes(lidar_preds)
```

而 `self.get_bboxes` 委托给学生的 `KDHead`，其继承配置阈值为 0.1。因此当前 ROI 会包含 0.1～0.15 的低置信 teacher proposal。物理坐标编码仍然对齐，因为两边最终 cell 都是 0.8 m，但 proposal 分数分布和注释不一致。

### 4.2 匹配允许同 task 异类别

匹配条件不是 exact class，而是 same task group。例如 truck proposal 可匹配 construction_vehicle GT，pedestrian proposal可匹配 traffic_cone GT。这是代码注释明确声明的策略，不一定是实现 bug，但会把“教师分类错、定位近”的预测当作可靠 ROI。建议同时记录：

- exact-class high/medium 比例；
- same-task-but-wrong-class 比例；
- unmatched 比例。

否则无法判断 ROI mask 的“质量感知”是否真的对应教师质量。

### 4.3 先按 task 拼接，再截前 256 个 ROI

`KDHead.get_bboxes` 按 car → truck/CV → bus/trailer → barrier → motorcycle/bicycle → pedestrian/cone 拼接，没有做全局 score 排序；`prepare_batch_dict_from_pred_box` 直接取前 256 个。因此超过 256 时，后面的 pedestrian/cone 最先被截掉。

本地教师 val JSON 中，6,019 个样本只有 4 个超过 256，共截掉 7 个 pedestrian 和 27 个 cone，所以在该产物上的实际影响有限；但训练时阈值更低到 0.1 且有 BDA，仍建议改成全类别按 score 的 top-k。

### 4.4 一个 ROI 可同时匹配多个 GT，且候选按最高分而非最近距离选

当前匹配没有 one-to-one 约束，同一个 proposal 可被多个相邻 GT 复用。对一个 GT 的 high/medium 候选又选择最高置信度，而不是距离最近或综合质量最高的框。对密集 pedestrian/cone 场景，可能产生与真实偏移无关的扩框方向。

## 5. 蒸馏损失中的高风险行为

### 5.1 medium ROI 与速度补偿的坐标系问题（已修复）

原实现将全局 x/y 坐标中的 offset 和 velocity 直接用于 box-local 的长宽边界。当 yaw 不为 0 时，全局方向与框的局部前后/左右方向不一致，因此会把扩展量加到错误边界。

当前修复统一执行二维旋转，将全局向量转换到 box-local frame：

- local_x = cos(yaw) * global_x + sin(yaw) * global_y；
- local_y = -sin(yaw) * global_x + cos(yaw) * global_y。

medium ROI offset 以及 high、medium、unmatched 三个速度补偿分支均使用局部向量计算 left、right、back、front 四条边界。该修复仅影响后续新训练，epoch 19 等旧 checkpoint 仍对应原实现。

### 5.2 mask 权重的绝对尺度被归一化抵消

feature loss 为：

```python
(pixel_loss * mask).sum() / mask.sum()
```

所以如果整张有效 mask 都乘同一个常数，loss 完全不变。`w_l/w_h` 控制的是不同区域之间的相对采样分布，不控制这张图对总 loss 的绝对贡献。若论文或实验表把这些值解释为“蒸馏强度”，则实现与解释不一致。

### 5.3 response bbox KD 没有 teacher 质量门控

response regression 在所有 GT center 位置都把学生 raw box code 拟合 teacher raw code，权重来自 GT valid mask，而不是 teacher confidence/ROI quality。即使 teacher 在该 GT 处低置信或分类错误，bbox KD 仍生效。ROI quality mask 只作用于 feature KD，不作用于 response KD。

建议至少用 teacher heatmap/confidence 对 bbox response loss 门控，并分别记录 heatmap KD 与 bbox KD，当前两者合在一个 `response_loss` 中不利于诊断。

## 6. 数据、评估和复现问题

### 6.1 P1：train/val 泄漏

当前学生显式设置：

```python
self.use_train_val = kwargs.get('use_train_val', True)
```

教师训练脚本也默认 `use_train_val=True`。因此：

- 教师见过 val；
- 学生也见过 val；
- 本地教师 `mean_ap=0.8637, NDS=0.8279` 是训练集泄漏后的 val 指标；
- 当前 Trainer 又设置 `limit_val_batches=0`，训练期间没有真正验证曲线。

如果目的是最终 trainval→test submission，这可以是有意设置；如果目的是做消融、选超参或报告 val，必须改为 train-only，并重新训练 teacher 和 student。

### 6.2 hparams.yaml 不是实际有效配置

父类在 `super().__init__()` 内先 `save_hyperparameters()`，子类随后才把：

- `output_channels: 80 -> 150`
- BEV head 输入：80 -> 750
- base channel：160 -> 300
- code weights 等

改成当前实验值。因此已有 `version_13/hparams.yaml` 仍保存父类旧值。checkpoint 依赖同一份 Python 源码可正常恢复，但只看 YAML 无法重建真实模型。

建议在所有子类修改完成、`self.model` 构建前后，保存一份 deep-copied resolved config，并在启动时打印关键 shape/LR/data split。

### 6.3 教师评估 metadata 写错模态

`DetNuscEvaluator` 默认 metadata 是 `use_lidar=False, use_camera=True`。教师是纯 LiDAR 模型，但产出的 JSON 仍声明 camera-only。它通常不改变 nuScenes 数值计算，但会使提交元数据和实验档案不真实。

### 6.4 绝对 checkpoint 路径

teacher checkpoint 被硬编码为工作区绝对路径。换机器、复制目录或整理产物后容易静默指向旧教师。建议做成 CLI 参数，并在启动日志中打印 checkpoint 哈希、训练 split 和 teacher metrics 的有效性声明。

## 7. 当前训练中的冗余与性能问题

### 7.1 label-input 被完整生成但完全没被当前模型消费

`KDHead.get_targets` 每步除 detection targets 外，还用 `gen_labelinput` 构造 `bev_mask/bev_box/bev_label`。训练脚本把它们传给 `LabelDistill.forward`，但当前 `models/lidardistill.py` 的 forward 参数虽然接收它们，函数体从未使用。

这会造成：

- 每个 GT 调用 skimage polygon；
- 多次 GPU tensor → CPU NumPy 同步；
- 额外的 128×128×(mask/9 box/10 class) 张量构建；
- 16 卡下每个 rank 都重复做。

应把 detection target 生成与 label-input rasterization 拆开，本配置只生成 `(heatmaps, anno_boxes, inds, masks)`。

### 7.2 学生框每步解码后未使用

`models/lidardistill.py` 每步执行 `image_pred_box = self.get_bboxes(preds)`，包含 sigmoid、decode 和逐 task CPU circle NMS；训练脚本收到后没有任何使用。应删除或仅在诊断开关下执行。

### 7.3 ROI/scaler 的逐框 CPU 同步

`ProposalTargetLayer` 用 Python list comprehension 判断 task group，scaler 对每个框反复 `.item()` 并调用 NumPy，mask generator 也逐框 `.item()`。这些操作会频繁同步 GPU。建议全部改成 tensorized class-group lookup、旋转矩阵、速度变换和批量 mask rasterization。

### 7.4 每步重复 `.cuda()` model

Lightning 已负责设备迁移，`training_step` 内的 `self.model = self.model.cuda()` 是冗余的。通常不会改变结果，但会混淆设备职责，不利于后续 FSDP/多机策略兼容。

## 8. 当前路径之外的明确框栅格 bug

`labeldistill/utils/bev_mask.py` 中：

```python
mask_w = width * 0.5
mask_l = length * 0.5
corners = [[+mask_l / 2, +mask_w / 2], ...]
```

`width/length` 已经是 feature-map 上的完整边长；先乘 0.5，角点又除 2，最终完整边长只剩输入的 0.5 倍，面积约为 0.25 倍。

当前 LiDAR feature distill 模型不读取这组 label-input，所以它不是本轮结果异常的直接原因；但 `labeldistill.py`、`labelencoder.py` 等真正使用 label-input 的配置会受影响。修复时角点应该直接使用 `±length/2, ±width/2`，并另行用可视化单测确认 yaw 和 x/y 轴约定。

## 9. 已有训练产物观察

本地 `outputs/LidarDistill_convnextb_900x1600_J4_wl05_wh08_cbgs` 中有多轮历史记录，不等于当前 DLC 实时状态：

- 早期 version 2/3/6/7 出现过 detection/response/distill NaN；
- 后来的 version 11～13 已稳定到约 50,200 step；
- version 11 从 step 49 的 total loss 约 16,588 降到 step 41,849 的约 8.30；
- version 13 最后一个记录约为：
  - total 8.845
  - detection 4.986
  - response 1.135（已乘 0.6）
  - feature distill 0.546（已乘 0.6）
  - depth 2.177

这些日志说明当前 FP32 voxel pooling、FP32 高风险 loss 和 loss finite 检查已解决旧版的主要数值稳定性问题。但由于没有合法 val、且框/蒸馏设计问题仍在，loss 正常不等于方法实现正确。

## 10. 建议修复顺序

### 第一阶段：决定当前任务是否保留

1. 明确本轮期望的是 `w_h=0.7` 还是 0.8。
2. 明确用途是 trainval→test 最终训练，还是 train→val 消融。
3. 若要验证自适应扩框或 `wh08` 消融，当前结果不应进入正式对比表。

### 第二阶段：先修正确性

1. [已完成] 将 offset 和 velocity 全部转换到 box-local frame。
2. 用统一 dead zone 修复正负不对称。
3. 按四条局部边界分别处理 ROI 定向和速度方向扩展；V3 已撤销 base floor。
4. 用 oriented rectangle / anisotropic Gaussian 生成连续 soft mask，移除过早的整数 radius 量化。
5. 将 teacher proposal threshold 的唯一来源显式化。
6. `wh08` 文件若保留名称，则实际设置为 0.8；否则重命名并同步实验表。

### 第三阶段：修实验有效性

1. 消融时 teacher/student 都改为 train-only，val-only 做选择。
2. 加回每 epoch 或固定 step 的 val，并保存 mAP/NDS 最优 checkpoint。
3. 保存 resolved config、数据 split、teacher checkpoint hash。
4. 修正 evaluator modality。

### 第四阶段：减少浪费并补诊断

1. 当前模型跳过 label-input rasterization。
2. 删除未使用的 student box decode。
3. tensorize ROI matching/scaling/mask。
4. 日志新增各类别 high/medium/unmatched 数、exact-class mismatch、mask 有效像素/权重和各 response 子损失。

## 11. 最小回归测试清单

- yaw=0°/90°/任意角，local-x ROI 偏移只能扩 local length。
- yaw=90°、global-x 速度应映射到 local lateral，而不是 local longitudinal。
- `+0.05/-0.05` 在同一 dead zone 下行为镜像一致。
- high 单侧扩展后，未扩的一侧边界保持不变。
- medium 双侧扩展后，中心保持不变。
- 不同 yaw 的 oriented mask 面积一致、方向正确。
- car/pedestrian/cone 扩 10% 后，soft mask 确实连续变化，而不是因 radius=2 完全不变。
- ROI 超过 256 时按全局 score top-k，不按 task 顺序丢小类。
- `hparams/resolved_config` 中的 150/750/300 通道与真实 module shape 一致。
- train-only 配置绝不加载 val pkl；最终 trainval 配置在实验名和报告中明确标注。

## 12. 关键代码位置

- 实验入口：`labeldistill/exps/nuscenes/labeldistill/big_backbone/LidarDistill_convnextb_900x1600_J4_wl05_wh08_cbgs.py`
- ConvNeXt-B 基础配置：`labeldistill/exps/nuscenes/base_exp_convnextb.py`
- 数据/BDA/LiDAR：`labeldistill/datasets/nusc_det_dataset_lidar.py`
- teacher/student forward：`labeldistill/models/lidardistill.py`
- detection/response targets 与 loss：`labeldistill/layers/heads/kd_head.py`
- ROI 匹配：`labeldistill/refine_head/target_assigner/roi_distill.py`
- 自适应扩框：`labeldistill/refine_head/target_assigner/adaptive_gt_scaler_v3.py`
- quality mask：`labeldistill/refine_head/target_assigner/quality_aware_mask_v3.py`
- label-input 栅格：`labeldistill/utils/bev_mask.py`
- CLI/DDP/验证开关：`labeldistill/exps/base_cli.py`
- 教师训练：`labeldistill/exps/nuscenes/labeldistill/train_teacher_centerpoint.py`

