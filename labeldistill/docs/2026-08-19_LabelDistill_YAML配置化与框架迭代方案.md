# LabelDistill YAML 配置化与框架迭代方案

> 日期：2026-08-19  
> 当前迭代基线：`labeldistill/exps/nuscenes/ablation_param/param_J4_wl05_wh08.py`  
> 说明：当前文件名包含 `wh08`，但实际配置是 `w_h=0.7`。后续 YAML 应按真实参数命名为 `j4_wl05_wh07`，旧名称只作为兼容别名保留。

> 实施状态（2026-08-19）：配置基础设施和 J4 builder 首版已经落地，包括 typed schema、`_base_` 继承、dot-list override、跨字段校验、派生参数、config diff、resolved config、环境/源码 SHA256 审计、versioned Python preset、通用 J4 experiment 以及 `tools/train.py`。`j4_wl05_wh07` 与 `j4_wl05_wh08` 已有 YAML。当前测试为 74 passed（因整套 pytest 在 DSW 上偶发长时间无输出，按 53 + 21 两组执行验证）。
>
> 当前边界：builder 已能从 YAML 构造完整 143,582,490 参数 J4 模型，并与 legacy J4 的完整模型配置对象一致；已完成 `1 GPU × batch 1 × 1 step`、双 PPU Gloo DDP 一步训练，以及从 `global_step=1` 恢复到 `global_step=2` 的 resume 验收。仍需固化同一真实 batch 的 legacy/YAML 中间输出与 loss 数值快照；在这项等价性验收完成前不得批量删除 legacy Python entry。

## 1. 目标

将当前“每个实验复制一份完整 Python 训练脚本”的管理方式，逐步迁移为：

```text
Python 负责算法实现和组件构造
YAML 负责实验参数、组件选择和运行设置
命令行负责少量临时覆盖
resolved_config.yaml 负责复现和审计
```

这次迁移主要解决以下问题：

1. 一个公共 bug 需要在多个 300～400 行实验脚本中重复修复。
2. 文件名、注释和实际参数可能不一致。
3. 无法快速确认两个消融实验究竟改变了哪些变量。
4. 子类修改父类配置后，保存的 `hparams.yaml` 可能不是实际生效配置。
5. 数据、模型、teacher、matching、mask、loss 和 runtime 参数散落在不同文件中。
6. checkpoint 缺少完整、不可变的最终配置记录。

## 2. 当前已完成的框架基础

J4 baseline 已完成以下接口重构，可作为 YAML 化的稳定起点：

- 配置进入模型前执行深拷贝，避免实验实例之间相互污染。
- J4 不再先构造父类 `BaseBEVDepth` 再覆盖，只构造一次最终 `LabelDistill`。
- teacher proposal decoder 已从学生 `KDHead` 解码路径中独立出来。
- teacher proposal 参数已显式固定：阈值 `0.1`、circle NMS 和对应半径。
- 训练输出改为 `StudentOutput`、`TeacherOutput` 和 `LabelDistillOutput`。
- teacher proposals 使用 ragged `ProposalBatch`，不再 padding/truncate 到 256。
- GT 不再 padding/truncate 到 128 后再恢复。
- matcher 输出按原始 GT 顺序对齐的 `MatchResult`。
- 学生 decode 能力继续保留用于推理；J4 训练阶段不执行未使用的学生 NMS。

因此，YAML 化不需要同时重写核心算法，只需要把当前有效配置提取出来并通过统一 builder 构造相同对象。

## 3. 设计原则

### 3.1 YAML 只保存纯数据

YAML 中只允许字符串、数字、布尔值、列表和字典，不写：

- `!!python/object`
- lambda
- 可执行 Python 表达式
- 隐式 import
- 设备相关 Tensor

组件类型使用稳定字符串，由 Python registry/builder 映射：

```yaml
region:
  mask:
    type: quality_aware_v3
```

而不是在 YAML 中直接实例化 Python 类。

### 3.2 单一事实来源

以下信息只能在一个公共配置节点中定义：

- point cloud range
- voxel size
- BEV output stride
- 类别顺序
- CenterPoint task groups
- feature map size 的推导规则
- teacher/student 的物理 BEV 范围

decoder、matcher、mask 和 loss 都从同一个 resolved config 读取，禁止分别硬编码。

### 3.3 实验 YAML 只描述变量差异

例如研究 `w_h` 时，实验文件只覆盖：

```yaml
experiment:
  name: j4_wl05_wh08

region:
  mask:
    w_high: 0.8
```

数据 split、teacher checkpoint、LR 等其余内容继承 baseline，不复制整份训练脚本。

### 3.4 配置必须经过校验

启动训练前至少验证：

- `0 < w_low <= w_high < 1`
- teacher task 数量等于 NMS `min_radius` 数量
- class/task 配置覆盖且只覆盖全部 10 类一次
- GT/proposal box code size 等于 9
- teacher/student feature KD 层数和通道对应
- teacher/student BEV 物理坐标范围一致
- teacher checkpoint 存在
- batch size、GPU 数量和 LR 缩放规则明确
- train-only 与 train+val 的目的明确
- experiment name 与关键参数一致

校验失败时直接终止，不使用静默默认值继续训练。

## 4. 推荐目录

```text
ROI_LABEL_DISTILL/
  configs/
    base/
      nuscenes.yaml
      student_r50_128x128.yaml
      teacher_centerpoint_vox01.yaml
      distillation_roi_v3.yaml

    experiments/
      j4_wl05_wh07.yaml
      j4_wl05_wh08.yaml
      ablation_exact_class.yaml
      ablation_no_scaler.yaml

  labeldistill/
    config/
      loader.py
      schema.py
      validation.py

    builders/
      experiment_builder.py
      student_builder.py
      teacher_builder.py
      distillation_builder.py

  tools/
    train.py
    evaluate.py
```

旧的 `labeldistill/exps/nuscenes/.../*.py` 在迁移完成前保留为 legacy entry，不立即删除。

## 5. 推荐技术方案

当前环境已经安装：

- PyYAML 6.0.2
- OmegaConf 2.3.0
- Hydra 1.3.2

第一阶段建议使用 **OmegaConf + argparse**，暂不直接接管为完整 Hydra 应用，原因是：

- 可以支持 YAML merge、dot-list 命令行覆盖和 resolved YAML 输出。
- 对现有 `run_cli` 和 PyTorch Lightning 启动方式改动较小。
- 不引入 Hydra working directory、launcher 和 sweep 行为带来的额外迁移复杂度。
- 以后需要批量 sweep 时仍可在相同 YAML schema 上接入 Hydra。

建议配置加载流程：

```text
读取 base YAML
  -> 递归读取 experiment._base_
  -> OmegaConf.merge(base, experiment)
  -> OmegaConf.merge(command_line_overrides)
  -> resolve interpolation
  -> schema/type/range validation
  -> 保存 resolved_config.yaml
  -> builder 构造训练对象
```

## 6. J4 baseline YAML 草案

```yaml
schema_version: 1

experiment:
  name: j4_wl05_wh07
  seed: 0
  description: >-
    R50 128x128 LiDAR distillation baseline with structured proposals and
    quality-aware ROI mask.

runtime:
  gpus: 2
  num_nodes: 1
  distributed_backend: gloo  # 当前 DSW PPU；NVIDIA 环境可 override 为 nccl
  batch_size_per_device: 16
  max_epochs: 24
  precision: 16
  gradient_clip_val: 35.0
  output_dir: outputs/j4_wl05_wh07
  resume_from: null

data:
  dataset: nuscenes
  root: data/nuScenes
  train_info: nuscenes_infos_train.pkl
  val_info: nuscenes_infos_val.pkl
  use_train_val: false
  num_workers: 4
  use_cbgs: false
  key_idxes: [-2, -4, -6, -8]
  return_depth: true
  return_lidar: true

geometry:
  point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
  lidar_voxel_size: [0.1, 0.1, 0.2]
  lidar_out_size_factor: 8
  student_bev_resolution: 0.8
  feature_map_size: [128, 128]

classes:
  names:
    - car
    - truck
    - construction_vehicle
    - bus
    - trailer
    - barrier
    - motorcycle
    - bicycle
    - pedestrian
    - traffic_cone

  tasks:
    - [car]
    - [truck, construction_vehicle]
    - [bus, trailer]
    - [barrier]
    - [motorcycle, bicycle]
    - [pedestrian, traffic_cone]

student:
  type: camera_bevdepth_r50
  output_channels: 150
  key_frame_count: 5
  temporal_kd_selection: legacy_half
  structured_output: true

teacher:
  type: frozen_centerpoint
  checkpoint: ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth
  checkpoint_prefix: model.centerpoint.

  proposal:
    score_threshold: 0.1
    max_num_before_nms: 500
    nms_type: circle
    min_radius: [4, 12, 10, 1, 0.85, 0.175]
    post_max_size: 83
    norm_bbox: true

matching:
  type: center_distance
  class_policy: same_task_group
  structured: true
  selection: highest_score
  one_to_one: false

  distance_thresholds:
    car:                  {high: 2.0, medium: 4.0}
    truck:                {high: 2.5, medium: 4.0}
    construction_vehicle: {high: 2.5, medium: 4.0}
    bus:                  {high: 3.5, medium: 4.0}
    trailer:              {high: 2.5, medium: 4.0}
    barrier:              {high: 0.5, medium: 2.5}
    motorcycle:           {high: 1.0, medium: 2.5}
    bicycle:              {high: 1.0, medium: 2.5}
    pedestrian:           {high: 0.5, medium: 2.0}
    traffic_cone:         {high: 0.5, medium: 2.0}

region:
  scaler:
    type: adaptive_gt_scaler_v3
    enabled: true
    mu: 0.15
    velocity_scale: 0.2
    max_distance: 50.0

  mask:
    type: quality_aware_mask_v3
    w_low: 0.5
    w_high: 0.7
    max_distance: 50.0
    boost_small_medium: true
    gaussian_overlap: 0.1
    min_radius: 2

loss:
  detection_weight: 1.0
  depth_weight: 1.0
  feature_weight: 0.6
  response_weight: 1.0

optimizer:
  type: AdamW
  base_lr_at_global_batch_64: 0.0004
  weight_decay: 0.01
  backbone_lr_mult: 0.5

scheduler:
  type: MultiStepLR
  milestones: [19, 23]
```

注意：以上 YAML 必须以当前 J4 源码为准逐字段迁移。`temporal_kd_selection`、LR 和数据 split 不应根据文件名或历史文档猜测。

## 7. 消融配置方式

### 7.1 修改单个参数

```yaml
_base_: j4_wl05_wh07.yaml

experiment:
  name: j4_wl05_wh08

region:
  mask:
    w_high: 0.8
```

### 7.2 修改 matching 策略

```yaml
_base_: j4_wl05_wh07.yaml

experiment:
  name: j4_exact_class

matching:
  class_policy: exact_class
```

### 7.3 关闭 scaler

```yaml
_base_: j4_wl05_wh07.yaml

experiment:
  name: j4_no_scaler

region:
  scaler:
    enabled: false
```

每次启动时输出相对于 base 的配置差异，例如：

```text
Config diff from j4_wl05_wh07:
  region.mask.w_high: 0.7 -> 0.8
```

如果差异中意外出现 teacher checkpoint、data split 或 LR，应在正式训练前人工确认。

## 8. 统一启动接口

目标启动方式：

```bash
cd /mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL

python tools/train.py \
  --config configs/experiments/j4_wl05_wh07.yaml
```

临时覆盖 runtime 参数：

```bash
python tools/train.py \
  --config configs/experiments/j4_wl05_wh07.yaml \
  runtime.gpus=16 \
  runtime.batch_size_per_device=4 \
  runtime.output_dir=outputs/j4_wl05_wh07_16gpu
```

继续训练：

```bash
python tools/train.py \
  --config outputs/j4_wl05_wh07/resolved_config.yaml \
  runtime.resume_from=outputs/j4_wl05_wh07/checkpoints/last.ckpt
```

恢复训练时优先使用输出目录中的 `resolved_config.yaml`，而不是可能已经被修改的源实验 YAML。

## 9. 输出目录和实验审计

每个训练输出目录至少保存：

```text
outputs/j4_wl05_wh07/
  source_config.yaml
  resolved_config.yaml
  config_diff.yaml
  environment.txt
  checkpoints/
  lightning_logs/
```

`resolved_config.yaml` 还应包含或关联以下派生信息：

- global batch size
- effective learning rate
- student 实际 BEV 输入通道
- teacher/student BEV cell 物理尺寸
- teacher checkpoint 路径和 SHA256
- 当前 git commit；若目录不是 git 仓库，记录关键源文件 SHA256
- Python、PyTorch、CUDA、MMEngine、MMDetection3D 版本
- train/val/trainval 数据 split

checkpoint 内部保存 `schema_version` 和 resolved config。加载时如果 schema 不兼容，应明确报错或执行显式迁移，不静默采用新默认值。

## 10. Python builder 的职责

推荐统一入口：

```python
config = load_and_resolve_config(path, overrides)
validate_config(config)
experiment = build_experiment(config)
trainer = build_trainer(config.runtime)
trainer.fit(experiment, ckpt_path=config.runtime.resume_from)
```

builder 负责：

- 根据 `student.type` 构造学生网络。
- 根据 `teacher.type` 构造并冻结 teacher。
- 根据 `teacher.proposal` 构造独立 decoder。
- 根据 `matching` 构造 matcher。
- 根据 `region.scaler/mask` 构造 region policy。
- 根据 `loss` 构造 loss 权重。
- 根据 `optimizer/scheduler` 构造优化器。

实验 YAML 不再继承一个包含完整 `training_step` 的 Python 子类。

## 11. 迁移步骤

### 阶段 A：冻结当前行为

当前状态：**部分完成**。已有接口回归、legacy J4 ↔ YAML 完整配置对象对齐，以及真实 batch 的新入口训练 smoke test；legacy/new 同 batch 的中间输出与 loss 快照仍未固化。

1. 保留当前 J4 Python entry。
2. 固定一个小型真实 batch 或记录一个 batch 的输入索引。
3. 保存当前输出：teacher proposals、MatchResult、mask 和各 loss 分量。
4. 保留现有单元测试作为接口回归。

### 阶段 B：实现配置基础设施

当前状态：**首版已完成**。

1. 新增 YAML loader。
2. 新增 `_base_` 合并与 dot-list override。
3. 新增 schema 和 validation。
4. 新增 resolved config 保存与 config diff。
5. 暂不改变模型和 loss 实现。

当前实现还额外记录：global batch size、effective learning rate、feature map size、BEV cell size、student BEV 输入通道、teacher checkpoint SHA256、配置继承链和关键源码 SHA256。

### 阶段 C：迁移 J4

当前状态：**builder 接管首版已完成，数值等价性验收进行中**。J4 YAML 已能通过 `tools/train.py` 构造通用 `J4Experiment`，不再依赖 legacy J4 class；preset 与 legacy 的 backbone/head/teacher/proposal 配置已做完整对象级 parity。单卡、双卡 DDP、checkpoint 保存和 resume 已通过，尚缺 legacy/new 同 batch 数值快照。

1. 将 J4 的全部有效参数逐项迁移到 YAML。
2. 通过 builder 构造与旧 J4 相同的模型和训练组件。
3. 比较参数数量、模型 state_dict key、proposal、mask 和 loss。
4. 通过单卡短程 smoke test。
5. 通过多卡 DDP smoke test。

### 阶段 D：迁移消融实验

1. J1/J2/J3/J5 只保留 YAML 参数差异。
2. P3 lambda 系列只覆盖 loss weight。
3. mu 系列只覆盖 scaler 参数。
4. 自动生成 base-to-experiment config diff。

不能在未审计源码差异时直接假设这些实验只改变一个参数。当前 J1/J2/J3/J5、P3 和 mu 脚本与已经重构的 J4 各有约 179～181 行差异，其中既可能包含旧实现，也可能包含真实实验语义。迁移时必须逐项分类为：

- 应统一到新 J4 的历史实现差异；
- 需要进入 schema/registry 的有效实验开关；
- 已确认不再需要复现的废弃行为。

### 阶段 E：清理 legacy

满足以下条件后再归档旧 Python 实验脚本：

- YAML J4 能从头训练。
- 能恢复旧/新 checkpoint。
- 单 batch 数值对比符合预期。
- 至少完成一次短程多卡训练。
- 实验记录已经改用 resolved config 标识。
- 新测试和审计逻辑已经不再 import 或硬编码该 legacy entry。

legacy 清理按实验族分别验收，不要求等待所有实验一次性迁移完毕。详细清理顺序和保护范围见第 14 节。

## 12. 验收标准

YAML 化迁移不能只以“程序能启动”为标准。至少需要验证：

### 12.1 配置一致性

- J4 实际 student/teacher 网络结构一致。
- state_dict key 和参数 shape 一致。
- teacher checkpoint 加载 key 数一致。
- optimizer param groups、LR multiplier 和 scheduler 一致。

### 12.2 单 batch 数值对比

相同随机种子、相同 batch、eval/no-augmentation 条件下比较：

- student raw predictions
- teacher raw predictions
- teacher ProposalBatch
- MatchResult quality/score/label/distance
- scaled GT
- BEV mask
- detection loss
- depth loss
- feature loss
- response loss
- total loss

框架迁移阶段原则上应完全一致；若因移除旧的 256/128 截断产生差异，应只出现在超过旧上限的样本，并单独记录。

### 12.3 运行验证

2026-08-19 实测：当前 DSW 的两张 `PPU-ZW810E` 通过 Gloo 完成单步 DDP；该环境无 `libcuda.so`，因此 NCCL 不适用。单卡训练、双卡 checkpoint 保存以及恢复后 `global_step: 1 → 2` 均已通过。

- `1 GPU × batch 1` 前向/反向 smoke test。
- `2 GPU` DDP smoke test。
- 目标卡数的启动与 checkpoint 保存验证。
- resume 后 global step、optimizer 和 scheduler 连续。

## 13. 不在第一阶段同时修改的内容

为避免 YAML 迁移和算法变化混在一起，第一阶段暂不同时修改：

- same-task-group 与 exact-class 的策略
- highest-score 与 nearest-distance 的选择
- one-to-one matching
- CPU circle NMS
- mask 的逐框 `.item()`
- 圆形 Gaussian 与旋转/各向异性 mask
- response bbox KD 的质量门控
- high/medium/unmatched 统计实现

这些应在 YAML baseline 完成数值对齐后，作为独立实验开关逐项迭代。

## 14. Legacy 清理与归档计划

### 14.1 清理原则

YAML 化后的目标确实是删除大量重复 Python entry，但删除动作必须与迁移完成度绑定：

```text
存在 YAML
≠ builder 已使用该 YAML
≠ 数值行为已经对齐
≠ legacy entry 可以删除
```

清理时遵循以下原则：

1. 生成物、缓存和错误 ABI 文件可以独立清理。
2. 历史源码先归档并记录 SHA256，再从运行目录移除。
3. 每个实验族分别完成 YAML、builder、checkpoint 和数值验收后再删除对应 Python entry。
4. `data/`、`ckpts/`、`outputs/` 和实验文档属于用户资产，不进入自动清理范围。
5. 当前目录不是 Git 仓库，不能依赖 Git 找回误删内容；在建立版本历史或外部归档前，对唯一源码采取保守策略。

### 14.2 当前仓库盘点

截至 2026-08-19：

- `labeldistill/exps/` 下有 44 个有效 Python 文件，共约 15094 行。
- `ablation_param/` 有 15 个参数实验脚本，每个约 367 行。
- `find_best_v3/` 有 6 个实验脚本；后 5 个相对 baseline 只变化 6～8 行，是最适合优先 YAML 化的一组。
- 有 63 个 `__pycache__` 目录、267 个 `.pyc` 文件。
- `build/` 约 48 MB。
- Python 3.8/3.9 的旧 CUDA 扩展约 18.09 MB；当前运行环境是 Python 3.10，实际加载 `cpython-310` 扩展。
- `.ipynb_checkpoints/` 约 0.58 MB，其中存在数份没有正式源码副本的历史文件。

### 14.3 第一批：可立即清理的生成物

以下内容不承载实验语义，可在不修改训练代码的情况下清理：

- `build/` 中的编译中间产物和重复 build 输出。
- 所有 `__pycache__/` 和 `.pyc`。
- `.pytest_cache/`。
- `LabelDistill.egg-info/`。
- `labeldistill/exps/base_cli.py.orig`；它只是当前 `base_cli.py` 一行修复前的副本。
- Python 3.8/3.9 的 `voxel_pooling_*_ext`，前提是明确不再使用 Python 3.8/3.9 环境。

必须保留当前 Python 3.10 实际加载的两个扩展：

```text
labeldistill/ops/voxel_pooling_train/
  voxel_pooling_train_ext.cpython-310-x86_64-linux-gnu.so

labeldistill/ops/voxel_pooling_inference/
  voxel_pooling_inference_ext.cpython-310-x86_64-linux-gnu.so
```

第一批清理预计释放约 68 MB。清理后应立即运行完整测试和两个扩展的 import smoke test。

### 14.4 第二批：先归档再删除的历史文件

以下文件没有当前代码引用，但在没有 Git 历史的情况下不直接删除：

- `labeldistill/models/备份lidardistill.py`
- `labeldistill/models/澶囦唤lidardistill.py`
- `草稿纸.py`
- `.ipynb_checkpoints/` 中与正式源码不同或没有正式副本的 Python 文件

其中两个 `lidardistill` 备份文件内容完全相同，只需保留一个归档副本。

当前发现的非空、且没有正式源码副本的 notebook checkpoint 包括：

```text
LidarDistill_r101_128x128_e24_4keyfp-checkpoint.py
LidarDistill_r50_128x128_e24_4keyfp-checkpoint.py
LidarDistill_swinb_640x1600_e24_4keyfp-checkpoint.py
Lidar_gt_label_r50_128x128_e_24-checkpoint.py
fp_distill-checkpoint.py
```

建议将需要保留的历史源码移到不参与 Python import 的显式归档目录，并生成包含原路径、归档路径、文件大小和 SHA256 的 manifest。归档完成后再统一清理 `.ipynb_checkpoints/`。

### 14.5 第三批：按实验族迁移并删除 legacy entry

推荐顺序如下。

#### 14.5.1 `find_best_v3`

优先迁移。该组脚本结构高度重复，后 5 个实验相对 baseline 只变化 6～8 行。应建立一个 baseline YAML 和若干差异 YAML，通过 builder 验收后删除：

```text
labeldistill/exps/nuscenes/find_best_v3/
```

#### 14.5.2 `ablation_param`

迁移 J1～J5、P3 lambda 和 mu 系列。由于这些脚本尚未同步 J4 的近期框架重构，必须先完成第 11 节阶段 D 的差异分类，不能只根据文件名提取几个数值后直接删除。

最终应由一个公共实现加 YAML override 替代当前 15 份完整脚本：

```text
labeldistill/exps/nuscenes/ablation_param/
```

旧 `param_J4_wl05_wh08.py` 在 builder 接管 J4、对齐测试不再 import 它、审计逻辑不再硬编码它之后才能归档。

#### 14.5.3 `ablation_module`

A0～A5 包含不同模型与 region policy 行为。应先把差异表达为明确的组件类型或开关，再删除：

```text
labeldistill/exps/nuscenes/ablation_module/
```

#### 14.5.4 其他实验和 backbone entry

依次迁移：

- `labeldistill/exps/nuscenes/labeldistill/`
- `labeldistill/exps/nuscenes/bevdepth/`
- `base_exp_convnextb.py`、`base_exp_res101.py`、`base_exp_swinb.py`
- teacher training entry

这些入口涉及 student backbone、teacher training、普通检测和蒸馏等不同用途，不作为一批无差别删除。

#### 14.5.5 旧 shell 启动器

YAML train/evaluate/sweep 工具可用后再删除：

```text
run_ablation_module.sh
run_ablation_param.sh
train_ablation_module.sh
train_ablation_param.sh
```

### 14.6 单个实验族的删除门槛

每组 legacy entry 只有同时满足以下条件才可从运行目录删除：

- 对应 baseline 和 override YAML 已存在且通过 schema/validation。
- builder 能从 YAML 构造完整实验，不依赖被删除的 Python class。
- base-to-experiment config diff 已人工检查。
- 模型参数数量、state_dict key/shape、optimizer 和 scheduler 已对齐。
- 旧 checkpoint 可以通过新入口恢复或评测。
- 单 batch 关键输出与 loss 对齐。
- 至少通过单卡 smoke test；训练实验还需通过 DDP smoke test。
- train、evaluate、resume 和审计命令均已有替代入口。
- 测试、文档、shell 脚本和审计源码列表中不再引用 legacy 路径。

### 14.7 明确保留的范围

下列内容不因 YAML 化而删除：

- `labeldistill/models/`、`layers/`、`datasets/`、`refine_head/` 中被 registry/builder 使用的算法实现。
- `labeldistill/config/`、未来的 `builders/` 和统一 `tools/` 入口。
- 当前 Python ABI 所需的 CUDA 扩展。
- `tests/` 和数值回归基准。
- `data/`、`ckpts/`、`outputs/`、`docs/`。

历史 outputs 是否归档或删除是独立的数据保留决策，不与源码清理绑定，也不执行自动清理。

### 14.8 清理执行流程

每批清理按以下顺序执行：

1. 生成候选文件清单、引用扫描结果和 SHA256 manifest。
2. 将唯一历史源码移入归档，确认归档可读。
3. 只删除当前批次明确批准的目标。
4. 运行配置解析、扩展 import、完整单元测试和相应 smoke test。
5. 记录实际删除内容、释放空间、验证结果和恢复方式。

## 15. 最终判断

YAML 管理适合当前项目，但关键不是简单地把 Python 字典复制到 YAML，而是建立：

```text
稳定 schema
+ 配置继承与命令行覆盖
+ 类型/范围/跨字段校验
+ 通用 builder
+ resolved config 固化
+ config diff
+ 单 batch 数值回归
```

配置基础设施和 J4 builder 首版已经完成，GPU 前反向、双卡 DDP、checkpoint 与 resume 验收也已通过。下一步应集中补齐阶段 A 的 legacy/new 同一真实 batch 数值快照和评测入口。与此同时可以独立执行第 14.3 节的生成物清理，但 legacy 实验脚本必须等对应 YAML 实验完成数值等价性验收后，再按实验族分批归档。
