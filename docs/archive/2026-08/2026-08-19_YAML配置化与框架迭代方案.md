# LabelDistill YAML 配置化与框架迭代方案（历史文档）

> **已于 2026-08-23 被 center-value baseline 取代。** 本文保留的是旧 J4
> 配置迁移过程与历史验收记录，不再代表当前训练入口。旧网络派生的
> `ablation_module`、`ablation_param` YAML、批量脚本、迁移测试和归档源码
> 已移除；当前入口见 `labeldistill/docs/训练命令.md` 和
> `configs/experiments/center_value_baseline_cp50200.yaml`。本文后续出现的旧命令
> 仅用于解释历史决策，不应直接执行。

> 日期：2026-08-19；最近更新：2026-08-20
> 当前迭代基线：`configs/experiments/j4_wl05_wh07.yaml`
> 说明：历史文件 `param_J4_wl05_wh08.py` 的实际配置是 `w_h=0.7`，现已归档；错误旧名称仅作为配置兼容别名保留。
> 实施分支：`codex/yaml-builder`；迁移前基线：`74cfe3f`；首版实现：`c3548b4`。

> 实施状态（2026-08-20）：配置基础设施、J4 builder、`tools/train.py`、`tools/evaluate.py`、checkpoint-only resume/evaluate 和数值快照工具已经落地。`ablation_param` 的 15 个参数实验和 `ablation_module` 的 6 个模块实验均已有 YAML；21 个 legacy 运行脚本已从 Python import 路径移除并归档到 `archive/legacy_experiments/2026-08-20/`，四个批量 shell 启动器也已改用统一 YAML 入口。
> 顶层项目回归集 `pytest -q tests` 为 `127 passed`。不带目录限制的全仓 pytest 还会收集 vendored nuScenes tracking tests；当前环境缺少其可选依赖 `motmetrics`，不属于本次 LabelDistill 回归口径。
>
> 当前边界：J4 legacy/YAML 已在同一真实 batch、共享 state_dict 下完成 110 个 tensor 的快照门禁，覆盖 raw predictions、proposals、MatchResult、scaled GT、mask、四项 loss 和 total loss；A0～A5 已建立新框架快照。A3 双 PPU Gloo DDP、checkpoint-only resume（`global_step=1 → 2`）、A5/P3 单卡一步和 A5 checkpoint-only evaluate prediction smoke 均通过。R50/J4 两个消融族的 legacy 删除硬门槛已满足并完成归档；`find_best_v3` 已按项目决策直接删除、不再迁移。ConvNeXt-B 896×1600 已进入 schema/preset/builder 并有独立打榜 YAML，真实对象、teacher 加载、单卡 optimizer step、checkpoint-only resume/evaluate 和双 PPU Gloo DDP 已通过。legacy Python entry 仅作历史对照；R101 尚未迁移。

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

本轮新增的实际执行链为：

```text
tools/train.py
  -> config/loader.py + validation.py
  -> presets/j4.py
  -> builders/experiment_builder.py
  -> experiments/j4.py
  -> builders/trainer_builder.py
  -> Lightning Trainer.fit(...)
```

同时修复了真实训练 smoke test 暴露的两个数据/设备契约问题：

- BEV mask 的高级索引改为设备内 torch 索引和显式 broadcast，兼容当前 PPU runtime。
- 空 GT box 固定为 `[0, 9]`，不再在无目标样本上退化为 `[0]`。

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

## 4. 当前目录与职责

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
      ablation_param/
        j1_wl02_wh05.yaml ... mu_020.yaml
      ablation_module/
        a0_baseline_full_gt.yaml ... a5_roi_scale.yaml

  labeldistill/
    config/
      audit.py
      loader.py
      schema.py
      validation.py

    builders/
      experiment_builder.py
      trainer_builder.py

    experiments/
      j4.py

    presets/
      j4.py

  tools/
    resolve_config.py
    train.py
    evaluate.py
    numerical_snapshot.py
    validate_ablation_migration.py
```

`presets/j4.py` 保存 J4 的版本化低层网络构造细节；`experiments/j4.py` 保存通用训练行为；builder 只负责把已校验配置组装为对象。`train.py` 和 `evaluate.py` 都可从 YAML、保存的 `resolved_config.yaml` 或 checkpoint 内嵌 resolved config 启动。有限 test batch 自动使用 prediction-only，不运行要求完整 sample token 的 nuScenes 指标器。

## 5. 已采用的技术方案

当前环境已经安装：

- PyYAML 6.0.2
- OmegaConf 2.3.0
- Hydra 1.3.2

当前首版使用 **OmegaConf + argparse**，未直接接管为完整 Hydra 应用，原因是：

- 可以支持 YAML merge、dot-list 命令行覆盖和 resolved YAML 输出。
- 对现有 `run_cli` 和 PyTorch Lightning 启动方式改动较小。
- 不引入 Hydra working directory、launcher 和 sweep 行为带来的额外迁移复杂度。
- 以后需要批量 sweep 时仍可在相同 YAML schema 上接入 Hydra。

当前配置加载流程：

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

## 6. J4 baseline 的实际配置边界

J4 baseline 已拆成四个公共 base YAML 和一个实验 YAML：

```yaml
# configs/experiments/j4_wl05_wh07.yaml
_base_:
  - ../base/nuscenes.yaml
  - ../base/student_r50_128x128.yaml
  - ../base/teacher_centerpoint_vox01.yaml
  - ../base/distillation_roi_v3.yaml

experiment:
  name: j4_wl05_wh07
  seed: 0
  legacy_aliases: [param_J4_wl05_wh08]

runtime:
  gpus: 2
  num_nodes: 1
  distributed_backend: gloo
  batch_size_per_device: 16
  accumulate_grad_batches: 1
  max_epochs: 24
  precision: "16"
  gradient_clip_val: 35.0
  deterministic: true
  output_dir: outputs/j4_wl05_wh07
  resume_from: null

ema:
  enabled: true

checkpoint:
  save_top_k: 3
  save_last: true
  every_n_epochs: 1

optimizer:
  type: AdamW
  base_lr_at_global_batch_64: 0.0004
  weight_decay: 0.01
  backbone_lr_mult: 0.5

scheduler:
  type: MultiStepLR
  milestones: [19, 23]
```

公共 YAML 分工如下：

| 文件 | 负责内容 |
| --- | --- |
| `configs/base/nuscenes.yaml` | 数据路径/split、时序帧索引、物理几何、类别和 task groups |
| `configs/base/student_r50_128x128.yaml` | student 类型、输出通道、时序 KD 选择 |
| `configs/base/teacher_centerpoint_vox01.yaml` | teacher checkpoint、checkpoint prefix、proposal decode/NMS |
| `configs/base/distillation_roi_v3.yaml` | feature KD 通道、matching、scaler、mask 和 loss 权重 |
| `configs/experiments/j4_wl05_wh07.yaml` | 实验身份、runtime、EMA、checkpoint、optimizer 和 scheduler |

### 6.1 保留在 Python preset 中的硬编码

以下内容属于实现契约或网络拓扑，不开放为普通实验参数：

- J4 的 backbone、neck、depth head、BEV head 和 CenterPoint teacher 的低层 topology。
- tensor 维度约定、box code layout、structured result dataclass 和 batch 字段顺序。
- 当前 J4 preset 的 feature adaptor 拓扑，以及 `[128, 256]` 的精确通道契约。
- 五帧时序实现约束和 `legacy_half` 的具体张量选择算法。
- decode、matching、scaling、mask 的数学实现；YAML 只能选择已注册且已验证的算法。
- 当前只实现的 optimizer、scheduler、dataset、student/teacher 类型白名单。

这些值集中在 `labeldistill/presets/j4.py` 或对应算法实现中，并将源码 SHA256 写入审计文件。若未来需要开放，应先新增 registry 分支、schema 校验和 contract test，不能只添加一个无消费者的 YAML 字段。

### 6.2 由 YAML 控制的参数

以下内容是实验变量或运行变量，必须通过 YAML/override 控制：

- 数据 root、info 文件、split、workers、CBGS 和历史帧索引。
- point cloud range、LiDAR voxel size、BEV output stride、类别和 task groups。
- student/teacher 类型、teacher checkpoint/prefix、proposal 阈值和 NMS 参数。
- matching class policy、selection、逐类别距离阈值和 `one_to_one` 开关。
- scaler/mask 类型与参数、四项 loss 权重、feature KD 通道声明。
- GPU/节点数、分布式 backend、batch、梯度累积、epoch、precision、梯度裁剪和随机性。
- EMA/checkpoint、optimizer、scheduler、输出目录和 resume checkpoint。

### 6.3 只读派生参数

以下值由 validator 计算，禁止出现在普通源 YAML 或命令行 override 中：

- `derived.global_batch_size`
- `derived.effective_learning_rate`
- `derived.key_frame_count`
- `derived.feature_map_size`
- `derived.bev_cell_size`
- `derived.student_bev_input_channels`
- `derived.teacher_checkpoint_sha256`

标准输出目录中的 `resolved_config.yaml` 可以包含这些字段，用于审计和恢复。加载时程序会先移除保存值、重新计算并校验；派生值不参与控制执行。完整逐字段消费者关系见 `labeldistill/docs/config_effectiveness_matrix.md`。

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

当前正式启动方式：

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

NVIDIA 环境需要显式切换分布式 backend：

```bash
python tools/train.py \
  --config configs/experiments/j4_wl05_wh07.yaml \
  runtime.distributed_backend=nccl
```

只解析、校验并查看最终配置：

```bash
python tools/resolve_config.py \
  --config configs/experiments/j4_wl05_wh07.yaml \
  --no-write
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

checkpoint 内部已经保存 `labeldistill_schema_version` 和 `labeldistill_resolved_config`。加载时如果 schema 不兼容，应明确报错或执行显式迁移，不静默采用新默认值。

## 10. Python builder 的职责

当前统一入口：

```python
bundle = load_and_resolve_config(path, overrides, project_root=project_root)
experiment = build_experiment(bundle)
trainer = build_trainer(bundle.config, experiment)
trainer.fit(experiment, ckpt_path=bundle.config.runtime.resume_from)
```

builder 负责：

- 根据 `student.type` 构造学生网络。
- 根据 `teacher.type` 构造并冻结 teacher。
- 根据 `teacher.proposal` 构造独立 decoder。
- 根据 `matching` 构造 matcher。
- 根据 `region.scaler/mask` 构造 region policy。
- 根据 `loss` 构造 loss 权重。
- 根据 `optimizer/scheduler` 构造优化器。

实验 YAML 不再继承一个包含完整 `training_step` 的 Python 子类。当前 J4 通用训练行为位于 `labeldistill/experiments/j4.py`，低层拓扑位于 versioned `labeldistill/presets/j4.py`。

## 11. 迁移步骤

### 阶段 A：冻结当前行为

当前状态：**已完成**。固定 batch 0 的输入 tensor 指纹、legacy/YAML 两份完整快照和逐 tensor comparison report 已保存到 `outputs/acceptance/j4_legacy_yaml_batch0/`。

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
5. 基础设施阶段不改变模型和 loss 数学实现。

当前实现还额外记录：global batch size、effective learning rate、feature map size、BEV cell size、student BEV 输入通道、teacher checkpoint SHA256、配置继承链和关键源码 SHA256。

### 阶段 C：迁移 J4

当前状态：**已完成 builder 接管和数值等价性验收**。J4 YAML 通过 `tools/train.py` 构造通用 `J4Experiment`，不依赖 legacy J4 class；preset 与 legacy 的 backbone/head/teacher/proposal 配置已做完整对象级 parity，真实 batch 的 110 个 tensor 门禁已通过。

1. **已完成**：将 J4 的全部有效实验参数逐项迁移到 YAML。
2. **已完成**：通过 builder 构造与旧 J4 相同的模型和训练组件。
3. **已完成**：参数数量、完整配置对象、raw predictions、proposal、MatchResult、scaled GT、mask 和 loss 快照已对齐。
4. **已完成**：通过单卡短程 smoke test。
5. **已完成**：通过双卡 DDP、checkpoint 和 resume smoke test。

### 阶段 D：迁移消融实验

当前状态：**配置、数值基准和代表性运行验收已完成**。`ablation_param` 已按 J1～J5、P3 lambda 和 mu 的有效单变量生成 15 个 YAML，并通过“排除 experiment/runtime/evaluation 身份字段后只允许目标科学变量变化”的自动 diff 门禁；A0～A5 六个模块实验各有固定 batch 新框架快照。迁移统一采用当前 structured/ragged J4 实现，不要求与历史脚本 total loss 相等。

1. J1/J2/J3/J5 只保留 YAML 参数差异。
2. P3 lambda 系列只覆盖 loss weight。
3. mu 系列只覆盖 scaler 参数。
4. 自动生成 base-to-experiment config diff。

迁移时没有根据文件名直接假设这些实验只改变一个参数，而是审计了 J1/J2/J3/J5、P3 和 mu 脚本相对已重构 J4 的约 179～181 行差异。审计结果已分类为：

- 应统一到新 J4 的历史实现差异；
- 需要进入 schema/registry 的有效实验开关；
- 已确认不再需要复现的废弃行为。

其中参数实验真正保留的科学变量只有 mask 权重、feature loss 权重或 scaler `mu`；proposal/GT padding、重复 teacher 和旧 feature-loss 通道求和均统一到当前公共实现。`tools/validate_ablation_migration.py` 会对每个参数 YAML 的实际 scientific config diff 做自动门禁。

### 阶段 E：清理 legacy

当前两个指定消融族已满足以下条件，并于 2026-08-20 完成归档：

- YAML J4 能从头训练。
- 能恢复旧/新 checkpoint。
- 单 batch 数值对比符合预期。
- 至少完成一次短程多卡训练。
- 实验记录已经改用 resolved config 标识。
- 新测试和审计逻辑已经不再 import 或硬编码该 legacy entry。

legacy 清理按实验族分别验收，不要求等待所有实验一次性迁移完毕。`ablation_param` 和 `ablation_module` 已完成；`find_best_v3` 已按项目决策直接删除且不再迁移；ConvNeXt-B 已完成 YAML 配置与构造阶段，下一步是 GPU 运行验收。R101 不阻塞 ConvNeXt-B 打榜入口。详细清理顺序和保护范围见第 14 节。

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

J4 对拍使用相同 batch 和共享 state_dict。在当前 PPU TF32 环境中，两次独立 student 卷积前向存在约 `6e-3` 的重复运行漂移，因此门禁为 `atol=1e-2, rtol=1e-4`；teacher、ProposalBatch、MatchResult、scaled GT 和 mask 为精确一致，110 个 tensor 全部通过。A0～A5 因有意统一 ragged proposal、单次 teacher 和 feature channel-mean，不与旧脚本 total loss 强制相等，而以当前框架快照作为后续回归基准。

### 12.3 运行验证

2026-08-19 实测：当前 DSW 的两张 `PPU-ZW810E` 通过 Gloo 完成单步 DDP；该环境无 `libcuda.so`，因此 NCCL 不适用。单卡训练、双卡 checkpoint 保存以及恢复后 `global_step: 1 → 2` 均已通过。

- **通过**：`1 GPU × batch 1 × 1 step` 前向、反向和 optimizer step。
- **通过**：A0 full-channel + GT heatmap 的 `1 GPU × batch 1 × 1 step` smoke test。
- **通过**：A5 和 P3 lambda=0.4 的 `1 GPU × batch 1 × 1 step` smoke test。
- **通过**：A3 的 `2 PPU × batch 1/device × 1 step` Gloo DDP。
- **通过**：checkpoint 包含 schema、resolved config、teacher SHA256、optimizer 和 scheduler 状态。
- **通过**：从 `last.ckpt` 恢复后 `global_step: 1 → 2`，optimizer 和 scheduler 连续。
- **通过**：`evaluate.py --checkpoint --limit-test-batches=1` 恢复 A5 并导出 prediction-only 结果。
- **通过**：epoch 23 legacy J4 checkpoint 使用显式 J4 YAML 加载并完成单 batch prediction-only；504 个 legacy 顶层 teacher 键仅在与 `model.centerpoint.*` 完全一致时安全移除。
- **待目标集群验证**：正式训练目标卡数、吞吐与长程稳定性；NVIDIA 环境使用 NCCL。

本轮顶层项目回归 `pytest -q tests`：`120 passed`。新增覆盖 checkpoint 内嵌配置、legacy 重复 teacher 键的严格兼容、prediction-only evaluate、参数族 scientific config diff 门禁和数值 snapshot 比较器。无目录限制的全仓 pytest 会额外收集 vendored nuScenes tracking tests，并因当前环境未安装可选依赖 `motmetrics` 在 collection 阶段报错；该项不是本项目测试失败。

### 12.4 验收产物与复跑入口

所有数值产物均位于 `outputs/acceptance/`，属于验收资产，不作为普通源码提交的大二进制：

| 产物 | 结论 | 内容 |
| --- | --- | --- |
| `j4_legacy_yaml_batch0/comparison.json` | `passed=true` | 110 个 tensor 的逐路径误差与门禁结果 |
| `j4_legacy_yaml_batch0/{legacy,yaml}_snapshot.pt` | 已固化 | 同一 batch、共享 state_dict 的两份完整快照 |
| `ablation_module/*/snapshot.pt` | A0～A5 齐全 | 新框架 raw predictions、中间态、mask 和 losses |
| `ablation_migration_report.json` | `passed=true` | 参数 diff 门禁、模块策略、snapshot SHA256 和 loss 摘要 |
| `runtime_acceptance.json` | `passed=true` | A5/P3 单卡、A3 DDP/resume、新/旧 checkpoint evaluate smoke 摘要 |

A0～A5 固定 batch 的 total loss 为：

| 实验 | A0 | A1 | A2 | A3 | A4 | A5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total loss | 18999.0137 | 18999.2969 | 18999.2930 | 18999.2734 | 18999.0430 | 18999.0371 |

核心复跑命令：

```bash
# 复核已经固化的 J4 legacy/YAML 同 batch 快照
python tools/numerical_snapshot.py compare \
  --left outputs/acceptance/j4_legacy_yaml_batch0/legacy_snapshot.pt \
  --right outputs/acceptance/j4_legacy_yaml_batch0/yaml_snapshot.pt \
  --output outputs/acceptance/j4_legacy_yaml_batch0/comparison.json \
  --atol 1e-2 --rtol 1e-4

# 单个 YAML 的新框架快照；A0～A5 分别执行
python tools/numerical_snapshot.py capture \
  --config configs/experiments/ablation_module/a3_full_method.yaml \
  --output outputs/acceptance/ablation_module/a3_full_method/snapshot.pt

# 参数族 diff + A0～A5 snapshot 完整性门禁
python tools/validate_ablation_migration.py

# checkpoint 内嵌 resolved config 恢复训练
python tools/train.py --checkpoint /path/to/last.ckpt runtime.max_epochs=2

# 完整评测：不限制 test batch，执行 nuScenes metrics
python tools/evaluate.py --checkpoint /path/to/last.ckpt

# 有限 batch 入口 smoke：自动 prediction-only，不产生伪指标
python tools/evaluate.py --checkpoint /path/to/last.ckpt \
  --limit-test-batches 1
```

旧 checkpoint 若没有内嵌 `labeldistill_resolved_config`，必须显式传入 `--config`。当前已验证新 checkpoint 的 checkpoint-only train/resume/evaluate，也已验证 `outputs/param_J4_wl05_wh08/checkpoints/epoch_epoch=23.ckpt` 使用显式 J4 YAML 加载和推理。该 legacy checkpoint 额外包含 504 个顶层 `centerpoint.*` 键，且都与对应 `model.centerpoint.*` 完全相同；兼容代码只移除这种可证明的重复项，缺少配对或内容冲突时会硬失败。

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
5. 当前目录已有 Git 历史，但工作区包含用户未提交变更；归档/删除只处理明确批准的实验族，且不能依赖覆盖工作区来恢复误删内容。

### 14.2 当前仓库盘点

2026-08-19 的迁移前盘点为：

- `labeldistill/exps/` 下有 44 个有效 Python 文件，共约 15094 行。
- `ablation_param/` 有 15 个参数实验脚本，每个约 367 行。
- `find_best_v3/` 在迁移前有 6 个实验脚本；2026-08-20 按项目决策直接删除，不再进入 YAML builder。
- 有 63 个 `__pycache__` 目录、267 个 `.pyc` 文件。
- `build/` 约 48 MB。
- Python 3.8/3.9 的旧 CUDA 扩展约 18.09 MB；当前运行环境是 Python 3.10，实际加载 `cpython-310` 扩展。
- `.ipynb_checkpoints/` 约 0.58 MB，其中存在数份没有正式源码副本的历史文件。

2026-08-20 本批执行结果：`ablation_param` 与 `ablation_module` 共 21 个可运行脚本、7557 行源码（另含 2 个 `__init__.py`）已移出运行目录。归档位于 `archive/legacy_experiments/2026-08-20/`，`manifest.json` 保存逐文件原路径、归档路径、大小、行数和 SHA256。

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

以下文件没有当前代码引用，但仍不直接删除；Git 历史不能替代对用户工作区和唯一历史源码的显式归档：

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

**已直接删除，不再迁移。** 2026-08-20 按项目决策从运行目录移除以下实验族，不建立对应 YAML：

```text
labeldistill/exps/nuscenes/find_best_v3/
```

#### 14.5.2 `ablation_param`

**已完成。** J1～J5、P3 lambda 和 mu 系列的 YAML、builder、数值基准、config diff 门禁和代表性旧 checkpoint 显式配置恢复均已通过。测试、审计、可视化工具、训练文档和 shell 启动器不再依赖 legacy class。

最终应由一个公共实现加 YAML override 替代当前 15 份完整脚本：

```text
labeldistill/exps/nuscenes/ablation_param/
```

原目录已从运行路径移除，15 个脚本保存在 `archive/legacy_experiments/2026-08-20/ablation_param/`。

#### 14.5.3 `ablation_module`

A0～A5 的模型与 region policy 差异已经表达为明确配置，并完成六份数值快照；A3 DDP/resume、A5 单卡、evaluate smoke 和 legacy J4 checkpoint 兼容均已通过。

```text
labeldistill/exps/nuscenes/ablation_module/
```

**已完成归档。** 原目录已从运行路径移除，6 个脚本保存在 `archive/legacy_experiments/2026-08-20/ablation_module/`。

#### 14.5.4 其他实验和 backbone entry

依次迁移：

- `labeldistill/exps/nuscenes/labeldistill/`
- `labeldistill/exps/nuscenes/bevdepth/`
- `base_exp_convnextb.py`、`base_exp_res101.py`、`base_exp_swinb.py`
- teacher training entry

这些入口涉及 student backbone、teacher training、普通检测和蒸馏等不同用途，不作为一批无差别删除。

打榜用 ConvNeXt-B 已新增 `configs/experiments/convnextb_896x1600_j4_cbgs.yaml`，由 `student.type=camera_bevdepth_convnextb` 选择独立 preset，并通过公共 J4 builder 构造。该 YAML 独立控制 896×1600、CBGS、train+val、gradient checkpoint、warmup-cosine、LR multiplier、20 epoch 和 DDP policy，不复用 R50 训练设置。单卡 batch 1 optimizer step、checkpoint-only `global_step: 1 → 2` resume、prediction-only evaluate 和双 PPU Gloo DDP 均已通过，checkpoint 包含 resolved config、optimizer 与 scheduler 状态。legacy Python entry 仅作历史对照；R101 尚无等价 YAML。

#### 14.5.5 批量 shell 启动器

以下四个脚本已保留为便捷批量入口，但内容已切换为调用 `tools/train.py --config ...` 或 `tools/evaluate.py --config ... --checkpoint ...`，不再调用 legacy Python experiment：

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
- 尚未 YAML 化的 R101 entry，以及尚未完成 GPU 运行验收的 ConvNeXt-B legacy entry。

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

配置基础设施、J4 builder、两个指定消融实验族、真实 batch 数值快照、代表性短程运行、旧 checkpoint 兼容以及统一 train/evaluate/resume 入口均已完成。`ablation_param` 与 `ablation_module` 的 21 个重复运行脚本已经退出运行目录；`find_best_v3` 也已按决策直接删除。ConvNeXt-B 现已由独立 YAML 控制并通过配置解析、contract test、真实模型构造、teacher checkpoint 加载、单卡 optimizer step、checkpoint-only resume/evaluate 和双卡 DDP。下一步是正式目标卡数长程训练与完整 6019-sample nuScenes 评测；是否删除 ConvNeXt-B legacy 对照入口和是否迁移 R101 可独立决定。
