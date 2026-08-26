# LabelDistill 配置有效性矩阵

每个源配置叶子节点必须有且只有一个明确消费者。`YAML` 表示 builder 或 runtime
真正读取的实验参数，`preset` 表示版本化 Python 实现细节，`derived` 表示只读派生值。
新增字段时必须同步补消费者、validation 和 contract test；禁止添加“能解析但不生效”的配置。

| Config path | Owner / consumer | Contract test |
| --- | --- | --- |
| `experiment.*` | audit, seeding, output identity | `test_config_system.py` |
| `runtime.*` | `trainer_builder.py`, optimizer LR derivation；含 DDP static graph/find-unused policy | `test_config_system.py`, `test_j4_builder.py`, `test_big_backbone_config.py` |
| `ema.*`, `checkpoint.*` | `trainer_builder.py` | trainer builder tests |
| `evaluation.run_metrics` | `J4Experiment.on_test_epoch_end`; false exports per-rank predictions | `test_j4_builder.py` |
| `data.root`, `data.*_info`, `data.split` | `J4Experiment` dataloaders | builder tests |
| `data.num_workers`, `data.use_cbgs`, `data.key_idxes` | `J4Experiment`, J4 preset | builder/parity tests |
| `geometry.*` | J4 preset, mask builder, derived geometry | config/builder tests |
| `classes.names`, `classes.tasks` | student/teacher presets, matcher, scaler | config/builder tests |
| `student.type` | preset dispatch；当前支持 R50 与 ConvNeXt-B | config/big-backbone tests |
| `student.output_channels`, `student.temporal_kd_selection` | J4 preset / `LabelDistill`; selects full or channel-split feature KD | parity/builder tests |
| `student.image.*` | ConvNeXt-B preset；控制 source/final size、resize、pretrained、gradient checkpoint 和 drop path | `test_big_backbone_config.py` |
| `teacher.type`, `teacher.checkpoint`, `teacher.checkpoint_prefix` | builder / `LabelDistill` | builder tests |
| `teacher.proposal.*` | teacher preset and independent decoder | parity/builder tests |
| `distillation.feature_channels` | `LabelDistill` adaptor contract | builder tests |
| `matching.*` | `ProposalTargetLayer` | config, ROI and builder tests |
| `region.scaler.*` | `AdaptiveGTScalerV3` | scaler/builder tests |
| `region.value.*`, `region.mask.*` | continuous teacher value and per-GT Gaussian policy | config/ROI/reducer/builder tests |
| `loss.*` | `J4Experiment.training_step` | experiment loss tests |
| `optimizer.*`, `scheduler.*` | `J4Experiment.configure_optimizers`；支持 MultiStepLR 与 step-based linear-warmup cosine | parity/big-backbone tests |
| `derived.*` | validation only; source YAML must not set it | config tests |

## Python preset 保留项

以下内容不作为普通实验参数开放：

- student/teacher 的低层网络 topology，以及 J4 feature adaptor 的结构。
- tensor shape、box-code layout、structured result types 和 batch 字段契约。
- 五帧时序实现和 `legacy_half` 的具体张量选择算法。
- decode、matching、scaling、mask 的数学实现。
- 当前实现支持的 dataset、student、teacher、optimizer 和 scheduler 白名单。

这些代码的 SHA256 会进入 `environment.txt`。修改 preset 即代表实现版本发生变化，不能伪装成一次纯 YAML 消融。

## 只读派生值

`derived.global_batch_size`、`effective_learning_rate`、`key_frame_count`、
`feature_map_size`、`bev_cell_size`、`student_bev_input_channels` 和
`teacher_checkpoint_sha256` 由 validation 统一计算。普通源 YAML 和命令行禁止设置；
`resolved_config.yaml` 中保存的值在恢复时会被移除、重算并校验。

## 当前验证结论

- 当前唯一 baseline 是 `center_value_baseline_cp50200.yaml`。
- matching 使用 exact-class、score-first nearest-unmatched、严格 `< tau` 和 one-to-one。
- teacher value 使用 normalized squared margin，不乘 teacher score。
- feature KD 使用 per-GT Gaussian 空间归一化和全局 effective `N_gt` 固定计数归一化。
- BDA 物理距离、official class range、空 GT、Gaussian 面积不变性和 value-gradient 比例均有 contract test。
- 清理旧消融门禁后，`pytest -q tests` 为 `107 passed`；单卡 FP16 和双卡 DDP 单步训练均已通过。
- 旧网络派生的 `ablation_module`、`ablation_param` 配置、入口、迁移测试和归档源码已于 2026-08-23 移除。
- legacy J4 YAML 只承担历史兼容和对照，不再是新消融的母配置。
- ConvNeXt-B/J4 仍为独立历史配置；迁移到 center-value 之前不与当前 baseline 直接比较。
