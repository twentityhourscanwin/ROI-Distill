# LabelDistill 配置有效性矩阵

每个源配置叶子节点必须有且只有一个明确消费者。`YAML` 表示 builder 或 runtime
真正读取的实验参数，`preset` 表示版本化 Python 实现细节，`derived` 表示只读派生值。
新增字段时必须同步补消费者、validation 和 contract test；禁止添加“能解析但不生效”的配置。

| Config path | Owner / consumer | Contract test |
| --- | --- | --- |
| `experiment.*` | audit, seeding, output identity | `test_config_system.py` |
| `runtime.*` | `trainer_builder.py`, optimizer LR derivation | `test_config_system.py`, `test_j4_builder.py` |
| `ema.*`, `checkpoint.*` | `trainer_builder.py` | trainer builder tests |
| `data.root`, `data.*_info`, `data.split` | `J4Experiment` dataloaders | builder tests |
| `data.num_workers`, `data.use_cbgs`, `data.key_idxes` | `J4Experiment`, J4 preset | builder/parity tests |
| `geometry.*` | J4 preset, mask builder, derived geometry | config/builder tests |
| `classes.names`, `classes.tasks` | student/teacher presets, matcher, scaler | config/builder tests |
| `student.type` | preflight registry validation | config tests |
| `student.output_channels`, `student.temporal_kd_selection` | J4 preset / `LabelDistill` | parity/builder tests |
| `teacher.type`, `teacher.checkpoint`, `teacher.checkpoint_prefix` | builder / `LabelDistill` | builder tests |
| `teacher.proposal.*` | teacher preset and independent decoder | parity/builder tests |
| `distillation.feature_channels` | `LabelDistill` adaptor contract | builder tests |
| `matching.*` | `ProposalTargetLayer` | config, ROI and builder tests |
| `region.scaler.*` | `AdaptiveGTScalerV3` | scaler/builder tests |
| `region.mask.*` | `QualityAwareMaskGeneratorV3` | mask/builder tests |
| `loss.*` | `J4Experiment.training_step` | experiment loss tests |
| `optimizer.*`, `scheduler.*` | `J4Experiment.configure_optimizers` | parity tests |
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

- 完整 legacy J4 ↔ YAML backbone/head/teacher/proposal 配置对象 parity 已通过。
- 配置与 builder 相关测试 `21 passed`。
- 算法、数据契约和 BEV mask 测试 `53 passed`。
- 单卡训练、双 PPU Gloo DDP、checkpoint 和 `global_step: 1 → 2` resume 已通过。
