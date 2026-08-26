# 2026-08 历史文档

本目录保存旧网络、旧教师、旧数据口径、阶段性诊断、迁移前方案和重复训练命令，仅用于追溯，不代表当前实验口径。

| 文档 | 归档原因 |
| --- | --- |
| `2026-08-17_LidarDistill_ConvNeXtB_16卡网络与问题诊断.md` | 针对旧 ConvNeXt/J4 路径；其中“BDA 已对齐”和“教师 9 sweep”等判断已被后续代码审计推翻，loss 与训练设置也已变化。 |
| `2026-08-20_R50教师GT匹配官方距离范围统计.md` | 使用旧教师和旧 LiDAR info，只保留为 trust-radius 决策的历史证据。 |
| `2026-08-20_nuScenes中心距离质量感知蒸馏指标设计.md` | 设计已经落地并由当前 B0/B1/B1T/B2 方案取代；其中 `overlap=sum`、`feature_weight=0.6` 和旧 checkpoint 已失效。 |
| `2026-08-21_CenterPoint50200教师GT匹配TrainVal统计.md` | step 50200 教师使用过 train+val，不能作为当前官方教师的干净 val 证据。 |

即使归档文件名中出现“当前”“baseline”或“正式”，也不得直接作为新实验依据。当前事实入口为：

```text
../../CURRENT_BASELINE.md
../../DESIGN_LOG.md
../../EXPERIMENT_LEDGER.md
```

旧指标必须带 `Historical` 或 `Invalid` 状态，不能与当前 B 系列直接横向比较；旧命令只用于复现历史环境。
