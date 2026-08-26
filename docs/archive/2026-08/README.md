# 2026-08 历史文档

本目录保存方案演进和问题追踪材料，仅用于追溯，不代表当前实验口径。

| 文档 | 归档原因 |
| --- | --- |
| `2026-08-17_LidarDistill_ConvNeXtB_16卡网络与问题诊断.md` | 针对旧 ConvNeXt/J4 路径；其中“BDA 已对齐”和“教师 9 sweep”等判断已被后续代码审计推翻，loss 与训练设置也已变化。 |
| `2026-08-20_R50教师GT匹配官方距离范围统计.md` | 使用旧教师和旧 LiDAR info，只保留为 trust-radius 决策的历史证据。 |
| `2026-08-20_nuScenes中心距离质量感知蒸馏指标设计.md` | 设计已经落地并由当前 B0/B1/B1T/B2 方案取代；其中 `overlap=sum`、`feature_weight=0.6` 和旧 checkpoint 已失效。 |
| `2026-08-21_CenterPoint50200教师GT匹配TrainVal统计.md` | step 50200 教师使用过 train+val，不能作为当前官方教师的干净 val 证据。 |

当前有效定义见：

```text
../../实验计划和配置.md
```
