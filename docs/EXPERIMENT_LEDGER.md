# 实验台账

本文件是实验运行与结论的唯一登记表。每次正式运行使用唯一 ID；失败、中止和无效结果同样保留。表格中的 `—` 表示尚未产生有效结果，不表示 0。

## 1. 当前 B 系列

共同实现快照：`d27eed6`

共同 teacher SHA256：`9061688e5f81adae87d28241143e2d33075f68908134264f0de8c901acf911d8`

共同数据：`nuscenes_infos_{train,val}_10sweeps_past9.pkl`

| Run ID | 实验 | 唯一变化 | Config | Seed | 状态 | mAP | NDS | 产物 | 结论 |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| `B0-20260826-s0` | B0 | 相对 B1 使用 uniform `q=1` | `configs/experiments/b0_full_gt_uniform_no_scale.yaml` | 0 | Planned | — | — | — | 待训练 |
| `B1-20260826-s0` | B1 | 当前候选 baseline | `configs/experiments/b1_teacher_value_no_scale.yaml` | 0 | Planned | — | — | — | 待训练 |
| `B1T-20260826-s0` | B1T | 仅改椭圆 Gaussian | `configs/experiments/b1t_teacher_value_elliptical_mask.yaml` | 0 | Planned | — | — | — | 待训练 |
| `B2-20260826-s0` | B2 | 仅启用 AdaptiveGTScalerV3 | `configs/experiments/b2_teacher_value_adaptive_scale.yaml` | 0 | Blocked | — | — | — | 先处理 mask 不变率与 deadzone 问题 |

正式启动时应把 `Planned` 改成 `Running`，补充 Git tag、硬件和 `outputs/...`/对象存储路径；完成独立 evaluation 后再填写指标和结论。若改变 seed 或代码，新增行，不覆盖原 run。

## 2. 历史结果（不可与当前 B 系列直接比较）

以下结果来自旧网络、旧教师/数据或不同帧数与损失口径，状态统一为 `Historical / Invalid for current comparison`。这里保留必要摘要；更完整的原始表格可从整理前 Git 快照 `d27eed6` 追溯。

### 旧模块消融

| Run | 旧设计 | mAP | NDS | 有效性说明 |
| --- | --- | ---: | ---: | --- |
| A0 | full-channel GT-guided | 0.3767 | 0.4982 | 旧双帧/后修改为五帧，需重训 |
| A1 | channel split | 0.3737 | 0.4946 | 同上 |
| A2 | channel split + ROI | 0.3741 | 0.4942 | 同上 |
| A3 | channel split + ROI + scaling | 0.3767 | 0.4974 | 同上 |
| A4 | full-channel + ROI | 0.3700 | 0.4907 | 同上 |
| A5 | full-channel + ROI + scaling | 0.3742 | 0.4971 | 同上 |

### 旧参数消融

| Run | 参数摘要 | mAP | NDS | 有效性说明 |
| --- | --- | ---: | ---: | --- |
| J2 | `w_l=0.3,w_h=0.65` | 0.4004 | 0.5125 | legacy J-series |
| J3 | `w_l=0.35,w_h=0.75` | 0.4009 | 0.5105 | legacy J-series |
| J4 | `w_l=0.5,w_h=0.7` | 0.4033 | 0.5160 | legacy J-series |
| J5 | `w_l=0.3,w_h=0.8` | 0.3981 | 0.5139 | legacy J-series |
| P3-020 | `feature_weight=0.2` | 0.3934 | 0.5092 | legacy parameter sweep |
| P3-040 | `feature_weight=0.4` | 0.4013 | 0.5137 | legacy parameter sweep |
| P3-050 | `feature_weight=0.5` | 0.3996 | 0.5124 | legacy parameter sweep |
| P3-070 | `feature_weight=0.7` | 0.3948 | 0.5109 | legacy parameter sweep |
| P3-080 | `feature_weight=0.8` | 0.3983 | 0.5140 | legacy parameter sweep |
| P3-100 | `feature_weight=1.0` | 0.3951 | 0.5084 | legacy parameter sweep |

## 3. 新增运行模板

追加新运行时复制以下字段：

```text
Run ID:
Hypothesis:
Parent baseline:
Single intended change:
Config:
Git SHA / tag:
Teacher checkpoint + SHA256:
Dataset info + hash:
Seed:
Hardware / global batch / precision:
Status: Planned | Running | Completed | Failed | Invalid | Historical
Metrics: mAP, NDS, mATE, mASE, mAOE, mAVE, mAAE
Artifacts:
Conclusion:
Validity notes:
```

对于 B2 额外记录 `mask_changed_rate`；对于通道分割实验额外记录两半通道遮蔽/互换的性能变化。
