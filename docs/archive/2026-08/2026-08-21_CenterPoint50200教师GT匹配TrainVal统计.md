# CenterPoint step 50200 Teacher–GT 匹配统计

日期：2026-08-21  
权重：`outputs/train_teacher_centerpoint/checkpoints/step_step=50200.ckpt`  
数据：nuScenes train / val  

> **后续 baseline 更新（2026-08-21）**：本文保留原始统计及其当时的“所有类别统一 2m”建议。当前确定的蒸馏 baseline 已更新为 `exact-class-only + small 1m / large 2m + score-first nearest-unmatched one-to-one + continuous teacher value`。train/val 中 small 从 1m 放宽到 2m 仅增加 0.62%/0.67% 覆盖，而 large 增加 3.39%/3.42%，且规律一致。本文现有覆盖率仍是 nearest exact-class 统计，不是 one-to-one 后的最终匹配率。

## 1. 统计设置

- 双 PPU，每卡 batch size 16，仅执行 teacher 前向，不计算梯度。
- GT 先按 nuScenes 官方类别范围过滤：barrier、traffic_cone 为 30m；pedestrian、motorcycle、bicycle 为 40m；其余类别为 50m。
- 中心距离统计使用最近的 exact-class teacher proposal。
- 当前 matcher 是每个 GT 在候选距离档内选择最高分 proposal，不是严格一对一；因此 proposal 复用率用于衡量该问题。

## 2. Train 统计

共处理 28,130 个样本、624,021 个官方距离范围内 GT。

### 2.1 当前 matcher

| 分组 | GT 数 | 匹配率 | Unmatched | 匹配中精确同类 | proposal 复用 | 多候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 269,888 | 98.74% | 1.26% | 99.66% | 10.25% | 24.62% |
| large | 354,133 | 97.07% | 2.93% | 99.95% | 1.82% | 2.48% |

### 2.2 最近同类框中心距离命中率

| 分组 | `<0.5m` | `<1m` | `<2m` | `<4m` | 2m→4m 增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 92.50% | 97.92% | 98.54% | 98.91% | 0.37% |
| large | 79.96% | 92.18% | 95.57% | 97.03% | 1.46% |

### 2.3 最近同类框 teacher score

| 中心距离 | small 均值 / 中位数 | large 均值 / 中位数 |
| --- | ---: | ---: |
| `<0.5m` | 0.656 / 0.699 | 0.678 / 0.713 |
| `0.5～1m` | 0.558 / 0.599 | 0.508 / 0.525 |
| `1～2m` | 0.449 / 0.453 | 0.372 / 0.346 |
| `2～4m` | 0.331 / 0.270 | 0.412 / 0.348 |

small 当前最高分框同时也是最近框的比例依次为 93.70%、79.63%、19.74%、1.78%，说明中心距离超过 1m 后，最高分选择与最近框选择明显分离。

## 3. Val 统计

共处理 6,019 个样本、121,855 个官方距离范围内 GT。

### 3.1 当前 matcher

| 分组 | GT 数 | 匹配率 | Unmatched | 匹配中精确同类 | proposal 复用 | 多候选 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 53,656 | 98.45% | 1.55% | 99.69% | 11.70% | 24.91% |
| large | 68,199 | 96.61% | 3.39% | 99.92% | 1.65% | 2.44% |

### 3.2 最近同类框中心距离命中率

| 分组 | `<0.5m` | `<1m` | `<2m` | `<4m` | 2m→4m 增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 92.14% | 97.59% | 98.26% | 98.76% | 0.50% |
| large | 79.45% | 91.71% | 95.13% | 96.54% | 1.40% |

### 3.3 最近同类框 teacher score

| 中心距离 | small 均值 / 中位数 | large 均值 / 中位数 |
| --- | ---: | ---: |
| `<0.5m` | 0.659 / 0.704 | 0.685 / 0.720 |
| `0.5～1m` | 0.555 / 0.601 | 0.516 / 0.538 |
| `1～2m` | 0.422 / 0.399 | 0.376 / 0.350 |
| `2～4m` | 0.348 / 0.274 | 0.392 / 0.314 |

small 当前最高分框同时也是最近框的比例依次为 93.43%、79.13%、23.17%、3.50%，与 train 具有相同趋势。

## 4. 当时的核心结论（已由后续 baseline 更新）

1. train 和 val 的匹配率、距离命中率及 score 分布接近，统计规律具有较好一致性。
2. 2m 内已经覆盖 train/val 的 98.54%/98.26% small GT 和 95.57%/95.13% large GT；放宽到 4m 的额外收益仅为 small 0.37%/0.50%、large 1.46%/1.40%。
3. small 的核心问题稳定存在：多候选比例约 25%，proposal 复用率为 train 10.25%、val 11.70%；large 的两项比例均较低。
4. teacher score 在 0～2m 内随中心距离增大明显下降，支持匹配完成后使用中心距离和 score 共同计算蒸馏质量。
5. 建议采用与 nuScenes TP metrics 一致的 2m 门限，并使用官方式一对一流程：同类 proposal 按 score 降序处理，匹配 2m 内最近的未匹配 GT。实施后需要重新统计一对一匹配率。

完整结果：

```text
outputs/teacher_gt_matching_cp50200_train_20260820/report_official_ranges.md
outputs/teacher_gt_matching_cp50200_train_20260820/summary_official_ranges.json
outputs/teacher_gt_matching_cp50200_val_20260820/report_official_ranges.md
outputs/teacher_gt_matching_cp50200_val_20260820/summary_official_ranges.json
```
