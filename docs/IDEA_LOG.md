# Idea 迭代记录

本文件只记录一条逻辑：实验结果说明了什么问题，我们下一步准备怎样修改。运行事实放在 `EXPERIMENT_LEDGER.md`，这里不重复完整配置。

## 当前目标

在保持相机学生推理成本不增加的前提下，提高 nuScenes val 的 mAP/NDS，并确认提升来自可解释、可重复的蒸馏机制，而不是单 seed 波动或实验变量混杂。

## 迭代 1：teacher-value 是否比 full-GT uniform 更好

### 实验结果

- B0：mAP 0.3919，NDS 0.5039。
- B1：mAP 0.3881，NDS 0.5047。
- 教师匹配率约 95.3%，B1 teacher value mean 约 0.88。

### 分析

B1 没有稳定优于 B0。当前比较又不是单变量：B0 同时使用 uniform feature KD 和 `all_gt` bbox response，B1 使用 teacher-value feature KD 和 `matched_gt` bbox response。因此不能判断是 q 无效，还是 response gate 导致差异。

union-mask-mass 归一化还会让单个孤立 ROI 的整体 q 在分子和分母中抵消，q 主要改变多实例之间的相对权重。当前 q 的实际作用可能比直觉更弱。

### 下一步

补一个桥接组：`uniform feature + matched-only bbox response`。先把 feature value/gate 和 bbox response gate 拆开，再决定是否继续优化 q 公式。

## 迭代 2：椭圆 mask 是否比圆形 mask 更符合目标几何

### 实验结果

- B1T 相对 B1：mAP +0.0010，NDS -0.0010。
- mATE、mAAE 改善，但 mAOE、mAVE 变差。

### 分析

椭圆 mask 改变了特征监督的空间分布，但没有转化成整体检测收益。当前不值得继续单独调 ellipse 的长宽比例；优先级低于拆分 B0/B1 混杂和验证 B2。

### 下一步

暂时保留实现，不作为默认配置。只有后续发现特定类别或距离段有稳定收益时再恢复该方向。

## 迭代 3：利用教师 proposal 偏差调整 ROI 几何

### 实验结果

- B2 相对 B1：mAP +0.0041，NDS +0.0022。
- B2 相对 B0：mAP +0.0003，NDS +0.0030。
- 改善主要来自 mATE、mASE 和 mAVE；mAOE、mAAE 没有改善。

### 分析

这是当前最有希望的方向，结果形态也符合“几何监督改变位置、尺度和运动学习”的预期。但现有 scaler 有正负 deadzone 不对称，而且没有记录最终 mask 是否真的变化，所以目前只能说明“这份具体实现的 seed 0 结果更好”，不能证明 adaptive scaling 机制成立。

### 下一步

1. 修复 center shift 的正负不对称，形成新的代码分支和实验身份。
2. 记录 mask changed rate、center-cell change rate、尺寸变化率，以及类别/距离/速度分桶。
3. 先做固定 batch 数值对比，确认 scaler 确实改变监督区域。
4. 再训练 seed 0；机制和结果一致后补 seed 1/2。

## 迭代 4：half-channel KD 是否真的形成私有特征

### 当前问题

`legacy_half` 只在通道维切前一半。检测仍使用全部通道，L1 又已经过卷积混合，因此不能保证前半是 LiDAR 子空间、后半是相机私有子空间。

### 下一步

在 B2 机制确认后，再比较：no feature KD、full KD、legacy half、explicit KD/private projections。配合通道遮蔽和梯度归因，不只看最终 mAP/NDS。

## 下一轮实验队列

| 优先级 | 实验 | 要回答的问题 | 唯一变化 | 成功标准 | 状态 |
| ---: | --- | --- | --- | --- | --- |
| 1 | Bridge-U-M | B0/B1 差异来自 feature policy 还是 bbox response gate？ | B0 的 feature 设置 + B1 的 `matched_gt` response | 能拆开两类贡献 | Planned |
| 2 | B2-fixed diagnostic | scaler 是否真实、对称地改变 mask？ | 修复 deadzone并增加统计 | mask/center change 可测且方向合理 | Planned |
| 3 | B2-fixed-s0 | 修复后能否保持 B2 收益？ | 使用修复后的 scaler | mAP/NDS 不低于 B1，几何指标保持改善 | Blocked by diagnostic |
| 4 | B0/B1/B2 seeds | 千分位差异是否稳定？ | seed 1/2 | 报告 mean/std，方向基本一致 | Blocked by code freeze |
| 5 | EMA A/B | EMA 是否值得保存和评测？ | 同一 run 比较 last/EMA | 明确指标收益与存储成本 | Planned |
| 6 | Gradient audit | 各 loss 是否发生梯度冲突？ | 固定 checkpoint/batches，仅做诊断 | 得到各参数组 norm/cosine/clipping | Planned |
