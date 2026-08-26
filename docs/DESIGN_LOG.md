# 设计与网络演进日志

状态定义：`Proposed`（提出）、`Implemented`（已实现待实证）、`Validated`（验证并采纳）、`Rejected`（验证后不采用）、`Superseded`（被新设计替代）。条目只追加，不因 feature branch 删除而删除。

## D001 — YAML 驱动实验框架

- 日期：2026-08-19
- 状态：`Validated`
- 实现：`c3548b4`，后续快照 `d27eed6`
- 决策：公共数据、学生、教师和蒸馏配置放在 `configs/base/`；实验入口放在 `configs/experiments/`，由 builder、validation 和 contract tests 保证字段确实生效。
- 原因：替代大量复制的 Python 实验脚本，避免配置名变化但代码未消费，以及不同消融无意间改变多个变量。
- 约束：修改 Python preset 代表实现变化；纯参数消融只能修改 YAML。

## D002 — B1 的 proposal–GT 质量值

- 日期：2026-08-26
- 状态：`Implemented`
- 配置：`configs/experiments/b1_teacher_value_no_scale.yaml`
- 决策：采用 exact-class、score-first nearest-unmatched、one-to-one 和严格距离阈值；使用 `q=max(0,1-(d/tau)^2)`，不乘 teacher score。
- 假设：教师 proposal 越接近 GT 中心，该实例的教师特征越值得被学生模仿。
- 验证：B0 对 B1；两者只能改变 `region.value`。
- 尚缺：按当前教师、past-9 数据和统一 seed 重新训练。

## D003 — Per-GT fixed-count feature loss

- 日期：2026-08-26
- 状态：`Implemented`
- 决策：单实例 Gaussian 空间归一化，实例损失按 effective GT 固定计数归一化；重叠区域取 `max`。
- 原因：避免大框或大 mask 仅因像素更多而获得更大权重，也避免 `q` 同时影响分子和分母后削弱质量差异。
- 门禁：禁止在未更名实验的情况下改成 `sum(q)` 或全局 `mask.sum()`。

## D004 — 学生 half-channel feature KD

- 日期：2026-08-26
- 状态：`Implemented`
- 当前实现：`student.temporal_kd_selection=legacy_half`。检测路径使用全通道；feature KD 在 L0/L1 只取前一半通道。
- 初始假设：一半表示受 LiDAR 教师约束，另一半保留相机学生的私有能力。
- 现实边界：共享上游参数和后续卷积会混合信息，切片并不能保证两个功能子空间真正解耦；L0 还与时间拼接顺序耦合。
- 后续方案：增加显式 KD/private 双投影分支，可选 decorrelation/orthogonality 约束；比较 no feature KD、full KD、legacy half 和 explicit split。
- 判定指标：除最终 mAP/NDS 外，增加两半通道遮蔽、互换和线性探针实验。

## D005 — B1T 椭圆 Gaussian

- 日期：2026-08-26
- 状态：`Implemented`
- 决策：在 B1 上只把 circular Gaussian 改为 oriented elliptical Gaussian。
- 假设：沿目标朝向和长宽分布的 mask 比圆形 mask 更贴合物体几何。
- 验证：B1T 对 B1；matching、q、原始 GT geometry、loss 和训练参数保持一致。

## D006 — B2 AdaptiveGTScalerV3

- 日期：2026-08-26
- 状态：`Implemented`
- 决策：在 B1 上只启用 scaler，允许 proposal 偏差改变 feature ROI 几何。
- 已知风险：整数 radius 和最小半径会吞掉多数连续尺度变化；局部 x 偏移 deadzone 存在正负不对称。
- 进入验证前：先修复或明确接受 deadzone；记录 `mask_changed_rate`、center-cell change rate 和按类别/距离分桶的几何变化。
- 注意：修复 scaler 是代码变化，应通过 feature branch/PR，并使旧 B2 运行失效或更名。

## D007 — Git 与文档治理

- 日期：2026-08-26
- 状态：`Validated`
- 决策：Git commit 保存代码行为，YAML 表达实验变量，tag 固化运行身份，实验台账保存结果，当前基线文档只保存已采纳状态。
- 分支策略：只有代码/网络变化建立短期 feature branch；B0/B1/B1T/B2 不建立长期分支。
- 文档策略：`CURRENT_BASELINE`、`DESIGN_LOG`、`EXPERIMENT_LEDGER` 是三个核心事实面；旧材料统一归档。
