# C1：仅 proposal 偏移驱动的中心移动

2026-09-06：C1 已完成训练及全量 val 评测，mAP **0.388522** / NDS **0.503194**；[完整结果](VAL_RESULTS_20260906.md)。以下尚未训练的描述为此前实现阶段记录。运行统一使用原仓库，额外工作树已删除。

## 分支交付状态（2026-09-05）

| 项目 | 状态 |
| --- | --- |
| 分支 | [codex/proposal-center-only](https://github.com/twentityhourscanwin/ROI-Distill/tree/codex/proposal-center-only) |
| 起点 / 算法提交 | `dev@cec9362` / `f1af868` |
| Git | 算法已提交并推送 origin；工作树跟踪同名远端分支；最新 HEAD 包含后续文档提交 |
| 完整训练 / 评测 | 未启动；无 run_id、训练 checkpoint 或新 mAP/NDS |
| 总览 | [几何消融分支与对照](GEOMETRY_ABLATIONS.md) |

用户假设：监督区域向教师 proposal 调整可能改善师生有效特征的空间对应。C1 检验该假设，不把 proposal 中心等同于已测得的特征中心，也不移动或 warp 任一特征图。

共同起点：`dev@cec9362`。两个分支分别从 dev 创建，没有互相合并。
本轮对照不采用 9/2 的 gt_nearest / proposal reuse 实验。

固定设置：B1 score-first exact-class 一对一匹配，原 q、matched_gt response、legacy_half、feature 权重 0.6、16 GPU x 16 samples、24 epochs、seed 0；评测普通 last.ckpt。
数据、教师权重、已编译 native extensions 通过软链接共享原仓库；Python 源码使用本 worktree。运行必须设置 `PYTHONPATH="$PWD"`，避免环境中的 editable install 指向原仓库。
本轮完成配置、CPU 机制/梯度验证与真实框抽查；未启动完整训练，尚无新 mAP/NDS。

## 唯一几何机制

复用 AdaptiveGTScalerV3 的 offset 分量，速度贡献严格为零。匹配 GT 在局部坐标系中的位移：

```text
a_long = min(abs(proposal_offset_long), 0.15 * original_length)
a_lat  = min(abs(proposal_offset_lat),  0.15 * original_width)
shift_local = 0.5 * [legacy_sign(offset_long)*a_long,
                     legacy_sign(offset_lat)*a_lat]
center_new = center_original + rotate_to_global(shift_local, yaw)
```

legacy_sign：offset > 0 为 +1，offset < -0.1m 为 -1，其余为 0。保留原正负 deadzone 不对称，避免在这次消融中同时修复方向规则。
只改 x/y；长宽高、yaw、velocity、匹配索引、q、matched/effective masks 保持不变。未匹配 GT 不移动。不使用速度，也不包含完整 Adaptive 中速度贡献造成的中心位移。
圆形 Gaussian 仍为原 `max(2, int(raw_radius))`，中心仍量化。每轴位移上限为该轴尺寸的 7.5%，不会直接把中心吸附到 proposal。

对照：C1 vs B1 检验 offset-only 中心移动；C1 vs 原 Adaptive B2 比较移除尺寸和速度贡献后的结果。后一个对照不能单独拆分尺寸与速度贡献。

## 运行

```bash
cd /mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL
PYTHONPATH="$PWD" python tools/train.py --config configs/experiments/c1_proposal_center_only.yaml
```

## 验证

相关 CPU 测试 113 passed；随后对非有限速度处理做了隔离，再运行 C1 测试 19 passed。
覆盖旧 offset-only 中心数值一致、原框不变、速度独立、匹配/q 不变、空 GT、中心跨格后的实际 mask 与梯度变化，以及 YAML 到真实 builder 的连接。真实教师权重 SHA256 与 B1 一致，resolved global batch=256。

真实样本抽查复用 speed-mask 审计的 sample 8/car、76/motorcycle、97/pedestrian、1766/bicycle。所有框尺寸保持不变，car 中心格 (19,63)->(18,62)，bicycle (77,56)->(78,56)，这两个样本实际 mask 改变；另两个未跨格。
这四例只是执行检查，不估计整体变化率。逐样本数值见 `outputs/geometry_ablation_smoke_20260905/summary.json`。
