# R1/R2：由框计算并限制在 1-2 格的圆形 Gaussian

2026-09-06：R1 已完成训练及全量 val 评测，mAP **0.382972** / NDS **0.503018**；[完整结果](VAL_RESULTS_20260906.md)。以下尚未训练的描述为此前实现阶段记录。运行统一使用原仓库，额外工作树已删除。

## 分支交付状态（2026-09-05）

| 项目 | 状态 |
| --- | --- |
| 分支 | [codex/box-radius-cap](https://github.com/twentityhourscanwin/ROI-Distill/tree/codex/box-radius-cap) |
| 起点 / 算法提交 | `dev@cec9362` / `02b2c6a` |
| Git | 算法已提交并推送 origin；工作树跟踪同名远端分支；最新 HEAD 包含后续文档提交 |
| 完整训练 / 评测 | 未启动；无 run_id、训练 checkpoint 或新 mAP/NDS |
| 总览 | [几何消融分支与对照](GEOMETRY_ABLATIONS.md) |

共同起点：`dev@cec9362`。两个分支分别从 dev 创建，没有互相合并。
本轮对照不采用 9/2 的 gt_nearest / proposal reuse 实验。

固定设置：B1 score-first exact-class 一对一匹配，原 q、matched_gt response、legacy_half、feature 权重 0.6、16 GPU x 16 samples、24 epochs、seed 0；评测普通 last.ckpt。
数据、教师权重、已编译 native extensions 通过软链接共享原仓库；Python 源码使用本 worktree。运行必须设置 `PYTHONPATH="$PWD"`，避免环境中的 editable install 指向原仓库。
本轮完成配置、CPU 机制/梯度验证与真实框抽查；未启动完整训练，尚无新 mAP/NDS。

## 半径定义和下限选择

```python
radius = max(1, min(2, int(raw_radius)))
```

raw_radius 仍来自原 gaussian_radius((width/0.8,length/0.8), min_overlap=0.1)。最终只有 1、2 两档，对应 3x3、5x5 的软高斯窗口。没有旋转、连续中心或贴框裁剪，因此它缩小了原监督窗口，但并不严格贴合有向框，也不能保证连续尺寸变化都可见。

下限选 1：0 虽然可绘制 1x1 Gaussian，却可能在 128->64、bilinear、align_corners=True 的插值中完全消失。测试复现中心格 (1,1) 的 radius=0 下采样质量为零；radius=1 保持正质量。利用可分离高斯核，枚举了两个坐标轴的全部 128 种中心位置（含边界），确认 r=1 在下采样后质量都大于零，并验证两层 loss/gradient 有限且非零。
半径 1 的 sigma=0.5 格=0.4m，完整单框掩码质量约 1.61460；半径 2 为约 4.35045。loss 仍按最终 union-mask-mass 归一化，不按面积修改 feature loss 权重。

新增 `region.mask.max_radius: Optional[int]`，默认 null，旧配置绘制行为完全保留。有限上限只允许 circular per_gt_gaussian，必须 >= min_radius。只改变 feature KD mask，不改 detection/response heatmap。

## 对照配置

| 配置 | 几何 | mask | 目的 |
|---|---|---|---|
| B1 原配置 | 原 GT | 原 max(2,int(r)) | 固定参考 |
| R1 | 原 GT | max(1,min(2,int(r))) | 单独测试半径规则 |
| R2 | 原 Adaptive 全几何 | 与 R1 相同 | 在相同新 mask 下测试 Adaptive |

R1 vs B1 仅半径规则不同；R2 vs R1 仅 scaler.enabled 不同。R2 是补充配套配置，可用于比较新半径下的 Adaptive，不代表本轮已运行第三次训练。此分支没有合入 C1 的中心-only 实现。

## 运行

```bash
cd /mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL
PYTHONPATH="$PWD" python tools/train.py --config configs/experiments/r1_teacher_value_radius_cap.yaml
# 配套 Adaptive 组需单独运行：
PYTHONPATH="$PWD" python tools/train.py --config configs/experiments/r2_adaptive_radius_cap.yaml
```

## 验证

相关 CPU 测试 54 passed，覆盖上下限、小目标、边界、插值存活、两层 loss/梯度、错误配置拒绝及 YAML 到真实 builder 的连接。R1/R2 均完成真实教师权重哈希核验及 resolved config 检查。
真实样本抽查采用已有 sample 8/car、76/motorcycle、97/pedestrian、1766/bicycle。R1 的四例半径均从 2 变为 1，框中心和尺寸保持不变，实际 mask 均改变。此为执行检查，不估计类别总体。逐例数值见 `outputs/geometry_ablation_smoke_20260905/summary.json`。
