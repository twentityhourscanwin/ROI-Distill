# S1：速度扩张与半径规则的 2x2 消融

2026-09-06：S1 已完成训练及全量 val 评测，mAP **0.384472** / NDS **0.500657**；[完整结果](VAL_RESULTS_20260906.md)。以下尚未训练的描述为此前实现阶段记录。运行统一使用原仓库，额外工作树已删除。

分支 `codex/speed-radius-cap` 从 `codex/box-radius-cap@c9387af` 创建，移植固定中心速度实现 `a6b38dd`（本分支对应提交 `d0b93ae`）。两项机制最终都来源于 `dev@cec9362`，没有合入 proposal-center-only，也没有改 matching。
组合实验、四组验证及审计工具提交：`06a4390`；后续文档提交不改变算法。当前实现和 CPU 审计已完成，尚未启动训练。

## 组合定义

```text
v_long = vx*cos(yaw) + vy*sin(yaw)
v_lat  = -vx*sin(yaw) + vy*cos(yaw)
L_new = L + 0.5 * abs(v_long) * (0.25 + 0.20)
W_new = W + 0.5 * abs(v_lat)  * (0.25 + 0.20)
center_new = original_center
raw_radius = gaussian_radius((W_new/0.8, L_new/0.8), min_overlap=0.1)
radius = max(1, min(2, int(raw_radius)))
```

只用 GT 速度，不含 proposal offset 扩张或中心移动；half fraction=0.5，时间窗口固定 past=0.25s、future=0.20s，沿用历史固定中心 Speed B2。场景边界不动态改变这两个时间参数，避免引入第三个变量。
只调整 feature KD 的 GT 几何和圆形 mask；检测 GT、response heatmap/bbox、匹配、q 和归一化保持原设置。

## 第一阶段：四组主实验

| 组 | 名称 / 配置 | 速度扩张 | 半径规则 |
| --- | --- | --- | --- |
| A | B1 / `b1_teacher_value_no_scale.yaml` | 关闭 | max(2,int(r)) |
| B | 固定中心 Speed B2 / `b2_speed_half_centered_circular.yaml` | 开启 | max(2,int(r)) |
| C | R1 / `r1_teacher_value_radius_cap.yaml` | 关闭 | max(1,min(2,int(r))) |
| D | S1 / `s1_speed_half_radius_cap.yaml` | 开启 | max(1,min(2,int(r))) |

四份配置都在本分支，使用同一干净 commit/tag 运行。匹配固定 exact-class、score-first、一对一，q 公式、matched_gt response、legacy_half、teacher/data、feature weight=0.6、global batch=256、24 epochs、seed=0、普通 last.ckpt 评测全部相同。

- B-A：速度在原绘制规则下的效果。
- D-C：速度在新绘制规则下的效果；这是本次最关键的对照。
- C-A：不扩框时，单独修改半径的效果。
- D-B：已有速度扩张时，修改半径的效果。
- `(D-C)-(B-A)`：分别对 mAP/NDS 计算交互项，用于判断新规则是否改变速度扩张的收益。先报告配对差值；不能凭单 seed 宣称统计显著。

D-A 只能说明组合效果，不能拆分速度和半径的贡献。旧 A/B 成绩可供探索时参考，但正式四组结论应在本分支固定相同训练口径复跑。原 Adaptive B2 和 C1 暂放在第二阶段，避免与本轮无中心移动的两因素混杂；9/2 的学生训练结果不参与本轮。

建议先跑 C、D 的 seed 0，看最关键的配对差值，再补同提交 A、B。四组机制/结果方向值得继续时，在同一 seed 1/2 上成对复跑四组，报告每组 mean/std 及配对差值。
如果还要区分“下限 2->1”和“新增上限 2”各自贡献，再增加 `min_radius=1,max_radius=null` 的速度关/开配对组；本轮四组把它们合称为一次半径规则改变。

## CPU 验证与真实掩码审计

相关测试 70 passed，包含速度 scaler、半径 reducer、四组 resolved config 的科学参数 diff、组合配置到真实 builder、半径阈值穿越以及两层 feature gradient 变化。固定案例验证了：同一速度扩张在原规则下掩码不变，在新规则下半径能从 1 变为 2，且改变实际梯度。

使用现有 train teacher candidate graph（不是 9/2 学生训练结果），固定 NumPy seed 0，从 28,130 帧中抽取 1,024 帧。1,023 帧有 GT，1,019 帧有正 q 监督，共 21,110 个 effective 且 q>0 GT，其中速度 >=0.8m/s 的为 5,394 个。
调用本分支实际 matcher/scaler/mask；identity BDA，四组共用匹配和 q。核验 teacher/data SHA256、10 sweeps、缓存 effective 标记、中心不变和 q/匹配索引不变；audit 运行时 Git 为干净 `06a4390`。

| 对照 | 单框 mask 改变 | 速度>=0.8m/s GT 的改变率 | 最终 max(q*mask) 合并图改变 |
| --- | ---: | ---: | ---: |
| B/A：原规则加速度 | 46/21,110 = 0.218% | 42/5,394 = 0.779% | 44/1,019 = 4.318% |
| D/C：新规则加速度 | 216/21,110 = 1.023% | 209/5,394 = 3.875% | 183/1,019 = 17.959% |
| C/A：只改半径 | 94.268% | 94.735% | 100% |
| D/A：同时改两项 | 93.245% | 90.860% | 100% |

合并图“完全相同”的比例在 128 和 64 层一致；不同层的连续权重变化量并不相同。D/C 的逐样本归一化 mask L1 均值在 128 层为 0.030643、64 层为 0.028548；B/A 分别为 0.005508、0.005357。它们是场景掩码分布指标，不是全局 DDP batch loss 或训练梯度的实测值。

D/C 改变的单框来自 car 171/9,542（1.792%）和 truck 45/1,430（3.147%）；本次 pedestrian、bicycle、motorcycle、bus、trailer、construction_vehicle、barrier、traffic_cone 都没有变化。不要把总体改变量理解为所有类别均生效。

解释：新半径实际为 `raw_radius<2 -> 1`、`raw_radius>=2 -> 2`。因此速度只能在跨过 2 这一个阈值时起作用；小目标大多仍在阈值以下，原本已达 2 的大目标受上限限制不能继续扩大。这比旧规则更容易让部分 car/truck 扩张生效，但仍不是连续、贴框或定向的运动区域。

已确认组合实现能工作，速度项相对旧规则更易传到 mask，但大部分 D/A 变化来自半径规则本身。训练结果是否改善仍待 C/D 及完整四组对照。统计基于固定 frame 抽样和 identity BDA，不代表完整随机训练增强分布，也不提供独立实例置信区间。

汇总与可追溯信息：[总体](evidence/speed_radius_20260905/summary.csv)、[运动组](evidence/speed_radius_20260905/moving_summary.csv)、[分类](evidence/speed_radius_20260905/per_class.csv)、[合并图](evidence/speed_radius_20260905/scene_summary.csv)、[抽样/源码/数据哈希](evidence/speed_radius_20260905/provenance.json)。逐 GT 和逐帧完整产物只保存在本 worktree 的 `outputs/speed_radius_audit_20260905`。

## 运行

组合组：

```bash
cd /mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL
PYTHONPATH="$PWD" python tools/train.py --config configs/experiments/s1_speed_half_radius_cap.yaml
```

其余组在同一 worktree 中替换为表中配置。每个训练配置均为 16 GPU x 16 samples，按组顺序运行。data、ckpts、native extensions 共用原服务器依赖，必须设置 PYTHONPATH 以加载本分支源码。

重做审计时使用新的输出目录：

```bash
cd /mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 PYTHONPATH="$PWD" python tools/audit_speed_radius.py \
  --cache /mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL/outputs/matching_graph_v2_train_20260902 \
  --samples 1024 --seed 0 --output outputs/speed_radius_audit_repeat
```
