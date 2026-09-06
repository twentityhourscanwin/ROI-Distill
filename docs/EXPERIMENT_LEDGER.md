# 实验详细记录

2026-09-06 更新：C1、R1、S1 已完成训练和完整 val 评测，见 [本轮完整指标与运行记录](VAL_RESULTS_20260906.md)。下方保留历史记录。

本文件保存每个实验的完整上下文。快速比较只看 [VAL_RESULTS.md](VAL_RESULTS.md)；这里用于复现、排查和判断结论是否有效。

## 2026-09-05 R1 / R2：已实现，待训练

| 字段 | 内容 |
| --- | --- |
| 代码身份 | `codex/box-radius-cap`；算法提交 `02b2c6a`；起点 `dev@cec9362`；已推送 origin |
| Config | `r1_teacher_value_radius_cap.yaml / r2_adaptive_radius_cap.yaml` |
| Changelog | feature KD 圆形半径 max(1,min(2,int(raw_radius)))；R1 使用原 GT，R2 使用原 Adaptive |
| 验证 | 相关测试 54 passed；实际 teacher SHA256 和配置核对；4 个真实框 CPU 抽查 |
| 训练 / Output | 未启动；未生成训练 run_id、checkpoint 和 evaluation 目录 |
| 效果 | 待训练评测，不填入 VAL_RESULTS |
| 说明 | [几何消融总览](GEOMETRY_ABLATIONS.md)；[RADIUS_CAP_ABLATION.md](RADIUS_CAP_ABLATION.md) |

以下共同实验口径和 B 系列运行身份保留为历史记录，不代表新分支已进行训练。

## 共同实验口径

| 项目 | 设置 |
| --- | --- |
| 学生 | BEVDepth R50；6 cameras × 5 timestamps；检测使用完整 750 channels；feature KD 使用 `legacy_half` |
| 教师 | frozen CenterPoint：`ckpts/centerpoint_vox01_128x128_20e_10sweeps.pth` |
| 教师 SHA256 | `850d60f7a3894000153e9479e3d67869d55f74ae94aa73abc945a7f5cdd72ddb` |
| 教师输入 | current 1 + past 5 + future 4，共 10 sweeps；仅用于离线训练 |
| train info | `nuscenes_infos_train.pkl`；SHA256 `4f38ffbf47c3225bab1ae4b42be81700475887e766db97cd675ad39eba2ad8b1` |
| val info | `nuscenes_infos_val.pkl`；SHA256 `eb2b63510e2709e0d18bd4a478af0e92f10820c0c7eee6fae618448362aec36f` |
| Loss 权重 | detection/depth/feature/response = `1.0/1.0/0.6/1.0` |
| Optimizer | AdamW；global batch 256 时 LR `4e-4`；weight decay `0.01`；backbone multiplier `1.0` |
| Scheduler | linear warmup 200 steps，ratio `0.001`；MultiStepLR `[19,23]`，gamma `0.1` |
| Runtime | 16 GPU × 16 samples/GPU；24 epochs；FP16；gradient clip 35；seed 0 |
| 评测 | nuScenes val 6019 samples；2 GPU × 16；普通 `last.ckpt` |
| 代码状态 | Git HEAD `f5f1400`，但运行使用未提交 working tree；没有正式实验 tag |

## B0 — Full-GT uniform

| 字段 | 内容 |
| --- | --- |
| 实验信息 | run_id `20260829_221359`；分支 `main`（dirty）；commit `f5f1400`；config `configs/experiments/b0_full_gt_uniform_no_scale.yaml`；seed 0 |
| Changelog | 基于 B1；`region.value=uniform_gt`，所有 effective GT 使用 `q=1`；bbox response 从 `matched_gt` 改为 `all_gt`；circular Gaussian；union-mask-mass 归一化；16 GPU × 16 |
| Output | train `outputs/b0_full_gt_uniform_no_scale_20260829_221359`；checkpoint `/mnt/nas_data/guqiupeng/checkpoint_nes/b0_full_gt_uniform_no_scale_20260829_221359/last.ckpt`；eval `outputs/evaluation/b0_20260829_221359_val_2x16` |
| 效果 | mAP 0.3919；NDS 0.5039；mATE 0.6144；mASE 0.2629；mAOE 0.4329；mAVE 0.3917；mAAE 0.2186 |
| 结论 | mAP 高于 B1，但 B0/B1 同时改变 feature value/gate 和 bbox response gate，不能解释为 teacher value 的单变量收益。 |
| 有效性 | 可作为当前口径的 seed 0 结果；需要 `uniform feature + matched-only bbox response` 桥接实验。 |

## B1 — Teacher-value reference

| 字段 | 内容 |
| --- | --- |
| 实验信息 | run_id `20260829_225349`；分支 `main`（dirty）；commit `f5f1400`；config `configs/experiments/b1_teacher_value_no_scale.yaml`；seed 0 |
| Changelog | 当前参考；exact-class、score-first、一对一匹配；`q=max(0,1-(d/tau)^2)`；circular Gaussian；union-mask-mass 归一化；bbox response `matched_gt`；16 GPU × 16 |
| Output | train `outputs/b1_teacher_value_no_scale_20260829_225349`；checkpoint `/mnt/nas_data/guqiupeng/checkpoint_nes/b1_teacher_value_no_scale_20260829_225349/last.ckpt`；eval `outputs/evaluation/b1_20260829_225349_val_2x16` |
| 效果 | mAP 0.3881；NDS 0.5047；mATE 0.6144；mASE 0.2635；mAOE 0.4143；mAVE 0.3825；mAAE 0.2192 |
| 训练观察 | TensorBoard 采样点中的 feature matched rate 约 0.953；teacher value mean 约 0.880。 |
| 结论 | 作为后续 idea 的固定参考；当前没有证据证明 teacher-value 优于 uniform GT。 |
| 有效性 | 只有 seed 0；需要桥接组和重复 seed。 |

## B1T — Oriented elliptical mask

| 字段 | 内容 |
| --- | --- |
| 实验信息 | run_id `20260830_104035`；分支 `main`（dirty）；commit `f5f1400`；config `configs/experiments/b1t_teacher_value_elliptical_mask.yaml`；seed 0 |
| Changelog | 基于 B1；唯一配置变化为 `per_gt_gaussian → per_gt_elliptical_gaussian`；其余 matching、q、归一化、batch 和训练参数保持一致。 |
| Output | train `outputs/b1t_teacher_value_elliptical_mask_20260830_104035`；checkpoint `/mnt/nas_data/guqiupeng/checkpoint_nes/b1t_teacher_value_elliptical_mask_20260830_104035/last.ckpt`；eval `outputs/evaluation/b1t_20260830_104035_val_2x16` |
| 效果 | mAP 0.3891；NDS 0.5037；mATE 0.6075；mASE 0.2648；mAOE 0.4299；mAVE 0.3915；mAAE 0.2145 |
| 结论 | mATE、mAAE 改善，但 mAP/NDS 没有超过 B1；当前不继续把 ellipse 作为默认 mask。 |
| 有效性 | B1/B1T resolved config 除实验身份、目录和 mask type 外一致；只有 seed 0。 |

## B2 — AdaptiveGTScalerV3

| 字段 | 内容 |
| --- | --- |
| 实验信息 | run_id `20260830_104032`；分支 `main`（dirty）；commit `f5f1400`；config `configs/experiments/b2_teacher_value_adaptive_scale.yaml`；seed 0 |
| Changelog | 基于 B1；唯一配置变化为 `region.scaler.enabled=false → true`；matching、q、circular Gaussian、union-mask-mass 归一化和训练参数保持一致。 |
| Output | train `outputs/b2_teacher_value_adaptive_scale_20260830_104032`；checkpoint `/mnt/nas_data/guqiupeng/checkpoint_nes/b2_teacher_value_adaptive_scale_20260830_104032/last.ckpt`；eval `outputs/evaluation/b2_20260830_104032_val_2x16` |
| 效果 | mAP 0.3922；NDS 0.5069；mATE 0.6066；mASE 0.2604；mAOE 0.4268；mAVE 0.3762；mAAE 0.2224 |
| 结论 | 当前 seed 0 最优；收益主要体现在位置、尺度和速度指标，方向与属性没有同步改善。 |
| 有效性 | scaler 存在正负 deadzone 不对称；未记录 mask changed rate、center-cell change rate 和几何分桶统计；修复和多 seed 前不能成为正式 baseline。 |

## D1 — Teacher–GT matching graph audit

这是一项只运行 frozen teacher 的诊断，不是学生训练实验，因此不写入 `VAL_RESULTS.md`。

| 字段 | 内容 |
| --- | --- |
| 实验信息 | run_id `matching_graph_v2_20260902`；分支 `codex/matching-analysis`；提取 commit `605c17857ddd3687f30fe190260956250c5e89a1`；config `configs/experiments/b1_teacher_value_no_scale.yaml`；seed 0 |
| Changelog | 基于 B1 的教师、decoder 和 GT 口径；不训练学生；identity BDA；逐样本确定性点云截断；缓存 exact-class、中心距离 `<4m` 的候选图；比较当前 greedy、GT-nearest/reuse、GT-highest-score/reuse、Hungarian distance 和 score×q/reuse |
| Input | teacher checkpoint 和 train/val info 使用本文件顶部 SHA256；严格使用 current + past 5 + future 4；场景边界用 current sweep 补至 10 帧 |
| Output | train `outputs/matching_graph_v2_train_20260902`；val `outputs/matching_graph_v2_val_20260902`；核心文件 `matching_analysis.json/.md`、`scope_summary.csv`、`threshold_sensitivity.csv`、`conflict_examples.png` |
| Process | train 28,130 samples，两个 rank 各 14,065；val 6,019 samples，两个 rank 为 3,010/3,009；每个 split 做 1,000 次 sample-level bootstrap |
| 有效性 | 两个 split 的 rank 完成标记、样本数、权重/数据 SHA256 和 10-sweep 时间差均核验；16 个 matcher 相关测试通过。原汇总器的上扩半径行会先截断候选图，已删除该两行并修复源码，核心五策略结果不受影响。 |

### 五种匹配策略

| Split | Policy | Matched / effective GT | Coverage | Mean q / all GT | Mean center distance | Mean teacher score | Mean BEV IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | P0 current score-first greedy | 595,734 / 624,021 | 95.4670% | 0.8833 | 0.3099 m | 0.6024 | 0.6243 |
| train | P1 GT-nearest, proposal reuse | 598,306 / 624,021 | 95.8791% | 0.8966 | 0.2981 m | 0.5959 | 0.6292 |
| train | P2 GT-highest-score, proposal reuse | 598,306 / 624,021 | 95.8791% | 0.8717 | 0.3270 m | 0.6080 | 0.6096 |
| train | P3 Hungarian normalized distance | 596,348 / 624,021 | 95.5654% | 0.8934 | 0.2983 m | 0.5945 | 0.6302 |
| train | P4 max score×q, proposal reuse | 598,306 / 624,021 | 95.8791% | 0.8932 | 0.3032 m | 0.6044 | 0.6249 |
| val | P0 current score-first greedy | 112,866 / 121,855 | 92.6232% | 0.8495 | 0.3272 m | 0.5916 | 0.6140 |
| val | P1 GT-nearest, proposal reuse | 113,554 / 121,855 | 93.1878% | 0.8651 | 0.3137 m | 0.5837 | 0.6188 |
| val | P2 GT-highest-score, proposal reuse | 113,554 / 121,855 | 93.1878% | 0.8388 | 0.3450 m | 0.5970 | 0.5980 |
| val | P3 Hungarian normalized distance | 113,036 / 121,855 | 92.7627% | 0.8609 | 0.3141 m | 0.5826 | 0.6201 |
| val | P4 max score×q, proposal reuse | 113,554 / 121,855 | 93.1878% | 0.8616 | 0.3191 m | 0.5930 | 0.6141 |

### 关键诊断

| 诊断 | train | val |
| --- | ---: | ---: |
| P3 − P0 matched GT | +614 | +170 |
| P3 − P0 coverage | +0.0984 pp；bootstrap 95% CI `[+0.0902,+0.1069]` pp | +0.1395 pp；bootstrap 95% CI `[+0.1168,+0.1646]` pp |
| P0 有候选但未匹配 | 2,572（0.4122% effective GT） | 688（0.5646%） |
| 最近 proposal = 最高分 proposal | 93.22%；多候选 GT 中 71.38% | 92.72%；多候选 GT 中 69.54% |
| P0 选中最近 proposal | 96.00% | 95.51% |
| GT 顺序 greedy：正序/倒序不同 assignment | 9,817 | 2,555 |
| edge score↔distance Spearman ρ | -0.5048 | -0.4740 |
| edge score↔BEV-IoU Spearman ρ | +0.4955 | +0.4636 |

结论：当前 P0 并非大面积失效，但目标不一致——它用 score 决定 proposal 顺序，最终 q 却只由距离决定。P3 在 train/val 都以更低距离、更高 q/IoU 和略高 coverage 稳定优于 P0，适合作为下一次单变量训练实验。P1 的额外 coverage 依赖 proposal reuse，只能视作上界；同一 proposal 监督多个相邻 GT 可能产生重复/冲突 target，不建议直接作为默认。按 GT 顺序做 one-to-one greedy 也不可靠，因为正序/倒序会改变数千个 assignment。

## 共同审计缺口

| 问题 | 影响 | 下一步 |
| --- | --- | --- |
| 运行代码未提交 | `f5f1400` 不能直接还原实际 working tree | 固化当前状态为 `dev` 起点 commit |
| 源码审计列表不完整 | 未记录 `kd_head.py`、LiDAR dataset 等关键源码哈希 | 补齐 audit source list |
| 数据哈希未写入 resolved config | 文档有哈希，但运行配置没有机器可读证据 | 把 train/val info SHA256 加入 derived config |
| EMA 未参与评测 | 保存约 24 份 EMA 权重，但指标来自普通 `last.ckpt` | 做一次 EMA/last 对照后决定保留或关闭 |
| sampler 行为未记录 | DDP 可能自动替换 sampler，实际顺序无法从产物确认 | 显式配置并打印实际 sampler |
| 只有 seed 0 | 千分位差异可能来自方差 | 对关键组补 seed 1/2 |

## 新实验记录模板

| 字段 | 内容 |
| --- | --- |
| 实验信息 | run_id；分支；commit/tag；config；seed |
| Changelog | 基于哪个 baseline；唯一修改变量；mask/归一化；batch size；GPU number |
| Input | teacher checkpoint + hash；train/val info + hash；sweep 组成 |
| Train | optimizer；各参数组 LR；scheduler；precision；gradient clip；epochs |
| Output | train log；checkpoint；evaluation 目录；metrics JSON |
| 效果 | mAP、NDS、mATE、mASE、mAOE、mAVE、mAAE |
| 结论 | 是否支持假设；观察到的副作用 |
| 有效性 | tests；多 seed；机制统计；已知混杂 |
