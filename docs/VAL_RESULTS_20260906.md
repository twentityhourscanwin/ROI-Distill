# 2026-09-06 C1 / R1 / S1 验证结果

三组均已训练完成并成功评测：nuScenes val 全部 6,019 samples；2 卡 x 16 samples/card；普通 last.ckpt（epoch=23，即第 24 个 epoch，global_step=2616），seed 0。未使用 EMA，未限制验证批次数。三个模型依次在同一原仓库切换对应分支评测，上一组进程退出后才切换下一组。

| 实验 | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.388066 | 0.504651 | 0.614355 | 0.263499 | 0.414301 | 0.382476 | 0.219192 |
| Adaptive B2 | 0.392224 | 0.506870 | 0.606624 | 0.260395 | 0.426783 | 0.376174 | 0.222442 |
| Speed B2 | 0.389817 | 0.502941 | 0.628615 | 0.262931 | 0.422245 | 0.394204 | 0.211683 |
| C1 | 0.388522 | 0.503194 | 0.614590 | 0.264829 | 0.423461 | 0.394683 | 0.213102 |
| R1 | 0.382972 | 0.503018 | 0.622978 | 0.261141 | 0.412593 | 0.378279 | 0.209686 |
| S1 | 0.384472 | 0.500657 | 0.629331 | 0.260873 | 0.419508 | 0.388675 | 0.217401 |

前三行为已保存的历史参照结果；后三行为本次新完成的评测。B1 为 8/29，Adaptive B2 为 8/30，Speed B2 为 9/4 固定中心速度扩张；不纳入 9/2 学生训练结果。mAP/NDS 越高越好，五项误差越低越好。

## 本轮观察

- C1 相比 B1：mAP +0.000456、NDS -0.001456。没有重现原 Adaptive 的整体优势。
- R1 相比 B1：mAP -0.005094、NDS -0.001632。仅收紧半径在这次运行中没有改善整体指标。
- S1 相比 R1：mAP +0.001500、NDS -0.002361。新半径下加入速度后 AP 略高，但位置、方向、速度、属性误差变差，NDS 下降。
- 三组均未超过原 Adaptive B2 的 mAP/NDS。以上是单 seed 观察，不能据此证明或否定全部中心对齐/速度建模假设。

## 运行与产物

### C1

- Run：`c1_proposal_center_only_20260906_003726`
- 分支：`codex/proposal-center-only`；评测源码提交：`d262ed607656034e2404463d27773ce899e93422`
- 权重：`/mnt/nas_data/guqiupeng/checkpoint_nes/c1_proposal_center_only_20260906_003726/last.ckpt`
- 评测目录：`/mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL/outputs/evaluation/c1_proposal_center_only_20260906_003726_val_2x16`
- 产物：`metrics_summary.json`、`metrics_details.json`、`results_nusc.json`、`resolved_config.yaml`、`environment.txt`。预测结果样本数已核验为 6,019。

### R1

- Run：`r1_teacher_value_radius_cap_20260905_232844`
- 分支：`codex/box-radius-cap`；评测源码提交：`c9387af6923dabb826ba963fba1a43287343c192`
- 权重：`/mnt/nas_data/guqiupeng/checkpoint_nes/r1_teacher_value_radius_cap_20260905_232844/last.ckpt`
- 评测目录：`/mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL/outputs/evaluation/r1_teacher_value_radius_cap_20260905_232844_val_2x16`
- 产物：`metrics_summary.json`、`metrics_details.json`、`results_nusc.json`、`resolved_config.yaml`、`environment.txt`。预测结果样本数已核验为 6,019。

### S1

- Run：`s1_speed_half_radius_cap_20260905_232924`
- 分支：`codex/speed-radius-cap`；评测源码提交：`2619de8f75e3048e4479efea745d6d190628f417`
- 权重：`/mnt/nas_data/guqiupeng/checkpoint_nes/s1_speed_half_radius_cap_20260905_232924/last.ckpt`
- 评测目录：`/mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL/outputs/evaluation/s1_speed_half_radius_cap_20260905_232924_val_2x16`
- 产物：`metrics_summary.json`、`metrics_details.json`、`results_nusc.json`、`resolved_config.yaml`、`environment.txt`。预测结果样本数已核验为 6,019。

## 分类 AP

| 类别 | B1 | Adaptive B2 | Speed B2 | C1 | R1 | S1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 0.585469 | 0.588717 | 0.589796 | 0.588769 | 0.583254 | 0.585957 |
| truck | 0.344348 | 0.331301 | 0.343665 | 0.340528 | 0.342247 | 0.332481 |
| bus | 0.398852 | 0.428844 | 0.404000 | 0.404076 | 0.390812 | 0.390940 |
| trailer | 0.176530 | 0.197522 | 0.192385 | 0.177150 | 0.192441 | 0.211257 |
| construction_vehicle | 0.106758 | 0.106738 | 0.107333 | 0.100265 | 0.113392 | 0.117981 |
| pedestrian | 0.378127 | 0.380706 | 0.385329 | 0.381185 | 0.364065 | 0.364666 |
| motorcycle | 0.385446 | 0.384756 | 0.378595 | 0.402724 | 0.372648 | 0.372219 |
| bicycle | 0.371130 | 0.395973 | 0.375095 | 0.372212 | 0.364210 | 0.377028 |
| traffic_cone | 0.556806 | 0.550542 | 0.559494 | 0.557453 | 0.548008 | 0.539991 |
| barrier | 0.577192 | 0.557142 | 0.562481 | 0.560857 | 0.558642 | 0.552197 |

## 原仓库运行约定

仅使用 `/mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL`，按方法切换分支；不新增 worktree。同一代码目录上的运行需串行，运行进程结束后再 checkout。此前三个额外工作树已删除，其中抽查产物保留在原仓库 `outputs/retained_ablation_checks_20260906/`。
