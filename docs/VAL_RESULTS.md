# 当前实验结果

2026-09-06 更新：C1、R1、S1 已完成训练和完整 val 评测，见 [本轮完整指标与运行记录](VAL_RESULTS_20260906.md)。下方保留历史记录。

统一口径：nuScenes val，6019 samples；训练使用 16 GPU × 16 samples/GPU；评测使用 2 GPU × 16 samples/GPU；评测权重均为普通 `last.ckpt`，不是 EMA `.pth`。

| 实验 | 实验信息 | Changelog | Output | 效果 |
| --- | --- | --- | --- | --- |
| B0 | run_id：`20260829_221359`<br>分支：`main`（dirty working tree）<br>commit：`f5f1400` | 基于 B1<br>`q=1`，覆盖全部有效 GT<br>circular Gaussian<br>union-mask-mass 归一化<br>bbox response：`all_gt`<br>batch：16×16 | `/mnt/nas_data/guqiupeng/checkpoint_nes/b0_full_gt_uniform_no_scale_20260829_221359/last.ckpt` | mAP **0.3919** / NDS **0.5039**<br>mATE 0.6144 / mASE 0.2629 / mAOE 0.4329 / mAVE 0.3917 / mAAE 0.2186 |
| B1 | run_id：`20260829_225349`<br>分支：`main`（dirty working tree）<br>commit：`f5f1400` | 当前参考<br>teacher-value `q`<br>circular Gaussian<br>union-mask-mass 归一化<br>bbox response：`matched_gt`<br>batch：16×16 | `/mnt/nas_data/guqiupeng/checkpoint_nes/b1_teacher_value_no_scale_20260829_225349/last.ckpt` | mAP **0.3881** / NDS **0.5047**<br>mATE 0.6144 / mASE 0.2635 / mAOE 0.4143 / mAVE 0.3825 / mAAE 0.2192 |
| B1T | run_id：`20260830_104035`<br>分支：`main`（dirty working tree）<br>commit：`f5f1400` | 基于 B1<br>仅将 circular 改为 oriented elliptical Gaussian<br>union-mask-mass 归一化<br>batch：16×16 | `/mnt/nas_data/guqiupeng/checkpoint_nes/b1t_teacher_value_elliptical_mask_20260830_104035/last.ckpt` | mAP **0.3891** / NDS **0.5037**<br>mATE 0.6075 / mASE 0.2648 / mAOE 0.4299 / mAVE 0.3915 / mAAE 0.2145 |
| B2 | run_id：`20260830_104032`<br>分支：`main`（dirty working tree）<br>commit：`f5f1400` | 基于 B1<br>仅启用 AdaptiveGTScalerV3<br>circular Gaussian<br>union-mask-mass 归一化<br>batch：16×16 | `/mnt/nas_data/guqiupeng/checkpoint_nes/b2_teacher_value_adaptive_scale_20260830_104032/last.ckpt` | mAP **0.3922** / NDS **0.5069**<br>mATE 0.6066 / mASE 0.2604 / mAOE 0.4268 / mAVE 0.3762 / mAAE 0.2224 |

当前结论：B2 的单次结果最好；B0/B1 同时改变 feature policy 和 bbox response scope，不能直接归因于 teacher value；所有结果只有 seed 0。

详细配置、数据哈希、训练参数和有效性见 [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md)。
