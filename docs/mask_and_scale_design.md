









### Score 变换
| 参数 | 值 | 位置 |
|------|-----|------|
| 高质量 score 提升 | `(1.2 + s) / 2` | 
| 中等小类别 score | `(s + 1) / 2` | 
| 中等其他类别 score | 0.5 | 
| 未匹配 near score | 0.4 | 
| 未匹配 medium score | 0.2 | 



### 放缩参数
| 参数 | 值 | 代码位置 | 论文对应 |
|------|-----|----------|----------|
| 高质量 roi_direction_scale | 0.3 | `roi_distill_scale.py:14` `high_quality_cfg['roi_direction_scale']` | 公式(8)(9)中 α，s_roi = α · score_roi |
| 高质量 max_roi_offset | 2.0m | `roi_distill_scale.py:15` `high_quality_cfg['max_roi_offset']` | 公式(8)(9)中 τ_offset，min(\|Δx_local\|, τ_offset) |
| 高质量 velocity_scale | 0.2 | `roi_distill_scale.py:16` `high_quality_cfg['velocity_scale']` | 公式(8)(9)中 s_vel，与公式(6) f(v) 配合 |
| 高质量 max_scale | 1.0 | `roi_distill_scale.py:17` `high_quality_cfg['max_scale']` | 限制 extend_l/w ≤ original_size × max_scale（论文未显式写出） |
| 中等质量 roi_direction_scale | 0.4 | `roi_distill_scale.py:23` `medium_quality_cfg['roi_direction_scale']` | 公式(11)(12)中 α，α · d_offset/τ_max · \|n_x\| |
| 中等质量 max_roi_offset | 3.0m | `roi_distill_scale.py:24` `medium_quality_cfg['max_roi_offset']` | 公式(11)(12)中 τ_max，归一化偏移距离 |
| 中等质量 velocity_scale | 0.25 | `roi_distill_scale.py:25` `medium_quality_cfg['velocity_scale']` | 公式(11)(12)中 β，β · f(v_gt) |
| 中等质量 max_scale | 1.5 | `roi_distill_scale.py:26` `medium_quality_cfg['max_scale']` | 限制 scale_x/y 上界（论文未显式写出） |
| 未匹配 near/mid/far scale | 0.2/0.25/0.2 | `roi_distill_scale.py:34-36` `unmatched_cfg['near/medium/far_scale']` | 公式(13)中 s_base(d_gt) 分段函数 |
| 未匹配 velocity_scale | 0.3 | `roi_distill_scale.py:37` `unmatched_cfg['velocity_scale']` | 公式(13)中 s_vel · f(v_gt) |
| 速度分级阈值 | 0.3/0.8 m/s | `roi_distill_scale.py:83-91` `compute_velocity_scale()` | 公式(6)分段条件，沿用 CRKD 设定 |

### 损失权重
| 参数 | 值 | 位置 |
|------|-----|------|
| lidar_distill_loss 系数 | 0.6 | 实验脚本 |
