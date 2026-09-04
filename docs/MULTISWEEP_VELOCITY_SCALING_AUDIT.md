# Multi-sweep 速度缩放统计与实验建议

本文回答一个具体问题：teacher 使用 current + 5 past + 4 future LiDAR sweeps，而 GT 位于当前帧时，动态目标的 feature KD 区域应该怎样随速度变化。

本文只做数据与机制审计，不改变 B1/B2 实现。统计日期为 2026-09-03，代码基线为 `dev@8c46dfe`。

## 结论摘要

1. Teacher 的 10 sweeps 典型覆盖过去 `0.25 s`、未来 `0.20 s`，总跨度约 `0.45 s`。
2. 多帧点云变换到当前 ego 坐标系，自车运动得到补偿；动态物体本身没有逐实例运动补偿。因此当前帧 GT 周围可能出现约为 `|v| × 时间跨度` 的空间拖影。
3. 物理包络的绝对尺寸增量应先写成 `|v_local| × T_span`；换成缩放比例后才与原始框尺寸成反比。当前 V3 的速度项直接使用 `0%/10%/20% × 框尺寸`，没有对应的时间尺度。
4. 大类高速样本中，理论纵向增量中位数为 `2.694 m`，当前 V3 速度项中位数为 `0.946 m`；`82.11%` 的高速样本中理论增量更大。
5. 小类高速样本中，理论纵向增量中位数为 `0.590 m`，当前 V3 速度项中位数为 `0.155 m`；`97.64%` 的高速样本中理论增量更大。pedestrian、motorcycle、bicycle 的相对增量尤其显著。
6. barrier 和 traffic cone 几乎全部静止，速度-only 机制应对它们基本不产生变化。
7. 建议先测试只改变长宽、不使用 teacher proposal offset、也不移动中心的 absolute-displacement scaler；中心移动作为后续独立变量。

## 数据与统计口径

| 项目 | 口径 |
| --- | --- |
| 数据 | `data/nuScenes/nuscenes_infos_train.pkl`，28,130 samples |
| info SHA256 | `4f38ffbf47c3225bab1ae4b42be81700475887e766db97cd675ad39eba2ad8b1` |
| GT 时刻 | 当前 sample 的 `ann_infos` |
| teacher sweeps | current + past 5 + future 4，索引 `0..9` |
| 坐标 | sweep 点变换到当前 key ego；保留时间差特征；不做动态实例运动补偿 |
| effective GT | 训练类别合法；LiDAR+radar 点数大于 0；尺寸/中心可绘制；位于 nuScenes official class range 内 |
| 大类 | car、truck、construction_vehicle、bus、trailer |
| 小类 | barrier、motorcycle、bicycle、pedestrian、traffic_cone |
| 样本计数 | 按 annotation-frame 计数，同一实例跨帧会出现多次，不是 unique instance 数量 |

统计与 D1 matching graph audit 的 `624,021` 个 train effective GT 对齐：大类 `354,134`，小类 `269,887`。其中小类有 19 个 GT 的 velocity 为非有限值；它们保留在 effective GT 总数中，但不进入速度、位移和缩放分布，因此小类速度有效样本数为 `269,868`。

nuScenes 中 wheelchair、stroller 和 personal_mobility 在数据代码中映射为 `ignore`，本文没有把它们计入 pedestrian。

## 10-sweep 时间窗口

| Sweep index | 相对当前帧的典型时间 |
| ---: | ---: |
| 0 | `0.00 s` |
| 1 | `-0.05 s` |
| 2 | `-0.10 s` |
| 3 | `-0.15 s` |
| 4 | `-0.20 s` |
| 5 | `-0.25 s` |
| 6 | `+0.05 s` |
| 7 | `+0.10 s` |
| 8 | `+0.15 s` |
| 9 | `+0.20 s` |

| 时间量 | mean | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| `T_past` | 0.245 s | 0.250 s | 0.251 s | 0.300 s |
| `T_future` | 0.191 s | 0.200 s | 0.200 s | 0.250 s |
| `T_span=T_past+T_future` | 0.436 s | 0.450 s | 0.451 s | 0.500 s |

场景边界缺少的上下文会用 key sweep padding，因此少量样本的实际动态时间覆盖短于典型窗口。正式实现应优先使用每个样本的实际 `T_past/T_future`，而不是无条件固定为 `0.45 s`。

## 物理包络模型

把速度旋转到当前 GT 的局部坐标：

```text
v_long =  vx cos(yaw) + vy sin(yaw)
v_lat  = -vx sin(yaw) + vy cos(yaw)
```

在匀速、朝向不变，并希望覆盖所有 sweep 中完整物体框的近似下：

```text
delta_length = |v_long| * (T_past + T_future)
delta_width  = |v_lat|  * (T_past + T_future)

length' = length + delta_length
width'  = width  + delta_width
```

对应相对缩放比例为 `1 + delta_length/length` 和 `1 + delta_width/width`。因此绝对增量由速度和时间决定；原始框大小只在换成 scale factor 时出现。

过去和未来覆盖不完全对称，精确包络还会产生中心偏移：

```text
center_shift_local = 0.5 * (T_future - T_past) * [v_long, v_lat]
```

典型窗口下 `0.5 × (0.20-0.25)=-0.025 s`。例如 `10 m/s` 目标的完整尺寸增量为 `4.5 m`，但中心只向过去方向偏移 `0.25 m`。尺寸扩张与中心移动是两个不同变量。

如果实验要求中心严格固定且仍完整覆盖两端，尺寸增量应改为：

```text
delta_length_center_fixed = 2 * |v_long| * max(T_past, T_future)
delta_width_center_fixed  = 2 * |v_lat|  * max(T_past, T_future)
```

典型窗口、`10 m/s` 下纵向增量为 `5.0 m`。第一轮实验可以把中心固定、使用 `|v|T_span` 作为干净近似，但必须记录它在较长的过去侧少覆盖、在未来侧多覆盖各 `0.25 m`。

## 当前 V3 的速度项

当前 `AdaptiveGTScalerV3` 对每个局部速度分量独立使用：

```text
|v_component| < 0.3 m/s       -> 0
0.3 <= |v_component| < 0.8    -> 0.10 * GT dimension
|v_component| >= 0.8 m/s      -> 0.20 * GT dimension
```

它没有使用 sweep 时间跨度。当前 B2 的总扩张还包含 teacher proposal center offset；本文表格中的 “V3 speed term” 只计算上述速度贡献，用于和 `|v|T_span` 做单变量比较。

## 大类统计

下表中的 `N` 为速度有效的 effective GT；大类没有发现非有限 velocity。

| 类别 | N | 长×宽 p50 | `abs(v_long)>=0.8` | 高速组物理 `delta L` p50 | 高速组 V3 speed term p50 | 高速组物理增量更大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| car | 281,563 | 4.583×1.935 m | 21.18% | 2.818 m | 0.924 m | 87.36% |
| truck | 44,873 | 6.025×2.385 m | 21.37% | 2.352 m | 1.304 m | 71.74% |
| construction_vehicle | 8,262 | 6.184×2.684 m | 3.86% | 0.792 m | 1.006 m | 39.50% |
| bus | 7,325 | 12.052×2.925 m | 49.28% | 2.085 m | 2.427 m | 43.43% |
| trailer | 12,111 | 13.028×2.880 m | 13.57% | 2.357 m | 2.565 m | 45.01% |
| 大类汇总 | 354,134 | 4.670×1.978 m | 21.12% | 2.694 m | 0.946 m | 82.11% |

大类汇总由 car 主导，不能替代逐类判断。当前按框尺寸给 `20%` 的规则，对 car/truck 的高速拖影通常明显偏小；对 bus/trailer 则更接近中位物理增量，部分样本会偏大。

### 大类整体分位数

| 指标 | p50 | p75 | p90 | p95 |
| --- | ---: | ---: | ---: | ---: |
| GT length | 4.670 m | 5.087 m | 6.924 m | 10.487 m |
| GT width | 1.978 m | 2.141 m | 2.654 m | 2.939 m |
| `abs(v_long)` | 0.025 m/s | 0.254 m/s | 6.408 m/s | 8.995 m/s |
| `abs(v_lat)` | 0.009 m/s | 0.061 m/s | 0.159 m/s | 0.256 m/s |
| 物理 `delta L` | 0.010 m | 0.102 m | 2.807 m | 3.997 m |
| `delta L / length` | 0.2% | 2.1% | 55.1% | 82.9% |

大类中 `75.89%` 的 GT 满足 `|v_long|<0.3 m/s`，`2.99%` 位于 `[0.3,0.8)`，`21.12%` 不低于 `0.8 m/s`。`17.34%` 的全部大类 GT 理论纵向增量超过原长度的 `20%`，`14.12%` 超过 `35%`。

横向运动明显更小：`|v_lat|` 的 p95 为 `0.256 m/s`，典型 `0.45 s` 对应约 `0.115 m` 横向位移。因此大类速度扩张应主要作用于 length，width 只在少数转弯或横向运动样本中明显变化。

## 小类统计

下表中的 `N` 为速度有效的 effective GT。小类 effective GT 总数为 `269,887`；19 个非有限 velocity 样本不进入本表的速度统计。

| 类别 | N | 长×宽 p50 | `abs(v_long)>=0.8` | 高速组物理 `delta L` p50 | 高速组 V3 speed term p50 | 高速组物理增量更大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| barrier | 76,487 | 0.477×2.392 m | 0.01% | 0.445 m | 0.122 m | 100.00% |
| motorcycle | 7,395 | 2.090×0.758 m | 20.15% | 3.165 m | 0.427 m | 97.11% |
| bicycle | 7,257 | 1.692×0.598 m | 18.63% | 1.768 m | 0.368 m | 97.56% |
| pedestrian | 126,484 | 0.708×0.655 m | 59.13% | 0.586 m | 0.153 m | 97.66% |
| traffic_cone | 52,245 | 0.390×0.388 m | 0.08% | 0.451 m | 0.080 m | 90.70% |
| 小类汇总 | 269,868 | 0.588×0.687 m | 28.79% | 0.590 m | 0.155 m | 97.64% |

barrier 和 traffic cone 的高速行只包含极少量尾部样本，不能用来代表类别主体；这两个类别超过 99% 的样本满足 `|v_long|<0.3 m/s`，速度-only scaler 应基本保持原框。

### 小类整体分位数

| 指标 | p50 | p75 | p90 | p95 |
| --- | ---: | ---: | ---: | ---: |
| GT length | 0.588 m | 0.761 m | 0.995 m | 1.524 m |
| GT width | 0.687 m | 1.913 m | 2.913 m | 3.125 m |
| `abs(v_long)` | 0.032 m/s | 1.039 m/s | 1.426 m/s | 1.582 m/s |
| `abs(v_lat)` | 0.016 m/s | 0.063 m/s | 0.146 m/s | 0.231 m/s |
| 物理 `delta L` | 0.013 m | 0.447 m | 0.639 m | 0.711 m |
| `delta L / length` | 2.4% | 55.4% | 85.0% | 101.1% |

小类聚合尺寸受 barrier 的轴定义和类别占比影响，必须结合逐类表理解。小类中 `68.16%` 的 GT 满足 `|v_long|<0.3 m/s`，`3.06%` 位于 `[0.3,0.8)`，`28.79%` 不低于 `0.8 m/s`。`31.60%` 的小类 GT 理论纵向增量超过原长度的 `20%`，`29.50%` 超过 `35%`。

类别差异如下：

- pedestrian：移动样本比例最高；全部样本的 `delta L / length` p50 为 `57.0%`，高速组为 `74.8%`。
- motorcycle：高速组 `delta L / length` p50 为 `150.5%`，p90 为 `286.1%`。
- bicycle：高速组 `delta L / length` p50 为 `95.5%`，p90 为 `152.2%`。
- barrier、traffic_cone：主体静止；少量异常高速尾部应单独检查 velocity 质量，不应据此改变类别默认区域。

小类横向增量总体也较小：聚合 `delta W / width` p50/p75/p90/p95 为 `0.8%/3.7%/9.3%/14.9%`。motorcycle 的横向尾部相对更明显，其 p95 为 `20.6%`。

## 大类与小类对比

| 指标 | 大类 | 小类 |
| --- | ---: | ---: |
| effective GT | 354,134 | 269,887 |
| velocity-valid GT | 354,134 | 269,868 |
| `abs(v_long)>=0.8` | 21.12% | 28.79% |
| 全部样本 `delta L / length` p50 | 0.2% | 2.4% |
| 全部样本 `delta L / length` p90 | 55.1% | 85.0% |
| 高速组物理 `delta L` p50 | 2.694 m | 0.590 m |
| 高速组物理 `delta L / length` p50 | 52.5% | 75.3% |
| 高速组 V3 speed term p50 | 0.946 m | 0.155 m |
| 高速组物理增量大于 V3 | 82.11% | 97.64% |

大类的绝对拖影更长，小类的相对拖影更严重。统一按尺寸百分比扩张会掩盖这一区别：它倾向于给大框更大的绝对增量，却可能低估快速的小框目标。

## 建议的解耦实验

### V-Abs：只使用速度的绝对位移扩张

保持 B1 的 matching、teacher value `q`、circular Gaussian、response KD 和所有训练参数不变，仅改变 feature KD 使用的 GT 几何：

```text
delta_length = alpha * |v_long| * T_span
delta_width  = alpha * |v_lat|  * T_span

length' = length + delta_length
width'  = width  + delta_width
center' = current GT center
```

第一阶段明确关闭 teacher proposal offset 扩张、proposal offset 驱动的中心移动、原 V3 的分段速度项和 deadzone 逻辑。

`alpha=1` 对应完整匀速物理包络，但它对快速小目标可能过强。训练前先在固定 batches 上离线比较 `alpha={0.25,0.5,1.0}`，根据最终 rasterized Gaussian mask 而不是连续框数值选择训练候选。

### V-Abs-C：单独加入物理中心偏移

只有 V-Abs 显示机制和指标均合理后，再增加：

```text
center_shift_local = beta * 0.5 * (T_future - T_past) * [v_long, v_lat]
```

固定 V-Abs 选出的 `alpha`，比较 `beta=0/1`。这样中心移动只由带符号速度和时间窗口不对称决定，不再由 teacher proposal 的微小偏移符号控制。

## 训练前门禁

在相同 checkpoint、相同固定 batches、相同 teacher proposals 和 matching result 上记录：

1. `matched_mask`、matched proposal index 和 `q` 逐元素不变；
2. box changed rate、center changed rate；
3. Gaussian radius changed rate、center-cell changed rate；
4. 最终 mask changed rate、mask IoU 和 mask mass ratio；
5. 按类别、速度、距离和大/小类分桶；
6. feature loss 差值、梯度 norm 和相对 B1 的梯度 cosine；
7. 静止类别 barrier/traffic_cone 的 mask 应几乎不变；
8. 速度越高，连续框增量和 rasterized mask 变化应整体单调增加。

如果连续框发生明显变化但 Gaussian mask changed rate 很低，说明 BEV 离散化抵消了机制；此时不应直接启动完整训练。

## 结果判定

首轮只跑 seed 0，用于筛掉明显错误的 `alpha`。除了 mAP/NDS，还应重点检查 car/truck/bus/trailer 的 mATE/mASE/mAVE、pedestrian/motorcycle/bicycle 的 per-class AP、静止类是否退化，以及按 GT 速度分桶后的检测变化。

只有当 mask 变化符合速度机制、seed 0 不退化且指标改善方向可解释时，再补 seed 1/2。最终报告 mean/std 和三个 seed 的方向一致性，不能只用单次千分位差异判断机制成立。

## 后续验证

`|v|T` 描述的是原始点云可能形成的物理包络，不等于 teacher feature 的最优蒸馏区域。下一步可复用 D1 candidate graph，统计 matched teacher proposal 相对 GT 的长宽偏差是否随 `|v_long|T`、`|v_lat|T` 增长；再进一步抽样查看 teacher BEV feature activation 与多 sweep 点云覆盖。只有 teacher 表征也呈现相同趋势，才支持把完整物理包络直接作为 feature KD mask。
