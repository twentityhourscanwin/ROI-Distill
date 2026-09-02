# 开发与运行说明

本文件保存项目的通用信息：当前开发基线、训练/评测命令、目录约定和 Git 工作流。

## 文档分工

| 文档 | 用途 |
| --- | --- |
| [VAL_RESULTS.md](VAL_RESULTS.md) | 简洁查看当前实验结果 |
| [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) | 查看单次实验的完整配置、产物和有效性 |
| [IDEA_LOG.md](IDEA_LOG.md) | 根据结果分析问题并规划下一轮 idea |
| [BRANCH_LEDGER.md](BRANCH_LEDGER.md) | 记录 dev 与各开发分支的分叉点、commit、影响范围和状态 |
| [README.md](README.md) | 通用训练命令、当前 dev 设置和 Git 规则 |

## 当前开发状态

| 项目 | 当前状态 |
| --- | --- |
| 当前 Git 分支 | `dev` |
| dev 起点 | 本文档所在提交；使用 `git rev-parse HEAD` 获取实际 SHA |
| Working tree | dev 起点提交后应保持 clean；新的修改只在短期 idea 分支进行 |
| 上一个 main HEAD | `f5f1400` |
| 当前固定参考 | B1：mAP 0.3881，NDS 0.5047 |
| 当前领先候选 | B2：mAP 0.3922，NDS 0.5069；尚未通过 scaler 机制和多 seed 门禁 |
| dev 起点验证 | Python compileall 通过；pytest 分组运行 `60 + 71 + 2 = 133 passed`；一次性全套运行存在设备层等待，尚需单独排查 |

当前 `dev` 起点保存了 B 系列实现、配置、测试和四份协作文档。新的代码 idea 从该提交切短期分支；正式运行仍需额外创建 experiment tag。

## 当前 dev 默认设置

| 项目 | 设置 |
| --- | --- |
| 学生 | `camera_bevdepth_r50`；图像 256×704；5 timestamps；输出 150 channels/frame |
| 时序 | current + `[-2,-4,-6,-8]` |
| Feature KD 通道 | `legacy_half`；L0/L1 前半通道进入 adaptor；检测仍使用完整通道 |
| 教师 | frozen CenterPoint；10-sweep checkpoint；`eval`；不参与梯度 |
| 教师 LiDAR | current 1 + past 5 + future 4；未来帧仅用于离线训练 |
| Feature loss | raw Gaussian union mask；overlap `max`；每层除以最终 `mask.sum()`；channel error 使用 sum |
| Response KD | heatmap：teacher heatmap × GT heatmap + GaussianFocalLoss；bbox scope 由配置控制 |
| Loss 权重 | detection/depth/feature/response = `1.0/1.0/0.6/1.0` |
| 训练规模 | 16 GPU × 16 samples/GPU；global batch 256；24 epochs；FP16 |
| Optimizer | AdamW；LR `4e-4`；backbone LR multiplier `1.0`；weight decay `0.01` |
| Scheduler | 200-step linear warmup；MultiStepLR `[19,23]`；gamma `0.1` |
| DDP backend | `gloo` |
| EMA | enabled；每 epoch 保存 `.pth`，但当前正式评测尚未使用 EMA |
| Dataset sampling | `use_cbgs=false`；源码 DataLoader 为 `shuffle=false, sampler=None`；Lightning DDP 可能自动注入 sampler，实际行为尚未落日志 |
| Fit validation | `limit_val_batches=0`；训练结束后独立执行 evaluation |

## 训练命令

在仓库根目录执行。由 Lightning 创建 DDP 进程，不要在外层再套 `torchrun`。

### B0

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
python tools/train.py \
  --config configs/experiments/b0_full_gt_uniform_no_scale.yaml
```

### B1

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
python tools/train.py \
  --config configs/experiments/b1_teacher_value_no_scale.yaml
```

### B1T

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
python tools/train.py \
  --config configs/experiments/b1t_teacher_value_elliptical_mask.yaml
```

### B2

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
python tools/train.py \
  --config configs/experiments/b2_teacher_value_adaptive_scale.yaml
```

在 `codex/matching-gt-nearest` 分支上，B1、B1T、B2 统一使用 M1：每个 GT 独立选择阈值内同类最近 proposal，并允许 proposal reuse；无需额外 M1 配置文件。`dev` 分支保留旧匹配机制。

如需改变 GPU 或 batch，使用显式覆盖，并把 resolved global batch/LR 写入实验台账：

```bash
python tools/train.py \
  --config configs/experiments/b1_teacher_value_no_scale.yaml \
  runtime.gpus=8 \
  runtime.batch_size_per_device=8
```

## 评测命令

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python tools/evaluate.py \
  --checkpoint /absolute/path/to/last.ckpt \
  --output-dir outputs/evaluation/<run_name>_val_2x16 \
  runtime.gpus=2 \
  runtime.batch_size_per_device=16
```

评测完成后把 `metrics_summary.json` 和 `metrics_details.json` 的结果写入 `VAL_RESULTS.md` 和 `EXPERIMENT_LEDGER.md`。

## Teacher–GT 匹配诊断

正式诊断使用当前训练代码一致的 teacher 输入：索引 `0..9`，即 current + past 5 + future 4；identity BDA；点云超限截断按 `seed + sample_index` 固定。

```bash
torchrun --standalone --nproc_per_node=2 \
  tools/extract_teacher_gt_graph.py \
  --config configs/experiments/b1_teacher_value_no_scale.yaml \
  --split train \
  --output-dir outputs/matching_graph_v2_train_<date> \
  --batch-size 16 \
  --num-workers 4

python tools/analyze_teacher_gt_graph.py \
  --input-dir outputs/matching_graph_v2_train_<date> \
  --bootstrap-reps 1000
```

把 `--split` 和输出目录改为 `val` 即可运行验证集。提取阶段只前向一次 teacher，缓存 `<4m` 的 exact-class candidate graph；P0–P4、分桶、bootstrap 和冲突图都从缓存离线重算。

不要用旧的 `tools/analyze_teacher_gt_matching.py` 产生新的正式结论：该脚本内部固定 `np.arange(9)`，实际得到 current + past 5 + future 3，再用 current padding 第十帧，与当前 teacher 的 future 4 输入不一致。历史输出只保留作追溯。

## 运行目录

新训练自动产生 `YYYYMMDD_HHMMSS` run_id：

```text
outputs/<experiment>_<run_id>/
/mnt/nas_data/guqiupeng/checkpoint_nes/<experiment>_<run_id>/
```

续训使用原 run_id，不创建新的实验身份。正式运行必须记录 output、checkpoint、evaluation 三个目录。

## Git 工作流

| 分支/标识 | 用途 |
| --- | --- |
| `main` | 已验证、可以稳定复现的版本 |
| `dev` | 日常集成分支和所有新 idea 的起点 |
| `codex/<idea>` | 算法或代码变化的短期分支 |
| `configs/experiments/*.yaml` | 同一代码实现上的配置消融 |
| `exp/YYYYMMDD-<experiment>-s<seed>` | 正式运行 tag，指向干净提交 |

建议流程：

```text
在 IDEA_LOG 写问题和成功标准
→ 从 dev 切 codex/<idea>
→ 实现、测试、固定 batch 数值检查
→ 合回 dev
→ 固化 YAML 和 experiment tag
→ 训练并更新实验结果/详细记录
→ 多 seed 和机制证据充分后合入 main
```

当前 working tree 已作为 `dev` 起点固化。不要为 B0/B1/B1T/B2 各建长期分支；只有代码行为不同才需要分支，运行身份用 YAML + tag 管理。
