# ROI-Distill

ROI-Distill 是一个面向 nuScenes 3D 检测的相机–LiDAR 跨模态蒸馏研究仓库。当前代码使用 YAML 驱动实验，学生为多帧 BEVDepth，相同样本上的冻结 CenterPoint 作为教师，并支持 proposal–GT 匹配、质量加权的特征蒸馏、响应蒸馏以及 ROI 几何消融。

## 当前状态

- 当前候选 baseline：**B1 — teacher value + circular Gaussian + no scale**。
- B0、B1、B1T、B2 已实现并通过合同测试，但都需要在修正后的教师输入与数据口径下重新训练，当前不能引用旧结果作为正式结论。
- 2026-08-26 的代码快照：`d27eed6`。
- 权重、数据集和训练输出不进入 Git；每次正式运行必须在实验台账中记录 Git SHA、配置、数据与教师哈希及产物位置。

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .

python tools/train.py \
  --config configs/experiments/b1_teacher_value_no_scale.yaml \
  runtime.gpus=8 \
  runtime.batch_size_per_device=4 \
  runtime.precision=16
```

按环境修改 GPU 数量和单卡 batch size。正式实验前请先核对教师 checkpoint 与 nuScenes info 文件；完整口径见 [当前基线](docs/CURRENT_BASELINE.md)。

## 文档入口

- [CURRENT_BASELINE.md](docs/CURRENT_BASELINE.md)：当前被接受的网络、数据、损失和训练口径。
- [DESIGN_LOG.md](docs/DESIGN_LOG.md)：网络与算法决策、分支演进及待验证问题。
- [EXPERIMENT_LEDGER.md](docs/EXPERIMENT_LEDGER.md)：实验运行、指标、产物与结论的唯一台账。
- [docs/README.md](docs/README.md)：文档维护和 Git 工作流。
- [CONFIG_CONTRACT.md](docs/reference/CONFIG_CONTRACT.md)：YAML 字段到代码消费者与测试的对应关系。

## 仓库结构

```text
configs/base/          可复用的公共配置
configs/experiments/   不可变的实验入口配置
labeldistill/          数据、模型、蒸馏与训练实现
tests/                 配置和算法合同测试
tools/                 训练、评测、统计与数值快照工具
docs/                  当前文档、参考资料和历史归档
```

`data/`、`ckpts/`、`outputs/` 只保存在训练环境或对象存储中，不应提交到 GitHub。

## 实验与 Git 约定

实验配置不是长期分支。配置消融应在同一代码提交上使用不同 YAML；只有算法或网络代码发生变化时才创建短期 feature branch 和 PR。正式运行用 Git tag 固化，例如：

```text
exp/20260826-b1-s0
```

推荐流程：Issue → `codex/<feature>` → PR/测试 → 合并 `main` → 固化 YAML 和 tag → 训练 → 更新实验台账；只有验证通过并被采纳的结果才能更新当前基线。

## 致谢

本项目的早期实现参考了 [LabelDistill](https://github.com/sanmin0312/LabelDistill) 与 [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)。仓库中的 ROI 蒸馏、配置化实验与后续改动请以当前代码和文档记录为准。
