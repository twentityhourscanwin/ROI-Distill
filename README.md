# ROI-Distill

本工作树为独立消融分支，当前实现和运行命令见 [分支实验说明](docs/RADIUS_CAP_ABLATION.md)。下方历史状态保留作参考。

ROI-Distill 是一个面向 nuScenes 3D 检测的相机–LiDAR 跨模态蒸馏研究仓库。学生为多帧 BEVDepth，相同样本上的冻结 CenterPoint 作为训练教师；实验由 YAML 配置驱动。

## 当前状态

- 固定参考 B1：mAP `0.3881`，NDS `0.5047`。
- 领先候选 B2：mAP `0.3922`，NDS `0.5069`；等待 scaler 机制检查和多 seed 验证。
- 当前开发代码已固化在 `dev` 集成分支；新的算法 idea 从 `dev` 切短期分支。

## 文档

- [当前实验结果](docs/VAL_RESULTS.md)：简洁结果表。
- [实验详细记录](docs/EXPERIMENT_LEDGER.md)：配置、产物、完整指标和有效性。
- [Idea 迭代](docs/IDEA_LOG.md)：根据结果分析问题并规划下一步。
- [开发与运行说明](docs/README.md)：训练命令、当前 dev 设置和 Git 工作流。

## 快速开始

```bash
pip install -r requirements.txt
pip install -e .

python tools/train.py \
  --config configs/experiments/b1_teacher_value_no_scale.yaml
```

完整训练与评测命令见 [开发与运行说明](docs/README.md)。

## 仓库结构

```text
configs/base/          公共配置
configs/experiments/   实验入口配置
labeldistill/          数据、模型、蒸馏与训练实现
tests/                 配置和算法合同测试
tools/                 训练、评测和分析工具
docs/                  结果、实验、idea 与开发说明
```

数据、权重和训练输出不提交到 Git；正式实验必须由干净 commit/tag、resolved config、数据/教师哈希和产物路径共同标识。
