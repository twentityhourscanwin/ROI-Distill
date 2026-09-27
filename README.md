# ROI-Distill

ROI-Distill 研究 nuScenes 3D 检测中的相机–LiDAR 特征蒸馏：多帧 BEVDepth 学生在训练时利用冻结 CenterPoint 教师的局部 BEV 监督。方法定义、论文主张和实验结论以 DSW 共享文档中的证据链为准；实验分支的结果不表示代码已经合入 dev。

## 文档入口

1. [研究文档首页](docs/README.md)
2. [方法定义](docs/方法定义.md)
3. [主张与证据](docs/主张与证据.md)
4. [实验矩阵](docs/实验矩阵.md)：按稳定实验 ID 进入日期目录，查看实验设计、事实和对比推理。
5. [项目工作约定](AGENTS.md)：DSW 单仓、分支、验证及共享文件规则。

docs/、tests/ 和辅助 tools/ 跨分支共用且不纳入 Git；tools/train.py 与 tools/evaluate.py 是受版本控制的正式入口。切换代码分支不会切换共享文档，新环境还需单独恢复它们。当前唯一工作仓库和运行入口见 AGENTS.md。

## 仓库结构

- configs/base/：公共配置。
- configs/experiments/：实验入口配置。
- labeldistill/：数据、模型、蒸馏与训练实现。
- tests/：共享验证文件。
- tools/：正式训练/评测入口及共享辅助工具。
- docs/：论文知识文档、按日期组织的实验证据与历史资料。

数据、教师权重、训练输出不提交到 Git；正式实验身份须结合固定代码版本、实际配置、输入身份和产物记录核对。
