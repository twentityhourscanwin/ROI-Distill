# 开发分支台账

本文件只记录 `dev` 以及从 `dev` 切出的开发分支。实验数值写入 `VAL_RESULTS.md` 和 `EXPERIMENT_LEDGER.md`，idea 分析写入 `IDEA_LOG.md`。

## 分支记录

| 分支 | 基线 / 分叉点 | 核心 commit | 目标与主要改动 | 影响范围 | 远端状态 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `dev` | `main@f5f1400` | `8c46dfe` | 集成 B 系列基线、配置系统、协作文档和 teacher–GT 匹配分析 | B0、B1、B1T、B2 的旧匹配基线 | 尚未设置 upstream | Active baseline |
| `codex/matching-analysis` | `dev@a0655e3` | `605c178`、`045c61d` | 提取 teacher–GT candidate graph，重做 train/val 匹配统计并记录结论 | 只增加分析工具和文档，不改变训练机制 | 尚未设置 upstream | 已合入 `dev@8c46dfe` |
| `codex/matching-gt-nearest` | `dev@8c46dfe` | `b230bac`、`7d4fd24`、`4c96ba6` | M1：每个 GT 独立选择阈值内同类最近 proposal，允许 proposal reuse；删除旧 scale-conditioned 分配机制 | B1/B1T/B2 已完成 seed 0；相对旧匹配 NDS 分别 `-0.0044/-0.0014/-0.0054` | `origin/codex/matching-gt-nearest` | Evaluated — negative, do not merge |

## 记录规则

1. 新开发分支必须从干净的 `dev` 切出，命名使用 `codex/<idea>`。
2. 建分支时立即新增一行，记录 `dev@<分叉 commit>`、目标、预计影响范围和初始状态。
3. 代码提交并 push 后补充核心 commit 与远端分支；不要只写“最新代码”。
4. 正式训练前保证 working tree clean，并在实验台账中记录分支名和精确 commit。
5. 状态统一使用：`Planned`、`In progress`、`Implemented, not run`、`Running`、`Evaluated`、`Merged`、`Abandoned`。
6. 分支合入 `dev` 后，记录 merge commit；已经产生正式实验的 commit 不改写历史。

## 新分支模板

| 分支 | 基线 / 分叉点 | 核心 commit | 目标与主要改动 | 影响范围 | 远端状态 | 当前状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `codex/<idea>` | `dev@<sha>` | `<sha>` | `<唯一机制变化>` | `<受影响配置/实验>` | `origin/codex/<idea>` | `Planned` |
