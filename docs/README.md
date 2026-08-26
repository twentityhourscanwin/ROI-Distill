# 文档索引与维护规则

本目录只维护三个核心事实面：当前状态、设计演进、实验事实。技术参考和旧材料分别进入 `reference/` 与 `archive/`，不再与当前结论混放。

## 三份核心文档

1. [CURRENT_BASELINE.md](CURRENT_BASELINE.md) — 当前唯一被接受的系统快照。只在某个实验被正式采纳时更新，不记录尝试历史。
2. [DESIGN_LOG.md](DESIGN_LOG.md) — 设计决策与网络演进。按决策编号追加，记录为什么改、如何验证、最终是否采纳。
3. [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md) — 实验运行唯一台账。按实验 ID 追加；失败、无效和中止的运行同样保留。

## 真相优先级

发生冲突时按以下顺序判断：

```text
指定 Git commit/tag 下的代码与 resolved_config.yaml
  > EXPERIMENT_LEDGER 中该次运行的事实记录
  > CURRENT_BASELINE 当前摘要
  > DESIGN_LOG 的设计说明
  > reference 技术参考
  > archive 历史材料
```

文档描述不能替代可执行配置。每次正式运行都应保存 `resolved_config.yaml`、环境快照、Git SHA、教师 checkpoint SHA256 和数据清单。

## Git 与实验工作流

- `main`：始终保持可运行，保存当前代码主线。
- `codex/<feature>`：只用于短期代码或网络改动；通过测试和 review 后合并并删除。
- `configs/experiments/*.yaml`：配置级消融的身份，不为 B0/B1/B1T/B2 建长期分支。
- `exp/YYYYMMDD-<experiment>-s<seed>`：正式运行 tag，指向运行所用提交。
- checkpoint、日志和数据放在对象存储/NAS；Git 只保存哈希、路径、指标和结论。

标准流程：

```text
Issue/假设
  → feature branch（仅代码变化时）
  → PR + tests + DESIGN_LOG
  → merge main
  → 固化 experiment YAML + Git tag
  → 运行并登记 EXPERIMENT_LEDGER
  → 结果被采纳后更新 CURRENT_BASELINE
```

## 更新规则

### CURRENT_BASELINE

- 必须给出 baseline ID、配置路径、Git SHA/tag、教师与数据版本。
- 只描述当前采用的系统，不追加流水账。
- 未跑完或不可比较的实验不能成为 baseline。

### DESIGN_LOG

- 使用 `DNNN` 编号；已有条目不删除。
- 状态仅使用 `Proposed`、`Implemented`、`Validated`、`Rejected`、`Superseded`。
- 设计分支即使被删除，决策条目仍保留并链接相关 commit/实验。

### EXPERIMENT_LEDGER

- 使用唯一 run ID；建议 `<实验>-<日期>-s<seed>`。
- 提交运行记录后原则上只追加勘误，不覆盖旧指标。
- 至少填写：父 baseline、唯一变化、config、Git SHA/tag、seed、数据/教师哈希、硬件、状态、指标、产物路径和结论。
- 不满足可比条件的结果必须标为 `Historical` 或 `Invalid`，禁止混入当前横向对比。

## 其他资料

- `reference/CONFIG_CONTRACT.md`：配置字段、代码消费者与合同测试。
- `archive/`：旧设计、旧命令、旧统计与不再有效的实验结果；仅用于追溯。
