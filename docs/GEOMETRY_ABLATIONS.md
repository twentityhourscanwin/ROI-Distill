# 2026-09-05 几何消融分支与实验状态

两个独立分支从 `dev@cec9362` 创建，代码实现已提交并推送到 origin；未启动完整训练、尚无新评测结果。后续运行以各分支实际 HEAD 和 resolved config 为准。

| 实验 | 分支 | 算法实现提交 | 变化 | 状态 |
| --- | --- | --- | --- | --- |
| C1 | [codex/proposal-center-only](https://github.com/twentityhourscanwin/ROI-Distill/tree/codex/proposal-center-only) | `f1af868` | 仅 proposal offset 中心移动，不扩尺寸、不加速度项 | 实现、CPU 验证完成；待训练 |
| R1 | [codex/box-radius-cap](https://github.com/twentityhourscanwin/ROI-Distill/tree/codex/box-radius-cap) | `02b2c6a` | 原 GT，半径限制在 1-2 格 | 实现、CPU 验证完成；待训练 |
| R2 | 同 R1 | `02b2c6a` | R1 仅启用原 Adaptive scaler | 配套配置已准备；待训练 |

算法实现提交不包含随后追加的文档提交。完整分支 HEAD 可用 `git rev-parse HEAD` 查询。两个分支没有合入 dev/main，也没有互相合并。

## 假设和对照

C1 用于检验 proposal 引导的监督位置调整是否有效；它不直接对齐或 warp 特征图，不能预先把 proposal 中心解释为已测得的特征中心。
C1 vs B1 仅引入 offset-only 中心移动；C1 vs 原 Adaptive B2 同时移除尺寸扩张和速度对位移的贡献，不能把后者称为只删除尺寸的单变量对照。
R1 vs B1 仅改变半径规则；R2 vs R1 仅启用 Adaptive；R2 vs 原 Adaptive B2 仅改变半径规则。

本轮参考 B1 20260829_225349（mAP 0.3881 / NDS 0.5047）和 Adaptive B2 20260830_104032（0.3922 / 0.5069）。Speed B2 20260904_103943 为固定中心速度扩张（0.3898 / 0.5029），与旧 Adaptive B2 区分命名。9/2 的 gt_nearest / proposal reuse 结果不参与本轮对照。
历史训练来自 dirty working tree，不能仅靠其 HEAD 完整还原；本轮新运行需固定干净提交、实验 tag、resolved config 和普通 last.ckpt 评测口径。

## 参数和检查

C1 保留 `mu=0.15` 和旧不对称方向 deadzone；位移来自限幅 proposal offset 的一半，尺寸不变，velocity contribution=0。匹配、q、response scope 和原半径规则不变。
R1/R2 使用 `max(1,min(2,int(raw_radius)))`，最小 3x3、最大 5x5 的软高斯窗口。下限 1 避免 radius=0 的单像素在 128->64 双线性插值中完全消失。它仍是离散圆形窗口，不严格贴合有向框。
C1 相关测试 113 passed，最后一次隔离非有限速度计算后定向复测 19 passed；R1/R2 相关测试 54 passed。均通过配置解析、真实 teacher checkpoint SHA256 核对及四个已有真实框的绘制抽查。这些是 CPU 检查，不代表完整训练已通过或已有指标收益。

## 文档和远端工作树

- [C1 机制、验证与运行命令](https://github.com/twentityhourscanwin/ROI-Distill/blob/codex/proposal-center-only/docs/CENTER_ONLY_ABLATION.md)
- [R1/R2 机制、验证与运行命令](https://github.com/twentityhourscanwin/ROI-Distill/blob/codex/box-radius-cap/docs/RADIUS_CAP_ABLATION.md)
- C1：`/mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL_center_only`
- R1/R2：`/mnt/workspace/guqiupeng/code/ROI_LABEL_DISTILL_radius_cap`

服务器为 `dsw-gqp-onwer-PPU`。运行时设置 `PYTHONPATH="$PWD"`，确保加载对应 worktree 的源码。data、ckpts 和 native extensions 的软链接为服务器本地依赖，不提交 Git；新 clone 需自行准备依赖。
每个 worktree 的 `outputs/geometry_ablation_smoke_20260905/summary.json` 为本地抽查产物，不随代码推送；关键样本结果已写入对应实验文档。正式运行后再补 run_id、train/checkpoint/evaluation 路径和指标，当前均未产生。
