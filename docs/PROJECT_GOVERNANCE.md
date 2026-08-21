# 项目工程治理规范

> status: active
> 日期：2026-08-16
> 验证依据：`AGENTS.md`、`docs/RUNBOOK.md`、现有回归脚本

本规范把大型数据/模型项目中常见的发布流程压缩成适合本项目的最小闭环，解决“代码能跑但无法证明、产物能交付但无法追溯、文档越来越乱”三个问题。

## 1. 变更分级

| 级别 | 典型变更 | 强制动作 |
|---|---|---|
| A 数据/防泄漏 | 数据同步、特征、cutoff、实际值使用 | 先读数据真实性规则；运行数据质量、泄漏和96点完整性检查 |
| B 模型/融合 | 模型池、权重、融合、分类器 | 运行模型池、门控、交付稳定性回归；记录 seed、resolution、cutoff |
| C 工程/文档 | 入口、依赖、目录、文档 | 运行 CLI/编译检查；更新 `README.md` 或 `docs/README.md` |

跨级变更按最高级别执行。禁止把 A/B 级改动只当作“重构”跳过验证。

## 2. 合并前质量门

生产代码进入主分支前至少满足：

1. `python -m py_compile` 对改动模块通过；
2. CLI 参数和范围解析回归通过；
3. 交付稳定性、目标日 NaN、同步流程回归通过；
4. 模型候选池与 `fusion/model_pool.py` 一致；
5. 融合输出包含 `model_quality_gate.csv`、`fused_debug.csv` 和 manifest 门控字段；
6. 不提交账本、runs、日志、模型权重等生成产物；
7. 依赖变更同时更新根 `requirements.txt`，并注明 Python/Torch/CUDA/JAX 基线。

任何一项不满足，都只能作为实验分支或明确标注的降级交付，不能宣称为正常交付。

## 3. 可复现记录

每次训练、回测或交付的 manifest 至少记录：

```text
git_commit
python_version
torch_version / cuda_version
jax_version
data_snapshot / data_max_timestamp
seed
resolution
realtime_cutoff_hour / last_visible_period
weight_learner
weight_prune_threshold
active_models / pruned_models
delivery_status
```

缺少上述字段时，结果可以用于调试，但不作为正式性能结论。

## 4. 失败、降级与回滚

- P0 数据真实性或泄漏失败：阻断模型写入预测账本；保留审计 JSON 和错误日志。
- P1 模型质量门失败：保留最高权重模型作为安全兜底，并在交付报告中标记降级。
- P2 单模型不可用：只从已通过门控且有预测的候选中重新归一化；不使用 `fillna(0)`。
- 正常链路失败但存在应急输出：交付状态必须为 `DEGRADED_DELIVERED`，不能伪装成 `NORMAL`。
- 修复后先使用同一数据快照重跑回归，再恢复正常交付；不直接覆盖历史基线。

## 5. 文档与实验生命周期

默认更新已有负责文档，不新增文件。只有确实无法归入现有领域时，智能体才可在回复中申请新增文档；申请获得用户明确同意后，才创建文件并登记在 `docs/README.md`。调研、实验记录、旧验收和 Agent 过程材料进入 `docs/archive/`，并保留原始内容与归档日期。实验结果只进入正式结论文档的条件是：

- 数据快照、代码版本和环境可复现；
- 指标口径、时间切分和防泄漏边界明确；
- 至少有基线、对照或失败记录；
- 没有把降级输出当作正常模型性能。
