# EFM3 文档入口与职责索引

> status: active
> 日期：2026-09-21
> 规则总文档：`DOCUMENT_ARCHITECTURE.md`

## 当前生效文档

当前生产、交接和 AI 阅读只以以下文档为准。普通内容必须更新对应文档，不得为一次修改新建报告。

| 顺序 | 文档 | 负责内容 |
|---|---|---|
| 1 | 根目录 `README.md` | 项目目标、快速开始、用户可见能力和当前状态 |
| 2 | `DOCUMENT_ARCHITECTURE.md` | 文档分域、目录职责、分支边界、新增审批和 AI 阅读顺序 |
| 3 | `PROJECT_LAYOUT.md` | 代码目录、文件夹分工、源码与产物边界 |
| 4 | `RUNBOOK.md` | 环境、同步、smoke、范围运行、交付、回归和部署 |
| 5 | `DATA_CONTRACT_96.md` | 24/96 分辨率、字段、业务日、数据质量和完整性 |
| 6 | `LEAKAGE_AUDIT_96.md` | cutoff、信息可得性、actual/fcast 使用和防泄漏门控 |
| 7 | `OUTPUT_CONVENTION.md` | ledger、runs、submission、manifest 和命名规则 |
| 8 | `PROJECT_GOVERNANCE.md` | 变更分级、质量门、复现、降级和回滚 |
| 9 | `metrics_calculation.md` | 模型指标、业务指标和统一计算口径 |
| 10 | `FeatureStore_特征预计算_设计.md` | 24/96 特征物化、缓存、切片和价差实验迁移计划；formal96 当前只作 compatibility/shadow 参考 |
| 11 | `SERVER_96_STANDARD_SOP.md` | 服务器 Codex 一步一步标准流程：Python/CUDA/TimesFM、ledger 30日学习器、full-source、range、audit、每日生产 |
| 12 | `SERVER_96_DEPLOYMENT_BACKFILL.md` | 服务器部署与历史接续的详细补充 Runbook；遇到异常或需要展开细节时读取 |
| 13 | `research_innovation_candidates/README.md` | 科研创新候选库；只登记具有长期研究价值、已有实验验证路线的候选，不作为生产契约 |

## 文档变更规则

1. **默认不新增文档。** 能归入现有领域的内容，直接增补对应负责文档。
2. **确需新增时必须先请求用户批准。** 智能体只能在回复中说明：文档名称、所属领域、为什么不能合并、预计维护内容和验证方式；在用户明确同意前不得创建文件。
3. **未获批准不得用“临时文档”绕过规则。** 临时结论写入对应文档的“待确认/实验记录”小节，或保留在回复中。
4. **代码行为变化必须同步更新文档。** 若代码、测试和文档冲突，以代码和测试为事实，并在同一任务修正文档。
5. **新文档若获批准，首部必须写 `status`、日期、责任领域和验证依据，并登记到本文。**

## 历史和实验材料

历史材料不删除，只归档，不能作为生产规则的唯一依据：

- `archive/historical/`：旧验收、旧部署、旧数据质量、旧范围运行和过程记录；2026-09 formal96 改造/收尾/问题账本/PhaseM1-M2/HistoricalProxy 执行稿统一收敛到 `archive/historical/formal96-2026-09-closeout/`；
- `archive/historical-audits-2026-07/`：被当前契约替代的历史审计和迁移方案；
- `archive/agent-research-2026-08/`：Agent 调研、实验、专项设计和自动化方案旧稿；当前价差专项研究计划见 `archive/agent-research-2026-08/价差预测_论文复现选择与执行计划_20260822.md`（active research plan，非生产规则）。
- `research_innovation_candidates/`：用户批准建立的长期科研创新候选库；当前训练方向见 `research_innovation_candidates/training/privileged_information_distillation.md`。这里保存“可能成为论文创新点”的长期机制设计，具体运行结果仍只写实验产物，未验证候选不得视为生产规则。
- `outputs/experiments/01_spread_24/spread_forecast_24_96_chain_v2/cycles/cycle_88_numeric_spread_da_minus_rt/docs/06_DIRECTION_MAGNITUDE_DECOMPOSITION_DESIGN.md`：用户于 2026-09-06 批准建立的 Cycle88 实验设计；只用于验证“方向分类 + 幅值回归”的单变量任务分解，不是生产规则，final holdout 在模型选择前保持封闭。

新实验只允许放在 `scripts/experiments/` 和 `outputs/experiments/`；实验结论只有在数据、代码、环境和指标口径可复现后，才能摘要回写到对应权威文档。24点价差历史事实复盘区位于 `outputs/experiments/01_spread_24/00_history_review/`，仅用于现有实验的阶段化索引、结果摘录与证据追溯，不作为生产契约或新实验路线。

## AI 推荐阅读顺序

1. 根目录 `AGENTS.md`；
2. 本文件；
3. `DOCUMENT_ARCHITECTURE.md`；
4. 根据任务读取唯一负责文档；服务器部署/历史接续任务先读 `SERVER_96_STANDARD_SOP.md`，再按需读 `SERVER_96_DEPLOYMENT_BACKFILL.md`；
5. 最后核对代码、测试和 manifest；
6. 只有追溯历史时才读取 `archive/`。

## 数据与输出域（当前约定）

- 24 点 canonical 输入：`data/24/canonical/`。
- 96 点权威实际：`data/96/authoritative/pmos_96_全量.csv`，只用于 actual
  交叉验证，不包含可替代价格模型宽表的语义。
- 96 点模型输入：`data/96/model_input/`；数据库原始镜像：`data/96/remote/`。
- FeatureStore 仅保留为显式兼容/候选 profile：24 点可用 `outputs/24/feature_store/{cache,ledger,runs}/`；96 点当前磁盘上已无持久 `outputs/96/feature_store/`，只有显式选择 compatibility profile 时才会重建。
- 96 历史 FeatureStore/server 资产已分别进入 `outputs/archive/server_backtest_96/` 与 `outputs/archive/legacy_96/`；旧同步/缓存目录已移入 `outputs/archive/legacy_sync/`，均不再作为正式写入目标。
- 当前 96 点正式 façade（`python main.py --96 DATE`）只写
  `outputs/96/{ledger,runs,cache,runtime,sync}/`，执行四阶段
  `ledger_predict → ledger_weight → ledger_fuse → final_outputs`；
  `ExtremePriceClf` 在该 profile 标记为
  `disabled_by_production_policy`，仅保留 legacy/shadow/replay 能力。

任何数据迁移或新增 96 点快照后，先运行：

```powershell
python scripts/tests/check_96_vs_24_actual.py
```
