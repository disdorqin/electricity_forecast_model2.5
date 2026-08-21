# EFM3 文档入口与职责索引

> status: active
> 日期：2026-08-16
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
| 10 | `24点硬编码与模型入口审计_20260816.md` | 当前入口、模型池和 24/96 参数化审计 |
| 11 | `FeatureStore_特征预计算_设计.md` | 24/96 特征物化、缓存、切片和价差实验迁移计划 |

## 文档变更规则

1. **默认不新增文档。** 能归入现有领域的内容，直接增补对应负责文档。
2. **确需新增时必须先请求用户批准。** 智能体只能在回复中说明：文档名称、所属领域、为什么不能合并、预计维护内容和验证方式；在用户明确同意前不得创建文件。
3. **未获批准不得用“临时文档”绕过规则。** 临时结论写入对应文档的“待确认/实验记录”小节，或保留在回复中。
4. **代码行为变化必须同步更新文档。** 若代码、测试和文档冲突，以代码和测试为事实，并在同一任务修正文档。
5. **新文档若获批准，首部必须写 `status`、日期、责任领域和验证依据，并登记到本文。**

## 历史和实验材料

历史材料不删除，只归档，不能作为生产规则的唯一依据：

- `archive/historical/`：旧验收、旧部署、旧数据质量、旧范围运行和过程记录；
- `archive/historical-audits-2026-07/`：被当前契约替代的历史审计和迁移方案；
- `archive/agent-research-2026-08/`：Agent 调研、实验、专项设计和自动化方案旧稿。

新实验只允许放在 `scripts/experiments/` 和 `outputs/experiments/`；实验结论只有在数据、代码、环境和指标口径可复现后，才能摘要回写到对应权威文档。

## AI 推荐阅读顺序

1. 根目录 `AGENTS.md`；
2. 本文件；
3. `DOCUMENT_ARCHITECTURE.md`；
4. 根据任务读取唯一负责文档；
5. 最后核对代码、测试和 manifest；
6. 只有追溯历史时才读取 `archive/`。

## 数据与输出域（当前约定）

- 24 点 canonical 输入：`data/24/canonical/`。
- 96 点权威实际：`data/96/authoritative/pmos_96_全量.csv`，只用于 actual
  交叉验证，不包含可替代价格模型宽表的语义。
- 96 点模型输入：`data/96/model_input/`；数据库原始镜像：`data/96/remote/`。
- 新 FeatureStore 链路：`outputs/24/feature_store/{cache,ledger,runs}/` 或
  `outputs/96/feature_store/{cache,ledger,runs}/`。
- 原链路仍保留在 `outputs/ledger*`、`outputs/runs*`，旧同步/缓存目录已移入
  `outputs/archive/legacy_sync/`，不再作为写入目标。

任何数据迁移或新增 96 点快照后，先运行：

```powershell
python scripts/tests/check_96_vs_24_actual.py
```
