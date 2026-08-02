# Phase-0 证据审计执行摘要

审计对象：`electricity_forecast_model2.5`；Git `main@484117361b5d8f7f2c43ac993d43cabdddfe213a`。审计时间见 `AUDIT_MANIFEST.json`。状态词仅表示本轮可追溯证据强度，不表示方法优劣。

## 项目诊断

- **VERIFIED**：仓库具备可调用的 24 点日级预测链路：日前使用 `lightgbm/timesfm/timemixer`，实时使用 `timesfm/sgdfnet/timemixer/rt916`，之后以 30 个完整历史日训练分时段融合权重，再运行负电价分类修正和交付兜底。证据：`pipelines/ledger_predict.py:44-46`、`pipelines/ledger_full.py:51-199`、`fusion/DailyLedgerGEF.py:39-119`。
- **VERIFIED**：当前 `ledger_full` 代码生成 submission 时读取未修正的 `realtime_final_predictions.csv`，而不是 `realtime_final_predictions_corrected.csv`；因此分类器修正按当前代码不会进入最终提交值。证据：`pipelines/ledger_classifier.py:68-102`、`pipelines/ledger_full.py:342-355,386-431`。
- **PARTIALLY VERIFIED**：归档 fixture 的 submission 反而与 corrected 文件逐点一致，说明归档产物与当前代码行为不一致；fixture 只能证明某次历史运行，不能证明当前提交链路。
- **VERIFIED**：小时数据中的负价/深负价高度集中在 09:00-16:00；实时负价事件中 77.08%、实时低于 -50 的事件中 77.53% 落在该时段。相反，以 `>500` 作审计用高价切片时仅 14.11% 落在 09:00-16:00。故“该时段集中了所有正负极端事件”不成立；正式正尖峰定义仍为 **UNKNOWN**。
- **VERIFIED**：代码已存在负价分类、硬修正、RT916 尾部加权/尖峰残差分支、SGDFNet 残差建模、TimeMixer 可选 regime/peak 校准、融合与应急回退。它们不是一套统一的 signed-tail、uncertainty-aware、no-harm 框架，也没有跨 backbone 的严格样本外证据。

## 五项最高风险

1. **VERIFIED - 信息边界风险**：默认 `epf_v1_mode=exact`；LightGBM adapter 计算 cutoff 但未传入底层，TimesFM 在 exact 模式直接接收完整数据。`runners/adapters/lightgbm_v1.py:99-128`；`runners/adapters/timesfm_v1.py:108-125`；`cli/parser.py:133-140`。
2. **VERIFIED - 最终值断链**：当前代码忽略 classifier corrected 文件，生产输出无法支持“分类器改善最终预测”的论断。
3. **VERIFIED - 指标失真**：floor-50 SMAPE 会把 `(-80,-70)`、`(-80,50)`、`(0,10)` 均处理为零误差，掩盖负价、符号翻转和近零误差。`pipelines/evaluate_pipeline.py:8-17`。
4. **PARTIALLY VERIFIED - 证据不足**：fixture 是静态验收/烟雾证据；2026-07-16/17 的运行记录包含 TimesFM 本地模型缺失、NaN 输出、融合无历史日和降级交付，不能形成论文级性能结论。
5. **UNKNOWN - 市场与特征语义**：价格是系统、分区还是节点价，时区/DST、各 forecast/actual 字段的发布时间、D-1 14:00/15:00 的制度性依据均未由数据字典或 as-of 快照证明。

## 已核实与仍未知

| 项目 | 状态 | 结论 |
|---|---|---|
| 24 点任务 | VERIFIED | 预测自然日式业务窗口 `D 01:00` 至 `D+1 00:00`，分 3 个 8 点段。 |
| 96 点任务 | PARTIALLY VERIFIED | 15 分钟同步、审计和数据契约存在；未发现与主模型/融合/提交等价的 96 点预测链路。 |
| 正尖峰标签 | UNKNOWN | 代码中的 `>300` 主要用于分段/峰态，不等价于研究标签；没有冻结阈值或事件定义。 |
| 负价规模 | VERIFIED | 小时 RT 有效值 39,768，负价 5,327（13.3952%），低于 -50 为 4,433（11.1472%）。 |
| 严格 OOS | UNKNOWN | 未找到冻结测试集、预注册阈值期和全链路 walk-forward 结果。 |
| 不确定性输出 | VERIFIED | 生产链路未发现 quantile/interval/conformal 输出；仅有条件 tail 指标和 Vault 候选笔记。 |
| Vault 文献覆盖 | PARTIALLY VERIFIED | 主要是 load/PV forecasting；仅定位到一个 day-ahead wholesale price 候选摘要，未形成负电价/正尖峰专门文献证据。 |

## 研究实验就绪判断

**PARTIALLY VERIFIED：工程试验可启动，论文级系统实验尚未就绪。** 可复用模型与流水线较丰富，但必须先冻结 forecast origin、建立 feature as-of contract、修复/选择最终 corrected 值、定义正负极端事件、划定验证/测试期，并将所有 backbone 接入同一无泄漏 walk-forward 协议。当前材料不能支持新颖性声明或最终创新命名。

## 安全与完整性

- **VERIFIED**：仓库 `.env` 含数据库连接类秘密字段；本报告不记录任何值。建议撤销/轮换可能暴露的凭据，仅保留 `.env.example`，并确认历史 Git 对象中不存在秘密；`.gitignore:20` 已忽略 `.env`。
- **VERIFIED**：审计过程未运行训练、未下载数据、未提交代码；一次读取/导入路径意外写回的 8 个 ledger 文件已精确恢复到 `HEAD`，最终工作树仅保留审计前/并发可见的用户改动与本报告目录。`scripts/crawler/run_crawler.py` 及其 auth 文件未被本审计触碰。
