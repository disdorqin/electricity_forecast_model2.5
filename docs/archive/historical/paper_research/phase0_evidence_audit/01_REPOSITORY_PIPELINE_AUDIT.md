# 仓库与流水线审计

## 1. Git 与入口

| 项目 | 状态 | 证据/结论 |
|---|---|---|
| 仓库 | VERIFIED | `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5` |
| 分支/提交 | VERIFIED | `main` / `484117361b5d8f7f2c43ac993d43cabdddfe213a` |
| 审计前工作树 | VERIFIED | 已修改：`cli/parser.py`、`pipelines/sync_dataset_pipeline.py`、`utils/database_operate.py`；另有既存未跟踪文件，详见 manifest。 |
| 主入口 | VERIFIED | `main.py:12-24,43-129` 导入并分派 evaluate、sync、ledger predict/weight/fuse/classifier/full。 |
| 默认主流水线 | VERIFIED | CLI 默认 `pipeline=ledger-full`、`target=both`、RT cutoff 14、`epf_v1_mode=exact`：`cli/parser.py:90-140`。 |
| 配置 | VERIFIED | `models/SGDFNet/configs/*.yaml`、`models/RT916/config*.json`、`requirements.txt`、`pyproject.toml`、`.github/workflows/*`。 |
| 秘密 | VERIFIED | `.env` 含 DB host/name/user/password/port/timeout 类字段；未复制值；`.gitignore:20` 忽略该文件。 |

## 2. 预测任务定义

| 维度 | 状态 | 实现事实 |
|---|---|---|
| 目标 | VERIFIED | 日前为 `日前电价`，实时为 `实时电价`；aliases 见 `pipelines/ledger_predict.py:804-815`。 |
| 市场 | PARTIALLY VERIFIED | 文件名和同步代码指向山东电力市场；价格是否系统/分区/节点价未编码，定义 **UNKNOWN**。 |
| 分辨率 | VERIFIED | 活跃模型链路按小时 24 点；同步 CLI 可选 `hourly/15min`：`cli/parser.py:167-175`。 |
| 预测窗口 | VERIFIED | 业务日 D 的 24 个点为 `D 01:00` 到 `D+1 00:00`，分 `[01-08],[09-16],[17-24]`；`models/TimeMixer/pipeline.py:107-145`、`pipelines/emergency_fallback.py:97-145`。 |
| 单/多步 | VERIFIED | 单次输出整日 24 点；部分模型内部为三个 8 点 multi-output 子任务，不是滚动逐小时统一任务。 |
| 历史窗口 | PARTIALLY VERIFIED | TimeMixer `seq_len=168` 且默认训练 12 个月；RT916 按时段使用 8 个历史日段加目标 8 点；不同模型不一致。`models/TimeMixer/repro_pipeline.py:36-70`、`models/RT916/core.py:196-225`。 |
| forecast origin | PARTIALLY VERIFIED | ledger 记录 DA 截止 D-1（只到日期）和 RT D-1 14:00：`ledger_predict.py:143-147`。TimeMixer DA 默认 15:00，RT 14:00；RT916 DA 路径使用 D-1 24:00；SGDFNet 默认 YAML 15:00但 active override 为14:00。没有统一制度定义。 |
| 24/96 并存 | PARTIALLY VERIFIED | 24 点预测完整；96 点仅发现同步、数据审计与兼容脚本，未发现主模型/融合/提交链路。 |
| 运行方式 | VERIFIED | 混合：每日滚动训练/预测、历史 ledger 融合和静态 fixture；并非单一 day-ahead 或单一 real-time 定义。 |

## 3. 活跃流水线与最终值

`ledger-full` 的代码路径为：

1. **VERIFIED** `ledger-predict`：DA `lightgbm,timesfm,timemixer`；RT `timesfm,sgdfnet,timemixer,rt916`。`ledger_predict.py:44-46,193-250`。
2. **VERIFIED** 写 prediction/actual ledger；每个候选输出标准化为 `ds,y_pred`，再构成长表。`ledger_predict.py:610-682,700-748`。
3. **VERIFIED** `ledger-weight`：从过去 30 个 complete days 训练分时段 BGEW 权重；fixture 显示 DA 2,160 行、RT 2,880 行。`fusion/DailyLedgerGEF.py:39-119`。
4. **VERIFIED** `ledger-fuse`：逐小时严格加权，要求 24 行、合法小时且权重非零。`pipelines/ledger_fuse.py:97-159`。
5. **VERIFIED** `ledger-classifier`：写出 uncorrected 和 corrected 文件。`pipelines/ledger_classifier.py:68-102`。
6. **VERIFIED** bridge 以 `final_pred==1 and y_fused<=100` 为 mask，将实时预测硬置为 `-80`。`fusion/classifier_bridge.py:78-95`。
7. **VERIFIED** `ledger_full.py:342-355` 复制两份实时文件，但 `ledger_full.py:386-431` 的 submission 明确读取未修正 `realtime_final_predictions.csv`。**当前最终值不受分类器影响。**
8. **PARTIALLY VERIFIED** fixture 2026-02-24/25 的 submission 与 corrected 逐点一致、与 uncorrected 不一致，证明历史生成行为与当前代码发生漂移，不能据 fixture 推断当前生产值。

应急路径：**VERIFIED** `pipelines/emergency_fallback.py:26-189` 以同小时历史中位数按 7 天、30 天、全历史逐级回退并直接交付。这是可用性 fallback，不是相对 base predictor 的统计 no-harm 保证。

## 4. 模型与模块清单

| 组件 | 合约与目标 | 极端机制 | 状态/接入 | 工件与证据 | 主要风险 |
|---|---|---|---|---|---|
| LightGBM v1 | 小时表 -> 24 点；分 valley/solar/peak | solar 负价分类+回归/规则修正 | VERIFIED implemented + DA active | 发现 `best_model_日前电价.pkl`；RT 权重不确定 | exact adapter 不传 cutoff；模型选择/测试期不明 |
| TimesFM 2.5 | foundation model wrapper -> 24 点 | 无显式 signed-tail | VERIFIED implemented + DA/RT active | 2026-07 运行因本地模型/cache缺失失败 | exact 模式使用完整表；本地权重不可复现 |
| TimeMixer | 168 小时输入、3x8 输出 | 可选 regime/peak/spike-day calibration | VERIFIED implemented + DA/RT active；默认 calibration `none` | 最新 24 点 metrics 为 NaN | 默认 L1；`full_refit` 文档与实现不一致；future forecast 发布时间未知 |
| SGDFNet | 实际为 HistGradientBoosting 预测 RT-DA delta，再分段 bias | tail sample weight 可配；active 值 1.0 | VERIFIED implemented + RT active | 最新 metrics 为 NaN | 名称与算法类别易误解；尾权重 active 时未启用 |
| RT916 | SpikeGatedTimesNet，三段各 8 点 | SpikeResidualBranch、DynamicPeriodGate、TailWeightedHuberMSELoss | VERIFIED implemented + RT active | 有内部 validation metrics/weights | 阈值来自训练分位数但 actual-side cutoff 处理不完整；无冻结 OOS |
| Negative classifier | two-stage daily/lightgbm + second-stage classifier | 深负价分类；硬置 -80 | VERIFIED implemented/connected；**当前 final disconnected** | fixture 有 4/5 个实际改点 | fixed -80 与阈值泛化未证明；stage1 动态阈值搜索未调用 |
| DailyLedgerGEF | 候选历史误差 -> 分段权重 | floor50 SMAPE/MAE_percent 复合 | VERIFIED active | fixture 30 天候选指标 | 不是 OOF stacking；错误指标会偏置权重 |
| Emergency fallback | 历史同小时中位数 | 无极端专门机制 | VERIFIED active on failure | 2026-07-17 `DEGRADED_DELIVERED` | 只保证文件交付，不保证性能无害 |
| LSTM/RNN/MLP standalone | 搜索到分类器/内部网络和文献，但无独立活跃 backbone | 不适用 | UNKNOWN/未发现 | 无统一 artifact | 暂不能声称已完成跨这些 backbone 验证 |
| Uncertainty/interval | 未发现生产 quantile/interval/conformal contract | 条件 tail metrics 不是不确定性 | VERIFIED absent from active path | 无区间工件 | uncertainty-aware 假设尚未实现 |

Registry 证据：`runners/registry.py:7-18`；CPU/GPU 执行器：`runners/executor.py:13-15`。归档 `_archive/legacy_staged_pipeline` 和 legacy fusion 仅为 **DOCUMENTED BUT NOT VERIFIED**，不计入活跃性能证据。

## 5. 指标与损失

| 实现 | 状态 | 代码公式/处理 | 聚合与用途 |
|---|---|---|---|
| raw SMAPE | VERIFIED | `mean(200*abs(y-p)/max(abs(y)+abs(p),eps))` | `evaluate_pipeline.py:8-17`，全行均值 |
| floor50 SMAPE | VERIFIED | 先分别执行 `y=max(y,50), p=max(p,50)`，再算 SMAPE | evaluation、fusion、TimeMixer、TimesFM；把全部负价和 0-50 压成 50 |
| MAE/RMSE | VERIFIED | `mean(abs(e))` / `sqrt(mean(e^2))` | SGDFNet/RT916/TimeMixer 等，通常点级平均 |
| Fusion MAE% | VERIFIED | `100*MAE/max(median(abs(max(y,50))),50)` | `fusion/DailyLedgerGEF.py:39-119`；与通常百分比误差不同 |
| Fusion objective | VERIFIED | `0.7*SMAPE_floor50 + 0.3*MAE_percent`；指数更新 eta=.8，weight floor=.03 | 3 个 8 小时段、过去 30 complete days |
| RT916 SMAPE | VERIFIED | floor50 SMAPE 返回比例而非百分数 | `models/RT916/core.py:756-800`；尺度与其他模块相差约 100 倍 |
| SGDFNet tail | VERIFIED | 在被评估 frame 上以 `abs(RT-DA)` 分位数定义 tail，再算条件 MAE/RMSE/SMAPE | `models/SGDFNet/metrics.py:89-112`；描述可用，不可从 test 反推阈值 |
| classifier | VERIFIED | precision/recall/F1/F2；stage2 在 gray-train 的后 20% 从 0.3..0.8 选最大 F2 | `models/classifier/cascade_daily.py:301-356,850-908` |
| RT916 loss | VERIFIED | Huber/MSE 基础上，对训练低/高分位数 tail 与剧烈差分加权 | 对正负尾对称，不是 signed router |

### 数值反例

按代码的 floor50：

| `y_true` | `y_pred` | MAE | raw SMAPE | floor50 SMAPE | 被隐藏的问题 |
|---:|---:|---:|---:|---:|---|
| -80 | -70 | 10 | 13.33% | 0% | 深负价幅度误差 |
| -80 | 50 | 130 | 200% | 0% | 符号翻转和巨大误差 |
| 0 | 10 | 10 | 200% | 0% | 近零误差 |

**VERIFIED**：floor50 只能作为某个业务自定义分数，不能替代负价/近零场景的误差指标；论文必须同时报告未裁剪 MAE/RMSE、signed-tail 条件指标和事件级指标。

## 6. 现有实证证据

| 证据 | 状态 | 可支持内容 | 不可支持内容 |
|---|---|---|---|
| `tests/fixtures/ledger_full/*` 2026-02-24/25/26 | PARTIALLY VERIFIED | 静态格式、30 日权重输入、历史 corrected submission 行为 | 正式 OOS 性能、当前代码成功运行、因果改善 |
| fixture candidate metrics | VERIFIED as artifact | 30 日 ledger 的候选分数可追溯 | 独立 test；阈值未受 test 影响 |
| 2026-07-16/17 run manifests/logs | VERIFIED | TimesFM 本地资源失败、部分模型 NaN、权重无完整日、最终降级交付 | 改善或稳定性 |
| RT916 latest metrics | PARTIALLY VERIFIED | 单次内部 chronological validation 有数值 | 跨日 walk-forward、冻结 test、跨市场泛化 |
| SGDFNet/TimeMixer latest metrics | VERIFIED | 24 行结果为 NaN/空 tail | 任何性能结论 |
| release/profile 文档中的年份分数 | DOCUMENTED BUT NOT VERIFIED | 配置中存在声明 | 缺少可重建预测、标签、split 和统一 metric 运行 |

**结论：PARTIALLY VERIFIED**。仓库证明“组件存在并曾部分执行”，没有证明“极端价格预测已被改善”。流水线完成、缓存命中或提交文件存在均不等价于经验提升。
