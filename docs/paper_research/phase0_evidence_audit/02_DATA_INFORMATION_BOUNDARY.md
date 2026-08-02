# 数据与信息边界审计

## 1. 数据集清单（仅聚合统计）

| 数据集 | 市场/期间/分辨率 | 行数与质量 | 极端事件 | 状态 |
|---|---|---|---|---|
| `data/shandong_pmos_hourly.csv` | 山东；2022-01-01 01:00 至 2026-07-18 00:00；1h | 39,816 行；时间戳重复 0；总缺失 552；23 列；naive datetime | RT 有效 39,768：负价 5,327 (13.3952%)，`<-50` 4,433 (11.1472%)；DA 有效 39,792：负价 4,427 (11.1254%)，`<-50` 3,521 (8.8485%) | VERIFIED |
| `data/shandong_pmos_96.csv` | 山东；2026-07-17 00:15 至 2026-07-19 00:00；15min | 192 行；重复 0；缺失 1,248；无价格列 | 无法统计价格事件 | VERIFIED |
| `data/remote_db_96/epf_market_data_96.parquet` | 表内 2022-01-01 00:15 至 2026-07-30 00:00；15min | 160,416 行；重复 0；缺失 227,328；无目标价格列 | 无法统计价格事件 | VERIFIED |
| `data/remote_db_96/epf_unit_data_96.parquet` | 表内 2022-01-01 00:15 至 2026-07-19 00:00；15min | 159,360 行；时间戳重复 0；缺失 0；含 unit clearing price 字段 | 未复制/展示任何单位级原始值 | VERIFIED |

编码：小时/96 CSV 可由 GB18030 正确读取。**UNKNOWN**：所有表均未携带 IANA timezone 或 UTC offset；未发现 DST 处理。不能仅凭“山东”把时区处理判定为正确。

## 2. 极端事件按小时聚合

以下阈值仅用于审计描述，不是建议阈值，也不从 test 期选择模型参数。

| 目标/切片 | 全期事件数 | 09:00-16:00 数 | 时段占比 | 状态 |
|---|---:|---:|---:|---|
| RT `<0` | 5,327 | 4,106 | 77.08% | VERIFIED |
| RT `<-50` | 4,433 | 3,437 | 77.53% | VERIFIED |
| RT `>500` | 5,820 | 821 | 14.11% | VERIFIED |
| DA `<0` | 4,427 | 3,648 | 82.40% | VERIFIED |
| DA `<-50` | 3,521 | 2,927 | 83.13% | VERIFIED |
| DA `>500` | 5,071 | 约 10%-15% 时段占比 | 未作为正式标签 | PARTIALLY VERIFIED |

RT 分位数：q01=-80，median=369.7862，q99=916.3289；DA q01=-80.8624，median=372.93，q99=772.8568。**VERIFIED**。高价的时段分布与负价不同，因此将 09-16 预设为统一 signed-tail 路由窗口会引入先验偏差。

## 3. Forecast origin 与业务日

- **VERIFIED**：主链路以目标日期 `D` 生成 24 点，时间语义是 `D 01:00 ... D+1 00:00`，不是标准的 `D 00:00 ... 23:00`。
- **VERIFIED**：`ledger_predict.py:143-147` 记录 DA cutoff 为 `D-1` 日期、RT cutoff 为 `D-1 14:00`。
- **PARTIALLY VERIFIED**：TimeMixer `models/TimeMixer/repro_pipeline.py:36-68,237-238` 使用 DA 15:00/RT 14:00；SGDFNet active pipeline 将 YAML 的 15:00 override 到 14:00；RT916 DA 代码使用 `D-1 24:00`。forecast origin 并未统一。
- **UNKNOWN**：D-1 14:00/15:00 是否对应真实市场发布时间、哪些外生 forecast 在该时刻已发布、是否允许后续修订。代码没有 as-of version。
- **UNKNOWN**：自然日、交易日和 `24:00` 的正式市场定义。代码给出实现映射，但没有制度数据字典证明。

## 4. 特征可用性矩阵

| 特征/组 | 来源与时间语义 | lag/shift | 最早可用点 | 在声称 origin 有效？ | 泄漏风险 | 证据/状态 |
|---|---|---|---|---|---|---|
| 历史 DA/RT 价格 | 小时表，event timestamp | TimeMixer 仅 past<=cutoff；RT916 构造 target lags | 取决于出清发布时间 | PARTIALLY VERIFIED | exact adapters 和 RT916 future frame 风险 | TimeMixer `repro_pipeline.py:282-335`; RT916 `core.py:241-296,658-666` |
| 目标日 DA 价格作 RT 特征 | 内部 DA 预测或 raw DA | 模型各异 | DA 出清后 | PARTIALLY VERIFIED | raw 实际 DA 与预测 DA 混用会污染 | `ledger_predict.py:675-682`; manifests 的 `da_feature_source` |
| 负荷 forecast | `用电负荷预测` 等 | target-day raw forecast | 发布时间未编码 | UNKNOWN | 修订版/事后值风险 | TimeMixer `repro_pipeline.py:140-191,338-388` |
| 实际负荷 | actual-side 列 | SGDFNet cutoff 后替换为 forecast；RT916 未对全部 actual-side future 列作同等替换 | 实际发生后 | SGDFNet PARTIALLY VERIFIED；RT916 UNKNOWN | RT916 目标日 realized exogenous 可见 | `protocol_b_cutoff.py:160-215`; RT916 `core.py:241-296` |
| 风/光 actual | actual-side realized generation | 同上 | 实际发生后 | UNKNOWN/无效于 D-1 origin | 高 | 同上 |
| 风/光 forecast | forecast 列 | target-day raw | 发布时间/版本未知 | UNKNOWN | 中-高 | 字段名不能证明发布时点 |
| weather | 未在核心小时 contract 中明确区分 forecast/realized | 不一 | UNKNOWN | UNKNOWN | 高 | 未找到带 as-of 的天气数据字典 |
| market clearing/机组价格 | 96 unit 表含 DA/RT clearing fields | 未接入 24 点主链路 | 出清后 | UNKNOWN | 高 | 15min sync/schema only |
| calendar/hour/weekday | event timestamp 派生 | 无 | 确定性预知 | VERIFIED | 低 | 多模型 calendar feature code |
| rolling stats | 历史窗口 | 取决于模型；TimeMixer cutoff 后构造 past | cutoff 后可构造 | PARTIALLY VERIFIED | 若先全表 rolling 再 split 有风险 | 各模型实现不统一 |
| normalization | StandardScaler | TimeMixer train_idx；RT916 train_df | 训练期结束 | VERIFIED for these two | 其他 wrapper 未逐一证明 | TimeMixer `repro_pipeline.py:1289-1325`; RT916 `core.py:479-525` |
| target-derived delta/tail | `RT-DA`、训练分位数、lags | 训练/历史 | labels 到达后 | PARTIALLY VERIFIED | test frame quantile 若用于选择阈值则泄漏 | SGDFNet `metrics.py:89-112`; RT916 loss/core |
| same-day/future hour | exact wrapper 可收到完整 input table | 无统一遮蔽 | 事后才可得 | NO for claimed origin | 极高 | LightGBM `lightgbm_v1.py:99-128`; TimesFM `timesfm_v1.py:108-125` |
| publication timestamps | 远程表部分含 create/update | 无 as-of snapshot | UNKNOWN | UNKNOWN | 高 | create/update 不证明当时版本可重建 |

## 5. 模型级边界结论

- **VERIFIED**：TimeMixer 明确切 past 到 cutoff、future 仅取 forecast 类字段，scaler 只 fit train index；结构上最接近 cutoff-safe，但 forecast 字段发布时间仍 **UNKNOWN**。
- **VERIFIED**：SGDFNet `protocol_b_cutoff` 会将 cutoff 后 RT 替为 DA、actual exogenous 替为 forecast；active config 使用 forecast raw、actual history false、delta history true。其可用性仍依赖 forecast 版本语义。
- **VERIFIED**：RT916 的 cutoff 代码主要替换 future RT target 并注入 DA；模型历史/窗口包含 actual-side exogenous，未找到对 cutoff 后全部 actual-side 列统一屏蔽。因此存在实现级潜在泄漏，是否在某次 run 实际触发为 **UNKNOWN**。
- **VERIFIED**：LightGBM v1 exact adapter 解析 cutoff 但调用 `run_lgbm_pipeline` 时未传；TimesFM exact 直接使用完整文件。默认 CLI 是 exact。因此 train/test 日期分离也不能排除目标日内部 future leakage。

## 6. Split、walk-forward 与阈值期

| 项目 | 状态 | 结论 |
|---|---|---|
| TimeMixer | VERIFIED | 默认回看 12 月；时间顺序 80/20 train/validation；逐日预测；未发现冻结 final test。`repro_pipeline.py:1742-1768`。 |
| SGDFNet | PARTIALLY VERIFIED | decision-day rolling、约 30 日 validation、minimum rows；active run 是否每次严格复现需 manifest。 |
| RT916 | VERIFIED | train_df/val_df chronological；现有指标是内部 validation，不是 final test。 |
| Classifier stage2 | VERIFIED | gray-train chronological 80/20，后 20% 选择 F2 threshold 0.3..0.8；stage1 动态搜索代码未被调用，默认 0.55。 |
| Fusion | VERIFIED | 当前日以前最多 30 complete ledger days；这是历史误差加权，不是每个 base 的统一 OOF 预测协议。 |
| 全链路 walk-forward | UNKNOWN | 未找到覆盖所有 base、分类修正、融合和 fallback 的统一 frozen-period 评估。 |
| threshold-selection period | UNKNOWN | 正尖峰定义、-80 修正值、`<=100` mask 和部分 tail quantile 的选择期未形成登记。 |
| final test influence | UNKNOWN | 无实验注册/搜索日志足以排除 test 对阈值、模型或报告选择的影响。 |

## 7. 需要用户提供的信息

1. **UNKNOWN**：日前/实时价格的正式市场定义、计价单位、价格上下限、系统/分区/节点层级。
2. **UNKNOWN**：每个 forecast 字段的首次发布时间、修订频率、可回溯 as-of 版本；实际字段何时结算可见。
3. **UNKNOWN**：DA 和 RT 的唯一 forecast origin，以及 `D 01:00 ... D+1 00:00` 的交易日制度依据。
4. **UNKNOWN**：正尖峰、负价、深负价的业务/统计事件定义和仅使用 training/validation 的冻结规则。
5. **UNKNOWN**：论文级 train/validation/test 日期，test 是否曾被人工查看，所有阈值的选择日志。
6. **UNKNOWN**：时区、节假日、缺失/重复修复规则，以及 2022-2026 市场规则变更日期。
7. **UNKNOWN**：可公开复现实验的数据授权边界；私有数据只能报告聚合统计。

## 8. 数据审计结论

**PARTIALLY VERIFIED**：数据量足够支持本地探索，且负尾现象真实存在；但缺少 publication-time contract、统一 forecast origin、冻结 split 与可公开复现数据。修复这些边界前，任何模型增益都不能被可靠归因于极端事件方法。
