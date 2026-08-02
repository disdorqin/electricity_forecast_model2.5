# 24 点 + 96 点双分辨率预测架构改造方案（设计规划稿）

- **性质**：全仓库分析 + 技术规划。**本任务未修改任何生产代码 / 模型 / 数据库 / 输出。**
- **日期**：2026-07-26
- **前置文档**：`docs/REMOTE_DATABASE_HOURLY_VS_15MIN_INSPECTION.md`（数据库巡检，本文 §4 直接引用其结论与来源分级）
- **后续更新（2026-08-01）**：文中提及的 `sync_data_96.py` 已删除，改由 `sync_data_96_core.py`（统一 CLI `main.py --pipeline sync_dataset --resolution 15min`）同步至 `data/remote_96/parquet/`；合并宽表见 `build_96_full_table.py`。
- **证据方法**：全部 7 条模型腿与编排层均逐文件读码核对，关键结论附 `文件:行号`；不依赖 README 断言。

---

## 1. Executive Summary（执行摘要）

客户允许在「24 点小时级」与「96 点 15 分钟级」两种预测分辨率中二选一或并存。现状：v2.5 交付链（ledger 五阶段）**全链路以 24 点为硬编码约定**，但其中最有价值的资产——业务日语义（区间末标注、跨午夜规则）、自适应 30 完整训练日选择、BGEW 权重更新数学、SGDFNet 时间戳驱动截断、交付状态机（NORMAL/DEGRADED/FAILED + exit code）——**本质上与分辨率无关**，硬编码集中在少数可枚举的位置（本文 §3 给出完整清单，约 60 处，跨 25 个文件）。

推荐 **策略 C（混合架构）**：不动 24 点生产主线；抽出一个共享的「业务时间分辨率抽象层」；96 点使用独立 ledger 根目录、独立数据集与独立输出契约；编排、manifest、交付状态机、资源调度共享。首个 96 点基线推荐 **TimesFM（零样本，checkpoint 原生支持 96 点 horizon 与 15 分钟频率，无需重训练）+ 朴素季节基线**；LightGBM 第二（每日重训，改造成本被吸收）；TimeMixer / RT916 第三批；SGDFNet 在 96 点 DA anchor 就绪后进入；极端价分类器初版**保持小时级运行、按小时广播 −80 修正**，不做原生 96 点移植。

两个前置闸门未过之前不建议动工：**数据闸门**（96 点电价目前只有单机组 `da_cq_price/rt_cq_price`，与市场价的等价性未验证；96 点特征表历史深度未测；机组表滞后约 9 个业务日）与**业务闸门**（目标价定义、96 点段位边界；实时截止已固定 14:00 / p56，见 §8，原 §14 Q5 已关闭）。

---

## 2. 现状架构（Current State）

```
main.py → cli/parser.py → pipelines/
  ledger_predict  : 7 模型（DA: lightgbm/timesfm/timemixer；RT: timesfm/sgdfnet/timemixer/rt916）
                    CPU 并行 + GPU 串行（runtime/resource_scheduler.py，与分辨率无关）
  ledger_weight   : 自适应向前扫描最近 30 个完整训练日 → BGEW 学习 (task, period, model) 权重
  ledger_fuse     : 按 weights.csv 逐小时加权融合（fusion/apply_daily_ledger_weights.py）
  ledger_classifier: 极端负电价级联分类 → −80 修正（旁路输出，不进 submission）
  final_outputs   : submission_ready.csv（24 行 6 列）+ postflight + fallback + 交付状态机
账本: outputs/ledger/{task}/{prediction,actual}/*.parquet，键 (task[,model_name,forecast_date,target_day],business_day,hour_business)
```

模型训练形态（对改造成本至关重要，逐一读码核实）：

| 模型 | 形态 | 每次运行是否重训 | 持久 checkpoint |
|---|---|---|---|
| LightGBM | 表格回归（三段模型） | 是（RT 逐日重训） | 无（pkl 每次覆盖） |
| TimesFM | 零样本基础模型 | 否 | 预训练 `google/timesfm-2.5-200m-pytorch` |
| TimeMixer | 自研池化编码器（非官方 TimeMixer） | 是（每次 run 从零训 6 个段模型） | 无 |
| RT916 | TimesBlock+Spike 残差分支，3 个时段模型 | 是（run 内训练并落盘复用） | 仅 run 内 |
| SGDFNet | HistGradientBoosting，walk-forward | 是（逐决策日重训） | 无（生产路径） |
| 分类器 | LGBM 级联两阶段 | 是（逐滚动日重训两阶段） | 仅 p1 OOF 缓存 |

**含义**：除 TimesFM 外全部模型「重训练是常态」，96 点化不存在「checkpoint 迁移」问题——需要的是 96 点历史数据 + 特征/形状参数化。

---

## 3. 全仓库 24 点假设清单（Repository-wide 24-point assumptions）

按危险度排序的核心清单（完整逐条见附录 A 思路，行号均已验证）：

### 3.1 最危险（会**静默**出错，不报错）

| # | 位置 | 问题 |
|---|---|---|
| D1 | `utils/business_day.py:64-66` | `hour_business_from_timestamp` 返回 `ts.hour` —— **丢弃分钟**。09:15/09:30/09:45 全部 →9。96 点数据经过它会 4 合 1 键碰撞 |
| D2 | `utils/business_day.py:53-55` | `business_day_from_timestamp` 只把 00:00:00 归前一日；00:15/00:30/00:45 归属未定义（15 分钟午夜块规则全仓库无一处定义） |
| D3 | `pipelines/prediction_ledger.py:43-44` | 账本唯一键只含 `hour_business`，**不含 `ds`**。96 点数据若入现有账本：hour 1..24 与小时级行互相**静默覆盖**，4 个刻度行被 dedup 折叠 |
| D4 | `TimesFMBackend/price_forecast_copy_分时段预测.py:253` | `df.index.floor("h")` + 去重 keep-last —— 喂入 15 分钟数据会被**直接压成小时数据**且不报错 |
| D5 | `SGDFNet/src/sgdfnet/data_contract.py:113-118` | `add_business_time_columns` 把所有 `hour==0` 的行（含 00:15/00:30/00:45）推回前一业务日 → 96 点下训练/推理切分被污染 |
| D6 | 所有 fusion adapters（如 `fusion/adapters/rt916.py:41-43`、`sgdfnet.py:19`、`timesfm.py:63`） | `dt.hour.replace({0:24})` 惯用法，分钟盲 |
| D7 | 行数即小时的滞后特征：LightGBM `train_fix.py:364-368`（48/168）、`train_da_fix.py:346-352`（shift/rolling 24/168）、SGDFNet `data_contract.py:237,288-361`（shift(24)/shift(168)/rolling(6/12/24/168)）、RT916 `dataprocess.py:185-220`（48/168/24/72/336）、TimeMixer `repro_pipeline.py:327-328`（diff(24)/diff(168)） | 96 点下 `shift(24)`=6 小时而非 1 天——**模型照常运行，特征语义全部错位**（最典型的静默失效） |
| D8 | `fusion/metrics.py:19-41` | 度电套利按「每行=1 个等能量可交易区间」计算，无时长项。96 点下 volume/profit 语义变为 0.25h，需显式 0.25 因子，否则与小时口径不可比 |
| D9 | 命名冲突：`pipelines/ledger_predict.py:713`、`ledger_smoke.py:135`、`verify_final_pipeline.py:66` 中 `"realtime": 96` | **现有代码里 "96" = 4 模型×24 小时**。96 点模式下 realtime 长表应为 4×96=384 行。任何检索/继承此常量的改造极易张冠李戴 |

### 3.2 硬失败类（会报错拦住，好修但量大）

- 24 行/小时集合校验：`business_day.py:234-241`、`delivery_quality.py:60,167,273,297,333,409-417,439-447`、`ledger_full.py:364-383,407,413`、`ledger_full_range.py:34-37,82-118`、`ledger_fuse.py:174-184`、`apply_daily_ledger_weights.py:77,173-177`（融合主循环 `for hour in range(1,25)`）、`emergency_fallback.py:97-137,247,278`、`ledger_weight.py:45,171-237,504`（`expected_rows=30×模型数×24`，即 2160/2880）。
- period 硬集合：`fusion/contracts.py:20,43-51`（`VALID_PERIODS={1_8,9_16,17_24}`，hb>24 直接 raise）、`daily_ledger_gef.py:149`、`business_day.py:30-38`、时段列表在 LightGBM 5 个文件中重复、TimeMixer `SEGMENTS=[(0,8),(8,16),(16,24)]`（`repro_pipeline.py:107-111`）、RT916 按小时拆段（`dataprocess.py:40-46`）。
- 模型侧形状：TimeMixer 目标日强制 24 行（`repro_pipeline.py:344-346`）、长度 8 的小时权重表仅在 `y.shape[1]==8` 时生效（`:1371-1427`，32 点段会**静默不生效**——兼具 D 类风险）、按**列索引**取通道（`:631-650,750-754`，重构特征顺序即断）；TimesFM `horizon!=24 → ValueError`（`:1386,1481`）、`_delta_hours` 下限 clip 1.0 小时（`:689`，15 分钟间隔被夹成 1h）；RT916 `OUTPUT_LEN_LIST=8`、`seq_len=72`（`core.py:205-206,509-510`）；分类器 `tail(24*9)`、`iloc[-24:]`、`-25h` 训练间隔（`cascade_daily.py:194,739-746,823`）。
- 校验/验收脚本：`verify_final_pipeline.py:66-144`（72/96/2160/2880/24/列清单）、`check_timemixer_alignment.py:44-106,222`、`check_delivery_stability.py`（合成数据全按 24 构造）。
- 契约常量重复 4 处：`SUBMISSION_COLUMNS` 在 `delivery_quality.py:19-22`、`ledger_full.py:407`、`ledger_full_range.py:34-37`、`emergency_fallback.py:20-23`。

### 3.3 已经与分辨率无关（不要重写的部分）

`runtime/resource_scheduler.py`（零耦合）；BGEW 权重更新数学（损失取均值，行数无关，`daily_ledger_gef.py:215-357`；唯一敏感点：`evidence_mass` 按行累计，96 点下 ×4，需把 `evidence_prior=5.0` 等比重标定或改按日计数）；自适应 30 完整日扫描算法本体；SGDFNet 截断核心 `_build_protocol_b_visible_frame`（`protocol_b_cutoff.py:160-192`，纯时间戳比较）；capped-SMAPE floor-50（逐点均值）；交付状态机与 exit code；RT916 的 TimesBlock（FFT 周期自适应，长度无关）；`utils/io.py::ensure_prediction_frame`。

---

## 4. 数据就绪度评估（Data Readiness）

来源分级沿用巡检报告：`[DB@07-26]` 实测 / `[快照]` 本地快照全量分析 / `[代码]` / `[假设]`。

| 维度 | 小时级 `epf_market_data` | 96 点特征 `epf_market_data_96` | 96 点价格 `epf_unit_data_96` |
|---|---|---|---|
| 目标价 | ✅ `price_dayahead/price_realtime`，市场级 | ❌ **无价格列** | `da_cq_price/rt_cq_price`，**机组级、单机组** `[DB@07-26]` |
| 特征 | 10 组 fcast/actual | 13 组 fcast/actual（多检修/正负备用） | 出力/电量/开机状态 |
| 历史 | 2022-01-01→业务日 07-27，1669 天零缺口 | DB 侧深度**未测** `[假设]`；本地仅 2 天 | 2022-01-01→07-18，1660 天，每天精确 96 行 |
| 新鲜度 | 基准 | 未测 | **落后约 9 个业务日** |
| 时间语义 | 区间末，h24=D+1 00:00 | `market_date`+`period_no 1..96` 区间末，p96=D+1 00:00 `[快照]` | 同左 `[DB@07-26]` |

已验证事实：小时特征 = 对应 4 个 15 分钟段的**精确算术平均**（重叠日 10 列预测特征 MAD=0.0000，mean 法唯一匹配）→ 两套数据同源一致，`hour=ceil(period_no/4)` 映射无歧义。**未验证**：价格列的聚合关系（小时市场价 vs 机组 96 点 cq 价）——巡检脚本 `output/_db_inspect_hourly_vs_96.py` §5 实验待跑，这是 96 点目标价选型的决定性证据。**未决业务语义**：cq 价含义（合同/清算口径）、单机组能否代表市场、96 点是否官方交付口径。

**结论**：数据侧可支撑「96 点特征 + 机组价目标」的技术验证开发（Phase 1-2），但在价格等价性证实或客户确认目标定义之前，**不可宣称 96 点预测的是市场电价**。

---

## 5. 模型兼容性矩阵（Model-by-model，读码结论）

| 模型腿 | 关键证据 | 分类 | 需重训 | 难度 | 风险 | 建议批次 |
|---|---|---|---|---|---|---|
| **TimesFM (DA+RT)** | TimesFM 2.5 **无频率嵌入**（`src/timesfm/` 全库 grep 无 freq），`max_horizon=256≥96`、分位头≤1024、`context 16384`（`timesfm_2p5_base.py:88-91`、`_torch.py:597-658`）→ checkpoint 原生支持 15 分钟/96 点；改造全在 wrapper（D4 floor("h")、`_delta_hours` clip、horizon=24 守卫、分段 24//3） | **配置+wrapper 形状改造**（模型本体零改动） | **否**（零样本） | S–M | 低–中（推理耗时×4；15 分钟精度未验证） | **第 1** |
| **LightGBM (DA)** | 表格模型逐日重训；滞后/时段/morning 窗口硬编码分布在 6 个文件（§3.1 D7）；RT 版不依赖 DA anchor | 特征+形状改造 | 是（本来就每次重训，零额外成本） | M | 中（常量重复 5 处；pickle 内嵌旧时段表） | **第 2** |
| **TimeMixer (DA+RT)** | 模型本体 pred_len 参数化、池化头长度无关；但数据/特征/校验层全 24 硬编码；长度 8 权重表静默失效风险；按列索引取通道 | 特征+形状改造 | 是（每 run 从零训练） | M–L | 中 | 第 3 |
| **RT916 (RT)** | 时段架构在 15 分钟下自然成立（按小时选行→每段 32 行）；主要是 dataprocess 行数滞后 ×4 + `OUTPUT_LEN_LIST=32`；`editable_horizon(9,16)` 在 24 点下即已失效（`annual_loss.py:43-48` 空切片）需顺手定性；`policy.py` 释放包锁死小时级，初版排除 | 特征+形状改造（偏配置） | 是（每 run 重训） | M | 中 | 第 3 |
| **SGDFNet (RT)** | 截断核心分辨率无关（最干净）；但 `rt_hat = da_anchor + delta`（`protocol_b_cutoff.py:319,343` 等）→ **硬依赖 96 点 DA anchor**；D5 业务日 bug；同小时 groupby 语义漂移 | 特征+形状改造 | 是（逐日重训） | M | 中（静默特征漂移） | 第 4（待 96 点 DA 就绪） |
| **极端价分类器** | ≤−50 标签在 15 分钟下事件碎片化、基率改变 → 灰度阈值/spw/F2 全部重标定；两个关键文件不在快照内（`run_daily.py`、`generate_oof_prob_feature.py`）无法审计；−80 规则本身逐行可移植 | **初版不建议原生移植** → 小时级运行 + 按小时广播修正（仅改 `classifier_bridge.py:78-95` 的 merge 为小时 join、`:25` 覆盖检查 +1h→+15min） | 广播路径否；原生路径是 | S（广播）/ L（原生） | 低（广播）/ 高（原生） | 广播随第 4；原生进 Phase 5 再评 |

**RT 对 DA 的依赖关系**（决定改造顺序）：TimesFM RT 与 LightGBM RT **不**依赖 DA 预测（读码证实，两者靠自身滞后/gap 模式）；TimeMixer RT 与 RT916 RT 各自消费**本 run 内自产**的 DA 预测（`repro_pipeline.py:1913,2037`；`core.py:1054-1066`）；SGDFNet 是唯一消费**外部** DA anchor 的腿。→ 96 点最小可行集可以完全不动 SGDFNet。

---

## 6. 时间与特征改造分析（哪些×4，哪些重调）

**按物理时长缩放（确定性 ×4，不需实验）**：一天滞后 24→96；一周 168→672；两天 48→192；14 天 336→1344；日内滚动 rolling(24)→rolling(96)；morning 窗 1..15→1..60；分类器 `tail(24*9)`→`tail(96*9)`；`min_val_rows 24*7`→`96*7`；`train_min_rows 2160(90天)`→8640；动态灰度 `min_samples 720(30天)`→2880。

**属于模型容量/统计效率，须实验重调（禁止机械×4）**：TimeMixer `seq_len 168→672?`（编码器池化，替代方案：下采样输入或 seq_len=336+多尺度）、`down_sampling scales 3→4-5`、`MovingAvg(25)→~97?`；RT916 `TOP_K 2→3-4`、`INPUT_LEN_LIST`（段历史天数不必×4——天数不变，行数自然×4）、D_MODEL/epochs；TimesFM 分段与否（96÷3=32 段长 vs 不分段 gap 192≤256 均可行，A/B 决定）；LightGBM `min 2000 行`（96 点下 2000 行仅≈21 天，应按「天数」口径重设为 ~90×96）；分类器灰度阈值/`scale_pos_weight`/F2 阈值（基率变化，重新搜索）；BGEW `evidence_prior`（×4 或改按日）。

**编码类**：hour_sin/cos ÷24 → 日内刻度 sin/cos ÷96（保留 hour 特征作为粗粒度并加 quarter-of-hour，通常优于只换）；RT916 `hour_embed Embedding(25)`→保 hour + 加 quarter（或 Embedding(97)）；节假日/星期/节气特征天级 join，零改动。

**period 分组**：初版**保持三段映射**：`1_8→槽 1..32`、`9_16→33..64`、`17_24→65..96`（32 点/段）。理由：BGEW 数学与段内点数无关；权重表/校准器/风险小时定义全部按段或按小时组织，三段延续可直接复用；更细分组（6×4h、按小时、逐点）在 30 天×32 点样本下先天过拟合风险，留待 96 点账本积累后作为实验项。`configurable period-group definition` 作为抽象层参数预留（见 §7）。

---

## 7. 业务时间抽象层设计（核心新增件）

新增 `utils/business_time.py`（或扩展 business_day.py），单一事实来源：

```python
@dataclass(frozen=True)
class Resolution:
    name: str              # "hourly" | "quarter"
    periods_per_day: int   # 24 | 96
    minutes_per_period: int# 60 | 15
    periods_per_hour: int  # 1 | 4
    freq: str              # "h" | "15min"
    period_groups: dict    # {"1_8": (1,32), ...} 槽区间，含 hourly 的 (1,8)...

HOURLY  = Resolution("hourly", 24, 60, 1, "h",   {"1_8":(1,8),"9_16":(9,16),"17_24":(17,24)})
QUARTER = Resolution("quarter",96, 15, 4, "15min",{"1_8":(1,32),"9_16":(33,64),"17_24":(65,96)})
```

**统一索引改用 `business_period`（1..periods_per_day，区间末标注）**；`hour_business` 在 hourly 模式下 ≡ business_period（外部契约零变化），在 quarter 模式下作为派生列保留（`hour_business = ceil(business_period/4)`）用于兼容、分段与人读。**不替换外部字段名，内部逐步以 business_period 为准**——即题面选项中的「generalize + retain as derived field」。

精确映射规则（与 DB 侧 `period_no` 已验证语义一致）：

```
business_period p of day D  ↔  ds = D 00:00 + p × minutes_per_period   (p=1..N)
p = N (24/96)               ↔  ds = D+1 00:00:00
business_day(ds) = (ds − 1 second).date()          # 单一规则，两种分辨率通用，
                                                    # 自动解决 00:15/00:30/00:45 归属：属于“当日开盘侧”
p(ds) = ((ds − 1s).hour × 60 + (ds − 1s).minute) // minutes_per_period + 1
hour_business = ceil(p / periods_per_hour)
period_group  = 由 period_groups 区间表查得
DB 对应: business_day ≡ market_date, business_period ≡ period_no（已验证 1..96 区间末）
```

**关于 00:15–00:45 的归属决策**：上式把 00:15 归入「当天」业务日（作为 p=1），与 DB `epf_unit_data_96` 实测语义完全一致（`market_date=D` 的 p1 = D 00:15）`[DB@07-26/快照]` —— 因此**不是**「00:00–00:45 整块像 h24 一样归前一日」；只有 00:00:00 一个点（p=96）归前一日。此规则修复 D2/D5 并与数据库天然对齐，无需数据变换。

午夜/排序/去重/跨日 join：输出严格按 business_period 升序；唯一性校验键 = (business_day, business_period)；跨日拼接用 ds 单调性校验（`ds(p=N) = D+1 00:00` 断言两分辨率同构）；对齐校验脚本按 `ds == D 00:00 + p×Δ` 逐行断言（推广 `check_timemixer_alignment.py:63-95` 的做法）。

---

## 8. 截断（cutoff）与防泄漏设计

现状核对：CLI 默认 `--realtime-cutoff-hour 14`（`cli/parser.py:140`），仅流经 `ledger_predict.py:145` 生成 `"D-1 14:00:00"` 字符串下发各模型；RT916 内部 CLI 默认 15（`cli.py:27`）但统一管线传 14；TimeMixer 审计口径 DA 15:00/RT 14:00（`repro_pipeline.py:66-67,1673-1686`）；LightGBM RT 用 `start − 11h` ≡ 14:00（`infer_fix.py:220-221`）；SGDFNet `decision_ts = D-1 + decision_hour`（时间戳比较，任意分辨率成立）。文档 15:00 与代码 14:00 的矛盾是**存量问题**，与 96 点改造解耦。

两种候选规则在 96 点下的精确语义（区间末标注）：

| 口径 | 最后可用段 | 首个被遮蔽段 | 是否含截止整点所在段 |
|---|---|---|---|
| **cutoff=14:00（已固定）**，规则 `ds ≤ cutoff_ts` | **p56**（13:45–14:00，标注 14:00） | p57（标注 14:15） | 含（14:00 恰为 p56 的区间末） |
| ~~cutoff=15:00（历史/非候选）~~，规则 `ds ≤ cutoff_ts` | p60（标注 15:00） | p61 | 含（仅历史记录，不作候选） |

即：沿用现有各处「`ts > decision_ts` 遮蔽」的比较式（SGDFNet `protocol_b_cutoff.py:180-181`、TimeMixer `:289`、RT916 `core.py:275-296`），96 点下**自动**得到 p≤56（或 p≤60）可见——无需任何按段位写死。**发布延迟**是另一维度：实测 RT 实际值按日回补（快照 §7），若 15 分钟 RT 价盘中发布有延迟，理论可见 ≠ 实际可得，需在 Phase 1 数据校验中按列实测「首个非空时间 vs data_time」后决定是否配置 `publication_lag` 保守回退。

**配置驱动设计（cutoff 已固定为 14:00）**：**2026-07-28 业主终裁：实时截止固定为 14:00 / `realtime_cutoff_period = 56`（见 `LEAKAGE_AUDIT_96.md` §3、`DATA_CONTRACT_96.md` §7）。** 不再作为待选方案；`cli/parser.py:140` 的 `--realtime-cutoff-hour 14` 即权威值。cutoff 统一表示为 `cutoff_time: "HH:MM"`（默认 `"14:00"`，分钟粒度，替代 hour 整数），派生 `cutoff_ts = D-1 + cutoff_time`、`last_visible_period = f(cutoff_ts)` 写入 manifest；两处非时间戳实现须改为从 cutoff_ts 派生：SGDFNet `blocked_hours=range(decision_hour+1,25)`（`protocol_b_cutoff.py:124`，仅 da_fill_bias 分支）与 LightGBM 的 `Timedelta(hours=11)`（已是壁钟时长，保留）。15:00（→p60）仅作历史歧义记录（`protocol_b_cutoff.py` 旧 `decision_hour=15` 默认值与部分旧文档），**不再作为候选**。防泄漏测试覆盖训练模拟、回测、生产推理三处（§13），且每腿需 p56/p57 边界断言（p56 可见可用、p57 遮蔽绝不泄漏）。

---

## 9. 账本、权重、融合、分类器、输出契约影响

### 9.1 账本（推荐：**独立 96 点账本根目录**）

同库共存被否决的硬证据：唯一键不含 ds/分辨率（D3），`business_day+hour_business(1..24)` 与 `business_day+period_no(1..96)` 在 1..24 区间语义冲突且互相覆盖；单体 parquet 每次 append 全量重写（`prediction_ledger.py:114-153`），96 点行数×4 再叠加混存会放大重写成本与损坏半径。**方案**：`outputs/ledger_96/{task}/{prediction,actual}/`，schema 在现有列基础上：`hour_business` 列承载 business_period（1..96，列名不变以复用全部管道代码），新增列 `resolution`（冗余防呆）+ 保留 `ds`；唯一键改为共享常量 `UNIQUE_KEY = [...,"business_day","hour_business"]`（96 树内语义即 period）。完整日定义参数化：`模型数 × N`，N 来自 Resolution（RT 96 点长表 = 4×96=384 行，**显式测试防 D9 命名混淆**）。30 完整日选择算法零改动复用。

### 9.2 权重与融合

BGEW 复用，仅：`GEFConfig.periods` 从 Resolution.period_groups 注入；`evidence_prior` 重标定（×4 或按日计 evidence）；`get_coverage_report n_expected` 参数化。融合循环 `for hour in range(1,25)`（`apply_daily_ledger_weights.py:77`）→ `for p in range(1, N+1)`。**混合分辨率融合：不做**——所有参与融合的腿必须同分辨率（客户也要求不得静默混用）；24↔96 转换仅允许两处受控使用：(a) SGDFNet 的 DA anchor 在 96 点 DA 腿缺席时可由小时 DA **显式插值**（manifest 记 `anchor_source: interpolated_hourly`，视为降级）；(b) 应急 fallback。均不得作为常规模型腿入账本。

### 9.3 分类器（初版广播方案）

小时级分类器照常跑（数据、阈值、OOF 缓存全不动）；`classifier_bridge.merge_clf_results` 由按 `ds` 精确 join 改为按小时 join 后广播到 4 个刻度（`final_pred==1 & y_fused≤100 → −80` 逐刻度应用）；覆盖检查 `+1h`→`+Δ`。修正结果**维持旁路输出**（现状：`submission_ready.csv` 用未修正 RT，`ledger_full.py:389` 已核实）——是否让修正进入官方 96 点交付属于业务问题（§14 Q10 附带）。原生 15 分钟分类器留 Phase 5：需 15 分钟 RT 实际值历史 + 阈值体系重标定 + 补齐快照缺失的两个入口文件。

### 9.4 输出契约

| 模式 | 文件 | 契约 |
|---|---|---|
| hourly | `final/submission_ready.csv` | **完全不变**（24 行 6 列，字节级兼容） |
| quarter | `final/submission_ready_96.csv` | 96 行，列：`business_day, ds, business_period(1..96), hour_business(派生), period, dayahead_price, realtime_price`；严格升序、无重复、无 NaN；p96 的 ds=D+1 00:00 |

分辨率必须显式可审计：`run_manifest.json` 新增顶层 `resolution` 字段 + `expected_rows`；`delivery_report.md` 首行标注；`--resolution` 进 CLI 并写入每个 stage 的 manifest 条目；两种输出**不并存于同一 run**（一次 run 一种分辨率；若客户日后要求双输出，按两次 run 两套 manifest 执行，主/派生口径见 §14 Q7）。`validate_daily_submission` 12 条规则全部参数化复用（行数、槽集合、p_N 的 ds 检查）。fallback 同法参数化（中位数按 business_period 分组，7/30/全历史层级不变）。

### 9.5 指标与业务评估

逐点指标（MAE/RMSE/SMAPE/capped-SMAPE/accuracy=1−SMAPE/SCR）**逐点等权直接成立**，无需改公式；但横向比较规则必须固定：**跨分辨率比较一律把 96 点预测 mean 聚合到 24 点后与小时模型同口径比**（数据侧已证 mean 是精确对应）。需要 0.25 因子的是**能量/收益类**：`arbitrage_metrics`（`fusion/metrics.py:19-41`）加 `interval_hours` 参数，`profit_i = q_i × spread_i × interval_hours`、`volume = Σq_i × interval_hours`（度电套利=商，两者同乘 0.25 后不变——但 total_profit/total_volume 绝对值必须带因子，否则日报表 4 倍虚高）。energy-weighted SMAPE 不建议引入（等时长区间下 = 等权）。极端价指标改「事件级」口径评估（连续 ≥−50 刻度段的检出率）作为 96 点新增报表项。验收阈值：96 点 capped-SMAPE 阈值**不得**直接沿用小时阈值（15 分钟波动更大），Phase 2 用朴素基线实测后另定。

---

## 10. 架构策略比较与推荐

| 维度 | A 统一分辨率感知重构 | B 完全独立 96 管线 | **C 混合（推荐）** |
|---|---|---|---|
| 开发复杂度 | 高（一次性触碰 ~60 处×25 文件） | 中（复制后各改各的） | 中（抽象层小步引入） |
| 对 24 点生产风险 | **高**（每处参数化都可能回归） | 最低 | 低（24 主线路径不动，仅注入默认 Resolution=HOURLY 的等价改写 + 回归测试钉死） |
| 代码重复 | 最低 | **高**（五阶段+校验+文档双份，已知 SUBMISSION_COLUMNS 重复 4 处的教训会翻倍） | 低-中（编排/状态机/校验共享，模型配置与账本分离） |
| 可测试性 | 差（同一路径两态） | 好但双倍维护 | 好（共享层单测 ×2 分辨率参数化） |
| 模型兼容 | 强迫 7 腿同步改 | 允许逐腿 | 允许逐腿（每腿一个 `points_per_day` 参数） |
| 未来加分辨率（如 5min） | 好 | 差 | 好（Resolution 数据类再加一行） |
| 迁移难度/运维 | 大爆炸 | 双系统运维 | 渐进，单 CLI 双模式 |

**推荐 C**，对应题面五项优先级：24 点主线不动（账本、输出、校验路径原样）；96 点可验证（独立账本+独立契约+显式 resolution 审计链）；无大重写（抽象层只替换已枚举的硬编码点，等价性由回归测试保证）;血缘清晰（ledger_96 目录、manifest resolution 字段、anchor_source 标注）；支持未来生产交付（五阶段/状态机/exit code 全复用）。

---

## 11. 分阶段实施计划

> 每阶段列出：目标 / 主要改动文件 / 依赖 / 交付物 / 测试 / 验收 / 风险 / 回滚隔离。**本任务不实施。**

**Phase 0 业务与数据确认（无代码）**
目标：关闭 §14 全部闸门问题。动作：跑 `output/_db_inspect_hourly_vs_96.py`（价格聚合实验+market_96 历史深度+schema）；恢复 96 爬虫日更；与客户书面确认目标价/主体/截止/输出口径/历史与新鲜度要求。交付：确认纪要 + 巡检 JSON。验收：Q1–Q10 有书面答案。风险：机组价≠市场价 → 触发备选（爬市场级 96 点价，工期另计）。回滚：纯调查，无。

**Phase 1 分辨率抽象 + 数据层**
目标：抽象层落地且 24 点行为逐字节不变。文件：新增 `utils/business_time.py`、`config/resolution.py`；改 `utils/business_day.py`（内部委托，公开 API 不变）、`cli/parser.py`（`--resolution`，默认 hourly；`--realtime-cutoff-time HH:MM` 兼容旧参数）、`fusion/contracts.py`、共享 `SUBMISSION_COLUMNS/UNIQUE_KEY` 常量收敛到单模块；`sync_data_96` 增加 96 点建模数据集构建（unit 价 + market_96 特征按 `market_date,period_no` join，含完整性/防泄漏校验报告）。依赖：Phase 0。交付：96 点数据集 + 校验报告。测试：新增 `scripts/check_resolution_abstraction.py`（两分辨率映射性质测试）+ 现有四件回归（40/40、29/29、16/16、41/41）+ **黄金基线**：改造前后对同一日期跑 hourly full chain，`submission_ready.csv`、`weights.csv` 逐字节 diff 为空。验收：diff 空 + 96 数据集通过完整性校验。风险：D1/D2 改动波及 hourly——以黄金基线钉死。回滚：抽象层默认值=HOURLY，revert 单模块即可。

**Phase 2 首批 96 点基线（不进账本，不动五阶段）**
目标：可信基线，而非模型数量。模型：**朴素季节基线**（昨日同刻/上周同刻/7 日同刻中位数——同时就是未来 fallback 与「安全退化」参照）→ **TimesFM**（wrapper 参数化 §5 清单）→ **LightGBM**。文件：`TimesFMBackend/price_forecast_copy_分时段预测.py`+`infer.py`、`lightGBM/*` 6 文件、`runners/adapters/*`。依赖：Phase 1 数据集。交付：独立回测报告（96 点 capped-SMAPE 分段/总体 + mean 聚合到 24 点与小时腿对比）。测试：形状（96 行、p1..96）、确定性 smoke、防泄漏（cutoff 后置 NaN 断言）、TimesFM `floor(step)` 单测。验收：两腿在 ≥60 天回测上稳定优于朴素基线；聚合口径下不劣于对应小时腿 −X%（X 由 Phase 0 定）。风险：15 分钟精度不达标 → 停在此阶段重新谈口径，损失最小。隔离：全部输出进 `outputs/experiments_96/`，不触碰 ledger。

**Phase 3 RT 链路与 DA 耦合**
目标：96 点 DA→RT 完整推理日流程。内容：TimeMixer、RT916 的 96 点化（§5/§6 清单，含权重表 8→32、通道命名化重构、editable_horizon 定性）；SGDFNet 96 点化（D5 修复 + 滞后参数化 + 96 点 DA anchor 接入，anchor 缺席时插值降级并标注）；cutoff 配置化贯通三处实现；逐日 walk-forward 回测含多市场情景（负价日、尖峰日、检修期）。验收：各腿防泄漏测试通过；RT 分段指标不劣于朴素基线；cutoff 固定 14:00 下 last_visible_period 断言 p56（p56/p57 边界测试：p56 可见可用、p57 起遮蔽绝不泄漏；15:00/p60 仅作历史回归参考，不要求）。风险：GPU 腿训练耗时×4 → 用 `--smoke-*` 参数族先行；RT916 policy 路径明确排除。

**Phase 4 融合与账本集成**
目标：96 点五阶段跑通。内容：`outputs/ledger_96` 落地（§9.1）；`prediction_ledger/ledger_weight/ledger_fuse/ledger_full(_range)/delivery_quality/emergency_fallback` 参数化（§3.2 清单逐条销号）；BGEW evidence 重标定；`submission_ready_96.csv` 契约 + manifest resolution 字段；96 点 backfill（先 30 完整日）。测试：`check_delivery_stability` 扩展 96 合成用例；384 行长表显式断言（防 D9）；fallback 不写账本断言复用。验收：某目标日 96 点 full chain `NORMAL / exit 0 / 96 行 0 NaN`，同机 hourly 链回归四件全绿。回滚：ledger_96 整目录可删，不碰 hourly 账本。

**Phase 5 分类器与业务指标**
目标：极端价处理与商业评估闭环。内容：广播式 −80 修正（§9.3）；`fusion/metrics.py` 加 interval_hours；交付/区间报告加 96 点分段指标与事件级极端价指标；（可选评审后）原生 15 分钟分类器立项。验收：修正报告逐刻度可审计；套利指标带 0.25 因子且与小时口径对账单调一致。

**Phase 6 生产加固**
目标：可运维。内容：range 模式 96 点 preflight；`verify_final_pipeline/verify_range_pipeline/check_timemixer_alignment` 参数化；监控（数据新鲜度检查已按 96 表实现，补机组表滞后告警）；RUNBOOK/OUTPUT_CONVENTION/README 双分辨率章节；降级演练（fallback、DEGRADED、exit 2）。验收：96 点连续 ≥14 天陪跑 NORMAL；文档评审通过。

---

## 12. 首批 96 点模型集推荐（结论）

- **最小可行集**：朴素季节基线 + TimesFM(DA,RT) + LightGBM(DA)。理由（码上证据）：TimesFM checkpoint 原生兼容（§5 行 1）、无训练依赖、双任务覆盖；LightGBM 每日重训吸收重训成本且 DA 腿特征最简；两者与 SGDFNet 的 anchor 依赖链解耦。
- **顺序**：TimesFM → LightGBM → TimeMixer/RT916（并行第三批）→ SGDFNet（第四，等 96 点 DA anchor）。
- **首个生产基线**：TimesFM——但**首个可信参照**是朴素季节基线（它同时是 fallback 与安全退化标尺，必须最先存在）。
- **初期保持 hourly-only**：SGDFNet（anchor 依赖）、极端价分类器（原生移植高风险 + 快照缺文件）、RT916 policy 释放包路径。
- **可能不适合/暂缓的理由**：无一腿因架构根本不适合被排除；分类器是唯一「原生移植不划算」的组件（事件语义改变 + 阈值体系重标定 + 审计盲区）。

---

## 13. 测试与验收框架（未来实现的 DoD）

**数据测试**：每完整业务日精确 N 行（24/96 参数化）；午夜映射（p=N 的 ds=D+1 00:00；00:15 归当日 p1）；(business_day, business_period) 无重复；目标列无 NaN（完整日定义内）；ds 严格单调；特征-目标对齐抽样断言（`y[t]` 只依赖 `ds≤cutoff` 信息，用构造数据验证）。
**防泄漏测试**：cutoff 后 RT 真值不可见（对三种实现各写断言：SGDFNet visible frame、TimeMixer past-window、RT916 asof 替换+特征重算）；train/val 时序切分；无未来 DA/RT 注入（目标日 DA 只允许来自预测列）；**分辨率感知滞后校验**——对同一物理时刻，96 点特征 `shift(96)` 与小时特征 `shift(24)` 指向同一历史时刻（防 D7 类静默错位的核心测试）。
**模型测试**：输入/输出形状（含 384 行长表）；固定种子 smoke 双跑逐字节一致；checkpoint-分辨率兼容（TimesFM compile 配置断言；RT916 ckpt 文件名含 pred_len 断言）；CPU/GPU 调度不变性；单腿失败隔离（其余腿照常，manifest 记错）。
**管线测试**：hourly 黄金基线逐字节不变（每个 Phase 出口都跑）；96 模式独立可跑；所有 artifact（manifest/报告/权重/账本行）携带 resolution；fallback 不写账本；后阶段可重生成（weight/fuse/classifier 幂等重跑）；range 模式两分辨率互不干扰（不同 runs-root/ledger-root 断言）。
**输出测试**：24/96 行、0 NaN、列清单与顺序、时间戳规则、manifest 完整、delivery_status/exit code 映射（0/2/1 三态各造一例）。
**回测测试**:严格样本外（预测日全部信息边界模拟）；多情景切片（负价日、尖峰日、节假日、新能源大发日）；聚合消融（96→24 mean 口径 vs 原生 24 腿）；对朴素基线的安全退化下界（任何 96 点腿劣于朴素基线即 FAIL）。

---

## 14. 风险登记册（Top）与阻塞性业务问题

**风险登记册**（缩略；D 编号见 §3.1）：R1 静默特征错位（D7）— 用分辨率感知滞后测试拦截；R2 账本键碰撞（D3）— 独立 ledger_96 根治；R3 wrapper 静默降采样（D4）— 单测钉死；R4 "96"命名混淆（D9）— 384 行显式断言 + 代码注释改名 `rows_per_task`；R5 24 点回归 — 黄金基线 diff；R6 机组价代表性 — Phase 0 闸门；R7 96 数据断供（滞后 9 日）— 爬虫恢复 + 新鲜度告警；R8 GPU 成本×4 — smoke 参数族 + 分批训练窗口实验；R9 快照缺文件（SGDFNet runtime_helpers 等 4 件、分类器 2 件）— 动工前在真实仓库核对；R10 evidence/阈值类超参失配 — Phase 2/4 重标定清单跟踪。

**阻塞性业务问题（动工前必须书面回答）**：

1. 96 点预测的**确切目标价**是什么（哪个字段/哪个口径的 15 分钟价）？
2. 目标是**市场级、节点级还是机组级**？
3. `da_cq_price` / `rt_cq_price` 能否作为官方目标？（等 Phase 0 聚合实验证据 + 客户确认，二者缺一不可）
4. 单机组是否足够，还是必须支持多机组（多机组 ⇒ 账本键加 unit_id、爬虫扩容）？
5. ~~官方截止是 **14:00 还是 15:00**？~~ **已终裁（2026-07-28）：固定 14:00 / `realtime_cutoff_period = 56`**（p56 含截止整点段、p57 起遮蔽；96 点段位边界已锁定）。详见 `LEAKAGE_AUDIT_96.md` §3、`DATA_CONTRACT_96.md` §7。
6. 系统需**运行时双分辨率可切换**，还是部署时二选一？
7. 是否需要**同时产出**两种分辨率？若是，哪个是主口径、哪个允许聚合派生？
8. 96 点输出是**官方提交**还是内部分析件？（决定其是否走 NORMAL/postflight 全套验收）
9. 最低历史深度与新鲜度要求（96 点账本 30 完整日从何日起算；机组表滞后容忍度）？
10. 15 分钟预测的**商业评估口径**（度电套利 0.25h 计法确认；−80 修正是否进入 96 点官方交付；验收阈值另定）？

---

## 15. 推荐的立即下一步

1. 在办公机运行 `output/_db_inspect_hourly_vs_96.py`（只读，约 1 分钟），拿到价格聚合实验与 market_96 历史深度——Q3 的技术半边。
2. 恢复 96 点爬虫日更并回填 07-19 以来缺口（Q9 的前提）。
3. 携本文 §14 十问 + 巡检报告与客户开一次确认会，锁 Phase 0。
4. 客户确认后，从 Phase 1 的 `utils/business_time.py` + 黄金基线回归框架动工（其本身对 24 点零行为变化，是风险最低的第一刀）。

---

*附注：本文引用行号基于 2026-07-26 设备快照；`_archive/` 遗留代码未纳入改造范围；SGDFNet 4 个辅助模块与分类器 2 个入口文件不在快照中，实施前须在真实仓库复核（R9）。*
