# 96 点（15 分钟）最小兼容性评估（修正版规划文档）

- **性质**：业务与技术确认轮。**本轮未修改任何生产代码 / 模型 / 数据库 / 账本 / 融合 / 分类器 / 小时级输出。**
- **日期**：2026-07-26
- **修正说明**：本文**取代** `docs/PLAN_24_AND_96_POINT_FORECASTING_ARCHITECTURE.md` 作为实施方向依据（该文的「硬编码 24 点清单」仍有效，作为附录证据引用；其「新基线组合 / 统一抽象层重构」等超出范围的实施取向作废）。目标改为：**保持现有架构、业务流程、模型组合、CPU/GPU 执行策略、交付契约与运维行为不变，找出让现有模型能接收/训练/预测 96 点数据的最小安全改动集。**
- **后续更新（2026-08-01）**：文中建议「放 `sync_data_96.py` 内或旁挂」的 `build_dataset_96` 数据层，现已落地为独立工具 `build_96_full_table.py`（合并 `epf_market_data_96` 特征 + `epf_unit_data_96` 价格 → 与 `shandong_pmos_hourly` 同列宽的 15 分钟宽表）。旧 `sync_data_96.py` 已删除，由 `main.py --pipeline sync_dataset --resolution 15min` 取代。
- **证据分级**（每条重要结论标注）：`[业主明示]` 项目负责人本轮陈述 / `[README]` 文档陈述 / `[代码]` 生产代码实测（附文件:行号）/ `[数据]` 数据库 07-26 只读实测或本地快照全量分析 / `[历史]` 历史或未启用代码 / `[矛盾]` 未决不一致（只报告，不擅自取舍）。

---

## 1. 修正后的业务流程

两条独立预测目标：**日前电价** 与 **实时电价**。在业务日 D 预测完整目标日 D+1（24 点或 96 点，二选一，整日全覆盖）`[业主明示]`。

### 1.1 日前（DA）
D 日可用截至 D 日的完整日前价信息（受现有数据契约约束）`[业主明示]`。生产 DA 模型腿（**代码实测**，`pipelines/ledger_predict.py:45`）：`DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]` —— 与业主所述「Time-LFM/TimesFM + LightGBM」一致，另含 TimeMixer（README 与代码一致）。TimesFM 腿不走 registry，经 `runners/adapters/timesfm_v1.py` 直连 `TimesFMBackend`（`runners/registry.py:18-20` 注释明示；老 `TimesFM/` wrapper 已归档 `[历史]`）。

### 1.2 实时（RT）
D 日实时信息**截至 14:00**；15:00 起的 RT 真值不可知；不可用区必须遮蔽；仍须预测 D+1 全天 `[业主明示]`。生产 RT 模型腿（`ledger_predict.py:46`）：`REALTIME_MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]` —— 与业主清单一致。**DA 与 RT 不共用输入构造**：各腿 DA/RT 特征表、截断处理完全不同（LightGBM DA/RT 是两套 train/infer 文件；TimesFM 用 `skip_style normal/gap` 区分，`TimesFMBackend/infer.py:46`；详见 §11）。

### 1.3 DA↔RT 依赖（代码实测，与业主认知存在一处重要差异）
`ledger_predict` 内 DA 腿**先于** RT 腿执行（`ledger_predict.py:193-226`，注释 "must run BEFORE realtime"）`[代码]`。但各 RT 腿的 DA 来源（`_get_da_feature_source`，`ledger_predict.py:674-684`，逐腿读码核实）：

| RT 腿 | DA 来源 | 结论 |
|---|---|---|
| timesfm | `timesfm_none` —— 不使用任何 DA 信息（gap 模式跨 D-1 预测，且显式排除另一价格列，`price_forecast_copy:1088-1090`） | 独立 |
| lightgbm-RT（注：RT 组合中无 lightgbm，仅列对照） | — | — |
| timemixer | `timemixer_internal_dayahead_prediction` —— 消费**本 run 内自产**的 TimeMixer DA 预测（`TimeMixer/repro_pipeline.py:1913,2037`） | 内部自产 |
| rt916 | `rt916_internal_joint_dayahead_prediction` —— 消费**本包自产**的 RT916 DA 回测预测（`core.py:1054-1066`） | 内部自产 |
| sgdfnet | `sgdfnet_config_da_fill` —— `da_anchor` 取自**数据文件的 `日前电价` 列**，目标日缺失时用历史同小时中位数回退（`data_contract.py:137-188`）；`rt_hat = da_anchor + delta_hat`（`protocol_b_cutoff.py:319,343`） | 依赖 DA **数据**，非官方 DA 预测输出 |

**`[矛盾]` 需与业主确认**：业主表述为「SGDFNet 需要日前预测锚点，故先跑 DA 再跑 RT」。代码现状是：执行顺序确实 DA 先行，但**没有任何 RT 腿消费官方融合 DA 结果**（融合发生在第 3 阶段 `ledger_fuse`，晚于全部模型预测）；SGDFNet 的锚是数据里的日前价列（对目标日 D+1 而言该列通常尚未发布→中位数回退，README §2 "SGDFNet target-day NaN FIXED" 印证）。若业务上希望 SGDFNet 锚定「官方 DA 预测」，那是一个**现状不满足的需求变更**，须单独立项，不属于 96 点兼容范围。DA 阶段失败对 RT 的影响：`_run_model_set` 逐腿捕获异常、结果进 manifest（失败隔离）；SGDFNet 因锚来自数据文件而非 DA 阶段输出，DA 腿失败**不会**级联击穿 SGDFNet；整体交付降级由五阶段状态机与 fallback 兜底 `[代码]`。

### 1.4 极端负电价分类器（仅 RT 路径）
入口：第 4 阶段 `ledger_classifier`（`pipelines/ledger_classifier.py:198-204` → `fusion/classifier_bridge.py`），消费 **RT 融合结果** `fused_predictions.csv`；规则 `final_pred==1 且 y_fused≤100 → −80`（`classifier_bridge.py:90-92`）。**非阻断**：默认失败不拦交付，`--strict-classifier` 才闭环（`ledger_classifier.py:107-113`）。**修正结果不进官方 submission**：`submission_ready.csv` 读取未修正的 `realtime_final_predictions.csv`（`ledger_full.py:389,404`），修正版仅旁路输出 `[代码]`。DA 路径无分类器 ✔。小时假设：训练数据 `时刻` 小时级、`tail(24*9)`、`iloc[-24:]`、覆盖检查 `+1h`（`cascade_daily.py:194,745-746,823`；`classifier_bridge.py:25`）。**本轮不重设计、不移植。**

### 1.5 CPU/GPU 调度（保留现设计）
`runtime/resource_scheduler.py` 无任何分辨率耦合（读码确认），无需替换。24→96 的影响评估：GPU 显存与训练时长——TimeMixer 过去窗 168→672 行、未来帧 24→96 行，其编码器为线性投影+池化，显存/时长约线性 ×4，`--timemixer-batch-size` 现参可下调（16→8）兜底；RT916 seq 72→288、d_model 64，规模仍小；推理时长 TimesFM 每段 context ×4；CPU 腿（LightGBM/SGDFNet）行数 ×4，HGB/LGBM 近线性，分钟级可承受。批维度：各腿 batch 均为样本维（天/段），不受点数直接约束。超时：executor 无显式超时（`runners/executor.py` 读码确认），无需改。失败隔离：现有逐腿 try/except + manifest 记录，沿用。**结论：调度层零改动；仅需在验收时实测一次 96 点 smoke 的显存与耗时**（现有 `--smoke-*`、`--timemixer-epochs/patience/batch-size` 参数已全线贯通，`ledger_predict.py:617-636`）。

---

## 2. 权威文档重读结论（README/文档 vs 代码）

| 主题 | README/文档 | 代码实测 | 判定 |
|---|---|---|---|
| 五阶段与顺序 | README §1 | `ledger_full.py` 依序调用，DA 组先于 RT 组在阶段 1 内部保证 | 一致 |
| 模型组合 | README §3（DA 3 + RT 4） | `ledger_predict.py:45-46`、`delivery_quality.py:46-49` | 一致 |
| RT 截止 | README/RUNBOOK：`--realtime-cutoff-hour` 默认 14；`docs/项目执行逻辑与陪跑步骤对齐.md:10` 写 **15:00** | CLI 默认 14（`cli/parser.py:140`）；生产 manifest 样本 `data_cutoff_realtime = 2026-02-23 14:00:00`（`fixtures/repro_bundle/sample_runs/2026-02-24/run_manifest.json`）；RT916 独立 CLI 默认 15（`cli.py:27`）但统一管线传 14 | `[矛盾]` 文档 15:00 为旧口径；运行态 14:00，与业主本轮陈述一致 |
| hour24 约定 | README §1/OUTPUT_CONVENTION | `business_day.py:45-78` | 一致（区间末） |
| 数据同步 | README §5.3/§17 | `sync_data.py`（db/http/local/auto 四源）+ `sync_data_96.py` | 一致 |
| 96 表用途 | README §5.4 称 `epf_unit_data_96` 为「机组级」多机组语义 | DB 实测单机组（distinct_units=1）`[数据]` | `[矛盾]`（数据未达文档语义） |
| classifier 进交付 | README §1 流程图注明「最终提交结果不加分类器矫正」 | `ledger_full.py:389` 证实 | 一致 |
| SGDFNet 锚 | `.workbuddy` 快照称「da_anchor = DA 的 D+1 预测」 | 实为数据列 + 中位数回退（§1.3） | `[矛盾]`（内部笔记与代码不符，以代码为准报告） |

---

## 3. 15 分钟数据库全量清点（现有证据 + 待跑全库扫描）

**执行限制**：本云端会话对数据库主机出站被网络白名单拦截（TCP 超时，双轮复测），依赖包安装亦被出口代理 403 拦截，故**无法在本环境直接遍历 information_schema**。为满足「不要只看两张候选表」的要求，已交付**全库只读扫描脚本** `output/_db_inspect_full_schema.py`（v2）：遍历账号可见全部 schema/表/视图/列/注释，按列名（price/cq/clear/settle/spot/da_/rt_/96/quarter/period…）与中文注释（价/出清/结算/电价…）双通道检索候选目标价列，并对每张候选表做覆盖度统计。在办公机 epf-2 环境一条命令即可运行（§8）。

**当前可核实的清点**（代码 DDL + 爬虫端点 + 07-26 实测 + 本地快照）：

### 3.1 已确认的 15 分钟表

**`epf_unit_data_96`**（表注释：机组级电力市场96点数据(15分钟粒度)，DDL 全文见 `scripts/db_migrations/001_create_epf_unit_data_96.sql`）`[代码+数据]`：

| 列 | 类型 | 注释 | 类别 |
|---|---|---|---|
| market_date / period_no / data_time | DATE / INT / DATETIME | 市场日期 / 96点序号1-96 / **完整时刻(区间结束时间)** | 时间 |
| unit_id | VARCHAR(64) | 机组ID | 主体 |
| **da_cq_price** | DECIMAL(14,4) | **日前出清价格(元/MWh)** | 价格（候选 DA 目标） |
| **rt_cq_price** | DECIMAL(14,4) | **实时出清价格(元/MWh)** | 价格（候选 RT 目标） |
| da_power / rt_power | DECIMAL | 日前/实时出力(MW) | 出力 |
| da_energy / rt_energy | DECIMAL | 日前/实时电量(MWh) | 电量 |
| da_status / rt_status | VARCHAR | 日前/实时开机状态 | 机组状态 |
| create_time / update_time | DATETIME | 创建/更新时间（ON UPDATE） | 入库审计（§9 关键） |

键：UNIQUE (market_date, period_no, unit_id)。覆盖：2022-01-01→2026-07-18（业务日），1660 天 × 每天恰 96 行，零缺日；**单机组**；较小时表滞后约 9 个业务日 `[数据@07-26]`。

**`epf_market_data_96`**（市场级 96 点特征，**无价格列**）`[代码+快照]`：时间键同上（market_date/period_no/data_time）；特征 13 组 actual+13 组 fcast——按类别：负荷（直调负荷 direct_load）、发电（地方电厂 local_plant、自备 self_owned、试验机组 test_unit、核电 nuclear）、新能源（风电 wind、光伏 solar、新能源总加 new_energy）、备用（正备用 pos_reserve、负备用 neg_reserve）、检修（机组检修 unit_maintenance）、联络线（外电 tie_line）、市场变量（竞价空间 bidding_space）；每列均有 actual/fcast 两版。本地快照实测：actual 按日后补、fcast 提前可得；period_no 1..96、区间末标注、p96=D+1 00:00 与小时约定同构。DB 侧历史深度**未测**（v2 脚本覆盖）。

### 3.2 「爬虫曾采集 15 分钟日前/实时价」的代码证据

爬虫仅有两个价格端点，均为**机组级**（`scripts/crawler/crawl.py:229-237`）：`DaJyjgfbPlantQuery24.do`（日前出清 96 点，字段 cqPrice/power/energy/kt）与 `YxJyjgfbPlantQuery24.do`（实时出清 96 点，同字段），按 `unitid` 请求；市场总览端点 `DaJyxxPlDa.do`（`crawl.py:180-215`）只采集负荷/新能源/自备等特征列，**无价格字段**。历史回填（`scripts/backfill_unit_data_96.py`）与智能补缺（`auto_fill_96.py`）写的都是这两张表。**本仓库不存在小时表 `epf_market_data` 的写入代码**（全仓 grep 仅 SELECT）——小时级市场价的采集在外部（历史 epf 项目/旧爬虫）`[代码]`。因此就本仓证据而言：**「15 分钟 DA/RT 价格」= 单机组出清价 `da_cq_price/rt_cq_price`**；若业主所指另有市场级 15 分钟价来源（旧爬虫或其他表），需 v2 全库扫描 + 业主指认（→ §15 Q1/Q2）。

### 3.3 候选目标价列汇总

| 分辨率 | DA 目标候选 | RT 目标候选 | 主体级别 | 状态 |
|---|---|---|---|---|
| 小时 | `epf_market_data.price_dayahead` | `.price_realtime` | 市场级 | 现役 |
| 15 分钟 | `epf_unit_data_96.da_cq_price` | `.rt_cq_price` | **机组级（单机组）** | 唯一已证实来源；与市场价等价性未验证（价格聚合实验在 v1 脚本 §5，待跑） |
| 15 分钟（其他表） | ？ | ？ | ？ | 待 v2 全库扫描证实或证伪 |

---

## 4. 小时数据官方同步刷新（本轮执行结果）

- **官方命令**（`cli/parser.py:161-175` + README §5.3）：
  `python main.py --pipeline sync_dataset --sync-source auto --force-sync --require-fresh-data`
  （源可选 `db`/`http`/`local`/`auto`；db 走 `utils/database_operate.fetch_web_grid_data`；http 走 `http://qiniu.dirx.com.cn/workspace/eprice_forecast/shandong_pmos_hourly_20220101_YYYYMMDD.xlsx` 逐日回溯 60 天，`sync_data.py:30,63-87`；输出 `data/shandong_pmos_hourly.xlsx/.csv` + `outputs/data_sync/sync_manifest.json`。）
- **本环境执行结果：未成功，未伪造。** 精确阻塞（非敏感）：① 云沙箱→DB 主机 TCP 超时（出站白名单）；② `pymysql`/`python-dotenv` 无法安装（pypi 与 apt 源均被出口代理 403），`sync_data.py` 顶层即 import 失败；③ http 源域名同样不在白名单。设备侧 VM shell 亦不可用。
- **须在本地（办公/开发机，epf-2 环境）执行**：上面那条官方命令原样运行即可；成功标志为输出 JSON `status: ok` 且 `outputs/data_sync/sync_manifest.json` 的 `max_timestamp` 接近当日。**顺带**请同轮运行两个只读巡检脚本（§8）。
- 本地现存快照（供参考，非本轮刷新）：`data/shandong_pmos_hourly.csv`，39,816 行，最新时间戳 **2026-07-18 00:00**（=业务日 07-17 h24），同步于 07-16（`outputs/data_sync/sync_manifest.json`）；DB 侧 07-26 实测已达业务日 07-27。

---

## 5.–6. 实时截止（14:00）核查 与 物理/业务时间语义

### 5.1 运行态证据（支持 14:00）

1. CLI 默认 `--realtime-cutoff-hour 14`（`cli/parser.py:140`）`[代码]`。
2. **生产 manifest 实样**：目标日 2026-02-24 的 `run_manifest.json`：`realtime_cutoff_hour=14`、`data_cutoff_dayahead=2026-02-23`、`data_cutoff_realtime=2026-02-23 14:00:00`（fixtures/repro_bundle）`[数据]` —— 即预测 T 日时 RT 截止于 D=T−1 的 14:00，与业主陈述一致。
3. 遮蔽实现均为时间戳比较 `ts > cutoff_ts`（SGDFNet `protocol_b_cutoff.py:180-184`；TimeMixer 过去窗 `repro_pipeline.py:288-291`；RT916 `core.py:275-296` cutoff 后 RT←DA 替换+特征重算；LightGBM `infer_fix.py:220-221` 的 `start−11h` ≡ D 14:00）`[代码]`。
4. `[矛盾]`（已终裁）：`docs/项目执行逻辑与陪跑步骤对齐.md` 写 15:00；RT916 独立 CLI 默认 15。2026-07-28 业主终裁**固定 14:00 / `realtime_cutoff_period = 56`**（见 `LEAKAGE_AUDIT_96.md` §3）；15:00 仅为历史歧义记录，不再作为候选，相关文档已订正。

### 5.2 「历史表已回填」警示（本轮重要发现）

**不能用最终入库的历史行反推预测时点可得性**：本地快照（07-16 11:50 BJT 同步）中，业务日 07-16（同步当天）的实时电价**整日为 NaN**、07-15 及更早整日齐全 —— 即经「爬虫每日 08:00 → MySQL → sync」路径，运行日当天的盘中 RT 值（含 14:00 前）**并不在项目数据文件里**；实际信息边界比 D 14:00 更保守（≈D−1 24:00）`[数据+README §17]`。这意味着：(a) 代码的 14:00 遮蔽当前起「保险丝」作用而非「刚好卡线」；(b) 若业务上要求真用到 D 日 14:00 前的 RT 值，需要盘中增量爬取/同步（现状没有）——这是**待业主确认的运营事实**（§15 Q6），不是本次要改的东西；(c) 判定「14:00 值是否在预测运行前发布」必须用**入库时间戳**审计：`epf_unit_data_96` 具备 `create_time/update_time`（DDL 确认），v2 脚本含该审计；小时表是否有等价列待扫描。

### 5.3 标注语义与 24↔96 映射表（现约定 = 区间末，已三方验证）

小时：label h 覆盖 (h−1):00→h:00；`hour_business=14` 代表 **13:00–14:00**；h24=D+1 00:00。96 点：`data_time` 注释「区间结束时间」`[DDL]`；p1=00:00–00:15（标 00:15）…p96=23:45–24:00（标 D+1 00:00）；爬虫标签 `00:15→1, 24:00→96`（`crawl.py:17-23`）。

| 业务日 | 物理区间 | 小时索引 | 96 索引 | ds（区间末） | DA 可得性（预测 T=D+1 时） | RT 可得性（14:00 截止，理论） |
|---|---|---|---|---|---|---|
| D | 00:00–00:15 | h1 的一部分 | p1 | D 00:15 | 全天已知（D 日发布于 D−1） | 已知 |
| D | 00:45–01:00 | h1 | p4 | D 01:00 | 已知 | 已知 |
| D | 13:00–14:00 | **h14** | p53–p56 | …D 14:00 | 已知 | **已知（最后可用：h14 / p56）** |
| D | 14:00–15:00 | **h15** | p57–p60 | …D 15:00 | 已知 | **未知（首个遮蔽：h15 / p57）** |
| D | 23:45–24:00 | h24 | p96 | D+1 00:00 | 已知 | 未知 |
| T=D+1 | 全天 | h1..24 | p1..96 | | 待预测（目标日 DA 价通常未发布→NaN 行保留） | 待预测 |

即：在「`ds ≤ cutoff_ts` 可见」的现行比较式下，**14:00 这个值本身（作为 13:45–14:00 段的区间末）属于可见区**；标 15 的值（14:00–15:00）不可见。96 点下边界为 **p56 可见 / p57 遮蔽**——该结论由现有代码语义自动导出，无需新写死。**2026-07-28 业主已终裁固定 14:00 / p56（原「是否含 14:00 整点」与 14:00/15:00 口径统一两个问题均已关闭，见 `LEAKAGE_AUDIT_96.md` §3）；「发布延迟」仍属运营事实（§15 Q6），非截止口径问题。**本轮不实现新时间抽象**；上表即「现状语义 + 期望语义」的确认基准。

---

## 7. 模型逐腿最小兼容评估（12 问框架）

> 通用结论先行：六条腿**全部**属于「模型本体不动、改数据构造/参数」的范畴；除 TimesFM 外重训练本来就是每次运行的常态（§2 表）；**没有任何一条腿需要更换或新增模型**。「hourly 模式回归不变」通过让全部新参数默认 24/60/现值来保证。

| 问题 | LightGBM (DA) | TimesFM (DA+RT) | TimeMixer (DA+RT) | RT916 (RT) | SGDFNet (RT) |
|---|---|---|---|---|---|
| ①输入 | `data/shandong_pmos_hourly.xlsx` 宽表 | 同左 | 同左 | 同左 | 同左（configs data_path） |
| ②目标列 | 日前电价 | 日前/实时电价（关键词自动识别） | 两列 | 实时电价（DA 内部自产） | 实时电价（delta 对 da_anchor） |
| ③样本粒度 | 1 行=1 点（表格回归） | 分段序列（段=8 点） | 1 天=1 样本（过去168+未来24） | 1 段日=8 行窗口 | 1 行=1 点 |
| ④物理时长常量（须×4） | shift 24/48/168、rolling24、morning 1..15 | 无（上下文=全段历史） | diff/rolling 24/168、seq_len168 | lag 24/48/72/168/336 | shift/rolling 6/12/24/168 |
| ⑤纯张量尺寸（参数化即可） | 无 | horizon 24、段长 24//3 | pred_len 24/8、future 24 行 | OUTPUT_LEN 8、seq 72 | 无 |
| ⑥可参数化维度 | N=24/96 全覆盖 | N + step("h"/"15min") | N + seq_len + 段表 | N + OUTPUT_LEN | N(PPD) |
| ⑦正确性必改 | 滞后×4、hour 索引 1..N、时段表×4、`validate_business_day_filled` 网格 | `floor("h")`→`floor(step)`、`_delta_hours` 下限=step、horizon 守卫、日帧 N | 24 行断言、`business_hour`→1..N、SEGMENTS×4、长度8权重表→32（否则**静默失效**）、按列索引取通道处 | dataprocess 滞后×4、`hour=dt.hour+1` 加刻度、pipeline `+1h`→`+15min` | `hour==0` 业务日 bug（须 `&minute==0`）、滞后×4、`groupby("hour")`→按刻度、train_min_rows 2160→8640 |
| ⑧仅性能调优 | min 2000 行阈值口径 | 分段 vs 不分段（96≤256 均可） | seq_len 672 vs 336、scales、MovingAvg(25) | TOP_K、epochs | 无（HGB 近线性） |
| ⑨因目标分辨率需重训 | 是（本就每 run 重训） | **否**（零样本；checkpoint 频率无关、horizon 256≥96，`timesfm_2p5_base.py:88-91`） | 是（本就每 run 重训） | 是（同） | 是（同） |
| ⑩模型代码不动、仅改 wrapper/数据层？ | 接近（模型=sklearn/LGBM API；改动均在特征构造） | **是**（`src/timesfm/` 零改动） | 模型 `backbones.py` 基本不动（MovingAvg 可选调） | `model.py/annual_*` 不动 | `models.py` 不动 |
| ⑪hourly 回归不变 | 参数默认 24 ⇒ 逐字节等价（须回归验证） | 同 | 同 | 同 | 同 |
| ⑫最小补丁面（文件） | 6 个（train/infer×2 + main_fix + pipeline） | **2 个**（price_forecast_copy_分时段预测.py、infer.py） | 2 个（repro_pipeline.py、pipeline.py 校验段） | 3 个（src/dataprocess.py、core.py 常量、pipeline.py） | 3 个（data_contract.py、configs×2） |

补充：SGDFNet 96 点的 `da_anchor` 来源顺理成章 = 96 点数据文件的「日前电价」列（即选定的 96 点 DA 目标列本身，目标日缺失时沿用中位数回退逻辑，`data_contract.py:137-188` 无需改动）——**不需要**为它接官方 DA 预测（那是 §1.3 的需求变更议题）。分类器：初版**保持小时级不动**；若业务要求 96 点交付也带 −80 修正，仅需后续在 `classifier_bridge.merge_clf_results`（`classifier_bridge.py:78-95`）加「按小时 join 广播到 4 刻度」小改（列为可选项，见 §12）。

---

## 8. 预测顺序保持（核对结论）

现行真实顺序：同步/校验 → 阶段1 `ledger_predict`（**内部** DA 组先、RT 组后；组内经调度器 CPU 并行/GPU 串行）→ 阶段2 权重 → 阶段3 融合（DA、RT 各自融合）→ 阶段4 分类器（仅 RT）→ 阶段5 final_outputs → postflight/fallback。与任务书 §8 的概念序列相比：**「产出供依赖 RT 模型使用的官方 DA 结果」这一步在现实现中不存在**（§1.3）——即概念第 3-4 步（先融合 DA 再喂 RT）与实际（先出全部腿预测、后统一融合）不同。96 点兼容方案**保持现有实际顺序不变**（依赖正确性已由「DA 组先行 + 各腿内部自产 DA」满足），CPU/GPU 并行策略原样保留。

---

## 9.（见 §5.2 已并入截止核查）

## 10.（见 §5.3 已并入时间映射）

## 11. 逐文件最小改动面与分类

**分类图例**：A=正确性必需 / B=数据兼容必需 / C=输出校验必需 / D=可选清理 / E=未来优化 / F=范围外。

| 文件 | 改动 | 类 |
|---|---|---|
| **数据层** 新增 `build_dataset_96`（建议放 `sync_data_96.py` 内或旁挂小模块） | 把 `epf_unit_data_96`（价）+ `epf_market_data_96`（特征）按 (market_date, period_no) 合成一张与 `shandong_pmos_hourly` **同列名**的 15 分钟宽表（`时刻/日前电价/实时电价/直调负荷预测值/...`）→ 各腿仅需换 `--data-path` + 点数参数即可复用全部现有列映射代码 | **B**（整个方案的杠杆点） |
| `cli/parser.py` | 加 `--points-per-day {24,96}`（默认 24）贯通下发；沿用**现有** `--data-path/--ledger-root/--runs-root` 实现 96 点独立树（`outputs/ledger_96`、`outputs/runs_96`）——**零新机制** | A |
| `utils/business_day.py` | **不重写**。追加两个带默认值的参数：`validate_daily_predictions(..., expected_points=24)`、`infer_period` 支持 96 槽映射（1_8↔1..32 等）；`business_day/hour_business_from_timestamp` 增加 15 分钟感知分支（默认路径不变） | A |
| `pipelines/ledger_predict.py` | `expected={"dayahead":3N,"realtime":4N}`（:713，修 D9 命名坑）；actuals 抽取窗起点 `D 01:00`→`D 00:15`（:786-790）；点数参数下发 | A/C |
| `pipelines/ledger_weight.py` | `_EXPECTED_HOURS`→1..N（:45）；`expected_rows=days×models×N`（:504） | C |
| `fusion/apply_daily_ledger_weights.py` + `ledger_fuse.py` | 融合循环 `range(1,25)`→`range(1,N+1)`（:77）；24 行断言参数化 | A |
| `fusion/contracts.py` | `VALID_PERIODS` 保持三段不变；`infer_period` 接受 1..96 槽（按 N 分支） | A |
| `fusion/adapters/*.py`（5 个） | `dt.hour.replace({0:24})` 惯用法→按 N 的刻度索引（每文件 2-3 行） | A |
| `pipelines/delivery_quality.py`、`ledger_full.py`、`ledger_full_range.py`、`emergency_fallback.py` | 24 行/1..24 槽/h24-ds 校验全部改为 `expected_points` 参数（默认 24）；fallback 中位数按槽分组；96 模式产出 `submission_ready.csv` **同 6 列契约、96 行**（`hour_business` 列承载 1..96 槽号——列名不改，行数与索引域变化；是否可接受列入 §15 Q9） | C |
| 模型 5 条腿 | §7 表 ⑦ 列，合计 16 个文件 | A/B |
| `pipelines/prediction_ledger.py` | **不改键、不改 schema**：96 点走独立 `--ledger-root`，物理隔离即无碰撞（键内 `hour_business`=槽号 1..96 语义自洽）；仅 `check_ledger_coverage n_expected` 参数化（:464-469） | B/C |
| `daily_ledger_gef.py` | 仅配置值：`evidence_prior` 按点数重标定（或 evidence 改按日累计）；periods 沿用三段 | C/E |
| 校验脚本 `verify_final_pipeline.py`/`check_timemixer_alignment.py`/`check_delivery_stability.py` | 常量参数化 + 96 合成用例 | C |
| `classifier_bridge.py` 广播修正 | 小时 join → 4 刻度广播（+覆盖检查 `+1h`→`+Δ`） | **D**（初版可不做；业务要求时再开） |
| README/RUNBOOK 双分辨率章节、cutoff 文档 15:00 订正（已完成：统一为 14:00 / `realtime_cutoff_period = 56`，见 `LEAKAGE_AUDIT_96.md` §3） | 文档 | DONE |
| 统一抽象层/Resolution 数据类、账本 schema 加列、SUBMISSION_COLUMNS 收敛重构、字段改名、新基线模型、分类器移植、盘中增量爬取 | — | **F（范围外）** |

**保持不动的模块**（明确承诺）：`runtime/resource_scheduler.py`、`runners/registry.py`、`runners/executor.py`、`main.py`、交付状态机与 exit code、`emergency_fallback` 策略逻辑（仅循环界参数化）、`delivery_report.py`、BGEW 更新数学、SGDFNet `models.py` 与截断核心、RT916 `model.py/annual_*`/`policy.py`、TimesFM `src/timesfm/` 全部、分类器全部核心、爬虫与同步脚本（除新增 96 数据集构建函数）、小时级账本与小时级输出契约（逐字节不变，黄金基线回归钉死）。

**静默语义错误风险清单**（实施时必须以测试拦截，沿用前版编号）：D1/D2 分钟盲業務日与小时映射、D3 账本键（已用独立根目录规避）、D4 TimesFM `floor("h")`、D5 SGDFNet `hour==0`、D6 adapters 惯用法、**D7 行数滞后语义漂移（最高危）**、D8 套利指标缺 0.25 时长因子（96 点报表启用时必须加）、D9 「96=4×24」命名坑、新增 D10：TimeMixer 长度 8 权重表在 32 点段静默不生效。

---

## 12.–14. 修正后的分阶段计划（以兼容为纲，未获批不实施）

**Round 0 — 业务与数据确认（无代码，本轮已就绪）**：业主答复 §15 问题；办公机执行三件套：官方小时同步命令（§4）、`output/_db_inspect_hourly_vs_96.py`（价格聚合实验）、`output/_db_inspect_full_schema.py`（全库清点 + 入库时间戳审计）；恢复 96 爬虫日更。产物：确认纪要 + 两份 JSON。

**Round 1 — 数据兼容层（B 类）**：`build_dataset_96` 合成同列名 15 分钟宽表 + 完整性/对齐校验（96 行/日、p96=D+1 00:00、与小时 mean 一致性抽检）。不触碰任何管线。验收：数据集校验报告全绿。

**Round 2 — 模型腿点数参数化（A 类，按依赖序）**：TimesFM（2 文件）→ LightGBM（6 文件）→ TimeMixer/RT916 → SGDFNet（含 D5 修复）。每腿两条验收：(a) 96 点 smoke 输出 96 行合法帧；(b) **hourly 黄金基线逐字节回归不变**。全程独立输出目录，不入账本。

**Round 3 — 编排与校验参数化（A/C 类）**：§11 表中 pipelines/fusion/校验脚本各点；96 点独立树跑通五阶段 `NORMAL / exit 0 / 96 行 0 NaN`；小时链四件回归（40/40、29/29、16/16、41/41）+ 黄金基线不变。

**Round 4 — 可选项（D 类，逐项经业主点菜）**：分类器广播修正；96 点业务指标报表（含 0.25 因子）；文档订正；连续陪跑验收。

每轮回滚策略：Round 1 仅新增文件；Round 2/3 所有改动带默认值参数（默认=现行为），revert 单文件粒度可行；96 点数据与输出全程独立目录，可整体删除不留痕。

---

## 15. 待业主确认的问题清单（回答前不动工）

1. 官方 15 分钟 DA/RT 目标是哪张表哪两列？（现仓库唯一已证实来源：`epf_unit_data_96.da_cq_price/rt_cq_price`，机组级出清价；「爬虫曾采集 15 分钟价」在本仓即指此二列——若另有市场级来源请指认，全库扫描脚本可验证）
2. 目标是市场级、节点级还是机组级？单机组是否足够（多机组⇒键含 unit_id、爬虫扩容）？
3. `da_cq_price/rt_cq_price` 可否作为正式目标？（配套证据：待跑的价格聚合实验——机组 96 价 mean 聚合 vs 市场小时价）
4. 15 分钟区间的时间戳约定确认：沿用现状**区间末**标注（p1=00:15、p96=D+1 00:00，与 DB/爬虫/小时约定三方一致）？
5. 14:00 截止下，13:45–14:00 段（标注 14:00，p56）确认**属于可见区**？（现行 `ds≤cutoff` 语义如此）
6. 生产数据源是否在预测运行前发布/入库当日 14:00 前的 RT 值？（§5.2 证据显示现有同步路径运行日当天 RT 整日缺失，实际边界≈D−1 24:00——业务上接受现状，还是要求补盘中增量同步？后者为独立立项）
7. 24 与 96 两模式须同部署共存（配置切换 `--points-per-day`），还是部署期二选一？是否需要同日双输出？
8. 96 点输出契约：接受「同 6 列、96 行、`hour_business` 列承载 1..96 槽号」，还是要求改列名/加列（那将偏离最小改动）？96 点产物是正式提交件还是内部分析件（决定是否走 NORMAL/postflight 全套）？
9. 分类器初版保持小时级（96 点交付不带 −80 修正，或按小时广播修正）——选哪个？
10. 七条腿是否必须全部支持 96 点才算交付，还是允许按 Round 2 顺序增量上线（建议先 TimesFM+LightGBM 出可用 96 点链）？
11. 96 点模式的精度与运维验收标准（capped-SMAPE 阈值不可直接沿用小时口径；建议以朴素季节参照的相对口径另定——注意：该「朴素参照」仅作**验收标尺**，不新增生产模型）。
12. SGDFNet 锚定语义确认：维持现状「数据列 da_anchor + 中位数回退」（§1.3 矛盾项），还是要求改接官方 DA 融合结果（=需求变更，另立项）？

---

## 16. 本轮产物

- 本文：`docs/PLAN_96_POINT_MINIMAL_COMPATIBILITY_ASSESSMENT.md`（取代前版规划的实施取向）。
- 新增只读脚本：`output/_db_inspect_full_schema.py`（全库 schema/列/注释清点 + create_time/update_time 可得性审计；与既有 `output/_db_inspect_hourly_vs_96.py` 配套，均临时、无凭据、gitignore 目录内）。
- 未修改任何生产代码、模型、数据库、账本、融合、分类器、小时输出、截止配置。等待业主逐条评审 §15 后再启动 Round 1。
