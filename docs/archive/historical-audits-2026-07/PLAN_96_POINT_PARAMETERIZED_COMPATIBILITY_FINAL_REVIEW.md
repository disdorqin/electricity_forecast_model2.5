# 24/96 点参数化兼容 — 实施前最终设计评审稿（v2：数据先行闸门版）

- **日期**：2026-07-26（v2）｜ **性质**：只读数据审计 + 设计更新（**未实施**；未改任何生产代码/CLI/模型/账本/融合/分类器/输出/截止/爬虫/数据库）
- **v2 变更**：并入「96 点数据完整审计」轮的要求与结果；新增数据先行硬闸门、数据字典、目标候选比较、历史对齐、可得性矩阵、原生分类器事件时长设计、官方修正输出契约、5 族/7 腿口径与停止条件。
- **后续更新（2026-08-01）**：文内 `sync_data_96.py` 已被 `sync_data_96_core.py`（统一 CLI `main.py --pipeline sync_dataset --resolution 15min`）取代并删除；96 点数据现同步至 `data/remote_96/parquet/`，合并宽表见 `build_96_full_table.py`。
- **结论标记**：`LIVE VERIFIED`（远程库实测）/ `LOCAL DATA VERIFIED`（本地快照全量分析）/ `CODE VERIFIED`（代码实证，附文件:行号）/ `OWNER APPROVED`（业主批准）/ `PENDING LOCAL RUN`（须办公机执行）/ `UNRESOLVED`（未决）。

---

## 0. 数据先行硬闸门（Data-First Hard Gate）与当前判定

> **原则**：任何模型/分类器/账本/融合/输出兼容工作，必须在 96 点数据被识别、对齐并证明充分之后才能开始。缺数据时停止并报告，不猜列、不造目标、不静默改预测口径。`OWNER APPROVED`

**停止条件状态表（当前判定：🔴 STOP——实施不得开始）**：

| # | 停止条件 | 状态 | 依据 |
|---|---|---|---|
| S1 | 官方 96 点 DA 目标未确认 | 🔴 未确认 | 唯一已证实候选为机组级 `da_cq_price`；全库扫描未跑（S10）→ 选定须待扫描+业主评审 `UNRESOLVED` |
| S2 | 官方 96 点 RT 目标未确认 | 🔴 未确认 | 同上（`rt_cq_price`） `UNRESOLVED` |
| S3 | 特征/目标公共历史不足 | 🟡 未知 | 价格侧 1,660 完整日 `LIVE VERIFIED`；market_96 特征表 DB 历史深度未测（本地仅 2 天）→ 公共完整日数未知 `PENDING LOCAL RUN` |
| S4 | 关键输入预测时可得性未知 | 🟡 未知 | 入库时间戳审计未跑（脚本已备） `PENDING LOCAL RUN` |
| S5 | 爬虫产出不完整/过期 | 🔴 命中 | unit 表滞后约 9 个业务日（07-18 vs 小时表 07-27）`LIVE VERIFIED`；须恢复日更并补缺 |
| S6 | 96 点逐日完整性破损 | 🟢 未命中 | unit 表 1,660 天每天恰 96 行、零缺日 `LIVE VERIFIED`（截至其覆盖终点） |
| S7 | 市场级 vs 机组级实体口径未决 | 🔴 命中 | 单机组 `LIVE VERIFIED`；与市场价等价性实验未跑 `PENDING LOCAL RUN`；业务口径待业主 `UNRESOLVED` |
| S8 | 本地无法复现完整 96 点数据 | 🔴 命中 | 本地 `unit_data_96.*` 不存在、`shandong_pmos_96.*` 仅 2 天 `LOCAL DATA VERIFIED`；须本地跑 `sync_data_96.py`（注：`sync_data_96.py` 已删除，改由 `main.py --pipeline sync_dataset --resolution 15min` 同步至 `data/remote_96/parquet/`） |
| S9 | 无法构造防泄漏训练表 | 🟡 未知 | 依赖 S3/S4 结果 |
| S10 | 全库 schema 扫描未在有 DB 权限的环境执行 | 🔴 命中 | 云端出站被白名单拦截（TCP 复测超时）+ 依赖安装被 403；脚本已交付 `PENDING LOCAL RUN` |

**解除路径**：办公机执行 §16 命令（三脚本 + 同步）→ 扫描/实验结果回填本文实测占位 → 业主拍板 S1/S2/S7 → 闸门复评。**在此之前 R1 不启动。**`OWNER APPROVED`

---

## 1. 已批准决定（不再是开放问题）`OWNER APPROVED`

1.1 同项目双分辨率，单次 run 单分辨率，`--resolution {hourly|15min}` 概念参数，缺省=现行 24 点；1.2 参数化兼容（保仓库/入口/模型组合/DA→RT 顺序/CPU-GPU/失败隔离/fallback/manifest/交付状态/exit code，不建平行仓库，不做无关重构）；1.3 终态 **5 个模型族 / 7 条任务腿**全部支持 96 点（DA：LightGBM、TimesFM、TimeMixer；RT：TimesFM、TimeMixer、RT916、SGDFNet），允许逐腿开发；1.4 SGDFNet 按现实态适配（96 点 DA 价列作数据锚 + 同刻度历史中位数回退；不喂官方融合 DA；意图-实态错配记 3.0 技术债）；1.5 截止 14:00 定案（区间末：h14/p56 可见，h15/p57 遮蔽；唯一遗留问题=现数据路径能否在预测前提供 D 日 14:00 前 RT）；1.6 时间索引：`period_no 1..96` + `data_time 区间末` + `hour_business=ceil(p/4)∈1..24` + 可选 `quarter_in_hour`，禁止 hour_business 冒充 1..96（五个样例 p1/p4/p56/p57/p96 已核验 `CODE VERIFIED`）；1.7 终态**原生** 15 分钟分类器（不永久省略、不以小时广播为终态）；1.8 **96 点官方输出采用分类器修正后 RT**，修正前后工件都保留；小时模式维持 2.5 现行为（官方输出不加修正）。

## 2.–3. 96 点数据字典（现有证据版 + 待扫描补全）

**检索面**：远程库元数据（待跑）、表/列注释（DDL 已得）、视图（待扫描）、爬虫代码、同步代码、历史回填、归档摄入代码、本地快照、SQL 迁移、查询定义、README/RUNBOOK、既有 manifest 审计——不限于两张已知表；中英双语关键词全集已编入扫描脚本。`CODE VERIFIED`

**字典（已证实部分）**——逐字段：

**表 `epf_unit_data_96`**（InnoDB；表注释「机组级电力市场96点数据(15分钟粒度)」；UNIQUE(market_date, period_no, unit_id)；`scripts/db_migrations/001_create_epf_unit_data_96.sql` `CODE VERIFIED`）

| 列 | 类型 | 注释 | 类别 | DA/RT | actual/…| 级别 | 目标候选 | 特征候选 | 预测时可得 | 证据 |
|---|---|---|---|---|---|---|---|---|---|---|
| market_date | DATE | 市场日期 | 时间 | 共用 | — | — | — | 键 | 是 | DDL |
| period_no | INT | 96点序号1-96 | 时间 | 共用 | — | — | — | 键 | 是 | DDL |
| data_time | DATETIME | 完整时刻(**区间结束时间**) | 时间 | 共用 | — | — | — | 键 | 是 | DDL |
| unit_id | VARCHAR(64) | 机组ID | 主体 | 共用 | — | 机组 | — | 键 | 是 | DDL；distinct=1 `LIVE VERIFIED` |
| **da_cq_price** | DECIMAL(14,4) | 日前出清价格(元/MWh) | 价格 | DA | actual(出清) | 机组 | **候选** | 亦可作 RT 锚 | D 日发布节奏待审计 | DDL+爬虫端点 |
| **rt_cq_price** | DECIMAL(14,4) | 实时出清价格(元/MWh) | 价格 | RT | actual(出清) | 机组 | **候选** | 滞后特征 | 事后结算，须 create/update 审计 | 同上 |
| da_power/rt_power | DECIMAL | 日前/实时出力(MW) | 机组出力 | DA/RT | actual | 机组 | 否 | 候选 | 同价格节奏 | DDL |
| da_energy/rt_energy | DECIMAL | 日前/实时电量(MWh) | 机组电量 | DA/RT | actual | 机组 | 否 | 候选（电量加权实验用） | 同上 | DDL |
| da_status/rt_status | VARCHAR | 日前/实时开机状态 | 机组状态 | DA/RT | status | 机组 | 否 | 候选（停机期数据形态核查） | 同上 | DDL |
| create_time/update_time | DATETIME | 创建/更新时间(ON UPDATE) | 审计 | — | — | — | — | 可得性审计键 | — | DDL |

覆盖统计 `LIVE VERIFIED(07-26)`：2022-01-01→2026-07-18（业务日），1,660 天，159,360 行=1,660×96×1，**每天恰 96 行零缺日**；单机组；较小时表滞后≈9 业务日。缺失率/重复率/价格逐列空值率 → `PENDING LOCAL RUN`（脚本 B 节）。

**表 `epf_market_data_96`**（市场级 96 点特征；**无价格列**；键 (market_date, period_no) upsert `CODE VERIFIED`）——26 个业务列 = 13 组 × {actual, fcast}：

| 类别 | 列（每列有 actual_ / fcast_ 两版） | 级别 | 目标候选 | 特征候选 | 预测时可得 |
|---|---|---|---|---|---|
| 负荷 | direct_load 直调负荷 | 市场 | 否 | 是 | fcast 提前可得、actual 按日后补 `LOCAL DATA VERIFIED` |
| 发电 | local_plant 地方电厂 / self_owned 自备 / test_unit 试验机组 / nuclear 核电 | 市场 | 否 | 是 | 同上 |
| 风电/光伏/新能源总加 | wind / solar / new_energy | 市场 | 否 | 是 | 同上 |
| 正/负备用 | pos_reserve / neg_reserve | 市场 | 否 | 是（小时表**没有**的增量特征） | 同上 |
| 检修 | unit_maintenance | 市场 | 否 | 是（增量特征） | 同上 |
| 联络线 | tie_line 外电 | 市场 | 否 | 是 | 同上 |
| 竞价空间 | bidding_space | 市场 | 否 | 是 | 同上 |
| 日历/时间 | market_date, period_no, data_time（+项目侧派生 hour_business/quarter_in_hour/星期/节假日/节气） | — | — | 是 | 是 |

DB 侧历史深度/行数/create_time 列有无 → `PENDING LOCAL RUN`。爬虫来源：市场总览端点 `DaJyxxPlDa.do`（无价格字段）+ 机组端点 `DaJyjgfbPlantQuery24.do`/`YxJyjgfbPlantQuery24.do`（cqPrice/power/energy/kt）`CODE VERIFIED`。项目内使用方：`utils/database_operate.fetch_market_data_96/fetch_unit_data_96`、`sync_data_96.py`、`check_data_freshness.py`、回填与补缺爬虫。

**其余表/视图/历史表**：`PENDING LOCAL RUN`（`_db_inspect_full_schema.py` + v3 脚本自动清点；本仓代码引用的 DB 表全集仅上述两张 96 表 + 小时表 `epf_market_data`，全仓 grep 证实 `CODE VERIFIED`）。

## 4. 候选 96 点 DA/RT 目标价对比（未定案）

| 项 | 候选①（现仓唯一已证实） | 候选②（若扫描发现市场级 96 价表） |
|---|---|---|
| 表.列 | `epf_unit_data_96.da_cq_price / rt_cq_price` | 待扫描 `UNRESOLVED` |
| 注释 | 日前/实时**出清**价格(元/MWh) | — |
| 级别/实体数 | 机组级 / 1 | — |
| 实体间价格差异 | 单实体不适用；多机组后需检验 | — |
| 96 期完整 | 是（1,660 完整日） | — |
| 历史/新鲜度 | 2022 起 / 滞后 9 日 | — |
| 可联特征表 | 同键 (market_date, period_no) 天然 join | — |
| 爬虫维护 | 是（run_crawler + auto_fill_96 + backfill） | — |
| 是否客户所需价 | `UNRESOLVED`（业务语义） | — |
| 数值上聚合≈小时市场价？ | `PENDING LOCAL RUN`（v3 脚本 C 节：逐实体 mean/first/last/min/max/电量加权 × DA/RT × 最近 15 重叠日 → MAE/RMSE/最大差/相关/容差精确率/对比与剔除小时数） | — |

**判定纪律**：高相关 ≠ 精确聚合等价 ≠ 业务语义等价，三者分开报告；除非数据库证据无歧义，目标选择保持待业主评审。`OWNER APPROVED`

## 5. 跨分辨率实验（已完成部分 + 待跑部分）

**特征（`LOCAL DATA VERIFIED`，样本受限须扩样）**：唯一本地重叠业务日 2026-07-17、10 个共有预测特征列 × 5 法：**mean 全列 MAE=0.0000/max=0.0000（精确复现）**；first/last/min/max 平均 MAE 249~259、max 至 2,534（仅 3 个常值列例外全法通过）。多日复核（最近 10 重叠日 × actual+fcast 12 列 × 5 法）→ `PENDING LOCAL RUN`（v3 D 节）。
**价格**：本地零重叠（unit 表未同步到本地）→ 全部 `PENDING LOCAL RUN`（v3 C 节，含电量加权法，权重取 da_energy/rt_energy）。

## 6. 特征-目标历史对齐与训练就绪矩阵（待实测回填）

| 候选目标 | 特征表 | 公共完整日数 | 最新日期 | 新鲜度缺口 | 训练就绪 |
|---|---|---|---|---|---|
| unit_96 cq 价 | epf_market_data_96 | `PENDING LOCAL RUN`（v3 E 节：公共起止、双侧完整日、价格全特征缺清单、特征全价格缺清单、实体随时间变化） | 价格侧 07-18 `LIVE VERIFIED`；特征侧待测 | 价格侧 ~9 日 | **待定**；若 market_96 历史仅始于 2026-07（本地迹象），公共窗口可能远短于价格历史 → 命中 S3，须先修爬虫/回填再谈实施 |

**纪律**：历史缺口不得用假设填补；对齐不足即停并报告「须先修爬虫或数据库」。`OWNER APPROVED`

## 7. 预测时点可得性矩阵（框架 + 已知行）

| 字段/类 | D 日预测前可得？ | 至 D 14:00 可得？ | 事后回填？ | 可安全作输入？ |
|---|---|---|---|---|
| market_96 fcast_*（13 列） | 是（快照：未来日已填）`LOCAL DATA VERIFIED` | 是 | — | 是（候选） |
| market_96 actual_*（13 列） | 仅历史日 | **未知**（快照显示按日后补；盘中节奏待入库戳审计） | 是 | 仅作滞后历史特征 |
| unit_96 da_cq_price（D 日当天值） | 未知（DA 发布 vs 爬虫节奏） | 未知 | 是 | 审计后定 |
| unit_96 rt_cq_price（D 日盘中） | **现路径不可得**（快照运行日 RT 整日缺）`LOCAL DATA VERIFIED` | 未知（若加盘中同步才可能） | 是 | 目前仅 ≤D−1 部分安全 |
| 小时表各列 | 同结构结论（巡检报告 §7/§9） | 同 | 是 | 同 |

判定规则：**「最终出现在历史库」不等于「预测时可用」**；逐列结论以 create_time/update_time + 爬虫/同步排程 + 历史输入快照为准（v3 F 节）→ `PENDING LOCAL RUN`。

## 8. 同步路径执行矩阵（本环境结果 + 待本地）

| 模式 | 执行？ | 结果 | 真实远端刷新？ | 备注 |
|---|---|---|---|---|
| db | 云端不可执行 | 阻塞（DB 出站白名单 + pymysql/dotenv 安装 403） | 是 | 07-16 办公机成功先例：39,816 行、max_ts 07-18 00:00（manifest `LOCAL DATA VERIFIED`；status=failed 仅因 freshness 48h>36h，同步本身成功） |
| http | 云端不可执行 | 阻塞（同依赖 + qiniu 域名不在白名单） | 是（逐日回溯 60 天下载） | 本地 data/ 无 http 命名文件 → 近期未用 |
| local | 逻辑核读 | — | **否**（仅本地候选文件提升为 canonical，`sync_data.py:96-116` `CODE VERIFIED`） | 不算刷新 |
| auto | 云端不可执行 | 阻塞 | db→http→local 依序回退 `CODE VERIFIED` | 待本地四连跑（§16），逐项记录 exit code/选中源/回退序列/行数/max_ts/最新非空 DA/RT 价时间戳/运行日盘中 RT 有无 |

**纪律**：云沙箱网络限制不构成生产路径不可用的证据。`OWNER APPROVED`

## 9. 96 点爬虫健康审计（代码侧已查 + 运行侧待跑）

写入面 `CODE VERIFIED`：market_96 由市场总览端点写 13 组特征（`run_crawler.py:219-`、`auto_fill_96.py:245-`）；unit_96 由两个机组端点写 da/rt 价+出力+电量+状态（cqPrice→cq_price 映射）；upsert 键=唯一键；`period_no_from_time("00:15")=1…"24:00"=96`；data_time=market_date+15p 分钟（区间末）；create/update_time 由 DDL 默认值维护；补缺爬虫 lookback 14 天自动补；Cookie 人工续期（README §17.1）；单 unit_id 由 config.json 决定。**健康问题（须修复后才过闸门，本轮不修）**：① 日更中断——数据止于 07-18（S5）；② 单机组（S7 关联）；③ market_96 历史深度存疑（若无历史回填，特征窗口不足，S3）；④ 修复动作=续 Cookie、跑 `auto_fill_96 --lookback`、必要时跑市场特征历史回填、`sync_data_96.py` 落地本地。

## 10. 原生 96 点负电价分类器：事件时长设计 `OWNER APPROVED（方向）`

**基础点标签**：沿用 `实时电价 ≤ −50`（阈值在 96 点分布检视前不改）。**事件参数**：新增可配置 `min_negative_duration_periods ∈ {1,2,3,4}`，**初始实验默认候选=2（30 分钟）**，非硬编码，终值由验证选定。**事件构造**：严格连续同标签期成事件；v1 不跨正价间断合并；每事件稳定 `event_id`；时长以期数+分钟双记。**确认与追溯修正**：达到最小时长即确认事件，并**追溯覆盖事件内更早的达标期**（p40、p41 连续两期、min=2 ⇒ p40 与 p41 同属确认事件、均可修正——不是从第二期才开始修）；点级概率与标签全程留档审计。**训练/验证指标**：点级 P/R/F1/F2/PR-AUC/FPR/FNR + 事件级召回/精确/起点误差/终点误差/时长误差/漏报数/误报数/被时长规则消除的孤立单期误报占比 + 修正前后 RT 预测业务指标对照。**数据依据（本轮新增）**：小时口径 `LOCAL DATA VERIFIED`（2022-01→2026-07-15）：RT≤−50 共 4,436 小时、1,150 个连续事件，**单小时孤立事件占 28.5%（328 个）**、中位时长 3 小时、均值 3.86 小时——事件天然成段，时长规则有真实抓手；15 分钟连段分布（决定 min_duration 终值）由 v3 G 节从 `rt_cq_price` 实测 → `PENDING LOCAL RUN`。**缩放 vs 重标定**分界、重训练必要性、OOF 重建、两个缺失文件审计盲区维持上版结论（详见 §12-classifier 前版内容）；本轮不实现不训练。

## 11. 官方 96 点输出 = 分类器修正后 RT `OWNER APPROVED`

**−80 规则现语义（`CODE VERIFIED`，`fusion/classifier_bridge.py:90-92`）**：`final_pred==1 且 y_fused ≤ 100` 时把预测**置为 −80.0**（赋值，非减 80、非其他变换）；本设计原样保留。注意区分：LightGBM 腿内部另有自有的负价段修正（DA→−80/RT→−100，`train_da_fix.py:127-128`、`train_fix.py:137-138`），属模型内部行为，与官方分类器修正无关。
**双工件保留**：`realtime_before_classifier`（现 `realtime_final_predictions.csv`）与 `realtime_after_classifier`（现 `_corrected.csv`）都保留；96 点模式 `submission_ready.csv` 的 `realtime_price` 取**修正后**值（与小时 2.5 现行为相反——小时模式不变 `OWNER APPROVED`）。审计字段落入修正明细工件与 `classifier_report.json` 扩展：`classifier_probability / point_label / event_confirmed / event_id / event_duration_periods / event_duration_minutes / correction_applied` 逐期记录。manifest 新增：`classifier_applied_to_official_output=true, classifier_resolution="15min", classifier_price_threshold=-50, classifier_min_duration_periods=<选值>, classifier_correction_rule="-80"`。**连带决定**（实施时）：96 点模式下分类器从「非阻断旁路」升级为官方链路一环——失败处理策略须相应明确（建议：分类器失败 ⇒ 官方输出回退未修正值并记 DEGRADED 类警示，具体待业主确认，列入 §13 遗留问题 Q-N1）。

## 12. 96 点输出契约（优先设计，待最终确认）

```text
business_day, ds, period_no, hour_business, period, dayahead_price, realtime_price
period_no=1..96；hour_business=ceil(period_no/4)∈1..24；period 沿用三段；
dayahead_price=融合 DA；realtime_price=分类器修正后 RT；另存未修正 RT 审计工件
```

客户侧模板核查 `CODE VERIFIED`：仓库内所有列契约消费方为 `delivery_quality.py:19-22`、`ledger_full.py:407`、`ledger_full_range.py:34-37`、`emergency_fallback.py:20-23`（四处重复常量）与校验脚本；README §1 记载 6 列口径；**未发现外部客户模板文件**。7 列 96 行契约与上述四处的参数化改动面已列入变更清单；是否存在仓库外的客户端校验模板 → 待业主确认（Q-N2）。

## 13. 5 族 / 7 腿口径与行数断言（防「96」歧义）

```text
5 个模型族：LightGBM、TimesFM、TimeMixer、RT916、SGDFNet
7 条任务腿：DA 3（lightgbm/timesfm/timemixer） + RT 4（timesfm/timemixer/rt916/sgdfnet）
96 点模式期望行数：DA 长表 3×96=288；RT 长表 4×96=384；合计 672
现存陷阱：代码中 "96" 现指「RT 4 模型×24 小时行」（ledger_predict.py:713、ledger_smoke.py:135、verify_final_pipeline.py:66）
实施要求：期望行数一律写作 n_models × points_per_day 并注释；新增显式断言测试
  assert len(da_long)==3*PPD and len(rt_long)==4*PPD   # PPD=24→72/96；PPD=96→288/384
```

## 14. 设计主体（承接 v1，经 v2 数据轮修订仍然有效）

CLI：单一 `--resolution {hourly,15min}` + 二十行查表助手派生 PPD/分钟/PPH/freq；传播图 15 触点（同步、数据集、adapters、wrappers、训练窗、horizon、ledger/runs root 复用现旗标切 `outputs/ledger_96`/`runs_96`、权重期望行、融合按 period_no 迭代、分类器、校验、输出、manifest（+resolution 字段）、报告、range、backfill/actuals）。逐腿参数化表（TimesFM 2 文件零重训 / LightGBM 6 文件 / TimeMixer 2 / RT916 3 / SGDFNet 3 + A 级 `hour==0&minute==0` 修正）、物理时长×PPD vs 容量重调分界、账本 15min 分支加 `period_no` 列并入键（hourly schema 零变化）、黄金基线逐字节清单与时间戳豁免、96 冒烟/集成/防泄漏测试、注释规范与十大高危注释点、不动模块承诺清单、技术债 TD-1..TD-6——**均维持前轮结论有效**（逐腿 12 问细表与逐文件改动分类见 `docs/PLAN_96_POINT_MINIMAL_COMPATIBILITY_ASSESSMENT.md` §7/§11；实施 PR 按该清单 + 本文 v2 增补逐项核销）。v2 增补：分类器在 96 点为官方链路（§11）→ `ledger_classifier` 严格性策略与 `ledger_full` 的 submission 构建源在 15min 分支切换为 corrected 文件（`ledger_full.py:389` 处的分辨率分支），列入变更清单 C 类。

## 15. 遗留问题（业主）

Q-N1 96 点模式分类器失败时官方输出回退策略（未修正值+DEGRADED 警示？）；Q-N2 是否存在仓库外客户校验模板约束 7 列契约；Q-N3 目标列终选（待扫描+聚合实验证据）；Q-N4 单机组口径与多机组扩容时点；Q-N5 market_96 历史不足时的回填/缩窗策略；Q-N6 盘中增量同步是否立项（决定 RT 特征能否真用到 D 14:00）；Q-N7 min_negative_duration_periods 终值确认流程（实验→业主签字）；Q-N8 96 点验收阈值口径。

## 16. 待本地执行命令（办公/开发机，项目根目录）——v3 起一键化

```powershell
powershell -ExecutionPolicy Bypass -File output\run_local_audit.ps1
```

runner 依次执行 `output/_db_audit_96_live.py`（v4 整合审计，取代 v1/v2/v3 三脚本的查询面并新增：market_96 逐列空值率按年/最早非空日期、unit_96 无过滤单机组判定、create/update 时延分位数、p56 前当日 RT 入库计数、六法价格聚合、异常日期清单，共 12 个机读工件）→ 官方同步四模式 → `sync_data_96.py`；stdout/stderr/exit code 全落 `output\audit_logs\`，同步矩阵写 `output\db_audit_96_sync_matrix.json`。回传全部 `output/db_audit_96_*` 与日志后回填本文 `PENDING LOCAL RUN` 占位并复评 §0 闸门。

**v3 addendum（2026-07-26 血缘轮新证据，详见 `docs/REMOTE_DB_96_DATA_AND_APPLICATION_AUDIT.md`）**：① 历史回填脚本每日同时 upsert market_96（8 个 actual 列）→ 支持「market_96 具 2022 起历史」的业主记忆（待 live 定量）；② market_96 的 fcast_* 13 列与 5 组扩展 actual 列由**仓库外第二写入方**维护（本仓三份写入脚本仅写 8 列、market_96 建表 DDL 不在本仓）→ 其排程决定 fcast 特征可得性与训练窗，live 审计逐列「最早非空 + 时延分布」为裁决证据；③ 本地 2 天快照最可能源于「限定日期/仅 market 的一次 sync」（该路径不写 manifest，与 outputs/data_sync_96 目录缺失自洽），不能反推远程历史；④ 96 点分类器失败的降级交付结构化设计（字段/三层日志/存放建议）已定稿于审计报告 §5，实施留 R4/R5。

---

*v2 未做任何生产改动。§0 闸门为 🔴 STOP：在 S1/S2/S3/S4/S5/S7/S8/S10 解除并获业主书面批准前，不进入 R1 实施。*
