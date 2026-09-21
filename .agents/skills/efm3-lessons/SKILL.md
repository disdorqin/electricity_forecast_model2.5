---
name: efm3-lessons
description: EFM3 电力预测项目（electricity_forecast_model2.5）经验教训与红线。改动本项目任何代码/数据/爬虫/模型前必须先加载本 skill——含数据真实性红线（96点 actual==fcast 拷贝问题）、24点vs96点口径区别、环境约束（epf-2 CPU）、交付纪律、已知坑清单。
license: MIT
compatibility: opencode
metadata:
  audience: coding-agents
  scope: project
---

# EFM3 电力预测项目 — 经验教训与红线

> 本 skill 是本项目踩坑记录。**任何改动（代码/数据/爬虫/模型/文档）前必须读完本文件**。
> 违反红线导致的后果：数据污染、回测失真、交付失败。

---

## 0. 一句话约束

### 当前 formal 96 production override（2026-09-19）

本文件后续历史条目保留为可追溯记录；若与当前正式链路冲突，以本节、代码和
契约测试为准：formal 96 入口为 `python main.py --96 DATE`，DA 为
`lightgbm/timesfm/timemixer`，RT 为 `timesfm/sgdfnet/timemixer/rt916`，
Dynamic-v1 snapshot/FeatureView serving（不按固定小时二次裁剪 RT），CPU split_process=2（DAG-aware），GPU=1 串行，
SGDFNet 使用 D-1 decision-day DA 96 点 anchor，RT916 production stride=24，
权重默认 `smape_reg/SLSQP`。正式阶段是
`ledger_predict → ledger_weight → ledger_fuse → final_outputs`；
`ExtremePriceClf` 为 `disabled_by_production_policy`，仅 legacy/shadow/replay。
历史“默认 nnls”“完整五阶段”“跑重模型前手工设置 RT916_TRAIN_STEPS”均不代表当前
formal 96 生产行为。

- **项目本质**：山东电力现货价预测，24点（小时级）正式交付 + 96点（15分钟级）辅助，7模型 + Ledger 自适应融合 + 极端价分类器。
- **环境**：本机 `conda epf-2`（CPU only，LightGBM/CatBoost 走 GPU 会崩）；GPU 云服务器 RTX3090 需 `export TIMESFM_DEVICE=cpu`。
- **论文红线**：山东/山西数据**绝不进论文**（仅内部动机）；多市场证据用宁夏/甘肃/陕西/青海 + 公开国际集（Lago/NEM/GEFCom/UniElecPrice）。

---

## 0b. 长期运行与部署设计原则

> 任何新增功能、性能优化、缓存、目录或接口设计，都默认以“长期持续运行的服务”而不是“一次性实验脚本”为目标；科研实验可单独放在 `scripts/experiments/` / `outputs/experiments/`，不得反向污染生产主链。

- **影响面优先**：生产链路修改前先检索 parser/runner/scheduler/model adapter/ledger/report/tests/docs 的关联调用，明确牵动范围后再动手；修一个点时必须检查被它影响的上下游契约。
- **最小改动优先**：只在最窄责任层修问题，不借机重构或改算法；每个行为变化都要补覆盖真实调用路径的最小回归，并同步 active 文档。单元测试绿色不能替代正式 CLI/live smoke。
- **adapter 必须测真实 wrapper 路径**：protocol/data-contract 单测不能证明 model adapter 可运行；至少覆盖一次 core 输出 → wrapper normalize/rename → metadata merge → canonical prediction 的真实路径。2026-09-19 SGDFNet 曾因 `timestamp` 先改名后又按旧名 merge 而在 live path 崩溃，protocol test 仍全绿。
- **cache hit 与 fresh run 必须 provenance 等价**：正式缓存不仅验证96槽/非NaN，还必须验证当前 production contract、cutoff、模型配置与模型专属审计字段；cache result 写入 manifest 的 audit metadata 必须与 fresh execution 等价。SGDFNet cache 必须保留 D-1 anchor contract，RT916 cache 必须证明 stride=24。
- **full 失败不得抹掉合法 prediction provenance**：root manifest 由 full attempt 独占，但新 full 在 cold-start/readiness 阶段失败时必须保留最近一次合法 `--predict` provenance；否则后续 `--finish` 会不可恢复。恢复链要用 `predict → failed full → finish` 顺序做回归。

- **接口最小化**：正式预测入口应尽量收敛为 `target_date + 少量显式业务参数`；数据同步、immutable snapshot、FeatureView 防泄漏、模型编排、融合和最终输出由内部完成，前端/API 不承担文件路径拼装或手工准备数据。
- **状态有界**：任何按天运行的中间大文件都必须有明确生命周期。可重复构造的 scratch / as-of / 临时特征 / 临时 checkpoint 默认滚动覆盖或成功后清理，禁止随运行天数无上限增长。
- **持久资产最小集**：长期保留只包括不可替代或有审计价值的资产，例如 prediction/actual ledger、最终预测、必要模型/增量 cache、manifest/状态、有限保留期日志。其余均视为可重建中间态。
- **原子与可恢复**：更新长期文件使用 temporary + atomic replace；任务中断后必须能从 manifest/ledger/cache 恢复，不依赖某个未记录的临时目录。重复执行同一日期应幂等或明确版本化。
- **读写成本受控**：先避免 O(days) 的存储膨胀和重复重算，再考虑微小 I/O 优化。单次十几 MB 的顺序 parquet 重写通常可以接受；只有实测成为瓶颈后才升级为按日 partition / 增量存储。
- **96 单仓原则**：生产长期只维护 `data/96/model_input/shandong_pmos_96_model_input_full.parquet` 一份模型仓；closed history 是逻辑筛选，不再每日物化 clean。共享 FeatureView 与各模型训练中间产物只属于一次任务的 scratch，成功后必须清理，immutable D/T snapshot 保留审计，失败时可短期保留诊断。
- **生产与研究隔离**：`outputs/experiments/`、旧 replay、诊断产物不得成为正式服务运行的隐式依赖。生产所需状态必须进入稳定、可解释的 domain 路径，并能从仓库 + 配置 + 数据源 + 明确持久资产恢复。
- **部署可复制**：fresh checkout 不假设仓库里存在历史 `data/` 或杂乱 `outputs/`；启动时自动创建稳定目录、检查外部依赖和数据就绪状态，并返回机器可读错误码，而不是运行到模型阶段才失败。
- **运维可观测**：每次运行记录 target、cutoff、数据版本/哈希、模型版本、阶段状态、降级与失败原因；日志实行 retention，不允许无限积累。

---

## 1. 数据真实性红线（最重要，本次事故）

### 1.1 96 点 actual==fcast 拷贝事故（2026-04 批量回填造成）
- **现象**：`epf_market_data_96.actual_*` 历史段（2022-01~2026-07-18）98%+ 行与 `fcast_*` 完全相同。
- **根因**：爬虫 `crawl.py:crawl_market_overview()` 用 **DaJyxxPlDa（日前接口）** 返回的是**预测值**，却被 `MARKET_FIELD_MAP`（run_crawler.py:84 / auto_fill_96.py:86 / backfill_unit_data_96.py:376）写进了 `actual_*` 列。
- **真实实际值来源**：`crawl_market_overview_actual()`（crawl.py:708，走 **DaJyxxPlYx 实时接口**，字段 systemload/dfdcload/excload/fdload/gfload/hdload/zbload/syjzload/cxload）+ `exportsj` 导出回退。
- **修复脚本**：`scripts/backfill_actual_96.py`（需公司内网 + 有效 Cookie 重爬覆盖云端）。
- **教训**：96 点表 `actual_*` 8/10 列不可信，只有 `actual_bidding_space` / `actual_new_energy` 带真实实际值。**回测/建模前必须验证 `actual_direct_load != fcast_direct_load`。**

### 1.1b 方案B修复（2026-08-14，已实施）
- 三爬虫（run_crawler/auto_fill_96/backfill_unit_data_96）+ backfill_actual_96 已改为**预测写 fcast_*、实际写 actual_*** 双映射。
- 预测接口 `DaJyxxPlDa` 字段：systemload/dfdcload/excload/fdload/gfload/sytsjz(核电)/selfunit(自备)/syjzzj(试验) → `fcast_*` 列。
- 实际接口 `DaJyxxPlYx` 字段：systemload/dfdcload/excload/fdload/gfload/hdload(核电)/zbload(自备)/syjzload(试验) → `actual_*` 列；**cxload=抽蓄无对应列，跳过**。
- 通用 upsert `_upsert_market_by_map`：只写该行有值的列（接口字段集不同，避免 NULL 覆盖）。
- HAR 实测：同一天预测 vs 实际 systemload 明显不同（00:15 预测58396 vs 实际58135）→ 两接口确为不同数据。

### 1.1c 历史96点预测账本实验标记（2026-08-17）
- `data/96/model_input/shandong_pmos_96_model_input.xlsx`（旧副本在 `data/96/quarantine/legacy_root/shandong_pmos_96_full_v2.xlsx`）的电网特征 actual/fcast 存在大面积重复，标记为 `historical-invalid-features`。
- 由其生成的 `outputs/ledger_96/{dayahead,realtime}` 预测账本与价格 actual 账本可用于**权重学习器/融合器相对实验**，不得用于真实数据精度宣称、生产模型训练或生产数据源。
- 已逐点核验历史账本 `y_true` 与旧宽表 `日前电价`/`实时电价`一致；实验必须直接读取 prediction/actual ledger，不得重新从污染宽表构造特征。新鲜有效预测集建立后，旧实验结果归档清理。

### 1.1d HAR5 爬虫来源隔离与拒写规则（2026-08-18）
- `dist/agent_artifacts/info/pmos.sd.sgcc.com.cn5.har` 核验：`DaJyxxPlDa` 是日前预测，`DaJyxxPlYx` 是实时实际；两者均可返回完整 96 点，但数值不同。
- 本地 exe 爬虫必须分别保存两套原始响应，核心预测/实际各自完整 96 点并通过同值比例审计后才能写总表；缺失、异常或疑似拷贝时只保存 `output_96/raw/YYYY-MM-DD.json`，禁止用另一来源补值。
- 日级检修/抽蓄、断面约束和未确认语义的图表数据不得广播成 96 点；先原样归档，待特征工程显式定义后再消费。

### 1.1e PMOS v3 本地优先同步（2026-09-14）
- 主链路先保存逐日 raw/CSV，再仅同步 `epf_pmos_96_full`；不完整日期保留已获取值和空值，报告为 `PARTIAL`，不得因不足 96 点直接丢弃或跨来源补值。
- 旧源表同步函数/SQL不得继续作为生产入口；数据库模块只能初始化和写入 canonical 合并表，详细日志统一进入 `output_96/crawler.log`，累计摘要进入 `output_96/report.json`。

### 1.1f QCTC 双层认证与中断审计（2026-09-15）

> ⚠️ **2026-09-16 更正**：本节原来断言「QCTC 请求必须在 sessionStorage 取得 token 并由 CDP 注入 Bearer，否则接口不可用」——**这条因果链从未被验证**，而且它让 09-15、09-16 两轮补数在**一次数据请求都没发出去**的情况下中止。修正结论见 §1.1h。下面保留原始记录仅作追溯。

- 旧 PMOS 门户 Cookie 登录成功不等于新版 QCTC 已认证；QCTC 请求还必须在同一浏览器的 `:18080/qctc-trade` 上下文取得 sessionStorage `token`，由 CDP fetch 注入 Bearer。**（此因果未证实，见 §1.1h）**
- 直接导航被重定向到 `/dashboard` 时，程序应保留窗口并等待用户从已登录门户进入新版 QCTC；`QCTC_CONTEXT_READY/MISSING` 必须写入日志和累计报告，QCTC 模式跳过会返回 502 的旧 `/trade` 附加接口。
- 报告调用中不要把 `status` 同时作为位置参数和 `**details` 传入；人工 Ctrl+C/顶层异常也要把 report 从 `START` 更新为 `INTERRUPTED/FAIL`。
- 2026-09-15 实测：直接导航 QCTC 会回到门户 `#/dashboard`/`#/outNet`，且等待期间旧 CDP 端口可能断开；需先尝试点击门户 QCTC 菜单生成 SSO，连续 CDP 失败立即记录并切换新浏览器，不能继续重试失效的 9222。

### 1.1h ⚠️ QCTC「Bearer 门槛」是误判：观测项 ≠ 放行条件（2026-09-16，重要方法论）

- **事故**：`ensure_qctc_context()` 要求 sessionStorage 有 Bearer，超时就 `raise`，导致 `_new_spider()` 直接抛错 → **一次 QCTC 数据请求都没发出**。09-15、09-16 两次真机补数全部这样中止。
- **证据（逐条可查）**：
  - 2026-09-13 17:19:48/17:19:52 日志有 `浏览器 fetch 成功 ... status=200` + `QCTC ... 96 rows` + `总表已原子更新` + `[DB] 上传成功` → **QCTC 接口在本部署里能通**；
  - 那次是否携带 Bearer **没有记录**（当时无 bearer 审计）→ 「没 Bearer 必失败」**无从证实**；
  - 全量日志 `bearer_present: True` 出现 **0 次**；
  - `502` 全部来自旧 `:18080/trade/*.do`；`status=0` 来自页面被弹回门户后的**跨源** fetch —— 两者各有独立成因；
  - fetch 用 `credentials:'include'`，Cookie 一定会带上；Chrome 导出 HAR 默认剥离 Cookie/Authorization 头，所以 HAR 也证明不了「前端用了 Bearer」。
- **教训（可迁移）**：**不要把观测不到的状态当成前置条件。** 观测项只能写进报告；放行判据必须是业务接口的真实返回码。凡是「拿不到 X 就整轮中止」的设计，先问一句：X 真的是必要条件吗，有没有反例日志？
- **修法**：
  1. 超时不再抛错，记 `QCTC_CONTEXT_SOFT_MISSING`（stage=`WARN`）并 `return False`，采集继续；`fetch_csrf_token()` 在 QCTC 模式恒返回 `True`（新接口不用 `_csrf`）；
  2. `_browser_fetch()` 若标签页**已在接口同源**就**不再导航**，直接 fetch（导航会把可用的同源文档换成可能被弹回门户的 SPA 路由）；
  3. fetch 返回 `status=0` 或 `502` 时，才用同源兜底页 `:18080/zcq/main/index.do` → `:18080/favicon.ico` 重试一次。**这是待验证的兜底假设**，不改成功路径。
- **仍未定论**：Cookie 是否就足够，要靠真机跑一次、看接口返回码才能定。若加了兜底仍返回 401/403/code≠0，才反过来确认 Bearer 是必要条件，届时再研究换 token。

### 1.1i ⚠️ UKey PIN 必须显式配置，否则永远人工（2026-09-16）

- **现象**：登录停在证书环节，人工在 UKey 弹窗输 PIN，26 秒后才 `logged_in`。
- **根因（不是代码 bug）**：`AuthConfig.resolved_pin = os.environ.get(pin_env, ukey_pin)`，公司电脑上环境变量 `PMOS_UKEY_PIN` 没设、`config.json` 里也**没有 `ukey_pin` 键** → 空值 → `build_pin_handler()` 主动降级 `ManualPinHandler`，日志 `pin.handler_fallback=manual reason=pin_not_configured`。
- **易踩的假钥匙**：`config.example.json` / 旧 config 里的 `ukey_auto_submit`、`ukey_pin_env` **在源码里没有任何引用**（真键是 `pin_env` / `ukey_pin`）——只配它们不会生效。
- **修法（改配置即可，无需重新打包）**：`config.json` 补 `"ukey_pin": "<PIN>"`，或 `setx PMOS_UKEY_PIN "<PIN>"`（环境变量优先）。
- **诊断日志**（09-16 新增）：`pin.window_no_edit`（没可见输入框）、`pin.window_multiple_edits`（多个 Edit，取第一个）、`pin.confirm_button_missing fallback=enter`、`pin.settext_failed`（多为权限问题，需管理员运行）、`pin.window_still_present attempt=N`（PIN 被拒，最多重试 3 次）。
- **部署对账**：项目目录的 `dist/crawler` 与公司电脑的部署副本**不是同一份配置**；查问题必须看 `report.json` 的 `config_path`/`output_dir`。

### 1.1g 统一认证入口回归（2026-09-16）
- 浏览器认证不能只打开 `/#/dashboard`；旧版可用链路的 `/?service=<trade_entry>` 会建立交易系统 SSO 上下文。登录按钮应优先使用原生 `element.click()`，仅返回“已触发 DOM 事件”不能视为登录成功；登录页仍可见时必须允许有限间隔重试。
- 实际部署日志要核对 `report.json` 的 `config_path`/`output_dir`，项目目录的 `dist/crawler` 与公司电脑的部署副本可能不是同一份配置。UKey PIN 未配置时应明确降级人工输入，不能每轮重复刷屏告警（详见 §1.1i）。

### 1.1j QCTC SSO 必须复现门户入口（2026-09-16）
- PMOS 门户 Cookie 成功不等于新版 QCTC 上下文成功；collect 层应先在真实门户 target 导航固定的 `/psso?service=.../qctc/admin/sdsso/SSOLogin`，再轮询 CDP `/json` 识别 `:18080/qctc/` 或 `/qctc-trade/` 的同/新 target，最后才进入业务路由。
- target 切换和 storage 只记录键名/布尔状态，不记录 token；直达 SSO 超时应保留原 DOM fallback，Bearer 仍是观测项，业务接口返回码才是最终可用性判据。

### 1.1k QCTC ticket 与快速失败（2026-09-16）
- `/psso?service=...` 是门户菜单描述地址，不是最终 SSO；应在门户同源上下文 POST `/px-common-authcenter/sso/token`，由浏览器内部拼接 `SSOLogin?ticket=...`，Python 侧只接收状态摘要。
- 已有同源 Bearer 上下文应立即 READY，不能再导航业务页；Bearer 缺失或接口明确 401 时要结束本轮，避免按日期循环制造重复认证失败。

### 1.2 爬虫历史事故与当前生产边界
- 旧 `auto_fill_96.py` / `run_crawler.py` 曾发生预测值写 actual 的污染事故；相关旧双表仅作历史审计，不得重新接回生产模型输入。
- 当前 96 生产唯一数据库源是 `epf_pmos_96_full`；`epf_market_data_96` / `epf_unit_data_96` 属历史兼容镜像。
- 爬虫当前冻结；除非用户明确要求，不修改 crawler 代码。若后续解冻，先读 `scripts/crawler/README.md` 并保持 forecast/actual 来源分离。

---

## 2. 24点 vs 96点口径区别

| 项 | 24点 | 96点 |
|---|---|---|
| 价格来源 | `epf_market_data` 全省市场均价 | 生产唯一源 `epf_pmos_96_full` 的 `日前出清价格/实时出清价格`（当前单机组 scope）|
| 相关性 | — | 24/96价格不是同一粒度，跨分辨率比较必须先按业务时段对齐 |
| 特征 | 10 组 fcast/actual | `epf_pmos_96_full` 预测/实际基本面；模型统一映射到 `data/96/model_input/` |
| 滞后特征 | shift(24)/168 | **shift(96)/672**，勿机械沿用 24 |
| 实时可见性 | — | Dynamic-v1 snapshot/FeatureView 路由；历史固定 cutoff 仅 legacy/experiment |

- 跨分辨率比较：96点 mean 聚合到 24点 后同口径比；度电套利需 ×0.25 因子。
- 生产输出统一进入 `outputs/96/{ledger,runs,cache,runtime,sync}`；旧 `outputs/ledger_96` / `outputs/runs_96` 仅保留为 legacy 兼容区，不再作为新生产默认写入目标。

---

## 2b. 业务时间规则与防泄漏红线（2026-08-15 用户立规，决定预测标准）

> 甲方数据对照已验证：96点非边界点(1-23点)与24点逐点一致(MAD=0.00)，预测≠实际(0%相同)。

### 2b.1 业务时间对齐（权威规则）
- **96点 p96(24:00) 与 24点 h24(00:00) 是同一物理时刻（D 的 24:00 = D+1 00:00）**，都归 `business_day=D` 的最后一个点。
- 两表数值取法不同（96点=最后15分钟区间值，24点=整点时刻值），差 ~1-2% 属正常口径差，**不影响业务时间对齐**。
- **预测标准的业务日归属必须统一**：`business_day=D` 覆盖 D 的 p1(00:15) ~ p96(D+1 00:00)，跨午夜规则（p96 归 D 非 D+1）。

### 2b.2 信息可得性（防泄漏铁律）
| 时点 | 可知信息 | 不可知 |
|---|---|---|
| **日前(DA)** | **D 当天完整日前数据**（日前价、预测特征全量） | D 当天实际值、实时价 |
| **实时(RT)** | 使用 immutable snapshot 中实际可见 RT，经 FeatureView fallback 路由 | target-day truth 全 mask，绝不泄漏 |
| **预测辅助特征** | 可用 **D+1（次日）电网特征预测值**（fcast_*）作模型辅助 | **绝不用实际值(actual_*)作 target 日特征** |
| 滞后特征 | 只用历史 actual/fcast（shift(96)/672） | target 日 actual 是标签不是特征 |

### 2b.3 三条硬规则
1. **预测≠实际**：任何特征列若预测==实际 比例 >1% 即视为爬虫污染（甲方数据已确认 0%）。
2. **target 日实际值绝不入特征**：`actual_*` 只能作历史 lag，target 日 actual 是预测标签。
3. **96生产 Dynamic-v1 边界**：formal path 由 DB sync → snapshot → FeatureView 统一决定可见性；24点 strict-spread 的 D-1 14:00 规则是另一任务，禁止混用。
4. **单一持久模型仓契约**：`authoritative/pmos_96_全量.csv` 忠实同步 DB；生产模型层长期只保留 `model_input_full.parquet` 一份，包含闭合历史+partial/forecast-only tail。闭合历史是 full 上的逻辑筛选，不再每日物化第二份 `clean.parquet`；旧 clean 仅作兼容历史资产。
5. **Dynamic-v1 外层防线**：预测 D 时，runner 先 DB sync，再冻结共享 immutable D/T snapshot，由 FeatureViewBuilder 路由 D/T；D 的 DA/RT/actual 全置 NaN。transient FeatureView 只存在于一次任务生命周期，所有模型共享，任务结束立即删除；snapshot/full/authoritative 按 provenance 保留且原始源永不修改。

---

## 3. 交付纪律（24点正式链路）

- 24 legacy 五阶段：`ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs`；formal 96 为四阶段，见 §0 override。
- NORMAL 交付前提：ledger 在 lookback 内为 Dayahead/Realtime 各找齐 **30 个完整训练日**。
- 交付文件 `outputs/runs/YYYY-MM-DD/final/submission_ready.csv`：24 行 6 列 0 NaN；96点=96 行。
- 只读校验脚本：`scripts/check_delivery_stability.py`(29/29)、`check_target_day_nan_regression.py`(16/16)、`check_sync_dataset.py`(41/41)、`check_adaptive_realtime_weight_days.py`(40/40)。
- 黄金基线：`outputs/golden_baseline_24/` 用于验证改造前后 submission_ready.csv **逐字节一致**。

---

## 4. 环境与执行约定

- 本机 conda：`D:/computer_download/environment/conda/epf-2/python.exe`；**CPU only**。
- GPU 云（智川云 sc01-ssh.gpuhome.cc:30486）：每次新终端 `source conda + export PROJECT_ROOT TIMESFM_DEVICE=cpu`；单 GPU 勿开第二进程。
- git push 本机代理坏：`git -c http.proxy= -c https.proxy= push origin main`。
- 负价漏判率**按事件数加权**，勿在市场间取算术平均。
- 历史 agent 报告/记忆可能过时：**以实际文件/代码/git 重新核验**，不盲信。

### 4.1 打包爬虫 exe 必须用 venv_build（OpenSSL 3.0.13），禁止用 epf-2（3.6.1）⚠️
- **现象**：epf-2 (OpenSSL 3.6.1) 打包的 exe 连 PMOS 报 `[ASN1: NOT_ENOUGH_DATA] not enough data`；旧 exe（run_full 等，venv_build/3.0.13 打包）能连。
- **根因**：国网 PMOS 的 TLS 证书与 OpenSSL 3.6+ 不兼容。
- **正确做法**：用 `dist/build_artifacts/venv_build/Scripts/python.exe` 打包；在**项目根**跑 PyInstaller（spec 放根，入口 `['scripts\\crawler\\xxx.py']`，pathex=['.']）。
- **exe 自检**：加 `--ssl-check` 参数打印 OpenSSL 版本+测连，甲方电脑一跑即知。
- **打包依赖**：hiddenimports 须含 requests/urllib3/certifi/websocket._core/websocket._exceptions；导入用 `from scripts.crawler.crawl import ...`（打包后模块全名）。
- **其他坑**：config.json cookie 带换行→JSON `Invalid control character`（load_config 已容错）；系统代理失效→`session.trust_env=False` 直连；双击闪退→用 cmd 或 `运行爬虫.cmd`。

### 4.2 实验前防事故检查（2026-08-15 立规，防止7天100元失败重演）
- **预测/backfill 前必跑** `scripts/tests/check_preflight_health.py`：检查24点完整性、96点近30天 actual≠fcast、Resolution 契约、Dynamic-v1 snapshot/FeatureView 边界和数据源健康；96 production ledger 天数只显示 INFO，因为允许从0开始建立新账本。
- **formal 96 readiness 前再跑** `scripts/tests/check_preflight_health.py --require-96-full-chain`：此时必须 `outputs/96/ledger` 至少30个完整历史日。旧 `outputs/ledger_96` 只作历史参考，禁止把它的天数冒充新 production readiness。
- **防泄漏确认**：96生产链 `ledger_predict` 统一生成 immutable D/T snapshot + FeatureView；模型内部 fixed-hour 参数仅兼容/训练语义，不能替代 serving boundary；24点 strict-spread 的 D-1 14:00 是另一任务契约，禁止混用。
- **96vs24对照**：`scripts/tests/check_96_vs_24_actual.py` 读甲方合并总表，按小时聚合比对。
  预演结果：近30天 MAD=83.9MW、corr=0.9984、MAPE=0.10% → 96点实际与24点一致（口径差异非零正常）。
- 模型重活（GPU）只能在服务器跑，本机 CPU-only 不跑会误导的判断。

### 4.3 链路稳健性 & 优化调查结论（2026-08-15 三份报告落盘）
- **风险本质**："崩得不够响"——静默污染/静默降级比崩溃更危险。Top 危险：
  ① 96点价格null时 `delivery_quality.py:499` KeyError 崩收尾（用slot_col修）；
  ② 96/24 data_path 混用污染 actual 账本（`_extract_actuals` 须断言分辨率）；
  ③ 缓存校验不查 y_pred NaN（坏缓存静默通过）。
- **训练加速 Top3**（3天→1天）：AMP混合精度、特征矩阵离线预计算共享、Warm-start续训。
- **融合改进**：BGEW方向正确，补 OOF融合守卫/事件加权/分类器前置/朴素平均对照。
- **极端价修正前置vs后置（2026-08-15 调研定论，详见 docs/极端电价修正_前置vs后置_调研报告.md）**：根因不是位置而是门控量——后置 `p≥θ∧y_fused≤100`（classifier_bridge.py:90-92）用被均化污染的 y_fused 当事件统计量，多数模型漏报时 y_fused 被拉高>100 → 真负价漏掉。共享 h(p) 时前置软修正 ≡ 后置软修正（=y_fused+h(p)(-80-y_fused)），前置真价值在 per-model 差异度。推荐混合：软修正（分级 h，p_lo≈0.25/p_hi≈0.80）+ 置信度兜底（p≥p_hi 才触底，去掉 y_fused≤100）。注意：共享分类器致 7 模型修正决策完全相关，硬前置会放大误报，软修正自限。
- **特征预计算+WarmStart 调研（2026-08-15，见 `docs/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`）**：不装 Feast/Chronon（重型框架），自研 50 行轻量 FeatureStore（特征注册表+parquet 缓存+指纹版本+asof 切片），插入点在 ledger_predict 之上 scheduler 之前；DA/RT 双命名空间，shift 常量只在注册表一处；SGDFNet 两阶段物化（DA→RT 的 da_anchor）；warm-start 用 LightGBM init_model / PyTorch checkpoint(含scaler+scheduler)，续训只是初始点≠含 target 日，漂移/节假日触发全量。
- **CPU-GPU 并行调度深化调研（2026-08-15，见 `docs/CPU_GPU_并行调度与数据流_深化调研报告.md`）**：GPU 模型批级重叠（num_workers+pin_memory+non_blocking+prefetch=2）已全部兑现（perf_knobs.py+core.py+repro_pipeline.py），别再加 side stream。唯一可回收=天级特征工程×训练串行：DayPrefetcher 线程+双缓冲（§3.5，<100行），加速比公式 S=(t_feat+t_gpu)/max(...)，t_gpu 越短越值钱；生产单日做"日内DA→RT两腿流水"。FeatureStore 落地后双缓冲降级兜底。不建议：模型内算子搬CPU/手写side stream/torch.distributed.pipelining两stage/ZeRO-Offload。
- 详见 `docs/链路稳健性容错筛查报告.md`、`docs/工业界时序预测训练加速与精度提升调研报告.md`、`docs/多模型融合策略调研报告.md`、`docs/EFM3_链路稳健性_训练加速_融合改进_实施计划.md`、`docs/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`、`docs/CPU_GPU_并行调度与数据流_深化调研报告.md`。

### 4.4 设计原则（用户立规 2026-08-15）
- **能运行是底线**：今天数据有问题也必须产出预测文件供上报；修复优先级=先保证有输出，再优化质量。
- **失败要响亮**：任何降级/回退/缺失写 manifest + delivery_report 告警段，前端/DB 可见，杜绝静默。
- **历史实验核验**：96点账本 business_period 96 唯一值 → 之前实验正确用 96 点文件，粒度混用是防患未然非已发生。
- **AMP 已内置**：`optim/perf_knobs.py`（BF16 默认+Scaler），加速重点放特征预计算(零精度损失)+warm-start续训(省时留更多训练时间收敛)。

### 4.5 极端价修正专项结论（2026-08-15，见 `docs/极端电价修正_前置vs后置_调研报告.md`）
- **根因不是前置/后置，是后置门控用了被均化污染的 y_fused**（`classifier_bridge.py:90-92` 的 `y_fused≤100` 挡住真负价）。
- 数学：共享 h(p) 时前置软修正≡后置软修正；前置价值在 per-model 差异强度。
- **推荐混合**：`y_final = y_fused + h(p)·(−80−y_fused)` 去掉 y_fused 门槛；分级软修正(p_lo 0.25/p_hi 0.80)+高置信触底+可选一致性门 min_m ŷ_m≤30。
- 实验 V0-V4 对照 + 事件级指标（漏报 M1/召回 M2/误报 M4）。

### 4.6 三专项落地设计（2026-08-15，见 `docs/EFM3_三专项落地设计_报错接口_特征预计算_极端价修正.md`）
- **报错接口**：`utils/degradation.py` DegradationHub（stage/kind/severity）→ manifest `degradations` 字段 + report 告警段 + 测试断言三形态；生产接口预留不实现。
- **特征预计算**：`utils/feature_store.py` 轻量 store；DA/RT 双命名空间防混淆（shift 只在注册表一处）；SGDFNet 两阶段物化（先DA后RT da_anchor）；CPU/GPU 线程池只读切片天然安全；零精度损失逐位 diff 验收。
- **warm-start**：LightGBM init_model（防树膨胀7-14天重置）；PyTorch state_dict（重建DataLoader+LR重启）；TimesFM 冻结主干；周一/节假日全量重训。
- **极端价混合修正**：`y_fused+h(p)(−80−y_fused)` 去 y_fused≤100 门槛；p_lo0.25/p_hi0.80 + 高置信兜底 + 一致性门 min_m ŷ_m≤30。

### 4.7 实验专区与对照实验（2026-08-15）
- **实验专区**：脚本 `scripts/experiments/`（git 跟踪，.gitignore 已放行）；产物 `outputs/experiments/`（gitignore）。
- **分类器 V0-V4 对照**：`scripts/experiments/classifier_ab/run_ab.py` 离线重放（读历史 fused+clf，不重跑模型）。分类器输出含 p1_prob/真实极值标签，可直接算事件级指标。
- **初步结果（3天样本）**：V2/V3/V4 软修正 M4 误报 0.444→0.286；V1 硬前置最差(0.615)→证实软优于硬。样本小需真实账本历史窗验证。
- **特征预计算量化**：96点特征矩阵 ~19MB、24点 ~4MB（parquet ~15-20MB）；单日省 3-7min 特征工程 → 214天省 10-25h；+warm-start → 3天→1天。

### 4.8 LightGBM 超参对比实验（2026-08-15 首轮，2个月窗）
- 脚本：`scripts/experiments/lightgbm_ab/run_ab.py`（本机 CPU 可跑）
- **发现**：不同时段段最优超参不同——谷段喜浅树(lr05_leaf15, SMAPE↓3.5%)，峰段喜小学习率(lr02_leaf63, ↓1.7%)，平段基线最优。
- **⚠️ 平段(光伏时段) SMAPE 高达 0.73-0.87**（远超峰段0.23）→ 光伏时段价格波动大，是比调参更大的改进空间。
- 训练极快(<0.1s/配置) → 可放心网格搜索；结论需扩大数据窗复验。

### 4.9 ⚠️ 指标公式铁律（2026-08-15 教训，用户质疑救回实验）
- **SMAPE 必须用生产公式**（`train_fix.calculate_smape`）：**值<50 先 clip 到 50**，再算 `|p-t|/((|p|+|t|)/2)`，**结果 ×100（百分比）**。
- **composite loss**（权重学习器 `compute_daily_loss`）= 0.7×SMAPE% + 0.3×MAE%（MAE% = 100×MAE/max(median(|y_clip|),50)），同一百分比尺度。
- **composite vs SMAPE 排名一致**（RT: sgdfnet>timesfm>rt916>timemixer；DA: timesfm>lightgbm>timemixer）——决策不受口径影响。
- **数值量级**：SMAPE% 就是 20-35% 量级（sgdfnet RT 23.6%、DA timesfm 25.6%），不是"0.21 小数值"。若看到小数值（0.2x）是**未 ×100 的旧口径**。
- 96点平段(光伏)负价率 32.7%，SMAPE 敏感区，clip 铁律必须遵守。

### 4.10 LightGBM 超参修正后结论（2026-08-15）
- **B 配置（lr02_leaf63 小学习率深树）三时段全优**：谷段0.3066/峰段0.2195/平段0.2145，↓1.5-2.0%。
- 首轮"各段最优不同"是公式 bug 造成，不成立。
- 训练 <0.1s/配置 → 全模型按"训练短先调"策略（见 `docs/全模型调参实验计划.md`）。

### 4.11 甲方96点数据对照结论（2026-08-15 ✅ 数据可靠）
- 甲方 `data/pmos_96_全量.csv`：162048 行，2022-01-01~2026-08-15，预测+实际全列。
- **对照 24 点：非边界点(1-23点) MAD=0.00 逐点完全一致，corr=0.9990** → 数据真实可靠。
- **p96(24:00) 边界点与 24 点表该点值不同**（~1685天全差异）→ 两表对"24:00"取值口径差异（96点p96 vs 24点h24 物理含义不同），**非数据污染**，只占 1/96，不影响价格目标建模。
- 对照命令：`python scripts/tests/check_96_vs_24_actual.py data/pmos_96_全量.csv`。

### 4.12 全模型调参进展（2026-08-15）
- **LightGBM**：B配置(lr02_leaf63)三时段全优，↓1.5-2.0%（谷0.3066/峰0.2195/平0.2145）。
- **SGDFNet**：训练<10s/配置（可本机调）；B配置(lr02_iter500)最优 0.2536，↓7.2%。**小学习率深迭代是CPU梯度模型普遍最优方向**。
- 实验脚本：`scripts/experiments/{lightgbm,sgdfnet}_ab/run_ab.py`。
- 下一步：timesfm（零样本 vs 微调）→ timemixer/rt916（服务器）。

### 4.13 ⚠️ TimesFM 环境复现铁律（2026-08-15 修复，必读）
- **jax 版本必须匹配 numpy**：本项目环境 numpy==1.26.4 → **jax==0.4.30**（requirements 已钉死）。
  jax 0.8.0 配 numpy 1.26 会报 `asarray() got an unexpected keyword argument 'copy'` → TimesFM 复现失败/经常漏。
- **timesfm 包**：用项目自带 `TimesFMBackend/src/timesfm`（PyTorch 实现，含 timesfm_2p5_torch），**不要 pip install timesfm**（editable 曾误指向 epf/TF 历史遗留，触发 jax 冲突）。
- `_import_timesfm` 已改为**强制优先本地 src**，本地缺失时明确报错。
- 修复后 TimesFM CPU 单日 96 点预测约 98s（零样本，segment_count=1）。

### 4.14 GPU 训练加速关键结论（2026-08-15，5份调研整合见 docs/GPU训练加速落地设计.md）
- **瓶颈不是算力**：两 GPU 模型（TimeMixer/RT916）是 launch-bound 小模型（hidden 64/128, batch 16-64），RTX3090 利用率个位数。主因=kernel数量+Python调度+CPU-GPU同步。
- **隐藏雷（P0，最高优先）**：`utils/reproducibility.py:34-38` set_global_seed **强制关 cudnn.benchmark + 关 TF32**（float32_matmul_precision("highest")），把 perf_knobs/core.py 想开的优化二次关掉。修复预期 20-40%。
- **RT916 最大单点**：`model.py:100` `int(period_list[i].item())` 每 forward 多次 CPU-GPU 同步 + Python loop → 需向量化。
- **P0 加速**：`optim.Adam(fused=True)` + `torch.compile(mode="reduce-overhead")`，预期 1.3-2.5×（RT916 需 fullgraph=False）。
- **CPU-GPU 并行**：批级重叠已吃满（num_workers+pin_memory 已配）；日级双缓冲收益敏感于 t_gpu，正确顺序=FeatureStore→warm-start→双缓冲兜底。
- **C++/CUDA 不需要**：Triton 足够且多数情况 torch.compile 就够；CUTLASS 对小矩阵无优势。

### 4.15 epf-2 GPU 环境 + 本机 GPU 实测（2026-08-16）
- **epf-2 已换 CUDA 版 torch**：torch 2.6.0+cu124 + torchvision 0.21.0+cu124，RTX 4060 Laptop 8GB 可用。
  清理了 CPU 版 torch 残留 + 损坏的 `~orch` 分发。requirements.txt 无 torch 版本（由环境提供）。
- **set_global_seed 修复**：不再强制关 TF32（`reproducibility.py`）；benchmark 交由模型自行决定（小模型开 benchmark 反而有搜索开销）。
- **RT916 core.py**：AdamW 加 `fused=True`（GPU 提速，CPU 自动降级）。
- **本机 GPU 单日耗时实测**（96点，12个月训练窗，RTX4060）：
  - TimeMixer DA：~310s（5.2min）
  - **RT916 realtime(DA+RT)：~1788s（29.8min）→ 回测最大瓶颈**
  - CPU 模型：LightGBM <0.1s / SGDFNet 5-9s / TimesFM 40s
- **RT916 提速重点**：`.item()` 同步（model.py:100）+ warm-start 续训（省 epoch）。服务器 3090 上会更快，但 RT916 仍是单日最长腿。
- **Windows 注意**：TimeMixer/RT916 脚本直接调用时 DataLoader `num_workers=4` 会 spawn 崩溃 → 用 `OPTIM_NUM_WORKERS=0` 或用 `__main__` 保护。

### 4.16 ⚠️ RT916 训练提速关键：TRAIN_STEPS（2026-08-16，实测 18-25 倍）
- **RT916 慢的结构性根因**：`core.py` `TRAIN_STEPS=1`（硬编码）→ 96点下每段 ~11393 样本（seq_len=288 滑动步长1），3 段×2 任务×8epoch = 48 次大训练。
- 历史实验（SUPERSEDED FOR FORMAL 96）：`TRAIN_STEPS` 曾改为环境变量 `RT916_TRAIN_STEPS` 可配置（默认仍 1 保守）。
- **实测（3个月窗，RTX4060）**：
  - TRAIN_STEPS=1 → ~1000s+（最慢）
  - **TRAIN_STEPS=24 → 98s（18倍提速），SMAPE 0.23-0.32 精度良好** ← 甜点
  - TRAIN_STEPS=96 → 70s 但样本太少(102)过拟合，SMAPE 0.45-0.82 ❌
- 历史调参结论（SUPERSEDED FOR FORMAL 96）：RT916 调参/回测用 `RT916_TRAIN_STEPS=24`。
- **注意**：样本数与训练月数成正比，训练月越大样本越多；实际以 12 个月窗复测为准。

### 4.17 分类器入口修复 + 全链路验证（2026-08-16 ✅）
- **缺失入口**：`ExtremPriceClf/merge_model_scripts/run_daily.py`（classifier_bridge 调用）曾缺失 → 分类器静默失败。已新建：
  - 调 `cascade_daily.run_rolling_daily_cascade`，输出 `<start>_<end>_clf.xlsx`（含 时刻/p1_prob/p2_prob/final_pred/真实极值标签）
  - **96点宽表需先剔除非数值列**（market_date/开机状态等）→ LightGBM 报 `pandas dtypes must be int, float or bool`。run_daily.py 已加数值列过滤。
- **全链路验证（2026-01-01，96点）**：`delivery_status=NORMAL, exit_code=0`，warnings/errors 全空，submission_ready 96行 0 NaN，period 1~96 完整。总耗时 ~16min（含 --force 重跑模型）。
- **模型清单确认**（ledger_predict.py:47-48）：
  - DA = lightgbm + timesfm + timemixer（3模型，sgdfnet/rt916 的 DA 是内部 anchor 不进融合）
  - RT = timesfm + sgdfnet + timemixer + rt916（4模型）
- **归档**：`outputs/runs_96`(501MB)、`RT916_SpikeMarketLab`、`unified_runs` → `dist/agent_artifacts/legacy_server_96/`。ledger_96 保留（权重学习用）。

### 4.18 ✅ SGDFNet 防泄漏审计确认（2026-08-16，代码级核实，锁死）
> 背景：用户质疑"SGDFNet 用源表实际日前价作 da_anchor 是否泄漏"。经逐行核实：**不泄漏**。
> 任何人再问此问题，直接引用本节，不要重新怀疑。

**业务时间口径（skill §2b 权威）**：
- D 日预测 D+1：**日前电价(D+1)在 D 日 14:00 前已发布**（日前市场提前出清）→ 预测时可得，合法
- 24 点历史协议（SUPERSEDED FOR FORMAL 96）：**实时电价(D+1)预测时不可得** → 必须遮蔽；实时只知道 D 日 14:00/p56 前的

**SGDFNet 代码证据（data_contract.py / protocol_b_cutoff.py）**：
| 特征 | 定义 | 时点 | 是否泄漏 |
|---|---|---|---|
| `da_anchor` | `out[DA_COL]`(L232) | D+1 日前价，已发布 | ✅ 合法 |
| `delta_lag_1` | `_safe_delta_history()` shift(24/96) (L157,330) | 前一天实时delta | ✅ 历史 |
| `delta_lag_24/168` | `delta.shift(resolution/7*res)` (L331,401) | 前1/7天 | ✅ 历史 |
| `rt_lag_168` | `_rt_history_source.shift(7*res)` (L405) | 前7天 | ✅ 历史 |
| `da_lag_24/168` | `da_anchor.shift(...)` (L403-404) | 滞后 | ✅ 历史 |
| `hist_*_lag24` | `.shift(resolution)` (L280) | 前1天 | ✅ 历史 |
| `visible_rt_anchor` | 整列实时价，仅 D 日 14:00 后被遮蔽 (L189-194) | D+1 行靠 shift 不用 | ✅ |

**核心机制**：`_safe_delta_history` 注释明确——"D-day post-cutoff RT truth **never backflows** into D+1 features through adjacent-hour shifts"（L155）。所有实时/实际特征都 shift 到历史，**D+1 行绝不包含 D+1 自己的实时价或 actual_***。

**权威背书**：LEAKAGE_AUDIT_96："day-ahead is fully known at prediction time"；"leakage in all legs is using target-day `rt_cq_price` or target-day `actual_*` as features"——da_anchor 是日前价非实时价非 actual_*，安全。

**caveat**：da_anchor 依赖源表"日前电价"列是预测时点已发布值；若是事后修正值则是数据质量问题非特征泄漏。

### 4.19 ✅ TimeMixer CUDA 运行契约与数据域迁移（2026-08-16）
- TimeMixer 在 epf-2（Torch 2.6.0+cu124、RTX 4060）以 `deterministic=False` 的 DA/RT 单日 smoke 均可完成并产出 96 行预测；旧的 deterministic CUDA 报错来自严格算法开关与 upsample backward 的组合。
- 生产代码现在明确拒绝 CUDA + `deterministic=True`，避免假装可复现；严格确定性改用 CPU，GPU 性能路径使用 `deterministic=False`，并由 manifest 记录。
- 96 点权威实际唯一来源为 `data/96/authoritative/pmos_96_全量.csv`；它只用于 actual 交叉验证，不是价格模型宽表。24 点 canonical 与 96 点 `actual_*` 的小时聚合交叉验证必须先通过 `scripts/tests/check_96_vs_24_actual.py`。
- 数据和输出按 `24/96` 域分离；旧根路径只作迁移兼容，新增链路使用 `outputs/{24,96}/feature_store/{cache,ledger,runs}`。

### 4.19 2026-01-01 全链路预测结果（2026-08-16 验证 ✅）
- `delivery_status=NORMAL, exit_code=0`，warnings/errors 全空
- submission_ready.csv：**96 行 0 NaN**，business_period 1~96 完整，period 三段(1_32/33_64/65_96)映射正确
- 分类器 complete（修复 run_daily.py 入口后）
- 总耗时 ~16min（--force 重跑全部模型：RT916 ~6min + TimeMixer ~5min + CPU模型 + 分类器训练）
- 服务器旧 runs_96 已归档，本机用甲方真实 96 点数据 + 最新代码跑通

### 4.20 ⚠️ 三路全链路审计（2026-08-16，发现2泄漏+1交付设计需确认）
> 多 agent 并行审计：数据质量 / 模型DA-RT链路 / 融合权重分类器。详见各报告。

**🔴 泄漏风险1：LightGBM RT cutoff 晚 24h（infer_fix.py:244）**
- `info_cutoff_dt = end_dt - 10h`，但 end_dt=(target+1)00:00 → cutoff=D+1 14:00，实为 D 14:00+24h。D+1 15:00 槽 lag 用了 D 15:00 实际实时价（D 14:00 不可得）。
- 当前 RT 融合不含 lightgbm（无实害），但 adapter 可达，一旦启用即泄漏。修复：按 start_dt/decision_day+14h 算。

**🔴 泄漏风险2：TimeMixer RT baseline 未截断（repro_pipeline.py:273-304）**
- `compute_blend_baseline` lag_days=1 取 D 全日实时价（无 cutoff 限制），residual_blend 下 D+1 下午预测被锚定到 D 全日实际 RT。
- 修复：baseline lag-1 对 RT 按 cutoff(D 14:00) 截断 + 前向填充。

**🟠 P0-1（设计确认）：分类器修正不进 submission（ledger_full.py:413）**
- ✅ **已修复（2026-08-16）**：`_build_submission_ready` 优先用 `realtime_final_predictions_corrected.csv`（探测 `y_fused_corrected` 列），result 记 `submission_realtime_source=classifier_corrected`。端到端验证：24/24 修正一致进入 submission。
- 历史决定（SUPERSEDED FOR FORMAL 96；仅 legacy/replay）：**分类器必须进主链路，最终预测经过分类器**。

**🟠 P0-2：分类器 ds 对齐错位（cascade_daily 24行 vs 融合 96点）**
- ✅ **已修复**：`merge_clf_results` 96 点下把 fused ds 归到所属业务小时（floor('h')）→ 与分类器小时 final_pred 按小时 map → 广播到 4 个刻度。同小时 4 刻度一致。

**🟠 P0-3：96 点分类器只覆盖最后 6 小时**
- ✅ **已修复（方案=入口聚合小时+广播回96点）**：`run_daily.py` 加 `--resolution 15min`，把 96 点数据按小时聚合（数值列均值）喂 cascade；`classifier_bridge` 自动检测 fused 分辨率传 `--resolution`。验证：2026-01-01 全天 24 小时输出（00:00~23:00），6 极值命中（Precision 100%），广播回 95/96 刻度（00:00 边界缺 1 刻钟，可接受）。

**🟡 附带修复**：`emergency_fallback._fallback_markdown` 96 点无 `hour_business` 列导致 KeyError → 按 rows 实际键用 `business_period` 或 `hour_business`。

**🔴 LightGBM RT cutoff 晚 24h（infer_fix.py:244）— 用户决定不改**
- 学长程序逻辑：`current_target_date`=目标日，`inference_end=(目标日+1)00:00`，`end_dt-10h` 实得**目标日当天14:00**（比"决策日D 14:00"晚24h）。注释与代码自相矛盾（注释称 D 14:00）。
- **实际影响=0**：ledger RT 模型集不含 lightgbm（ledger_predict.py:48），DA 走 infer_da_fix.py。RT 路径仅在单独调 adapter target=realtime 时可达。
- 用户判断：学长程序可能有其用意，**维持原样**。仅当将来 lightgbm 进 RT 融合时需复核。

**🟡 其他**：缓存不查 NaN（ledger_predict.py:327-345）；SGDFNet/RT916 独立入口默认 15h（ledger 都传14，仅独立调用退15）；账本重跑改写历史（keep=last）；分类器 y_fused≤100 硬门未上软修正。

**✅ 已确认无泄漏**：SGDFNet（da_anchor=日前价合法+delta全shift）、RT916（asof=14+DA注入用自身预测）、TimesFM（段机制天然不触目标日）、模型间无交叉污染、账本幂等。

### 4.21 ✅ 权重学习器升级：NNLSGEF 替代 BGEW（2026-08-16 实证，用户要求重新设计）
> 用户判断 BGEW "有创意但不够"，要求能准确收敛到加权最优。多 agent 上网调研（AdaHedge/NNLS/BMA/regime 自适应）+ 本机实证后定案。

**实证发现（96 点 ledger_96，2025-12-01~2026-07-18，230 天滚动回测）**：
- **BGEW 实际输给等权**：赢 45.8%，相对提升 -2.07%，距 oracle(sgdfnet) 差 16.63 loss → 当前算法不收敛。
- **SLSQP 约束最小二乘退化等权**（局部最优）：NNLS 融合 46.9 vs 等权 41.97（更差）。
- **scipy.nnls（纯非负，无 sum=1 约束，事后归一）显著最优**：段1 29.15（等权38.13 ↓24%）、段3 21.43（等权26.66 ↓20%），**赢等权 68.6%，相对提升 +8.1%**，多处超越单模型最优。
- 根因：BGEW 固定 eta + 固定 day_gate + 证据收缩拖累强模型；sgdfnet 在段1/段3 主导但权重被均化。

**新 learner（`fusion/learners/daily_ledger_gef.py` 新增 `NNLSGEF`）**：
- 每 (task, period) 用最近 21 天 OOF 预测拼 X、实际拼 y → `scipy.optimize.nnls` 学非负系数 → 归一化 → 下界 weight_floor=0.02 重归一。
- 冷启动/退化回退 AdaHedge 在线更新。
- 历史兼容口径（SUPERSEDED FOR FORMAL 96）：接入 `--weight-learner {nnls,bgew}`，曾以 **nnls** 为默认。
- 实测权重合理：RT 段1 sgdfnet 0.74 / 段2 timesfm 0.62 / 段3 sgdfnet 0.90；DA 段3 lightgbm 0.62。
- `_validate_weights` 已兼容 NNLS trace（无 age_days，用 method/n_obs）。
- 保留 BGEW 作对照（`--weight-learner bgew`）。

**注意**：ledger_weight 训练窗仍取 30 完整日（select_complete_training_days），NNLS 内部只用最近 21 天 OOF。

**是否已达最优（2026-08-16 最终核算）**：NNLSGEF **已收敛到现实可达的最优**。
- 整体核算（2025-12~2026-07 滚动，309 单元）：等权 41.97 → **NNLS 37.46（↓10.7%）**，赢 68.6% 单元；BGEW 42.46；oracle（事后最优单模型）28.49。
- NNLS 距 oracle 仅 ~9 loss，且 **14% 单元超越事后最优单模型**（加权组合利用模型互补性）。
- oracle 是理论下界（需事后实际值，现实不可达）→ **剩余差距不是算法问题，是信息边界**。
- **继续提升方向**（调研结论）：更细 period（每小时块 24 块/天）、regime 自适应（volatility 动态遗忘 λ）、条件权重（星期/节假日分桶）。换权重算法本身收益已尽。

### 4.22b 多组权重实验结论（2026-08-16，用户要求"96点多学几组权重"）
> 完整实验见 `scripts/experiments/nnls_ab/`（run_ab.py 窗口/粒度/参数、run_negative_w.py 负权重、run_hour_select.py 小时选择）。产出在 `outputs/experiments/03_fusion_weighting/nnls_ab/`。

**用户目标**：96 点数据级更细、量更大，希望 ≥70% 单元超越最优单模型。

**实验结论（全部实证，230 天滚动）**：
1. **窗口长度**：60d 相对提升最高（+12.2% vs 等权），45d +10.4%，30d +10.1%——更长窗略有提升，但提升边际递减。**默认 window=30（select 取30）+ NNLS 内部 21 天 OOF 已够**。
2. **粒度**：3段(period) 最优；hour(24组)/point(96组) **因每块样本稀释而降级**（hour 赢 period 仅 44.3%，整体 -20%）。混合粒度（光伏 hour + 其他 period）也未提升（赢 40%）。
3. **负权重（用户提出）**：允许 lo<0 让强模型配>1、弱模型配负——**整体大幅恶化**（fuse 47-99 vs sgdfnet 28-42）。根因：96 点模型高度相关 + OOF 噪声 → 病态解过拟合。**否定负权重方向**。
4. **oracle 上限**：即使事后选当天最优单模型，RT 超越 sgdfnet 也仅 **44.7%**（sgdfnet 太强 23.4 vs 次优 31.2）。**逐单元 70% 超越率是信息理论边界，非算法可及**。
5. **整体口径（关键）**：NNLS period 粒度整体 loss 已优于所有单模型（DA 31.09 < lightgbm 26.75/timesfm 26.39 的融合值 24.67；RT 31.73 < 除 sgdfnet 外全部 31.17-35.07 的融合值 28.19）。**"融合超越最优单模型"用整体口径已达成，逐单元口径受 oracle 限制**。

**落地**：`--weight-granularity {period,hour,point}`（默认 period），NNLSGEF 支持 hour 粒度（weights period 列 `h1..h24`），apply_daily_ledger_weights 按 hour_business 匹配。**生产默认 period**（实证最优），hour/point 仅实验用。

**下一步（若仍要追 70%）**：regime 门控（星期/节假日/负荷波动分桶）+ 预测分歧度特征——需特征工程，ROI 待评估。

### 4.22c RT 优先 sgdfnet + 差模型限权/负权实验（2026-08-16，用户方向）
> 用户判断：sgdfnet 对 RT 极强（已确认无泄漏），应优先；timemixer/rt916 一直差，可手动限权/负权。实验脚本 `scripts/experiments/nnls_ab/run_rt_strategy.py`、`run_negative_robust.py`。

**RT 各模型长期表现（按月 loss，全 230 天）**：sgdfnet **每月最优**（18.5-28.8），timemixer **每月最差**（27.6-45.0），rt916 次差（28.6-42.8），timesfm 居中。排名稳定无翻身。

**策略实验（30d 窗滚动）**：
| 策略 | fuse | 超越sgdfnet | 超越等权 |
|---|---|---|---|
| nnls 纯学（现状） | **35.66** | **33.7%** | 67.0% |
| prior sgd07（初始偏置） | 40.86 | 26.1% | 58.3% |
| cap 差模型≤0.2 | 27.72* | 27.6% | 77.6%* |
| 负权差模型-0.1（fine_proj） | 36.3 | 28-30% | **71-72%** |

*（部分样本，SLSQP 失败致 n 少，有偏）

**全样本（300 单元）结论**：
1. **nnls 纯学整体 fuse 最优**（35.66），超越 sgdfnet 33.7% 最高。
2. **初始偏置 sgdfnet（prior=0.7）有效**：早期实验 44.3→40.9，但全样本下不如纯学。
3. **负权重（仅差模型负，fine_proj）**：超越等权 71-72% 高于 nnls 的 67%，但整体 fuse（36.3）略差于 nnls（35.66）、超越 sgdfnet 也低。**负权方向部分有效但整体不敌纯学**。
4. **生产维持 nnls 纯学**（period 3段）。

**SGDFNet 深调参**（时间无所谓，9 配置网格）：**B 配置（lr02_iter500）仍最优**（SMAPE 0.2536）；更小学习率 lr01_iter2000 MAE 略降但 SMAPE 升（0.2615）。**已到甜点，调参空间尽**。模型层面改动风险大无收益，不动。

### 4.22e ⚠️ 只用最强单模型 vs 融合 决策调研（2026-08-16，见 docs/只用最强单模型_vs_继续融合_调研报告.md）
> 用户方向：sgdfnet 全面最优且负权无法改善，考虑只用 sgdfnet（深度学习模型慢且差）。

**实证（201 单元滚动回测）**：
| 任务 | 只用最强单模型 | NNLS融合 | oracle上限 |
|---|---|---|---|
| RT | **sgdfnet 32.02** | 37.08(+5.07) | 27.17(赢26.9%) |
| DA | 历史最优 32.05 | 32.74(+0.69) | 24.67(赢60.2%) |

- **RT 融合显著拖累**（+5.07）；DA 融合略逊（+0.69）但口径敏感（整体口径融合 24.67 优）。
- sgdfnet **68.6% 单元最优**，所有 regime（工作日/周末/光伏/四季）都最优；误差相关 0.63-0.68。
- **文献支持**：MoE "expert collapse"——路由塌缩到主导专家是正确行为；"Do We Really Need Deep Learning"——时序不必 DL；GBDT 转折点比 LSTM 好 22-34%。
- **结论**：RT 应只用 sgdfnet（用户已确认方向，报告已写，落地待用户定选项 A/B/C）。DA 暂保持 NNLS 融合（用户决定）。

### 4.22f ✅ SLSQP 软门控学习器（用户加入，实证 RT 最优，2026-08-16）
> 用户自己在 `fusion/weights.py` 写的 SLSQP 权重学习（软门控），要求融合进现有策略并实验验证。已接入生产。

**算法**（`fit_weights_from_long_table` / `fit_segment_weights`）：
- 目标 = **smape_floor50(y_true, Σw·pred) + reg×‖w−prior‖²**（直接优化 SMAPE，prior=1/MAE 初始化）
- `scipy SLSQP` 优化，bound 可调（生产用 [0,1]），sum=1 约束，scipy 不可用时回退投影梯度下降
- 96 点兼容：需传 `resolution=res` 且从 ds 推导 business_period（已修 weights.py）

**实证（201 单元滚动回测，reg=0.2, bound[0,1]）**：
| 任务 | NNLS(现有) | SLSQP软门控 | 最优单模型 |
|---|---|---|---|
| RT composite | 37.08 | **33.54** | 27.17 |
| RT SMAPE% | 25.91 | **24.16** | 20.11 |
| RT 赢等权 | 66.7% | **77.6%** | — |
| RT 赢最优单模型 | 15.9% | **26.9%** | — |
| DA composite | 32.74 | 34.36 | 24.67 |
| DA SMAPE% | 25.46 | **24.48** | 19.55 |

- **RT 全面优于 NNLS**（composite -3.54, SMAPE -1.75, 赢等权/赢单模型大幅提升）——SMAPE 目标比 NNLS 的 MSE 更匹配评价指标。
- **DA composite 略差（+1.62）但 SMAPE 优**——SLSQP 直接优化 SMAPE 所以 SMAPE 好、MAE 略差。
- **负权 bound（[-0.5,1.2]）不如非负 [0,1]**（33.54 vs 36.13）——超参扫描确定。
- **超参**：reg=0.2（0.05-0.5 扫描，0.2 最优）、bound [0,1]。

**接入**：`--weight-learner smape_reg`（cli/parser.py 已加）；ledger_weight `_learn_weights_for_task` 分支。融合阶段叠加 `model_quality_gate`（weight-prune-threshold=0.05）自动剪低权模型。端到端：RT fuse 后 pruned rt916/timemixer/timesfm（段1/段3 只剩 sgdfnet）。
历史实验建议（SUPERSEDED FOR FORMAL 96）：RT 用 smape_reg，DA 可保留 nnls 或 smape_reg。

### 4.23 ⚠️ 指标审核教训：采样窗口会翻转结论（2026-08-16，用户质疑"数字对不上"）
> 用户发现会议文档实验部分数字混乱（NNLS 出现 37.46/35.66/37.08 三个值、oracle 28.49/22.74/27.17 三个值）。根因=**不同实验脚本采样不同**（step=3→201单元 / step=2→300单元 / step=1→3000单元），不同采样下相对排序会变。

**权威结果（全程 step=1，3000 单元，30d 窗，同口径 composite+SMAPE 双列）**：
- **RT**：BGEW 32.99 / SLSQP 33.06（并列最优，赢等权 78.5%/76%）> NNLS 36.74（66.3%）> 等权 40.34；oracle 27.16。
- **DA**：BGEW 30.98 > NNLS 31.19 > SLSQP 31.54 > 等权 32.58；oracle 22.93。
- **关键翻转**：之前"BGEW 输给等权 45.8%"是 step 采样偏差（201 单元噪声）；全程 BGEW 从没输过。**NNLS 是 RT 三者中最差**（MSE 目标不匹配 SMAPE 评价）。

**教训（写死）**：
1. **实验必须统一采样**：对比方法一律 step=1 全程（或明确标注采样窗口），否则结论不可比。
2. **报告必须双指标**：composite + SMAPE% 都给，且注明口径（composite=0.7*SMAPE%+0.3*MAE%）。
3. **排名比绝对值重要**：采样少时看排名方向（如 slsqp>nnls），采样多时看具体值。
4. 统一基准脚本：`scripts/experiments/nnls_ab/run_unified_bench.py`（step=1 全程，输出 dual-metric 表）。

**结论修正**：RT 融合用 BGEW 或 SLSQP（并列最优），NNLS 不推荐；DA 三个融合都优于等权，BGEW 略优。oracle 上限 RT 27.16/DA 22.93，融合与其差距是信息边界非算法问题。

### 4.24 FeatureStore 特征预计算设计（2026-08-16 启动，S1 盘点完成）
> 用户确认启动。设计稿 `docs/FeatureStore_特征预计算_设计.md`。背景调研见 `docs/archive/agent-research-2026-08/特征预计算_FeatureStore_与WarmStart增量训练_调研报告.md`。

**动机**：每个模型每次预测都 read_excel（96 点 30MB 实测 **49.8s**）+ 各自算 shift/rolling → 重复劳动。物化一次后读 parquet ~0.1s → 单日省 3-7min，214 天回测省 10-25h。

**S1 特征盘点关键结论**（explore 全代码核验）：
- shift 常量必须 **resolution 化**：各模型"48h/168h"命名是 24 点遗留，96 点实为 2N/7N 行。
- **双命名空间** da/rt（物理分离），p56 遮蔽在 RT 物化时统一施加。
- **两阶段依赖**：SGDFNet da_anchor / TimeMixer da_values / RT916 da_pred 都依赖 DA 产物 → 物化顺序 DA→RT。
- ⚠️ **SGDFNet `delta_lag_1` 与 `_safe_hourly_history` 硬编码 shift(24)**：96 点下 = 6 小时（非前一日），是唯一 resolution 隐患，物化时需显式 resolution 化（当前 RT 生产用 SGDFNet，A/B 确认）。
- daily 统计特征（LightGBM morning/prev_day、RT916 prevday）= "groupby 业务日→shift(1天)"，注册表声明为 daily_stats。
- TimesFM 是段窗口+exog 无行式特征，物化价值最低。

**S2 待办**：实现 `utils/feature_store.py`（ensure/build/slice）+ 单日逐位 diff 零损失验证。建议先做 1 模型（SGDFNet 或 LightGBM DA）验证再铺开。

### 4.25 ✅ FeatureStore S2 零损失验证通过（2026-08-16）+ SGDFNet 硬编码修复
> 用户确认先 1 模型验证。验证脚本 `scripts/experiments/feature_store_ab/verify_zero_loss.py`。

**S2 结果（LightGBM DA 96 点）**：
- `utils/feature_store.py` 实现（注册表 + 物化 + 切片 + manifest 指纹）
- **零损失 PASS**：24 特征列 × 154848 行，最大绝对差 = 0，与现状 feature_engineering 逐位一致
- **提速实测**：物化后切片 17.58ms/次 vs 现状 read_excel 76.1s/次 → **~4326x**；单日 5 模型省 ~380s（6.3min）；214 天回测约 22.6h → 19s
- 缓存键 = 特征版本 + 源文件指纹（mtime+size）；DA 矩阵先做，RT（p56 遮蔽）后续

**SGDFNet delta_lag_1 硬编码修复（模型自适应）**：
- `_safe_delta_history`/`_safe_hourly_history` 所有调用显式传 `lag_hours=resolution`
- 修复 96 点下 shift(24)=6h 隐患 → shift(96)=1天（单元测试验证）；24 点行为不变
- 直接 96 点调用输出 96 行正确（修改安全）
- ⚠️ ledger 里 sgdfnet 曾输出 24 行 = `--data-path` 默认指向 24 点文件（`data/shandong_pmos_hourly.xlsx`）的用法问题，96 点需传 `--data-path data/shandong_pmos_96_full_v2.xlsx` 或走 sync，非代码 bug

**S3 待办**：ledger_predict 前接 ensure + 模型 adapter 改读切片 + 全量回归。建议先 LightGBM DA（已验证零损失）。

### 4.26 硬编码隐患排查 + FeatureStore 全模型 parquet 接入（2026-08-16）
> 用户要求：排查 24→96 硬编码隐患；FeatureStore 扩展到所有模型后验证。

**24→96 硬编码隐患（explore 全代码排查）**：
- 🔴 **TimeMixer is_peak/is_solar + sin/cos 分母错误（已修）**：`repro_pipeline.py` 96 点下 `hour_business` 是业务小时(1..24)，却按"槽1..96"用 `pp=32` 判断 → is_peak 恒1/is_solar 恒0；sin/cos 分母 `/resolution`(96) 使日周期只覆盖 1/4 圈。**修复**：is_peak/is_solar 用业务小时规则（`hb>=17|hb<=8` 峰、`9<=hb<=16` 光伏），sin/cos 分母固定 24。验证：is_peak/is_solar 非恒值、sin 覆盖全圈。
- 🟠 assign_period 96 点 period 错标（被 ledger 标准化掩盖）；lightGBM validate_business_day_filled 96 点首尾 6 槽漏检（RT 未启用）；SGDFNet train_min_rows=24*90（短窗会欠训）；指标段列表硬编码 24 点三段。
- 🟡 分类器 cascade_daily 24 点硬编码（96 点已由聚合入口规避）；RT916 find_initial_term "24"=回溯24天找节气（非行数）；死路径 optimize_data_window。
- 历史验证（SUPERSEDED FOR FORMAL 96）：LightGBM DA/RT 滞后、RT916、TimesFM 段机制、ledger 五阶段全部 resolution 化。

**FeatureStore 全模型 parquet 接入（消灭 read_excel ~30s/次）**：
- `utils/data_loader.py`：`load_table(path)` 自适应 parquet/csv/xlsx（parquet 优先）。
- 各模型 loader 替换：
  - LightGBM `infer_da_fix/infer_fix/train_da_fix/train_fix` → `load_table`
  - SGDFNet `load_dataset` → `load_table`
  - TimeMixer `load_data` → `load_table`
  - RT916 `core.py` 4 处 read_excel → `_load_raw()`（load_table）
  - TimesFM 原本 ok
- `utils/feature_store.py` 加 `load_raw()`：xlsx → parquet 缓存（16MB），`--data-path` 指向 parquet 即可提速。
- **实测**：read_parquet 137ms vs read_excel 30s（**226x**）；RT916/TimeMixer 读 parquet 单测通过（160800 行 0.1-0.2s）。
- ⚠️ ledger 全链路验证：LightGBM DA/TimesFM/SGDFNet 在 parquet 下 status ok；timemixer/rt916 训练慢（5min+）数据读取已单测通过，完整链路待跑。

**下一步**：ledger_predict 全模型 parquet 端到端（timemixer/rt916 需长超时）；FeatureStore 特征矩阵物化扩展到 SGDFNet/RT 后接入。

### 4.27 ✅ FeatureStore + SLSQP 软门控全链路实验（2026-08-16，试验区跑通）
> 脚本 `scripts/experiments/feature_store_ab/run_pipeline_slsqp.py`（predict→weight(smape_reg)→fuse，隔离账本 ledger_96 复制避免污染）。

**结果（2026-01-01，96 点）**：
- **链路跑通**：predict(parquet) 0.68s + weight(SLSQP) 5.7s + fuse 2.8s = **~9s**（轻量模型，parquet 缓存命中）
- **SLSQP 权重合理**：RT 段1 sgdfnet 0.935 / 段3 sgdfnet 0.837 / 段2 timesfm 0.588（光伏段）；DA 段2 timesfm 0.878
- **fused 输出**：DA/RT 各 96 行 0 NaN，范围含负价（合理）
- **隔离账本**：复制 outputs/ledger_96 到实验 runs 下，weight 用共享历史（30 完整训练日），predict 产物 append 到隔离账本不污染生产

**教训**：
1. `--models` 是逗号分隔字符串非 nargs list（`--models lightgbm,timesfm`）。
2. **timemixer/rt916 96 点 CPU 训练 >30min**（本机 CPU 瓶颈，skill §4.2），链路验证用快模型（lightgbm/timesfm/sgdfnet）+ 账本历史即可；重模型需服务器。
3. FeatureStore raw parquet 是全链路提速关键（predict 0.68s vs 原 xlsx ~30s+）。

### 4.28 GPU 训练确认 + TimeMixer deterministic 崩溃修复（2026-08-16）
> 用户问"timemixer/rt916 为什么不用 GPU"。实测确认：**GPU 可用且两个模型都走 GPU**。

**环境**：epf-2 torch 2.6.0+cu124，RTX 4060 Laptop 8.6GB，`torch.cuda.is_available()=True`。ledger 调度 GPU_MODELS={timemixer,rt916}，两 pipeline 默认 device_type=gpu。

**实测**：
- 历史实验（SUPERSEDED FOR FORMAL 96）：RT916 设 `RT916_TRAIN_STEPS=24` 后 `设备: cuda`，单日 **151s** 成功。
- **TimeMixer**：直接调 run_monthly_reproduction，cuda 可用，epochs=10/1月窗 **81s** 完成。ledger 里 15min 超时 = 默认 **train_months=12 + epochs=80** 训练量大（估算 10-16min），非不用 GPU。
- 🔴 **TimeMixer GPU 崩溃修复**：ledger 链路曾报 `upsample_linear1d_backward_out_cuda ... use_deterministic_algorithms(True)`——GPU 训练被残留确定性标志卡住。已在 `TimeMixer/pipeline.py:31` predict_range 开头显式 `torch.use_deterministic_algorithms(False)` + `cudnn.deterministic=False`。

**历史教训（SUPERSEDED FOR FORMAL 96）**：跑重模型前设 `RT916_TRAIN_STEPS=24`；formal96 façade 不依赖 shell override。

### 4.29 历史账本冠军学习器代理实验（2026-08-17）
> 仅用于旧预测账本的相对融合验证；由于历史 96 点模型宽表 `actual_*`/`fcast_*` 重复，不能作为真实特征精度结论。

- 输入仅限 `outputs/ledger_96/{dayahead,realtime}/{prediction,actual}`，禁止重新读取错误宽表造特征。
- 30 日滚动窗（23 日训练 + 7 日验证）总计 200 个目标日，运行约 26 秒；训练学习器远低于 4 分钟模型训练耗时。
- DA：冠军锚定的加权 NNLS/有符号候选经验证门控后，三段均优于滚动冠军，均值 composite 改善约 `2.82/0.39/1.50`。
- RT：SGDFNet 在历史上占优，不能靠无约束 NNLS 直接击败；强正则预测形态 meta 门控并以 `rho=0.30` 做冠军锚定软融合用于 `1_32/33_64`、有符号冠军锚定用于 `65_96`，总体优于冠军且日级胜率高于全量替换，但单段仍有半期波动，必须保留按 `(task,period)` 的质量门控与分段审计。
- 结论：只允许作为实验候选，真实数据重新生成并通过独立回测前不得替换生产学习器。

### 4.30 负权文献与本项目解释（2026-08-17）
- Radchenko, Vasnev & Wang, *Too Similar to Combine? On Negative Weights in Forecast Combination*：负权常在高度相关、方差相近的预测之间出现；无约束权重方差大，直接截断/收缩负权通常更稳，并建议把截断阈值作为调参量。
- 这与 RT 账本一致：SGDFNet 与其他模型误差相关性高时，无约束负权会放大外推；冠军锚定、负权上限、冠军最小权重和验证门控是必要的稳定化约束。
- 因此当前实验不采用“允许任意负值”的生产方案，而采用有界残差修正/软融合，并记录权重审计。

### 4.31 稳定性审计（2026-08-17）
- `scripts/experiments/nnls_ab/analyze_stability.py` 对 200 个目标日做成对 bootstrap 与 Wilcoxon 审计。
- 任务总体差值（policy - champion）为 DA `-1.571`、RT `-0.426`，95% bootstrap 区间均在 0 以下；但 RT 单段区间仍跨 0，且部分半段有波动。
- 结论：可以称为“任务总体均值有统计证据改善”，不能夸大为“每个 period/每天稳定超过冠军”；生产接入仍需干净新账本的独立复验。

### 4.32 验证集防泄漏修复（2026-08-17）
- 发现并修正 `run_meta_hybrid_ab.py` 中 RT `65_96` 有符号候选的门控错误：此前用全 30 日拟合权重评估后 7 日验证集，造成验证信息泄漏。
- 修复后验证只使用前 23 日拟合权重，目标日才用全 30 日重拟合；RT 均值改善由 `-0.426` 修正为 `-0.465`，结果更可信。
- 规则：所有验证候选必须先在 train split 拟合，再在 validation split 只评估；不得用 window/all 权重回看 validation。

### 4.33 短窗口方案（2026-08-17）
- 针对“学习器不能比模型训练还慢”的要求，新增 `run_short_window_ab.py`：14 日窗口 = 7 日训练 + 7 日验证；最终预注册配置 DA/RT 均半衰期 7 日（DA 半衰期 5 日仅作敏感性对照）。
- 同一 200 个目标日上，短窗口运行约 17 秒；最终配置任务总体 composite：DA `32.842→31.520`（-1.322），RT `34.675→33.861`（-0.814）。
- 相比 30 日方案，RT 改善更大且近期权重更集中；但 RT `65_96` 单段置信区间仍跨 0，短窗口候选仍须真实干净账本复验，不直接进生产。

### 4.34 窗口扫参脚本的门控审计（2026-08-17）
- 早期临时窗口扫参曾漏把 `champion` 放入 DA 候选集合，导致“候选验证不劣”时即使略差于冠军也会被选中，扫参结果偏乐观。
- 已用包含冠军基线的修正版重跑；正式 `run_short_window_ab.py` 始终将 champion 放入 eligible 集合，且 14 日最终结果以 `short14_h7` 为准。

### 4.35 冠军学习器实验接入与 96 点槽位键（2026-08-17）
- `--weight-learner champion_short` 已作为显式实验分支接入 `ledger_weight`；**历史记录（SUPERSEDED FOR FORMAL 96）**：当时默认 `nnls`。在旧账本上通过 `ledger_fuse` 端到端验证，DA/RT 各输出完整 96 点且无 NaN。
- `build_ledger_training_table` 的通用训练表保留 `ds`、不保留 `business_period`；96 点学习器缺失首选槽列时必须回退到 `ds`，不能回退 `hour_business`，否则 96 点会被压扁成 24 个小时。
- 该接入使用 `--weight-prune-threshold 0` 才能审计负权；仅限污染历史账本相对实验，干净新账本独立复验前不得替换生产默认学习器。
- `scripts/tests/check_champion_short.py` 已覆盖 hourly/15min 两种分辨率，并显式删除训练表 `business_period` 验证 `ds` 回退。

### 4.36 目标日模型完整性门控（2026-08-17）
- 在目标日 `2026-07-19` 的代理运行中，`champion_short → ledger_fuse` 成功输出 DA/RT 各96点，说明无目标日 actual 也可只依赖预测账本执行权重学习与融合。
- 目标日 `2026-07-20` 的旧账本仅有 DA 1/3、RT 3/4 模型预测；学习器明确报 `target prediction table is incomplete`，融合阶段随后拒绝缺失权重。该失败是正确的质量门控，不得用残缺模型集合伪装融合结果。
- 失败信息已细化到 `period` 和 `missing_models`，便于定位是哪一段、哪一腿预测未完成。

### 4.37 权威丰富96点表适配规则（2026-08-18）
- 权威爬虫表的 `时段` 是 `00:15`~`24:00` 字符串，不一定是 1~96 整数；适配器必须先标准化为 period 1~96，不能直接 `to_numeric` 后把全列变成 NaN。
- 24兼容的 `竞价空间` 不是“直调负荷−风电−光伏−外电”，而是：
  `直调负荷 − 地方电厂 − 外电 − 风电 − 光伏 − 核电 − 自备机组 − 试验机组`；预测和实际都必须使用该完整供需公式，已与 canonical 24表逐点核验最大差 0。
- 96权威表到模型输入必须经过独立适配层；默认从 `2022-07-12` 截取，实际值不从24点填充，仅允许缺失预测从24点预测按业务日+小时映射到四个15分钟槽，并写 provenance manifest。

### 4.38 权威价格列名必须覆盖真实 CSV 口径（2026-08-18）
- 权威文件 `data/96/authoritative/pmos_96_全量.csv` 使用 `日前出清价格`、`实时出清价格`；旧模型宽表常用 `日前电价`、`实时电价`，两套列名都可能是实际价格标签。
- 直接读取权威 CSV 的预测/服务器入口必须同时覆盖这两套别名；否则模型预测可以成功，但 actual ledger 会静默缺失，服务器前置校验与回测账本均不完整。
- 服务器启动前必须用 `--actual-data-path data/96/authoritative/pmos_96_全量.csv` 做单日 smoke，并审计 DA/RT actual ledger 均为 96 点、无 NaN。

### 4.39 actual/forecast 真实性审计要排除真实零值退化列（2026-08-18）
- 权威表中的 `试验机组` 实际/预测序列在当前范围内均为 0；按“全值相等比例”会误报为爬虫拷贝，导致干净模型输入无法通过服务器前置校验。
- 拷贝审计应在 `actual` 或 `forecast` 至少一个非零的观测行上计算相等比例；全零/常量退化特征仍需记录，但不应单凭相等判定为污染。

### 4.40 96 点双进程预测与账本写入（2026-08-18）
- 96 点候选链路使用 `--resource-mode split_process`：CPU 子进程强制 CPU 串行，GPU 子进程独占一张卡并串行；不要用共享 Torch 状态的线程池替代进程隔离。
- Windows `spawn` 子进程通过 Queue 返回结果时，父进程必须在 `join()` 后给 feeder thread 留出 flush 时间，不能直接 `get_nowait()`，否则会把成功子进程误判成“无结果 manifest”。
- 预测模型文件和 ledger 文件都必须临时文件写入后原子替换；范围回测每日写 `parts/<target_day>.parquet`，全部成功后再 compact 成 canonical ledger，避免每天重写全历史账本。
- 96 点 split 预测必须对实际账本做严格 96 行门控；actual 缺失时整日失败，不得仅标记 `complete_with_warnings`。

### 4.41 96 点回放严格产物门控（2026-08-18）
- `ledger_full` 计算出的 `strict_classifier` 必须传入 classifier 子阶段；只在父流程判断严格而不传递参数，会让分类器失败后仍生成 `complete_with_warnings`。
- 96 点 `delivery_quality`、`is_existing_final_valid`、`ledger_fuse` 和 final collector 必须把缺槽位、NaN、缺质量门控文件视为失败；不能只写 warning 后继续生成“完整”回放。
- 服务器回测完成后运行只读审计：`python scripts/server/audit_96_artifacts.py --phase prediction --start ... --end ...`；回放完成后用 `--phase replay`，它会逐日检查模型集合、96 槽、账本键、权重、融合质量门控、分类器和 submission。

### 4.42 96 点直接单日全链路账本压缩（2026-08-18）
- `ledger_predict` 的断点续跑模式按目标日写入 `parts/YYYY-MM-DD.parquet`；直接调用 `ledger_full` 时，必须在 `ledger_weight` 读取历史窗口前先执行一次 `compact_ledger`，否则预测阶段虽成功，学习器会误报 canonical `prediction_ledger.parquet` 不存在。
- 96 点分类器桥接当前按小时输出分类决策；p96 是 D+1 00:00，若没有对应小时分类决策，必须显式写入 `final_pred=0`（表示未应用分类器修正），不能把 NaN 带入 96 点纠正产物。

### 4.43 24 点直接价差实验隔离与适配（2026-08-19）
- 价差任务的监督目标是 `实时电价-日前电价`，不是把价差当作普通输入特征；目标日 RT、价差和 `actual_*` 必须在模型输入中遮蔽，目标日 DA 只按已知锚点角色保留，实际价差仅在预测完成后连接评估。
- 隔离实验入口为 `scripts/experiments/spread_direction_24/run_spread_experiment.py`，产物只写 `outputs/experiments/01_spread_24/invalid_leakage/legacy_cutoff_leakage_spread_direction_24/`；模型集合从 `DAYAHEAD_MODELS + REALTIME_MODELS` 去重得到，禁止另抄生产模型列表。
- LightGBM 训练会通过 `LightGBM_MODEL_PATH` 落模型，实验必须把该环境变量重定向到日级实验目录，不能覆盖 `models/LightGBM/`；RT916 的 `PACKAGE_OUT_ROOT` 同样必须重定向到实验区。
- TimesFM 的 dataset/backtest 代码虽支持 `spread`，forecast 的 `TARGET_CFG` 历史上未登记该键；实验只在进程内登记 `价差` 别名，未验证前不得直接改生产 backend。
- SGDFNet 原生 `delta_target=RT-DA` 可直接作为价差预测；Windows 深层中文项目路径会使其原生审计文件超过 MAX_PATH，先在短临时目录执行，再把审计摘要复制回实验目录。
- `FeatureStore.ensure_base()` 与 parquet 缓存已验证可复用于 hourly 价差实验；当前 `split_process` 仍只开放给 96 点，第二阶段应按 `(task=spread, resolution)` 泛化调度，而不是再复制一条 24/96 专用管道。
- 三日探索结果只用于验证链路和发现方向偏置；方向总准确率必须同时报告正向、负向和 balanced accuracy，不能在正负样本失衡时仅按总准确率选模。
- **作废记录（2026-08-19 P0）**：首次 30 日结果及其三日结果不得引用。旧 `build_asof_input` 仅从 target business day 开始遮蔽 RT/spread/actual，导致预测 D 时 D-1 15:00~24:00（决策时点 D-1 14:00 后）仍可见；`spread_lag24` 的 p15-p24 更直接使用了这些不可得实时价差。旧根 `outputs/experiments/01_spread_24/invalid_leakage/legacy_cutoff_leakage_spread_direction_24/` 已写 `INVALID_DUE_TO_CUTOFF_LEAKAGE.json`，旧 ledger/融合/准确率全部失效。
- “互补 oracle 上限高”不等于可融合。所有门控/权重必须做 prequential 评价：目标日只能使用严格更早日期的评价结果；本实验通过修改当天真值但当天融合预测不变的契约测试验证该边界。Pandas `MultiIndex.get_level_values()` 返回 `Index`，布尔匹配用 `==`，不要调用不存在的 `.eq()`。
- 修复后的硬边界是：forecast origin=`D-1 14:00`；RT、spread 和所有 `*实际值` 在物理时间 `> cutoff` 后全部遮蔽（不只是 target day）；D 目标日完整 DA 与 D 目标日 `*预测值` 可见；D 之后预测特征也遮蔽。缓存必须带 `spread_cutoff_dminus1_14_v2` schema，旧缓存禁止复用。
- 裸 `spread_lag24` 永久拒绝：p1-p14 可用 D-1 同时刻，但 p15-p24 的 D-1 同时刻尚未发生。安全基线改为 `spread_asof_lag`（p1-p14 lag24、p15-p24 lag48）、全日 `spread_lag48`、`spread_weekly` 和只取 cutoff 前历史的 `spread_rolling_median`；每条基线输出 `source_max_ds<=cutoff` 供审计。

### 4.44 24 点价差分段 v3 实验契约（2026-08-19）
- 用户最终口径：D 日 DA 也不可作为价差预测输入；D 日 DA/RT/价差标签必须全部遮蔽。训练可使用历史实际类电网特征，但 target-day 推理必须使用完整预测类电网特征；验证/测试预测始终使用历史保存的 forecast 特征，不能用验证日 actual 特征制造虚高结果。
- v3 实现位于 `scripts/experiments/spread_direction_24/spread_contract.py` 和 `run_masked_spread_experiment.py`。完整历史实际电网字段只允许 D-2 及更早；D-1 仅保留截止14:00的价差观测。每个模型必须通过 `segment_model_id` 记录 `1_8/9_16/17_24` 三个独立时段输出，每段严格8点。
- Masked Direct 以 D-1 p1-p14 价差+显式可见标记作为输入，p15-p24 遮蔽；Safe Mixed Lag 只用 D-2 同时段替代不可用块，并记录 `价差来源滞后日=2`。两者不得共用输出目录；缓存签名必须包含输入方案、模型、训练窗口、epoch、seed 等运行参数。
- 连续价差融合脚本 `fit_spread_fusion.py` 只学习非负归一化连续价差权重，按三个时段分别加权后再对最终值取 sign；不建立正/负两套权重，也不把连续预测提前变成类别投票。权重只使用开发窗拟合，测试窗冻结；测试结果同时对照最佳单模型、等权和统一全日权重。

### 4.45 24 点价差正式生产模拟的执行隔离（2026-08-19）
- 正式模拟参数门控：`training_months>=12`、`TimeMixer epochs>=10`、`patience>=5`、逐日运行、cutoff=14:00；CPU 模拟要求 deterministic，CUDA TimeMixer 因 `upsample_linear1d_backward` 不支持严格确定性，必须记录 `deterministic=false`。
- Windows epf-2 环境中，即使设置 `CUDA_VISIBLE_DEVICES=-1`，长窗 CPU TimeMixer 仍可能触发 `python.exe` 的 `nvdxgdmal64.dll_unloaded / 0xc0000005`；不要把该崩溃误判为数据泄露或模型逻辑错误。实测单日 CPU 可通过，但 30 天长跑不稳定。
- 稳妥方案是模型批次隔离：LightGBM/SGDFNet CPU 批次与 TimeMixer CUDA 批次分别运行，再用带源批次、SHA256、行数和状态校验的 aggregate manifest 合并；禁止用“部分模型完成”的账本直接做融合。
- 高参数 30 天模拟验收必须同时检查：30×模型数×24 行、每 `(target_day,model,period)` 恰好 8 行、`segment_model_id` 全量存在、`segment_training=true`、预测/实际价差无 NaN、所有日模型状态均为 `ok`，并用冻结的 15/15 日切分评估融合泛化。

### 4.46 24 点价差共享缓存与首轮填充策略结果（2026-08-20）
- `utils/feature_store.py::ensure_spread_base()` 负责一次物化 hourly spread 公共基座；as-of 视图必须按源 SHA、FeatureStore 版本、target day、input scheme 和 cutoff 签名，跨模型批次只读复用。
- Windows 深项目路径下，sidecar manifest 不能跟在长 parquet 文件名后追加 `.manifest.json.tmp`，会触发 MAX_PATH；使用短 sidecar 名称，并继续使用 tmp+replace 原子写。
- 30 天 Safe Mixed 预筛结果：同槽位滚动中位数 60.42%，lag48 54.03%，weekly 50.83%，as-of lag 54.31%；滚动中位数是下一步 Safe Mixed 派生 view 的优先候选。
- 加入候选后静态分段融合测试 61.67%，严格历史 prequential dynamic reliability 测试 62.08%；回顾性分段-日 oracle 83.06%，逐点 oracle 95.56%。oracle 只能用于衡量互补余量，不能直接学习生产权重。

### 4.47 24 点价差扩展权重学习窗验证（2026-08-20）
- 为验证缓存加速后扩大权重学习窗的收益，已将 2026-06-16～2026-08-14 的严格 Masked Direct 三模型与 Safe Mixed 四个基线合并为 60 天、7 模型、10080 行候选账本；truth 对齐、NaN、模型日完整性通过 `combine_ledgers.py`。
- 在同一候选池上比较 30/30、40/20、45/15、50/10 的冻结时序切分。学习出的分段权重测试准确率分别为 56.53%、57.29%、58.06%、57.92%；对应等权基线为 61.11%、62.29%、60.56%、58.33%。扩窗改善了证据量但没有自动消除权重过拟合。
- 当前推荐“滚动同槽位中位数 + SGDFNet”作为主/互补组合，等权或收缩到等权/全局锚点，不直接上线自由分段权重。任何动态门控必须只使用严格早于目标日的历史结果；本次 15 天 warm-up、30 天滚动历史的 45 天 prequential 测试 pair-gate 为 61.94%，仍属于实验区证据。

### 4.48 24 点价差 LEAR/共享主干验证与工程调研（2026-08-20）
- 价差数值 sMAPE 必须使用 signed-spread 定义：`100/n * Σ 2|pred-true|/(|pred|+|true|)`；不能套用价格 sMAPE 的 `max(value, 50)` 地板，否则会破坏负价差的相对误差含义。两者均为0记0%，仅一方为0记200%，报告字段为 `spread_smape_pct`，方向准确率仍是主指标。
- 在同一严格因果、60天、24×7模型账本上运行实验区 LEAR：`lear_shared_lasso` 方向 **62.15%**、balanced **50.20%**、spread sMAPE **144.41%**；`lear_segmented_lasso` **60.21%/50.09%/145.35%**；共享主干 MLP **52.99%/49.28%/145.66%**。全日实际符号先验为负向 `900/1440=62.5%`，所以 LEAR 的总准确率不能被误读为有效方向能力；共享主干 MLP 当前不应进入候选池。
- 旧候选池中滚动同槽位中位数仍是单模型方向最优（61.39%，balanced 52.33%，sMAPE 133.88%）；SGDFNet 为57.71%/53.31%/141.26%。滚动中位数与 SGDFNet 的逐点回顾性 oracle 为76.32%，但固定数值加权在60天上最高仅约61.60%，说明主要瓶颈是状态/时段识别而非简单平均；oracle 只用于估计互补余量，不得用于生产权重。
- 预验收必须同时报告负向先验、正向召回、负向召回、balanced、MAE/RMSE、signed-spread sMAPE、按日波动和最差日；至少执行一个严格 prequential 测试窗。单纯优化总 accuracy 会把“全预测负”误当作进步。
- 外部工程经验与本项目一致：LEAR 的价值在于高维滞后结构的稀疏正则和可复现基线，而不是换一个黑盒；电价研究应使用强简单基线、多年/多市场或更长滚动窗口、滚动起点评估和显著性检验。迁移到24/96价格链路时，优先复用“因果账本 + FeatureStore + 缓存签名 + walk-forward/DM检验”四件套，不直接复用价差模型名称或24点小时特征。
- 共享主干迁移规则：先做 resolution-aware 的 shared trunk + slot embedding/segment head A/B；只有在 balanced 和 signed-spread sMAPE 同时不劣于滚动中位数/SGDFNet、且 prequential 不劣时，才考虑接入价差生产模拟；不得因理论 oracle 较高而绕过因果和门控。

### 4.49 96点权重学习器负权与滚动策略门控（2026-08-20）
- 实验入口为 `scripts/experiments/weight_learner_96/run_weight_learner_96.py`，只读复制后的 `outputs/96/feature_store/.../ledger`，产物统一写入 `outputs/experiments/`，不调用 `ledger_weight`、`ledger_fuse`、`ledger_classifier` 或 `final_outputs`。
- 权重拟合使用 floor-50 sMAPE 目标、`sum(w)=1` 和有界 signed weights；负权只允许实验区使用，适合让 RT SGDFNet 作为锚点、由其他模型做反向修正。模型子集由历史 gate/候选策略决定，不能用目标日真值回溯筛选。
- 逐日即时选择容易在 96 点三个时段之间抖动。`validation_rolling` 策略只汇总严格早于目标日的 validation 分数，并用45日历史稳定策略；本次 2026-02-01～08-14 因果回测中 DA selected floor-50 sMAPE 21.967% vs TimesFM 24.845%，RT 22.764% vs SGDFNet 22.885%。RT 的总体 bootstrap CI 跨0，不能宣称显著优于；但 2026-06～08-14 留出段 RT 为25.172% vs25.747%。
- 必须同时保存 `weights_audit.csv`、`selection_audit.csv`、`significance_daily_smape.csv` 和输入哈希；验收至少检查权重和为1、预测有限值、月度指标、留出段和负权/剔除模型统计，不能只看总体一个数字。

### 4.50 96点预测值状态感知融合与理论上限（2026-08-20）
- 理论上限必须拆成不可部署的 hindsight oracle：逐点 oracle、日内分段 oracle、以及候选策略 oracle；本次真实账本上原始模型逐点 oracle 约13.55%/13.57%，但不能作为生产精度目标。候选策略日-分段 oracle 约19.85%/21.53%，用于估计仍有多少状态识别空间。
- 新增 `regime_selector`：只用目标日各模型预测的均值、波动、极值、低价比例和模型间分歧作为特征；标签和门控只来自严格更早日期的候选策略损失，历史尾部验证不通过则回退 champion。目标日实际值只在当前预测输出后写入未来历史。
- 2026-02-01～08-14 的最佳实验组合是 DA `fixed_nonnegative`、RT `regime_selector`：DA 21.863% vs TimesFM 24.845%，RT 22.636% vs SGDFNet 22.885%；RT bootstrap CI 约[-0.481,-0.030]，Wilcoxon 仍未达到显著性门槛，不能直接上线。
- 实验结果说明“固定非负融合 + RT 状态门控”比自由逐日策略更稳，但实时月度仍有个别月份略差；接入生产前必须先做 shadow，保留 SGDFNet 回退和逐任务/时段门控审计。

### 4.22d 96/24 链路分离设计（2026-08-16）
- **96 是主链路，24 是新增**。已隔离：
  - **历史实现（SUPERSEDED）**：当时96用 `outputs/ledger_96`+`outputs/runs_96`；当前 formal96 已切换为 `outputs/96/{ledger,runs,cache,runtime,sync}`，旧根只保留 legacy/research；24 仍用 `outputs/ledger`+`outputs/runs`。
  - NNLSGEF **resolution 感知**：24 点三段（1_8/9_16/17_24，24行/天）、96 点三段（1_32/33_64/65_96，96行/天）自动适配，同一 learner 代码。
  - `--resolution hourly|15min` 全局切换。
- **96 成果可迁移 24**：NNLS 权重学习器天然支持 24 点（同代码），`--weight-granularity` 通用。
- **注意**：24 点账本（outputs/ledger）当前不完整（2026-01-02~01-24 缺大部分天），30 天窗权重学习暂无法跑通——**需数据积累**。代码层已验证 NNLSConfig 对 hourly 正确输出三段。

### 4.22 ⚠️ 服务器回测环境（2026-08-16）
- 智川云 `sc01-ssh.gpuhome.cc:30486` 当前 **SSH 连接被拒**（可能关机/迁移）。
- 历史研究账本：`outputs/ledger_96`（prediction 2025-12-01~2026-07-20 / actual 到 07-18，232 天）仍可做 legacy/research 权重实验；**不得**作为 formal96 learner 默认输入。formal96 只读 `outputs/96/ledger`。
- 全量 GPU 回测（TimeMixer/RT916 训练）需服务器；本机 CPU 只跑轻量验证。
- git push 本机代理坏：`git -c http.proxy= -c https.proxy= push origin main`。
- **现象**：perf_knobs 的 TF32 + cudnn.benchmark 看似打开，实际运行被 seed 函数废掉。
- **根因**：`utils/reproducibility.py:set_global_seed` 无条件 `cudnn.benchmark=False` + `set_float32_matmul_precision("highest")`（=关 TF32）；RT916 自己的 `core.py:set_seed` 更狠，无条件 `cudnn.deterministic=True` + `benchmark=False`（conv 重模型最伤）。TimeMixer/RT916 训练前都会调用。
- **教训**：改 perf_knobs 必须同时查 seed 路径；想恢复加速应给 set_global_seed 加 `OPTIM_*` 开关或去掉 `"highest"`/deterministic 硬编码。
- **其他已核实**：`torch.fft.rfft` 不支持 BF16（RT916 每前向 2 处 FP32 往返）；TimeMixer 短程预测（pred_len 8/32）论文官方建议 scales=1 非 3（PDM FLOPs 省 43%）；两模型均未用 torch.compile，且训练循环每步 `loss.item()` 有同步开销。详见 `docs/TimeMixer_RT916_训练耗时热点调研报告.md`。

### 4.62 96点 R1 paper-protocol 与 24点桥接的信息边界（2026-08-22）
- R1 paper-style 15min 协议在山东可稳定复现约89%~91%方向准确率，但使用 `spread_lag1`、目标日进行中的实际负荷/状态等近时刻信息，**不是 D-1 14:00 一次性预测 D 日96点的可部署能力**。
- 任何 `96→24` bridge 若输入来自该 paper-protocol 预测，即使最终用24点 canonical truth 验收并超过70%，也只能标记为 privileged/oracle bridge；不能计入正式24点70%门槛。
- 真正可部署 bridge 的上游96预测必须单独通过 forecast-origin 审计：target-day actual 不入输入、target-day DA 按当前 spread-v3 契约不入输入、D-1 cutoff 后 spread/RT 不入输入。
- 2025→2026 的首版 safe 96 Ahead q50 direction 仅约55.8%，其96→24 bridge近期约61.9%；当前跨分辨率瓶颈在“把短horizon/R1强状态迁移给提前Student”，而不是简单聚合算法。
- Das et al. 2022 Seq2Seq 在山东48h→next-hour可复现约79.8% direction，但机械扩成D-1 14:00→D日24点会塌缩；短horizon论文高分不能直接外推到长horizon业务链路。

---

## 5. 项目目录地图（防踩乱）

- `outputs/96/` = formal96 唯一生产域，当前只含 `ledger/runs/cache/runtime/sync`；NORMAL 后 `runtime/` 应为空
- `outputs/ledger` + `outputs/runs` = 当前24点独立生产状态；暂不为目录对称迁移
- `outputs/ledger_96` + `outputs/runs_96` = legacy/research 兼容状态，仍被旧分析脚本直接读取，冻结新生产写入但暂不搬
- `outputs/crawl/` = 爬虫 source-mode/调试运行域；源码主 crawler 固定写 `outputs/crawl/runtime_96/`，项目根旧 `output_96/` 已归档；frozen EXE 继续固定写 `<exe目录>/output_96/`
- `outputs/archive/` = server backtest、legacy96、历史 export/diagnostic 的只读归档；旧 `prediction_results` 已迁入 `archive/historical_exports/`
- `scripts/sync/` = 数据同步/合并/回填脚本（sync_data、sync_data_96_core、build_96_full_table、backfill_*）
- `scripts/tests/` = 回归/验证测试脚本（check_*.py、verify_*.py）
- `scripts/crawler/` = 爬虫子模块（按 `auth/`、`collect/`、`sync_db/` 三类组织）
- `dist/crawler/` = 甲方交付包（当前唯一生产入口 `crawl_96_auto_v6.exe`；40,807,703 bytes，SHA256=`2ad77acfd34bd188b982638fb35f015747711540a342e714c7e7bc1ddf904f31`，与归档的 v3-before-auth-recovery-v6 二进制完全相同）
- `dist/archived_crawlers/` = 旧爬虫 exe 归档（auto_fill_96/run_full/auto_crawler_v2/backfill_*，git 忽略）
- `dist/audit/` = db_audit 审计工具；`dist/build_artifacts/` = 构建中间产物（build/venv_build/pyi_tmp）
- `dist/agent_artifacts/` = agent 遗留归档（旧 runs/HAR/调试产物）
- **dist/ 整体 gitignore**（.gitignore `dist/` + `*.exe` + `*.spec`），exe 不进仓库
- `_archive/` = 遗留代码（保留追溯，不参与生产）
- `data/remote_96/` = 96点本地镜像（parquet/raw/metadata）；`data/shandong_pmos_96_full_v2.xlsx` = 合并宽表
- `docs/` 权威文档：`PROJECT_LAYOUT.md`、`PLAN_24_AND_96_POINT_FORECASTING_ARCHITECTURE.md`、`24_VS_96_FEATURE_COMPARISON.md`、`96_DEPLOYMENT_GUIDE.md`

### 产出纪律（2026-08-14 立规）

- **新产物禁止散落根目录 / outputs 根**。按类型归位：
  - formal96 管道产物 → `outputs/96/{ledger,runs,cache,runtime,sync}`；历史 export 不再新写 `outputs/prediction_results/`，统一进入 formal final 或 `outputs/archive/historical_exports/`
  - 爬虫/调试日志 → `outputs/crawl/`；打包产物 → `dist/<分类>/`
  - agent 遗留 → `dist/agent_artifacts/`
- **移动脚本必须同步改 import/路径/README/docs/workflows**，并跑回归验证，确保全链路畅通。
- 改脚本路径时留意 `Path(__file__).resolve().parents[n]` 层级：`scripts/tests/` 下用 `parents[2]`。

---

### 4.67 96点源码分类与部署包整理（2026-09-14）
- 当前生产 EXE 是自动登录、数据采集、数据库同步的一体化入口，源码可以按 `auth/`、`collect/`、`sync_db/` 分类，但不能只移动文件；必须同步修正包导入、`__file__` 相对路径、PyInstaller spec 和 SQL 路径后再打包。
- 生产部署目录只保留当前 EXE、配置文件和一个总 README；日前独立程序只能作为辅助补数工具，HAR/JS/备用 EXE/运行产物全部归档，避免误用旧链路。
- 日前正式口径固定为 `DaJyjgfbPlantQuery` 二次出清/最终版，实时使用 `YxJyjgfbPlantQuery` 正式版并允许临时版兜底；两者都必须在写入 canonical 表前通过 96 点和价格非空校验。

### 4.68 96生产验收门禁与续跑来源证明（2026-09-18）
- `--require-target-actual` 是历史结算/验收硬门禁，必须在所有 resource mode 下执行；门禁只能检查**当前 target day 的实际行数**，不能拿累计 actual ledger 的 `rows_after` 与 96 比较，否则第二天起会误判。
- **历史 replay 规则（SUPERSEDED FOR FORMAL 96）**：resume 不能因为 ledger 某天“模型齐全 + 96点齐全”就直接 skip；旧协议的 fixed cutoff/as-of 证据不得冒充 Dynamic-v1。当前 formal runner 必须读取同日 `run_manifest.json`，证明 immutable snapshot、FeatureView PASS/target truth mask、完整生产模型池和一致 resource mode。
- `split_process` 在服务器 A/B 通过前只是候选调度，正式连续 replay 默认 `legacy`。A/B 建议使用独立 output root，只有预测值等价、无稳定性下降且总 wall time 明显下降后才允许晋升默认。
- classifier cache 迁移只能通过语义历史前缀验证：预测特征前缀 + 真正训练标签前缀一致才收养旧 p1。2026-09-18 本机验收已证明 8/15 可从旧缓存收养至 8/14 23:00，8/16 source 延伸仍为 `semantic_prefix_reuse`。
- retention 在服务器验收前只允许 dry-run 计划；ledger、classifier cache、final、manifest 永久排除，自动 destructive cleanup 必须保持关闭。

### 4.69 历史 forecast vintage 不能由 latest-state 表冒充（2026-09-18）
- `epf_pmos_96_full` 唯一键是 `market_date + 时段 + unit_id`，后续采集使用 upsert 更新非空字段；它保存的是最新状态，不保存 forecast 修订历史。`source_captured_at/create_time/update_time` 是审计时间，不等价于历史 forecast 版本库。
- as-of 遮蔽只能保证 D-1 cutoff 后的 realized/RT 不进入模型，不能证明 target-day forecast 就是当时 D-1 15:00 发布的版本。严格历史回放必须依赖独立、带 capture time 的 D-1 forecast snapshot，并且模型输入实际使用该 snapshot。
- 2026-09-18 对 2026-08-15..2026-09-15 的 audit：0/32 天有独立 D-1 snapshot，32/32 标记 `UNVERIFIED_LEGACY_VINTAGE`。这段历史可用于生产机械链路/调度/恢复验收，但不能作为“严格无 forecast 修订泄漏”的最终效果证据。
- server range manifest 必须显式记录 forecast vintage 状态；post-run auditor 提供 `--require-strict-forecast-vintage`，一旦要求严格历史版本，当前 legacy latest-state history 应 fail closed。

### 4.70 96正式链路 anchor fallback 与冷启动门禁（2026-09-18）
- SGDFNet 的正式 target D anchor 必须来自 D-1 DA p1..p96；当 D-1 某槽缺失时，历史中位数 fallback 也不能把 target-day DA 原值混入候选。实现时要保留 target-day mask，并用 counterfactual（篡改 D DA 不改变 anchor）验证。
- `outputs/96/feature_store` 即使具备 30 日以上模型行，也不能因为行数齐全就直接提升为 production ledger；必须逐日核对 provenance、resolution、模型池、完整 DA3/RT4/actual96 与 cutoff 安全边界。无法证明时标为 legacy/影子并 fail closed。
- 正式 `ledger_weight` 不得偷偷回退 `outputs/ledger_96` 或 `historical-invalid-features` 历史。不足 30 个完整日写 `INSUFFICIENT_STRICT_HISTORY`；若新服务器已有旧服务器历史账本，只能通过有审计的 warm-start migration 进入 `outputs/96/ledger`。

### 4.72 formal96 G.1 收尾门禁（2026-09-19）
- formal96 完整入口必须先用生产 `ledger_weight.select_complete_training_days()` 做严格 30 日 DA3/RT4/actual96 readiness；不足时写 `INSUFFICIENT_STRICT_HISTORY`、`FAILED_NO_DELIVERY` 并在模型启动前 fail closed。`--predict ...` 单任务入口仍可运行以积累正式账本。
- `ledger_full` 是 root `run_manifest.json` 唯一 owner；子阶段只能写 `runtime/stage_manifests/`，不能覆盖 root。每次 attempt 先原子写 `running`，中断/异常留下当前 attempt 的 `interrupted/failed`，不得复用上一轮 `complete`。
- `--finish` 必须验证完整 prediction provenance（production/split_process/both、DA3/RT4 每日96槽、as-of/cutoff、SGDF D-1 anchor、RT916 stride=24、ledger append）后才能进入 weight；formal96 合同失败禁止 emergency/degraded fallback 和新 submission。

### 4.73 formal96 ledger 是可迁移、每日增长的生产状态（2026-09-19）
- `ledger_weight` 只从 `outputs/96/ledger` 读取最近 30 个**因果完整日**。**历史 fixed D-1 15:00 serving 说明（SUPERSEDED FOR FORMAL 96）**：当时 formal96 固定 `history_lag_days=2`；当前 Dynamic-v1 的 serving 可见性由 DB sync → immutable snapshot → FeatureViewBuilder 决定，T-2 完整真值约束仍保留。完整 `--96 T` 在 readiness 前先幂等结算 T-2 actual；预测结果继续 append 到 ledger。因此换服务器时优先迁移 ledger，而不是复制一堆 daily runs 后重建学习历史。
- warm-start 必须走 `scripts/server/bootstrap_96_production_ledger.py`：与正式 learner 同义，从 T-2 向前最多回看90日并选择最近30个 DA3/RT4+actual96 完整日 → source audit → staging 合并 current production ledger → `history_lag_days=2` 正式 selector readiness → 原子 promote；T-1 prediction 只有整池96槽且 cutoff 合法才可选携带。失败不得污染 production，重复执行必须幂等，并写 `bootstrap_manifest.json` + source hashes。
- 迁入历史必须保留原始 `data_cutoff/source_file/run_id/model_version`，禁止为了满足新协议而篡改旧 cutoff。早于当前 serving boundary 的历史可作为 conservative operational warm-start；这只证明不会多看未来且可启动 learner，不代表旧行具备当前模型 protocol 或 strict historical forecast-vintage 资格。
- 2026-08-16 实证：warm-start prediction=2026-07-16..08-15、actual=2026-07-16..08-14；DA/RT learner 明确选 08-14→07-16 30日，T-1=08-15 不参与训练。`--finish` 与完整 `python main.py --96 2026-08-16` 均 NORMAL/exit0，final=96行0NaN。该证据用于当前暂定边界下的链路可运行验收，不用于最终时间边界/模型精度宣称。
- 单日生产 artifact audit 不应强制 range manifest；多日范围或显式 `--require-range-manifest` 仍必须有 range manifest。daily/range 两种运行形态必须分别可机械审计。

### 4.74 formal96 运行目录必须由调用者拥有（2026-09-19）
- formal96 的 runtime 不能硬编码到 canonical `outputs/96/runtime` 后再让测试/部署自己绕开；应从 resolved `runs_root` 推导 sibling `runtime`。这样自定义/隔离 `runs_root` 会自动隔离 scratch，24/legacy profile 不需要为了对称而迁移。
- 一次 formal96 invocation 只允许一个 `attempt_<date>_<attempt_id>/`：共享 as-of 在 `attempt/asof/input.parquet`，五模型 scratch 在 `attempt/models/`。NORMAL 后整 attempt 删除；失败只保留一个 attempt 给诊断/TTL，避免 as-of 与 `models_<pid>` 平铺泄漏。
- 同一业务日重复运行不得无限新增 `stale_delivery_<attempt>`；只保留固定 `runtime/diagnostics/stale_delivery_previous/`，下一次 rerun 覆盖它。最终 weights + model-quality gate 必须进入持久 root `run_manifest.decision_snapshot`，大中间 CSV 才能安全按 retention 清理。
- 旧服务器 prediction/backtest evidence 已从 `outputs/96/feature_store/remote_20260101_20260814` 整包迁至 `outputs/archive/server_backtest_96/original_server_prediction_20251218_20260814/`，保留240/240 DA+RT daily runs、ledger/cache/metrics，并增加 archive manifest + 全文件 SHA256。历史 evidence 与当前 production ledger 必须分域；归档不等于晋升为 learner 数据。
- split-process 使用 spawn，子模型不会继承父进程 FileHandler；“模型改标准 logging”本身不足以保证 daily file log。scheduler 必须把 worker root logger 显式追加到本次 `runs/D/logs/pipeline.log`，并用 fresh model smoke 查真实模块 marker。2026-09-19 LightGBM 强制重预测已验证三条 `infer_da_fix` marker 进入 pipeline.log，根 diag 文件未重生。
- retention 的安全前置不是“run status=complete”而是“最终决策已持久化”。`maintenance_96.py` 仅在 DA/RT `decision_snapshot` 均有非空 weights + model_quality_gate 时才允许30天后的 prediction/weight/fuse 进入候选；否则标记 `blocked_missing_decision_snapshot`。这样未来开放 apply 也不会先删证据再发现 manifest 不够解释 final。

## 6. 执行 checklist（改动前逐项自检）

- [ ] 本次改动是否涉及 96 点数据？→ 先验证 actual≠fcast，确认目标列口径
- [ ] 是否动了爬虫？→ 确认用实时接口拿实际值，不污染 actual 列
- [ ] 是否改了共享代码？→ 跑 4 件套回归 + 黄金基线 diff
- [ ] 是否 24/96 分辨率混淆？→ 查 shift 滞后值、账本目录、runs-root
- [ ] 是否动了 `.env`/config.json？→ 确认不含引号、不进 git
- [ ] 是否有新经验教训？→ 写入本 skill + 共享记忆(memory_put, category=domain:efm3/mech)

---

## 7. 本 skill 的维护规则

- 每次踩坑/修复/教训，追加到对应小节（先验证再写入，注明日期与证据）。
- skill 只对本 git 仓库生效，其他文件夹对话不受影响。

### 4.77 formal96 甲方部署必须做 clean-release acceptance（2026-09-20）
- 开发仓 `main.py --96 T` NORMAL 仍不足以证明甲方部署可用；必须额外验证“白名单 predictor release + 独立 ledger state + 外部 DB”在全新目录可独立运行。
- 当前 `build_predictor_release.py` 采用白名单 manifest：只带 application、必要 server/sync scripts、LightGBM/TimesFM 静态模型和 active docs；明确排除 data/outputs/crawler/experiments/tests/build/Agent tooling/ExtremePriceClf/secrets，并对每个 release 文件记录 size+SHA256。禁止覆盖非空旧 release。
- `bootstrap_96_production_ledger.py` 必须与 learner 同义：lag=2、max-lookback=90、选择最近30个完整 DA3/RT4+actual96 日；不能再硬要求30个连续日。T-1 prediction 仅当整池96槽且 cutoff 合法时可选携带。
- `doctor_96_deployment.py --strict-release` 必须验证 release hash、8个 production import、CUDA、DB、runtime write/delete、30日 readiness，以及 TimesFM 实际解析到部署根自己的 `models/timesFM`。旧 generic `PROJECT_ROOT` 曾让干净 candidate 回跳开发机模型目录并尝试联网，已作为部署硬门。
- 2026-09-20 clean candidate v2 已真实 DB full sync 后执行 `python main.py --96 2026-09-20`：DA3/RT4 七腿各96点，TimesFM 从 candidate 本地 checkpoint 加载，SGDFNet anchor=D-1 DA96/fallback=0，weights=9/12，fuse=96/96，submission=96，postflight PASS，next-day readiness PASS，fallback=false，delivery=NORMAL，runtime 清空；candidate artifact audit 与 strict final doctor 都 PASS。
- `outputs/ledger_96/runs_96` 当前 formal reader=0，但仍有 legacy/research reader，因此“不搬”是正确生产决策；科研 reader 参数化与旧资产冷存储属于非阻塞治理，不得为了目录整齐破坏复现。

### 4.76 Dynamic-v1 必须用真实 `main.py --96 T` 才能签生产验收（2026-09-20）
- controlled smoke 会绕过重模型内部路径，不能替代 production acceptance。2026-09-20 真实入口先暴露 SGDFNet Windows 深层 scratch 路径过长；缩短 Dynamic run suffix 后，SGDFNet 真模型输出96点，D-1 DA anchor=96、fallback_used=false。
- live target-day actual 未闭合时，`ledger_predict=complete_with_warnings` 可以是正常状态；`--finish` 只能在 DA3/RT4、96 slots、snapshot/protocol、SGDF anchor、RT916 stride、ledger append 全部严格通过后复用该 Stage1。
- formal96 身份必须由 `production_config.formal_96=true` 直接参与 fail-closed 判定，不能只靠 classifier policy 间接识别；formal96 postflight failure 永远不得 emergency/degraded fallback。
- next-day readiness 必须调用与 learner 同义的 adaptive selector：lag=2、最多回看90天、选最近30个完整日；不能用“连续30个日历日必须齐全”的旧 validator 误报警。
- prediction artifact audit 默认是 live-serving 语义：目标日 actual 可 partial/absent，但已有行必须合法唯一 finite；历史结算需要 exact 96/96 时显式 `--require-target-actual`。
- 生产放行至少需要：标准入口 exit0/NORMAL、DA3+RT4 各96点且同 snapshot/protocol、weight/fuse/final 完整、postflight PASS、fallback=false、artifact audit PASS。真实模型一次成功 + 后续相同 snapshot 的严格 cache reuse 是两层互补证据。

### 4.75 Dynamic-v1 routed serving view 不能直接当训练 truth（2026-09-19）
- 主审复核发现：中央 FeatureView 把 decision day D 的未知 actual/RT 合法填成 forecast/DA/history 后，如果“预测时顺手训练”的模型仍把 D 放进 supervised fit，就会把 synthetic effective 值误当真实 label。controlled seven-leg test double 无法发现这一类泄漏。
- 当前修复：TimeMixer Dynamic-v1 的 train/validation days 必须严格 `< decision_day D`；RT916 Dynamic-v1 的 supervised train end 固定到 D 00:00（即业务日 D-1 的 p96），SGDFNet 本来已按 `< D` 切 train/val。serving context 可以使用 routed D，但 supervised truth 不可以。
- Snapshot 必须按 attempt 独立落盘；同日 rerun 不得覆盖上一成功 Stage1 的 snapshot。`--finish` 不仅检查路径存在，还要读取持久 snapshot manifest 校验 protocol/snapshot_id/target_day/values_path。
- Critical-source readiness 必须在 authoritative raw facts 上先检查，再允许 model-store/24点 fallback；否则旧 fallback 会把“ForecastData/DA 整体断源”伪装成正常输入。
- Preflight 的 PASS 文案也属于 contract：禁止保留“fixed 15:00/p60 是 formal serving 真源”的旧检查。当前 preflight 应明确验证 `formal96_dynamic_snapshot_v1` + SnapshotBuilder/FeatureViewBuilder。
- formal façade 之外的 direct production `ledger_predict` 也必须 fail-closed 拒绝 `feature_store_mode=raw/materialized`；不能只依赖 `main.py` 把 façade 参数改成 off，否则 API/测试/服务器内部调用仍可绕过唯一 FeatureView contract。

### 4.74 Dynamic-v1 snapshot/FeatureView contract hardening（2026-09-19）
- FeatureViewBuilder 必须先验证 snapshot protocol、snapshot_id 和唯一 D/T 网格；缺失 protocol 或重复 `(market_date, period_no)` 必须 fail-closed，不能让 `Path("")` 误把当前目录当成存在的 snapshot。
- formal façade 强制 `feature_store_mode=off`；FeatureStore 只保留显式 legacy/shadow/experiment 入口，不能覆盖 Dynamic-v1 的共享 FeatureView。
- primitive RealityTmp 映射必须覆盖全部8个 canonical actual 字段；每次新增字段后要跑 cell-gap/counterfactual smoke。

### 4.71 96 server A/B 的 worker 口径（2026-09-19）
- 服务器 replay runner 的 legacy 对照可以保持 CPU=1，但 `split_process` 正式候选必须实际传入 CPU=2；只在 scheduler 内部改成2、而 benchmark command 仍传1，会让 A/B 失去验收意义。
- range manifest 必须记录 `cpu_workers`、`gpu_workers`、`dag_aware`、`gpu_serial`，artifact audit 按 resource mode 校验（split_process=2/1/DAG-aware，legacy=1/1）。对应回归覆盖 `benchmark_96_resource_modes.py` command 构造和 `audit_96_artifacts.py` manifest 门禁。

### 4.51 多段权重实验（2026-08-20）
- 在最新 feature_store 96点账本（2026-02-01~08-14）上，将现有三段各拆成两段的六段因果权重实验已完成；产物位于 `outputs/experiments/03_fusion_weighting/weight_learner_96_6segment_20260201_20260814`，未触碰正式链路。
- 六段 `fixed_nonnegative` DA + `regime_selector` RT：DA selected floor-50 sMAPE 21.7400%，RT 22.5565%；优于此前三段对应 21.8627%/22.6357%，但这是未加入相邻平滑约束的实验结果，不能直接上线。

### 4.52 24点价差TimeMixer结构实验（2026-08-21）
- 实验入口为 `scripts/experiments/spread_direction_24/run_timemixer_structure_experiment.py`，只写 `outputs/experiments/`；固定使用12个月（365天）滚动训练，评估窗口为开发30天、确认15天、留出15天，未调用任何权重学习/融合/分类/最终交付阶段。
- 所有结构使用同一严格输入：D-1 p1-p14可见价差，p15-p24使用同槽位滚动中位数，缺失回退D-2同槽位；源最大时间必须不超过D-1 14:00。不能因为训练集使用全部历史标签而把目标日实际价差或cutoff后RT放回输入。
- 留出15天结果：统一24→24 `tm_unified24` 方向 **60.28%**、balanced **55.95%**、MAE **46.13**、signed-spread sMAPE **143.81%**；无权分段 **56.39%/51.48%/46.15/147.41%**；输入加权分段 **59.44%/54.48%/46.63/142.87%**；共享编码器等权三头 **56.94%/51.86%/47.72/143.97%**；困难度加权三头（由开发30日确认9-16最难后固定为 `[0.80,1.35,0.85]`）**55.83%/51.00%/47.47/146.40%**。
- 结论：在TimeMixer内部，“全天统一24→24”比三段独立头和共享三头更稳；输入加权只改善数值sMAPE，未改善balanced方向；困难段加大loss反而过拟合。现有留出窗口SGDFNet balanced约 **61.24%**，因此统一结构虽明显优于旧TimeMixer分段结果，但还不足以直接迁移到其他生产模型或进入权重学习。
- 后续若迁移，只迁移“全天统一输出”的结构假设到SGDFNet/LightGBM实验，并重新使用同一12个月滚动、30/15/15切分；不得把TimeMixer的结构结果直接当作跨模型结论。权重学习必须等迁移对比完成后再开始。
- 实验运行约484秒；SLSQP 出现 bounds clipping warning 但结果有限且 manifest/指标已生成，后续应加入权重平滑正则并审计权重稳定性。

### 4.52 TimesFM cutoff 时间戳修补与动态 LightGBM 链路（2026-08-20）
- `runners/adapters/timesfm_v1.py` 原先对所有 `cutoff_date` 无条件加一天；当 RT 传入 `YYYY-MM-DD HH:00:00` 时会暴露 cutoff 后数据。现改为：纯日期按日末解释，显式时间戳严格按时刻截断，并已用 13:45/14:00/14:15 临时 CSV 测试通过。
- LightGBM 动态窗口已从 `cli/parser.py` → `pipelines/ledger_predict.py` → `LightGBMV1Adapter` → `lightGBM/main_fix.py` 打通，参数默认关闭；24/96 都走分辨率感知 `main_fix` 动态路径，旧 24 点 direct v1 入口仅在未启用候选窗口时保留。
- 六段权重平滑离线复算：RT λ≈0.15~0.20 仅有约0.08个百分点以内微小收益，DA 变差；不能把平滑默认上线，需保留未平滑六段方案作为当前实验候选。
- 动态 LightGBM 96点实验（clean model_input，严格实验区）：候选[6,12]在 2026-06-16~07-15 将 floor-50 sMAPE 从33.0953%降到25.6043%，在 07-16~08-14 从25.0523%降到22.4577%；但后一窗口 MAE 从118.68升至129.84，说明不能只凭 sMAPE 直接生产化，需同时审查 MAE/极端价与更长留出。
- 60日权重独立留出（2026-06-16~08-14）：三段 selected DA/RT=22.9325%/23.3342%，六段=22.8194%/23.3568%；六段DA小幅改善、RT小幅退化，平滑λ=.2为22.9539%/23.3701%，确认平滑不应默认启用。三段和六段RT均显著优于SGDFNet（60日 bootstrap CI均不跨0），但六段尚未同时稳定优于三段。
- 正式 `smape_reg` 96点试运行暴露并修复 `fusion/weights.py` 元数据污染：`business_period` 曾因 wide 表排除列表遗漏而被当作模型学习权重；现已显式排除 `business_period`，新增 `scripts/tests/check_smape_weight_resolution_metadata.py`，并验证DA/RT权重模型集合只含生产模型、每段权重和为1。

### 4.53 正式 smape_reg 60日 walk-forward（2026-08-20）
- 按正式 `ledger_weight` 的 `smape_reg`、30日严格历史窗口、96点三段和自适应完整日选择，独立回测 2026-06-16～08-14 完成 120 次任务学习、11520 行预测，无目标日真值回看。
- 结果：DA floor-50 sMAPE 22.8219%、MAE 107.37，RT 23.7826%、MAE 77.69；分别优于同期最佳单模型 TimesFM DA 25.4339% 和 SGDFNet RT 23.9915%，但这是同一历史账本上的融合验证，不等于新模型重新预测后的生产精度。
- 产物位于 `outputs/experiments/99_legacy_unclassified/formal_smape_reg_walkforward60_20260616_20260814`。接入正式链路前仍需 shadow、模型集合/权重门控审计和新鲜有效预测集复验；默认 `nnls` 不直接改写。

### 4.54 smape_reg 96点 shadow 融合（2026-08-20）
- 使用正式 `ledger_fuse` 接口、实验 runs-root 和 `weight_prune_threshold=0.05` 完成 2026-08-14 96点 shadow；日前/实时均输出96行、无NaN、质量门控各3段并写出 `model_quality_gate.csv`/`fused_debug.csv`。
- 门控会按段剔除低权重模型并归一化剩余权重；实时本次有64个槽位发生重归一化，说明正式接入必须保留门控审计，不能只检查 fused_predictions.csv。

### 4.55 改进模型与服务器账本合并（2026-08-20）
- 改进版 LightGBM/TimesFM 已在本机隔离 CPU worker 完成 2025-12-18～2026-08-14 共240天重预测，0失败；严格合并后 DA 69120行、RT 92160行，模型集合和每个日期96槽均通过检查。
- 合并器为 `scripts/experiments/re_prediction_96/merge_with_server_ledger.py`，只替换 DA 的 LightGBM/TimesFM 与 RT 的 TimesFM，保留服务器 TimeMixer/SGDFNet/RT916；首次目录层级错误已修正为 `task/prediction|actual/prediction_ledger.parquet`。
- 最新合并账本上的 LightGBM 全窗 floor-50 sMAPE 由25.9601%降至24.6661%；TimesFM因本次修补主要影响边界，整体数值基本不变。不得据此宣称所有任务都提升，需按最终融合和留出窗复核。
- 2026-08-14 shadow 完成 smape_reg→门控融合→分类器→96行 submission_ready，最终输出只写 `outputs/experiments/04_pipeline_audits/re_prediction_96_final_chain_20260814`，未改正式输出。

### 4.56 24/96 训练表分辨率合并键（2026-08-20）
- 旧24点账本虽带 `business_period` 列，但该列全为 NaN；训练表若仅按 `task,business_day,business_period` 合并，会把每天24个预测槽与24个实际槽做成576行笛卡尔积，导致权重覆盖误判。现按该列是否有有效值选择96点 `business_period`，否则回退24点 `hour_business`，并新增回归测试；24点30日权重训练已恢复为2160/2880行并通过覆盖审计。

### 4.57 1.0 动态训练复核与六段权重留出（2026-08-20）
- 1.0 的 LightGBM 确实是逐目标日动态寻优：实时验证截止 D-1 14:00、候选窗口步长2个月；日前使用目标日前一日窗口。2.5 已具备因果候选窗口，但仍保留固定12个月默认，不能把动态模式误当默认生产行为。
- 最新服务器96点账本上的六段（每32点再拆为16点）独立留出 2026-06-16～08-14：日前 selected floor-50 sMAPE 22.8189%，实时23.3569%，均优于同期最佳单模型 TimesFM 25.4339%/SGDFNet 23.9915%；相邻权重后处理平滑 λ=.2 反而变为22.9537%/23.3702%，因此暂不启用平滑。六段结果仍是实验区策略，不自动改正式默认。

### 4.58 smape_reg 正式默认切换验收（2026-08-20）
- **正式当前口径（2026-09-19）**：`cli/parser.py` 的默认权重学习器为因果 `smape_reg/SLSQP`；`nnls` 仅可显式 internal/experiment 调用。24点与96点默认参数分别完成30日 `ledger_weight`，训练行数/覆盖均通过；96点默认 `ledger_fuse` 输出96行且质量门控审计通过。

### 4.59 96点全历史重放的分类器缓存隔离（2026-08-20）
- 并行 replay 多个日期/分片时，分类器默认缓存会共同写 `outputs/cache/classifier/.../manifest.json`，在 Windows 上会触发 `WinError 32`。每个并行分片必须传独立的 `feature_store_root`（分类器桥接取其 parent 作为缓存根），否则不能把单日成功外推为并行全量成功。

### 4.60 改进96点全链路重放验收（2026-08-20）
- 改进账本 2026-01-17～08-14 共210天已完成三分片并行 replay，weight→fuse→classifier→final 全部 NORMAL，无失败；每个日期日前/实时/提交文件均96行且无NaN，共20160行/任务。最终 floor-50 sMAPE：日前22.1136%、实时23.3550%。产物位于 `outputs/experiments/04_pipeline_audits/replay_full_96_improved_20251218_20260814`，并生成月度与汇总指标文件。

### 4.61 24点价差 TimeMixer 结构与跨模型迁移（2026-08-21）
- 统一24→24、三段独立、输入加权三段、共享编码器三头等结构均用同一 FeatureStore、同一 cutoff、12个月滚动训练和30/15/15时序切分；留出结果显示统一24→24最佳（balanced 55.95%），困难段加权 `[0.80,1.35,0.85]` 反而降至51.00%，不能据此启动权重学习。
- 将可迁移的滚动同槽位填充策略应用到 SGDFNet/LightGBM 后，SGDFNet 留出 balanced 61.24%、LightGBM 51.62%，没有证明该填充策略能普遍提升模型；TimeMixer结构收益不能未经模型级重写直接外推到其他模型。
- 证据目录：`outputs/experiments/01_spread_24/invalid_leakage/legacy_cutoff_leakage_spread_direction_24_timemixer_structure_base_20260616_20260814`、`spread_direction_24_timemixer_structure_difficulty_20260616_20260814`、`spread_direction_24_model_migration_20260616_20260814`。正式链路、融合器、分类器均未运行。

### 4.62 96点价差→24点桥接的泄露审计与 strict-v3 重跑（2026-08-22）
- R1 paper-protocol 的约90% 96点方向依赖 target-day/近时刻 realized 15min 信息（如 spread lag1、实际负荷），只能作为 oracle/privileged 上限；即使下游最终用24点 canonical truth 验收，衍生出的66.83%或73%~78% bridge也**不是 D-1 14:00 可部署成绩**。
- 新增 `source_contract.py`：96→24 simple/learned aggregator 默认拒绝任何非 `D-1 14:00`、target-day actual=false、target-day DA=false、D-1 post14 realized=false 的96预测源；故意传入旧 paper-protocol 文件已验证会直接 FAIL。
- 新增反事实 feature audit：篡改目标日整天 DA/RT/spread 或 D-1 p57~p96 后，44个 Ahead 特征最大变化均为0；篡改允许的 D-1 p1~p56 后特征显著变化，审计通过。
- 修复评估偏差：旧 Ahead 对“任一特征 NaN”整行 drop，导致部分月份被静默删样本；strict-v3 改为 LightGBM 原生处理特征 NaN，只要求 target spread 真值存在，并强制每个测试日96/96完整。
- strict-v3 96 q50方向仅约56%（2024 56.21%、2025 55.76%、2026YTD 55.79%，balanced约53%~55%）；2026严格96→24 mean/median/vote约57.9%，balanced约50.5%。OOF learned bridge与2026前向窗口均未形成稳定增益。最近07-01~08-14 raw约61.94%，但全负基线61.67%，因此不得误读为有效方向能力。
- 规则：任何“高频→低频”实验先审上游 forecast-origin；下游真值干净不代表整条链路无泄露。正式24点指标只接受整个上游链路都通过 source manifest + counterfactual boundary audit 的结果。

### 4.63 🔴 24点 direct-spread 的 D-1 完整标签泄露事故（2026-08-23，永久红线）
- 事故：`P6 + causal Similar-Day` 一度在 2026-07-31~08-14 得到约 **69.44%** direction，后续代码级复核发现滚动训练使用 `train_days = ... : idx`，把 **D-1 完整24点 spread label** 纳入训练；但 forecast origin 是 D-1 14:00，D-1 15~24点尚未发生，因此该成绩 **INVALID-LEAKAGE，永久作废**。
- 根因：只按“calendar day < target day”判断历史，没有按“forecast origin 时 label 是否已完整落地”判断。**日历上的过去 ≠ 业务时点上的可见。**
- strict 规则：预测 D 时，完整监督训练标签最晚 **D-2**；D-1 只允许 p1-p14 作为 partial context 特征，绝不能整日进入 fit / calibration / threshold / stacking / feature selection / similar-day label pool / online routing。
- 代码规则：24点 strict-spread 新实验必须统一使用 `strict_train_days()`（或语义等价的唯一 helper），并在 ledger 中写 `training_last_day`，逐目标日断言 `training_last_day <= D-2`。任何裸 `idx-v.training_days:idx`、`day < target_day`、`shift(1 day)` 逻辑都必须视为高风险并人工审查。
- 评估规则：所有新成绩先标 `STRICT/PASS` 或 `INVALID-LEAKAGE`；发现泄露后不得把旧漂亮数字留在“当前最优”表中。跨月份验收必须同时报告 direction / positive / negative / balanced / all-negative baseline。
- 工程教训：**实验前先画信息时间线，再写特征和训练切分；先做 leakage audit，再跑模型。** 以后宁可晚出结果，也不能先跑后审。

### 4.64 24点 numeric spread 动态窗口公平比较与方向双头实验（2026-08-28）
- 6/9/12个月窗口不得各自按比例切验证集；预测 D 时统一使用 `D-1 calendar month ~ D-2` 的同一验证日期，三个候选只改变验证集之前的训练历史。综合损失中的数值项统一为 `validation MAE / mean(abs(shared validation spread))`，不得再用各候选自己的训练幅度作分母。
- LightGBM 若 `objective=regression` 同时传 `eval_metric=l1`，保存模型可能同时出现 L1/L2，默认 early stopping 并非“只看L1”。direct numeric spread 当前使用 `objective=regression_l1 + metric=l1 + first_metric_only=True`；修正后选中模型中位树数从大量1棵恢复到约38~64棵。
- STRICT/PASS 的2026-06~07开发回测：共享验证动态L1方向58.40%、balanced53.19%、MAE87.12；同数据冻结180日分段L1为59.77%/54.59%/87.97。动态方案改善MAE但尚未超过冻结方向基线。
- “二分类方向头 + L1连续值头”仍输出具体数值：分类器只决定连续回归幅度的符号。两月结果 classifier-sign 57.04%/52.64%/87.41，confidence-gate 56.49%/52.34%/86.96，均未稳定超过纯回归符号；尤其六月改善而七月退化，因此不进入1~8月晋级回测。单日高分只作可执行性检查，不作效果结论。

### 4.65 24点 spread 特征筛选、基本面误差与方向损失（2026-08-28）
- Cycle 88 严格实验统一使用 D-1 14:00 origin、完整训练标签截至 D-2。3月+5月18组增删实验中 `full_minus_F7` 最好，方向59.48%、balanced56.52%；但6月+8月1-26确认仅57.14%/55.63%，未通过预设跨月稳定门槛。特征删减必须看独立月份，不能把筛选月高点当稳定提升。
- 基本面误差一阶段 OOF 对新能源/竞价空间误差可有较高相关性，但把预测误差加入 spread 二阶段后全部低于不加误差的 B0；“中间物理量预测得准”不等于“下游指标必然提升”，必须以严格 OOF 的端到端增益验收。
- pseudo-Huber + smooth-sign 在筛选月宏观方向最高59.61%，但 λ=0.4 在3月改善、5月下降4.57个百分点；加入方向梯度并不能替代跨月约束。损失选择应先执行月度跌幅、recall、balanced、MAE门槛，再按方向排序。
- Windows 长项目根接近 MAX_PATH；实验 run 子目录应使用 `confirm`、`fund_error` 等短名。长目录加 `training_boundary_audit.csv` 曾在结果已算完后写文件失败；这不是模型错误，后续应在开跑前检查最终产物路径长度。

### 4.66 24点 spread 去冗余、固定树数与负类校准（2026-08-28）
- `Full-F7` 的220维中删除12个严格常数特征后，筛选和确认预测逐点不变，可采用208维作为等价简化；继续删除完全相同/相反别名后筛选方向从59.48%降至56.52%，再删 `|Spearman|>=.995` 相关簇降至55.51%。LightGBM在 `feature_fraction<1` 时重复列会改变每棵树可见的候选集合，不能把“数学冗余”机械等同于“删除后模型不变”。
- 关闭早停并由同一800树模型截取50/100/220/400/800轮后，3月+5月方向分别59.07%/59.34%/59.48%/58.87%/58.60%；220树最佳，6月+8月确认220树57.14%、100树56.50%。因此原早停的2~5树确实过少，但强制超过220树也无收益；下一基准固定220树，不启动复杂早停重构。
- `DA-RT` 口径下弱类是负价差。负样本权重1.25使负召回49.36%升至56.83%，但方向降至55.31%；更大权重和quantile同样以明显方向损失换召回。因果加性阈值在确认集由57.14%降至56.30%，未通过门槛。类别偏差不能只靠全局权重/阈值修正，后续若继续应做条件化弱类识别，而不是继续加大全局偏置。

### 4.67 服务器冷启动传输与 release 复盘（2026-09-20）
- GitHub 只负责源码/release commit；models、`.env`、formal96 `outputs/96/ledger` 和 old-server source 均是外部部署状态。远端可访问不等于远端含有本地最新 commit，必须先比较 commit，再决定 clone/pull 或 bundle。
- 925MB TimesFM checkpoint 应采用可恢复分片传输，合并后必须 SHA256 与源一致；SSH 并发过多会 reset，因此大文件传完应先关闭传输连接，再安装依赖。
- 部署先构建 `build_predictor_release.py --apply` 的最小 release。直接拿 full research checkout 跑 strict doctor 会被 experiments、crawler、fixtures、`.env` 等开发资产拒绝。
- Python 3.11、Torch 2.6.0+cu124、CUDA 12.4、triton 3.2.0 和根 requirements 必须一次性核验；大 wheel 卡住时可使用官方/镜像的本地 wheel，但不得改版本或安装外部 pip timesfm。
- DB 凭据和 canonical data 应在安装后立即做 doctor/sync 检查；preflight 报缺数据时，不得把“代码已拉下”当成数据库已同步。预测前必须通过 strict doctor、TimesFM 本地 import、ledger restore/readiness 和真实标准入口 smoke。

### 4.68 formal96 日常同步与最终产物覆盖（2026-09-21）
- formal `python main.py --96 DATE` 已切换为：已有本地 mirror 时按可配置最近重叠窗口（默认7天）同步，并额外用 `update_time` 一天 watermark overlap 捕获旧业务日的修正；无 mirror 仍自动走 exact full sync。显式 `sync_dataset --resolution 15min --sync-mode full` 保留为冷启动/全量审计入口，不能把增量路径误称为删除级全库对账。
- `ledger_fuse` 是 formal96 RT 的权威结果，classifier 只 legacy/shadow/replay。`realtime/final/realtime_final_predictions.csv` 每次 finalization 必须从当前 `realtime/fuse/fused_predictions.csv` 覆盖写出；绝不能因旧 final 已存在而复用，否则会出现 fuse 是新结果、submission_ready 仍是旧结果。回归必须同时比较 fuse→RT final→submission 三个数值产物。
