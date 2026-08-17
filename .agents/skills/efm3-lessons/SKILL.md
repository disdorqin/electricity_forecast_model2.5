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

- **项目本质**：山东电力现货价预测，24点（小时级）正式交付 + 96点（15分钟级）辅助，7模型 + Ledger 自适应融合 + 极端价分类器。
- **环境**：本机 `conda epf-2`（CPU only，LightGBM/CatBoost 走 GPU 会崩）；GPU 云服务器 RTX3090 需 `export TIMESFM_DEVICE=cpu`。
- **论文红线**：山东/山西数据**绝不进论文**（仅内部动机）；多市场证据用宁夏/甘肃/陕西/青海 + 公开国际集（Lago/NEM/GEFCom/UniElecPrice）。

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

### 1.2 爬虫日常任务仍在污染
- 定时任务 `auto_fill_96.py`（每天 08:00）和 `run_crawler.py` **仍调预测值接口写 actual 列**，尚未切到 `crawl_market_overview_actual()`。
- 机组价表 `epf_unit_data_96` 滞后约 9 个业务日；`rt_cq_price` 近几日常为 NaN（发布延迟）。
- 改动爬虫时：先读 `scripts/crawler/README.md`，分清**两套爬虫**（国网 PMOS vs AI交易平台 47.114.107.96）。

---

## 2. 24点 vs 96点口径区别

| 项 | 24点 | 96点 |
|---|---|---|
| 价格来源 | `epf_market_data` 全省市场均价 | `epf_unit_data_96` **机组级出清价**（da_cq_price/rt_cq_price）|
| 相关性 | — | 强相关(corr~0.95)但不等价，diff 可达 ±300 元/MWh |
| 特征 | 10 组 fcast/actual | 13 组（多检修/正负备用）；fcast 与24点精确对应(均值聚合)，actual 多列是拷贝 |
| 滞后特征 | shift(24)/168 | **shift(96)/672**，勿机械沿用 24 |
| 实时截止 | — | 固定 **14:00 / period 56**（p56 可见、p57 起遮蔽）|

- 跨分辨率比较：96点 mean 聚合到 24点 后同口径比；度电套利需 ×0.25 因子。
- 独立账本 `outputs/ledger_96`、独立 runs `outputs/runs_96`、`--resolution 15min` 切换，勿混存。

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
| **实时(RT)** | **只到 D 当天 14:00 / p56** 的实时数据 | p57 之后遮蔽，绝不泄漏 |
| **预测辅助特征** | 可用 **D+1（次日）电网特征预测值**（fcast_*）作模型辅助 | **绝不用实际值(actual_*)作 target 日特征** |
| 滞后特征 | 只用历史 actual/fcast（shift(96)/672） | target 日 actual 是标签不是特征 |

### 2b.3 三条硬规则
1. **预测≠实际**：任何特征列若预测==实际 比例 >1% 即视为爬虫污染（甲方数据已确认 0%）。
2. **target 日实际值绝不入特征**：`actual_*` 只能作历史 lag，target 日 actual 是预测标签。
3. **RT 截止 14:00**：p56 可见、p57 起遮蔽（DATA_CONTRACT_96 §7 固定）。

---

## 3. 交付纪律（24点正式链路）

- 五阶段：`ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs`。
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
- **必跑** `scripts/tests/check_preflight_health.py`（11项全绿才可开跑）：
  24点完整性、96点近30天actual≠fcast、Resolution契约、p56截止、账本≥30天。
- **防泄漏确认**：生产链路 ledger_predict 传 cutoff=14（SGDFNet decision_hour/TimeMixer cutoff_hour_rt/RT916 asof_hour 全为14）；
  `pipeline_timemixer_single_task.py` 与 `protocol_b_cutoff.py` 的15:00是**遗留默认**，不走生产主链路。
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
- **修复**：`TRAIN_STEPS` 改为环境变量 `RT916_TRAIN_STEPS` 可配置（默认仍 1 保守）。
- **实测（3个月窗，RTX4060）**：
  - TRAIN_STEPS=1 → ~1000s+（最慢）
  - **TRAIN_STEPS=24 → 98s（18倍提速），SMAPE 0.23-0.32 精度良好** ← 甜点
  - TRAIN_STEPS=96 → 70s 但样本太少(102)过拟合，SMAPE 0.45-0.82 ❌
- **结论**：RT916 调参/回测用 `RT916_TRAIN_STEPS=24`，12个月窗估计 ~6min/天（原 29.8min）。
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
- **实时电价(D+1)预测时不可得** → 必须遮蔽；实时只知道 D 日 14:00/p56 前的

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
- 用户决定：**分类器必须进主链路，最终预测经过分类器**。

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
- 接入：`--weight-learner {nnls,bgew}`，**默认 nnls**（ledger_weight.py + cli/parser.py）。
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
> 完整实验见 `scripts/experiments/nnls_ab/`（run_ab.py 窗口/粒度/参数、run_negative_w.py 负权重、run_hour_select.py 小时选择）。产出在 `outputs/experiments/nnls_ab/`。

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
**生产建议**：RT 用 smape_reg，DA 可保留 nnls（composite 优）或 smape_reg（SMAPE 优，用户偏好决定）。

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
- ✅ 已确认无问题：LightGBM DA/RT 滞后、RT916、TimesFM 段机制、ledger 五阶段全部 resolution 化。

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
- **RT916**：设 `RT916_TRAIN_STEPS=24` 后 `设备: cuda`，单日 **151s** 成功（3 段）。之前实验 30min 超时根因 = **缺 `RT916_TRAIN_STEPS` 环境变量**（默认 1 极慢），非不用 GPU。
- **TimeMixer**：直接调 run_monthly_reproduction，cuda 可用，epochs=10/1月窗 **81s** 完成。ledger 里 15min 超时 = 默认 **train_months=12 + epochs=80** 训练量大（估算 10-16min），非不用 GPU。
- 🔴 **TimeMixer GPU 崩溃修复**：ledger 链路曾报 `upsample_linear1d_backward_out_cuda ... use_deterministic_algorithms(True)`——GPU 训练被残留确定性标志卡住。已在 `TimeMixer/pipeline.py:31` predict_range 开头显式 `torch.use_deterministic_algorithms(False)` + `cudnn.deterministic=False`。

**教训**：跑重模型前设 `RT916_TRAIN_STEPS=24`；TimeMixer 长训练（12月/80epoch）需长超时或减小窗口；GPU 崩溃先查 deterministic 标志。

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
- `--weight-learner champion_short` 已作为显式实验分支接入 `ledger_weight`，默认 `nnls` 不变；在旧账本上通过 `ledger_fuse` 端到端验证，DA/RT 各输出完整 96 点且无 NaN。
- `build_ledger_training_table` 的通用训练表保留 `ds`、不保留 `business_period`；96 点学习器缺失首选槽列时必须回退到 `ds`，不能回退 `hour_business`，否则 96 点会被压扁成 24 个小时。
- 该接入使用 `--weight-prune-threshold 0` 才能审计负权；仅限污染历史账本相对实验，干净新账本独立复验前不得替换生产默认学习器。
- `scripts/tests/check_champion_short.py` 已覆盖 hourly/15min 两种分辨率，并显式删除训练表 `business_period` 验证 `ds` 回退。

### 4.36 目标日模型完整性门控（2026-08-17）
- 在目标日 `2026-07-19` 的代理运行中，`champion_short → ledger_fuse` 成功输出 DA/RT 各96点，说明无目标日 actual 也可只依赖预测账本执行权重学习与融合。
- 目标日 `2026-07-20` 的旧账本仅有 DA 1/3、RT 3/4 模型预测；学习器明确报 `target prediction table is incomplete`，融合阶段随后拒绝缺失权重。该失败是正确的质量门控，不得用残缺模型集合伪装融合结果。
- 失败信息已细化到 `period` 和 `missing_models`，便于定位是哪一段、哪一腿预测未完成。

### 4.22d 96/24 链路分离设计（2026-08-16）
- **96 是主链路，24 是新增**。已隔离：
  - 目录：96 用 `outputs/ledger_96`+`outputs/runs_96`；24 用 `outputs/ledger`+`outputs/runs`（各 pipeline 按 res.label 自动选）。
  - NNLSGEF **resolution 感知**：24 点三段（1_8/9_16/17_24，24行/天）、96 点三段（1_32/33_64/65_96，96行/天）自动适配，同一 learner 代码。
  - `--resolution hourly|15min` 全局切换。
- **96 成果可迁移 24**：NNLS 权重学习器天然支持 24 点（同代码），`--weight-granularity` 通用。
- **注意**：24 点账本（outputs/ledger）当前不完整（2026-01-02~01-24 缺大部分天），30 天窗权重学习暂无法跑通——**需数据积累**。代码层已验证 NNLSConfig 对 hourly 正确输出三段。

### 4.22 ⚠️ 服务器回测环境（2026-08-16）
- 智川云 `sc01-ssh.gpuhome.cc:30486` 当前 **SSH 连接被拒**（可能关机/迁移）。
- 权重学习数据就绪：`outputs/ledger_96`（prediction 2025-12-01~2026-07-20 / actual 到 07-18，232 天）→ 可本机做权重学习 + 回测准备。
- 全量 GPU 回测（TimeMixer/RT916 训练）需服务器；本机 CPU 只跑轻量验证。
- git push 本机代理坏：`git -c http.proxy= -c https.proxy= push origin main`。
- **现象**：perf_knobs 的 TF32 + cudnn.benchmark 看似打开，实际运行被 seed 函数废掉。
- **根因**：`utils/reproducibility.py:set_global_seed` 无条件 `cudnn.benchmark=False` + `set_float32_matmul_precision("highest")`（=关 TF32）；RT916 自己的 `core.py:set_seed` 更狠，无条件 `cudnn.deterministic=True` + `benchmark=False`（conv 重模型最伤）。TimeMixer/RT916 训练前都会调用。
- **教训**：改 perf_knobs 必须同时查 seed 路径；想恢复加速应给 set_global_seed 加 `OPTIM_*` 开关或去掉 `"highest"`/deterministic 硬编码。
- **其他已核实**：`torch.fft.rfft` 不支持 BF16（RT916 每前向 2 处 FP32 往返）；TimeMixer 短程预测（pred_len 8/32）论文官方建议 scales=1 非 3（PDM FLOPs 省 43%）；两模型均未用 torch.compile，且训练循环每步 `loss.item()` 有同步开销。详见 `docs/TimeMixer_RT916_训练耗时热点调研报告.md`。

---

## 5. 项目目录地图（防踩乱）

- `outputs/` = 正式管道产物（ledger/runs/ledger_96/runs_96/platform_review/data_sync*）
- `outputs/crawl/` = 爬虫运行产物（原 `output/`，含日志、prediction_96、config_backup、验证码）
- `outputs/prediction_results/` = 预测结果表归档
- `scripts/sync/` = 数据同步/合并/回填脚本（sync_data、sync_data_96_core、build_96_full_table、backfill_*）
- `scripts/tests/` = 回归/验证测试脚本（check_*.py、verify_*.py）
- `scripts/crawler/` = 爬虫子模块（crawl.py、run_crawler、auto_fill_96、run_full 等）
- `dist/crawler/` = 甲方交付包（当前使用：crawl_96_local.exe + config.example.json + README_甲方部署.txt）
- `dist/archived_crawlers/` = 旧爬虫 exe 归档（auto_fill_96/run_full/auto_crawler_v2/backfill_*，git 忽略）
- `dist/audit/` = db_audit 审计工具；`dist/build_artifacts/` = 构建中间产物（build/venv_build/pyi_tmp）
- `dist/agent_artifacts/` = agent 遗留归档（旧 runs/HAR/调试产物）
- **dist/ 整体 gitignore**（.gitignore `dist/` + `*.exe` + `*.spec`），exe 不进仓库
- `_archive/` = 遗留代码（保留追溯，不参与生产）
- `data/remote_96/` = 96点本地镜像（parquet/raw/metadata）；`data/shandong_pmos_96_full_v2.xlsx` = 合并宽表
- `docs/` 权威文档：`PROJECT_LAYOUT.md`、`PLAN_24_AND_96_POINT_FORECASTING_ARCHITECTURE.md`、`24_VS_96_FEATURE_COMPARISON.md`、`96_DEPLOYMENT_GUIDE.md`

### 产出纪律（2026-08-14 立规）

- **新产物禁止散落根目录 / outputs 根**。按类型归位：
  - 管道产物 → `outputs/<约定子目录>`；预测结果表 → `outputs/prediction_results/`
  - 爬虫/调试日志 → `outputs/crawl/`；打包产物 → `dist/<分类>/`
  - agent 遗留 → `dist/agent_artifacts/`
- **移动脚本必须同步改 import/路径/README/docs/workflows**，并跑回归验证，确保全链路畅通。
- 改脚本路径时留意 `Path(__file__).resolve().parents[n]` 层级：`scripts/tests/` 下用 `parents[2]`。

---

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
