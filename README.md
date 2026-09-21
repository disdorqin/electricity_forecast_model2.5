# Electricity Forecast Delivery Pipeline v2.5

山东电力现货价格预测交付链路：**7 模型预测 + Ledger 自适应动态权重融合 + 最终交付校验**；24 点 legacy 兼容链仍保留 Realtime 极端价格分类校正。

24 点 legacy compatibility 已完成 2026-07-03 正式陪跑验收：五阶段全部 `complete`，`postflight=PASS`，`delivery_status=NORMAL`，`exit_code=0`，`fallback_used=false`，最终 `submission_ready.csv` 为 24 行、0 NaN；这不是 formal 96 生产验收。

> **96 点（15min）服务器部署**：Codex/操作员先按 [docs/SERVER_96_STANDARD_SOP.md](docs/SERVER_96_STANDARD_SOP.md) 顺序执行；通用运行规则见 [docs/RUNBOOK.md](docs/RUNBOOK.md)，更详细的 release/state/异常处理见 [docs/SERVER_96_DEPLOYMENT_BACKFILL.md](docs/SERVER_96_DEPLOYMENT_BACKFILL.md)。
> 96 点调试过程与 7 个兼容 bug 记录见 [docs/archive/historical/96_POINT_DEBUG_LOG_20260803.md](docs/archive/historical/96_POINT_DEBUG_LOG_20260803.md)。
> 工程复现、质量门和回滚规则见 [docs/PROJECT_GOVERNANCE.md](docs/PROJECT_GOVERNANCE.md)。

---

## 1. 正式链路

本项目保留两套正式可运行链路：24 点用于小时级 legacy/兼容交付，96 点用于当前
15 分钟级正式生产。两套链路共用账本、权重融合和交付审计思想，但输入粒度、模型
池和最终输出目录不同，不应混用。

### 1.1 24 点（小时级 legacy/兼容链路）

24 点链路面向小时级山东电力现货价格，保留五阶段流程和 Realtime 极端价格分类器：

```text
ledger_predict → ledger_weight → ledger_fuse
→ ledger_classifier → final_outputs → postflight
```

它使用日前/实时模型预测 24 个小时，账本从历史完整日中自适应选择训练样本，按
task、时段和模型学习融合权重；分类器只属于 24 点 legacy/兼容链路。正式结果写入
`outputs/24/runs/YYYY-MM-DD/final/submission_ready.csv`，验收要求为 24 行、无数值
NaN、manifest 完整且 `delivery_status=NORMAL`。

### 1.2 96 点（15 分钟级正式生产链路）

96 点是当前正式生产链路，每天覆盖 96 个 15 分钟槽位，固定模型池为：

- 日前：`lightgbm`、`timesfm`、`timemixer`
- 实时：`timesfm`、`sgdfnet`、`timemixer`、`rt916`

正式流程为：

```text
DB sync → immutable D/T snapshot → FeatureView
→ ledger_predict → ledger_weight → ledger_fuse
→ final_outputs → postflight
```

96 点正式生产不执行 ExtremePriceClf；实时最终结果直接使用未校正的融合结果。正式
入口会先同步数据库，再依据目标日路由 `LIVE_DYNAMIC`、已保存的 LIVE snapshot 回放
或 `HISTORICAL_PROXY_V1`，三条路径统一经过 FeatureView 和同一套后半链路。

正式约束固定为 `split_process`、CPU=2/GPU=1、`smape_reg/SLSQP`、30 日窗口、
lag2、最多回看 90 天、period 三段权重、prune threshold=0.05、RT916 stride=24，
并保留 SGDFNet 的决策日前 DA 96 点 anchor。生产状态位于
`outputs/96/{ledger,runs,cache,runtime,sync}`，最终交付为：

```text
outputs/96/runs/YYYY-MM-DD/final/submission_ready.csv
```

标准入口只有：

```text
python main.py --96 YYYY-MM-DD
python main.py --96 YYYY-MM-DD --predict both|dayahead|realtime
python main.py --96 YYYY-MM-DD --finish
```

96 点正式放行必须同时满足 DA3/RT4 每腿 96 点、同一 snapshot/protocol、权重和融合
门控通过、postflight PASS、artifact audit PASS、`fallback=false`，并且当前生产模型
结构、模型池、信息边界和资源配置不得被运行参数覆盖。

最终交付文件：

```text
24 legacy: outputs/24/runs/YYYY-MM-DD/final/submission_ready.csv
formal 96: outputs/96/runs/YYYY-MM-DD/final/submission_ready.csv
```

24 legacy 标准列：

```text
business_day, ds, hour_business, period, dayahead_price, realtime_price
```

---

## 2. 回测与融合结果

本节汇总当前仓库中可复现的 24 点与 formal96 结果。指标使用项目的 floor50
SMAPE 口径，`accuracy = 1 - SMAPE`；96 点按 15 分钟槽位统计，系统级指标按日前
与实时融合结果联合计算。

### 2.1 账本与权重学习口径

`ledger_weight` 对日前和实时都从目标日 `D-2` 向前选择最近 30 个完整日，最多回看
90 个日历日；不完整日跳过，不足时 fail-closed，不使用等权或历史降级。当前 formal96
模型池为：日前 `lightgbm/timesfm/timemixer`，实时 `timesfm/sgdfnet/timemixer/rt916`。

完整日必须具备全模型池的完整预测槽位、无 NaN 的实际账本，并满足当前信息边界。
`smape_reg/SLSQP` 学习 period 三段权重，低于 0.05 的权重按门控规则审计并从该段
融合中排除。

### 2.2 24 点（小时级）回测结果

数据来自 `outputs/platform_review/` 的 AI 电力交易平台复盘集，覆盖
**2026-01-01～2026-08-06，共 218 天**；其中平台 2.0 为当前展示的融合结果。

| 任务 | 模型 | SMAPE 总 | 准确率总 | MAE 总 | MSE 总 | R2 总 |
|---|---|---:|---:|---:|---:|---:|
| 日前 | 平台 1.0 | 17.14% | 82.86% | 53.6 | 5308 | 0.859 |
| 日前 | 平台 2.0（融合） | **16.07%** | **83.93%** | **51.2** | **4716** | **0.875** |
| 实时 | 平台 1.0 | 29.85% | 70.15% | 97.7 | 21365 | 0.575 |
| 实时 | 平台 2.0（融合） | **26.53%** | **73.47%** | **86.5** | **18544** | **0.631** |

24 点融合系统改良套利为 **13.16 元/MWh**，基础套利为 **11.74 元/MWh**；该
结果属于平台复盘数据，不与 formal96 的 15 分钟结果混合比较。

### 2.3 96 点（15 分钟级）正式融合结果

96 点模型/实际账本从 **2025-12-18** 开始存在，formal96 严格融合从
**2026-01-18～2026-09-20，共 246 天**开始统计，共计日前、实时各 **23616 个
融合样本**（246 × 96）。2025-12-18～2026-01-17 是严格 30 日权重学习预热期，
不使用降级融合；2026-09-21 为实时实际尚未闭合的 live partial 日，未纳入本次闭合
报告。

| 日前模型 | SMAPE 总 | 准确率总 | MAE 总 |
|---|---:|---:|---:|
| lightgbm | 25.50% | 74.50% | 103.9429 |
| timemixer | 28.08% | 71.92% | 111.6950 |
| timesfm | 25.14% | 74.86% | 104.8012 |
| **融合模型** | **21.73%** | **78.27%** | **95.8673** |

| 实时模型 | SMAPE 总 | 准确率总 | MAE 总 |
|---|---:|---:|---:|
| rt916 | 38.13% | 61.87% | 132.7485 |
| sgdfnet | 23.88% | 76.12% | 84.3823 |
| timemixer | 34.26% | 65.74% | 116.8675 |
| timesfm | 30.12% | 69.88% | 99.9103 |
| **融合模型** | **23.17%** | **76.83%** | **82.8348** |

系统级融合结果：

| 样本数 | SCR | 基础套利·度电 | 改良套利·度电 |
|---:|---:|---:|---:|
| 23616 | 43.77% | 25.2341 | **56.4088** |

完整 CSV 报告为 `outputs/metrics_96_report.csv`；逐日融合文件位于
`outputs/96/runs/YYYY-MM-DD/{dayahead,realtime}/fuse/`，包括
`fused_predictions.csv`、`fused_debug.csv` 和 `model_quality_gate.csv`。

> **正式交付记录**：2026-07-03 单日 NORMAL 验收（`ledger_predict → … → final_outputs` 五阶段 complete，`postflight=PASS`，`exit_code=0`，`fallback_used=false`，`submission_ready.csv` 24 行 0 NaN）归档见 `docs/archive/historical/ACCEPTANCE_REPORT.md`。

---

## 3. 快速开始

### 3.1 安装环境

```bash
conda create -n epf-2 python=3.11 -y
conda activate epf-2
pip install -r requirements.txt
```

Windows + CUDA 已验证。GPU 模型建议保持串行，避免 OOM；TimeMixer 在当前
Torch/CUDA 基线不支持严格 CUDA 确定性，因此 GPU 交付不要传
`--deterministic`，严格复现实验请切 CPU。

### 3.2 准备数据

默认输入：

```text
data/24/canonical/shandong_pmos_hourly.xlsx
```

必需字段：

```text
时刻 / ds / 时间
日前电价
实时电价
```

自定义路径：

```bash
--data-path path/to/shandong_pmos_hourly.xlsx
```

### 3.3 24点（小时级）数据同步

> 本项目跑通两套粒度：**24 点（小时级）** 与 **96 点（15分钟级）**，数据同步命令不同，分别见 §3.3 与 §3.4。

推荐两步式，便于区分数据问题和模型问题：

```bash
python main.py --pipeline sync_dataset --sync-source auto --force-sync --require-fresh-data
python main.py YYYY-MM-DD --data-path data/24/canonical/shandong_pmos_hourly.xlsx
```

也可以一条命令：

```bash
python main.py YYYY-MM-DD --sync-data-before-run --require-fresh-data
```

### 3.4 96点（15分钟级）数据同步

同步15分钟粒度数据（96点/日），用于更精细的电力市场分析。推荐走统一 CLI：

```bash
# 全量同步（生产唯一远端源 epf_pmos_96_full）
python main.py --pipeline sync_dataset --resolution 15min --sync-source db --sync-mode full --force-sync

# 增量同步（只回拉最近7天，但本地仍保持完整历史）
python main.py --pipeline sync_dataset --resolution 15min --sync-source db --sync-mode incremental
```

96 点同步不再依赖旧的 `epf_market_data_96 + epf_unit_data_96` 双表拼接。生产模型层长期只维护一份统一模型仓：

```text
data/96/authoritative/pmos_96_全量.csv
  └─ 数据库业务字段忠实镜像，可包含最新 partial / forecast-only 日

data/96/model_input/shandong_pmos_96_model_input_full.parquet
  └─ 唯一持久模型仓：闭合历史 + partial/forecast-only tail
```

闭合历史由 `model_input_full.parquet` 上的逻辑筛选获得，不再每日物化第二份 clean parquet；旧 clean 文件仅保留作历史兼容。

完整 DB 镜像另存
`data/96/remote/parquet/epf_pmos_96_full.parquet`。历史 `核电预测` 的缺失只允许从
24 点 canonical **forecast** 做已验证 fallback；actual 和价格目标绝不跨源填充。

运行 96 点 `ledger_predict/ledger_full` 时无需手工生成 masked 文件：formal runner 先同步 DB，按 invocation 持久化 `runs/<date>/snapshot/attempt_<id>/`，再由 FeatureViewBuilder 生成共享 transient parquet。每次 invocation 使用一个 `runtime/attempt_<date>_<attempt>/` sandbox；NORMAL 后 transient 目录删除，snapshot/provenance 保留。Dynamic-v1 不按固定小时再次裁剪 RT；D 日 DA/RT/actual truth 在 FeatureView 中全部 mask。

**定时任务说明**
- 爬虫负责写入 `epf_pmos_96_full`，其中完整 D+1 forecast 可提前写 forecast-only 行；
- `sync_dataset --resolution 15min` 同步 DB，并以最近 overlap 日增量刷新唯一 `model_input_full.parquet`；
- 同步报告输出至 `outputs/96/sync/`。

---

### 3.5 最小运行命令

24 点链路通常需要先准备小时级数据，再按日期运行：

```bash
python main.py --pipeline sync_dataset --sync-source auto --force-sync --require-fresh-data
python main.py YYYY-MM-DD --data-path data/24/canonical/shandong_pmos_hourly.xlsx
```

96 点正式生产只需要以下三个入口：

```bash
# 单日：自动 DB sync、预测、学习权重、融合并交付
python main.py --96 YYYY-MM-DD

# 已有目标日 prediction provenance 时，仅完成后半链路
python main.py --96 YYYY-MM-DD --finish

# 已闭合区间逐日 resume；END 使用批次开始时 DB sync 得到的 latest_closed_day
python main.py --96 --start START --end END --require-target-actual --skip-existing-final
```

---

## 4. 正式复现与生产运行

> **核心机制**：融合权重学习器（`ledger_weight`）只从 persistent `ledger` 读取最近 30 个完整历史日；每日 `ledger_predict` 会自动把当日 DA3/RT4 prediction 和可得 actual 追加到 ledger。因此 `outputs/96/ledger/` 本身就是可随服务器迁移、会每日增长的生产状态。
> 新服务器若已有历史 prediction/actual ledger，不必重新暖机30天，但 formal96 禁止裸复制 legacy/FeatureStore 目录：必须先走审计式 warm-start migration，再进入正式链。

| 运行方式 | 前提 | 干什么 |
|---|---|---|
| **正式陪跑（24 legacy）** | 没有任何预测结果 | 从 7 模型预测 `ledger_predict` 开始，跑完整五阶段，边跑边积累账本 |
| **正式 façade（96）** | `outputs/96/ledger` 已有最近30个完整历史日 | 使用 `--96 DATE` 一条命令跑四阶段；历史不足时在模型前 fail-closed |
| **96 新服务器 cold-start** | 已有旧服务器 prediction/actual ledger | 先用 `bootstrap_96_production_ledger.py` 审计迁移30日，再运行 `--96 DATE` |
| **复现** | 已有预测结果（如直接上传 30 天预测/账本文件） | 跳过预测，直接用 `ledger_weight` 学习权重并出结果 |

### 4.1 正式生产运行（24 legacy 五阶段；96 使用四阶段 façade）

24 点 legacy 完整五阶段：

```text
ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs
```

#### 4.1.1 24 点（hourly）正式陪跑

Linux / macOS：

```bash
python main.py 2026-07-03 \
  --data-path data/shandong_pmos_hourly_0702.xlsx \
  --ledger-root outputs/ledger \
  --weight-max-lookback-days 180 \
  --max-cpu-workers 2 \
  --max-gpu-workers 1 \
  --seed 42 \
  --deterministic
```

Windows PowerShell：

```powershell
python main.py 2026-07-03 `
  --data-path data/shandong_pmos_hourly_0702.xlsx `
  --ledger-root outputs/ledger `
  --weight-max-lookback-days 180 `
  --max-cpu-workers 2 `
  --max-gpu-workers 1 `
  --seed 42 `
  --deterministic
```

成功标准（24 点）：

```text
delivery_status = NORMAL
exit_code = 0
postflight = PASS
final/submission_ready.csv = 24 rows, 0 NaN
fallback_used = false
```

#### 4.1.2 96 点（15min）正式陪跑

推荐 façade（不再进入 ExtremePriceClf）：

```powershell
python main.py --96 YYYY-MM-DD
python main.py --96 YYYY-MM-DD --predict both
python main.py --96 YYYY-MM-DD --predict dayahead
python main.py --96 YYYY-MM-DD --predict realtime
python main.py --96 YYYY-MM-DD --finish
```

上述正式链路固定 CPU DAG=2、GPU serial=1、RT916 stride=24、
`smape_reg/SLSQP`、30 日窗口/max lookback=90、period 三段和 prune=0.05；
manifest 会记录 `classifier_policy=disabled_by_production_policy`。formal96
权重学习仍只使用完整闭合历史；Dynamic-v1 serving 的可见性由 snapshot/
FeatureView 决定，不按固定小时再次裁剪 RT。完整 `--96 DATE` 会先强制 DB
sync、生成 D/T snapshot，再做30日 readiness；不足时显式 fail-closed。
已有服务器历史可以作为**有审计的 operational warm-start** 迁入正式 ledger：迁移必须逐日验证 DA3/RT4/actual96、槽位、NaN 和原始 provenance；不得把旧历史改写成当前 Dynamic-v1 协议或严格 forecast-vintage 的效果证据。

**Dynamic-v1 生产验收（2026-09-20）：** 已用标准入口 `python main.py --96 2026-09-20` 完成真实 DB-backed 验收。该日 decision day 的 RT 为 partial，真实覆盖了动态 RT→DA 路由场景；7 个正式模型腿均已在真实模型计算中产出 96 点，SGDFNet 记录 `anchor_source_day=2026-09-19 / rows=96 / fallback_used=false`。最终标准入口再次运行得到 `delivery_status=NORMAL`、`exit_code=0`、postflight PASS、next-day adaptive readiness PASS、fallback=false；严格 cache rerun 只复用同 snapshot/protocol 的已验模型输出。生产机械验收可用 `scripts/server/audit_96_artifacts.py`，live prediction 默认允许目标日 actual partial；历史结算验收才追加 `--require-target-actual`。

**当前三态 Snapshot 路由（2026-09-20 收口）：** `python main.py --96 T` 是唯一正式入口。除 `--finish` 外先强制 DB sync，并用数据库 `latest_closed_day` 判定运行类型：历史日若已有成功 manifest 绑定的 canonical LIVE Snapshot，则直接走 `STORED_LIVE_SNAPSHOT_REPLAY`；历史日没有真实 Snapshot 时走 `HISTORICAL_PROXY_V1`，仅在 Snapshot 层把 D 日 final actual/RT 暴露到 p56，后段继续走现有 FeatureView fallback；当前正式目标走 `LIVE_DYNAMIC`，数据库当时可见多少就冻结多少。三条路之后完全共用同一个 FeatureViewBuilder、DA3/RT4、30日 learner、SLSQP 与 final。正式 LIVE 成功 Snapshot 长期保留，FeatureView/runtime scratch 在 NORMAL 后清理。

**Historical Proxy 与服务器历史接续（2026-09-21）：** `python main.py --96 2026-08-17` 已真实跑通 `HISTORICAL_PROXY_V1` / p56，7 个模型腿各96点，SGDFNet anchor=`2026-08-16` DA96、RT916 stride=24，learner 严格只使用到 T-2=`2026-08-15`，weight/fuse/final/postflight 全部 PASS。随后服务器完成 old-server 240 天 full-source ledger apply，并用正式 range 从 `2026-08-17` 接续到 `2026-09-19`；8/17 已有合法结果被 skip，8/18～9/19 全部 complete，逐日 artifact audit **34/34 PASS**。服务器验证基线为 Python 3.11.14 + Torch 2.6.0+cu124 + RTX 4090 48GB。历史 range 只在批次开始做一次 DB full sync；正式日仍单独运行 `python main.py --96 TARGET_DATE` 以获取预测时点的新鲜 Snapshot。

正式命令速查：

```bash
# 单日正式生产
python main.py --96 YYYY-MM-DD

# 历史闭合区间接续 / resume
python main.py --96 --start START --end END \
  --require-target-actual --skip-existing-final
```

`END` 必须取批次开始时 DB sync 得到的 `latest_closed_day`；range 内按日期串行执行完整 `Snapshot → FeatureView → DA3/RT4 → ledger → 30日 learner → SLSQP → fuse → final → postflight`，不是 prediction-only runner。服务器部署请先读 `docs/SERVER_96_STANDARD_SOP.md`。

推荐冷启动迁移：

```powershell
# 先 dry-run，只审计不写 production ledger
python scripts/server/bootstrap_96_production_ledger.py `
  --source-ledger "<old-server-ledger>" `
  --target-date YYYY-MM-DD `
  --days 30

# dry-run PASS 后再原子导入 outputs/96/ledger
python scripts/server/bootstrap_96_production_ledger.py `
  --source-ledger "<old-server-ledger>" `
  --target-date YYYY-MM-DD `
  --days 30 `
  --apply
```

迁移完成后 `bootstrap_manifest.json` 与四份 canonical ledger 一起保留。warm-start 与正式 learner 使用同一个 adaptive 规则：从 T-2 向前最多回看90个日历日，选择最近30个 DA3/RT4 + actual96 完整日；不要求最近30个日历日连续完整。T-1 prediction 仅在整池96槽且 cutoff 合法时作为可选 future state 一并迁入，缺失不会阻断30日 readiness。

**甲方最小部署（2026-09-20 已 clean-room 验收）：** 使用 `scripts/server/build_predictor_release.py --apply --output-dir <NEW_DIR>` 生成白名单 predictor（187 files，约0.866GiB，含静态模型，不含 data/outputs/crawler/experiments/tests/build/Agent/secrets），再用 `bootstrap_96_production_ledger.py` 迁 state，最后运行 `scripts/server/doctor_96_deployment.py --root <NEW_DIR> --strict-release --require-cuda --check-db --check-writable --target-date T`。真实全新 candidate 已从 DB full sync 后执行 `python main.py --96 2026-09-20`，7模型全部真实96点、Stage2~4 完成、`delivery=NORMAL`、postflight PASS、fallback=false；candidate 内 artifact audit 与 final doctor 均 PASS。TimesFM 必须解析到 candidate 自身 `models/timesFM`，strict doctor 已对此 fail-closed。

`--resolution 15min` 默认使用 `production` profile：`outputs/96/ledger` + `outputs/96/runs` + `outputs/96/cache`，临时输入位于 `outputs/96/runtime`；也可以通过 `--ledger-root` 和 `--runs-root` 显式覆盖：

```bash
# 默认直接读取唯一长期模型仓 + authoritative truth；无需再传 data-path
python scripts/server/run_96_prediction_backtest.py \
  --report-start 2026-08-15 --end 2026-09-15
```

服务器预测每天统一走 `DB sync -> immutable D/T snapshot -> FeatureView -> models`，
transient FeatureView 在任务结束删除，snapshot/provenance 保留。96 点 `split_process` 已使用
CPU DAG-ready queue（最多2 worker）与严格串行 GPU worker，并以 `(model, task/internal_node)`
区分 DA/RT；正式 façade 接线完成前旧 `--pipeline` 仍是 advanced compatibility。resume 只会跳过同时通过账本完整性、
actual 完整性和严格 as-of run manifest 协议审计的日期。先用
`--report-start 2026-08-16 --end 2026-08-16 --no-prewarm` 做单日耗时和 96 点完整性
smoke，再启动完整区间。预测账本写入 `outputs/96/ledger`，逐日结果和范围 manifest
写入 `outputs/96/runs`；无需再使用 `outputs/96/feature_store` 作为正式服务器根。
范围 manifest 会显式标记 `forecast_vintage=UNVERIFIED_LEGACY_VINTAGE`：当前 latest-state
历史可以用于生产链路验收，但不能被表述为已证明的 D-1 原始 forecast 版本回放。

### Legacy / FeatureStore 兼容链路

当前新生产默认且**实际只维护** `outputs/96/{ledger,runs,cache,runtime,sync}`。24 点仍沿用已验证的 `outputs/ledger + outputs/runs`，暂不为了目录对称迁移。旧 `outputs/ledger_96` + `outputs/runs_96` 仅供 legacy/research；formal `--96` façade 已增加 fail-closed，显式把 `--ledger-root/--runs-root` 指向这些旧根会直接拒绝。原 `outputs/96/feature_store/` 已于 2026-09-19 全部迁出正式域：服务器原始回测包进入 `outputs/archive/server_backtest_96/original_server_prediction_20251218_20260814/`，其余 candidate/cache/smoke 残余进入 `outputs/archive/legacy_96/feature_store_residual_20260919/`。`feature_store` profile 仅作为显式兼容模式保留，主动选择时才会重新创建 candidate root：

```bash
python main.py 2026-01-01 \
  --resolution 15min \
  --output-profile feature_store \
  --feature-store-mode raw \
  --data-path data/96/model_input/<clean_96_model_input>.xlsx
```

上面的 `<clean_96_model_input>.xlsx` 只是占位符，不能替换成
`shandong_pmos_96_model_input.xlsx`：该历史文件已标记为
`historical-invalid-features`，服务器预测脚本会主动拒绝它。正式服务器运行请使用
`scripts/server/run_96_prediction_backtest.py`；其默认 actual source 为
`data/96/authoritative/pmos_96_全量.csv`。

服务器验收前的 retention 只允许 dry-run：
`python scripts/server/maintenance_96.py --report outputs/96/sync/retention_plan_preacceptance.json`。
当前 `--apply` 会硬拒绝，禁止在验收前自动删除历史 outputs。

历史 `outputs/96/feature_store/*` 已全部归档；当前磁盘上的 formal96 不包含该目录。显式 `feature_store` compatibility profile 仍可用于旧候选链复现，但不得承接新生产状态。正式模型池唯一来源为
`fusion/model_pool.py`：日前为 `lightgbm + timesfm + timemixer`，实时为
`timesfm + sgdfnet + timemixer + rt916`；LightGBM 实时入口不进入生产池。

96 点权威实际数据为 `data/96/authoritative/pmos_96_全量.csv`，只用于
`scripts/tests/check_96_vs_24_actual.py` 的交叉验证；价格和预测输入必须来自
`data/96/model_input/`，两者禁止混用。

多日预热 + 全链路（服务器推荐，放 tmux 里跑）：

```bash
bash scripts/auto_preheat_backtest.sh
# 阶段1: ledger_backfill 2025-12-01~12-31 预热，补足 30 天权重学习历史（~8h）
# 阶段2: 账本 ≥30 天后自动 ledger_full_range 2026-01-01 起逐日跑正式四阶段（24 legacy 才执行 classifier）
```

成功标准（96 点正式链路；不满足时明确 fail-closed，不退化冒充正式 96）：

```text
final/submission_ready.csv = 96 行（15min 粒度）
长表 288 行（dayahead）/ 384 行（realtime）→ 96 点正确
```

### 4.2 复现（已有预测结果，直接学权重）

融合权重学习器要学**前 30 天的预测结果**才能出权重。两条路二选一：

1. **正式陪跑**：没有任何预测结果 → 先按 §4.1 跑完整链路，边跑边积累账本；
2. **复现**：直接把 30 天的预测/账本文件上传、拷进 ledger → 跳过模型预测，直接学权重出结果。

复现命令如下（`ledger_predict` 缓存命中秒过，重点在学权重）。

#### 4.2.1 24 点（hourly）复现

从复现包拷贝 32 天账本（含预测 + 实际，见 `fixtures/repro_bundle/README.md`），跳过 `ledger_backfill`：

Linux / macOS：

```bash
mkdir -p outputs/ledger
cp -r fixtures/repro_bundle/ledger/* outputs/ledger/
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force outputs/ledger | Out-Null
Copy-Item fixtures/repro_bundle/ledger/* outputs/ledger -Recurse -Force
```

然后直接跑目标日（账本已有预测 → `ledger_predict` 命中缓存，重点是学权重）：

```bash
python main.py 2026-02-24 \
  --data-path data/24/canonical/shandong_pmos_hourly.xlsx \
  --ledger-root outputs/ledger \
  --weight-max-lookback-days 180
```

只想验证后半链路（7 模型已跑完、不重跑模型）的完整做法见 §4.5 副线 C。

#### 4.2.2 96 点（15min）复现

已有 96 点预测 CSV / runs 结果时，先用已有预测重建账本，再直接跑目标日学权重（不重跑模型）。以下命令仅用于 legacy/internal 96 replay（旧 `outputs/ledger_96` / `outputs/runs_96`）；formal 96 生产请使用 `python main.py --96 YYYY-MM-DD`：

```bash
# 用已有预测结果重建 prediction ledger
python scripts/sync/rebuild_prediction_ledger_96.py --runs-root outputs/runs_96 --ledger-root outputs/ledger_96

# 或从 output/prediction_96/*.csv 把预测种回 runs 缓存（可选）
python scripts/seed_96_ledger_cache.py --date 2026-07-16

# 直接跑目标日：predict 缓存命中，直接学权重
python main.py 2026-07-16 \
  --resolution 15min \
  --ledger-root outputs/ledger_96 \
  --runs-root outputs/runs_96 \
  --weight-max-lookback-days 180

# formal96 当前目标先强制 DB sync；历史目标复用 LIVE snapshot 或 p56 proxy；再由同一 FeatureView 生成共享
# FeatureView；只有调试/隔离实验才建议显式传 --data-path。
```

### 4.3 副线 A：简单跑 / 快速验收

用于快速确认代码、数据路径、ledger、权重融合有没有明显问题。适合演示、 smoke test、交付前最后检查。

推荐顺序：

```bash
python -m py_compile main.py cli/parser.py pipelines/ledger_weight.py pipelines/prediction_ledger.py pipelines/delivery_quality.py pipelines/ledger_classifier.py
python scripts/tests/check_adaptive_realtime_weight_days.py
python scripts/tests/check_delivery_stability.py
python scripts/tests/check_target_day_nan_regression.py
python scripts/tests/check_sync_dataset.py
```

然后跑单日 full chain：

```bash
python main.py 2026-07-03 \
  --data-path data/shandong_pmos_hourly_0702.xlsx \
  --ledger-root outputs/ledger \
  --weight-max-lookback-days 180
```

简单跑特点：

```text
目标：快速判断能不能跑通
输入：已有 data + 已有 ledger
输出：submission_ready.csv / run_manifest.json / delivery_report.md
不负责补齐长历史 ledger
不建议提交 outputs/runs 到 Git
```

### 4.4 副线 B：复杂全量跑 / 生产完整跑

用于更接近生产的完整流程：先同步数据，再补 ledger，再跑正式 full chain。

推荐流程：

```bash
# 1. 同步最新数据
python main.py --pipeline sync_dataset \
  --sync-source auto \
  --force-sync \
  --require-fresh-data

# 2. 回填历史 ledger，确保权重学习能选到更近的 30 个完整训练日
python main.py --pipeline ledger_backfill \
  --start 2026-06-03 \
  --end 2026-07-02 \
  --data-path data/shandong_pmos_hourly_0702.xlsx \
  --max-cpu-workers 2 \
  --max-gpu-workers 1 \
  --seed 42 \
  --deterministic \
  --force

# 3. 正式跑目标日
python main.py 2026-07-03 \
  --data-path data/shandong_pmos_hourly_0702.xlsx \
  --ledger-root outputs/ledger \
  --weight-max-lookback-days 180 \
  --max-cpu-workers 2 \
  --max-gpu-workers 1 \
  --seed 42 \
  --deterministic
```

复杂全量跑特点：

```text
目标：尽量贴近正式生产
输入：最新数据 + 尽可能完整的历史 ledger
重点：ledger_backfill 让权重学习使用更近的完整训练日
耗时：明显长于简单跑
适用：正式交付前、生产机部署、长区间回测
```

### 4.5 副线 C：已有预测结果，只验证后半链路

如果生产模型已经跑完，只想验证权重、融合、分类器、最终输出：

```powershell
$TARGET_DATE = "2026-07-03"
$LEDGER_ROOT = "outputs/ledger"
$RUNS_ROOT = "outputs/_final_chain_verify_20260703/runs"

Copy-Item -Recurse -Force "outputs/runs/2026-07-03" "$RUNS_ROOT/"

python main.py --pipeline ledger_weight --date $TARGET_DATE --ledger-root $LEDGER_ROOT --runs-root $RUNS_ROOT --weight-max-lookback-days 180
python main.py --pipeline ledger_fuse --date $TARGET_DATE --ledger-root $LEDGER_ROOT --runs-root $RUNS_ROOT
python main.py --pipeline ledger_classifier --date $TARGET_DATE --ledger-root $LEDGER_ROOT --runs-root $RUNS_ROOT
```

这个模式不重新跑生产模型，只验证：

```text
ledger_weight → ledger_fuse → ledger_classifier → final_outputs/postflight
```

### 4.6 历史副线 D：AI电力交易平台复盘数据获取（已归档）

我们现在已经有了**交易可视化平台**（AI电力交易平台 http://47.114.107.96/，账号 user/user123），
复盘模块有「电价预测复盘」，可以直接用爬虫程序把 日前/实时 电价 + 各模型预测价抓成数据集，不用再手搓 Excel。

> ⚠️ 注意：该平台是自建演示站，**与国网 PMOS 爬虫无关**，是独立数据源。

该工具已从当前爬虫目录移至 `scripts/crawler/archive/legacy/`，不属于 PMOS
96 点生产链路。历史数据仍保留在 `outputs/platform_review/`，如需追溯才运行：

命令行更新数据集：

```bash
# 更新到最新（自动：从数据集最早日期 ~ 今天）
python scripts/crawler/archive/legacy/platform_review_update.py

# 指定抓取区间
python scripts/crawler/archive/legacy/platform_review_update.py --start 2026-01-01 --end 2026-08-06
```

数据集落在 `outputs/platform_review/`（已放行 git 跟踪），字段说明与更完整用法见 §18。

---

## 5. 不推荐用于正式 NORMAL 的参数

下面参数只用于诊断或应急，不作为 NORMAL 交付依据：

```text
--allow-missing-models
--allow-equal-weight-fallback
--no-range-preflight
```

如果用了这些参数跑通，只能说明工程链路可继续，不代表正式 NORMAL。

---

## 6. Ledger 目录

默认 ledger 根目录按 profile 区分：

```text
24 legacy: outputs/ledger
96 formal production: outputs/96/ledger
```

也可指定：

```bash
--ledger-root <your_ledger_root>
```

核心文件：

| 类型 | 路径 |
|---|---|
| Dayahead prediction | `{ledger_root}/dayahead/prediction/prediction_ledger.parquet` |
| Dayahead actual | `{ledger_root}/dayahead/actual/actual_ledger.parquet` |
| Realtime prediction | `{ledger_root}/realtime/prediction/prediction_ledger.parquet` |
| Realtime actual | `{ledger_root}/realtime/actual/actual_ledger.parquet` |

权重学习只读取 ledger，不直接读取 `outputs/runs`。每日 `ledger_predict` 会把当日预测追加到 prediction ledger；actual ledger 会按可得实际值更新。formal 96 的长期状态根是 `outputs/96/ledger/`，它可以随服务器迁移；若历史不是由当前 formal96 直接生成，必须先经 `bootstrap_96_production_ledger.py` 审计式 warm-start 导入并保留 `bootstrap_manifest.json`。

---

## 7. 验证命令

基础回归：

```bash
python -m py_compile main.py cli/parser.py pipelines/ledger_weight.py pipelines/prediction_ledger.py pipelines/delivery_quality.py pipelines/ledger_classifier.py
python scripts/tests/check_adaptive_realtime_weight_days.py
python scripts/tests/check_delivery_stability.py
python scripts/tests/check_target_day_nan_regression.py
python scripts/tests/check_sync_dataset.py
```

期望：

```text
check_adaptive_realtime_weight_days.py = 40/40 PASS
check_delivery_stability.py = 29/29 PASS
check_target_day_nan_regression.py = 16/16 PASS
check_sync_dataset.py = 41/41 PASS
```

检查 adaptive training days：

```bash
python - <<'PY'
from pathlib import Path
from pipelines.ledger_weight import select_complete_training_days, DAYAHEAD_MODELS, REALTIME_MODELS
import json
for task, models in [('dayahead', DAYAHEAD_MODELS), ('realtime', REALTIME_MODELS)]:
    result = select_complete_training_days(
        task=task,
        target_date='2026-07-03',
        ledger_root=Path('outputs/ledger'),
        expected_models=models,
        required_days=30,
        max_lookback_days=180,
    )
    print(task)
    print(json.dumps({
        'status': result['status'],
        'selected_count': result['selected_count'],
        'latest_selected_day': result['selected_days'][0] if result['selected_days'] else None,
        'skipped_count': len(result['skipped_days']),
        'errors': result['errors'],
    }, ensure_ascii=False, indent=2))
PY
```

---

## 8. 如果需要补 ledger

如果 adaptive 在 lookback 范围内凑不够 30 个完整训练日，需要 backfill：

```bash
python main.py --pipeline ledger_backfill \
  --start 2026-06-03 \
  --end 2026-07-02 \
  --data-path data/shandong_pmos_hourly_0702.xlsx \
  --max-cpu-workers 2 \
  --max-gpu-workers 1 \
  --seed 42 \
  --deterministic \
  --force
```

若 `D-1` 当天 actual 不完整，adaptive 会自动跳过该日，并继续向前找完整训练日。

---

## 9. 输出文件

| 文件 | 说明 |
|---|---|
| `outputs/runs/YYYY-MM-DD/final/submission_ready.csv` | 最终交付文件 |
| `outputs/runs/YYYY-MM-DD/run_manifest.json` | 24 legacy 五阶段运行元信息（96 使用 `outputs/96/runs` 四阶段 manifest） |
| `outputs/runs/YYYY-MM-DD/delivery_report.md` | 交付报告 |
| `outputs/runs/YYYY-MM-DD/dayahead/weight/weights.csv` | Dayahead 融合权重 |
| `outputs/runs/YYYY-MM-DD/realtime/weight/weights.csv` | Realtime 融合权重 |
| `outputs/runs/YYYY-MM-DD/{task}/fuse/fused_predictions.csv` | 融合结果 |
| `outputs/runs/YYYY-MM-DD/realtime/final/realtime_final_predictions_corrected.csv` | 分类器校正后 realtime |

---

## 10. Delivery Status

| delivery_status | exit code | 含义 |
|---|---:|---|
| NORMAL | 0 | 对应分辨率的正式阶段正常完成，postflight PASS |
| DEGRADED_DELIVERED | 2 | 正常链路失败，但 emergency fallback 生成可交付文件 |
| FAILED_NO_DELIVERY | 1 | 正常链路和 fallback 均失败，无可用交付 |

正式验收优先使用 NORMAL。24 legacy 若使用 DEGRADED，必须说明 fallback 原因和后续修复计划；formal 96 合同失败必须 `FAILED_NO_DELIVERY`，禁止 emergency/degraded fallback。

---

## 11. Troubleshooting

| 问题 | 判断 | 处理 |
|---|---|---|
| LightGBM `NoneType` | 旧版本未兼容目标日 NaN | 拉取最新 main |
| SGDFNet 24 行 NaN | 旧版本 `da_anchor` 为 NaN | 拉取最新 main |
| `ledger_weight` 凑不够 30 天 | ledger 不足或 lookback 太短 | 补 ledger / backfill / 提高 `--weight-max-lookback-days` |
| `UnicodeDecodeError: gbk` | Windows 默认编码读 JSON | 拉取最新 main，JSON 读写已显式 UTF-8 |
| `submission_ready.csv` 有 NaN | fuse/final 缺某个 task | 查 `delivery_report.md` 与 `run_manifest.json` |
| exit code 2 | fallback 交付 | 查看 `fallback_report.md/json`，修复后 `--force` 重跑 |
| exit code 1 | 无交付 | 查看 `run_manifest.json.errors` |
| `DATABASE_SYNC_FAILED` / row-count mismatch | 远程表在读取期间发生并发写入，或旧本地镜像与远程不一致 | formal96 先使用默认增量同步；需要审计时显式运行 15 分钟级 full sync，不要手工只补某两天 |
| formal96 没有融合样本 | 目标日尚未满足 lag2 + 30 个完整训练日 | 这是 fail-closed 预热期；等待账本满足 readiness，不使用等权或旧模型降级 |
| RT 最终结果看起来是旧文件 | 旧 attempt 留下了 stale final | 重新执行 `python main.py --96 DATE --finish`；formal96 会以当前 `realtime/fuse/fused_predictions.csv` 覆盖 RT final |
| TimesFM 模型缺失或路径错误 | 部署根没有自带 `models/timesFM`，或被开发机路径覆盖 | 先运行 `doctor_96_deployment.py --strict-release`，确认 TimesFM 从 candidate 自身解析 |
| live target actual 不满 96 点 | 当前日实际数据尚未闭合 | 单日正式生产允许 partial actual；历史闭合验收再追加 `--require-target-actual` |
| `--finish` 报 prediction provenance 不完整 | 该日期没有合法 Stage1 snapshot/manifest | 不要强行复用旧 runs；先用标准入口重新运行 `python main.py --96 DATE` |

---

## 12. Git 安全

不要提交：

```text
data/
models/
outputs/runs/
outputs/_*/
```

检查：

```bash
git status --short
git ls-files data models outputs/runs outputs/_*
```

`outputs/runs/YYYY-MM-DD/final/submission_ready.csv`、`run_manifest.json`、`delivery_report.md` 可以作为交付附件单独发送，不建议作为代码提交。

---

## 13. 最近关键修复

| commit | 内容 |
|---|---|
| `bbe9b8c` | 修复 LightGBM target-day NaN / SGDFNet target-day NaN |
| `3cd629e` | Realtime adaptive complete training days |
| `55465be` | 严格 hour_business `{1..24}` + position-based `age_days` |
| `0214aaf` | 修复 classifier manifest Windows UTF-8 问题 |
| `40965eb` | README 交付版 |
| `f379a4c` | Dayahead 也改为 adaptive complete training days |
| 最新 main | 恢复并保留简单跑 / 复杂全量跑 / 后半链路验证三条副线说明 |

---

## 14. 一句话结论

模型预测流程、DA/RT 自适应权重学习、融合、分类器、最终输出与 postflight 均已通过 2026-07-03 正式陪跑验收。完整 NORMAL 交付的核心前提是：**ledger 中能在 lookback 范围内为 Dayahead 和 Realtime 各自找到最近 30 个完整训练日。**

---

## 15. 爬虫 & 数据同步 FAQ

### 15.1 PMOS 连不上 / 爬虫报错

**现象：** 运行爬虫或回填脚本时出现：
- `Remote end closed connection without response`
- `HTTP 911`
- `CSRF token not found`
- `Expecting value: line 1 column 1 (char 0)`

**原因：** Cookie 过期。国网 PMOS 的 Cookie 有效期通常为数天至一两周，过期后服务器直接断开连接。

**解决：**
1. 在办公电脑上打开浏览器，访问 `https://pmos.sd.sgcc.com.cn:18080/trade/`
2. 按 F12 → Network（网络）标签 → 刷新页面
3. 点任意请求 → 找到 Request Headers 中的 `Cookie` 字段
4. 复制整段 Cookie 值
5. 更新 `config.json` 中的 `"cookie"` 字段
6. 重新运行爬虫

当前 96 点企业机版本 `dist/crawler/crawl_96_auto_v6.exe` 支持自动打开浏览器认证：
配置 `auth_mode=browser` 后，程序通过 CDP 读取登录 Cookie，原子写回同目录
`config.json`，再继续调用原有 96 点数据接口；滑块模板识别失败时可在弹出的浏览器中手工完成。

### 15.2 电脑关机 / 休眠 → 当天没数据

**现象：** 某天数据缺失，GitHub Actions 发出告警 Issue。

**原因：** 定时任务依赖办公电脑在 08:00 处于开机或睡眠状态。

**解决：**
1. 在任务计划程序（Task Scheduler）中找到"PMOS数据爬虫"
2. 右键 → 属性 → **条件（Conditions）** 选项卡
3. 勾选 **"唤醒计算机运行此任务"**（Wake the computer to run this task）
4. 电脑保持**睡眠（Sleep）**状态即可，不要完全关机

**补充：** 睡眠状态耗电极低（≈ 台式机 3-5W），可以长期不关。

### 15.3 GitHub Actions 检查失败

**现象：** 收到 GitHub Issue 告警 "[数据告警] 数据检查异常"

**系统已自动执行：**
- GitHub Actions 每天 BJT 08:30 运行 `scripts/tests/check_data_freshness.py`
- 检查项：昨日数据完整性、近7天连续性、数据新鲜度
- 发现异常 → 自动创建 Issue → 邮件通知仓库所有者

**收到告警后排查步骤：**
1. 检查办公电脑是否开机（远程桌面或请同事查看）
2. 查看爬虫日志（办公电脑上 `output/crawler.log` 或 `output/crawler_scheduled.log`）
3. 如果是 Cookie 过期 → 按 17.1 更新 Cookie 后重新运行爬虫
4. 数据恢复后手动触发同步：`python main.py --pipeline sync_dataset --resolution 15min --sync-mode incremental`

**提示：** 建议在 GitHub Settings → Notifications 中开启 Issues 邮件通知，确保第一时间收到告警。

### 15.4 backfill_unit_96.exe 闪退 / 无反应

**现象：** 双击 exe 后窗口一闪而过，没有输出。

**原因：** exe 依赖 `config.json` 和 `.env`，必须放在同一目录下。

**解决：**
1. 确认目录下有这 3 个文件：
   ```
   D:\爬虫电网\
   ├── backfill_unit_96.exe
   ├── config.json      （Cookie + unit_id）
   └── .env             （数据库连接信息）
   ```
2. 建议用命令行运行（窗口不会自动关闭）：
   ```bash
   cd D:\爬虫电网
   backfill_unit_96.exe --dry-run
   ```

### 15.5 本地同步 vs 数据库数据不一致

**现象：** 本地 96 点数据（`data/96/remote/` 镜像或 `shandong_pmos_96_full.xlsx`）与数据库不一致。

**原因：** 本地镜像同步和爬虫交付表是两条不同用途的链路。爬虫先把
`epf_pmos_96_full` 写入云端 MySQL；本地镜像同步只读取源表，不会反向写库。

**解决：**
```bash
# 增量同步（重拉最近7天并合并到本地镜像）
python main.py --pipeline sync_dataset --resolution 15min --sync-mode incremental

# 全量覆盖本地镜像（从数据库重新拉取）
python main.py --pipeline sync_dataset --resolution 15min --sync-source db

# 从权威 CSV 回灌/校准云端爬虫交付表（业务列与 CSV 同名）
python scripts/crawler/archive/migrations/migrate_authoritative_96_to_full.py --dry-run
python scripts/crawler/archive/migrations/migrate_authoritative_96_to_full.py
```

```bash
# 从源表重新生成本地模型输入
python scripts/sync/build_96_full_table.py
```

`build_96_full_table.py` 只用于构建本地模型输入，不是
`epf_pmos_96_full` 的写入入口。该脚本会拒绝已知的
`actual_* == fcast_*` 历史污染镜像，不能用污染数据训练模型；清洁模型输入应使用
`scripts/sync/build_96_model_input_from_authoritative.py`。公司电脑使用
`dist/crawler/crawl_96_auto_v6.exe`，它会把当天通过审计的数据直接写入
`epf_pmos_96_full`。

### 15.6 云端数据库连接失败

**现象：** 脚本报错 `Database env vars are incomplete` 或 `Can't connect to MySQL server`

**原因：** `.env` 文件缺失、格式错误，或数据库服务不可用。

**检查：**
```bash
# 确认 .env 文件存在且格式正确
cat .env
# 期望输出（不要有引号）：
# DB_HOST=8.136.218.112
# DB=ai_epf_platform
# DB_USER=ai_epf_platform
# DB_PWD=x:!9puh3-wu%
# DB_PORT=3306
```

**注意：** `.env` 中的值**不要加引号**，否则会被当作值的一部分。

---

## 16. AI电力交易平台 · 电价预测复盘数据更新

> ⚠️ **平台区分：** 本节面向的是「AI电力交易平台」自建演示站
> **http://47.114.107.96/**（账号 `user` / `user123`），
> **与国网山东电力交易平台 PMOS（`pmos.sd.sgcc.com.cn`）是完全不同的两个系统**。
> 第 17 节 FAQ 里讲的 Cookie / 数据库 / 定时任务全部针对国网 PMOS；
> 本节的工具不走 Cookie、不碰数据库，是独立的第二数据源（平台自带的「电价预测复盘」模块）。

### 16.1 数据集位置

稳定路径 `outputs/platform_review/`（已放行 git 跟踪，可直接提交推送）：

| 文件 | 内容 |
|---|---|
| `电价预测复盘.xlsx` | 平台原始导出（详细数据 + 统计报告 两个 sheet） |
| `电价预测复盘_详细数据.csv` | 逐小时：实时电价 / 日前电价 + 1.0/2.0 模型预测价 |
| `电价预测复盘_统计报告.csv` | 全量 + 分月综合准确率统计 |

当前覆盖：**2026-01-01 ~ 2026-08-06**（218 天 × 24 小时 = 5232 行）。

### 16.2 更新命令

```bash
# 更新到最新（自动：从数据集最早日期 ~ 今天，幂等）
python scripts/crawler/archive/legacy/platform_review_update.py

# 指定抓取区间（明细按 time 合并去重，区间外旧数据保留）
python scripts/crawler/archive/legacy/platform_review_update.py --start 2026-01-01 --end 2026-08-06

# 只指定结束日期（从数据集最早日开始）
python scripts/crawler/archive/legacy/platform_review_update.py --end 2026-08-06

# 换账号 / 换输出目录
python scripts/crawler/archive/legacy/platform_review_update.py --user user --password user123 --out outputs/platform_review
```

说明：
- 区间内数据重新拉取并覆盖，区间外历史数据自动保留（按 `time` 去重合并）。
- 「统计报告」仅在本次区间能覆盖现有全部数据时才刷新（避免聚合口径变小）。
- 平台最后一天的实时电价可能晚发布（如 08-06 15–24 点实时价），隔天重跑一次即可补上。
