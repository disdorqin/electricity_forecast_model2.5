# Electricity Forecast Delivery Pipeline v2.5

山东电力现货价格预测交付链路：**7 模型预测 + Ledger 自适应动态权重融合 + 最终交付校验**；24 点 legacy 兼容链仍保留 Realtime 极端价格分类校正。

24 点 legacy compatibility 已完成 2026-07-03 正式陪跑验收：五阶段全部 `complete`，`postflight=PASS`，`delivery_status=NORMAL`，`exit_code=0`，`fallback_used=false`，最终 `submission_ready.csv` 为 24 行、0 NaN；这不是 formal 96 生产验收。

> **96 点（15min）部署**：通用运行规则见 [docs/RUNBOOK.md](docs/RUNBOOK.md) 第 10 节；新服务器环境、full-source ledger、从 2026-08-17 接续到最近并切换每日生产见 [docs/SERVER_96_DEPLOYMENT_BACKFILL.md](docs/SERVER_96_DEPLOYMENT_BACKFILL.md)。
> 96 点调试过程与 7 个兼容 bug 记录见 [docs/archive/historical/96_POINT_DEBUG_LOG_20260803.md](docs/archive/historical/96_POINT_DEBUG_LOG_20260803.md)。
> 工程复现、质量门和回滚规则见 [docs/PROJECT_GOVERNANCE.md](docs/PROJECT_GOVERNANCE.md)。

---

## 1. 正式链路

以下五阶段图和 24 行验收记录描述的是 **24 点 legacy/advanced compatibility**；96 点
正式生产只执行后面的四阶段 façade，classifier 仅保留 shadow/replay。

```text
输入小时级山东电力现货数据
    ↓
    ledger_predict：7 条生产模型腿预测目标日 24 小时（DA 3 + RT 4）
    ↓
ledger_weight：默认从 ledger 中自适应选择最近 30 个完整训练日，学习动态融合权重
    ↓
ledger_fuse：按 task / period / model 权重融合
    ↓
    ledger_classifier：24 legacy/96 shadow-replay 的缓存版级联分类器；formal 96 production 不执行 classifier
    ↓
    final_outputs：生成 final/submission_ready.csv（优先使用分类器校正后的实时预测）
    ↓
postflight：校验 24 行、6 列、无 NaN、manifest 完整
```

> 实验候选：`--weight-learner champion_short` 使用 14 日窗口（7 日训练 + 7 日
> 验证）和 DA/RT 半衰期 7 日，支持冠军门控与有界负权。该选项仅用于
> `outputs/experiments/` 下的历史账本相对实验。当前 CLI 默认学习器为 `smape_reg`
>（因果 SLSQP 软门控，三段权重）；`nnls`/`bgew`/`champion_short` 均需显式选择。

24 点 legacy 五阶段顺序：

```text
ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs
```

96 点正式生产四阶段顺序（classifier 软下线）：

```text
ledger_predict → ledger_weight → ledger_fuse → final_outputs → postflight
```

96 点正式生产 façade：

```text
python main.py --96 YYYY-MM-DD
python main.py --96 YYYY-MM-DD --predict both|dayahead|realtime
python main.py --96 YYYY-MM-DD --finish
```

正式 96 使用 `split_process`、CPU=2/GPU=1、RT916 stride=24、
`smape_reg/SLSQP`、period 三段权重和 prune threshold=0.05；旧 `--pipeline`
接口继续保留为 advanced compatibility。正式 `--96` 先强制 DB sync，再生成
current target 执行 DB sync 后生成 immutable D/T snapshot 与 FeatureView；closed historical target 优先复用成功 LIVE snapshot，没有时使用 `HISTORICAL_PROXY_V1`，两者共用 FeatureView；`--finish` 只复用 Stage1 snapshot/provenance。
prediction cache 只复用带当前
当前 formal96 route/protocol contract 的产物，并校验 snapshot/FeatureView、RT916 stride/SGDFNet anchor；
失败的 full attempt 会保留最近一次合法 prediction provenance，使后续 `--finish` 可恢复。

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

## 2. 当前交付状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 数据同步 `sync_dataset` | PASS | 支持 db / http / local / auto |
| LightGBM target-day NaN | FIXED | 目标日 `日前电价` 未发布时保留推理行，不再 NoneType |
| SGDFNet formal 96 anchor | FIXED + LIVE PASS | D-1 decision-day DA p1..p96；wrapper live path 已实跑 96/96，正常 `fallback_used=false` |
| DA/RT adaptive weight days | FIXED | Dayahead 和 Realtime 都从 D-1 向前找最近 30 个完整训练日 |
| hour_business 严格校验 | FIXED | prediction / actual 必须严格为 `{1..24}` |
| `age_days` 位置计算 | FIXED | adaptive 选中日按列表位置计算，最近完整日为 1 |
| Windows UTF-8 manifest | FIXED | JSON 读写显式 `encoding="utf-8"` |
| 回归测试 | PASS | adaptive 40/40、stability 29/29、NaN regression 16/16、sync 41/41 |
| 2026-07-03 正式陪跑 | PASS | NORMAL / exit 0 / postflight PASS / 24 行 0 NaN |

---

## 3. Adaptive Complete Training Days

`ledger_weight` 对 **Dayahead 和 Realtime 都使用同一套自适应训练日选择逻辑**：

1. 从目标日 `D-1` 开始向前扫描；
2. 跳过不完整日；
3. 收集最近 30 个完整训练日；
4. 选中日按从近到远排序；
5. `selected_days[0] → age_days=1`，最近完整日权重最高；
6. 在 `--weight-max-lookback-days` 范围内凑不够 30 天则失败，并在 manifest/log 中写明 skipped days 和 errors。

完整日定义：

| Task | Prediction 要求 | Actual 要求 |
|---|---|---|
| Dayahead | 3 模型 × `hour_business={1..24}`，`y_pred` 无 NaN | `hour_business={1..24}`，`y_true` 无 NaN |
| Realtime | 4 模型 × `hour_business={1..24}`，`y_pred` 无 NaN | `hour_business={1..24}`，`y_true` 无 NaN |

模型列表：

```text
Dayahead: lightgbm, timesfm, timemixer
Realtime: timesfm, sgdfnet, timemixer, rt916
```

训练表期望规模：

```text
Dayahead: 30 × 3 × 24 = 2160 rows
Realtime: 30 × 4 × 24 = 2880 rows
Actual: 每个 task 30 × 24 = 720 rows
```

`validate_ledger_window()` 仍保留为 audit-only 检查，但不再作为 Dayahead hard gate。真正决定是否能学习权重的是 `select_complete_training_days()`。

---

## 4. 预测结果汇报

本节展示两套粒度的预测指标，口径统一按 `docs/metrics_calculation.md` 的**改良 SMAPE**（floor50 裁剪，准确率 = 1 − SMAPE，时段按 1-8h / 9-16h / 17-24h 划分）：

- **24 点（小时级）**：数据源 = AI电力交易平台复盘数据集（`outputs/platform_review/`，2026-01-01 ~ 08-06，218 天），模型 = 平台 1.0 / 2.0 两个版本，其中 **2.0 即当前要展示的融合模型**。
- **96 点（15 分钟级）**：172 天回测（2026-01-01 ~ 06-21，n = 16512），结果取自 8.6 会议纪要第二部分。

### 4.1 24 点预测指标（平台复盘，2026-01-01 ~ 08-06）

**日前（DA）** —— 2 个版本模型

| 指标 | 1.0模型 | 2.0模型（融合） |
|---|---|---|
| SMAPE 总 | 17.14% | **16.07%** |
| SMAPE 1-8h | 17.79% | **17.78%** |
| SMAPE 9-16h | 21.56% | **17.61%** |
| SMAPE 17-24h | **12.05%** | 12.81% |
| 准确率 总 | 82.86% | **83.93%** |
| MAE 总 | 53.6 | **51.2** |
| MSE 总 | 5308 | **4716** |
| MAPE 总 | **66.64%** | 69.18% |
| R2 总 | 0.859 | **0.875** |

**实时（RT）** —— 2 个版本模型

| 指标 | 1.0模型 | 2.0模型（融合） |
|---|---|---|
| SMAPE 总 | 29.85% | **26.53%** |
| SMAPE 1-8h | 27.45% | **21.12%** |
| SMAPE 9-16h | **38.63%** | 38.92% |
| SMAPE 17-24h | 23.45% | **19.54%** |
| 准确率 总 | 70.15% | **73.47%** |
| MAE 总 | 97.7 | **86.5** |
| MSE 总 | 21365 | **18544** |
| MAPE 总 | **66.83%** | 74.18% |
| R2 总 | 0.575 | **0.631** |

**系统级指标（DA+RT 联合，融合系统 2.0）**

| 指标 | 合计 | 1-8h | 9-16h | 17-24h |
|---|---|---|---|---|
| 度电套利·基础版（元/MWh） | 11.74 | 1.32 | 38.27 | -4.43 |
| 度电套利·改良版（元/MWh） | **13.16** | 8.22 | 45.04 | -9.54 |
| 总套利·基础版（元） | 26684 | 976 | 29124 | -3416 |
| 总套利·改良版（元） | 1935 | 206 | 2387 | -658 |

### 4.2 96 点预测指标（172 天回测，2026-01-01 ~ 06-21）

**日前（DA）** —— 3 个模型 + 融合，共 4 列

| 指标 | lightgbm | timemixer | timesfm | 融合模型 |
|---|---|---|---|---|
| SMAPE 总 | 25.12% | 27.61% | 25.74% | **23.89%** |
| SMAPE 1-8h | 27.76% | 31.22% | 28.65% | **25.94%** |
| SMAPE 9-16h | **22.25%** | 26.96% | 23.88% | 22.25% |
| SMAPE 17-24h | 25.34% | 24.65% | 24.70% | **23.48%** |
| 准确率 总 | 74.88% | 72.39% | 74.26% | **76.11%** |
| MAE 总 | 100.3 | 110.1 | 103.5 | **94.5** |
| MSE 总 | 33407 | 36921 | 31930 | **30871** |
| MAPE 总 | **48.07%** | 69.38% | 67.70% | 49.84% |
| R2 总 | 0.579 | 0.535 | 0.598 | **0.611** |

**实时（RT）** —— 4 个模型 + 融合，共 5 列

| 指标 | rt916 | sgdfnet | timemixer | timesfm | 融合模型 |
|---|---|---|---|---|---|
| SMAPE 总 | 33.56% | **23.17%** | 35.41% | 31.78% | 23.63% |
| SMAPE 1-8h | 32.91% | **17.75%** | 37.22% | 31.34% | 18.11% |
| SMAPE 9-16h | 41.10% | 33.29% | 42.25% | 37.78% | **34.11%** |
| SMAPE 17-24h | 26.66% | **18.47%** | 26.77% | 26.22% | 18.68% |
| 准确率 总 | 66.44% | **76.83%** | 64.59% | 68.22% | 76.37% |
| MAE 总 | 123.2 | **82.8** | 125.7 | 107.3 | 84.0 |
| MSE 总 | 29215 | 26384 | 35327 | 24833 | **24057** |
| MAPE 总 | 72.74% | **46.88%** | 74.29% | 61.30% | 47.58% |
| R2 总 | 0.502 | 0.550 | 0.398 | 0.577 | **0.590** |

**系统级指标（DA+RT 联合，融合系统）**

| 指标 | 合计 | 1-8h | 9-16h | 17-24h |
|---|---|---|---|---|
| 度电套利·基础版（元/MWh） | 31.62 | 1.86 | 78.28 | 12.66 |
| 度电套利·改良版（元/MWh） | **92.57** | 10.24 | 110.19 | 26.78 |
| 总套利·基础版（元） | 245128 | 4565 | 207055 | 33508 |
| 总套利·改良版（元） | 97383 | 737 | 93003 | 3642 |

> 改良版 = 在基础版售电触发条件上，额外要求「预测实时价 > 预测日前价」，两条件同时成立才触发售电。96 点下度电套利提升约 3 倍（31.62 → 92.57 元/MWh）；24 点下改良版单笔收益更高但触发更少（总套利 1935 元 vs 基础版 26684 元）。

---

> **正式交付记录**：2026-07-03 单日 NORMAL 验收（`ledger_predict → … → final_outputs` 五阶段 complete，`postflight=PASS`，`exit_code=0`，`fallback_used=false`，`submission_ready.csv` 24 行 0 NaN）归档见 `docs/archive/historical/ACCEPTANCE_REPORT.md`。

---

## 5. 快速开始

### 5.1 安装环境

```bash
conda create -n epf-2 python=3.11 -y
conda activate epf-2
pip install -r requirements.txt
```

Windows + CUDA 已验证。GPU 模型建议保持串行，避免 OOM；TimeMixer 在当前
Torch/CUDA 基线不支持严格 CUDA 确定性，因此 GPU 交付不要传
`--deterministic`，严格复现实验请切 CPU。

### 5.2 准备数据

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

### 5.3 24点（小时级）数据同步

> 本项目跑通两套粒度：**24 点（小时级）** 与 **96 点（15分钟级）**，数据同步命令不同，分别见 §5.3 与 §5.4。

推荐两步式，便于区分数据问题和模型问题：

```bash
python main.py --pipeline sync_dataset --sync-source auto --force-sync --require-fresh-data
python main.py YYYY-MM-DD --data-path data/24/canonical/shandong_pmos_hourly.xlsx
```

也可以一条命令：

```bash
python main.py YYYY-MM-DD --sync-data-before-run --require-fresh-data
```

### 5.4 96点（15分钟级）数据同步

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

## 6. 运行阶段：正式陪跑 与 复现

> **核心机制**：融合权重学习器（`ledger_weight`）只从 persistent `ledger` 读取最近 30 个完整历史日；每日 `ledger_predict` 会自动把当日 DA3/RT4 prediction 和可得 actual 追加到 ledger。因此 `outputs/96/ledger/` 本身就是可随服务器迁移、会每日增长的生产状态。
> 新服务器若已有历史 prediction/actual ledger，不必重新暖机30天，但 formal96 禁止裸复制 legacy/FeatureStore 目录：必须先走审计式 warm-start migration，再进入正式链。

| 运行方式 | 前提 | 干什么 |
|---|---|---|
| **正式陪跑（24 legacy）** | 没有任何预测结果 | 从 7 模型预测 `ledger_predict` 开始，跑完整五阶段，边跑边积累账本 |
| **正式 façade（96）** | `outputs/96/ledger` 已有最近30个完整历史日 | 使用 `--96 DATE` 一条命令跑四阶段；历史不足时在模型前 fail-closed |
| **96 新服务器 cold-start** | 已有旧服务器 prediction/actual ledger | 先用 `bootstrap_96_production_ledger.py` 审计迁移30日，再运行 `--96 DATE` |
| **复现** | 已有预测结果（如直接上传 30 天预测/账本文件） | 跳过预测，直接用 `ledger_weight` 学习权重并出结果 |

### 6.1 正式陪跑（24 legacy 五阶段；96 使用四阶段 façade）

24 点 legacy 完整五阶段：

```text
ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs
```

#### 6.1.1 24 点（hourly）正式陪跑

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

#### 6.1.2 96 点（15min）正式陪跑

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

**Historical Proxy 首日实机验收：** `python main.py --96 2026-08-17` 已真实跑通，`HISTORICAL_PROXY_V1` / p56 生效，7 个模型腿各96点，SGDFNet anchor=`2026-08-16` DA96、RT916 stride=24，learner 严格只使用到 T-2=`2026-08-15`，weight/fuse/final/postflight 全部 PASS，delivery=NORMAL。该机本次 full DB sync 约4分37秒、正式四阶段约10分17秒，端到端约14分56秒；服务器实际耗时以 GPU 与数据库网络为准。

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

### 6.2 复现（已有预测结果，直接学权重）

融合权重学习器要学**前 30 天的预测结果**才能出权重。两条路二选一：

1. **正式陪跑**：没有任何预测结果 → 先按 §6.1 跑完整链路，边跑边积累账本；
2. **复现**：直接把 30 天的预测/账本文件上传、拷进 ledger → 跳过模型预测，直接学权重出结果。

复现命令如下（`ledger_predict` 缓存命中秒过，重点在学权重）。

#### 6.2.1 24 点（hourly）复现

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

只想验证后半链路（7 模型已跑完、不重跑模型）的完整做法见 §6.5 副线 C。

#### 6.2.2 96 点（15min）复现

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

### 6.3 副线 A：简单跑 / 快速验收

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

### 6.4 副线 B：复杂全量跑 / 生产完整跑

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

### 6.5 副线 C：已有预测结果，只验证后半链路

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

### 6.6 历史副线 D：AI电力交易平台复盘数据获取（已归档）

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

## 7. 不推荐用于正式 NORMAL 的参数

下面参数只用于诊断或应急，不作为 NORMAL 交付依据：

```text
--allow-missing-models
--allow-equal-weight-fallback
--no-range-preflight
```

如果用了这些参数跑通，只能说明工程链路可继续，不代表正式 NORMAL。

---

## 8. Ledger 目录

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

## 9. 验证命令

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

## 10. 如果需要补 ledger

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

## 11. 输出文件

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

## 12. Delivery Status

| delivery_status | exit code | 含义 |
|---|---:|---|
| NORMAL | 0 | 对应分辨率的正式阶段正常完成，postflight PASS |
| DEGRADED_DELIVERED | 2 | 正常链路失败，但 emergency fallback 生成可交付文件 |
| FAILED_NO_DELIVERY | 1 | 正常链路和 fallback 均失败，无可用交付 |

正式验收优先使用 NORMAL。24 legacy 若使用 DEGRADED，必须说明 fallback 原因和后续修复计划；formal 96 合同失败必须 `FAILED_NO_DELIVERY`，禁止 emergency/degraded fallback。

---

## 13. Troubleshooting

| 问题 | 判断 | 处理 |
|---|---|---|
| LightGBM `NoneType` | 旧版本未兼容目标日 NaN | 拉取最新 main |
| SGDFNet 24 行 NaN | 旧版本 `da_anchor` 为 NaN | 拉取最新 main |
| `ledger_weight` 凑不够 30 天 | ledger 不足或 lookback 太短 | 补 ledger / backfill / 提高 `--weight-max-lookback-days` |
| `UnicodeDecodeError: gbk` | Windows 默认编码读 JSON | 拉取最新 main，JSON 读写已显式 UTF-8 |
| `submission_ready.csv` 有 NaN | fuse/final 缺某个 task | 查 `delivery_report.md` 与 `run_manifest.json` |
| exit code 2 | fallback 交付 | 查看 `fallback_report.md/json`，修复后 `--force` 重跑 |
| exit code 1 | 无交付 | 查看 `run_manifest.json.errors` |

---

## 14. Git 安全

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

## 15. 最近关键修复

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

## 16. 一句话结论

模型预测流程、DA/RT 自适应权重学习、融合、分类器、最终输出与 postflight 均已通过 2026-07-03 正式陪跑验收。完整 NORMAL 交付的核心前提是：**ledger 中能在 lookback 范围内为 Dayahead 和 Realtime 各自找到最近 30 个完整训练日。**

---

## 17. 爬虫 & 数据同步 FAQ

### 17.1 PMOS 连不上 / 爬虫报错

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

### 17.2 电脑关机 / 休眠 → 当天没数据

**现象：** 某天数据缺失，GitHub Actions 发出告警 Issue。

**原因：** 定时任务依赖办公电脑在 08:00 处于开机或睡眠状态。

**解决：**
1. 在任务计划程序（Task Scheduler）中找到"PMOS数据爬虫"
2. 右键 → 属性 → **条件（Conditions）** 选项卡
3. 勾选 **"唤醒计算机运行此任务"**（Wake the computer to run this task）
4. 电脑保持**睡眠（Sleep）**状态即可，不要完全关机

**补充：** 睡眠状态耗电极低（≈ 台式机 3-5W），可以长期不关。

### 17.3 GitHub Actions 检查失败

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

### 17.4 backfill_unit_96.exe 闪退 / 无反应

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

### 17.5 本地同步 vs 数据库数据不一致

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

### 17.6 云端数据库连接失败

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

## 18. AI电力交易平台 · 电价预测复盘数据更新

> ⚠️ **平台区分：** 本节面向的是「AI电力交易平台」自建演示站
> **http://47.114.107.96/**（账号 `user` / `user123`），
> **与国网山东电力交易平台 PMOS（`pmos.sd.sgcc.com.cn`）是完全不同的两个系统**。
> 第 17 节 FAQ 里讲的 Cookie / 数据库 / 定时任务全部针对国网 PMOS；
> 本节的工具不走 Cookie、不碰数据库，是独立的第二数据源（平台自带的「电价预测复盘」模块）。

### 18.1 数据集位置

稳定路径 `outputs/platform_review/`（已放行 git 跟踪，可直接提交推送）：

| 文件 | 内容 |
|---|---|
| `电价预测复盘.xlsx` | 平台原始导出（详细数据 + 统计报告 两个 sheet） |
| `电价预测复盘_详细数据.csv` | 逐小时：实时电价 / 日前电价 + 1.0/2.0 模型预测价 |
| `电价预测复盘_统计报告.csv` | 全量 + 分月综合准确率统计 |

当前覆盖：**2026-01-01 ~ 2026-08-06**（218 天 × 24 小时 = 5232 行）。

### 18.2 更新命令

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
