# Electricity Forecast Delivery Pipeline v2.5

山东电力现货价格预测交付链路：**7 模型预测 + Ledger 自适应动态权重融合 + Realtime 极端价格分类校正 + 最终交付校验**。

当前版本已经完成 2026-07-03 正式陪跑验收：五阶段全部 `complete`，`postflight=PASS`，`delivery_status=NORMAL`，`exit_code=0`，`fallback_used=false`，最终 `submission_ready.csv` 为 24 行、0 NaN。

> **96 点（15min）部署**：请阅读 [docs/RUNBOOK.md](docs/RUNBOOK.md) 第 10 节 —— 同步、范围运行和 GPU 部署入口。
> 96 点调试过程与 7 个兼容 bug 记录见 [docs/archive/historical/96_POINT_DEBUG_LOG_20260803.md](docs/archive/historical/96_POINT_DEBUG_LOG_20260803.md)。
> 工程复现、质量门和回滚规则见 [docs/PROJECT_GOVERNANCE.md](docs/PROJECT_GOVERNANCE.md)。

---

## 1. 正式链路

```text
输入小时级山东电力现货数据
    ↓
    ledger_predict：7 条生产模型腿预测目标日 24 小时（DA 3 + RT 4）
    ↓
ledger_weight：默认从 ledger 中自适应选择最近 30 个完整训练日，学习动态融合权重
    ↓
ledger_fuse：按 task / period / model 权重融合
    ↓
    ledger_classifier：使用24/96通用的缓存版级联分类器，仅对实时电价进行-80分类，输出分类校正后的实时电价预测结果
    ↓
    final_outputs：生成 final/submission_ready.csv（优先使用分类器校正后的实时预测）
    ↓
postflight：校验 24 行、6 列、无 NaN、manifest 完整
```

> 实验候选：`--weight-learner champion_short` 使用 14 日窗口（7 日训练 + 7 日
> 验证）和 DA/RT 半衰期 7 日，支持冠军门控与有界负权。该选项仅用于
> `outputs/experiments/` 下的历史账本相对实验；默认学习器仍为 `nnls`，干净新账本
> 独立复验前不替换生产链路。

五阶段顺序：

```text
ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs
```

最终交付文件：

```text
outputs/runs/YYYY-MM-DD/final/submission_ready.csv
```

标准列：

```text
business_day, ds, hour_business, period, dayahead_price, realtime_price
```

---

## 2. 当前交付状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 数据同步 `sync_dataset` | PASS | 支持 db / http / local / auto |
| LightGBM target-day NaN | FIXED | 目标日 `日前电价` 未发布时保留推理行，不再 NoneType |
| SGDFNet target-day NaN | FIXED | `da_anchor` 缺失时使用历史同小时中位数 fallback |
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
# 全量同步（从云端数据库拉取 epf_market_data_96 + epf_unit_data_96）
python main.py --pipeline sync_dataset --resolution 15min --sync-source db

# 增量同步（重拉最近7天并合并）
python main.py --pipeline sync_dataset --resolution 15min --sync-mode incremental
```

**数据来源**
- `epf_market_data_96` — 全省级市场特征（直调负荷、地方电厂出力、外电、风电、光伏、核电、竞价空间、检修、备用等）
- `epf_unit_data_96` — 机组级日前/实时电价、出力、电量、开机状态

**同步输出（本地镜像）**
```text
data/96/remote/parquet/epf_market_data_96.parquet   — 全省市场特征96点（含 actual/fcast）
data/96/remote/parquet/epf_unit_data_96.parquet     — 机组级96点电价/出力
```

**合并成一张宽表**（对标 24 点 `shandong_pmos_hourly.xlsx`，含 `日前电价/实时电价`）：

```bash
python scripts/sync/build_96_full_table.py
# 输出 data/96/model_input/shandong_pmos_96_model_input_clean.xlsx(.csv)
```

> 注意：96 点 `日前电价/实时电价` 来自机组级 `da_cq_price/rt_cq_price`，是**单机组出清价**，
> 与 24 点的全省市场均价口径不同。差异详见 `docs/DATA_CONTRACT_96.md` 第 8 节。

**定时任务说明**
- 爬虫每日 08:00 自动爬取最新96点数据写入MySQL
- 本地镜像由 `--resolution 15min` 同步或 `scripts/sync/build_96_full_table.py` 手动/定时刷新
- 同步报告输出至 `outputs/96/sync/`

---

## 6. 运行阶段：正式陪跑 与 复现

> **核心机制**：融合权重学习器（`ledger_weight`）需要学习**前 30 天的预测结果**才能学到权重。
> 因此按「有没有预测结果」分两种运行方式，命令也分 24 点（hourly）与 96 点（15min）两套：

| 运行方式 | 前提 | 干什么 |
|---|---|---|
| **正式陪跑** | 没有任何预测结果 | 从 7 模型预测 `ledger_predict` 开始，跑完整五阶段，边跑边积累账本 |
| **复现** | 已有预测结果（如直接上传 30 天预测/账本文件） | 跳过预测，直接用 `ledger_weight` 学习权重并出结果 |

### 6.1 正式陪跑（无预测结果，全五阶段）

完整五阶段：

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

`--resolution 15min` 默认使用独立的 `outputs/ledger_96` + `outputs/runs_96`；也可以通过
`--ledger-root` 和 `--runs-root` 显式覆盖：

```bash
python scripts/server/run_96_prediction_backtest.py \
  --data-path data/96/model_input/pmos_96_model_input_clean.xlsx \
  --actual-data-path data/96/actual_price/pmos_96_price_actual.xlsx \
  --report-start 2026-01-01 --end 2026-08-15 \
  --output-root outputs/96/feature_store
```

服务器完整回测必须使用未被标记污染的 96 点模型输入；脚本会拒绝
`shandong_pmos_96_model_input.xlsx` 等历史污染文件。先用
`--report-start 2026-01-01 --end 2026-01-01 --no-prewarm` 做单日耗时和 96 点完整性
smoke，再启动 2025-12-18（14 日预热）至 2026-08-15 的完整预测阶段。预测完成后，
只拉取 `outputs/96/feature_store/ledger`、`runs` 和范围 manifest 做本地学习器回放。

### FeatureStore 候选链路（与原链路隔离）

原有 `outputs/ledger_96` + `outputs/runs_96` 链路保持不变。FeatureStore
验证链路使用按分辨率隔离的新目录，不会污染原有权重学习账本：

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
`scripts/server/run_96_prediction_backtest.py`，并同时提供独立的 actual-price source。

候选链路的结果位于
`outputs/96/feature_store/ledger/`、
`outputs/96/feature_store/runs/`，缓存位于
`outputs/96/feature_store/cache/`。正式模型池唯一来源为
`fusion/model_pool.py`：日前为 `lightgbm + timesfm + timemixer`，实时为
`timesfm + sgdfnet + timemixer + rt916`；LightGBM 实时入口不进入生产池。

96 点权威实际数据为 `data/96/authoritative/pmos_96_全量.csv`，只用于
`scripts/tests/check_96_vs_24_actual.py` 的交叉验证；价格和预测输入必须来自
`data/96/model_input/`，两者禁止混用。

多日预热 + 全链路（服务器推荐，放 tmux 里跑）：

```bash
bash scripts/auto_preheat_backtest.sh
# 阶段1: ledger_backfill 2025-12-01~12-31 预热，补足 30 天权重学习历史（~8h）
# 阶段2: 账本 ≥30 天后自动 ledger_full_range 2026-01-01 起逐日跑五阶段
```

成功标准（96 点，不满足就是退化成 24/72 点）：

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

已有 96 点预测 CSV / runs 结果时，先用已有预测重建账本，再直接跑目标日学权重（不重跑模型）：

```bash
# 用已有预测结果重建 prediction ledger
python scripts/sync/rebuild_prediction_ledger_96.py --runs-root outputs/runs_96 --ledger-root outputs/ledger_96

# 或从 output/prediction_96/*.csv 把预测种回 runs 缓存（可选）
python scripts/seed_96_ledger_cache.py --date 2026-07-16

# 直接跑目标日：predict 缓存命中，直接学权重
python main.py 2026-07-16 \
  --resolution 15min \
  --data-path data/96/model_input/shandong_pmos_96_model_input_clean.xlsx \
  --ledger-root outputs/ledger_96 \
  --runs-root outputs/runs_96 \
  --weight-max-lookback-days 180
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

### 6.6 副线 D：AI电力交易平台复盘数据获取

我们现在已经有了**交易可视化平台**（AI电力交易平台 http://47.114.107.96/，账号 user/user123），
复盘模块有「电价预测复盘」，可以直接用爬虫程序把 日前/实时 电价 + 各模型预测价抓成数据集，不用再手搓 Excel。

> ⚠️ 注意：该平台是自建演示站，**与国网 PMOS 爬虫无关**，是独立数据源。

命令行更新数据集：

```bash
# 更新到最新（自动：从数据集最早日期 ~ 今天）
python scripts/crawler/platform_review_update.py

# 指定抓取区间
python scripts/crawler/platform_review_update.py --start 2026-01-01 --end 2026-08-06
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

默认 ledger 根目录：

```text
outputs/ledger
```

也可指定：

```bash
--ledger-root <your_ledger_root>
```

核心文件：

| 类型 | 路径 |
|---|---|
| Dayahead prediction | `outputs/ledger/dayahead/prediction/prediction_ledger.parquet` |
| Dayahead actual | `outputs/ledger/dayahead/actual/actual_ledger.parquet` |
| Realtime prediction | `outputs/ledger/realtime/prediction/prediction_ledger.parquet` |
| Realtime actual | `outputs/ledger/realtime/actual/actual_ledger.parquet` |

权重学习只读取 ledger，不直接读取 `outputs/runs`。每日 `ledger_predict` 会把当日预测追加到 prediction ledger；actual ledger 会按可得实际值更新。

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
| `outputs/runs/YYYY-MM-DD/run_manifest.json` | 五阶段运行元信息 |
| `outputs/runs/YYYY-MM-DD/delivery_report.md` | 交付报告 |
| `outputs/runs/YYYY-MM-DD/dayahead/weight/weights.csv` | Dayahead 融合权重 |
| `outputs/runs/YYYY-MM-DD/realtime/weight/weights.csv` | Realtime 融合权重 |
| `outputs/runs/YYYY-MM-DD/{task}/fuse/fused_predictions.csv` | 融合结果 |
| `outputs/runs/YYYY-MM-DD/realtime/final/realtime_final_predictions_corrected.csv` | 分类器校正后 realtime |

---

## 12. Delivery Status

| delivery_status | exit code | 含义 |
|---|---:|---|
| NORMAL | 0 | 五阶段正常完成，postflight PASS |
| DEGRADED_DELIVERED | 2 | 正常链路失败，但 emergency fallback 生成可交付文件 |
| FAILED_NO_DELIVERY | 1 | 正常链路和 fallback 均失败，无可用交付 |

正式验收优先使用 NORMAL。若使用 DEGRADED，必须说明 fallback 原因和后续修复计划。

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

**原因：** 本地同步需要手动或定时执行。爬虫只写入云端 MySQL，不直接更新本地文件。

**解决：**
```bash
# 增量同步（重拉最近7天并合并到本地镜像）
python main.py --pipeline sync_dataset --resolution 15min --sync-mode incremental

# 全量覆盖本地镜像（从数据库重新拉取）
python main.py --pipeline sync_dataset --resolution 15min --sync-source db

# 重新生成合并宽表
python scripts/sync/build_96_full_table.py
```

建议在办公电脑定时任务中追加以上同步命令，使爬虫完成后自动同步到本地文件。

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
python scripts/crawler/platform_review_update.py

# 指定抓取区间（明细按 time 合并去重，区间外旧数据保留）
python scripts/crawler/platform_review_update.py --start 2026-01-01 --end 2026-08-06

# 只指定结束日期（从数据集最早日开始）
python scripts/crawler/platform_review_update.py --end 2026-08-06

# 换账号 / 换输出目录
python scripts/crawler/platform_review_update.py --user user --password user123 --out outputs/platform_review
```

说明：
- 区间内数据重新拉取并覆盖，区间外历史数据自动保留（按 `time` 去重合并）。
- 「统计报告」仅在本次区间能覆盖现有全部数据时才刷新（避免聚合口径变小）。
- 平台最后一天的实时电价可能晚发布（如 08-06 15–24 点实时价），隔天重跑一次即可补上。
