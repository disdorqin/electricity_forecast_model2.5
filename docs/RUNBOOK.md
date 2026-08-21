# Runbook

## Prerequisites

- Conda environment `epf-2` with dependencies installed (`pip install -r requirements.txt`)
- Input data file at `data/24/canonical/shandong_pmos_hourly.xlsx`
- CUDA-capable GPU for TimeMixer and RT916 models
- For GPU TimeMixer, keep `--deterministic` off: strict CUDA determinism is
  rejected because the configured Torch/CUDA build has no deterministic
  `upsample` backward kernel. Use CPU for strict deterministic experiments.
- The delivery scheduler serializes CPU/GPU queues by default to isolate this
  Torch runtime policy; overlap is benchmark-only via
  `EFM3_ALLOW_GPU_CPU_OVERLAP=1`.

> **Self-contained:** LightGBM and TimesFMBackend are bundled in this repository.
> No external EPF v1.0 repository is required. The `--epf-v1-root` option is
> retained only for legacy compatibility.

## 1. Environment Check

```powershell
conda run -n epf-2 python scripts/env_check.py
```

Expected output:
```
ENV_CHECK
python: 3.11.x OK
cuda: available OK
dependencies: all OK
local_paths: all OK
status: PASS
```

## 2. Smoke Test (Quick Validation)

Run a minimal end-to-end test with reduced training scope:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_smoke --date 2026-02-24 ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx ^
    --smoke-training-months 3 --smoke-timemixer-epochs 3 --smoke-timemixer-patience 1 ^
    --max-cpu-workers 2 --max-gpu-workers 1 ^
    --seed 42 --deterministic --force
```

Expected: ~10-15 minutes. All production model legs produce 24-hour predictions.

## 3. 30-Day Backfill

Populate the prediction ledger with historical data:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_backfill ^
    --start YYYY-MM-DD --end YYYY-MM-DD ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx ^
    --max-cpu-workers 2 --max-gpu-workers 1 ^
    --seed 42 --deterministic
```

Example: 2026-01-25 through 2026-02-23 (30 days).

Expected runtime: overnight (~10-12 hours depending on GPU).

Progress tracking: each completed day logs to `outputs/runs/backfill_manifest.json`.

## 4. Full Pipeline (Single Day)

Run all 5 stages for a target date:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_full ^
    --date YYYY-MM-DD ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx ^
    --max-cpu-workers 2 --max-gpu-workers 1 ^
    --seed 42 --deterministic
```

Expected runtime: ~30-40 minutes (first run; subsequent runs are faster due to prediction cache).

## 5. Verify Existing Outputs

Validate a completed run without re-running models:

```powershell
conda run -n epf-2 python scripts/tests/verify_final_pipeline.py --date YYYY-MM-DD --runs-root outputs/runs
```

Expected output:
```
FINAL_VERIFY: YYYY-MM-DD
ledger_predict: complete
ledger_weight: complete
ledger_fuse: complete
ledger_classifier: complete
final_outputs: complete
errors: 0
warnings: 0
FINAL_STATUS: PASS
```

## 6. Reproducibility Check

Verify that two runs with the same seed produce identical outputs:

```powershell
conda run -n epf-2 python scripts/tests/check_reproducibility.py YYYY-MM-DD ^
    --seed 42 --deterministic --epf-v1-root "path\to\epf-v1" --keep-tmp
```

Expected: `=== Result: PASS (all outputs identical) ===`

## 7. TimeMixer Alignment Check

Verify TimeMixer timestamp alignment (hour 24 = D+1 00:00 rule):

```powershell
conda run -n epf-2 python scripts/tests/check_timemixer_alignment.py --date YYYY-MM-DD
```

Expected: `ALL OK` for both dayahead and realtime.

## 8. Stage-by-Stage Commands (Debug)

Run individual stages if rerunning from scratch:

```powershell
# Stage 1: Predict
conda run -n epf-2 python main.py --pipeline ledger_predict --date YYYY-MM-DD

# Stage 2: Weight learning
conda run -n epf-2 python main.py --pipeline ledger_weight --date YYYY-MM-DD

# Stage 3: Fusion
conda run -n epf-2 python main.py --pipeline ledger_fuse --date YYYY-MM-DD

# Stage 4: Classifier
conda run -n epf-2 python main.py --pipeline ledger_classifier --date YYYY-MM-DD
```

分类器正式 bridge 使用可复用的 `range_runner`，不再通过生产 bridge 启动旧的
`run_daily.py` 子进程。24 点和 96 点均先规范化到分类器的小时语义；96 点输入按小时
聚合，最终校正结果再由 bridge 广播回 96 个 15 分钟槽位。共享缓存位于：

```text
outputs/24/feature_store/cache/classifier/realtime/<source-spec-hash>/
outputs/96/feature_store/cache/classifier/realtime/<source-spec-hash>/
```

缓存包括规范化输入、Stage1/Stage2 特征、p1 概率和 manifest；扩展日期范围时只补齐
缺失时间戳，不重复执行历史预热。旧入口仍保留用于兼容和对照，不是正式 bridge 的
默认执行路径。实验区可直接运行：

```powershell
python scripts/experiments/classifier_range/run_range.py `
  --source data/24/canonical/shandong_pmos_hourly.xlsx `
  --start 2026-01-01 --end 2026-01-07 `
  --resolution hourly --task realtime
```

切换前的 24 点单日逐点回归已通过：决策列一致，概率最大绝对误差约
`5.6e-17`；7 日增量回放使用同一缓存完成。分类器只改变执行组织和缓存，不改变
原有特征、模型、阈值、滚动训练和校正规则。

## 9. Force Re-run

To force a full rerun (bypass prediction cache):

```powershell
conda run -n epf-2 python main.py --pipeline ledger_full --date YYYY-MM-DD ^
    --data-path data/24/canonical/shandong_pmos_hourly.xlsx --force
```

To force a specific stage, add `--force` after the cleaned outputs.

## Common CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--date` | — | Target date YYYY-MM-DD |
| `--start` / `--end` | — | Date range for backfill |
| `--epf-v1-root` | None | [Optional legacy compatibility] External EPF v1.0 root |
| `--seed` | 42 | Global random seed |
| `--deterministic` | False | Strict deterministic mode; CPU-only for TimeMixer |
| `--max-cpu-workers` | 2 | CPU parallel workers |
| `--max-gpu-workers` | 1 | GPU serial workers |
| `--realtime-cutoff-hour` | 14 | Realtime cutoff hour on D-1 |
| `--force` | False | Force rerun, bypass cache |
| `--data-path` | `data/24/canonical/shandong_pmos_hourly.xlsx` | Input data file path |
| `--ledger-root` | resolution-dependent | Override ledger storage root |
| `--runs-root` | resolution-dependent | Override daily run output root |
| `--output-profile` | `legacy` | `legacy` or isolated `feature_store` candidate roots |
| `--feature-store-root` | profile-derived | Candidate raw/materialized feature cache root |

To run the isolated FeatureStore candidate chain without touching the legacy
ledger:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_full --date YYYY-MM-DD ^
    --resolution 15min --output-profile feature_store --feature-store-mode raw
```

The business model pool is fixed in `fusion/model_pool.py`: Dayahead uses
`lightgbm + timesfm + timemixer`; Realtime uses
`timesfm + sgdfnet + timemixer + rt916`. Do not repeat or override these
lists in pipeline scripts.

## Verification Scripts Reference

```powershell
# Full environment check
python scripts/env_check.py

# Verify final pipeline outputs
python scripts/tests/verify_final_pipeline.py --date 2026-02-24 --runs-root outputs/runs

# Reproducibility
python scripts/tests/check_reproducibility.py 2026-02-24 --seed 42 --deterministic

# TimeMixer alignment
python scripts/tests/check_timemixer_alignment.py --date 2026-02-24

# Smoke verification
python scripts/tests/verify_smoke.py
```

## 10. 数据同步、范围运行与部署职责

本文件统一承接数据同步、范围回测和服务器运行的操作入口；旧版专项手册已归档，后续只更新本节。

### 10.1 数据同步

```powershell
python main.py --pipeline sync_dataset --resolution hourly --sync-source db
python main.py --pipeline sync_dataset --resolution 15min --sync-source db
```

同步前后必须检查源表、分辨率、最新时间、重复键、每日 96 点完整性和 manifest。同步失败不得覆盖上一份有效快照。

### 10.2 多日范围运行

```powershell
python main.py --pipeline ledger_full_range --start YYYY-MM-DD --end YYYY-MM-DD --resolution hourly --seed 42 --deterministic
```

范围运行仍按五阶段执行：`ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs`。正式运行前必须确认每个任务具备最近 30 个完整训练日；不满足时只能输出降级状态，不能伪装成 NORMAL。

### 10.3 GPU/CPU环境

- Python、Torch、JAX 和 CUDA 版本以根目录 `requirements.txt` 为准；
- LightGBM、TimesFM 的轻量验证可在 CPU 执行；TimeMixer/RT916 完整训练按 GPU 环境执行；
- Windows 下训练脚本必须遵守 DataLoader worker 约束；
- 运行 manifest 记录 `resolution`、`cutoff`、`seed`、数据快照、git commit 和权重门控阈值。

### 10.4 回归顺序

1. 环境和数据预检；
2. CLI/编译检查；
3. 单模型或 smoke 检查；
4. ledger、权重门控、融合和最终输出检查；
5. 需要正式交付时再执行完整链路。

### 10.5 冠军门控学习器（实验候选）

`champion_short` 是显式实验选项，不改变默认 `nnls`，当前只允许用于已标记为
`historical-invalid-features` 的旧 96 点预测账本做相对融合验证。协议固定为
14 日窗口（7 日拟合 + 7 日验证）、DA/RT 半衰期均 7 日；负权只在有界冠军锚定
候选中出现。关闭融合权重门控必须显式传 `--weight-prune-threshold 0`，不得作为
生产默认值。

```powershell
python main.py --date YYYY-MM-DD --pipeline ledger_weight --resolution 15min `
  --validation-days 14 --weight-learner champion_short `
  --ledger-root outputs/ledger_96 `
  --runs-root outputs/experiments/champion_weight_ab/integration_runs `
  --weight-prune-threshold 0

python main.py --date YYYY-MM-DD --pipeline ledger_fuse --resolution 15min `
  --ledger-root outputs/ledger_96 `
  --runs-root outputs/experiments/champion_weight_ab/integration_runs `
  --weight-prune-threshold 0
```

验证重点是 `weight/candidate_metrics.csv`、`weights.csv`、`fuse/fused_debug.csv`
和 manifest；新鲜有效预测集建立后，必须在独立账本上重新回测，满足稳定性门槛后
才能讨论替换生产学习器。目标日预测必须包含模型池中的全部模型；缺失任一模型时
`champion_short` 应拒绝学习并由融合阶段报错，不得用残缺模型集合交付。

### 10.6 服务器 96 点预测账本（先预测、后回放）

服务器回测先只运行模型预测，不要让学习器实验重复触发模型训练。推荐使用
`scripts/server/run_96_prediction_backtest.py`，它会先检查输入是否为完整 96 点、
拒绝已标记的污染宽表、检查 actual/forecast 拷贝，并逐日记录模型完整性、耗时和
断点状态。TimesFM 默认留在 CPU，TimeMixer/RT916 使用单张 GPU；不要并行启动第二
个 GPU 进程。

先做单日耗时 smoke：

```bash
export TIMESFM_DEVICE=cpu
python scripts/server/run_96_prediction_backtest.py \
  --data-path data/96/model_input/pmos_96_model_input_clean.xlsx \
  --actual-data-path data/96/actual_price/pmos_96_price_actual.xlsx \
  --report-start 2026-01-01 --end 2026-01-01 --no-prewarm
```

确认日志和 `prediction_range_manifest.json` 中 DA/RT 全模型均为 96 点后，再运行完整
范围。完整运行会自动增加 14 天预热，实际预测区间为 2025-12-18 至 2026-08-15，
最终评估区间从 2026-01-01 开始：

```bash
python scripts/server/run_96_prediction_backtest.py \
  --data-path data/96/model_input/pmos_96_model_input_clean.xlsx \
  --actual-data-path data/96/actual_price/pmos_96_price_actual.xlsx \
  --report-start 2026-01-01 --end 2026-08-15 \
  --output-root outputs/96/feature_store
```

历史 RTX 3090 参考值为完整五阶段约 17 分钟/日；现有 96 点 FeatureStore 日志中，
预测阶段样本约 4.3--4.8 分钟/日，但这不是 3090 服务器保证值。241 天（含 14 天
预热）可先按约 17--20 小时的预测阶段规划，并保留完整五阶段约 68 小时的保守上限；
单日 smoke 完成后必须以 `prediction_range_manifest.json` 的实测日均耗时和 ETA 为准。
模型预测完成后，只需拉取 `outputs/96/feature_store/ledger` 和对应 `runs`/manifest，
在本地执行学习器 replay，不再重复运行模型。

`--actual-data-path` 必须是含有 `日前电价`、`实时电价` 的权威价格表；
`data/96/authoritative/pmos_96_全量.csv` 只用于实际特征/96 点真实性校验，通常不含这两列，
不能直接替代 actual-price source。模型输入与实际价格源均必须通过脚本的 96 点、重复槽位、
actual/forecast 拷贝门控。
