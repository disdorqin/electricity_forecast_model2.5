# Output Convention

## Overview

The pipeline generates three categories of **formal outputs**:

1. **Persistent ledger** (`outputs/ledger/`) — cross-date accumulated prediction and actual value storage
2. **Daily run artifacts** (`outputs/runs/{date}/`) — per-date model predictions, weights, fused results, and final deliverables
3. **Range summary** (`outputs/runs/range_{start}_to_{end}/`) — multi-day range pipeline manifest and summary CSV

A fourth category is defined for testing:

4. **Smoke test outputs** (`outputs/smoke/`) — lightweight test output, separate from production data

All outputs live under `outputs/` which is `.gitignore`d and never committed to Git.

## Legacy and FeatureStore candidate output profiles

The existing chain is preserved as the default. The FeatureStore candidate
chain has separate ledger, daily-run, and cache roots, so shadow validation
cannot contaminate the legacy training history.

| Profile | 24-point roots | 96-point roots | Status |
|---|---|---|---|
| `legacy` (default) | `outputs/ledger/`, `outputs/runs/` | `outputs/ledger_96/`, `outputs/runs_96/` | Existing chain |
| `feature_store` | `outputs/24/feature_store/ledger`, `outputs/24/feature_store/runs` | `outputs/96/feature_store/ledger`, `outputs/96/feature_store/runs` | Candidate/shadow chain |

FeatureStore cache artifacts for the candidate profile are stored under
`outputs/24/feature_store/cache/` or `outputs/96/feature_store/cache/`.
The former `outputs/feature_store_chain/` and `outputs/feature_store/` trees
are archived and are not write targets.

Select the candidate roots explicitly:

```powershell
python main.py --pipeline ledger_full --date YYYY-MM-DD `
  --resolution 15min --output-profile feature_store --feature-store-mode raw
```

96 点服务器预测阶段使用物化特征和双子进程调度：

```powershell
python main.py --pipeline ledger_predict --date YYYY-MM-DD `
  --resolution 15min --output-profile feature_store `
  --feature-store-mode materialized --resource-mode split_process
```

`split_process` 只对 96 点候选链路启用：CPU 子进程和 GPU 子进程同时启动，
两个队列内部均严格串行。范围预测期间 ledger 写入按目标日保存到
`prediction/parts/` 和 `actual/parts/`，范围成功结束后再压缩为 canonical
`prediction_ledger.parquet` / `actual_ledger.parquet`；中断时 parts 可直接用于续跑。

Explicit `--ledger-root` and `--runs-root` values override the profile.

Canonical candidate tree:

```text
outputs/
  24/feature_store/{cache,ledger,runs}/
  96/feature_store/{cache,ledger,runs}/
  24/sync/                 # 24-point sync manifest/report
  96/sync/                 # 96-point sync manifest/report
  archive/legacy_sync/     # old data_sync/data_sync_96 and old caches
```

For every `ledger_full` run, `run_manifest.json` is the root audit record;
fusion additionally writes `model_quality_gate.csv` and `fused_debug.csv` per
task. A final normal delivery must contain 24 or 96 rows according to the
resolution and zero numeric NaN in `final/submission_ready.csv`.

### Formal vs. Non-formal Outputs

**Formal outputs (part of the production ledger pipeline):**

| Directory | Purpose |
|-----------|---------|
| `outputs/ledger/` | Cross-date accumulated prediction/actual ledger for weight learning |
| `outputs/runs/YYYY-MM-DD/` | Single-day full pipeline output + submission_ready.csv |
| `outputs/runs/range_*_to_*/` | Range pipeline manifest + summary |
| `outputs/smoke/` | Smoke test prediction outputs (lightweight validation) |

**All other directories under `outputs/` are local-only legacy, experimental, or debug artifacts.** If present on your machine, they may include:

| Directory | Origin | Status |
|-----------|--------|--------|
| `outputs/2026-02-01` (direct date) | Old staging runs | Legacy — not part of ledger pipeline |
| `outputs/unified_runs/` | EPF v2.0 unified output | Legacy — unused by ledger pipeline |
| `outputs/audit_30day_*/` | `scripts/audit_30day_backfill.py` | Debug/audit artifact |
| `outputs/repro_check/` | `scripts/check_reproducibility.py` | Debug/verification artifact |
| `outputs/RT916_SpikeMarketLab/` | RT916 model debug output | Debug artifact — safe to delete |

These can be safely ignored or deleted; they are not required by the ledger_full pipeline.

---

## Ledger Storage

```
outputs/ledger/
  dayahead/
    prediction/
      prediction_ledger.parquet   ← Parquet format (primary)
      prediction_ledger.csv       ← CSV format (inspection)
    actual/
      actual_ledger.parquet
      actual_ledger.csv
  realtime/
    prediction/
      prediction_ledger.parquet
      prediction_ledger.csv
    actual/
      actual_ledger.parquet
      actual_ledger.csv
```

- Ledger rows are keyed by `(task, business_day, hour_business)`.
- Duplicate entries are automatically deduplicated on append.
- Ledger is the source of truth for weight learning (`ledger_weight` reads D-30 to D-1 from ledger).

---

## Daily Run Directory

Each day runs the full five-stage pipeline:

```
  ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs
```

```
outputs/runs/{YYYY-MM-DD}/
  run_manifest.json                         ← Full run metadata (all five stages)
    ├── ledger_predict: status, models, rows, model_runtime_config
    ├── ledger_weight: status, training_rows, day_gate range, weight_dir
    ├── ledger_fuse: status, fused_rows, fuse_dir
    ├── ledger_classifier: status, corrections_applied
    └── final_outputs: status, submission_ready_rows

  dayahead/
    prediction/
      all_model_predictions_long.csv        ← All models concatenated (72 rows)
      lightgbm_predictions.csv              ← Per-model (24 rows each)
      timemixer_predictions.csv
      timesfm_predictions.csv
    weight/
      weights.csv                           ← Learned weights per (task, period, model)
      dynamic_weight_trace.csv              ← Day-by-day weight evolution
      candidate_metrics.csv                 ← Per-model metrics on training window
      coverage_report.csv                   ← Prediction coverage by model
    fuse/
      fused_predictions.csv                 ← Weighted fusion result (24 rows)
      fused_debug.csv                       ← Per-model contribution debug info
    final/
      dayahead_final_predictions.csv        ← Final DA output (24 rows)

  realtime/
    prediction/
      all_model_predictions_long.csv        ← All models concatenated (96 rows)
      timesfm_predictions.csv               ← Per-model (24 rows each)
      sgdfnet_predictions.csv
      timemixer_predictions.csv
      rt916_predictions.csv
    weight/
      weights.csv                           ← Learned weights
      dynamic_weight_trace.csv
      candidate_metrics.csv
      coverage_report.csv
    fuse/
      fused_predictions.csv                 ← Weighted fusion result (24 rows)
      fused_debug.csv
    final/
      realtime_final_predictions.csv        ← Pre-classifier output (24 rows)
      realtime_final_predictions_corrected.csv ← Post-classifier output (24 rows)
      classifier_report.json                ← Classifier run metadata

  final/
    dayahead_final_predictions.csv          ← Copy of DA final
    realtime_final_predictions.csv          ← Copy of RT final (pre-classifier)
    realtime_final_predictions_corrected.csv ← Copy of RT final (corrected)
    submission_ready.csv                    ← Merged DA+RT final (24 rows)
```

---

## Range Daily Run Directory

```
outputs/runs/range_{YYYY-MM-DD}_to_{YYYY-MM-DD}/
  range_manifest.json               ← Range-level manifest (all days)
  range_summary.csv                 ← CSV summary of all days in range
```

### `range_manifest.json`

```json
{
  "pipeline": "ledger_full_range",
  "start_date": "2026-02-24",
  "end_date": "2026-02-28",
  "total_days": 5,
  "completed_days": 5,
  "failed_days": 0,
  "skipped_days": 0,
  "status": "complete",
  "daily_results": [
    {
      "date": "2026-02-24",
      "status": "complete",
      "manifest_path": "outputs/runs/2026-02-24/run_manifest.json",
      "submission_ready_path": "outputs/runs/2026-02-24/final/submission_ready.csv",
      "warnings_count": 0,
      "errors_count": 0
    }
  ]
}
```

### `range_summary.csv`

```
date,status,submission_ready_exists,submission_ready_rows,errors_count,warnings_count,manifest_path,submission_ready_path
2026-02-24,complete,True,24,0,0,outputs/runs/2026-02-24/run_manifest.json,outputs/runs/2026-02-24/final/submission_ready.csv
2026-02-25,complete,True,24,0,0,outputs/runs/2026-02-25/run_manifest.json,outputs/runs/2026-02-25/final/submission_ready.csv
...
```

---

## File Naming Conventions

| Directory Name | Content | Notes |
|---------------|---------|-------|
| `prediction/` | Raw model predictions (per-model CSVs + long table) | Cache key for rerun |
| `weight/` | Learned fusion weights, trace, metrics | Regenerated each run |
| `fuse/` | Fused (weighted) predictions + debug info | **Not `fused/`** |
| `final/` | Final deliverables including `submission_ready.csv` | What gets submitted |

---

## Key Output Files

### `submission_ready.csv`

The final deliverable — 24 rows, one per hour:

```
business_day,ds,hour_business,period,dayahead_price,realtime_price
2026-02-24,2026-02-24 01:00:00,1,1_8,343.1948,344.6605
...
2026-02-24,2026-02-25 00:00:00,24,17_24,348.7663,334.345
```

- `hour_business`: 1..24 (hour 24 = D+1 00:00)
- `dayahead_price`: fused dayahead prediction
- `realtime_price`: post-classifier realtime prediction (may include -80.00 corrections)
- No `_x`/`_y` suffix columns

### `run_manifest.json`

Complete metadata for all pipeline stages including model status, row counts, warnings, errors, runtime config, and timestamps.

### `weights.csv`

Learned NNLSGEF weights per `(task, period, model)`:

```
task,period,model_name,weight
dayahead,1_8,lightgbm,0.108293
dayahead,1_8,timemixer,0.065361
dayahead,1_8,timesfm,0.826346
```

- Weights within a `(task, period)` sum to 1.0.
- Periods: `1_8`, `9_16`, `17_24`.

### `dynamic_weight_trace.csv`

Day-by-day evolution of learned weights across the 30-day training window:
- `age_days`: 1 (yesterday) to 30 (30 days ago)
- `day_gate`: learning rate per day (0.3-0.85)
- `loss`, `normalized_loss`: per-model per-day loss
- `weight_after`: weight after each day's update

The fusion stage additionally writes `model_quality_gate.csv`. Models whose
learned weight is below `--weight-prune-threshold` are excluded from that
task/period and the decision is recorded in `fused_debug.csv` and the run
manifest. Pass `--weight-prune-threshold 0` to disable pruning explicitly.

### `fused_predictions.csv`

Weighted fusion output:
```
task,business_day,ds,hour_business,period,y_fused
dayahead,2026-02-24,2026-02-24 01:00:00,1,1_8,343.1948
```

- `y_fused` = weighted sum of all model predictions for that hour.

### `classifier_report.json`

Classifier metadata:
```json
{
  "target_date": "2026-02-24",
  "method": "classifier_bridge",
  "success": true,
  "fallback_used": false,
  "n_corrections": 4,
  "corrected_hours": [
    {"hour_business": 5, "ds": "2026-02-24 05:00:00", "before": -63.9575, "after": -80.0},
    {"hour_business": 6, "ds": "2026-02-24 06:00:00", "before": -68.1796, "after": -80.0}
  ]
}
```
- `corrected_hours` is an array of objects, each with `hour_business`, `ds`, `before`, `after`.
- Empty list `[]` if no corrections were applied.

---

## Legacy / Debug Output Directories

These directories are **not formal outputs** of the ledger pipeline. They may appear on disk if you have run legacy pipelines, audit scripts, or debug tools:

| Directory | Created by | Description | Formal? |
|-----------|-----------|-------------|---------|
| `outputs/smoke/` | `ledger_smoke` | Smoke test outputs (lightweight) | Test only |
| `outputs/2026-02-01` (direct date) | Old staging runs | Legacy staging output | No |
| `outputs/unified_runs/` | Old pipeline (legacy) | Old unified output format | No |
| `outputs/repro_check/` | `scripts/check_reproducibility.py` | Reproducibility verification artifacts | No |
| `outputs/audit_30day_*` | `scripts/audit_30day_backfill.py` | 30-day backfill audit report | No |
| `outputs/RT916_SpikeMarketLab/` | RT916 model debug | RT916 daily joint debug output | No |

Only the selected profile's ledger/runs pair is part of that invocation's
ledger pipeline. `legacy` remains the production default; the
`feature_store` pair is candidate/shadow output until its full-chain gates pass.

---

## Caching Behavior

- **`prediction/`** is cached: if per-model CSVs exist, `ledger_predict` skips model inference (cache HIT).
- **`weight/`**, **`fuse/`**, **`classifier/`**, **`final/`** are not cached: each run regenerates them.
- Use `--force` on `ledger_full` or `ledger_predict` to clear prediction cache and force rerun.
