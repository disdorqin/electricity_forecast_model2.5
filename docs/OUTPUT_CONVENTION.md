# Output Convention

## Overview

The pipeline generates three categories of **formal outputs**:

1. **Persistent ledger** (`outputs/{24,96}/ledger/`) — cross-date accumulated prediction and actual value storage
2. **Daily run artifacts** (`outputs/{24,96}/runs/{date}/`) — per-date model predictions, weights, fused results, and final deliverables
3. **Range summary** (`outputs/{24,96}/runs/range_{start}_to_{end}/`) — multi-day range manifest and summary
4. **Runtime scratch** (`outputs/{24,96}/runtime/`) — transient inputs; successful runs clean these automatically

All outputs live under `outputs/` which is `.gitignore`d and never committed to Git.

## Production and compatibility output profiles

`production` is the default. Legacy and FeatureStore trees are retained only
for reproducibility/compatibility and must be selected explicitly.

| Profile | 24-point roots | 96-point roots | Status |
|---|---|---|---|
| `production` (default) | `outputs/ledger`, `outputs/runs` (phased migration pending) | `outputs/96/ledger`, `outputs/96/runs` | Current production state |
| `legacy` | `outputs/ledger/`, `outputs/runs/` | `outputs/ledger_96/`, `outputs/runs_96/` | Compatibility only |
| `feature_store` | `outputs/24/feature_store/ledger`, `outputs/24/feature_store/runs` | `outputs/96/feature_store/ledger`, `outputs/96/feature_store/runs` | Historical candidate/shadow chain |

FeatureStore cache artifacts for the candidate profile are stored under
`outputs/24/feature_store/cache/` or `outputs/96/feature_store/cache/`.
The former `outputs/feature_store_chain/` and `outputs/feature_store/` trees
are archived and are not write targets.

Select the candidate roots explicitly:

```powershell
python main.py --pipeline ledger_full --date YYYY-MM-DD `
  --resolution 15min --output-profile feature_store --feature-store-mode raw
```

96 点正式服务器预测使用 production roots；FeatureStore 物化不是生产必需条件。
24 点 legacy/advanced compatibility 仍可使用 `legacy` resource mode；formal 96
production 固定使用 `split_process`（CPU=2、DAG-aware；GPU=1 串行）。双子进程
不再只是候选：

```powershell
python main.py --pipeline ledger_predict --date YYYY-MM-DD `
  --resolution 15min --output-profile production `
  --feature-store-mode off --resource-mode split_process
```

`split_process` 对 96 点链路启用时：CPU 子进程和 GPU 子进程同时启动；CPU 使用
最多 2 个 **DAG ready-queue** worker，只有前置节点成功后才提交子节点，GPU 保持 1 个
worker 严格串行。调度身份是 `(model_name, task/internal_node)`，不能以模型名混淆 DA/RT。
只有 `scripts/server/benchmark_96_resource_modes.py` 的服务器 A/B
同时证明7模型完整、prediction/ledger等价、weight/fuse/final 等价（classifier 仅核对 policy marker）、无 OOM/明显模型互抢，
且达到显式 speedup 门槛后，才进入默认晋升复核。
生产 profile 当前使用原子 canonical ledger append；`parts/` 是显式 fragmented/兼容链的
可续跑存储形式，不是 production profile 的必然写法。

Explicit `--ledger-root` and `--runs-root` values override the profile.

Formal 96 manifests record `classifier_policy=disabled_by_production_policy`
and use `realtime/fuse/fused_predictions.csv` as the uncorrected RT final.
Classifier-corrected files are shadow/legacy outputs and are not consumed by
the formal 96 submission.

Canonical production tree during the phased migration:

```text
outputs/
  ledger/                  # validated 24-point ledger; migrate separately later
  runs/                    # validated 24-point daily runs
  24/{cache,runtime,sync}/ # new 24-point bounded auxiliary state
  96/{cache,ledger,runs,runtime,sync}/
  crawl/                   # crawler-owned runtime/reporting
  experiments/             # research/backtest artifacts only
  archive/                 # historical assets, never a production write target
```

Do not move the 24-point ledger merely for symmetry: the validated 24 chain still owns `outputs/ledger` + `outputs/runs`, while its auxiliary domain state may live under `outputs/24/`. This intentional asymmetry remains until a dedicated 24 migration verifies byte/row equivalence and operational parity.

For 96-point production, `runtime/` is scratch-only. A formal invocation first
syncs DB data, persists an immutable D/T snapshot under the daily run, and owns
exactly one `attempt_<date>_<attempt_id>/` sandbox containing the shared
FeatureView input and model scratch. NORMAL completion deletes the transient
view; the snapshot and provenance remain for cache/finish. A controlled failure
may retain that single sandbox for diagnosis until TTL cleanup. Tests that use
an isolated runs root therefore cannot pollute canonical runtime.

### Long-running retention policy

Production storage must remain bounded over years of daily operation:

| Asset | Retention | Reason |
|---|---|---|
| prediction / actual ledger | persistent + migratable | fusion training state and historical truth; formal96 daily runs append here and new servers warm-start from an audited migrated ledger |
| classifier incremental cache | persistent, bounded by cache policy | avoids historical recomputation |
| top-level `final/` outputs | persistent | business result/history |
| `run_manifest.json` + delivery report | persistent | audit/recovery record |
| successful run prediction/weight/fuse intermediates | short-term (target: 30 days) | reproducible from ledger + config; exact final weights + quality-gate rows are embedded in persistent `run_manifest.json.decision_snapshot` before these large intermediates are eligible for retention |
| normal logs | target 30 days | operations/debugging |
| failed-run diagnostics | target 90 days | longer incident investigation window |
| `runtime/` scratch | delete on success; stale TTL cleanup | never a durable dependency |
| `experiments/` | manual research archive policy | never a production dependency |

The automatic destructive retention job remains disabled until the production
server acceptance run verifies the new output roots. Existing historical
research trees must not be bulk-deleted or migrated implicitly. Before that
acceptance, `scripts/server/maintenance_96.py` is dry-run only: it can emit a
retention plan, while `--apply` is deliberately rejected. Even after age expiry,
a successful run's prediction/weight/fuse intermediates are **blocked** from the
retention candidate list unless the root manifest already contains a complete
DA+RT `decision_snapshot` (non-empty weights and model-quality gate records).

For every `ledger_full` run, `run_manifest.json` is the root audit record;
fusion additionally writes `model_quality_gate.csv` and `fused_debug.csv` per
task. A final normal delivery must contain 24 or 96 rows according to the
resolution and zero numeric NaN in `final/submission_ready.csv`.

### Formal vs. Non-formal Outputs

**Formal outputs (part of the production ledger pipeline):**

| Directory | Purpose |
|-----------|---------|
| `outputs/{24,96}/ledger/` | Cross-date accumulated prediction/actual ledger for weight learning; `outputs/96/ledger/` is a deployable production-state asset and may include an audited `bootstrap_manifest.json` |
| `outputs/{24,96}/runs/YYYY-MM-DD/` | Single-day output + final deliverable + audit manifest |
| `outputs/{24,96}/runs/range_*_to_*/` | Range manifest + summary |
| `outputs/{24,96}/cache/` | Reusable bounded caches such as classifier p1 state |
| `outputs/{24,96}/runtime/` | Scratch only; normal completion deletes large transient inputs |

**All other directories under `outputs/` are local-only legacy, experimental, or debug artifacts.** If present on your machine, they may include:

| Directory | Origin | Status |
|-----------|--------|--------|
| `outputs/2026-02-01` (direct date) | Old staging runs | Legacy — not part of ledger pipeline |
| `outputs/archive/legacy_96/unified_runs/` | EPF v2.0 unified output (moved from `outputs/unified_runs/` on 2026-09-19) | Archived — unused by ledger pipeline |
| `outputs/audit_30day_*/` | `scripts/audit_30day_backfill.py` | Debug/audit artifact |
| `outputs/repro_check/` | `scripts/check_reproducibility.py` | Debug/verification artifact |
| `outputs/archive/legacy_96/RT916_SpikeMarketLab/` | Historical RT916 self-managed output (moved from `outputs/RT916_SpikeMarketLab/` on 2026-09-19) | Archived/debug; formal runner redirects RT916 scratch into invocation runtime |

These trees are not required by the current `ledger_full` production path, but they must **not**
be bulk-deleted. The migration rule is always: stop new writes → verify no formal reader →
migrate/archive unique state → validate → only then delete a specific legacy tree.

### Local output asset map (2026-09-18 pre-acceptance snapshot)

This table is an audit snapshot, not a retention command. Sizes are approximate and will drift.

| Top-level tree | Size | Producer / current reader-writer | Rebuild / uniqueness | Production need | Target / delete condition |
|---|---:|---|---|---|---|
| `_diagnose_20260703/` | 0 MB | old diagnostic/manual; no formal code reference | empty | No | archive/remove only after acceptance; no action needed now |
| `24/` | 101.09 MB | domain-scoped 24 cache/runtime/sync plus historical FeatureStore state | mixed | Partly | keep `24/{cache,runtime,sync}`; handle old candidate state separately |
| `96/` | 42.50 MB after M1 cleanup | formal ledger/cache/final persistent; `runtime` transient | **Yes** | contains only `ledger,runs,cache,runtime,sync`; feature_store/legacy/runtime residue removed from production domain |
| `archive/` | historical/read-only evidence | may contain unique history | No runtime dependency | keep as archive; original server prediction evidence now lives under `archive/server_backtest_96/` with hashes |
| `cache/` | moved | legacy classifier cache | preserved read-only | No | archived to `archive/legacy_96/classifier_cache_legacy/`; formal cache is `96/cache/` |
| `crawl/` | 4.01 MB | crawler/export runtime | raw/report artifacts can be unique | Crawler-owned, not prediction model state | keep under `crawl/`; crawler policy owns retention |
| `diagnostics/` | 1.01 MB | historical diagnostics | mostly reproducible, some incident evidence | No | archive after server acceptance and reference check |
| `experiments/` | 17120.05 MB | research scripts only | research history often unique | **Never** production dependency | manual research archive policy only; never automatic cleanup |
| `ledger/` | 1.83 MB | validated 24-point production pipeline | persistent business/fusion state | **Yes (24)** | keep in place until a dedicated 24 migration proves equivalence |
| `ledger_96/` | 45.81 MB | legacy 96 tools/research only | historical/expensive; polluted-feature warning applies | **No** | formal96 and client deployment do not need it; archive after legacy reference inventory |
| `platform_review/` | 0.86 MB | reporting scripts | generally rebuildable | No | archive after reference check |
| `prediction_results/` | moved | historical export/report path | rebuildable | No | archived to `archive/historical_exports/prediction_results/` |
| `RT916_SpikeMarketLab/` | moved | historical RT916 core debug output | preserved read-only | No | archived to `archive/legacy_96/RT916_SpikeMarketLab/` |
| `runs_96/` | 0.18 MB | legacy 96 run profile/tools only | historical manifests/runs | **No** | formal96 and client deployment do not need it; archive with legacy research assets |
| `unified_runs/` | moved | legacy unified wrappers | rebuildable legacy output | No | archived to `archive/legacy_96/unified_runs/` |

The former `outputs/96/feature_store/` tree is no longer present in the formal production domain. On 2026-09-19 the complete `remote_20260101_20260814` server package moved intact to `outputs/archive/server_backtest_96/original_server_prediction_20251218_20260814/`; the remaining 180.15 MB candidate/cache/ledger/runs/smoke tree moved to `outputs/archive/legacy_96/feature_store_residual_20260919/`. The formal classifier cache remains under `outputs/96/cache/classifier/`. Explicit `feature_store` profile selection may recreate a compatibility candidate root, but formal96 does not write there.

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

24-point legacy/advanced compatibility runs the full five-stage pipeline:

```
  ledger_predict → ledger_weight → ledger_fuse → ledger_classifier → final_outputs
```

```
outputs/runs/{YYYY-MM-DD}/
  run_manifest.json                         ← 24 legacy run metadata (five stages)
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

### Formal 96-point production daily run

The `--96 DATE` façade uses a four-stage production chain.  The classifier
stage is recorded as `disabled_by_production_policy` for audit only and is not
consumed by final delivery:

```
ledger_predict → ledger_weight → ledger_fuse → final_outputs → postflight
```

```
outputs/96/runs/{YYYY-MM-DD}/
  run_manifest.json                         ← production_config + four-stage statuses + persistent decision_snapshot (exact weights + quality gate)
    ├── ledger_predict: DA 3 / RT 4, Dynamic-v1 snapshot/FeatureView and RT916 stride=24
    ├── ledger_weight: smape_reg/SLSQP, 30-day period weights
    ├── ledger_fuse: uncorrected DA/RT fused outputs + quality gates
    ├── ledger_classifier: disabled_by_production_policy
    └── final_outputs: 96-row submission_ready.csv (combined scope)

  snapshot/attempt_<id>/                    ← immutable D/T values.parquet + snapshot_manifest.json for each prediction attempt
  dayahead/prediction/                      ← 3 × 96 model rows
  realtime/prediction/                      ← 4 × 96 model rows
  dayahead/fuse/fused_predictions.csv       ← DA final source
  realtime/fuse/fused_predictions.csv       ← RT final source (uncorrected)
  final/submission_ready.csv                ← merged 96-row delivery
```

Formal 96 manifests must include `classifier_policy`, `production_config`, model pool, route-bound `serving_protocol`, `snapshot_id`, `dynamic_snapshot`, `feature_view` with `target_truth_mask=true`, `rt916_train_steps=24`, weight gate threshold, and `runtime_input_cleanup`. `LIVE_DYNAMIC` and stored LIVE replay use `formal96_dynamic_snapshot_v1`; Historical Proxy uses `formal96_historical_proxy_v1` plus p56/proxy-vintage metadata.
Legacy/FeatureStore trees remain compatibility or
experiment assets and are not production ledger sources.

For NORMAL formal96 delivery, `run_manifest.json` must also record `postflight.status=PASS`, `fallback.fallback_used=false`, and `next_day_readiness.mode=adaptive_complete_days`. The readiness block reports whether T+1 can select 30 complete days within the 90-day lookback with lag=2; it is not a contiguous-calendar-day requirement. Live prediction actual ledgers may contain only the already settled subset of target-day RT truth, while the seven prediction files and both fused/final products remain exact 96-slot artifacts. Failed/degraded attempts keep their attempt scratch for diagnosis; a later NORMAL run cleans only its own transient attempt and never deletes earlier diagnostic attempts implicitly.

---

## Range Daily Run Directory (profile-aware)

```
24 legacy: outputs/runs/range_{YYYY-MM-DD}_to_{YYYY-MM-DD}/
formal 96: outputs/96/runs/range_{YYYY-MM-DD}_to_{YYYY-MM-DD}/
  range_manifest.json               ← Range-level manifest (all days)
  range_summary.csv                 ← CSV summary of all days in range
```

### `range_manifest.json`

The JSON below is a 24-point legacy example; formal 96 uses the same schema under `outputs/96/runs/` and its four-stage policy.

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
- `realtime_price`: 24 legacy post-classifier realtime prediction (may include
  -80.00 corrections); formal 96 uses the uncorrected fused RT output because
  the classifier is disabled by production policy
- No `_x`/`_y` suffix columns

### `run_manifest.json`

Complete metadata for all pipeline stages including model status, row counts, warnings, errors, runtime config, and timestamps.

Formal 96 SGDFNet prediction rows use
`da_feature_source=sgdfnet_decision_day_da_anchor`; the manifest records the
D-1 anchor source day/type, `rows=96`, and `fallback_used`. The former
`sgdfnet_config_da_fill` label is historical only.
During `ledger_full`, child-stage manifests live under `outputs/96/runs/<date>/runtime/stage_manifests/`; they never overwrite the root `run_manifest.json`. They are resumable scratch only and are deleted after a NORMAL delivery. Same-day reruns move the immediately previous delivery into the single fixed slot `runtime/diagnostics/stale_delivery_previous/`; the next rerun overwrites that slot instead of creating another timestamped directory.

### `weights.csv`

Learned weights per `(task, period, model)`; formal 96 uses `smape_reg/SLSQP`,
while NNLSGEF remains an explicit legacy/experiment option:

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

This artifact is legacy/shadow/replay only. Formal 96 production does not run
ExtremePriceClf; its manifest value is `classifier_policy=
disabled_by_production_policy` and RT final consumes the uncorrected fused file.

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
| `outputs/archive/legacy_96/unified_runs/` | Old pipeline (legacy) | Archived old unified output format | No |
| `outputs/repro_check/` | `scripts/check_reproducibility.py` | Reproducibility verification artifacts | No |
| `outputs/audit_30day_*` | `scripts/audit_30day_backfill.py` | 30-day backfill audit report | No |
| `outputs/archive/legacy_96/RT916_SpikeMarketLab/` | RT916 model debug | Archived historical joint debug output | No |

Only the selected profile's ledger/runs pair is part of that invocation's ledger pipeline. `production` is the default. `legacy` and `feature_store` remain compatibility/history profiles and should not receive new production state.

---

## Caching Behavior

- **`prediction/`** is cached: if per-model CSVs exist, `ledger_predict` skips model inference (cache HIT).
- **`weight/`**, **`fuse/`**, **`classifier/`**, **`final/`** are not cached: each run regenerates them.
- Use `--force` on `ledger_full` or `ledger_predict` to clear prediction cache and force rerun.

### Formal96 snapshot route metadata (2026-09-20)

Persistent `outputs/96/runs/<target>/snapshot/<attempt>/` contains snapshot values plus `snapshot_manifest.json`; run/Stage1 manifests bind the exact `snapshot_id`, path and protocol. Route metadata distinguishes `LIVE_DYNAMIC`, `STORED_LIVE_SNAPSHOT_REPLAY`, and `HISTORICAL_PROXY_V1`; cache reuse across route/protocol/policy identities is not allowed. A stored replay is selected only from successful manifest-bound LIVE provenance, never from filesystem recency. Historical proxy snapshots carry `formal96_historical_proxy_v1`, `proxy_cutoff_period=56`, `UNVERIFIED_LEGACY_VINTAGE` and `strict_historical_vintage_proven=false`; current LIVE snapshots remain Dynamic-v1.

Successful canonical LIVE snapshots are **durable production history**, not 30-day scratch. Formal96 `--force` preserves the `snapshot/` subtree while clearing rebuildable run artifacts. FeatureView/model runtime remains transient and NORMAL delivery removes only the invocation-owned runtime attempt. Current snapshots are small (roughly tens of KB per target day), so daily canonical retention is the default.

`outputs/96/ledger/` is separately persistent/migratable state. Full-source import may merge the audited 2025-12-18..2026-08-14 server history through `bootstrap_96_production_ledger.py --history-scope full-source`; source rows never overwrite current-production keys. `bootstrap_manifest.json` records source/final ranges, imported days, overlap/conflict summary and promoted hashes.
