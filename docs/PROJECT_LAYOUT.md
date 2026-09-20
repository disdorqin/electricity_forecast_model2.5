# Project Layout

## Root Directory Reference

| Folder | Role | Used by ledger pipeline | Commit policy | Recommended action |
|--------|------|------------------------|---------------|-------------------|
| `cli/` | CLI argument parser; main.py entry point | **Yes** | commit | KEEP |
| `pipelines/` | Pipeline orchestration: formal 96 `ledger_predict → ledger_weight → ledger_fuse → final_outputs` plus legacy/24 `ledger_classifier` compatibility | **Yes** | commit | KEEP |
| `runners/` | Model registry + EPF v1 adapters (lightgbm_v1, timesfm_v1) | **Yes** | commit | KEEP |
| `runtime/` | CPU/GPU resource scheduler; 96 split mode uses CPU DAG-ready queue (2 workers) + GPU serial queue (1 worker) | **Yes** | commit | KEEP |
| `fusion/` | Fusion core: NNLSGEF/BGEW learners, weight gate/application, classifier bridge, per-model adapters, metrics, legacy experiment scripts | **Yes** (select files) | commit | KEEP (needs cleanup) |
| `lightGBM/` | LightGBM model pipeline (standalone) | **Yes** | commit | KEEP |
| `TimesFMBackend/` | **Active TimesFM prediction engine** (EPF v1 backend, NOT TensorFlow). Contains full timesfm_2p5 PyTorch+Flax implementation. Renamed from `TF/` to avoid TensorFlow confusion | **Yes** (via runners/adapters/timesfm_v1.py) | commit | KEEP |
| `TimeMixer/` | TimeMixer model pipeline (standalone, GPU) | **Yes** | commit | KEEP |
| `RT916_SpikeFusionNet/` | RT916 (SpikeFusionNet) model pipeline (standalone, GPU) | **Yes** | commit | KEEP |
| `SGDFNet/` | SGDFNet model pipeline (standalone, CPU) | **Yes** | commit | KEEP |
| `ExtremPriceClf/` | Extreme price classifier retained for legacy/shadow replay; formal 96 production bypasses it | **No** (formal 96); legacy/shadow only | commit | KEEP |
| `utils/` | Shared utilities: business_day.py, reproducibility.py, io.py | **Yes** | commit | KEEP |
| `scripts/` | 工具与测试：crawler/ 爬虫、sync/ 数据同步、tests/ 回归验证、env_check 等 | No (tooling) | commit | KEEP |
| `docs/` | Project documentation | No | commit | KEEP |
| `data/` | Local input data (Excel/CSV) | Yes (model input) | **ignore** | KEEP |
| `outputs/` | All pipeline run artifacts: ledger storage, daily runs, smoke, repro check | No (generated) | **ignore** | KEEP |

## Documentation ownership

`docs/README.md` 是索引，`docs/DOCUMENT_ARCHITECTURE.md` 是文档职责规则。当前长期维护文档按以下边界分工：

| 文档 | 唯一职责 |
|---|---|
| `README.md` | 项目入口与用户可见状态 |
| `RUNBOOK.md` | 运行、同步、范围回测、部署和回归 |
| `DATA_CONTRACT_96.md` | 24/96 数据、时间、质量和字段契约 |
| `LEAKAGE_AUDIT_96.md` | 信息可得性、cutoff 和防泄漏 |
| `OUTPUT_CONVENTION.md` | ledger、runs、submission、manifest |
| `PROJECT_GOVERNANCE.md` | 变更、复现、质量门、回滚 |
| `DOCUMENT_ARCHITECTURE.md` | 文档新增、归档、分支和 AI 阅读规则 |
| `SERVER_96_DEPLOYMENT_BACKFILL.md` | 新服务器部署、full-source ledger 合并、2026-08-17 起历史接续、resume 与每日生产切换 |
| `models/` | Pre-trained model weight caches (~885 MB) | No (model weights) | **ignore** | KEEP |
| `_archive/` | Legacy code preserved for traceability: legacy_timesfm_wrapper, legacy_staged_pipeline, fusion_legacy, dev_scripts | No | commit | KEEP |
| `optim/` | Training performance knobs (TF32, AMP, DataLoader) | Partial (imported by TimeMixer/RT916) | commit | KEEP |
| `.claude/` | Claude Code memory/persistence (local tooling) | No | **ignore** | KEEP |
| `.workbuddy/` | Workbuddy workspace data (local tooling) | No | **ignore** | KEEP |

## Pipeline Architecture

```
main.py
  └─ cli/parser.py              ← argument parsing
  └─ pipelines/
       ├─ ledger_backfill.py    ← 30-day historical backfill
       ├─ ledger_predict.py     ← run all models, append to ledger
       │   ├─ runners/registry.py              → direct model pipeline
       │   │   ├─ TimeMixer.pipeline           ← TimeMixer/
       │   │   ├─ RT916_SpikeFusionNet.pipeline ← RT916_SpikeFusionNet/
       │   │   ├─ SGDFNet.pipeline             ← SGDFNet/
       │   │   # Note: TimesFM removed from registry — ledger uses adapter directly
       │   ├─ runners/adapters/timesfm_v1.py  → TimesFMBackend/infer.py (active TimesFM)
       │   ├─ runners/adapters/lightgbm_v1.py → lightGBM/ (EPF v1)
       │   └─ runtime/resource_scheduler.py   ← CPU/GPU queuing
       ├─ ledger_weight.py
       │   └─ fusion/learners/daily_ledger_gef.py  ← NNLSGEF/BGEW weight learners
       ├─ ledger_fuse.py
       │   └─ fusion/apply_daily_ledger_weights.py ← weight application
       ├─ ledger_classifier.py
       │   └─ fusion/classifier_bridge.py           ← legacy/shadow extreme-price correction
       ├─ ledger_full.py       ← formal 96 stages 1-4; legacy/24 stages 1-5
       ├─ prediction_ledger.py ← ledger append/dedup/query
       └─ ledger_smoke.py      ← smoke test wrapper
```

## TimesFMBackend/ (formerly TF/)

**Current state:** The `TF/` directory has been renamed to `TimesFMBackend/` to eliminate the misleading `TF/` name (which was never TensorFlow). The old legacy `TimesFM/` wrapper has been archived to `_archive/legacy_timesfm_wrapper/`.

| Directory | Role | Used by ledger_full |
|-----------|------|---------------------|
| `TimesFMBackend/` | **Active TimesFM backend.** Contains full `src/timesfm/` package (PyTorch + Flax). Entry: `TimesFMBackend.infer.predict_price_for_date()` | **Yes** (via `runners/adapters/timesfm_v1.py`) |

**History:** `TimesFM/` was the original 2.0 wrapper. `TF/` was added as the EPF v1.0 backend. Both contained duplicate `src/timesfm/` copies. After the staged pipeline was retired, `TF/` was renamed to `TimesFMBackend/` and `TimesFM/` was archived.

## Fusion Module Status

**Active files (used by ledger pipeline):**

| File | Used by |
|------|---------|
| `fusion/learners/daily_ledger_gef.py` | `ledger_weight` — formal default `smape_reg/SLSQP`; NNLS/BGEW are explicit internal/experiment learners |
| `fusion/apply_daily_ledger_weights.py` | `ledger_fuse` — weight application |
| `fusion/classifier_bridge.py` | `ledger_classifier` — legacy/shadow extreme-price correction; disabled by policy in formal 96 |
| `fusion/metrics.py` | Shared metrics (imported by learner and weights) |
| `fusion/contracts.py` | Staged pipeline (not ledger) |
| `fusion/weights.py` | Staged pipeline (not ledger) |
| `fusion/run_fixed_window_fusion.py` | Staged pipeline (not ledger) |
| `fusion/adapters/*.py` | Fusion adapter layer (per-model csv long table) |
| `fusion/registry.py` | Fusion model registry |
| `fusion/coverage_utils.py` | Coverage reports |
| `fusion/pipeline_common.py` | Shared pipeline helpers |
| `fusion/project_defaults.py` | Defaults config |

**Legacy files (archived to `_archive/fusion_legacy/`):**

All legacy fusion experiment runners and scripts have been moved to `_archive/fusion_legacy/`:

- `fusion/run_pipeline.py`, `fusion/run_dayahead_pipeline.py`, `fusion/run_realtime_pipeline.py`
- `fusion/run_fit.py`, `fusion/run_end_to_end_fixed_fusion.py`, `fusion/run_final_fusion_pipeline.py`
- `fusion/run_rolling_backtest.py`, `fusion/run_full_fusion_suite.py`
- `fusion/run_repro_training_length_suite.py`, `fusion/prepare_history_outputs.py`
- `fusion/prepare_manifest.py`, `fusion/repro_suite.py`, `fusion/meta_learner.py`
- `fusion/manifest_template.csv`
- `fusion/runners/*.py` (all 17 files, various experiment runners)

These files are preserved for traceability but not imported by any active pipeline.

## Output Structure

See [`OUTPUT_CONVENTION.md`](OUTPUT_CONVENTION.md) for full details.

Key paths:
- `outputs/ledger/{task}/prediction/prediction_ledger.parquet` — current validated **24-point** persistent prediction ledger
- `outputs/ledger/{task}/actual/actual_ledger.parquet` — current validated **24-point** actual ledger
- `outputs/runs/{date}/...` — current 24-point daily run metadata/final delivery. The 24-point layout intentionally remains different from formal96 and is not migrated merely for symmetry; a later dedicated 24 migration must prove equivalence first.
- `outputs/96/ledger/{task}/{prediction,actual}/...` — formal 96 persistent production state; weight learner reads the rolling 30-day history here, daily prediction/actual append here, and server migration preserves this tree plus `bootstrap_manifest.json`
- `outputs/96/runs/{date}/...` — formal 96 run metadata/final delivery; root manifest is owned by `ledger_full`. Same-day reruns keep only one `runtime/diagnostics/stale_delivery_previous/` rollback slot instead of creating unlimited timestamped backups.
- `outputs/96/runtime/attempt_<date>_<attempt>/...` — formal96 invocation-owned
  FeatureView/model scratch after DB sync and immutable snapshot creation.
  `runs_root` isolation automatically isolates this sibling runtime; NORMAL
  delivery removes transient scratch while the attempt-scoped daily snapshot/provenance under `runs/{date}/snapshot/attempt_<id>/` remains; same-day reruns never overwrite an older successful Stage1 snapshot.
- `outputs/24/feature_store/{cache,ledger,runs}/` — 24-point candidate chain
- `outputs/96/` — formal96 production domain now contains **only** `ledger/`, `runs/`, `cache/`, `runtime/`, `sync/`. There is no persistent production `feature_store/` or `legacy/` subdirectory.
- `outputs/96/feature_store/...` — compatibility profile path only; currently absent on disk and created only by explicit `--output-profile feature_store` use.
- `outputs/archive/server_backtest_96/original_server_prediction_20251218_20260814/` — read-only original server prediction/backtest evidence, 240 complete DA+RT days, with `SERVER_EVIDENCE_MANIFEST.json` + `SHA256SUMS.json`.
- `outputs/archive/legacy_96/feature_store_residual_20260919/` — archived historical candidate cache/ledger/runs/smoke tree formerly under `outputs/96/feature_store/`.
- `outputs/archive/legacy_96/{unified_runs,RT916_SpikeMarketLab,classifier_cache_legacy,formal96_legacy_dir}/` — retired legacy output roots; formal96 has no readers/writers there.
- `outputs/archive/legacy_sync/` — moved legacy sync/cache roots, read-only reference

Data domains are equally explicit: `data/24/canonical/` is the hourly source;
`data/96/authoritative/pmos_96_全量.csv` is the faithful synchronized DB business-column mirror (including forecast/partial tail);
`data/96/model_input/` is the price/forecast model wide table; and
`data/96/remote/` is the native database mirror.

The default `legacy` profile keeps the existing `outputs/ledger*` and
`outputs/runs*` locations. The `feature_store` profile is selected with
`--output-profile feature_store` and must not share a ledger with the legacy
profile.

## Dynamic-v1 production serving architecture

Formal96 has one visibility authority rather than five model-local cutoff policies. The formal façade first syncs the DB (except `--finish`) and resolves one of three Snapshot sources:

```text
main.py --96 T
  -> DB sync -> latest_closed_day
  -> Snapshot route
       A. successful canonical LIVE snapshot exists -> STORED_LIVE_SNAPSHOT_REPLAY
       B. closed historical day, no LIVE snapshot    -> HISTORICAL_PROXY_V1 (p56)
       C. current/live target                         -> LIVE_DYNAMIC
  -> immutable runs/T/snapshot/attempt_<id>/
  -> FeatureViewBuilder transient input.parquet
  -> model adapters (DA3 / RT4)
  -> prediction ledger -> weight -> fuse -> final
```

Route A is selected only from success provenance bound by the run/Stage1 manifest, never by choosing the newest directory. Route B uses historical final actual/RT only through p56 as an operational proxy; target truth stays masked and the tail uses the same FeatureView fallback as LIVE. Route C uses every cell actually present at the live forecast origin. All three routes keep `dynamic_serving=true`; model-local 14/15-hour knobs are training/legacy compatibility only.

The persistent `data/96/model_input/...full.parquet` remains the training/history model store; the Snapshot/FeatureView layer is serving-only. Successful LIVE canonical snapshots under `outputs/96/runs/<T>/snapshot/` are durable replay assets and are preserved even by formal96 `--force`; at current size they are only tens of KB per day. `outputs/96/runtime/attempt_*` owns transient FeatureView/model scratch and is removed after NORMAL delivery; failed/degraded attempts may remain temporarily for diagnosis/TTL. On Windows, SGDFNet Dynamic scratch intentionally uses a short experiment/run suffix so the deep production attempt path does not exceed path limits; this changes scratch naming only, not the model or D→T DA anchor contract.

`--finish` is a provenance replay of the exact successful Stage1: it never syncs or creates a new snapshot and accepts `complete_with_warnings` only when the strict per-model/snapshot validation passes.

### Minimal predictor release boundary

甲方 predictor 由 `scripts/server/build_predictor_release.py` 的白名单生成，而不是复制整个开发仓。release 只包含 application dirs、`models/LightGBM`、`models/timesFM`、必要 sync/server scripts 和 active production docs；明确排除 `data/`、`outputs/`、crawler、experiments、tests、build、Agent tooling、ExtremePriceClf、`.env`/secret config。mutable `outputs/96/ledger` 由 `bootstrap_96_production_ledger.py` 单独迁入，数据由 DB sync 重建。`doctor_96_deployment.py --strict-release` 校验 release manifest、逐文件 SHA256、production imports、CUDA、TimesFM bundled checkpoint 路径、DB/runtime/ledger readiness。2026-09-20 已在系统临时全新目录完成 clean deployment + DB sync + `main.py --96` NORMAL 验收。

## Production model pool

The only production candidate source is `fusion/model_pool.py`:

| Task | Candidate models | Fusion meaning |
|---|---|---|
| Dayahead | `lightgbm`, `timesfm`, `timemixer` | All three enter DA fusion |
| Realtime | `timesfm`, `sgdfnet`, `timemixer`, `rt916` | All four enter RT fusion |

LightGBM realtime is disabled for production. Historical/experimental
adapters are not candidates unless added to this canonical pool and passed
through the resolution and leakage checks.

## Git Policy

| Pattern | Tracked | Committed | Notes |
|---------|---------|-----------|-------|
| `data/` | Now untracked | Was tracked (historical) | `git rm --cached` applied |
| `outputs/` | Never tracked | No | `.gitignore` |
| `models/` | Never tracked | No | `.gitignore` |
| `.claude/` | Never tracked | No | `.gitignore` (added) |
| `.workbuddy/` | Never tracked | No | `.gitignore` (added) |
| All source code | Yes | Yes | Core project code |
| `*.xlsx` (data) | Now untracked | Was tracked (historical) | `.gitignore` + index removed |

### Formal96 canonical snapshots (2026-09-20)

Successful LIVE snapshot values/manifests are retained below `outputs/96/runs/<target>/snapshot/<attempt>/` and referenced by immutable Stage1 provenance. FeatureView scratch remains transient; historical proxy snapshots are explicit operational replay artifacts, never strict publication-vintage evidence.
