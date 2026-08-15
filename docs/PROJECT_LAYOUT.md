# Project Layout

## Root Directory Reference

| Folder | Role | Used by ledger pipeline | Commit policy | Recommended action |
|--------|------|------------------------|---------------|-------------------|
| `cli/` | CLI argument parser; main.py entry point | **Yes** | commit | KEEP |
| `pipelines/` | Pipeline orchestration: ledger_predict, ledger_weight, ledger_fuse, ledger_classifier, ledger_full, ledger_backfill, ledger_smoke, prediction_ledger | **Yes** | commit | KEEP |
| `runners/` | Model registry + EPF v1 adapters (lightgbm_v1, timesfm_v1) | **Yes** | commit | KEEP |
| `runtime/` | CPU/GPU resource scheduler for concurrent model execution | **Yes** | commit | KEEP |
| `fusion/` | Fusion core: BGEW learner, weight application, classifier bridge, per-model adapters, metrics, legacy experiment scripts | **Yes** (select files) | commit | KEEP (needs cleanup) |
| `lightGBM/` | LightGBM model pipeline (standalone) | **Yes** | commit | KEEP |
| `TimesFMBackend/` | **Active TimesFM prediction engine** (EPF v1 backend, NOT TensorFlow). Contains full timesfm_2p5 PyTorch+Flax implementation. Renamed from `TF/` to avoid TensorFlow confusion | **Yes** (via runners/adapters/timesfm_v1.py) | commit | KEEP |
| `TimeMixer/` | TimeMixer model pipeline (standalone, GPU) | **Yes** | commit | KEEP |
| `RT916_SpikeFusionNet/` | RT916 (SpikeFusionNet) model pipeline (standalone, GPU) | **Yes** | commit | KEEP |
| `SGDFNet/` | SGDFNet model pipeline (standalone, CPU) | **Yes** | commit | KEEP |
| `ExtremPriceClf/` | Extreme price classifier; used by fusion/classifier_bridge.py | **Yes** (via classifier_bridge) | commit | KEEP |
| `utils/` | Shared utilities: business_day.py, reproducibility.py, io.py | **Yes** | commit | KEEP |
| `scripts/` | 工具与测试：crawler/ 爬虫、sync/ 数据同步、tests/ 回归验证、env_check 等 | No (tooling) | commit | KEEP |
| `docs/` | Project documentation | No | commit | KEEP |
| `data/` | Local input data (Excel/CSV) | Yes (model input) | **ignore** | KEEP |
| `outputs/` | All pipeline run artifacts: ledger storage, daily runs, smoke, repro check | No (generated) | **ignore** | KEEP |
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
       │   └─ fusion/learners/daily_ledger_gef.py  ← BGEW weight learner
       ├─ ledger_fuse.py
       │   └─ fusion/apply_daily_ledger_weights.py ← weight application
       ├─ ledger_classifier.py
       │   └─ fusion/classifier_bridge.py           ← extreme price correction
       ├─ ledger_full.py       ← orchestrate stages 1-5
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
| `fusion/learners/daily_ledger_gef.py` | `ledger_weight` — BGEW weight learner |
| `fusion/apply_daily_ledger_weights.py` | `ledger_fuse` — weight application |
| `fusion/classifier_bridge.py` | `ledger_classifier` — extreme price correction |
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
- `outputs/ledger/{task}/prediction/prediction_ledger.parquet` — persistent prediction ledger
- `outputs/ledger/{task}/actual/actual_ledger.parquet` — persistent actual ledger
- `outputs/runs/{date}/run_manifest.json` — full run metadata
- `outputs/runs/{date}/final/submission_ready.csv` — final deliverable

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
