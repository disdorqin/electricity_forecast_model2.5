# Legacy & Archive Status

> **Status: Archive complete.** All legacy code has been moved to `_archive/`. The `ledger_*` pipeline family is the sole production path.

---

## Archive Summary

| Original path | Archived to | Date |
|---------------|-------------|------|
| `TimesFM/` (legacy wrapper) | `_archive/legacy_timesfm_wrapper/` | 2026-06 |
| `services/` | `_archive/legacy_staged_pipeline/services/` | 2026-06 |
| `pipelines/staged_pipeline.py` | `_archive/legacy_staged_pipeline/` | 2026-06 |
| `pipelines/fusion_pipeline.py` | `_archive/legacy_staged_pipeline/` | 2026-06 |
| `pipelines/predict_pipeline.py` | `_archive/legacy_staged_pipeline/` | 2026-06 |
| `pipelines/train_pipeline.py` | `_archive/legacy_staged_pipeline/` | 2026-06 |
| `fusion/` legacy experiment scripts (14 files + 17 runners) | `_archive/fusion_legacy/` | 2026-06 |
| `scripts/` dev scripts (5 files) | `_archive/dev_scripts/` | 2026-06 |

---

## What Changed

### `TF/` → `TimesFMBackend/`

The active TimesFM backend was renamed from `TF/` to `TimesFMBackend/` to eliminate confusion with TensorFlow. All internal and external imports were updated.

### `TimesFM/` → `_archive/legacy_timesfm_wrapper/`

The old 2.0 TimesFM wrapper (used only by the retired staged pipeline) is archived. It had its own `pipeline.py` and a duplicate `src/timesfm/` package.

### `services/` + `pipelines/staged_pipeline.py` → `_archive/legacy_staged_pipeline/`

The entire old staged pipeline (model_stage → learner_stage → fuse_stage → classifier_stage → full) and its service layer is archived. The `ledger_*` pipeline family (`ledger_full`) is the sole production path.

### `fusion/` legacy scripts → `_archive/fusion_legacy/`

Legacy fusion experiment scripts (meta_learner, repro_suite, all fusion runners) are archived.

**Active fusion files (NOT archived):**
- `fusion/learners/daily_ledger_gef.py` — BGEW weight learner (used by `ledger_weight`)
- `fusion/apply_daily_ledger_weights.py` — weight application (used by `ledger_fuse`)
- `fusion/classifier_bridge.py` — extreme price correction (used by `ledger_classifier`)
- `fusion/metrics.py` — shared metrics
- `fusion/registry.py` — fusion adapter registry
- `fusion/adapters/*.py` — per-model adapters
- `fusion/coverage_utils.py` — coverage reports
- `fusion/pipeline_common.py` — shared pipeline helpers
- `fusion/project_defaults.py` — defaults config

### `scripts/` dev scripts → `_archive/dev_scripts/`

Temporary/ad-hoc development scripts (SMOKE_RUN.md, _check_data.py, check_data_coverage.py, run_one_day_release.py, validate_release_output.py) are archived.

**Active scripts (NOT archived):**
- `scripts/env_check.py` — environment check
- `scripts/verify_final_pipeline.py` — output verification
- `scripts/verify_smoke.py` — smoke result verification
- `scripts/check_reproducibility.py` — reproducibility check
- `scripts/check_timemixer_alignment.py` — TimeMixer alignment check

---

## Remaining Cleanup (Not Critical)

| Item | Status | Priority |
|------|--------|----------|
| Legacy output dirs (`daily_runs/`, `outputs/unified_runs/`, `outputs/2026-02-01/`, `outputs/RT916_SpikeMarketLab/`) | Still present (gitignored) | Low — no code impact |
| `fusion/contracts.py`, `fusion/weights.py`, `fusion/run_fixed_window_fusion.py` | Still active, not imported by ledger pipeline | Low — no code impact |

These items are non-critical because they don't affect the `ledger_*` pipeline and are already covered by `.gitignore`.

---

## Restore Instructions

To restore any archived file:

```bash
git mv _archive/path/to/file original/path/to/file
```

Each archive subdirectory has its own context in `_archive/README.md`.
