# Runbook

## Prerequisites

- Conda environment `epf-2` with dependencies installed (`pip install -r requirements.txt`)
- Input data file at `data/shandong_pmos_hourly.xlsx`
- CUDA-capable GPU for TimeMixer and RT916 models

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
    --data-path data/shandong_pmos_hourly.xlsx ^
    --smoke-training-months 3 --smoke-timemixer-epochs 3 --smoke-timemixer-patience 1 ^
    --max-cpu-workers 2 --max-gpu-workers 1 ^
    --seed 42 --deterministic --force
```

Expected: ~10-15 minutes. All 7 models produce 24-hour predictions.

## 3. 30-Day Backfill

Populate the prediction ledger with historical data:

```powershell
conda run -n epf-2 python main.py --pipeline ledger_backfill ^
    --start YYYY-MM-DD --end YYYY-MM-DD ^
    --data-path data/shandong_pmos_hourly.xlsx ^
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
    --data-path data/shandong_pmos_hourly.xlsx ^
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

## 9. Force Re-run

To force a full rerun (bypass prediction cache):

```powershell
conda run -n epf-2 python main.py --pipeline ledger_full --date YYYY-MM-DD ^
    --data-path data/shandong_pmos_hourly.xlsx --force
```

To force a specific stage, add `--force` after the cleaned outputs.

## Common CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--date` | — | Target date YYYY-MM-DD |
| `--start` / `--end` | — | Date range for backfill |
| `--epf-v1-root` | None | [Optional legacy compatibility] External EPF v1.0 root |
| `--seed` | 42 | Global random seed |
| `--deterministic` | False | Enable deterministic algorithms |
| `--max-cpu-workers` | 2 | CPU parallel workers |
| `--max-gpu-workers` | 1 | GPU serial workers |
| `--realtime-cutoff-hour` | 14 | Realtime cutoff hour on D-1 |
| `--force` | False | Force rerun, bypass cache |
| `--data-path` | `data/shandong_pmos_hourly.xlsx` | Input data file path |
| `--ledger-root` | `outputs/ledger` | Ledger storage root |
| `--runs-root` | `outputs/runs` | Daily run output root |

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
