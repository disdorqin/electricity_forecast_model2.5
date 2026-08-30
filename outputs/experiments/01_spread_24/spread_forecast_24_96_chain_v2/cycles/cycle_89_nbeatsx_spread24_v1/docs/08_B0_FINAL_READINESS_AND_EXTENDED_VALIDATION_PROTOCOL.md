---
status: active
date: 2026-08-29
scope: Cycle89 final B0 readiness repair + pre-registered extended-date MAE validation
purpose: close remaining reproducibility/preprocessing/reporting issues before full Jun+Jul B0, then obtain a non-cherry-picked multi-day signal against strict Cycle88 baselines
---

# Cycle89 B0 Final Readiness + Extended Validation Protocol

## 1. Why this document exists

Cycle89 has already passed the major structural gates: strict D-1 14:00 origin, H34 alignment, D-2 training cutoff, origin-safe CORE5 inference, train/validation separation, target-day OOS inference, holdout registry, checkpointing, official-source lock, and a first 3-day formal mini-backtest.

Independent review found that the program is close to a formal B0, but several remaining issues would make a full Jun+Jul run scientifically weaker than necessary if left unfixed:

1. model parameters are initialized **before** `seed_everything(42)` is called;
2. calendar sin/cos channels are documented as identity-scaled but are currently passed through RobustScaler;
3. config execution audit compares some configured thresholds to copied configured values instead of actual runtime facts;
4. aggregate OOS metrics can hide daily majority-class collapse;
5. run provenance is not yet strong enough to explain changing results across reruns;
6. business runner is hard-coded to CPU, which is inefficient for 61 daily cold retrains;
7. paper parity is correctly at SOURCE-EQUATION level, but business-adapted architecture must not be described as a full official block parity;
8. Git ignore exceptions make source discoverable, but `git status` still shows Cycle89 as untracked; this must not be reported as committed/tracked unless `git ls-files` proves it.

This protocol closes those issues **without tuning model performance**. After closure, it runs a pre-registered 14-day development panel to obtain a stronger preliminary comparison against existing strict Cycle88 LightGBM results.

---

## 2. Frozen scientific model for this round

No scientific model change is permitted during the readiness/extended-panel round.

```text
Target                 DA - RT
Forecast origin        D-1 14:00
Backcast               168 hourly target points
Horizon                34
Bridge                 offsets 1..10 = D-1 h15..h24
Scored                  offsets 11..34 = D h1..h24
Training label cutoff  <= D-2
Training history       rolling 9 months
Validation             latest 28 legal daily origins
Recalibration          daily cold retrain
Feature profile        CORE5 + calendar
Architecture           Identity -> Exogenous-TCN
Blocks                  [1, 1]
Hidden                  [256, 256]
TCN channels            8
Kernel                  3
Loss                    MAE only
Optimizer               Adam
LR                      5e-4
Batch                   32 daily origins
Clip norm               1.0
Precision               float32
Directional loss        forbidden in this round
B208                     forbidden in this round
Input336                 forbidden in this round
Specialist periods      forbidden in this round
```

The goal is to determine the behavior of the frozen MAE baseline, not to improve it while looking at development dates.

---

## 3. P0 repair — seed must cover model initialization

Current bad ordering:

```text
build_model()
seed_everything(42)
Trainer.fit()
```

This does not make the initial weights reproducible.

Required formal ordering:

```text
seed_everything(seed, deterministic=True)
build_model(config)
Trainer(...)
```

The formal runner must own the seed lifecycle; `Trainer.fit()` may assert/reseed dataloader/dropout state, but it must not be the first place where the seed is applied.

### Required reproducibility evidence

For one fixed target day (2026-06-01), run the full formal OOS procedure twice in two fresh processes on the same selected device and same source/config snapshot.

Required equality/evidence:

```text
initial_model_state_sha256  identical
split_manifest_sha256       identical
config_sha256               identical
best_step                   identical
checkpoint/model-state hash identical or documented serialization-only difference
24 point predictions        identical (preferred) or <= 1e-7 float32 tolerance
headline metrics            identical
```

If CUDA cannot satisfy deterministic execution for the required operations, formal B0 must fall back to CPU rather than silently accepting nondeterminism.

---

## 4. P0/P1 repair — calendar scaling contract

Current config says:

```text
CORE5 numeric     train-only robust median/IQR
calendar sin/cos identity
```

Current implementation scales all nine channels together. This changes the geometry of the cyclical representation.

Required behavior:

```text
channels 0..4  CORE5 numeric -> RobustScaler(train only)
channels 5..8  hour/dow sin/cos -> identity, remain in [-1, 1]
```

The scaler artifact must explicitly record which channel transformation is used.

Required tests:

- changing validation/target truth does not change numeric scaler;
- calendar channels are byte/numerically unchanged by transform;
- inverse/transform semantics for numeric channels are stable;
- no target-day label participates in scaler fitting.

Because preprocessing changes, all previous Cycle89 readiness OOS numbers are retained as historical evidence only and the 14-day panel must be rerun after this fix.

---

## 5. P1 repair — config execution audit must use runtime facts

Do not self-certify values.

Bad example:

```text
configured minimum train origins = 180
runtime_value = config[minimum] = 180
MATCH
```

Correct audit:

```text
field                    configured       actual runtime       relation    result
train_daily_origins      >=180            245                  >=          PASS
validation_daily_origins >=21             28                   >=          PASS
parameter_count          <=2,000,000      1,206,722            <=          PASS
batch_size               32               32                   ==          PASS
precision                float32          float32              ==          PASS
```

Every audit row must distinguish `constraint` from `actual_runtime_value`.

A formal run is invalid when a required execution row fails.

---

## 6. Device policy for full deep-learning backtests

Hard-coding CPU is not appropriate for a 61-day daily cold-retrain experiment.

Add an explicit runtime device policy without changing numeric precision:

```text
device_policy = cuda_if_deterministic_else_cpu
precision     = float32
AMP           = off
```

Gate:

1. detect CUDA;
2. enable deterministic algorithms;
3. run a deterministic forward/backward smoke;
4. run the same-target reproducibility test twice;
5. if deterministic equality/tolerance gate passes, freeze the entire extended panel and subsequent B0 to that CUDA device;
6. otherwise record blocker and use CPU for the entire panel.

Never mix CPU and GPU days inside one aggregate B0 result.

Record:

```text
device_type
GPU model if applicable
CUDA version
PyTorch version
deterministic_algorithms
precision
```

---

## 7. Gradient clipping interpretation

Current evidence:

```text
clip norm                 1.0
clip fraction             100%
median pre-clip norm      roughly 3-6
p90                       roughly 7-10
non-finite gradients      0
training loss             decreases
validation MAE            decreases
```

The official NBEATSx training code also applies global gradient clipping at norm 1.0. Therefore this is currently classified as:

```text
PERSISTENT_GRADIENT_CLIPPING_WARNING
```

not automatically as gradient explosion.

Do not change clip norm, LR, optimizer or architecture before B0. After seed/scaler fixes, rerun the 200-step convergence sanity. B0 may proceed with warning if:

- nonfinite count = 0;
- loss does not diverge;
- validation MAE has a decreasing/improving phase;
- gradient norm does not show persistent monotonic explosion.

---

## 8. Paper parity status must remain precise

Locked official source:

```text
repository: https://github.com/cchallu/nbeatsx
commit: fe116d21785fca55670d258756e7c35fcb613eca
```

Current valid claim:

```text
SOURCE-EQUATION PASS
```

because Identity/Trend/Seasonality basis and TCN mapped numerical checks are present.

Do **not** claim full official NBEATSx block parity or full EPF reproduction. The business model intentionally differs from the official EPF block in at least the following areas:

- explicit CORE5 business covariate path;
- local block MLP receives flattened backcast/future covariates;
- official `include_var_dict` EPF filtering semantics are not the business input path;
- full official training/search reproduction is not yet run.

Paper-faithful core and H34 business-adapted core must remain separately named.

Paper reproduction can continue later; it is not allowed to mutate the frozen B0 business model in this round.

---

## 9. Run provenance — required before extended panel

Every formal OOS run and aggregate panel must include enough information to explain a rerun.

Required `provenance.json` fields:

```text
run_id
target_day
canonical_data_path
canonical_data_sha256
canonical_data_mtime
config_path
config_sha256
Cycle89 source manifest/tree hash
selected official source commit
Python version
PyTorch version
device
CUDA version if applicable
seed
initial_model_state_sha256
train split sha256
validation split sha256
scaler sha256
checkpoint/model-state sha256
Git HEAD
Git dirty flag
Cycle89 source git status
```

Do not label Cycle89 source `git_tracked=true` unless `git ls-files` actually lists the source files. If it is merely unignored but untracked, report exactly:

```text
GIT_DISCOVERABLE_UNTRACKED
```

Do not stage or commit unrelated repository changes.

---

## 10. H34 output naming repair

Do not overload `forecast_offset`.

Use explicit columns:

```text
h34_offset          1..34
section             bridge | D-day
business_hour       bridge: 15..24, D-day: 1..24
```

For the scored target file:

```text
h34_offset = 11..34
business_hour = 1..24
```

This is required for later offset-performance research.

---

## 11. Extended validation panel — pre-registered dates

The panel is chosen by a fixed calendar rule, not by model/baseline performance.

### June 2026

```text
2026-06-01
2026-06-05
2026-06-10
2026-06-15
2026-06-20
2026-06-25
2026-06-30
```

### July 2026

```text
2026-07-01
2026-07-05
2026-07-10
2026-07-15
2026-07-20
2026-07-25
2026-07-31
```

Total:

```text
14 target days
336 scored D-day hours
```

These dates are frozen before running the repaired NBEATSx model. Do not remove a bad day or add a good day after seeing results.

This panel is a **development signal panel**, not the final Jun+Jul monthly B0 and not the future lockbox.

---

## 12. Same-date Cycle88 comparators

Primary strict comparator available across both June and July:

```text
Cycle88 LGBM v2 full-existing F0-F9
runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv
```

Reference monthly performance from the frozen artifact:

```text
2026-06: raw 57.08%, balanced 54.47%, MAE 91.74
2026-07: raw 60.35%, balanced 51.42%, MAE 84.08
```

Secondary comparator across the broader 2026 development range:

```text
runs/baseline_2026_01_01_2026_08_26_lgbm/predictions.csv
```

Additional June-only feature-pruned comparator:

```text
runs/trees2/confirm_pred.csv
config = trees_220
```

Do not compare headline metrics from mismatched date sets. For every comparator, subset **exactly the same 14 target days** before aggregation.

Required `baseline_comparison_daily.csv` columns:

```text
target_day
model
raw
positive_recall
negative_recall
balanced
MAE
actual_positive_rate
predicted_positive_rate
majority_baseline
raw_minus_majority
balanced_minus_0p5
```

Required paired deltas:

```text
NBEATSx - Cycle88 comparator
```

computed day-by-day before summarizing.

### 12.1 Pre-registered comparator snapshot on the exact 14-day panel

After the date panel above was frozen, the existing Cycle88 artifacts were subset to exactly those same 14 dates. These values are immutable comparison references for the repaired NBEATSx panel; they must not be used to alter the selected dates.

Primary comparator `cross_month_2026_01_08_14_lgbm_v2_full`:

```text
n scored hours             336
micro raw                  61.90%
micro positive recall      77.32%
micro negative recall      40.85%
micro balanced             59.08%
MAE                        84.74
daily-macro raw            61.90%
daily-macro balanced       56.79%
daily-macro MAE            84.74
```

Secondary comparator `baseline_2026_01_01_2026_08_26_lgbm`:

```text
n scored hours             336
micro raw                  57.74%
micro positive recall      48.45%
micro negative recall      70.42%
micro balanced             59.44%
MAE                        84.94
daily-macro raw            57.74%
daily-macro balanced       58.29%
daily-macro MAE            84.94
```

June-only `trees_220` B208 comparator on the seven pre-registered June dates:

```text
n scored hours             168
micro raw                  58.93%
micro positive recall      63.44%
micro negative recall      53.33%
micro balanced             58.39%
MAE                        95.15
daily-macro raw            58.93%
daily-macro balanced       47.10%
```

The two all-date LGBM comparators illustrate why both micro and daily-macro metrics are mandatory: one can rank higher on pooled raw while the other ranks higher on daily robustness.

---

## 13. Micro vs daily-macro metrics — both are mandatory

A pooled 72/336-hour metric can hide daily class collapse. Therefore report two aggregation families.

### 13.1 Micro hourly

Pool all scored hourly predictions in the panel and calculate:

```text
direction accuracy
positive recall
negative recall
balanced accuracy
MAE/RMSE
```

### 13.2 Macro daily

First calculate metrics separately for each day, then report:

```text
mean
median
std
p10
worst day
best day
```

for at least:

```text
raw
balanced
positive recall
negative recall
MAE
```

The primary robustness diagnostic is:

```text
DAILY_MACRO_BALANCED
```

not pooled balanced alone.

---

## 14. Majority-collapse and intraday-transition diagnostics

The first 3-day readiness run suggested a possible pattern: the model may identify day-level sign regime while failing to resolve intra-day sign transitions.

This is a hypothesis only and must not change B0 training.

Add diagnostics for every day:

```text
actual_positive_rate
predicted_positive_rate
all_positive_baseline
all_negative_baseline
majority_baseline
raw_minus_majority
positive_recall
negative_recall
minority_recall
actual_sign_switch_count
predicted_sign_switch_count
```

Flag:

```text
MAJORITY_COLLAPSE_DAY
```

when the model predicts only one sign (or effectively one sign) and obtains no meaningful minority recall.

Also output:

```text
majority_collapse_day_count
majority_collapse_day_rate
```

This diagnostic is especially important before designing future directional/offset-aware losses.

---

## 15. Offset diagnostics

Aggregate the pre-registered panel by true H34 offset:

```text
h34_offset 1..34
```

For each offset report:

```text
n
MAE
bias
raw direction
positive recall
negative recall
balanced
```

Do not create specialists yet. The output is evidence for later research into whether difficult regions align with:

- forecast distance;
- bridge vs D-day;
- solar ramp regions;
- sign-transition regions;
- repeated error clusters.

---

## 16. Extended panel interpretation gate

This 14-day panel is not a promotion decision and must not be used for hyperparameter tuning in the same run.

Classify only:

### STRONG_SIGNAL

- NBEATSx has a meaningful positive paired delta versus the primary Cycle88 comparator on both micro raw and daily-macro balanced;
- neither positive nor negative recall collapses systematically;
- majority-collapse rate is acceptably low;
- improvement is not driven by only 1-2 dates.

### MIXED_SIGNAL

- micro metrics improve but daily-macro balanced does not;
- or gains are driven by day-level majority regime;
- or one class recall remains unstable.

### WEAK_SIGNAL

- NBEATSx does not beat same-date strict comparator on key metrics;
- or majority-collapse dominates the panel;
- or performance is strongly date-fragile.

No architecture/loss change is permitted until the entire fixed panel is complete and summarized.

---

## 17. Required artifacts

At panel root:

```text
extended_panel_manifest.json
provenance.json
daily_metrics.csv
micro_hourly_metrics.json
macro_daily_metrics.json
monthly_partial_metrics.csv
baseline_comparison_daily.csv
baseline_comparison_summary.json
majority_collapse_diagnostics.csv
metric_by_h34_offset.csv
panel_predictions.csv
panel_readiness_review.md
```

Each target day keeps its independent run directory/checkpoint/split/audits.

`panel_readiness_review.md` must explicitly state that 14-day results are a fixed development panel and are not the full monthly B0 or final holdout.

---

## 18. Final gate before full Jun+Jul B0

Full monthly B0 may start only after:

```text
seed-before-model-init PASS
same-day two-process reproducibility PASS
calendar identity scaling PASS
runtime-fact config audit PASS
provenance PASS
H34 naming PASS
29+ all tests PASS
project preflight PASS
200-step convergence sanity PASS/WARN (no fatal instability)
14-day fixed panel complete
same-date Cycle88 comparison complete
micro + daily macro reporting complete
majority-collapse diagnostics complete
```

The 14-day model performance itself does not need to be excellent to make the software ready. If the signal is weak, preserve the result and then decide scientifically whether to run the full B0 or move to a controlled B1 hypothesis. Do not silently tune on panel results.

---

## 19. Deferred research registered from current evidence

Do not implement these in this round, but retain them for later controlled experiments:

1. pseudo-Huber magnitude loss;
2. balanced positive/negative differentiable sign loss;
3. raw direction surrogate;
4. bridge weight ablation;
5. L=336 and L=672 backcast;
6. 6m/12m training history;
7. LightGBM-selected engineered-feature compatibility profile;
8. offset-aware / transition-aware objective;
9. specialist regions learned from H34 error clusters rather than legacy 1-8/9-16/17-24 segmentation;
10. gap-fill H24 strategy versus direct H34 strategy.

The current working hypothesis to test later is:

> NBEATSx may learn coarse day-level sign regime more easily than intra-day sign-transition structure. Full B0 and offset diagnostics must establish whether this persists across many days before any loss or specialist is designed around it.

---

## 20. Codex execution checklist for this round

This section is the authoritative execution instruction for the next local Codex run. Do not reconstruct the task from chat history; execute this document as written.

### 20.1 Required startup

Before making changes, reread:

```text
AGENTS.md
.agents/skills/efm3-lessons/SKILL.md
outputs/experiments/01_spread_24/spread_forecast_24_96_chain_v2/AGENTS.md
cycle_89_nbeatsx_spread24_v1/AGENTS.md
cycle_89_nbeatsx_spread24_v1/README.md
cycle_89_nbeatsx_spread24_v1/experiment_manifest.json
cycle_89_nbeatsx_spread24_v1/docs/08_B0_FINAL_READINESS_AND_EXTENDED_VALIDATION_PROTOCOL.md
cycle_89_nbeatsx_spread24_v1/configs/business_strict34_core.json
cycle_89_nbeatsx_spread24_v1/configs/b0_extended_validation_panel.json
cycle_89_nbeatsx_spread24_v1/configs/holdout_registry.json
```

The scientific model is frozen for this round. No tuning or feature/model-family changes are allowed.

### 20.2 Repair order

Execute repairs in this order and stop if a hard gate fails:

1. seed before `build_model()` and full fresh-process reproducibility;
2. numeric-only robust scaling with calendar identity transform;
3. config execution audit based on real runtime values and relations;
4. provenance hashes and source/data/config/environment identity;
5. H34 offset naming semantics;
6. deterministic device gate (`cuda_if_deterministic_else_cpu`, float32 only);
7. rerun 200-step convergence sanity with fixed 300/600/900 LR milestones;
8. rerun 2026-06-01 twice in fresh processes and prove reproducibility;
9. only after all prior gates pass, run the fixed 14-day panel;
10. compare NBEATSx against the same-date Cycle88 frozen comparators;
11. produce micro, daily-macro, majority-collapse, sign-transition and H34-offset diagnostics;
12. update repository documents and readiness status without tuning the model.

### 20.3 Required new/updated tests

At minimum cover:

```text
test_seed_before_model_init.py
test_fresh_process_reproducibility.py
test_calendar_identity_scaling.py
test_numeric_only_robust_scaling.py
test_runtime_fact_config_audit.py
test_provenance_manifest.py
test_h34_offset_semantics.py
test_device_policy.py
test_same_device_determinism.py
test_daily_macro_aggregation.py
test_majority_collapse_detection.py
test_same_date_baseline_comparison.py
test_panel_dates_are_frozen.py
```

All existing tests must remain green. Run project preflight again before the extended panel.

### 20.4 Extended panel dates are immutable

Use exactly the 14 dates registered in `configs/b0_extended_validation_panel.json`. Do not add/remove/replace dates based on model behavior.

Every day must run as an independent cold retrain with its own 9-month rolling training window, 28-day chronological validation split, best checkpoint and strict target-day OOS inference.

### 20.5 Required comparison contract

Use exact same target days for every comparator. The primary comparator is:

```text
cycle_88_numeric_spread_da_minus_rt/
runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv
feature_package = full_existing_F0_F9
model = lgbm
```

Also retain the secondary baseline and June-only `trees_220` comparison described in Section 12.

Never compare metrics produced from different date sets.

### 20.6 Required panel metrics

For NBEATSx and each frozen comparator, report:

```text
micro hourly:
  raw direction
  positive recall
  negative recall
  balanced accuracy
  MAE
  RMSE

daily macro:
  raw mean/median/std/p10/worst/best
  balanced mean/median/std/p10/worst/best
  positive recall distribution
  negative recall distribution
  MAE distribution

paired daily deltas:
  delta_raw
  delta_balanced
  delta_MAE

regime diagnostics:
  actual_positive_rate
  predicted_positive_rate
  majority_baseline
  raw_minus_majority
  minority_recall
  actual_sign_switch_count
  predicted_sign_switch_count
  majority_collapse flag

H34 diagnostics:
  h34_offset 1..34
  section
  business_hour
  MAE
  bias
  raw
  positive recall
  negative recall
  balanced
```

The result must not be judged from pooled raw accuracy alone.

### 20.7 Required artifacts

The repaired 14-day run must create a dedicated run root such as:

```text
runs/b0_extended_panel_14d/
```

with the root artifacts listed in Section 17 and independent subdirectories for every target day. Every target-day run must contain provenance, split, scaler/config execution audit, training curve, gradient statistics, checkpoint hash, leakage audit and true 24-point D-day OOS prediction.

### 20.8 Interpretation rule

At the end classify the panel only as:

```text
STRONG_SIGNAL
MIXED_SIGNAL
WEAK_SIGNAL
```

Do not change the model in response to the result during this execution round.

A high raw score that merely equals the daily majority-class baseline is not model improvement. Daily-macro balanced accuracy and minority recall are mandatory safeguards.

### 20.9 Status update

After completion, update:

```text
README.md
experiment_manifest.json
docs/09_B0_EXTENDED_PANEL_RESULTS.md
```

The results document must include:

- repairs made and why;
- reproducibility evidence;
- selected device and deterministic evidence;
- test/preflight results;
- convergence warning status;
- all 14 daily NBEATSx metrics;
- same-date Cycle88 comparator metrics;
- paired deltas;
- micro and daily-macro summaries;
- majority-collapse/sign-transition diagnostics;
- H34 offset diagnostics;
- `STRONG_SIGNAL / MIXED_SIGNAL / WEAK_SIGNAL` conclusion;
- recommendation on whether to proceed to the full June+July MAE B0.

### 20.10 Hard prohibitions

During this round do not activate or search:

```text
directional loss
Pseudo-Huber as the headline B0 loss
B208 / selected engineered features
input length 336/672
6m/12m windows
new architecture families
hidden size / LR / batch search
period specialists
warm-start training
AMP
```

Do not delete poor dates. Do not rerun only favorable dates. Do not touch the registered future holdout.
