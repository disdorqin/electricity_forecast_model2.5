# Program Architecture — NBEATSx Spread24 v1

status: active  
date: 2026-08-29  
responsibility: concrete implementation layout and module contracts  
validation basis: official NBEATSx pipeline structure + EFM3 strict experiment rules

## 1. Design objective

This directory is a self-contained research program, not a wrapper around `NeuralForecast.NBEATSx`.

A high-level library may be used later as a cross-check baseline, but the primary implementation must expose and test the model internals required by the paper:

- basis functions;
- block/stack construction;
- double-residual mechanism;
- TCN/WaveNet exogenous encoder;
- decomposition;
- training loop;
- reference/paper hyperparameter profile;
- business-origin dataset construction;
- strict leakage audits;
- rolling backtest and reporting.

## 2. Target directory

```text
cycle_89_nbeatsx_spread24_v1/
├── AGENTS.md
├── README.md
├── experiment_manifest.json
├── configs/
│   ├── paper_repro_generic.json
│   ├── business_strict34_core.json
│   └── later/
│       ├── paper_repro_interpretable.json
│       ├── business_strict34_directional.json
│       ├── business_strict34_6m.json
│       ├── business_strict34_12m.json
│       ├── business_strict34_b208_compat.json
│       └── research_gap_fill.json
├── docs/
│   ├── 00_RESEARCH_DECISIONS.md
│   ├── 01_PAPER_REPRODUCTION_PROTOCOL.md
│   └── 02_PROGRAM_ARCHITECTURE.md
├── src/
│   └── nbeatsx_spread/
│       ├── __init__.py
│       ├── contracts.py
│       ├── config.py
│       ├── data/
│       │   ├── canonical_source.py
│       │   ├── origin_index.py
│       │   ├── paper_dataset.py
│       │   ├── business_dataset.py
│       │   ├── covariates.py
│       │   ├── normalization.py
│       │   └── collate.py
│       ├── model/
│       │   ├── basis.py
│       │   ├── tcn.py
│       │   ├── wavenet.py
│       │   ├── block.py
│       │   ├── nbeatsx.py
│       │   ├── initialization.py
│       │   └── factory.py
│       ├── losses/
│       │   ├── paper_mae.py
│       │   └── business_spread.py
│       ├── training/
│       │   ├── trainer.py
│       │   ├── early_stopping.py
│       │   ├── checkpoint.py
│       │   └── reproducibility.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── decomposition.py
│       │   ├── backtest.py
│       │   └── reporter.py
│       └── audits/
│           ├── origin.py
│           ├── covariate_availability.py
│           ├── label_cutoff.py
│           ├── counterfactual.py
│           └── holdout.py
├── scripts/
│   ├── preflight.py
│   ├── run_paper_repro.py
│   ├── run_business_backtest.py
│   ├── inspect_decomposition.py
│   └── audit_business_contract.py
├── tests/
│   ├── test_basis.py
│   ├── test_double_residual.py
│   ├── test_reference_parity.py
│   ├── test_origin_index.py
│   ├── test_horizon34_alignment.py
│   ├── test_feature_availability.py
│   ├── test_no_post14_leakage.py
│   ├── test_target_actual_counterfactual.py
│   ├── test_training_cutoff.py
│   └── test_metric_scope.py
├── third_party/
│   └── nbeatsx_source_manifest.json
└── runs/                       # generated artifacts only
```

## 3. Data model

### 3.1 One business sample

For target day D:

```text
origin = D-1 14:00

Y_backcast:
  168 observed hourly DA-RT spread values
  final point = D-1 h14

X_backcast:
  CORE5 forecast trajectories + calendar values
  aligned to the same 168 historical timestamps

X_future:
  34 forecast trajectories + calendar values
  D-1 h15..h24 + D h01..h24

Y_future_train:
  34 realized DA-RT labels for historical training examples only

Y_future_inference:
  unavailable
```

The dataset emits explicit masks:

```text
bridge_mask = [1]*10 + [0]*24
score_mask  = [0]*10 + [1]*24
```

These masks prevent the evaluator from accidentally including bridge points in headline D-day metrics.

### 3.2 Origin index

`origin_index.py` must generate timestamps using the repository resolution/business-day utility rather than raw row-count assumptions.

Each sample record stores:

```text
target_day
origin_timestamp
backcast_first_timestamp
backcast_last_timestamp
forecast_first_timestamp
forecast_last_timestamp
training_label_latest_day
```

Assertions:

```text
backcast_last == D-1 14:00
forecast contains exactly D-1 h15..h24 + D h1..h24
scored portion contains exactly D h1..h24
```

### 3.3 Training/validation split

Business runner:

- rolling history: 9 calendar months before the validation block;
- validation block: latest approximately 28 complete days whose labels are available by the origin;
- for each target D, every training/validation label must be complete by D-2;
- checkpoint selection is performed only on historical validation labels available at that target origin.

The implementation may cache daily-origin tensors, but cache fingerprints must include source hash, target sign, origin hour, feature profile, L, H and cutoff policy.

## 4. Covariate program

### 4.1 CORE5

The first model uses raw primitive forecast sequences:

```text
fcast_直调负荷
fcast_联络线受电负荷
fcast_风电总加
fcast_光伏总加
fcast_竞价空间
```

Calendar channels:

```text
hour_sin
hour_cos
dow_sin
dow_cos
```

So the temporal exogenous encoder initially sees 9 channels.

### 4.2 Why no explicit ramp/profile features in v1

The generic NBEATSx exogenous TCN/WaveNet receives time-indexed covariate trajectories. Its purpose is to learn temporal transformations of those variables. Adding manual ramp, min/max, mean/std and duplicated lag summaries in the first run would partially defeat the model-specific test.

### 4.3 Availability audit

`covariate_availability.py` must not merely test non-NaN values. It must maintain a registry entry per channel:

```text
source column
semantic role
publication/availability rule
allowed historical range
allowed future range
```

For all 34 future rows, the audit must state why the forecast was known by D-1 14:00. If source metadata cannot prove this, the run is non-promotable until the contract is resolved.

## 5. Model implementation

### 5.1 Bases

`basis.py`:

- `IdentityBasis`
- `TrendBasis`
- `SeasonalityBasis`
- exogenous basis interface

### 5.2 Exogenous encoder

Primary generic implementation:

- `TemporalConvNet` matching the paper/reference behavior;
- `WaveNet` variant matching the paper/reference behavior.

Both take covariates spanning backcast + forecast and generate a learned exogenous basis used by the block.

This is one of NBEATSx’s model-specific strengths and must remain visible in the program rather than being hidden behind a generic library call.

### 5.3 Blocks/stacks

`block.py` owns:

```text
MLP -> theta -> basis -> backcast + forecast
```

`nbeatsx.py` owns:

```text
stack ordering
double residual
forecast aggregation
optional decomposition output
```

Generic paper stack search:

```text
identity only
identity -> exogenous_tcn
exogenous_tcn -> identity
identity -> exogenous_wavenet
exogenous_wavenet -> identity
```

### 5.4 Decomposition

Even if the generic model wins on accuracy, every model run should optionally return per-block forecasts.

The interpretable profile additionally exposes:

```text
trend contribution
seasonality contribution
exogenous contribution
```

For the business task this can answer useful questions such as whether a negative spread window is being driven by renewable/bidding-space exogenous components rather than pure historical persistence.

## 6. Loss program

### 6.1 `paper_mae.py`

Exact MAE behavior used for paper reproduction and first business baseline.

### 6.2 `business_spread.py` — implemented but disabled in B34_MAE

Later controlled variant:

```text
L = w_mag * L_magnitude
  + w_dir * L_direction_surrogate
  + w_pos * L_positive_surrogate
  + w_neg * L_negative_surrogate
```

True direction accuracy and recalls are never differentiated directly.

Initial research proposal after B34_MAE:

```text
magnitude 0.70
direction 0.05
positive  0.125
negative  0.125
```

The bridge and target portions may use different weights, but this is not activated until the pure MAE baseline has been established.

## 7. Training program

### 7.1 Paper profile

Faithful profile:

- Adam
- StepLR gamma 0.5, three scheduled halvings
- gradient clip 1.0
- max 30k iterations
- eval every 100
- early-stop patience 10 validation checks
- MAE
- batch 256/512 search
- paper initialization/activation/normalization search space

### 7.2 Business profile

The architecture core remains identical. Business differences are explicit config fields.

Because a fixed D-1 14:00 origin yields roughly one supervised sample per historical day, the local 9-month set is much smaller than the multi-year paper dataset. Therefore business batch size must be tuned for sample count rather than blindly copying 256/512.

Initial business value:

```text
batch_size = 32
```

with 16/64 as later ablations if required.

The first business run uses the paper optimizer/scheduler semantics as far as practical so model behavior is isolated before introducing AdamW/cosine or other modern training changes.

## 8. Backtest program

`backtest.py` owns a strict daily replay:

1. choose target day D;
2. resolve D-1 14:00 origin;
3. assert all model-selection labels available by D-2;
4. materialize the 9-month train + historical validation set;
5. train/select checkpoint;
6. generate H=34 in one forward pass;
7. discard bridge points from headline score;
8. append D h1-h24 predictions to an experiment ledger;
9. write boundary audit.

No target-day label can alter the model, checkpoint, normalization or threshold used for that same target day.

## 9. Evaluation

Headline D-day metrics:

```text
direction_accuracy
positive_recall
negative_recall
balanced_accuracy
all_positive_baseline
all_negative_baseline
MAE
RMSE
```

Additional deep-model diagnostics:

```text
bridge_10h_MAE
bridge_10h_direction
horizon_error_by_step (1..34)
Dday_error_by_hour (1..24)
decomposition_by_step
parameter_count
train_steps
best_validation_step
GPU time
```

Period metrics 1-8 / 9-16 / 17-24 are diagnostic only in v1.

## 10. Execution gates

### M0 — skeleton + paper structural tests

Must pass before any expensive training.

### M1 — paper parity

Run R0/R1 tests, then reference EPF reproduction.

### M2 — business contract smoke

One historical target day, H=34, CORE5, MAE. Must pass all counterfactual leakage audits.

### M3 — June/July strict pilot

Only after M2. Compare B34_MAE against frozen Cycle 88 baselines under the same target sign/metric convention.

### M4 — controlled improvements

Only one change per experiment:

1. direction-aware loss;
2. 12m vs 9m;
3. 336h vs 168h;
4. CAUSAL_RICH features;
5. B208_COMPAT;
6. gap-fill strategy.

### M5 — cross-month confirmation / fresh holdout

Only frozen candidates proceed.
