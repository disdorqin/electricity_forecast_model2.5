# NBEATSx Paper Reproduction Protocol

status: active  
date: 2026-08-29  
responsibility: define what “strict paper reproduction” means before local adaptation  
validation basis: IJF/arXiv paper and official `cchallu/nbeatsx` implementation

## 1. Reference

Paper:

- Kin G. Olivares, Cristian Challu, Grzegorz Marcjasz, Rafał Weron, Artur Dubrawski
- *Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx*
- International Journal of Forecasting 39(2), 2023, 884-900
- DOI: `10.1016/j.ijforecast.2022.03.001`
- arXiv: `2104.05522`

Official source:

- `https://github.com/cchallu/nbeatsx`
- MIT license

Official README reproduction command for the Nord Pool generic NBEATSx experiment:

```text
python src/hyperopt_nbeatsx.py --dataset 'NP' --space "nbeats_x" --data_augmentation 0 --random_validation 0 --n_val_weeks 52 --hyperopt_iters 1500 --experiment_id "nbeatsx_0_0"
```

## 2. Architecture parity requirements

### 2.1 Core N-BEATS double residual

Each block returns a backcast and forecast. The global forward pass must preserve:

```text
residual_0 = reverse(y_backcast)
forecast_0 = last_observed_level

for block in blocks:
    backcast_b, forecast_b = block(residual, X)
    residual = (residual - backcast_b) * mask
    forecast = forecast + forecast_b
```

Tests must verify:

- correct residual subtraction;
- additive forecast aggregation;
- level/naive initialization behavior;
- identical shapes and masking behavior to the reference formulation.

### 2.2 Generic NBEATSx-G

The published official search space includes:

```text
[identity]
[identity, exogenous_wavenet]
[exogenous_wavenet, identity]
[identity, exogenous_tcn]
[exogenous_tcn, identity]
```

For two-stack variants:

```text
n_blocks = [1, 1]
n_layers = [2, 2]
hidden units per stack = 50..500
```

The v1 codebase must implement the paper-style TCN and WaveNet exogenous bases. A modern convenience API that only projects future covariates linearly is not considered an exact reproduction of the generic paper search space.

### 2.3 Interpretable NBEATSx-I

Supported reference stack forms include trend, seasonality and exogenous convolutional components. The interpretable mode must expose forecast decomposition by block/stack so the effect of trend, seasonality and exogenous factors can be inspected.

Reference search values:

```text
trend polynomial degree: 2, 3, 4
seasonality harmonics: 1, 2
```

### 2.4 Forecast problem

Published EPF comparison:

```text
L = 168 hourly backcast
H = 24 hourly forecast
```

The exogenous matrix covers both backcast and forecast periods. Typical markets use day-ahead load/wind/solar/generation forecasts and calendar variables.

## 3. Paper optimization parity

The strict reproduction profile must expose the original search domain:

```text
initialization:
  orthogonal | he_normal | glorot_normal

activation:
  softplus | selu | prelu | sigmoid

learning_rate:
  log-uniform 5e-4 .. 1e-2 (generic)

batch_size:
  256 | 512

optimizer:
  Adam

lr_decay:
  gamma = 0.5
  number_of_decays = 3

max_iterations:
  30000

eval_steps:
  100

early_stopping:
  10 non-improving validation evaluations

gradient_clip_norm:
  1.0

loss:
  MAE

weight_decay:
  0 in the official generic hyperopt space

normalization:
  none | median | invariant (official code search)
```

The paper table also discusses broader regularization/normalization values. For code parity, the executable search space follows the official repository first; any divergence is documented rather than silently merged.

## 4. EPF variable-selection parity

The official implementation does not indiscriminately feed hundreds of engineered columns. It searches structured lag inclusion for:

- target price lags (including recent/daily/weekly positions);
- exogenous series at current/day-lag/week-lag positions;
- day-of-week.

This is an important design clue for the project adaptation: NBEATSx should first receive compact temporal trajectories and let the temporal exogenous encoder learn representations.

## 5. Reproduction levels

### Level R0 — structural parity

Pass unit tests for:

- IdentityBasis
- TrendBasis
- SeasonalityBasis
- TCN exogenous basis
- WaveNet exogenous basis
- block dimensions
- double residual
- decomposition sum
- deterministic seed behavior on CPU

### Level R1 — reference forward/training parity

On a fixed tiny reference tensor, compare our implementation against the official implementation with copied weights or controlled initialization:

- forecast shape equality;
- decomposition shape equality;
- numerical agreement within tolerance for matching architecture;
- one-step MAE and gradient sanity.

### Level R2 — EPF pipeline reproduction

Run the public paper dataset/split with the paper profile. Reproduce the expected ordering and approximate reference accuracy. Do not demand bit-identical numbers across modern PyTorch/CUDA versions, but deviations must be investigated and documented.

### Level R3 — full paper hyperopt reproduction

Optional high-compute confirmation:

- Nord Pool
- `nbeats_x`
- no data augmentation
- non-random validation
- 52 validation weeks
- 1500 hyperopt iterations

Only R3 may be described as a full search reproduction. R0-R2 are implementation/pipeline parity levels.

## 6. Separation from the project business task

The following are **not** allowed in paper-reproduction mode:

- H=34;
- DA-RT spread target;
- project CORE5/B208 inputs;
- project directional loss;
- project D-1 14:00 cutoff;
- project rolling 9-month history.

Those belong to the business-adaptation runner and must not contaminate reference parity tests.

## 7. Required artifacts

```text
runs/paper_repro/<run_id>/
  environment.json
  source_manifest.json
  config.json
  split_manifest.json
  training_curve.csv
  validation_metrics.json
  test_metrics.json
  reproduction_delta.json
  model_state.pt
  decomposition_sample.npz
```

No business experiment is considered based on a verified NBEATSx implementation until at least R0 and R1 pass; R2 should be completed before strong project conclusions are drawn.
