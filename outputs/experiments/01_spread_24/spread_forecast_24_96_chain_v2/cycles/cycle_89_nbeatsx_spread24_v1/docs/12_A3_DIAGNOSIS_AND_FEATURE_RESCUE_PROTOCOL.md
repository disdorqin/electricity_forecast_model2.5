---
status: active
date: 2026-08-29
scope: post-A3 diagnosis and compact-feature rescue stage
purpose: stop further history/validation-window tuning after A3, preserve long-history gains, and test whether compact origin-safe features can recover intra-day sign structure without sacrificing MAE
---

# Cycle89 — A3 Diagnosis and Compact Feature Rescue Protocol

## 1. Stage conclusion

The history/validation study has now answered the main history-length question sufficiently for the next stage.

Frozen DEV14 evidence:

```text
A0  9m / val28
raw                  57.14%
micro balanced       55.05%
daily-macro balanced 50.27%
MAE                  96.71
collapse days        3/14

A1 24m / val28
raw                  61.61%
micro balanced       58.64%
daily-macro balanced 51.24%
MAE                  86.75
collapse days        5/14

A2 36m / val28
raw                  67.86%
micro balanced       65.56%
daily-macro balanced 52.01%
MAE                  84.85
collapse days        7/14

A3 36m / val84
raw                  65.48%
micro balanced       64.16%
daily-macro balanced 49.23%
MAE                  80.41
collapse days        9/14
transition F1        0
```

Interpretation:

1. More history clearly helps the neural model learn magnitude and coarse day-level regime.
2. 36m/28d is substantially stronger than 9m in pooled raw and MAE, and the gain is spread across many DEV14 dates rather than one isolated day.
3. Widening validation from 28d to 84d further improves MAE but worsens daily robustness, majority collapse, and transition timing.
4. Therefore the main remaining bottleneck is no longer insufficient history alone.
5. The current MAE/direct-H34 model increasingly learns a smooth/coarse daily sign regime instead of stable intra-day sign transitions.

This means A3 is rejected, but the evidence does **not** justify returning to 9m as though long history had no value.

## 2. Frozen anchors for the next stage

Two anchors are retained for different scientific roles:

```text
ROBUSTNESS_CONTROL = A0_HISTORY_9M_VAL28
PERFORMANCE_CHASSIS = A2_HISTORY_36M_VAL28
```

A3_HISTORY36_VAL84 is closed as `LONG_HISTORY_REGRESSION` and must not be used as the next-stage chassis.

Why A2 is the main chassis:

- raw improves by ~10.7pp versus A0;
- MAE improves by ~11.9;
- pooled positive and negative recall both improve;
- best checkpoints move later, showing that the larger training set is actually being used;
- the remaining failure mode is identifiable: majority collapse / weak intra-day minority and transition structure.

Why A0 remains a control:

- fewer collapse days;
- useful reference for deciding whether a feature genuinely adds structural information or merely interacts with long-history regime fitting.

Do not run every feature package on both histories. First screen features on A2. Only the best compact feature candidate may be cross-checked once on A0.

---

## 3. What A3 teaches about the failure mechanism

A3 is particularly informative because its pooled metrics look good while its day-level structure collapses.

A3 pooled:

```text
raw              65.48%
balanced         64.16%
positive recall  72.68%
negative recall  55.63%
MAE              80.41
```

but:

```text
daily-macro balanced = 49.23%
majority collapse     = 9/14
transition F1         = 0
predicted sign switches average only ~1.43/day
```

The apparent contradiction is explained by **cross-day regime aggregation**:

- on some days the model predicts almost entirely positive;
- on other days almost entirely negative;
- across all 336 hours these day-level choices can produce respectable pooled positive/negative recall;
- inside an individual day, minority-class recall is often zero and sign transitions are missed.

Therefore pooled balanced accuracy is no longer sufficient for selection.

For all next experiments, the primary structural metrics are:

```text
daily-macro balanced
macro minority recall
majority-collapse count
transition recall / F1
```

Micro raw, micro balanced and MAE remain required, but they cannot alone promote a candidate.

---

## 4. Stop further history/validation-window search

Do not test:

```text
48m
60m
all-history
VAL112
VAL168
other arbitrary history/validation combinations
```

at this stage.

The current evidence already shows the direction of the trade-off. Additional history/validation search would likely overfit DEV14 and delay the more relevant question: whether the model is missing causal information needed to resolve intra-day structure.

CONFIRM21 and September remain untouched.

---

# PART A — Compact feature rescue

## 5. Main hypothesis

The current model sees:

```text
168h historical spread
5 future forecast trajectories
calendar sin/cos
```

This is sufficient for coarse regime/magnitude learning but may require the network to infer important physical and price-state relationships from raw trajectories with relatively little inductive guidance.

The first feature stage therefore adds **small, interpretable, origin-safe state packages**, one package at a time.

Do not inject B208 wholesale.

Primary chassis for feature screening:

```text
history = 36m
validation = 28d
strategy = DIRECT_H34
loss = MAE
architecture = frozen Identity -> Exogenous-TCN 2x256
```

Reference candidate:

```text
F0 = A2_HISTORY36_VAL28 + CORE5_RAW
```

---

## 6. F1 — PHYSICAL_SHAPE

Purpose: make future intra-day physical shape explicit instead of forcing the network to derive all interactions from five raw forecast channels.

Add only transformations constructed from already legal origin-known forecasts:

```text
renewable_total = wind + solar
net_load_proxy  = direct_load - wind - solar
bidding_stress  = bidding_space / max(abs(direct_load), denominator_floor)

first differences / ramps for:
- direct load
- interconnection received load
- wind
- solar
- bidding space
```

Optional only if the first implementation remains compact and audited:

```text
renewable ramp = delta(wind + solar)
net-load ramp  = delta(net_load_proxy)
```

Do not add second differences in the first F1 experiment.

For the first H34 future step, any ramp requiring a previous value must use the last legal forecast value available at/before origin, never an actual future observation.

Expected mechanism:

- identify sunrise/solar ramps;
- identify load-renewable imbalance changes;
- identify bidding-space stress changes;
- provide explicit slope information that may help locate sign transitions.

---

## 7. F2 — CAUSAL_PRICE_STATE

Purpose: provide recent market-regime information explicitly without using future labels.

Use only data visible by D-1 14:00 or fully matured by D-2.

Compact first package:

```text
last_spread_at_origin
D1_h1_h14_mean
D1_h1_h14_std
D1_h1_h14_positive_fraction
D1_h1_h14_linear_slope
D1_h1_h14_sign_switch_count
trailing_7d_spread_mean
trailing_7d_spread_std
trailing_28d_spread_mean
trailing_28d_spread_std
```

Do not add dozens of lags/statistics in F2.

Representation:

- keep the base architecture unchanged as much as possible;
- first implementation may broadcast these day-level states as constant auxiliary temporal channels;
- document this explicitly;
- fit any scaling using train split only.

Expected mechanism:

- distinguish persistent positive/negative regimes from transition-prone days;
- give the model explicit recent sign persistence / volatility context;
- reduce the tendency to infer a single daily sign solely from long-history central tendency.

---

## 8. F3 — CAUSAL_FORECAST_ERROR_STATE (deferred until F1/F2 review)

Do not run F3 in the same first feature round.

If F1/F2 show useful signal, then separately test compact causal historical forecast-error state derived from Cycle88 F5/F6 concepts.

Possible later inputs:

```text
historical forecast bias
historical forecast MAE/dispersion
forecast reliability / uncertainty state
```

All underlying actuals must be fully matured no later than D-2.

---

## 9. First feature screen matrix

Run only:

```text
F0_A2_CORE5             frozen A2 reference
F1_A2_PHYSICAL_SHAPE
F2_A2_CAUSAL_PRICE_STATE
```

All three on the same DEV14 dates.

Do not run F1+F2 until individual effects are known.

Do not run F3/B208/rollout/directional loss in this round.

---

## 10. Promotion criteria relative to A2

A feature package should address the actual failure mode, not merely improve MAE.

Prefer promotion if most of the following are satisfied:

```text
1. daily-macro balanced >= A2 + 2pp
   target >= 54.01%

2. majority-collapse days <= 5
   A2 reference = 7

3. macro minority recall materially improves
   target improvement >= 5pp where defined

4. transition recall or transition F1 improves
   with no artificial increase caused solely by noisy over-switching

5. micro raw >= A2 - 1pp
   target >= 66.86%

6. MAE <= A2 * 1.05
   target <= ~89.09
```

No single metric is sufficient.

A feature that lowers MAE but increases collapse is not a structural success.

A feature that improves transitions but destroys raw/MAE is also not a promotion candidate.

---

## 11. Cross-history check

After F1/F2 are complete:

- select at most one best compact feature package;
- run that package once on A0_HISTORY_9M_VAL28 using DEV14;
- this is an interaction check, not a new search.

Interpretation:

```text
feature helps both A0 and A2 -> robust information gain
feature helps only A2        -> long-history interaction
feature helps only A0        -> long history may suppress local state
feature helps neither        -> close that feature hypothesis
```

Do not open CONFIRM21 yet.

---

# PART B — What happens after feature screening

## 12. Forecast-strategy stage remains next, but not yet

Once history + one compact feature profile are frozen, compare forecasting strategies.

Because the current failure is intra-day structure, the preferred order is:

```text
C0 DIRECT_H34              reference
C2 BLOCK_DIRMO_6           first alternative
C3 BRIDGE10_TO_DIRECT24    if causal cross-fitting is correct
C1 RECURSIVE_H1_FREE_RUN   after the lower-risk structured alternatives
```

Rationale:

- pure recursive H1 has the highest classical error-propagation risk;
- block-DIRMO reduces output horizon while limiting the number of recursive feedback events;
- bridge10->direct24 directly separates the unknown 10-hour bridge from the scored D-day vector;
- full recursive H1 remains scientifically valuable, but should not be assumed superior merely because it is rolling.

No rollout experiment starts in the feature round.

---

## 13. Loss stage remains separate

Current MAE is useful as a baseline but does not directly optimize direction or minority transitions.

Do not mix feature and loss changes.

After feature/strategy attribution is understood, later controlled experiments may test:

```text
scored24 MAE + 0.25 * bridge10 MAE
pseudo-Huber magnitude loss
balanced positive/negative direction surrogate
raw direction surrogate
```

The current evidence strongly suggests a direction-aware objective may eventually be necessary, but it must not be used to rescue an unexplained feature experiment.

---

## 14. Statistical/reporting requirements

For each F0/F1/F2 candidate report:

```text
micro raw
micro positive recall
micro negative recall
micro balanced
MAE/RMSE

daily-macro raw
daily-macro balanced
daily-macro positive recall
daily-macro negative recall
macro minority recall

majority-collapse days
actual/predicted positive fraction by day
actual/predicted sign-switch count
transition precision
transition recall
transition F1

metric by H34 offset
paired day-level delta vs A2
paired day-level delta vs Cycle88
```

Also report win/tie/loss days and paired bootstrap intervals where useful.

Do not use pooled metrics alone for promotion.

---

## 15. Required implementation artifacts

Create a dedicated feature-stage runner rather than modifying history-run outputs in place.

Suggested:

```text
scripts/run_compact_feature_study.py
runs/compact_feature_study/
    F0_A2_CORE5/
    F1_A2_PHYSICAL_SHAPE/
    F2_A2_CAUSAL_PRICE_STATE/
    comparison/
```

Required comparison artifacts:

```text
feature_daily_metrics.csv
feature_micro_metrics.csv
feature_macro_metrics.csv
feature_majority_collapse.csv
feature_transition_metrics.csv
feature_h34_offset_metrics.csv
feature_paired_vs_A2.csv
feature_paired_vs_Cycle88.csv
feature_review.md
```

Each target-day run keeps strict provenance, config/source/data hashes, leakage audit, split manifest and 24 scored rows.

---

## 16. Required tests before DEV14 feature runs

At minimum:

```text
test_physical_shape_features_origin_safe.py
test_future_ramp_uses_legal_previous_forecast.py
test_price_state_cutoff.py
test_price_state_no_post14_actual.py
test_feature_scaler_train_only.py
test_feature_package_dimensions.py
test_feature_counterfactual_target_isolation.py
test_compact_feature_runner_frozen_panel.py
```

Project preflight and all existing Cycle89 tests remain mandatory.

---

## 17. Current stage decision

```text
A3_HISTORY36_VAL84      CLOSED / REJECT
A0_HISTORY9_VAL28       ROBUSTNESS CONTROL
A2_HISTORY36_VAL28      FEATURE-STAGE PERFORMANCE CHASSIS
CONFIRM21               UNTOUCHED
SEPTEMBER LOCKBOX       UNTOUCHED
NEXT ACTIVE QUESTION    CAN COMPACT ORIGIN-SAFE FEATURES RECOVER INTRADAY STRUCTURE?
```

The next stage should not ask whether more historical data helps. That has already been answered: it helps magnitude and coarse regime but does not by itself solve daily structural robustness.
