---
status: active
date: 2026-08-29
scope: Cycle89 next-stage research protocol after weak-signal H34 CORE5 MAE v1
purpose: test whether longer calibration history, richer origin-safe features, and alternative multi-step strategies can materially improve NBEATSx without relaxing the strict D-1 14:00 causal contract
---

# Cycle89 Next Stage — Long-History, Feature, and Rollout Research Protocol

## 1. Why this next stage exists

The repaired 14-day pre-registered panel established a trustworthy baseline for the current model:

```text
Model                  NBEATSx H34 CORE5 MAE v1
Input backcast          168 h
Forecast horizon        34 h
Training history        rolling 9 months
Validation              latest 28 legal target days
Features                CORE5 + calendar
Strategy                direct multiple-output H34
Target                  DA - RT
Origin                  D-1 14:00
Training labels         <= D-2
```

The baseline is scientifically usable but produced a `WEAK_SIGNAL` against the same-date Cycle88 LightGBM comparator. The major symptoms are:

- only roughly 245 training daily origins under a 9-month calibration window;
- a ~1.2M-parameter model reaching its best validation checkpoint very early (typically before step 200);
- weak daily-macro balanced direction performance;
- class/majority collapse on some days;
- poor alignment of intra-day sign transitions;
- CORE5-only inputs may omit useful causal price-regime / forecast-error state;
- the current H34 direct formulation avoids recursive error propagation, but its 10 bridge outputs and long 34-step horizon may still make the learning task unnecessarily difficult.

The next stage therefore answers three separate scientific questions **one at a time**:

1. **History length:** Is 9 months simply too little data for this neural model, and do 2–3 years of rolling history improve generalization?
2. **Feature information:** Does adding carefully selected origin-safe physical/state/uncertainty information improve the model beyond raw CORE5 trajectories?
3. **Forecast strategy:** Is the current direct H34 strategy preferable to recursive/block/two-stage alternatives, or can a controlled rollout strategy handle the 10-hour bridge and D-day horizon better?

These axes must not be changed simultaneously in the first comparison, otherwise attribution becomes impossible.

---

## 2. Literature basis

### 2.1 Longer calibration history is well supported in electricity price forecasting

The NBEATSx electricity-price paper by Olivares et al. uses six years of data per market. Its methodology defines the initial training set as the first three years, the following year as validation, and the last two years as test; during the test period the model is re-trained daily to incorporate newly available observations. The EPF application uses a 168-hour backcast and a 24-hour direct forecast horizon.

Reference:

- Olivares, Challu, Marcjasz, Weron, Dubrawski, *Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx*, International Journal of Forecasting 39(2), 2023, 884–900. DOI: 10.1016/j.ijforecast.2022.03.001. arXiv: https://arxiv.org/abs/2104.05522
- Official code: https://github.com/cchallu/nbeatsx

Important methodological facts from the paper:

```text
initial train coverage    ~3 years
validation                ~1 year
held-out test             ~2 years
daily recalibration       yes
backcast L                168 h
forecast H                24 h
loss                      MAE
optimizer                 Adam
```

The widely used EPF Toolbox LEAR benchmark also defaults to a 1092-day (3×364) calibration window, while examples use 4×364 days with daily recalibration. This is strong evidence that a 9-month window is not a general EPF default and may be especially restrictive for deep models.

References:

- EPF Toolbox LEAR documentation: https://epftoolbox.readthedocs.io/en/latest/modules/lear/LEAR.html
- LEAR implementation: https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/models/_lear.py
- Lago et al., *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark*, Applied Energy 293 (2021), 116983. DOI: 10.1016/j.apenergy.2021.116983.

### 2.2 Exogenous variables and feature selection are central to EPF

NBEATSx itself was introduced specifically to incorporate exogenous information and reported major gains over the original NBEATS. In electricity-price applications, standard useful information families include:

- historical prices;
- day-ahead load forecasts;
- wind and solar forecasts;
- generation / system-state forecasts;
- calendar information;
- lagged and regime information;
- selected transformations rather than uncontrolled feature inflation.

Recent EPF work continues to show that feature selection can improve performance, while redundant or irrelevant features can harm models. Probabilistic load/renewable information is also increasingly useful in renewable-heavy markets.

References:

- Olivares et al. NBEATSx, DOI above.
- Lago et al. Applied Energy 2021 benchmark/best-practice review.
- Uniejewski, Marcjasz, Weron, *Understanding intraday electricity markets: Variable selection and very short-term price forecasting using LASSO*, International Journal of Forecasting 35(4), 2019, 1533–1547. DOI: 10.1016/j.ijforecast.2019.02.001.
- *Electricity price forecasting in New Zealand: A comparative analysis of statistical and machine learning models with feature selection*, Applied Energy 347 (2023), 121446. DOI: 10.1016/j.apenergy.2023.121446.
- *A robust electricity price forecasting framework based on heteroscedastic temporal Convolutional Network*, International Journal of Electrical Power & Energy Systems (2024): the model uses historical prices, generation/load forecasts, future forecast fundamentals, and time-related variables with explicit feature selection.
- *The role of probabilistic load and renewable prediction in enhancing day-ahead electricity price forecasts*, Renewable Energy 269 (2026), 125844, DOI: 10.1016/j.renene.2026.125844: probabilistic load/wind/solar inputs materially improve EPF in the reported study.

### 2.3 Recursive rolling is a hypothesis, not an assumed solution

The multi-step forecasting literature distinguishes:

```text
Recursive / Iterated      one-step model recursively feeds its prediction back
Direct                    separate horizon-specific prediction
MIMO                      one model predicts the full future vector jointly
DirRec                    direct + recursive hybrid
DIRMO / block-MO          predicts blocks and recursively advances between blocks
```

Recursive forecasting **can accumulate errors** because its own predictions become subsequent inputs. Direct/MIMO methods avoid that particular feedback loop but may be harder to estimate for long horizons. Hybrid/block strategies can trade off horizon dependence and recursive error propagation.

References:

- Ben Taieb, Bontempi, Atiya, Sorjamaa, *A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition*, Expert Systems with Applications 39(8), 2012, 7067–7083. DOI: 10.1016/j.eswa.2012.01.039.
- Bontempi & Ben Taieb, *Conditionally dependent strategies for multiple-step-ahead prediction in local learning*, International Journal of Forecasting 27(3), 2011, 689–699. DOI: 10.1016/j.ijforecast.2010.09.004.
- Ben Taieb & Hyndman, *Boosting multi-step autoregressive forecasts*, ICML/PMLR 2014.
- Lim & Zohren, *Time-series forecasting with deep learning: a survey*, Philosophical Transactions of the Royal Society A 379 (2021), DOI: 10.1098/rsta.2020.0209.
- Mendez-Garces et al., *Comparative Evaluation of Direct and Recursive Multi-Step Forecasting for Electricity Demand Using Deep Learning and Gradient Boosting Models*, Energies 19(15), 2026, 3563: recursive configurations can show horizon-dependent error accumulation.

If recursive training uses true previous targets but inference uses model predictions, it additionally suffers a training/inference mismatch often called exposure bias. Scheduled sampling was proposed to reduce this mismatch, although it should be treated as an empirical technique rather than assumed theoretically optimal.

References:

- Bengio et al., *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*, NIPS 2015, arXiv:1506.03099.
- Huszár, *How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?*, arXiv:1511.05101 (important theoretical caveat).

Therefore Cycle89 must compare rollout strategies rather than claiming rolling prediction is automatically superior.

---

## 3. Frozen causal contract for every next-stage experiment

No experiment may relax:

```text
Target                       DA - RT
Formal forecast origin       D-1 14:00
D-1 realized target allowed  only h1..h14
D-1 h15..h24 actual target   forbidden as model input
D-day actual price           forbidden as model input
D-day actual fundamentals    forbidden as model input
Target-day DA                forbidden as model input in strict direct-spread task
Full supervised labels       <= D-2
Final September lockbox      untouched
```

For any recursive strategy, predicted values may be fed back because they are generated by the model. Future actual values may not be substituted during fixed-origin inference.

A separate real-time receding-horizon experiment that consumes newly observed actuals after D-1 14:00 is scientifically allowed **only as a different operational task** and may not be compared as the same D-1 14:00 headline forecast.

---

# PART A — Training-history study

## 4. Main hypothesis

The current 9-month window provides only roughly 245 training daily origins after reserving validation. This is small relative to the current neural network capacity.

Longer history can:

- increase the number of daily samples from hundreds to roughly 700–1000+;
- expose the model to more sign regimes, renewable conditions and price spikes;
- make exogenous encoders better identified;
- reduce sensitivity to one recent 28-day validation regime.

But excessively old data can introduce concept drift. Therefore the question is empirical.

## 5. History configurations

Run first with **all other settings frozen**:

```text
A0_HISTORY_9M   current baseline
A1_HISTORY_24M  rolling 24 calendar months
A2_HISTORY_36M  rolling 36 calendar months
```

Keep initially:

```text
validation = latest 28 complete legal target days
L = 168
H = 34 direct
CORE5 + calendar
Identity -> TCN
hidden 256
MAE
seed 42
```

Do not change model size at the same time.

### Optional follow-up only if longer history helps

The NBEATSx paper uses a substantially larger early-stopping set than Cycle89. If A1/A2 materially improves the development panel, then separately test:

```text
A3_HISTORY36_VAL84
36 months history
latest 84 legal target days validation
```

Do **not** jump directly to the paper's 42-week early-stopping set because that would change too many effective training samples in one step. `VAL84` is a controlled intermediate test.

### Optional all-available history

Only if 36m wins consistently and older data quality is clean:

```text
A4_EXPANDING_AVAILABLE
all causally available data subject to minimum-start quality audit
```

This is not part of the first history screen.

## 6. History-stage selection rule

Use the existing pre-registered `DEV14` dates for screening because they are already a development panel.

Primary comparison is against current A0 NBEATSx, not against the final holdout.

A history candidate is considered meaningfully better if it satisfies most of:

```text
daily-macro balanced >= A0 + 2 pp
micro raw              >= A0 - 1 pp
MAE                    <= A0 * 1.05
majority-collapse days <= A0
performance gain not concentrated in <=2 days
```

Prefer Pareto dominance to a single opaque scalar score.

If neither 24m nor 36m improves robustness, stop the long-history hypothesis rather than adding 48m/60m blindly.

---

# PART B — Feature study

## 7. General principle

Do not dump the full B208 LightGBM feature cube into NBEATSx.

NBEATSx already consumes an autoregressive target backcast and temporal exogenous trajectories. Added features should supply **new causal state**, not duplicate hundreds of correlated transformations.

Feature experiments begin only after the best training-history configuration is frozen.

## 8. Feature packages

### B0 — CORE5_RAW

Current baseline:

```text
5 forecast trajectories:
- direct load
- interconnection received load
- wind
- solar
- bidding space

+ hour/dow sin/cos
```

### B1 — PHYSICAL_SHAPE

Add deterministic transformations of already legal CORE5 forecasts:

```text
renewable_total      = wind + solar
net_load_proxy       = direct_load - wind - solar
bidding_stress_ratio = bidding_space / robust_scale(direct_load)
CORE5 first ramps    = delta of each forecast trajectory
```

Important:

- these transformations introduce no new future information;
- all ratios use denominator floors / clipping;
- transformations are computed from forecast values available at the formal origin;
- ramps for the first future step may use the last legal forecast value at/before origin, never an actual future value.

This package tests whether the network benefits from explicit physical inductive bias.

### B2 — CAUSAL_PRICE_STATE

Add a compact day-level state vector computed solely from legal historical target values, e.g.:

```text
last legal spread at D-1 h14
mean / median / std of D-1 h1..h14 spread
positive fraction of D-1 h1..h14
linear slope of D-1 h1..h14
sign-switch count of D-1 h1..h14
trailing 7d spread mean/std
trailing 28d spread mean/std
```

For a temporal NBEATSx input these state values may be repeated as constant future channels or passed through a dedicated static encoder, but the representation choice must be fixed before testing.

The preferred first implementation is constant future/static channels because it changes the architecture less.

### B3 — CAUSAL_FORECAST_ERROR_STATE

Reuse only the strict causal concepts already developed in Cycle88 F5/F6:

```text
historical forecast-error bias
historical forecast-error MAE / dispersion
causal uncertainty bands / reliability state
```

Rules:

- every statistic must use observations whose complete labels/actual fundamentals are available no later than D-2;
- origin availability must be machine-audited;
- no target-day actual fundamental may enter the calculation;
- do not silently backfill missing future errors.

This package is strongly motivated by the fact that point load/wind/solar forecasts contain uncertainty, and recent EPF literature shows gains from uncertainty-aware exogenous inputs.

### B4 — LIGHTGBM_STABILITY_SHORTLIST

Only if B1/B2/B3 show that extra state helps.

Use Cycle88 as a feature-discovery tool, not as a permission to inject B208 wholesale.

Procedure:

1. run feature importance/stability only within legal training folds;
2. require stability across multiple months/folds;
3. exclude any feature whose publication-time contract is unclear;
4. select `top16`, optionally `top32`;
5. freeze the list before DEV14 evaluation.

No target-day or DEV14 performance may be used to choose individual features.

## 9. Feature experiment sequence

Recommended order:

```text
B0 CORE5
 -> B1 PHYSICAL_SHAPE
 -> B2 CAUSAL_PRICE_STATE
 -> B3 CAUSAL_FORECAST_ERROR_STATE
 -> B1+B2 only if each has independent signal
 -> top16 stability shortlist only after the compact packages are understood
```

Do not run dozens of combinatorial feature packages.

---

# PART C — Multi-step / rollout strategy study

## 10. Important clarification

The current H34 direct model is a MIMO-style fixed-horizon model. It does **not** recursively feed predictions back, therefore it does not suffer classical recursive error accumulation.

Its weakness is different:

- long horizon 34;
- difficult unknown 10-hour bridge;
- single network must learn bridge + D-day jointly;
- bridge consumes part of the modeling/loss capacity.

Recursive rolling prediction may help by reducing output dimensionality and generating more local forecasting tasks, but it introduces the very feedback-error mechanism the literature warns about.

Therefore rollout must be compared as an ablation.

## 11. Strategy C0 — DIRECT_H34

Current baseline:

```text
168 -> 34
one forward pass
first 10 bridge auxiliary
last 24 scored
```

This remains the reference.

## 12. Strategy C1 — RECURSIVE_H1_FREE_RUN

Train one model to produce the next target value, then recursively generate 34 values:

```text
origin y-history
 -> predict t+1
 -> append predicted t+1 to target history
 -> predict t+2
 ...
 -> predict t+34
```

Future exogenous forecasts for each step are legal because they are origin-known forecast trajectories.

### Training requirement

Do not train only under pure teacher forcing and then free-run at inference.

Preferred first implementation:

```text
free-running rollout within each historical daily-origin sample
model-generated previous target is fed into later rollout steps
rollout losses are accumulated across all 34 steps
```

For stability, the first experiment may detach the previous prediction before inserting it into the next input window, while sharing parameters across all calls. Record this explicitly as:

```text
REC1_DETACHED_FREE_RUN
```

Only after that baseline is understood should full backpropagation through all 34 recursive calls or scheduled sampling be considered.

### Required diagnostics

```text
MAE by rollout step
bias by rollout step
balanced direction by rollout step
error growth slope
prediction variance collapse
sign-switch timing
```

If errors grow strongly with horizon, recursive rollout is rejected.

## 13. Strategy C2 — BLOCK_DIRMO_6

A compromise between direct and fully recursive prediction:

```text
168 -> next 6 targets
append 6 predictions
predict next 6
...
```

For H34 this is approximately 6 rollout blocks, with the final block partially scored.

Why test it:

- fewer feedback cycles than H1 recursion;
- preserves within-block output dependence;
- lower output dimension than H34;
- conceptually aligned with DIRMO/block multi-output literature.

Freeze block size to `6` for the first test. Do not search 2/4/6/8/12 simultaneously.

Only if block-6 shows clear promise may block size become a later ablation.

## 14. Strategy C3 — BRIDGE10_TO_DIRECT24

This strategy directly addresses the business gap:

```text
Stage 1: origin D-1 14 -> predict missing bridge10
Stage 2: completed history using predicted bridge -> predict D-day 24 jointly
```

This has only one major recursive handoff rather than 34 recursive steps.

### Critical training rule: no bridge teacher-forcing shortcut

Stage 2 may not be trained exclusively on true bridge10 and then evaluated with predicted bridge10, because that creates a train/inference mismatch.

Preferred rigorous design:

```text
for Stage-2 training samples:
  create bridge predictions with causal cross-fitting / out-of-fold Stage-1 models
  use those predicted bridges as Stage-2 inputs
```

All cross-fitted Stage-1 models must be trained only on data legally prior to the sample being generated.

If causal cross-fitting is too expensive, do not implement a contaminated shortcut. Mark the experiment blocked until a valid approximation is designed.

## 15. Strategy C4 — REAL_TIME_RECEDING_HORIZON (separate task only)

A real operational rolling system can refresh the forecast after D-1 14 as new actuals arrive:

```text
15:00 actual becomes available -> refresh remaining horizon
16:00 actual becomes available -> refresh remaining horizon
...
```

This can reduce uncertainty but answers a different question.

It must use a separate label such as:

```text
REALTIME_REFRESH
```

and may never be reported as the fixed D-1 14:00 headline result.

Do not implement this until fixed-origin strategies are complete.

---

# PART D — Experiment order and anti-confounding rules

## 16. Required sequence

Do not run all three axes together.

### Phase 0 — repair remaining engineering evidence

Before new experiments, close the known non-model issues:

```text
parameter-count audit must compare actual count to threshold
Git provenance must distinguish repository vs Cycle89 scope
artifact reuse must verify config/source/data/device/seed hashes
sign-transition total/reporting bug must be corrected
add transition timing precision/recall/F1
```

### Phase A — history length

```text
A0 9m
A1 24m
A2 36m
```

Same model, same features, same direct H34.

### Phase B — features

Freeze the best history window, then:

```text
B0 CORE5
B1 PHYSICAL_SHAPE
B2 CAUSAL_PRICE_STATE
B3 CAUSAL_FORECAST_ERROR_STATE
```

Only compact combinations that are justified by individual results.

### Phase C — forecasting strategy

Freeze the best history + selected compact feature package, then:

```text
C0 DIRECT_H34
C1 RECURSIVE_H1_FREE_RUN
C2 BLOCK_DIRMO_6
C3 BRIDGE10_TO_DIRECT24
```

C3 runs only if causal bridge cross-fitting is implemented correctly.

### Phase D — confirmation

Select no more than the top two candidates after DEV14, then run once on an untouched confirmation panel.

---

# PART E — Development and confirmation panels

## 17. DEV14 remains the tuning/development panel

Already consumed and therefore explicitly development-only:

```text
June: 01,05,10,15,20,25,30
July: 01,05,10,15,20,25,31
```

It may be used to choose among A/B/C configurations.

It may not later be described as untouched confirmation.

## 18. New pre-registered CONFIRM21 panel

The following dates are frozen now, before any next-stage model results are observed.

### March 2026

```text
2026-03-01
2026-03-05
2026-03-10
2026-03-15
2026-03-20
2026-03-25
2026-03-31
```

### May 2026

```text
2026-05-01
2026-05-05
2026-05-10
2026-05-15
2026-05-20
2026-05-25
2026-05-31
```

### August 2026

```text
2026-08-01
2026-08-05
2026-08-10
2026-08-15
2026-08-20
2026-08-25
2026-08-26
```

Total:

```text
21 target days
504 scored D-day hours
```

Rules:

- do not inspect candidate performance on these dates until Phase D;
- do not delete/replace dates after seeing results;
- only top 1–2 DEV14 configurations enter CONFIRM21;
- September remains the final future lockbox and is untouched.

---

# PART F — Metrics and interpretation

## 19. Mandatory metrics

Every experiment reports:

### Micro hourly

```text
raw direction accuracy
positive recall
negative recall
balanced accuracy
MAE
RMSE
```

### Daily macro

```text
mean / median / std / p10
raw
positive recall
negative recall
balanced
MAE
```

### Regime diagnostics

```text
actual positive rate
predicted positive rate
majority-collapse days
raw-minus-majority diagnostic
actual sign-switch count
predicted sign-switch count
transition precision
transition recall
transition F1
```

### Horizon diagnostics

```text
H34 offset
MAE
bias
balanced
positive/negative recall
error-growth slope for recursive strategies
```

## 20. Same-date baselines

Continue to compare against strict Cycle88 predictions on exactly matching dates.

Do not compare different date sets.

Primary baseline remains the strongest compatible strict LightGBM artifact available for that date panel. Record source artifact/hash in every comparison.

## 21. Statistical reporting

DEV14 is too small to over-interpret p-values. Use:

```text
paired day-level deltas
paired bootstrap confidence intervals
win/tie/loss day counts
```

On larger confirmation/full-month sets, add Diebold-Mariano or Giacomini-White style daily L1 error comparison where assumptions and sample size are reasonable. NBEATSx itself uses the Giacomini-White conditional predictive ability framework for EPF comparisons.

---

# PART G — Go / stop logic

## 22. History stage

Proceed to feature study only after selecting the best history configuration.

If 24m/36m do not improve current NBEATSx robustness, preserve 9m and continue only if there is a strong feature/strategy rationale.

## 23. Feature stage

A new feature package must improve information quality, not merely training fit.

Preferred evidence:

```text
daily-macro balanced improves
minority recall improves
MAE does not materially deteriorate
majority-collapse does not increase
benefit appears across several days
```

## 24. Strategy stage

A rollout strategy must beat DIRECT_H34 after considering:

```text
headline metrics
error growth by horizon
sign-transition timing
runtime
number of model calls
stability
```

A recursive model with slightly better first hours but strong later error accumulation is not considered a success.

## 25. Confirmation stage

After DEV14 selection, run top 1–2 candidates once on CONFIRM21.

A candidate is considered credible only if the direction of improvement broadly survives confirmation.

Do not tune again on CONFIRM21 and then call it confirmation.

---

# PART H — Recommended immediate implementation order

## 26. Next coding round

The next Codex execution should **not** implement every stage at once.

Implement in this order:

```text
1. close Phase-0 engineering evidence bugs
2. generalize calibration-window code beyond 9m
3. add 24m and 36m configs
4. validate sample counts / D-2 boundary for all windows
5. run DEV14 A0/A1/A2 history study
6. write history comparison report
7. stop and review before adding features
```

Reason:

The literature and current diagnostics make insufficient neural training history the cleanest first hypothesis. If a 2–3 year history materially improves the baseline, subsequent feature and rollout experiments will be evaluated on a more appropriate neural training regime.

Do not implement recursive rollout before the long-history result is known unless the software abstraction can be added without running the experiment.

---

## 27. Required history-study artifacts

Create:

```text
runs/history_window_study/
  9m/
  24m/
  36m/
  comparison/
```

Comparison root must contain:

```text
history_window_daily_metrics.csv
history_window_micro_metrics.csv
history_window_macro_metrics.csv
history_window_paired_deltas.csv
history_window_training_sample_counts.csv
history_window_gradient_summary.csv
history_window_majority_collapse.csv
history_window_transition_metrics.csv
history_window_review.md
```

The review must explicitly answer:

1. Does more history reduce validation/OOS variance?
2. Does it improve daily-macro balanced accuracy?
3. Does minority recall improve?
4. Does MAE improve?
5. Does majority collapse decrease?
6. Do best checkpoints move later than the current 75–175-step range?
7. Does 36m outperform 24m enough to justify extra computation?
8. Which single history window should be frozen for the feature stage?

---

## 28. Current scientific expectation

The next stage should **not** assume a winner.

Reasonable hypotheses are:

```text
H1: 24–36m history improves a small-sample deep model.
H2: compact physical/state features improve sign-transition learning.
H3: direct H34 may remain better than H1 recursion because recursive error accumulation is real.
H4: block-DirMO or bridge10->direct24 may outperform both extremes.
```

All four hypotheses are falsifiable.

The research goal is to identify which mechanism is actually limiting the current model, not to force NBEATSx to beat LightGBM by uncontrolled tuning.
