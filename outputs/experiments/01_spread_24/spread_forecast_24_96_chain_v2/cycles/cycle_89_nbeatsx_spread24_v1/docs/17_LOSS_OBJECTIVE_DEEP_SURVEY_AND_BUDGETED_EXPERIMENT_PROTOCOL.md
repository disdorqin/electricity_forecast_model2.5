---
status: active
date: 2026-08-30
scope: Cycle89 Loss / Objective research after FULLDEV5
purpose: deeply review loss/objective literature and define a compute-budgeted, causally strict experimental ladder that avoids repeating 6-hour full cross-month runs for every candidate
---

# Cycle89 Loss / Objective — Deep Literature Survey and Budgeted Experiment Protocol

## 1. Why the research direction changes now

FULLDEV5 materially changed the interpretation of Cycle89.

Frozen full-month evidence across Jan/Feb/Apr/Jun/Jul (150 target days, 3600 scored D-day hours per model):

```text
C0 DIRECT_H34
raw                       52.53%
positive recall           55.49%
negative recall           47.84%
balanced                  51.66%
MAE                       99.18
daily-macro balanced      50.22%
majority collapse         46/150
transition F1             0.1227

C3 DIRMO
raw                       52.75%
balanced                  51.33%
MAE                       103.29
daily-macro balanced      50.93%
majority collapse         33/150
transition F1             0.1136

Cycle88 LightGBM
raw                       56.12%
balanced                  51.91%
MAE                       87.24
daily-macro balanced      51.60%
majority collapse         37/150
transition F1             0.1342
```

The central failure mode is therefore not simply one forecast structure:

- pooled DEV14 results were optimistic relative to complete months;
- changing history length helped magnitude/raw on a narrow panel but did not solve daily structural collapse;
- compact features did not rescue the issue;
- gap/direct, bridge-two-stage and DIRMO did not establish cross-month structural gains;
- all C0/C3/Cycle88 full-month daily-macro balanced scores are near 50%;
- C0 still has high collapse frequency and low transition F1.

The next falsifiable hypothesis is:

> **The training objective is misaligned with the business objective.**
>
> The model is optimized almost entirely for point magnitude error, while the business headline also depends on sign, positive/negative recall balance, day-level robustness and intra-day switching.

This protocol studies that hypothesis while preserving the frozen C0 model architecture, features, history and forecast strategy.

---

# PART A — Literature survey

## 2. Baseline: MAE is reasonable for EPF, but it does not optimize sign structure

NBEATSx for electricity price forecasting uses MAE with Adam and direct multi-output forecasting. MAE is standard and robust relative to squared error, and therefore remains the proper reproduction/business baseline.

Reference:

- Olivares et al., *Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx*, International Journal of Forecasting 39(2), 2023, 884–900. DOI: 10.1016/j.ijforecast.2022.03.001.

However, MAE only penalizes absolute magnitude error. It does not distinguish a same-sign error from a sign-flipping error of equal absolute size, and it does not explicitly balance positive vs negative events.

This distinction matters directly for Cycle89 because FULLDEV5 shows:

```text
C0 MAE is finite and trainable,
but daily-macro balanced ≈ 50%,
collapse = 46/150,
transition F1 = 0.1227.
```

Thus loss research is motivated by observed model behavior rather than by generic loss-function experimentation.

---

## 3. Robust magnitude objectives: useful but not sufficient by themselves

### 3.1 Huber / pseudo-Huber / Charbonnier family

Robust statistics motivates losses whose influence on large residuals grows more slowly than L2. Huber loss is quadratic around zero and linear in the tails. Smooth relatives include pseudo-Huber / Charbonnier-style penalties.

References:

- Huber, *Robust Statistics*, Wiley, 1981; 2nd ed. Huber & Ronchetti, 2009.
- Barron, *A General and Adaptive Robust Loss Function*, CVPR 2019. The paper explicitly unifies Charbonnier, pseudo-Huber/L1-L2, Cauchy and L2-like losses.
- 2026 systematic load-forecasting study in Scientific Reports reports that Charbonnier can outperform several traditional and task-specific losses under a fixed deep forecasting architecture. This is supportive evidence for optimization stability, not direct proof for EPF sign prediction.

Important Cycle89 interpretation:

- MAE is already tail-robust compared with MSE.
- A pseudo-Huber/Charbonnier replacement should therefore **not** be expected to solve sign collapse on its own.
- Its main plausible benefit is smoother optimization and different treatment of small vs large residuals.

Therefore robust magnitude loss is a secondary test, not the first main rescue hypothesis.

### 3.2 Heteroscedastic likelihood

Recent EPF work explicitly models time-varying variance and uses a heteroscedastic likelihood to reduce the influence of volatility and extreme-price regimes.

Reference:

- Shi & Wang, *A robust electricity price forecasting framework based on heteroscedastic temporal Convolutional Network*, International Journal of Electrical Power & Energy Systems, 2024, DOI 10.1016/j.ijepes.2024.110177.

This is scientifically relevant because electricity-price errors are heteroscedastic. However, it changes the output distribution/head and targets magnitude uncertainty rather than the current sign-collapse failure. It is therefore deferred until a pure loss-only stage is exhausted.

---

## 4. Direction-aware forecasting losses: strong conceptual support, but target semantics matter

### 4.1 Recent direction-aware time-series research

A 2026 paper, *Beyond Magnitude and Shape: A Direction-Aware Loss for Time Series Forecasting*, proposes CosDir, which aligns prediction and target difference vectors through cosine similarity and reports consistent directional-accuracy gains while preserving magnitude accuracy across extensive experiments.

Reference:

- Lee et al., *Beyond Magnitude and Shape: A Direction-Aware Loss for Time Series Forecasting*, arXiv:2608.01857, 2026.

A recent Scientific Reports study also uses a hybrid direction-aware loss and notes that conventional point losses can fail to capture practically important directional behavior; it also highlights that hard sign operations are problematic for gradient optimization.

Reference:

- *Decomposition-Enhanced Network for financial time series forecasting*, Scientific Reports, 2026.

### 4.2 Why CosDir is not copied directly into Cycle89

CosDir targets the **direction of change** between adjacent values:

```text
sign(y_t - y_{t-1})
```

Cycle89's business target is instead:

```text
sign(spread_t) = sign(DA_t - RT_t)
```

These are different scientific targets.

Therefore the literature supports the principle "make direction differentiable and trainable", but the exact Cycle89 surrogate should target the sign of the spread itself, not blindly use change-direction cosine loss.

---

## 5. Classification-regression coupling is particularly relevant to electricity prices

Electricity-price literature contains direct evidence that classification and regression can be useful together when the economically important event is categorical.

Examples:

- 2022 Applied Energy work reframes short-term electricity-price interval forecasting as a pattern-classification problem and reports that classification-oriented representations can be competitive for difficult price behavior.
- A 2026 IEEE paper proposes a hybrid classification-regression method for negative electricity prices: a classifier predicts event probability and a regressor predicts magnitude.
- A 2026 electricity-price multi-task method introduces price direction classification and volatility prediction as auxiliary tasks alongside point price prediction.

References:

- Shao et al., *A pattern classification methodology for interval forecasts of short-term electricity prices based on hybrid deep neural networks*, Applied Energy 327 (2022), 120115, DOI 10.1016/j.apenergy.2022.120115.
- Vahedi et al., *A Hybrid Classification-Regression Method for Forecasting Negative Electricity Prices*, IEEE ICCE 2026, DOI 10.1109/ICCE67443.2026.11449799.
- Zhao et al., *Electricity Price Forecasting Method Based on Transformer Encoder and Multi-Scale Deep Convolutional Multi-Task Learning*, ISAEECE 2026, DOI 10.1109/ISAEECE70502.2026.11635368.

This supports testing a magnitude + sign objective in Cycle89.

---

## 6. Class imbalance: global accuracy can hide minority failure

Cycle89 repeatedly exhibits days dominated by one sign. Raw accuracy can therefore improve while the minority class collapses.

General class-imbalance literature supports explicit reweighting rather than relying on raw sample frequency.

Reference:

- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019, DOI 10.1109/CVPR.2019.00949.

Cycle89 does **not** need to copy effective-number weighting directly. The more direct task-aligned construction is:

```text
for each daily origin:
  compute positive-sign loss over positive D-day hours
  compute negative-sign loss over negative D-day hours
  average the two classes when both exist
then average over days
```

This aligns much more closely with the reported `daily-macro balanced` metric than a global BCE over all hours.

The loss must define explicit behavior for all-positive/all-negative days; such days may use the present-class directional term while the final evaluation metric remains unchanged.

---

## 7. Multi-task loss weighting: avoid a large manual lambda grid

Magnitude and sign losses can conflict. Their raw scales are also different.

General multi-task literature offers two relevant families:

### 7.1 Uncertainty weighting

Kendall, Gal & Cipolla learn task weights from homoscedastic uncertainty rather than hand-tuning many combinations.

Reference:

- Kendall, Gal & Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, CVPR 2018.

### 7.2 Gradient balancing / gradient conflict

GradNorm dynamically controls task weights through gradient magnitudes and can reduce the need for exhaustive loss-weight searches.

Reference:

- Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, ICML 2018.

Recent EPF work has also used PCGrad-style gradient surgery to mitigate conflicts in hybrid electricity-price models.

Reference:

- *Transformer-GNN fusion with gradient surgery for electricity price forecasting*, Scientific Reports, 2026.

Cycle89 decision:

- do **not** start with GradNorm/PCGrad;
- first prove that adding a sign objective has signal;
- only if magnitude improves while sign degrades or vice versa, and gradient diagnostics show conflict, activate an adaptive weighting experiment.

This avoids replacing one simple hypothesis with another large methodological branch.

---

## 8. Horizon/task weighting is legitimate when business importance is unequal

Multi-horizon forecasting literature permits weighted losses when different horizons have different application importance. Multi-horizon forecast comparison also explicitly recognizes weighted average loss across horizons as a valid business-dependent criterion.

References:

- Quaedvlieg, *Multi-Horizon Forecast Comparison*, Journal of Business & Economic Statistics, 2019.
- Temporal Fusion Transformer jointly sums losses across forecast horizons; horizon-aware weighting is also used in recent multi-horizon application literature.

Cycle89 has a uniquely strong reason for horizon weighting:

```text
H34 outputs 1..10 = bridge diagnostics/auxiliary
H34 outputs 11..34 = actual D-day headline business target
```

Therefore equal 34-point MAE optimizes 29.4% of its loss on points that are not part of the headline D-day metric.

A scored-D24-weighted loss is not metric hacking; it makes the training risk match the actual declared task.

---

# PART B — Loss hypotheses, one at a time

## 9. Frozen baseline for all loss experiments

Unless a later protocol explicitly changes it:

```text
architecture          C0 DIRECT_H34
history               36 calendar months
validation            latest 28 legal target days
features              CORE5 + calendar
forecast strategy     direct H34
L                     168
H                     34
precision             float32
AMP                   false
seed                   42
cold retrain           per target day
origin                 D-1 14:00
supervised labels      <= D-2
final metrics          unchanged
```

No new feature, history, architecture or rollout experiment may run in parallel with this loss direction.

---

## 10. L0 — MAE reference

Current C0 behavior. Reuse immutable FULLDEV5 baseline artifacts by hash whenever dates overlap.

Do not retrain L0 merely to create a comparison file.

---

## 11. L1 — Business-horizon weighted MAE

First experiment because it is the cleanest alignment change.

Recommended definition:

```text
L_D24     = mean absolute error on offsets 11..34
L_bridge  = mean absolute error on offsets 1..10

L1 = L_D24 + 0.25 * L_bridge
```

Alternative `bridge_weight=0` is deliberately not the first test because bridge prediction can still regularize the shared representation. If 0.25 is clearly beneficial, zero bridge may be tested later as a single follow-up.

Question answered:

> Is C0 wasting too much learning capacity on the non-headline bridge?

Do not add any directional term in L1.

---

## 12. L2 — Smooth robust magnitude objective

Only after L1 is reviewed.

Use normalized residuals and one frozen pseudo-Huber/Charbonnier-like scale; no scale grid.

One valid form:

```text
PH(r; delta) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
delta = 1 in normalized target units

L_D24_PH    = mean PH(r_D24)
L_bridge_PH = mean PH(r_bridge)

L2 = L_D24_PH + 0.25 * L_bridge_PH
```

Interpretation:

- tests optimization smoothness/robustness;
- is **not** expected by itself to solve sign balance;
- promote only if it improves magnitude without worsening structure.

---

## 13. L3 — Direct spread-sign auxiliary objective

This is the main scientific hypothesis.

The prediction itself is used as a differentiable sign logit; no new classifier head is introduced in the first experiment.

For normalized prediction `z_hat`, target `z`, temperature `tau`:

```text
positive target: softplus(-z_hat / tau)
negative target: softplus( z_hat / tau)
```

Recommended first temperature:

```text
tau = 0.35
```

Do not grid-search tau in the first round.

### 13.1 Near-zero reliability weight

Sign labels near zero are easy to flip with negligible magnitude difference. Training may use:

```text
w_zero = clip(abs(z) / 0.25, 0, 1)
```

This changes only the **training directional weight**. Final direction metrics continue to use the original exact zero semantics.

### 13.2 Day-balanced sign loss

Preferred sign term:

```text
for every training daily-origin sample:
  L_pos_day = weighted mean positive softplus terms
  L_neg_day = weighted mean negative softplus terms
  if both classes present:
      L_dir_day = 0.5*L_pos_day + 0.5*L_neg_day
  else:
      L_dir_day = present-class loss

L_dir = mean(L_dir_day across batch)
```

This is intentionally aligned with the observed failure in **daily-macro balanced accuracy**, rather than only pooled class balance.

### 13.3 Magnitude + direction composition

Do not launch a lambda grid.

First pre-registered composition:

```text
magnitude = L1 business-weighted MAE or L2 only if L2 earned promotion

training progress 0–20%:
    magnitude only
20–40%:
    linearly introduce direction
40–100%:
    0.75 * normalized magnitude + 0.25 * normalized day-balanced direction
```

Loss components must be normalized to comparable baseline scale before weighting.

The warm-in period protects early optimization from an unstable sign surrogate before magnitude structure is learned.

If L3 gives real signal but magnitude/sign conflict remains obvious, adaptive weighting becomes a later L4 candidate.

---

## 14. L4 — Adaptive multi-task weighting (deferred, conditional)

Run only if L3 demonstrates that directional supervision is useful but fixed weighting creates a clear Pareto conflict.

Preferred first adaptive method:

```text
GradNorm OR uncertainty weighting
```

not both.

No manual 10-point lambda grid is allowed.

---

## 15. Transition-specific loss is not yet authorized

Transition F1 is poor, but an adjacent-transition loss introduces another objective and may encourage oscillatory predictions.

CosDir/change-direction literature is scientifically interesting but does not exactly match `sign(DA-RT)`.

Therefore transition-specific loss is deferred until direct sign supervision has been tested.

---

# PART C — Compute-budgeted experimental design

## 16. Why the old DEV14 screening design is retired

DEV14 consisted only of selected June/July dates. It produced C0 raw around 67–68%, while FULLDEV5 complete months reduced C0 raw to 52.53%.

Therefore:

```text
DEV14 is retained as historical diagnostic evidence,
but is no longer a valid primary screening panel for new loss research.
```

The new low-fidelity screen must cover multiple months.

---

## 17. Literature basis for multi-fidelity / successive resource allocation

Successive Halving, Hyperband and ASHA allocate small budgets to many candidates and progressively increase resources only for promising candidates. The core idea is to terminate poor candidates early rather than fully train every configuration.

References:

- Li et al., *A System for Massively Parallel Hyperparameter Tuning*, MLSys 2020 — introduces ASHA and aggressive early stopping/resource allocation.
- Hyperband / Successive Halving literature: configurations receive progressively larger resource budgets; poor configurations are discarded early.
- Multi-objective ASHA work shows that retaining a Pareto view can be preferable to collapsing conflicting objectives into one scalar.
- EPF literature has used BOHB/Hyperband specifically to reduce costly tuning burden; e.g. the 2023 Energy hybrid day-ahead EPF framework applies BOHB across multiple PJM datasets.

Cycle89 does not need a general HPO engine. We borrow the **resource-allocation principle** and apply it to scientifically pre-registered loss candidates.

---

## 18. Local fidelity study from FULLDEV5

A read-only retrospective analysis used the completed FULLDEV5 artifacts to ask whether a small panel can approximate complete-month results.

A stratified five-month panel was searched using C0 and Cycle88 as calibration models, with C3 used only as a sanity-check model.

Important result:

### 18.1 15-day proxy

Three dates per month could reproduce C0 and Cycle88 aggregate metrics closely, but C3 raw differed from its full result by about 2.2pp. Therefore 15 days is too risky for promotion decisions.

### 18.2 25-day proxy

Five dates per month produced:

```text
C0 panel vs full
raw error        -0.03 pp
balanced error   +0.14 pp
MAE error        +1.02

Cycle88 panel vs full
raw error        +0.05 pp
balanced error   +0.35 pp
MAE error        -0.53

C3 sanity-check panel vs full
raw error        +0.75 pp
balanced error   +1.06 pp
MAE error        -4.56
```

This is useful but not exact.

Conclusion:

> **A 25-day five-month panel is suitable for rejecting obvious failures and ranking large effects, but not for declaring a 0.5–1pp improvement as real.**

This uncertainty margin is explicitly built into the promotion rules below.

---

## 19. RACE25 — frozen primary screening panel

The following 25 development dates are frozen before loss results are observed:

```text
2026-01-06
2026-01-07
2026-01-17
2026-01-19
2026-01-31

2026-02-08
2026-02-09
2026-02-12
2026-02-24
2026-02-27

2026-04-06
2026-04-09
2026-04-12
2026-04-16
2026-04-18

2026-06-02
2026-06-11
2026-06-14
2026-06-26
2026-06-29

2026-07-03
2026-07-07
2026-07-15
2026-07-25
2026-07-28
```

RACE25 is derived entirely from already-consumed FULLDEV5 months and is therefore **development-only**.

It may be used for:

- candidate rejection;
- relative ranking of large effects;
- gradient/collapse/transition diagnostics.

It may never be called confirmation or holdout.

---

## 20. Screening uncertainty guard

Because RACE25 is an approximation, any small difference is `INCONCLUSIVE`, not a win.

A candidate may advance from RACE25 only if it demonstrates a clear multi-objective signal, e.g. most of:

```text
daily-macro balanced      >= C0 + 2.0 pp
micro balanced            >= C0 + 1.5 pp
raw                       >= C0 - 1.0 pp
MAE                       <= C0 * 1.05
collapse rate             materially lower OR no worse
transition F1             no material deterioration
benefit observed in >=3/5 months
```

Because the retrospective C3 sanity check showed about 1pp panel error in balanced and ~4.6 MAE units, changes smaller than those scales are not sufficient to authorize confirmation.

---

## 21. RACE5 engineering gate

Before RACE25, every new loss must pass a tiny five-date engineering gate, one date from each FULLDEV5 month:

```text
2026-01-17
2026-02-12
2026-04-09
2026-06-14
2026-07-15
```

Purpose:

- no NaN/non-finite gradients;
- no collapse to constant zero;
- correct loss decomposition;
- learning curves improve;
- D-day/bridge weighting executed exactly;
- directional term gradients point the intended way on unit tests/counterfactual examples.

RACE5 is **not** an accuracy screen unless the candidate catastrophically fails.

---

## 22. RACE50 finalist development gate

Only one candidate at a time may advance from RACE25 to RACE50.

RACE50 uses ten dates per FULLDEV5 month, deterministically spaced to cover the month. Exact dates must be generated/frozen by a script before the first candidate is run and stored in the config artifact.

Purpose:

- reduce the proxy error of RACE25;
- test month-level consistency;
- avoid paying 150-day cost for weak candidates.

A candidate that shows only a marginal effect on RACE25 should **not** consume RACE50 budget.

---

## 23. Confirmation and final evidence

After the loss direction is frozen:

### CONFIRM21

The existing untouched pre-registered March/May/August 21-day panel remains the first independent confirmation.

Only top 1–2 fully frozen candidates may enter it.

No tuning after reading CONFIRM21 is allowed if it is to retain confirmation status.

### September lockbox

Remains untouched until architecture/features/history/strategy/loss are all frozen.

### Full cross-month rerun

Do **not** run a new 150-day FULLDEV5 for every loss candidate.

A full multi-month rerun is reserved for:

```text
one final paper-ready/frozen candidate,
or a genuinely ambiguous finalist whose RACE50 + CONFIRM21 evidence cannot separate it from C0.
```

Because C0 FULLDEV5 already exists, baseline results are always reused by hash rather than retrained.

---

# PART D — Compute-saving engineering rules

## 24. Reuse all immutable baseline evidence

For any date already run under C0:

```text
reuse C0 prediction/metric/provenance artifacts
verify source/config/data hashes
never retrain the baseline solely for comparison
```

Only the candidate loss needs training.

---

## 25. Reuse data preparation, not learned model state

Safe to cache by content hash:

- canonical source slices;
- training-day registries;
- exogenous tensors;
- calendar tensors;
- split manifests;
- train-only scaler fit inputs/results when the exact split is unchanged.

Do not warm-start candidate day D from candidate day D-1 during this scientific comparison; that would change the learning protocol.

Daily cold retrain remains frozen until warm-start itself becomes a separate research question.

---

## 26. No seed multiplication during screening

RACE5/RACE25/RACE50 use only:

```text
seed = 42
```

Multi-seed confirmation is allowed only for a finalist that has already demonstrated a clear effect.

This prevents spending 3x compute before there is a real scientific signal.

---

## 27. No loss-weight grid search

Forbidden during the first objective stage:

```text
10 lambda values
5 temperatures
multiple bridge weights
multiple warm-up lengths
```

Each stage tests one pre-registered mechanism.

If directional supervision works but weighting is the bottleneck, use one adaptive weighting method rather than a large manual Cartesian grid.

---

# PART E — Immediate experiment sequence

## 28. Stage O1 — L1 business-weighted MAE

Run order:

```text
unit tests
RACE5 engineering gate
RACE25
STOP AND REVIEW
```

Do not immediately run RACE50.

Primary scientific question:

> Does reducing bridge loss weight improve D-day generalization without changing any sign objective?

If no clear signal, close L1 and move to L2 or L3 according to diagnostics.

---

## 29. Stage O2 — smooth robust magnitude

Run only after O1 review.

Again:

```text
RACE5 -> RACE25 -> STOP
```

Promote only if it clearly improves magnitude/optimization while not worsening sign structure.

If it has no signal, do not spend RACE50.

---

## 30. Stage O3 — day-balanced sign auxiliary objective

This is expected to be the most important experiment, but it should be run after the magnitude-only controls establish attribution.

Required diagnostics beyond normal metrics:

```text
magnitude loss trajectory
direction loss trajectory
magnitude-gradient norm
direction-gradient norm
cosine similarity between the two gradients on shared parameters
fraction of days with both signs
near-zero directional weight distribution
predicted positive rate by day/month
collapse count
transition F1
```

If the sign term improves daily-macro balanced/collapse but materially damages MAE, record a real Pareto conflict rather than declaring failure immediately; that is the condition under which adaptive weighting is scientifically justified.

---

# PART F — Decision tree

## 31. Direction stop/go logic

```text
L1 weighted horizon
  ├─ clear positive -> freeze as magnitude baseline
  └─ no signal      -> keep C0 magnitude baseline

L2 robust magnitude
  ├─ clear positive -> freeze robust magnitude
  └─ no signal      -> close robust-only branch

L3 day-balanced sign
  ├─ balanced/collapse improve, MAE acceptable
  │      -> finalist -> RACE50
  ├─ balanced improves but MAE degrades strongly
  │      -> one adaptive weighting experiment
  └─ no structural signal
         -> close Loss / Objective direction
```

If Loss / Objective closes with no signal, the next deferred directions remain:

- causal forecast-error/uncertainty state;
- LightGBM-stability feature shortlist;
- longer backcast L336;
- specialized regime/router models;
- RecMO only if there is a new mechanistic reason.

They must not be opened simultaneously.

---

# PART G — Hard prohibitions

During this loss research direction:

```text
NO new features
NO history-window search
NO validation-window search
NO model-size search
NO C3/RecMO/Recursive-H1 work
NO threshold tuning on final metrics
NO CONFIRM21 until a finalist exists
NO September lockbox
NO full 150-day rerun for ordinary screening
NO multi-seed screening
NO large lambda grid
```

Every run remains STRICT D-1 14:00 / D-2 causal.

---

# PART H — Practical conclusion

The compute lesson from FULLDEV5 is now part of the scientific protocol:

> Complete multi-month testing is essential for final claims, but it is too expensive to use as the first filter for every idea.

Therefore Cycle89 adopts a multi-fidelity evidence ladder:

```text
RACE5   -> engineering correctness only
RACE25  -> primary cross-month development rejection screen
RACE50  -> one finalist only
CONFIRM21 -> independent confirmation
September -> final lockbox
full cross-month rerun -> final frozen model only if needed
```

This preserves cross-month discipline while reducing repeated GPU cost by an order of magnitude for failed ideas.
