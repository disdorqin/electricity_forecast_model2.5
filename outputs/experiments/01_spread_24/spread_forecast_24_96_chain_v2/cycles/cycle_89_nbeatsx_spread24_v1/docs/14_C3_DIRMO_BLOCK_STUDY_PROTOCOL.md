---
status: active
date: 2026-08-29
scope: Cycle89 Forecast Strategy Stage-2 / C3 only
purpose: test whether direct block-wise multi-output decomposition improves intra-day structure without recursive feedback
---

# Cycle89 C3 DIRMO Block Study Protocol

## 1. Why C3 is the next and only experiment

Forecast Strategy Stage-1 completed three strict DEV14 comparisons under the frozen A2/F0 chassis:

- C0 `DIRECT_H34`: current reference, 168 -> 34 in one direct MIMO-style forecast.
- C1 `GAP_DIRECT_D24`: directly skips the unknown 10-hour bridge and predicts D-day 24 points.
- C2A `BRIDGE_TF`: Stage-1 predicts Bridge10, Stage-2 predicts D24; Stage-2 training receives historical true bridge while inference receives predicted bridge.

Stage-1 result: `NO_FORMULATION_SIGNAL_YET`.

C1 materially regressed relative to C0, so simply removing bridge outputs does not solve the task. C2A also regressed, so adding a predicted bridge before one full D24 prediction does not create a robust gain. C2A bridge error is positively associated with D24 MAE, therefore the bridge-prediction channel can propagate error.

However, C1/C2A do **not** answer a different question:

> Is the current single 34-dimensional direct output task too heterogeneous, such that shorter **independent direct multi-output blocks** are easier to learn even without any recursive feedback?

This is the exact question of C3. No other research axis may be changed in this round.

## 2. Frozen causal/scientific contract

All C3 runs retain:

- target: `DA - RT`
- forecast origin: `D-1 14:00`
- realized D-1 target allowed only through h14
- D-1 h15..h24 actual target forbidden as input
- D-day actual price/fundamentals forbidden as input
- supervised labels no later than D-2
- training history: 36 months
- chronological validation: latest 28 legal target days
- backcast: 168 hourly target points
- feature package: F0 CORE5 + calendar only
- loss: MAE
- optimizer: Adam
- seed: 42
- float32, deterministic CUDA if available
- daily cold retrain
- DEV14 only
- CONFIRM21 untouched
- September lockbox untouched

No feature, history, validation-width, loss, optimizer, model-size, AMP or threshold tuning is allowed.

## 3. C3 definition — true DIRMO-style direct blocks

Use the fixed partition:

```text
Block B0: Bridge10       H34 offsets  1..10   = D-1 h15..h24
Block B1: D-day first12  H34 offsets 11..22   = D h1..h12
Block B2: D-day last12   H34 offsets 23..34   = D h13..h24
```

The three blocks are predicted **independently from the same D-1 14:00 origin**.

Critical property:

```text
B0 prediction is NOT fed into B1.
B1 prediction is NOT fed into B2.
No predicted target is recursively appended.
```

Therefore C3 isolates **output-horizon decomposition** from recursive error propagation.

This terminology matters:

- independent direct blocks = DIRMO-style decomposition;
- predicted previous block fed into next block = RecMO/DirRecMO-like strategy and belongs to a later experiment.

## 4. Why use 10 + 12 + 12 first

The partition is frozen before C3 results for three reasons:

1. `10` is dictated by the business gap between D-1 14:00 and end of D-1, not selected from DEV14 performance.
2. D-day is split into two equal 12-hour blocks, avoiding the legacy hand-crafted 1-8 / 9-16 / 17-24 segmentation.
3. The same partition can later be reused in a RecMO experiment, allowing a clean comparison of `independent direct blocks` versus `recursive block feedback` without changing block boundaries simultaneously.

Do not test 6-hour, 8-hour, 4-hour or learned boundaries in this round.

## 5. Implementation options and capacity confounding

### 5.1 Stage-1 operational implementation

The first C3 implementation may use three independent NBEATSx block predictors, each built from the same frozen architecture family but with its forecast head/horizon restricted to its own block.

This deliberately tests the operational question:

> Does specialization by horizon block produce a useful system-level forecast?

### 5.2 Required capacity accounting

Three independent predictors increase total system parameter count and training cost relative to C0. This is a known confound and must be reported explicitly.

Every C3 artifact must record:

- parameter count per block model;
- total deployed parameter count;
- number of forward passes;
- total training wall time if available;
- inference wall time if available.

If C3 shows a meaningful positive signal, **do not immediately declare DIRMO superior**. The next follow-up must be a capacity-control study, e.g. reduced-width block models or a comparably enlarged direct-H34 control, before promotion.

If C3 fails even with the extra capacity, close the direct-block decomposition hypothesis without spending a round on capacity matching.

## 6. Training/evaluation semantics

For each target day D:

1. Construct the legal 36m/VAL28 split ending no later than D-2.
2. Fit all scalers on the legal training partition only.
3. Seed before each block-model construction.
4. Train B0/B1/B2 independently; no warm start or weight transfer.
5. At formal inference, build one origin-safe input snapshot at D-1 14:00.
6. Run the three direct block predictors independently.
7. Concatenate predictions in H34 order: B0 + B1 + B2.
8. Join target-day labels only after forward inference.
9. Headline metrics use D-day B1+B2 = 24 points only.
10. Bridge B0 is diagnostic, not headline.

Each block must use only origin-known future exogenous trajectories corresponding to the timestamps it predicts.

## 7. Required outputs

At minimum write:

- `C3_daily_metrics.csv`
- `C3_micro_metrics.json`
- `C3_macro_metrics.json`
- `C3_transition_metrics.csv`
- `C3_majority_collapse.csv`
- `C3_h34_offset_metrics.csv`
- `C3_block_metrics.csv`
- `C3_runtime_capacity.csv`
- `C3_paired_vs_C0.csv`
- `C3_paired_vs_Cycle88.csv`
- `C3_review.md`

Per target-day artifacts retain strict split, leakage, provenance, checkpoint/model-state and prediction evidence for all three block models.

## 8. Frozen reference values

C0 DIRECT_H34 DEV14 reference:

```text
micro raw              67.86%
micro positive recall  80.41%
micro negative recall  50.70%
micro balanced         65.56%
MAE                     84.85
daily macro balanced   52.01%
majority collapse      7 / 14
transition F1          0.1677
```

Stage-1 failed alternatives are retained only as evidence:

```text
C1 GAP_DIRECT_D24   raw 57.74%, balanced 56.13%, MAE 89.13, macro balanced 45.64%, collapse 5/14, transition F1 0.0595
C2A BRIDGE_TF       raw 61.01%, balanced 60.01%, MAE 90.12, macro balanced 49.38%, collapse 7/14, transition F1 0.0574
```

## 9. C3 success / stop gate

C3 is considered a useful formulation signal only if it improves **intra-day robustness**, not merely pooled raw.

Preferred positive pattern:

- daily macro balanced >= 54.0%;
- transition F1 >= 0.20;
- majority-collapse days <= 5;
- micro raw >= 66.0%;
- MAE <= 89.1;
- gain distributed across multiple DEV14 dates.

Interpretation:

### `DIRMO_POSITIVE_SIGNAL`
At least two of the three structural targets improve materially:

- daily-macro balanced;
- transition F1;
- collapse count;

while Raw/MAE remain within the frozen tolerances.

Then perform a **capacity-control follow-up before promotion**.

### `DIRMO_MIXED_SIGNAL`
Some structural metric improves, but Raw/MAE or another structural metric materially regresses. Preserve the evidence; do not tune block boundaries immediately.

### `DIRMO_NO_SIGNAL`
No credible structural improvement over C0, or the model only benefits from majority-class behavior / extra capacity without improved daily structure.

Then close independent direct-block decomposition and move to the next distinct formulation hypothesis: RecMO block feedback using the **same 10+12+12 partition**.

## 10. What is explicitly forbidden in C3

Do not run:

- C2B / C2C bridge variants;
- alternative block sizes;
- RecMO / DirRecMO;
- Recursive H1;
- scheduled sampling;
- new features;
- directional / pseudo-Huber loss;
- history or validation search;
- model-width search;
- CONFIRM21;
- September lockbox.

## 11. Why C2B is deferred

C2A is already worse than C0 by roughly 6.85pp raw and +5.26 MAE. OOF-matched bridge training could reduce train/inference mismatch, but would need to recover a large deficit. It remains a legitimate deferred diagnostic, not the highest-value next experiment.

If later block-feedback experiments show that predicted-history distribution mismatch is the dominant bottleneck, C2B may be reopened with a stronger mechanistic justification.

## 12. Current monthly-status interpretation

NBEATSx has **not** been run as a complete full-month backtest. Current Cycle89 evidence contains seven pre-registered DEV14 dates in June and seven in July only. These are sampled-month diagnostics, not full-month scores.

For C0 DIRECT_H34 on those sampled dates:

```text
June sample (7 days / 168 scored hours):
raw 67.26%
positive recall 74.19%
negative recall 58.67%
balanced 66.43%
MAE 99.66

daily-macro balanced on the 7 June days: ~50.10%

July sample (7 days / 168 scored hours):
raw 68.45%
positive recall 86.14%
negative recall 41.79%
balanced 63.96%
MAE 70.05

daily-macro balanced on the 7 July days: ~53.91%
```

The high pooled monthly-sample raw/balanced values coexist with weak daily-macro robustness. This is precisely why future formulation experiments must optimize structural stability rather than pooled raw alone.

Do not access March/May/August CONFIRM21 merely to create more monthly-looking numbers. Those dates remain reserved for later confirmation of the final 1–2 DEV14-selected strategies.

## 13. Completion criteria

C3 is complete only after:

- all existing Cycle89 tests pass;
- C3-specific block alignment / no-feedback / D-2 / label-after-forward tests pass;
- project preflight passes;
- all 14 days complete with 24 scored D-day rows each;
- leakage status is STRICT/PASS for every block/date;
- C3/C0/Cycle88 paired comparison is complete;
- transition and majority-collapse diagnostics are complete;
- system capacity/runtime cost is reported;
- one final classification is recorded: `DIRMO_POSITIVE_SIGNAL`, `DIRMO_MIXED_SIGNAL`, or `DIRMO_NO_SIGNAL`.

Stop after C3 review. Do not automatically run RecMO in the same round.
