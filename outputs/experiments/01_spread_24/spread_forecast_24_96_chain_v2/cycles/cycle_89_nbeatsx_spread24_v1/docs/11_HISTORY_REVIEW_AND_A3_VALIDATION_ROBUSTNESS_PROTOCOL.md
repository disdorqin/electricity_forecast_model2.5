---
status: active
date: 2026-08-29
scope: independent review of Phase-A history study and one-variable A3 robustness follow-up
purpose: decide whether the strong 36-month magnitude/raw gains can be retained while reducing majority collapse by widening the chronological validation window before any feature or rollout experiment
---

# Cycle89 — History Review and A3 Validation-Robustness Protocol

## 1. Independent review conclusion

Phase A was executed correctly enough to support scientific interpretation:

- Cycle89 pytest: 50 passed on independent rerun;
- project preflight: 14/14 PASS on independent rerun;
- 42 daily cold-retrain target-day runs were reported STRICT/PASS;
- A0/A1/A2 used the same DEV14 dates, architecture, features, loss, seed and direct-H34 strategy;
- training counts increase as intended from 245 -> 702 -> 1068 daily origins;
- final holdout/CONFIRM21 were not used for selection.

However, the Phase-A report's recommendation to immediately freeze A0=9m is **too conservative relative to the wording and intent of the pre-registered protocol**.

The original rule said a longer-history candidate is meaningful if it satisfies **most** of:

```text
daily-macro balanced >= A0 + 2pp
micro raw             >= A0 - 1pp
MAE                   <= A0 * 1.05
majority collapse     <= A0
gain not concentrated in <=2 days
```

A2=36m does not dominate A0, but neither does A0 dominate A2. A2 has a clear magnitude/raw advantage and a clear robustness trade-off. That should be treated as a **mixed positive long-history signal**, not as evidence that long history failed.

## 2. Verified Phase-A evidence

### Aggregate DEV14

| history | train origins | micro raw | micro balanced | MAE | daily-macro balanced | collapse days | median best step |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 9m | 245 | 57.14% | 55.05% | 96.71 | 50.27% | 3 | 150 |
| A1 24m | 702 | 61.61% | 58.64% | 86.75 | 51.24% | 5 | 287.5 |
| A2 36m | 1068 | 67.86% | 65.56% | 84.85 | 52.01% | 7 | 400 |

Compared with A0, A2 changes:

```text
micro raw              +10.71 pp
micro balanced         +10.51 pp
MAE                     -11.85
macro daily balanced    +1.74 pp
collapse days            3 -> 7
```

The paired A2-vs-A0 daily evidence is broad rather than concentrated in one or two days:

```text
raw improves on          9 / 14 days
balanced improves on     6 / 14 days
MAE improves on         11 / 14 days
median daily delta raw   +8.33 pp
median daily delta MAE   -13.47
```

Against the strict same-date Cycle88 primary LightGBM comparator, A2 reaches:

```text
mean paired delta raw        +5.95 pp
mean paired delta balanced   -4.79 pp
mean paired delta MAE        +0.11  (approximately tied)
raw win/tie/loss              6 / 4 / 4
```

Thus 36m materially changes the neural model's behavior and cannot be discarded as a null result.

## 3. What still goes wrong at 36m

Longer history increasingly teaches coarse sign prevalence/regime but does not reliably improve within-day minority behavior:

```text
majority-collapse days:  3 -> 5 -> 7
macro minority recall:   ~0.230 -> ~0.152 -> ~0.145
transition recall:        0.147 -> 0.128 -> 0.084
```

A2's transition F1 is numerically higher because precision changes, but transition recall becomes worse. Therefore the raw gain is not yet equivalent to stronger intra-day sign-structure learning.

The majority diagnostic also improves in one sense:

```text
mean raw-minus-daily-majority:
9m   -14.88 pp
24m  -10.42 pp
36m   -4.17 pp
```

A2 is closer to the daily majority benchmark and exceeds it on one day, but it still often achieves raw performance by leaning strongly toward one sign.

## 4. Why the next experiment should be A3, not features yet

The 36m candidate is trained on about 1068 daily origins but still selects its checkpoint using only the latest 28 days.

That means model estimation becomes much more diverse while model selection remains highly local in regime/time. A 28-day validation slice can select a checkpoint that is excellent for recent MAE but biased toward one recent sign regime.

The original protocol already pre-registered an optional follow-up:

```text
A3_HISTORY36_VAL84
training history = 36 months
chronological validation = latest 84 legal target days
```

A3 is therefore the cleanest next scientific test because it changes exactly one thing relative to A2: validation horizon.

It tests the hypothesis:

> The 36m data volume is useful, but the 28-day early-stopping/selection window is too narrow to select a robust checkpoint for a high-capacity neural model.

No feature, architecture, loss or forecasting-strategy change is permitted in A3.

## 5. Frozen A3 configuration

```text
ID                         A3_HISTORY36_VAL84
Target                     DA - RT
Origin                     D-1 14:00
Backcast                   168 h
Horizon                    direct H34
Bridge                     10
Scored                     24
Training history           36 calendar months
Validation                 latest 84 complete legal target days
Expected train origins     about 1012 (varies only with exact calendar/data completeness)
Feature package            CORE5 + calendar
Architecture               Identity -> Exogenous-TCN
Blocks                     [1,1]
Hidden                     [256,256]
TCN channels               8
Kernel                     3
Loss                       MAE
Optimizer                  Adam
LR                         5e-4
Batch                      32
Clip                       1.0
Seed                       42
Device                     same deterministic CUDA policy
Precision                  float32
Warm start                 forbidden
AMP                        forbidden
Directional loss           forbidden
Feature expansion          forbidden
Rollout                    forbidden
```

Training labels remain <=D-2. Validation is chronological and must also be <=D-2.

## 6. DEV14 only

A3 is evaluated only on the existing DEV14 development panel:

```text
2026-06-01
2026-06-05
2026-06-10
2026-06-15
2026-06-20
2026-06-25
2026-06-30
2026-07-01
2026-07-05
2026-07-10
2026-07-15
2026-07-20
2026-07-25
2026-07-31
```

Do not inspect CONFIRM21. September remains untouched.

## 7. A3 promotion gate — frozen before A3 results

The objective is not to maximize one scalar. A3 must preserve most of A2's long-history gains while improving robustness.

### Hard safety gates

All must pass:

```text
14/14 leakage STRICT/PASS
24 scored rows per day
training and validation labels <= D-2
same config/source/data/device/seed provenance
no warm start
no feature/loss/architecture change
```

### Scientific gate

Prefer A3 as the Phase-B history anchor if it satisfies at least 4 of the following 5 conditions, with no catastrophic per-day failure pattern:

```text
G1 daily-macro balanced >= 52.27%
   (= A0 50.27% + pre-registered 2pp robustness target)

G2 majority-collapse days <= 5
   (must improve materially from A2's 7; A1's 5 is the maximum acceptable first rescue target)

G3 micro raw >= 65.86%
   (= A2 67.86% - 2pp tolerance)

G4 MAE <= 89.10
   (= A2 84.85 * 1.05)

G5 negative/minority robustness improves versus A2:
   pooled negative recall >= 50.70%
   OR daily macro minority recall improves by >= 3pp
```

Also report daily paired deltas against A0, A2 and Cycle88.

### Interpretation

```text
ROBUST_LONG_HISTORY_PASS
  A3 satisfies >=4/5 and reduces collapse.
  Freeze 36m/VAL84 for feature stage.

LONG_HISTORY_MAGNITUDE_ONLY
  A3 preserves raw/MAE but fails robustness rescue.
  Do not treat long history as the sole Phase-B anchor.

LONG_HISTORY_REGRESSION
  A3 loses both A2 magnitude gains and robustness.
  Freeze A0 9m for Phase B.
```

## 8. Why A1=24m is not automatically promoted as a compromise

A1 is numerically intermediate but does not solve the central robustness problem:

```text
collapse 5 > A0 3
macro balanced +0.97pp only
negative recall lower than A0 in pooled results
```

Therefore A1 is retained as evidence but is not an automatic compromise anchor. A3 directly tests the mechanism that could explain A2's trade-off.

## 9. Required A3 artifacts

Create under:

```text
runs/history_window_study/36m_val84/
```

and comparison artifacts under:

```text
runs/history_window_study/comparison_a3/
```

Required:

```text
A3 daily_metrics.csv
A3 micro_metrics.json
A3 macro_metrics.json
A3 majority_collapse.csv
A3 transition_metrics.csv
A3 h34_offset_metrics.csv
A3 training_sample_counts.csv
A3 gradient_summary.csv
A3 paired_vs_A0.csv
A3 paired_vs_A2.csv
A3 paired_vs_Cycle88.csv
A3_review.md
```

## 10. Required tests before running A3

At minimum add/verify:

```text
test_validation_84_is_chronological.py
test_validation_84_cutoff_is_d2.py
test_history36_val84_train_count.py
test_a3_changes_only_validation_length.py
test_a3_no_confirm21_access.py
```

All existing Cycle89 tests and root preflight must remain green.

## 11. Phase-B decision after A3

Do not begin feature experiments in the same execution that produces A3 results.

After A3 review:

### If ROBUST_LONG_HISTORY_PASS

Freeze:

```text
history = 36m
validation = 84d
```

Then run compact features B1/B2/B3 on that anchor.

### If LONG_HISTORY_MAGNITUDE_ONLY

Use a **dual-reference** Phase B rather than pretending one history setting is universally best:

```text
A0 9m/28d  = robustness reference
A3/A2 36m  = magnitude/data-volume reference
```

The first feature package should be tested on the long-history anchor because its purpose is to determine whether added causal state can fix collapse while retaining raw/MAE gains. A0 remains an immutable reference, not a full cross-product feature grid.

### If LONG_HISTORY_REGRESSION

Freeze A0 9m/28d and proceed to compact features.

## 12. Feature-stage plan retained but not executed yet

After the A3 stop/review, Phase B remains:

```text
B0 CORE5 reference
B1 PHYSICAL_SHAPE
B2 CAUSAL_PRICE_STATE
B3 CAUSAL_FORECAST_ERROR_STATE
```

No B208 dump. No B4 shortlist until compact packages are understood.

The key scientific target of Phase B is now sharper:

> Can compact origin-safe state information improve daily-macro balanced/minority recall and reduce majority collapse without surrendering the raw/MAE benefit created by more training history?

## 13. Final instruction for this round

This round is successful when A3 answers the validation-width hypothesis. Accuracy does not need to improve.

Do not:

- run B1/B2/B3;
- run recursive/DIRMO/two-stage rollout;
- modify loss;
- tune hidden size;
- tune learning rate;
- inspect CONFIRM21;
- touch September lockbox.

Stop after A3 review and report which of the three statuses applies:

```text
ROBUST_LONG_HISTORY_PASS
LONG_HISTORY_MAGNITUDE_ONLY
LONG_HISTORY_REGRESSION
```
