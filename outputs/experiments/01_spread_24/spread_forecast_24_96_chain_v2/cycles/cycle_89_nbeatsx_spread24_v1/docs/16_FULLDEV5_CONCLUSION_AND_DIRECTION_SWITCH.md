---
status: active
date: 2026-08-30
scope: Cycle89 FULLDEV5 conclusion and forecast-strategy branch closure
---

# Cycle89 FULLDEV5 Conclusion and Direction Switch

## 1. Executive conclusion

FULLDEV5 changes the interpretation of Cycle89 more than any prior DEV14 experiment.

The DEV14 development panel had suggested that C0 DIRECT_H34 could reach roughly 67-68% pooled direction accuracy, but the full-month cross-month backtest across Jan/Feb/Apr/Jun/Jul shows that this was not a stable cross-month property.

Frozen FULLDEV5 results:

| strategy | Raw | +Recall | -Recall | Balanced | MAE | Daily Macro Balanced | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 DIRECT_H34 | 52.53% | 55.49% | 47.84% | 51.66% | 99.18 | 50.22% | 46/150 |
| C3 DIRMO 10+12+12 | 52.75% | 57.62% | 45.04% | 51.33% | 103.29 | 50.93% | 33/150 |
| Cycle88 strict LGBM | 56.12% | 70.44% | 33.38% | 51.91% | 87.24 | 51.60% | 37/150 |

Therefore:

1. C3 does not earn promotion over C0.
2. C0 itself does not demonstrate robust cross-month superiority to Cycle88.
3. DEV14 was useful for development, but its headline 67% result was optimistic relative to the full-month distribution.
4. The Forecast Formulation direction has reached diminishing returns and should be closed for the mainline.
5. RecMO / Recursive H1 remain archived hypotheses, not the next mainline experiment.

## 2. Monthly evidence

### C0 DIRECT_H34

| month | Raw | Balanced | MAE |
|---|---:|---:|---:|
| Jan | 50.67% | 52.56% | 111.06 |
| Feb | 46.56% | 47.78% | 94.14 |
| Apr | 55.97% | 55.98% | 92.89 |
| Jun | 53.19% | 51.38% | 105.25 |
| Jul | 55.78% | 50.24% | 92.06 |

### C3 DIRMO

| month | Raw | Balanced | MAE |
|---|---:|---:|---:|
| Jan | 51.21% | 52.60% | 115.86 |
| Feb | 51.80% | 51.21% | 94.94 |
| Apr | 54.17% | 53.77% | 99.18 |
| Jun | 52.50% | 50.60% | 111.62 |
| Jul | 54.03% | 48.25% | 94.18 |

### Cycle88 strict comparator

| month | Raw | Balanced | MAE |
|---|---:|---:|---:|
| Jan | 55.65% | 51.22% | 104.39 |
| Feb | 50.60% | 47.59% | 77.59 |
| Apr | 56.39% | 54.37% | 77.29 |
| Jun | 57.08% | 54.47% | 91.74 |
| Jul | 60.35% | 51.42% | 84.08 |

No NBEATSx formulation is consistently dominant across the five complete months.

## 3. Why DIRMO is closed rather than promoted

C3 has one genuine benefit: it reduces majority-collapse days from 46/150 to 33/150. This suggests that independent block decomposition changes output diversity.

However, it does not convert that benefit into the key final metrics:

- Raw only +0.23 pp versus C0;
- Daily Macro Balanced only +0.71 pp;
- pooled Balanced is slightly lower;
- MAE worsens by 4.11;
- Transition F1 decreases from 0.1227 to 0.1136;
- only 2/6 pre-registered promotion gates pass;
- C3 costs roughly three models / three forward passes and substantially more daily training time.

Thus the lower collapse rate is a diagnostic clue, not enough evidence for promotion.

## 4. What the full-month result says about the current bottleneck

The current bottleneck is no longer adequately described as:

- too little training history;
- too few handcrafted features;
- one-shot H34 output dimensionality alone;
- missing explicit bridge reconstruction alone.

Those hypotheses have already been tested and either produced only partial gains or failed to generalize.

The most persistent failure signature is now:

```text
Daily Macro Balanced ~ 50%
Transition F1 low
minority recall unstable
majority-collapse frequent
MAE-selected checkpoints can look numerically reasonable while sign structure remains weak
```

This points to a mismatch between the training/selection objective and the business target.

Current training asks the model to minimize magnitude error through MAE. The business task ultimately cares strongly about:

- sign correctness;
- balanced positive/negative recall;
- minority-class performance;
- intra-day sign transitions;
- stability across months.

Therefore the next scientific direction should be **Loss / Objective**, while keeping the data/model/forecast formulation frozen.

## 5. Mainline freeze before the next direction

Freeze as the reference model for the next study:

```text
Model: C0 DIRECT_H34
History: 36 months
Validation: 28 chronological legal days
Input: CORE5 + calendar
Backcast: 168 h
Horizon: H34, scored last 24
Architecture: Identity -> Exogenous-TCN, [1,1], hidden 256
Precision: float32
Device: deterministic CUDA
Training: daily cold retrain
```

C3 remains a diagnostic archived candidate only.

Do not combine C3 with a new loss in the first loss study; otherwise attribution is lost.

## 6. Forecast Strategy direction status

Mainline status:

```text
FORECAST_STRATEGY_CLOSED_DIMINISHING_RETURNS
```

Evidence tree:

```text
C0 DIRECT_H34            reference
C1 GAP_DIRECT_D24        rejected
C2A BRIDGE_TF            rejected
C3 DIRMO 10+12+12        mixed on DEV14, not promoted on FULLDEV5
C2B/C2C                  deferred
RecMO                    deferred
Recursive H1             deferred
```

RecMO/H1 should only be revived later if a separate theoretical reason or new evidence specifically targets recursive state propagation. They should not consume the next mainline experiment budget.

## 7. Next direction: Loss / Objective

The next study must change only the loss / checkpoint-selection objective.

Everything else remains frozen at C0.

The initial sequence should be small and interpretable rather than a large weight search:

```text
L0 = current H34 MAE reference
L1 = scored-D24 weighted magnitude loss (bridge auxiliary down-weighted)
L2 = robust magnitude loss, preferably pseudo-Huber
L3 = magnitude + balanced differentiable sign objective
L4 = only if L3 has signal: add a small raw-direction term
```

The first loss experiment should specifically answer whether the 10 bridge points are consuming too much optimization weight before introducing directional surrogates.

A later loss protocol should define exact equations, warm-up schedule, validation score, gradient monitoring, and promotion gates before any result is observed.

## 8. Validation discipline from this point

FULLDEV5 is now consumed development evidence.

Do not repeatedly tune on Jan/Feb/Apr/Jun/Jul and then call those months confirmation.

Still protected:

- CONFIRM21: March / May / August pre-registered dates;
- September: final future lockbox.

Loss-stage screening may use the already-consumed development data, but no more than a small pre-registered candidate set should be selected before CONFIRM21.

Do not access CONFIRM21 until the loss candidate definitions and selection rule are frozen.

## 9. Scientific lesson

The important negative result is:

> Increasing training history, compact feature engineering, gap skipping, bridge teacher forcing, and direct block decomposition did not produce a robust cross-month NBEATSx direction advantage under pure MAE training.

This is useful evidence. It narrows the research question from broad architecture search to whether the model objective is aligned with the spread-direction task.
