---
status: active
date: 2026-08-29
scope: Cycle89 Forecast Strategy full-month cross-month development validation
purpose: determine whether C3 DIRMO mixed-signal behavior survives complete months before spending effort on recursive block strategies
---

# Cycle89 Full-Month Cross-Month Validation Protocol

## 1. Why this validation is necessary

DEV14 is useful for fast causal screening, but it contains only seven target days from June and seven from July. It cannot establish complete-month robustness, month-to-month stability, or whether a candidate merely benefits from a small set of favorable days.

Current frozen findings are:

```text
C0 DIRECT_H34
raw                  67.86%
micro balanced       65.56%
MAE                  84.85
daily macro balanced 52.01%
collapse             7/14
transition F1        0.1677

C3 DIRMO 10+12+12
raw                  63.39%
micro balanced       61.50%
MAE                  84.32
daily macro balanced 51.31%
collapse             5/14
transition F1        0.1000
```

C3 is therefore not a DEV14 winner. It provides only two potentially useful signals: slightly lower MAE and fewer majority-collapse days. At the same time it loses raw accuracy, balanced accuracy, daily-macro balanced accuracy and transition F1, while using about three times the total model parameters and three forward passes per target day.

This is a legitimate `DIRMO_MIXED_SIGNAL`, not a promotion.

Before concluding that block decomposition is unhelpful, one complete-month cross-month validation is warranted because majority-regime composition varies substantially across months. This validation is still development evidence; it is not confirmation and not final holdout evidence.

---

## 2. Candidate set is frozen

Only the following models are allowed:

```text
C0_DIRECT_H34
C3_DIRMO_10_12_12
Cycle88 primary strict LightGBM comparator
```

Do not rerun or promote:

```text
C1_GAP_DIRECT_D24
C2A_BRIDGE_TF
```

They have already shown clear DEV14 regression and are archived as rejected formulation variants.

Do not implement in this run:

```text
C2B / C2C
alternative DIRMO block sizes
RecMO
Recursive H1
scheduled sampling
new features
directional loss
history / validation search
model-size search
AMP
warm start
```

The scientific question for this run is only:

> Does independent direct-block decomposition provide a robust complete-month advantage over the one-shot C0 DIRECT_H34 baseline?

---

## 3. FULLDEV5 month registry

To obtain genuine complete-month evidence while preserving the pre-registered CONFIRM21 panel, use exactly five complete months:

```text
2026-01-01 .. 2026-01-31   31 days
2026-02-01 .. 2026-02-28   28 days
2026-04-01 .. 2026-04-30   30 days
2026-06-01 .. 2026-06-30   30 days
2026-07-01 .. 2026-07-31   31 days
```

Total:

```text
150 target days per strategy
3,600 scored D-day hours per strategy
7,200 new C0+C3 scored predictions before comparator rows
```

### Why March, May and August are excluded

The existing next-stage protocol pre-registered CONFIRM21 using dates from:

```text
March 2026
May 2026
August 2026
```

Running complete NBEATSx months there now would reveal candidate performance on those confirmation months and would contaminate the confirmation role. Therefore FULLDEV5 deliberately excludes all three months, not merely the 21 registered dates.

September remains the untouched final future lockbox.

---

## 4. Two reporting strata

FULLDEV5 must be reported both as one five-month panel and as two transparent strata.

### 4.1 UNSEEN_FULL3

```text
January
February
April
```

These months were not used in the Cycle89 DEV14 strategy selection. Once this run is executed they become development evidence, but before this run they provide an important out-of-month generalization check.

### 4.2 SEEN_FULL2

```text
June
July
```

These months contain DEV14 dates already used for model development. Full-month results remain useful but must not be described as unseen validation.

Every headline report must show:

```text
FULLDEV5
UNSEEN_FULL3
SEEN_FULL2
```

separately.

---

## 5. Frozen C0 and C3 scientific contract

Both strategies use the same frozen business contract:

```text
target                  DA - RT
forecast origin         D-1 14:00
training labels         <= D-2
training history        36 calendar months
validation              latest 28 legal daily origins
backcast                168 h
features                CORE5 + calendar only
loss                    MAE
optimizer               Adam
seed                    42
precision               float32
AMP                     off
daily recalibration     independent cold retrain
```

C0:

```text
one shared direct model
168 -> H34
bridge offsets 1..10 diagnostic
D-day offsets 11..34 scored
```

C3:

```text
three independent direct blocks
B0 = bridge10
B1 = D-day h1..h12
B2 = D-day h13..h24
same formal origin
no predicted block feedback
```

Do not alter architecture, hidden size, LR, training steps or block boundaries during FULLDEV5.

---

## 6. Artifact reuse rule

Existing DEV14 C0/C3 target-day runs may be reused only when all of the following match the current requested run:

```text
config SHA256
source-code SHA256
canonical-data SHA256
seed
device policy
precision
training history
validation days
feature profile
strategy id / block definition
strict leakage status
```

If any field differs, retrain that day from scratch.

The FULLDEV5 manifest must record for every target day:

```text
REUSED_HASH_VERIFIED
or
COLD_RETRAIN_NEW
```

No silent artifact reuse is allowed.

---

## 7. Strict execution gates

Before training:

1. run Cycle89 pytest;
2. run project preflight;
3. verify holdout registry;
4. verify all 150 dates exclude CONFIRM21 months and September;
5. verify Cycle88 comparator artifact/hash;
6. verify deterministic CUDA policy;
7. verify C0/C3 frozen config hashes.

Every target day must independently pass:

```text
origin alignment
training cutoff <= D-2
future covariate availability
D-1 post14 counterfactual audit
target-day actual counterfactual audit
holdout untouched assertion
```

Headline must contain exactly 24 D-day rows per target day.

Any failed day blocks aggregate publication until repaired or explicitly marked invalid.

---

## 8. Required monthly metrics

For each of the five months and each model, report complete-month pooled hourly metrics:

```text
n
raw direction accuracy
positive recall
negative recall
balanced accuracy
all-positive baseline
all-negative baseline
MAE
RMSE
```

Also report daily-macro distributions within each month:

```text
raw mean / median / std / p10
balanced mean / median / std / p10
positive recall mean
negative recall mean
minority recall mean
MAE mean / median / std
```

And structural diagnostics:

```text
majority-collapse day count/rate
actual sign-switch count
predicted sign-switch count
transition precision
transition recall
transition F1
H34 / D24 horizon diagnostics
```

C3 additionally reports each block's monthly MAE and balanced accuracy.

---

## 9. Cross-month aggregation

Do not rely on one pooled 3,600-hour score.

Report three aggregation levels.

### 9.1 Hour-micro FULLDEV5

Pool all 3,600 D-day hours for each strategy.

### 9.2 Day-macro FULLDEV5

Treat all 150 days as equal units.

### 9.3 Month-macro FULLDEV5

Compute each metric independently inside each month, then equally average the five monthly values.

This is the primary cross-month robustness layer.

At minimum report:

```text
month-macro mean
month median
month std
worst month
best month
month win/tie/loss count
```

for:

```text
raw
balanced
MAE
daily-macro balanced
collapse rate
transition F1
```

A candidate that wins pooled hours but collapses in multiple months is not considered robust.

---

## 10. Same-date paired comparisons

For every day calculate:

```text
C3 - C0
C0 - Cycle88
C3 - Cycle88
```

for:

```text
raw
balanced
MAE
minority recall
transition F1 when defined
```

Then report:

```text
mean paired delta
median paired delta
win / tie / loss days
95% paired bootstrap CI by target day
```

Monthly paired deltas must also be reported.

Cycle88 predictions must be subset to exactly the same calendar dates. Never compare a NBEATSx partial month to a Cycle88 full month in the same table cell.

---

## 11. C3 full-month promotion gate

C3 has greater complexity and cost than C0, so it must earn promotion through robust structural gain, not merely a tiny MAE improvement.

C3 is considered a `DIRMO_FULLMONTH_POSITIVE` only if all hard safety conditions pass and the evidence satisfies most of the following:

```text
G1 month-macro daily balanced >= C0 + 1.0 pp
G2 overall collapse rate < C0 and lower/equal in at least 3/5 months
G3 transition F1 >= C0 overall and no broad unseen-month collapse
G4 month-macro raw >= C0 - 1.0 pp
G5 month-macro MAE <= C0 * 1.03
G6 UNSEEN_FULL3 does not show systematic regression: at least 2/3 months non-inferior on the main robustness picture
```

Because C3 uses roughly three times the total parameters and three forward passes, a result such as:

```text
MAE -0.5
Raw -4 pp
transition worse
```

is not enough for promotion.

Possible decisions:

```text
DIRMO_FULLMONTH_POSITIVE
DIRMO_FULLMONTH_MIXED
DIRMO_FULLMONTH_REJECT
```

---

## 12. What happens after FULLDEV5

### If C3 is positive

Freeze the formulation as DIRMO for later research. Do not touch CONFIRM21 yet. Move to the next scientific direction only after recording the full-month evidence.

### If C3 is mixed/rejected

Archive C3 as a useful negative/mixed result and restore C0 as the forecast-formulation baseline.

Then, and only then, allow one final formulation hypothesis:

```text
C4_RECMO_10_12_12
```

using exactly the same block boundaries as C3 so that the principal difference is predicted-block feedback.

C4 first runs on DEV14 only. It earns a full-month test only if DEV14 shows a meaningful structural benefit.

### If C4 also fails

Close the Forecast Strategy direction. Do not run Recursive H1 merely because it exists. Move to the next independent scientific direction: Loss / Objective.

This implements the project rule:

> investigate one mechanism deeply enough to reach diminishing returns, then stop and switch mechanisms rather than endlessly proliferating variants.

---

## 13. Required artifacts

Recommended root:

```text
runs/full_month_cross_month_dev5/
```

Required aggregate files:

```text
full_month_manifest.json
monthly_micro_metrics.csv
monthly_daily_macro_metrics.csv
monthly_structure_metrics.csv
cross_month_micro_metrics.json
cross_month_daily_macro_metrics.json
cross_month_month_macro_metrics.json
unseen_full3_metrics.json
seen_full2_metrics.json
paired_daily_deltas.csv
paired_monthly_deltas.csv
paired_bootstrap_ci.json
majority_collapse_by_month.csv
transition_by_month.csv
horizon_by_month.csv
runtime_cost_by_month.csv
cycle88_comparator_manifest.json
full_month_cross_month_review.md
```

Every model/date keeps independent strict-run artifacts and provenance.

---

## 14. Final report must answer these questions

1. What are C0, C3 and Cycle88 complete-month raw/balanced/MAE results for Jan, Feb, Apr, Jun and Jul?
2. Does C0's DEV14 ~68% raw survive complete months, or was DEV14 optimistic?
3. Does C3's lower-collapse signal survive complete months?
4. Does C3 recover transition F1 when many more days are observed?
5. Does C3 improve unseen Jan/Feb/Apr months or only months already used for development?
6. Which model has the best month-macro balanced accuracy?
7. Which model has the best worst-month balanced accuracy?
8. Which model has the lowest cross-month variance?
9. Does either NBEATSx strategy beat strict Cycle88 across most months rather than in only one favorable month?
10. Is the extra ~3x DIRMO model capacity/runtime justified?
11. Should C3 be frozen, rejected, or remain mixed?
12. Is one C4 RecMO DEV14 experiment scientifically justified after this evidence?

---

## 15. Interpretation discipline

FULLDEV5 is intentionally broader than DEV14 but is still development data because its results will influence the next research decision.

Do not call it:

```text
final holdout
untouched confirmation
final paper headline
```

CONFIRM21 remains reserved for later top-candidate confirmation, and September remains the final future lockbox.

The purpose of FULLDEV5 is to prevent the project from choosing forecasting strategies based on 14 scattered days when the actual deployment objective requires month-to-month stability.
