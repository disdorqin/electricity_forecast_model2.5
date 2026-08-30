---
status: active
date: 2026-08-29
scope: Cycle89 forecast-formulation research after history and compact-feature stages
purpose: isolate whether the current fixed-origin 168->34 direct formulation is the main cause of weak intraday sign-transition performance, without simultaneously changing history, features, architecture family, or loss
---

# Cycle89 Forecast Strategy Survey and Experiment Protocol

## 1. Current frozen baseline entering this stage

The previous stages have established the following working baseline:

```text
Baseline ID             A2/F0
Target                  spread = DA - RT
Formal origin           D-1 14:00
Backcast                168 hourly spread values ending D-1 h14
Training history        rolling 36 months
Validation              latest 28 legal target days
Features                CORE5 raw forecast trajectories + calendar
Architecture            current NBEATSx business-adapted Identity -> Exogenous-TCN
Hidden                  256 / 256
Loss                    MAE
Optimizer               Adam
Precision               float32
Device                  deterministic CUDA when gate passes
Current strategy        one-shot DIRECT_H34 / MIMO-style
Bridge outputs          offsets 1..10 = D-1 h15..h24
Headline outputs        offsets 11..34 = D h1..h24
```

The compact feature stage did not promote any feature package. Therefore this forecast-strategy stage MUST use the frozen A2/F0 information set. No new feature family, loss, history window, validation width, hidden width, learning rate, or input length may be changed in the same experiment.

The scientific question is now only:

> Does changing the way the 10-hour unknown bridge and 24-hour D-day trajectory are forecast improve daily robustness, minority-sign recognition, sign-transition timing, and numerical accuracy?

---

# PART A — Literature conclusions

## 2. Multi-step forecasting is not one method

The multi-step forecasting literature distinguishes several fundamentally different strategies. The standard taxonomy includes:

```text
Recursive / Iterated
Direct
DirRec
MIMO (multi-input multi-output)
DIRMO (direct multiple-output blocks)
RecMO (recursive multiple-output blocks)
DirRecMO (direct-recursive multiple-output blocks)
```

Key references:

- Chevillon (2007), *Direct Multi-Step Estimation and Forecasting*, Journal of Economic Surveys 21(4), 746–785. DOI 10.1111/j.1467-6419.2007.00518.x.
- Ben Taieb, Bontempi, Atiya & Sorjamaa (2012), *A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition*, Expert Systems with Applications 39(8), 7067–7083. DOI 10.1016/j.eswa.2012.01.039.
- Gasparin et al. (2022), *Deep learning for time series forecasting: The electric load case*, CAAI Transactions on Intelligence Technology. DOI 10.1049/cit2.12060.
- Stratify (2025), *Stratify: unifying multi-step forecasting strategies*, Data Mining and Knowledge Discovery, for a modern unified taxonomy including RecMO / DirMO / DirRecMO.

The literature does not support a universal claim that recursive forecasting is better than direct forecasting. Instead, the strategy changes the bias/variance, dependency, training-sample, and error-propagation trade-offs.

## 3. Current Cycle89 DIRECT_H34 is a MIMO-style strategy

Current Cycle89 performs:

```text
y_hat[1:34] = f(y_history_168, X_history, X_future_34)
```

in one forward pass.

This is MIMO-style multiple-output prediction. It has a major advantage:

- no model prediction is recursively inserted as a future target input;
- therefore classical recursive error accumulation is absent.

Its potential disadvantages in the Cycle89 business task are different:

- one network must jointly learn 34 future target values;
- the first 10 outputs are an unknown bridge but are not headline targets;
- bridge and D-day may have different statistical roles;
- MAE optimization allocates capacity to all 34 outputs;
- the D-day trajectory begins 11 hours after the final observed target.

NBEATSx itself was evaluated in electricity-price forecasting with a direct 168-hour input to 24-hour output setup. Cycle89's H34 business adaptation is therefore a harder and structurally different horizon than the canonical paper setting.

Reference:

- Olivares et al. (2023), *Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx*, International Journal of Forecasting 39(2), 884–900. DOI 10.1016/j.ijforecast.2022.03.001.

## 4. Recursive H1 is useful as a diagnostic, not an assumed solution

The recursive strategy trains a one-step model and repeatedly feeds its predictions back:

```text
y_hat[t+1] = f(history)
y_hat[t+2] = f(history + y_hat[t+1])
...
```

Potential advantage:

- one-step local mapping is simpler;
- many more one-step training windows can be generated;
- temporal dependence is represented through the rolling state.

Main risk:

- prediction errors become future model inputs;
- errors can propagate or amplify over the horizon;
- inference requires 34 sequential calls;
- training/inference distributions can differ if training always sees true previous targets.

Recent direct-vs-recursive electricity-demand experiments also show architecture-dependent error accumulation, especially over intermediate/long horizons. Therefore pure recursive H1 belongs in the experiment matrix, but it should not be the first strategy assumed to win.

## 5. Correct terminology: DIRMO is not recursive block rollout

This stage corrects an earlier shorthand.

### DIRMO

DIRMO splits the horizon into blocks, but each block-specific model predicts directly from the same origin. Previous predicted blocks are NOT fed into the next block.

Example conceptually:

```text
f1(origin) -> h1..h6
f2(origin) -> h7..h12
f3(origin) -> h13..h18
...
```

Advantages:

- preserves local multi-output dependencies inside each block;
- avoids recursive feedback error;
- permits horizon-specialized models.

Cost:

- multiple models/heads;
- higher model-management/training cost;
- inter-block dependence is not explicitly transmitted.

### RecMO

RecMO predicts a block, appends that predicted block to the rolling target history, and predicts the next block using the same shared model.

Example:

```text
origin -> predict block1
history + predicted block1 -> predict block2
...
```

This reduces the number of recursive calls compared with H1 recursion, while retaining some recursive state transfer.

### DirRecMO

DirRecMO uses block-specific models/heads, and later blocks also consume previous predicted blocks. It combines horizon specialization with recursive dependency.

These three must not be conflated because their error propagation mechanisms differ.

## 6. Teacher forcing is legal here, but it creates deployment mismatch

The user's proposed bridge idea is scientifically valid:

> During Stage-2 training, historical true Bridge10 values may be visible; during prediction of D, Stage-2 cannot see the true D-1 h15..h24 bridge and instead receives Stage-1 predicted bridge values.

This is NOT target-day leakage if all Stage-2 training samples satisfy the strict label boundary and the true bridge values belong only to mature historical samples.

For a training sample with target day d <= D-2:

```text
true bridge = d-1 h15..h24
```

is fully historical at the time the final model for D is trained.

The scientific issue is instead:

```text
training context  = perfect historical bridge
inference context = imperfect predicted bridge
```

This is teacher forcing / exposure mismatch.

Bengio et al. (2015) introduced scheduled sampling to reduce analogous training/inference mismatch in sequence prediction. Time-series studies also report that teacher forcing may converge faster but can suffer exposure bias, while free-running training may be slower/less stable. Scheduled sampling is not theoretically guaranteed; Huszár (2015) gives an important consistency critique.

References:

- Bengio et al. (2015), *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*, NeurIPS 2015, arXiv:1506.03099.
- Huszár (2015), *How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?*, arXiv:1511.05101.
- Teutsch & Mäder (2022), *Flipped Classroom: Effective Teaching for Time Series Forecasting*, TMLR.

Cycle89 therefore WILL test teacher-forced bridge training as a legitimate empirical candidate. It is not disqualified in advance. If it produces strict fixed-origin OOS gains and later survives confirmation, it can be promoted while being accurately labeled `TF_TRAIN / PREDICTED_BRIDGE_INFERENCE`.

However, a deployment-matched cross-fitted version should be tested if the teacher-forced variant shows signal, so that the mechanism is understood.

## 7. A missing but important baseline: skip the bridge entirely

Direct multi-step forecasting does not mathematically require predicting every intermediate point before the target horizon of interest.

Therefore Cycle89 should test a gap-direct formulation:

```text
observed target history ends D-1 h14
forecast target vector = D h1..h24 only
```

The 10 intermediate target values D-1 h15..h24 are neither predicted nor used.

Future exogenous covariates for D-day remain legal if known at the formal origin.

This experiment answers a simple question:

> Is the auxiliary Bridge10 itself helping, or is forcing the network to model those ten non-headline targets consuming capacity and destabilizing the actual D24 task?

This is a direct-strategy experiment and has no recursive error propagation.

---

# PART B — Scientific contract

## 8. Frozen causal contract for every strategy

All candidates MUST retain:

```text
Formal headline origin        D-1 14:00
Target sign convention        DA - RT
Observed target at origin     through D-1 h14 only
D-1 h15..h24 true target      forbidden at target-day inference
D-day actuals                 forbidden at target-day inference
Training labels               <= D-2
History                       36 months
Validation                    28 chronological legal target days
Features                      F0 CORE5 + calendar only
Model family                  current NBEATSx business core unless a strategy mathematically requires a different output head
Loss                          MAE only
Seed                          42
Precision                     float32
Final holdout                 September untouched
CONFIRM21                     untouched until strategy selection is finished
```

No strategy may gain an information advantage at inference.

## 9. Distinguish fixed-origin forecast from operational receding horizon

Two different tasks must never be mixed.

### Fixed-origin task — headline

At D-1 14:00 all 24 D-day predictions must be generated without consuming newly realized D-1 h15..h24 targets.

This remains the only headline task in this stage.

### Receding-horizon operational task — deferred

A separate future experiment may update predictions after h15, h16, etc. become observed in real time.

That can be operationally useful, but it is a different forecast origin and cannot be compared directly with the fixed D-1 14:00 headline.

This stage does NOT run real-time actual-updating forecasts.

---

# PART C — Experiment families

## 10. C0 — frozen reference: DIRECT_H34

```text
Input target history: 168
Output:              34
Bridge:              predicted offsets 1..10
Headline D-day:      offsets 11..34
Calls:               1
Feedback:            none
```

Use existing A2/F0 results as the reference. Do not retrain C0 unnecessarily unless artifact integrity/source hashes require it.

## 11. C1 — GAP_DIRECT_D24

### Hypothesis

The network does not need to spend output/loss capacity on Bridge10 if the real objective is only D24.

### Formulation

```text
Observed history: D-8 ... D-1 h14, L=168
Unknown gap:      D-1 h15..h24, not modeled as targets
Output target:    D h1..h24 only
Future exogs:     D h1..h24 aligned to target timestamps
Output size:      24
Calls:            1
Feedback:         none
```

The model must never silently relabel the 24 output positions as the immediate next 24 chronological hours. The dataset contract must explicitly map each output to D h1..h24.

### Why this is high priority

It is the cleanest formulation change:

- no recursive feedback;
- no teacher forcing;
- no second-stage model;
- fewer output targets;
- directly aligns training loss with headline D24.

Run this FIRST.

## 12. C2 — BRIDGE10 -> DIRECT24 two-stage family

This family explicitly models the missing 10-hour bridge, then conditions the D24 model on a bridge representation.

### Stage 1

```text
Input:  legal 168h history + F0 covariates
Output: Bridge10 = D-1 h15..h24
```

### Stage 2

```text
Input:  legal historical context + Bridge10 representation + D-day legal exogs
Output: D h1..h24
```

The primary scientific question is how Stage-2 should be trained.

### C2A — BRIDGE_TF

Training:

```text
Stage-2 receives TRUE historical Bridge10
```

Inference:

```text
Stage-2 receives Stage-1 PREDICTED Bridge10
```

Status:

```text
LEGAL_TEACHER_FORCING
DEPLOYMENT_MISMATCH_PRESENT
```

This variant is explicitly allowed.

If its DEV14 OOS results are strongly positive, do not reject it merely because teacher forcing was used. Its OOS inference still uses only legal predicted bridge values.

Required diagnostics:

```text
Stage1 bridge MAE
Stage1 bridge sign accuracy
Stage2 D24 metrics
correlation of bridge error with D24 error
performance vs bridge-error quartile
```

This will show whether Stage2 is robust to imperfect bridge estimates.

### C2B — BRIDGE_OOF_MATCHED

Run only if C2A shows meaningful signal OR if C2A strongly fails in a way consistent with bridge mismatch.

Training:

- Generate Stage-1 predictions for historical Stage-2 training samples using rolling/cross-fitted models that did not train on that sample's bridge target.
- Feed these out-of-fold predicted Bridge10 values into Stage-2.

Inference:

- Feed Stage-1 predicted Bridge10.

Status:

```text
DEPLOYMENT_MATCHED
NO_TRUE_BRIDGE_CONTEXT_IN_STAGE2_TRAIN_FEATURES
```

This is more expensive but gives the cleanest training/inference match.

### C2C — BRIDGE_NOISY / RESIDUAL_PERTURBED

Optional only after C2A/C2B evidence.

Train Stage-2 on historical true bridge values perturbed using Stage-1 out-of-fold bridge residuals, or a controlled mixture of true/predicted bridge contexts.

Motivation:

- expose Stage2 to realistic bridge error without requiring every training step to use a fully free-running pipeline;
- analogous in spirit to noise-perturbed recursive strategies and curriculum approaches.

Do not run this in the first bridge screen.

## 13. C3 — DIRMO block-direct family

DIRMO predicts blocks independently from the same formal origin; there is no predicted-block feedback.

For H34, unequal final blocks are permitted in the implementation, but the partition must be frozen before results.

### Recommended first partition

Use a semantic partition rather than a broad block-size hyperparameter sweep:

```text
[10, 12, 12]
```

Interpretation:

```text
Block 1 = Bridge10
Block 2 = D h1..h12
Block 3 = D h13..h24
```

Each block-specific model/head sees the same legal origin context and its aligned legal exogenous covariates.

Why [10,12,12]:

- respects the actual bridge boundary;
- D-day is split only once into two half-day blocks;
- requires only three block predictors;
- avoids arbitrary 6+6+... fragmentation;
- no recursive feedback.

This is `C3_DIRMO_SEMANTIC_10_12_12`.

### Optional alternative block granularity

Only if C3 shows signal and the partition itself appears limiting:

```text
[10, 6, 6, 6, 6]
```

Do not run both partitions immediately unless compute cost is small and they were pre-registered before viewing C3 results.

## 14. C4 — RecMO block-recursive family

RecMO uses one shared block model or compatible shared architecture and feeds predicted blocks into later calls.

Rather than automatically choosing six-hour blocks, Cycle89 uses a business-aware first design:

```text
Block 1: 10h Bridge
Block 2: 6h D1-D6
Block 3: 6h D7-D12
Block 4: 6h D13-D18
Block 5: 6h D19-D24
```

Partition:

```text
[10, 6, 6, 6, 6]
```

This requires five sequential block predictions, not 34 H1 predictions.

Scientific trade-off:

- more local than one-shot H34;
- explicitly carries predicted state forward;
- much less recursive depth than H1;
- error propagation is possible and must be measured.

Required horizon diagnostics:

```text
block index
cumulative feedback depth
MAE by block
balanced by block
transition F1 by block
error-growth slope
```

If error sharply worsens with block depth, close the RecMO idea rather than reducing block size and making recursion deeper.

## 15. C5 — pure RECURSIVE_H1_FREE_RUN

This is now a later diagnostic, not the first recommended candidate.

### Inference

```text
predict 1 step
append predicted target
shift history
repeat 34 times
```

### Training variants

The first valid experiment should not silently train only on perfect one-step histories and then call free-running inference equivalent.

Preferred first implementation:

```text
REC_H1_FREE_RUN_DETACHED
```

Within a historical origin sample:

- roll forward using model predictions;
- predicted values are inserted into subsequent target-history context;
- recursive feedback values may be detached to reduce backpropagation-through-34-step instability;
- accumulate losses across all 34 predicted steps.

This produces training inputs closer to inference inputs.

### Teacher-forced H1 as a diagnostic

A classic one-step teacher-forced model can also be run if needed:

```text
REC_H1_TF_TRAIN_FREE_RUN_INFER
```

but it must be explicitly labeled exposure-mismatched.

### RecNoisy follow-up

If plain recursive H1 performs well at early offsets but deteriorates with horizon, a single follow-up may inject realistic residual noise into training histories. This is motivated by recursive forecasting literature such as `RECNOISY`, which perturbs data to reduce recursive error accumulation.

Do not run scheduled sampling, RecNoisy, and free-run variants all at once.

---

# PART D — Execution sequence and stopping rules

## 16. One research direction, sequential experiments

This entire document is one research direction: **forecast formulation**.

Within it, candidates are run sequentially so that failures stop unproductive branches early.

### Step 1 — C1 GAP_DIRECT_D24

Compare:

```text
C0 DIRECT_H34
vs
C1 GAP_DIRECT_D24
```

If C1 clearly improves D24 robustness and does not damage MAE, it becomes the new reference for later formulation comparisons.

### Step 2 — C2A BRIDGE_TF

Run the user's proposed teacher-forced bridge training exactly as a legitimate experiment.

Compare with C0/C1.

If C2A is positive, continue to C2B OOF-matched to understand whether the benefit survives deployment-matched Stage-2 training.

If C2A is clearly worse and Stage1 bridge error strongly predicts Stage2 failure, C2B may still be run once as a mismatch test. Otherwise close the bridge family.

### Step 3 — C3 DIRMO semantic blocks

Run block-direct `[10,12,12]`.

This isolates whether horizon specialization helps without recursive feedback.

### Step 4 — C4 RecMO semantic blocks

Run only after C3 so that the incremental effect of predicted-block feedback can be attributed.

### Step 5 — C5 Recursive H1

Run only if previous block/direct strategies leave evidence that local autoregressive modeling may help.

Pure H1 recursion is not mandatory if C4 already shows strong error propagation.

## 17. Promotion criteria inside DEV14

The current A2/F0 reference is roughly:

```text
Raw                  67.86%
Micro Balanced       65.56%
MAE                  84.85
Daily Macro Balanced 52.01%
Collapse              7/14
Transition F1         0.1677
```

A strategy is considered a meaningful formulation improvement if it satisfies most of:

```text
Daily Macro Balanced >= 54.0%              preferred +~2pp
Collapse days         <= 5
Transition F1         >= 0.20
Raw                   >= 66.0%              allow small trade for robustness
MAE                   <= 89.1               <= A2 * 1.05
Negative recall       does not collapse
improvement spans several dates
```

No single metric is sufficient.

A candidate with Raw 70% but collapse 10/14 and transition F1 near zero is NOT a successful formulation.

Conversely, a candidate with Raw 66.5%, Macro Balanced 57%, Collapse 3/14, and much better transition F1 may be preferable.

## 18. Pairwise attribution rules

Every comparison must use the exact same DEV14 target dates.

Report paired daily deltas:

```text
Delta Raw
Delta Balanced
Delta MAE
Delta transition F1
Delta minority recall
```

Also report:

```text
win / tie / loss day counts
paired bootstrap intervals
```

For recursive/block-feedback strategies add:

```text
metric vs recursion depth
error-growth slope
cumulative feedback-error correlation
```

## 19. Stop a sub-family when marginal returns are exhausted

Examples:

### Bridge family stop

Stop after C2A if:

- no robust improvement;
- Stage1 bridge error strongly degrades D24;
- no evidence mismatch is the reason.

Run C2B only with an explicit rationale.

### DIRMO stop

Do not search block sizes 2/3/4/5/6/8/12/17.

Run one semantic partition first. Only one additional granularity is permitted if the first result has a clear positive signal.

### RecMO stop

If later blocks show monotonic or strong cumulative error growth and no headline benefit, close the recursive-block branch.

### H1 stop

If H1 shows material horizon error accumulation, do not launch multiple teacher-forcing/scheduled-sampling variants in the same cycle.

Preserve the negative result and move to the next research direction (loss/objective).

---

# PART E — Required implementation safeguards

## 20. Bridge teacher-forcing legality audit

For C2A, write a machine-readable audit proving:

```text
Stage2 training sample target day d <= D-2
true bridge timestamps = d-1 h15..h24
true bridge appears only in historical Stage2 training features
true bridge for requested target day D never appears in inference input
inference Stage2 bridge source = Stage1 prediction artifact only
```

Name the status:

```text
LEGAL_TF_TRAIN_PREDICTED_BRIDGE_INFERENCE
```

Do not call it leakage merely because train/inference inputs differ.

## 21. Cross-fitted bridge audit

For C2B:

Every Stage2 training bridge prediction must come from a Stage1 model whose training labels exclude that sample's Stage1 bridge target.

Record:

```text
stage1_training_last_day
stage1_predicted_sample_day
OOF/cross-fit relationship
bridge_prediction_hash
```

## 22. Recursive feedback audit

For C4/C5 each generated target used as later context must carry provenance:

```text
value_source = MODEL_PREDICTION
model_call_index
predicted_timestamp
```

Any true future target appearing after the formal origin invalidates the run.

## 23. Exogenous alignment audit

Every strategy must map each exogenous vector to its actual forecast timestamp.

This is especially important for:

- GAP_DIRECT_D24, because output index 1 corresponds to D h1, not D-1 h15;
- block models, because each block has a different timestamp slice;
- recursive models, because the input window shifts while exogenous timestamps advance.

Add tests that intentionally shift exogenous timestamps by one hour and require failure.

---

# PART F — Required diagnostics

## 24. Common metrics

Every candidate reports:

```text
Micro Raw
Positive Recall
Negative Recall
Micro Balanced
MAE
RMSE
Daily Macro Raw
Daily Macro Balanced
Daily Macro Minority Recall
Collapse days/rate
Transition precision
Transition recall
Transition F1
```

## 25. Strategy-specific diagnostics

### GAP_DIRECT_D24

```text
D24 offset 1..24 metrics
compare D24 output to C0 H34 offsets 11..34
parameter count difference
training-step difference
```

### BRIDGE family

```text
Bridge10 MAE / Balanced
Stage2 D24 metrics
Bridge error -> D24 error correlation
Bridge-error quartiles
teacher-forced vs predicted-bridge distribution distance
```

### DIRMO

```text
block-level metrics
between-block discontinuity
sign switches at block boundaries
```

### RecMO/H1

```text
feedback depth
error vs depth
sign error vs depth
transition timing vs depth
number of sequential model calls
runtime per target day
```

## 26. Block-boundary artifact check

Block methods can generate artificial discontinuities where one predictor ends and another starts.

For C3/C4 calculate:

```text
abs(pred[h_boundary+1] - pred[h_boundary])
```

and compare with the empirical true transition distribution.

A strategy that improves aggregate MAE by producing unrealistic block jumps is not promoted without investigation.

---

# PART G — DEV14 and confirmation governance

## 27. DEV14 remains the only development panel for this direction

Use exactly the existing 14 dates.

Do not replace bad dates.

All C-family choices are made on DEV14.

## 28. CONFIRM21 remains untouched until the entire formulation direction is complete

Do not open CONFIRM21 after C1 or C2.

First finish the forecast-formulation direction and select at most two candidates:

```text
1 frozen reference
1 best formulation challenger
```

Then run each candidate ONCE on CONFIRM21.

No further strategy tuning is allowed after CONFIRM21 results are viewed.

September remains the final future lockbox.

---

# PART H — Immediate implementation round

## 29. Next AI execution should NOT implement all C variants

The next coding/experiment round must only implement and run:

```text
C0 existing DIRECT_H34 reference
C1 GAP_DIRECT_D24
C2A BRIDGE_TF
```

Why these first:

- C1 tests whether bridge prediction is unnecessary.
- C2A tests the user's bridge-conditioning hypothesis with the simplest legal teacher-forcing implementation.
- Both are non-recursive at D24 headline inference except Stage1->Stage2 bridge transfer.
- Their results determine whether more complex block/recursive strategies are justified.

Do NOT yet run:

```text
C2B OOF bridge
C2C noisy bridge
C3 DIRMO
C4 RecMO
C5 H1 recursive
scheduled sampling
RecNoisy
new features
new loss
new history search
```

## 30. Required first-round outputs

Create:

```text
runs/forecast_strategy_stage1/
  C0_DIRECT_H34_reference/
  C1_GAP_DIRECT_D24/
  C2A_BRIDGE_TF/
  comparison/
```

Required comparison artifacts:

```text
strategy_stage1_daily_metrics.csv
strategy_stage1_micro_metrics.csv
strategy_stage1_macro_metrics.csv
strategy_stage1_transition_metrics.csv
strategy_stage1_majority_collapse.csv
strategy_stage1_paired_deltas.csv
strategy_stage1_horizon_metrics.csv
bridge_stage1_metrics.csv
bridge_error_to_d24_error.csv
strategy_stage1_review.md
```

## 31. First-round decision labels

Only use:

```text
GAP_DIRECT_POSITIVE
BRIDGE_TF_POSITIVE
BOTH_MIXED
NO_FORMULATION_SIGNAL_YET
```

Then decide whether the next single experiment should be:

```text
C2B BRIDGE_OOF_MATCHED
or
C3 DIRMO_SEMANTIC_10_12_12
```

Do not jump to every remaining strategy simultaneously.

---

# PART I — Research interpretation map

## 32. What each possible result would mean

### If C1 GAP_DIRECT_D24 wins

Interpretation:

> The 10 bridge outputs were mainly an optimization burden; direct prediction of the actual business horizon is preferable.

Then future block/recursive experiments should target D24, not automatically recreate H34.

### If C2A BRIDGE_TF wins strongly

Interpretation:

> Bridge state contains useful information for D-day forecasting, and a two-stage decomposition is promising.

Then C2B is required to determine whether the advantage survives deployment-matched predicted-bridge training.

### If C2A wins but C2B fails later

Interpretation:

> Stage2 benefits from a clean bridge representation, but current Stage1 bridge errors create an exposure mismatch.

Then one controlled robustness follow-up such as residual-perturbed bridge training may be justified.

### If C1 and C2 both fail

Interpretation:

> Bridge handling alone is not the dominant bottleneck.

Then move to block strategies C3/C4 rather than repeatedly adjusting the two-stage design.

### If direct block DIRMO helps but recursive RecMO does not

Interpretation:

> Horizon specialization helps, but recursive feedback error is harmful.

### If RecMO helps over DIRMO

Interpretation:

> Carrying predicted state between sub-horizons contains useful dependency information and the error propagation is manageable.

### If all formulation strategies fail

Close forecast-formulation research and move to the next distinct scientific direction:

```text
LOSS / OBJECTIVE ALIGNMENT
```

At that point the evidence for direction-aware loss will be stronger because history, compact feature information, and forecast formulation have each been investigated separately.
