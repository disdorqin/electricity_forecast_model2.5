# Research Decisions — NBEATSx Spread24 v1

status: active  
date: 2026-08-29  
responsibility: record frozen choices and deliberately deferred research branches  
validation basis: literature review + Cycle 88 local evidence

## D1. First version does not split 1-8 / 9-16 / 17-24

Decision: **one shared model, one joint output path**.

Rationale:

- NBEATSx is a multiple-output sequence model whose main advantage is shared representation and residual basis expansion across the complete horizon.
- Previous local TimeMixer structure experiments showed the unified 24-output model was stronger than several segmented variants.
- Segment-specific models remain a later ablation if the unified model has a persistent, reproducible period-specific failure.

The evaluator may still report 1-8 / 9-16 / 17-24 diagnostics; they do not alter v1 training structure.

---

## D2. Forecast-origin gap: choose direct H=34, do not fill D-1 h15-h24

### Real problem

At forecast origin D-1 14:00, the last observed spread is D-1 h14. The final business requirement is D h1-h24. There are ten unknown hourly spread values between origin and the target day:

```text
D-1 h15 ... h24   = 10 unknown bridge points
D   h01 ... h24   = 24 scored target points
```

### Candidate A — gap fill / recursive bridge

Possible methods include:

- persistence / interpolation / seasonal fill;
- a separate bridge model predicting h15-h24, then treating those predictions as inputs;
- recursive one-step prediction from h15 onward;
- masked reconstruction / denoising training.

Advantages:

- preserves a conventional 24-hour target horizon after a synthetically completed D-1 trajectory;
- can be useful if a downstream architecture fundamentally requires contiguous observed target history ending immediately before D h1.

Risks:

1. synthetic bridge errors become downstream inputs;
2. training often sees true bridge values while inference sees predicted bridge values, creating train/inference mismatch unless specially designed;
3. recursive strategies can accumulate error over the 10-step bridge;
4. a second model/component makes attribution harder;
5. engineering mistakes can accidentally leak true D-1 h15-h24 values.

This branch is retained as research only: `R_GAP_FILL`.

### Candidate B — direct multiple-output H=34

Define the model output as:

```text
origin D-1 14:00
  -> D-1 h15 ... h24  (10 auxiliary outputs)
  -> D   h01 ... h24  (24 scored outputs)
```

Total H=34.

Advantages:

1. no predicted target is fed back as if it were observed;
2. one forward pass jointly learns dependencies across all 34 future steps;
3. bridge values receive real historical supervision during training without becoming inference-time target inputs;
4. NBEATSx naturally supports a fixed multi-output forecast horizon and future exogenous trajectories;
5. leakage boundary is simpler: the target backcast ends exactly at D-1 14:00.

Literature on multi-step forecasting consistently identifies recursive error accumulation as a central weakness of iterated forecasting, while multiple-output forecasting predicts the future vector jointly and can preserve inter-horizon dependence. See Chevillon (2007), Ben Taieb et al. (2012), Ben Taieb & Hyndman (2014), and modern deep-forecasting surveys.

### Frozen choice

**v1 = Candidate B, direct multiple-output H=34.**

The 10 bridge outputs are auxiliary diagnostics/training targets. Headline project metrics are calculated only on the final 24 D-day outputs.

---

## D3. Paper reproduction and project adaptation are separate scientific stages

The phrase “strictly reproduce the paper” has a precise meaning here.

### Stage P — paper reproduction

Must preserve the published EPF setup:

- autoregressive input length L=168;
- forecast horizon H=24;
- NBEATSx-G and/or NBEATSx-I stack logic;
- original temporal-convolution exogenous encoders (TCN/WaveNet search space), not a simplified API approximation;
- two FC layers per block;
- paper hyperparameter ranges;
- MAE training loss;
- Adam optimizer;
- StepLR-style learning-rate decay by 0.5 three times;
- maximum 30,000 optimization iterations;
- validation every 100 iterations;
- early stopping after 10 non-improving validation evaluations;
- gradient clipping at 1.0;
- paper EPF dataset/split and official reproduction comparison.

The official repository exposes a complete Dataset/DataLoader and hyperopt pipeline; our implementation must be parity-tested against it rather than merely calling a current high-level library class.

### Stage B — business adaptation

Only after the NBEATSx core passes structural/numerical parity checks do we change:

- target: electricity price -> DA-RT spread;
- origin: paper day-ahead boundary -> project D-1 14:00;
- H: 24 -> 34;
- training-history policy -> strict rolling 9 months;
- covariates -> project origin-visible primitives;
- metrics -> direction/recalls/balanced/MAE plus bridge diagnostics.

This result is labelled `BUSINESS_ADAPTATION`, never `PAPER_REPRODUCTION`.

---

## D4. Main v1 features are not B208

Decision: **do not feed all 208 engineered LightGBM features into the first NBEATSx model.**

This is a deliberate model-aware decision.

### Why

NBEATSx was designed to combine:

- a raw autoregressive target backcast;
- time-varying exogenous trajectories extending through the forecast horizon;
- basis-expansion/residual blocks;
- temporal convolution for exogenous effects in the generic configuration.

Many B208 variables manually summarize information that the sequence model already receives in higher-resolution form:

- F0 lag/rolling spread -> raw 168h target backcast;
- F1/F8 D-1 context statistics/raw p1-p14 -> same raw backcast, which ends at h14;
- F4 ramps -> temporal convolution can learn changes/ramps;
- F9 target-day mean/std/min/max -> full future exogenous trajectories preserve the shape before aggregation;
- many F5/F6 summaries -> useful later, but they would obscure whether NBEATSx’s native inductive bias works.

### CORE5 profile

The first business model uses five primitive forecast trajectories:

1. `fcast_直调负荷`
2. `fcast_联络线受电负荷`
3. `fcast_风电总加`
4. `fcast_光伏总加`
5. `fcast_竞价空间`

plus hour/day-of-week calendar information.

Why not `fcast_新能源总加` in CORE5? It is highly redundant with wind + solar. Why retain bidding space? It is a compact market-balance quantity containing information from several supply categories and is directly tied to the market mechanism.

The current Cycle 88 cube contains complete values for the principal forecast fundamentals, but **stored availability is not sufficient proof of origin availability**. A dedicated D-1 14:00 availability audit is mandatory for the complete 34-step covariate horizon.

### Later feature profiles

- `CAUSAL_RICH`: selected F3 business ratios + carefully audited historical forecast-error/uncertainty state.
- `B208_COMPAT`: the LightGBM-style engineered feature set, only as an ablation.
- `SELECTED_FEATURES`: eventual handoff from the independent feature-selection study, evaluated after CORE5 establishes the native NBEATSx baseline.

---

## D5. Loss design sequence

Strict paper reproduction and the first business architecture test both use MAE first.

Reason: if the first run simultaneously changes architecture, horizon, features and a custom directional loss, a positive or negative result cannot be attributed.

Run order:

1. `PAPER_MAE`: exact paper reproduction;
2. `B34_MAE`: same verified core, local H=34 task, MAE;
3. `B34_BUSINESS_LOSS`: only after B34_MAE, add differentiable direction/positive/negative terms.

True direction accuracy/positive recall/negative recall are nondifferentiable metrics. A future business loss must use differentiable surrogates while the actual metrics remain the checkpoint/validation criteria.

Cycle 88 already showed that a stronger direction term can improve one month and damage another, so the business loss must be a controlled ablation rather than an assumption baked into v1.

---

## D6. Training history

Business v1 freezes the rolling training history to **9 months**.

Local Cycle 88 6/9/12-month validation showed 9 and 12 months had nearly identical composite scores, while 9 months is cheaper and less exposed to old-regime drift. The 6-month and 12-month histories remain explicit later ablations.

This is separate from L=168: 9 months is the pool of daily training examples; 168 hours is the target backcast supplied to each example.
