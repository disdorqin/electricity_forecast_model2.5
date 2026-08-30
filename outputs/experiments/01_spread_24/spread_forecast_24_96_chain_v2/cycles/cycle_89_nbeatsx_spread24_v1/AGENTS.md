# Cycle 89 — NBEATSx Spread24 v1 local rules

status: active  
date: 2026-08-29  
scope: this directory and children  
parent rules: inherit repository `AGENTS.md` and `spread_forecast_24_96_chain_v2/AGENTS.md`

## 1. Frozen scientific contract

- Task sign in this cycle is **DA - RT**, not the parent chain's legacy RT - DA sign.
- Forecast origin is **D-1 14:00**.
- D-1 realized spread/RT is visible only through h1-h14.
- D-1 h15-h24 realized spread/RT is forbidden as model input, imputation teacher at inference, calibration input, feature-selection input, early-stopping input, or routing input.
- Target-day actual DA/RT/fundamentals are labels/audit only and are forbidden as features.
- Complete supervised labels used to forecast D must be from D-2 or earlier.
- Final holdout remains untouched until model, loss, feature profile and training protocol are frozen.

## 2. v1 modeling decision

- v1 uses one shared NBEATSx model; **no separate 1-8 / 9-16 / 17-24 models**.
- v1 business horizon is a direct multi-output horizon from D-1 14:00 to D 24:00:
  - bridge outputs: D-1 h15-h24 (10 points)
  - scored business outputs: D h1-h24 (24 points)
  - total horizon: 34 points
- v1 MUST NOT synthesize/fill D-1 h15-h24 and feed those synthetic targets back as observed target history.
- The alternative gap-fill/recursive strategy remains a documented research branch only.

## 3. Paper reproduction before business adaptation

The program has two explicit profiles and must never conflate them:

1. `paper_repro_*`
   - faithful NBEATSx electricity-price reproduction profile
   - L=168, H=24
   - paper architecture/search space, MAE, Adam, StepLR, original validation logic
   - no project-specific directional loss
2. `business_strict34_*`
   - same verified NBEATSx core adapted to DA-RT spread
   - L=168, H=34
   - daily origin anchored at D-1 14:00
   - strict D-2 supervision and project metrics

A business result must never be labelled a paper reproduction result.

## 4. Feature policy

The main v1 does NOT use B208 as its primary input. NBEATSx should first be tested in the structured form for which it was designed:

- autoregressive target: raw historical DA-RT spread, ending at D-1 14:00
- temporal exogenous primitive forecasts: a compact, auditable set of raw forecast trajectories
- calendar variables known in advance

Main v1 primitive future/history covariates:

- `fcast_直调负荷`
- `fcast_联络线受电负荷`
- `fcast_风电总加`
- `fcast_光伏总加`
- `fcast_竞价空间`
- calendar hour/day-of-week encodings

Do not include `fcast_新能源总加` in the core profile because wind+solar already encode it; do not include F4/F9 because full trajectories allow the temporal encoder to learn ramps/profile shape; do not include F1/F8 because the raw target backcast already contains the visible D-1 target context; do not include F5/F6 until a later controlled ablation.

`B208_COMPAT` is an ablation profile only.

## 5. Leakage gates required before any training

Every run must produce and pass:

- origin alignment audit
- horizon-34 timestamp alignment audit
- feature availability audit for all 34 future covariate rows
- training label cutoff audit (`<= D-2`)
- D-1 post-14 counterfactual audit
- target-day actual counterfactual audit
- final-holdout untouched assertion

If any audit fails, status is `INVALID-LEAKAGE` and training must not start.

## 6. Result reporting

Always report on the scored D-day 24 points:

- direction accuracy
- positive recall
- negative recall
- balanced accuracy
- all-positive/all-negative baselines
- MAE/RMSE
- month-level results

Bridge h15-h24 metrics are diagnostic and must never be mixed into the headline D-day score.
