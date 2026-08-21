# PLAN_96_POINT_MODEL_COMPATIBILITY_AFTER_LOCAL_SYNC

> **Status:** Design only. No model, classifier, ledger, fusion, or prediction
> code was modified in this task. Grounded in the 2026-07-28 local 96-point sync
> (see `DATA_CONTRACT_96.md`, `DATA_QUALITY_96.md`, `LEAKAGE_AUDIT_96.md`).
>
> **Hard gates (owner-approved):** every future model change must pass
> `96-point smoke test + 24-point golden-baseline regression` before merge. Do not
> start R1 automatically — wait for explicit owner approval.

---

## 0. Scope Recap

- Targets (owner-approved native 96-point pair): `epf_unit_data_96.da_cq_price`
  (DA) and `epf_unit_data_96.rt_cq_price` (RT), unit-level, 元/MWh.
- Features: `epf_market_data_96` 13 `fcast_*` + 13 `actual_*` (market grid).
- Local mirror: `data/remote_96/{parquet,raw}/` — already produced and validated.
- The cutoff/anchor logic in `protocol_b_cutoff.py` is **resolution-agnostic**
  and is reused (only the cutoff *period* is re-expressed in 15-minute units).

---

## 1. Golden-Baseline Implementation Order (R0–R7)

| Phase | Work | Exit gate |
|---|---|---|
| **R0** | Complete + validate local 96-point sync | ✅ DONE (this task): 33/33 tests, real sync `ok`, both tables validated. |
| **R1** | Formal 96-point data contract + processed-dataset adapter | Adapter emits joined, coerced, leakage-safe 96-point frames + processed manifest. |
| **R2** | Freeze current 24-point golden baseline | Snapshot metrics of all 7 legs @24-point as the regression reference. |
| **R3** | Introduce `resolution` parameter through the pipeline | `business_day.py` + feature builders accept `resolution`. |
| **R4** | Adapt seven model legs one by one | Each leg: 96-point smoke + 24-point golden regression. |
| **R5** | Adapt ledgers, weighting, fusion, validation, official output | Resolution-aware period buckets; ledger re-weighting at 96-point. |
| **R6** | Implement + retrain native 96-point classifier | Native 15-min negative-price classifier (task §19). |
| **R7** | Dual-resolution integration, companion runs, delivery validation | Both resolutions delivered; parity report. |

---

## 2. Dataset Adapter Plan (R1) — design only

**Join:** `epf_market_data_96` ⋈ `epf_unit_data_96` on `(market_date, period_no)`
(+ `unit_id` when multi-unit). Market is the left table (it extends 11 days past
the unit table); trailing market-only days carry `NaN` targets and are dropped
from training but kept for feature back-fill.

**Rename map to the existing Chinese model-column contract** (illustrative;
adapter translates remote English → the legacy Chinese feature names used by the
24-point models so the model code changes are minimal):

| Remote (96) | Legacy 24-point equivalent |
|---|---|
| `fcast_direct_load` | 直调负荷预测 |
| `actual_direct_load` | 直调负荷实际 |
| `fcast_wind` / `fcast_solar` / `fcast_new_energy` | 风电/光伏/新能源预测 |
| `da_cq_price` | 日前电价 (target DA) |
| `rt_cq_price` | 实时电价 (target RT) |
| derived `hour_business` | `ceil(period_no/4)` |
| derived `quarter_in_hour` | `((period_no-1)%4)+1` |

**Missing-value rules:** `object→float64` coercion (`pd.to_numeric(errors=
"coerce")`); `fcast_*` are 0% null (pass-through); **market** `actual_*` (e.g.
`actual_wind`, `actual_solar`, `actual_direct_load`) ~0.12% null → forward-fill
within day, then inter-day median fallback. **Scope limit:** this fill applies
**only** to market actual-value fields in `epf_market_data_96`; it does **not**
fill `rt_cq_price`, which is a prediction **target** whose post-cutoff truth is
never imputed or used as input (DATA_CONTRACT_96 §4–§5). Targets 0% null.

**Late-start handling:** guard skips any feature before its first valid date
(none observed in current data, but kept for future backfills).

**Forecast-time filtering (LEAKAGE_AUDIT_96 §4):** drop same-period `actual_*`;
mask post-cutoff `rt_*`; exclude `id/create_time/update_time`.

**Target-day placeholder rows:** one `(period_no[, unit_id])` row per target day
with `NaN` targets so the model can score it; historical `actual_*` of the target
day are NOT filled.

**Complete-day validation:** a scored day must contain exactly 96 non-placeholder
target rows.

**Splits:** chronological only; boundaries on business-day midnight; frozen
train/val/test day list persisted in the processed manifest.

**Incremental actual ingestion:** when the remote back-fills recent days, the
adapter re-reads the updated local mirror (R0 incremental sync) and regenerates
only affected processed partitions.

**Local processed output:** `data/processed_96/` (parquet per split) + a
`processed_96_manifest.json` (schema, dtypes, split days, cutoff period, unit
list).

---

## 3. Seven Model-Leg Compatibility Plan (R4) — design only

For each leg: *current* (24-point) contract → *new* (96-point) contract +
files / params / scaling constants / horizon / cutoff / output / CPU-GPU /
retrain / tests / leakage / rollback.

### 3.0 Cross-leg cutoff principle (minimal-change, owner-approved)

- **Do NOT** refactor all RT legs into one unified `protocol_b_cutoff` call. Keep
  each leg's **existing** cutoff implementation; they only **share the single
  business parameter** `realtime_cutoff_hour = 14` / `realtime_cutoff_period = 56`
  (LEAKAGE_AUDIT_96 §3).
- Each RT leg (**RT916 RT, TimeMixer RT, SGDFNet RT, TimesFM RT**) must add an
  explicit **p56 / p57 boundary test**: p56 must be visible & used, p57 must be
  masked / substituted and never leaked as a feature or label at scoring time.
- A unified cutoff module is **future 3.0 technical debt**, out of scope for this
  minimal-change round.

### 3.0 Cross-leg cutoff principle (minimal-change, owner-approved)

- **Do NOT** refactor all RT legs into one unified `protocol_b_cutoff` call. Keep
  each leg's **existing** cutoff implementation; they only **share the single
  business parameter** `realtime_cutoff_hour = 14` / `realtime_cutoff_period = 56`
  (LEAKAGE_AUDIT_96 §3).
- Each RT leg (**RT916 RT, TimeMixer RT, SGDFNet RT, TimesFM RT**) must add an
  explicit **p56 / p57 boundary test**: p56 must be visible & used, p57 must be
  masked / substituted and never leaked as a feature or label at scoring time.
- A unified cutoff module is **future 3.0 technical debt**, out of scope for this
  minimal-change round.

### 3.1 LightGBM DA (`lightGBM/`)
- **Current:** point-wise regression, 24 hourly rows/day, lag_24h/48h/168h,
  `hour_sin/cos`.
- **New (96):** point-wise, 96 rows/day; `lag_96 / lag_192 / lag_672`;
  `quarter_sin/cos` + `hour_business` + `quarter_in_hour`; per-period GBDT.
- **Files:** `lightGBM/main_fix.py`, feature builder.
- **Params:** no arch change; sample count ×4 (96 vs 24).
- **Physical constants:** hourly→quarter multipliers (24→96).
- **Tensor:** N/A (tabular).
- **Horizon:** DA D+1 unchanged.
- **Cutoff:** N/A for DA.
- **CPU/GPU:** CPU (LightGBM GPU optional; see project GPU caveats).
- **Retrain:** full refit on 96-point; golden regression vs R2.
- **Tests:** 96-smoke (one day scores 96 rows) + 24-golden regression.
- **Leakage:** no `actual_*` same-period; `da_cq_price` is label only.
- **Rollback:** keep 24-point wrapper; flag-gate 96 behind `--resolution`.

### 3.2 TimesFM DA (`TimesFMBackend/`)
- **Current:** sequence model, context=24·k hours.
- **New:** context in 15-min ticks (×4); native variable-length OK.
- **Files:** TimesFMBackend config/wrapper.
- **Params:** `context_len`, `horizon_len` in periods (96); patch/horizon rescale.
- **Tensor:** seq len ×4 → memory ×~4; watch GPU VRAM.
- **Horizon:** 96 periods D+1.
- **Cutoff:** N/A DA.
- **Tests/leakage/rollback:** as 3.1.

### 3.3 TimeMixer DA (`TimeMixer/`)
- **Current:** patch-based mixing over 24-point days.
- **New:** patch size / seasonality periods rescaled to 96 (e.g. daily=96,
  weekly=672); patch embedding adapts.
- **Files:** TimeMixer config.
- **Tensor:** seq ×4.
- **Tests/leakage/rollback:** as above.

### 3.4 TimesFM RT (`TimesFMBackend/`) — keep current RT logic (NOT DA_anchor + delta)

> **Correction (2026-07-28):** TimesFM RT must **not** be described or implemented
> as a `DA_anchor + delta` model. It keeps its **current project realtime
> prediction logic**: it runs in **gap** mode (independent of the DA price anchor),
> confirmed at `TimesFMBackend/infer.py:46` (`default_skip_style = "gap" if target
> == "realtime"`) and `price_forecast_copy_分时段预测.py:1088-1090` (the other
> price column is excluded for realtime). `DA_anchor + delta` belongs to the
> **SGDFNet** path (§3.7), **not** TimesFM.

- **Current (24-point):** sequence model, `target="realtime"`, gap skip style;
  predicts RT directly without a DA-price delta.
- **New (96):** only **parametrize** for 96-point — no logic rewrite:
  - input / frequency → 15-minute ticks (×4);
  - `context_len` / `horizon_len` expressed in **periods** (96);
  - time index → `(market_date, period_no)` / `data_time`;
  - cutoff → **p1..p56 visible, p57..p96 masked** per the existing project
    cutoff logic, driven by the shared `realtime_cutoff_period = 56`.
- **Cutoff:** NOT a new unified `protocol_b_cutoff` call; reuse TimesFM's existing
  cutoff handling, just parameterized to period 56.
- **Files:** RT wrapper (period/index rescale only) + existing cutoff handling.

### 3.5 TimeMixer RT (`TimeMixer/`)
- As 3.3 + RT cutoff mask (post-cutoff periods excluded from RT labels at scoring).

### 3.6 RT916 RT (`RT916_SpikeFusionNet/`)
- Spike/fusion net; RT leg. Adapt input frame to 96 periods; cutoff mask; spike
  resolution is naturally finer (15-min) — an advantage for the spike module.

### 3.7 SGDFNet RT (`SGDFNet/src/sgdfnet/`) — preserve current behavior (task §18)
- **DA anchor** comes from the input data's DA-price column (the 96-point
  `da_cq_price`), **not** from official fused DA predictions.
- Missing target-day DA values → **same-period historical median** fallback.
- Do **not** feed official fused DA into SGDFNet; record the intended-vs-actual
  mismatch as **future 3.0 technical debt**.
- **No new RT fill strategy:** the existing same-period historical-median fallback
  for missing target-day DA (`SGDFNet/src/sgdfnet/data_contract.py:_fill_da_anchor_fallback`)
  is preserved **only in its current semantics**. Do **not** extend it into a NEW
  global "DA price + historical median weighted" RT fill for the 96-point round —
  that is deferred to 3.0 / future experiments, not part of this minimal change.
- Adapt only the `period_no` indexing; no redesign of the delta model.

---

## 4. Native 96-Point Negative-Price Classifier Plan (R6) — task §19

**Direction (retained):** native 15-minute training & inference; base point
label `realtime_price <= -50`; configurable `min_negative_duration_periods`
(candidates 1/2/3/4); **initial experimental candidate = 2 periods (30 min)**.

**Construction:**
- Strictly **consecutive** event construction in v1 (no gap-bridging).
- **Stable event IDs** (hash of `market_date + first_period + unit_id`) for
  reproducible metrics.
- **Retrospective correction:** once an event is confirmed, correct all
  qualifying periods within it (point-level + event-level metrics).

**Recalibration:** rebuild OOF features on 96-point; recalibrate thresholds from
15-minute data (the 24-point thresholds do not transfer directly).

**Official output:** classifier-corrected RT is the official 96-point RT.

**Failure behavior (frontend-safe + operational):**
```text
deliver realtime_before_classifier
delivery_status = DEGRADED_DELIVERED
exit code        = 2
classifier_applied_to_official_output = false
official_rt_source = realtime_before_classifier
```
Plus an operational manifest block (status, reason, affected periods) and
developer diagnostic logging (no secrets).

---

## 5. Acceptance Gates (every R4–R7 change)

1. `96-point smoke test` — model loads 96-point adapter output, scores one full
   day (96 rows), no shape/NaN errors.
2. `24-point golden-baseline regression` — all 7 legs @24-point reproduce R2
   metrics within tolerance (non-functional metadata exempt).
3. Leakage checklist (LEAKAGE_AUDIT_96 §5) passes, **including the p56/p57 cutoff
   boundary test** for every RT leg (p56 visible & used, p57 masked, never leaked).
4. **Minimal-change principle (owner-approved gate):** every change is a
   *parametrization* of the existing 2.5 local code, grounded in real source
   (`file:line`). No model logic is rewritten or imagined from this plan. A diff
   that introduces new model architecture, a new RT fill strategy, or a unified
   cutoff rewrite **fails** this gate (see §3.0, §3.4, §3.7).

---

## 6. Remaining Risks / Owner Decisions

1. **Single-unit scope** — is unit-level `epf_unit_data_96` price the delivered
   "market" price, or must a province-level 96-point price still be sourced?
2. ~~**14:00 vs 15:00 cutoff** — resolved: fixed at 14:00 / `realtime_cutoff_period
   = 56` (LEAKAGE_AUDIT_96 §3). No longer open.~~
3. **96-point lags ×4** increase training time/memory ~4× for sequence models —
   budget GPU VRAM accordingly (respect project GPU caveats).
4. **RT916/TimeMixer RT** must inherit the `protocol_b_cutoff` mask — verify
   before R5 fusion.
