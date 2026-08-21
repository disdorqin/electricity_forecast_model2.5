# DATA_QUALITY_96 — 96-Point Data Quality Report

> **Status:** Design/analysis document. Numbers below are measured directly from the
> local mirror synchronized on 2026-07-28 (full, source = remote DB). No model code
> was modified. Companion: `DATA_CONTRACT_96.md`, `LEAKAGE_AUDIT_96.md`.

---

## 1. Completeness (per table)

| Metric | `epf_market_data_96` | `epf_unit_data_96` |
|---|---|---|
| Rows (local = remote) | 160,416 | 159,360 |
| Date range | 2022-01-01 → 2026-07-29 | 2022-01-01 → 2026-07-18 |
| Distinct days | 1,671 | 1,660 |
| **Complete 96-days** | **1,671** | **1,660** |
| Incomplete days | 0 | 0 |
| Duplicate keys | 0 | 0 |
| `period_no` range | 1..96 | 1..96 |
| Units (distinct) | — | **1** (`7B2B…211E`) |

Both tables are **perfectly daily-complete**: every present day has exactly 96
distinct periods, zero duplicates. `epf_market_data_96` extends 11 days beyond
`epf_unit_data_96` (unit prices stop at 2026-07-18 while market features continue
to 2026-07-29) — expected, and the adapter must left-join on the unit table so
those trailing market-only days carry no target.

---

## 2. Null Patterns

| Group | Column | Null % (measured) |
|---|---|---|
| Market forecast `fcast_*` | all 13 columns | **0.00 %** |
| Market actual `actual_*` | all 13 columns | **~0.12 %** (e.g. `actual_direct_load` 0.12%) |
| Unit target `da_cq_price` | — | **0.00 %** |
| Unit target `rt_cq_price` | — | **0.00 %** |
| Unit aux `da_*/rt_*` | power/energy/status | **0.00 %** |

- **Forecasts are pristine** (0 nulls from 2022-01-01).
- **Actuals have a tiny ~0.12% gap** (≈190 of 160,416 rows per column). These
  require imputation in the adapter (forward-fill within day, then inter-day
  median fallback). They are *historical-only* fields anyway (see LEAKAGE_AUDIT).
  **Scope note (2026-07-28):** this imputation applies **only** to the market
  `actual_*` fields in `epf_market_data_96`. It does **not** fill `rt_cq_price`,
  which is a prediction *target* whose post-cutoff truth is never imputed or used
  as input (see `DATA_CONTRACT_96.md` §4–§5).
- **Both targets are fully populated** — no target-day imputation needed.

---

## 3. Numeric Coercion Requirement

All `fcast_*` / `actual_*` / price columns are MySQL `varchar` and import as
pandas `object`. The adapter must coerce to `float64`. Any non-numeric token
(after coercion) becomes `NaN` and is handled by the imputation rules above.

---

## 4. Value Ranges & Sanity

| Field | Min | Max | Notes |
|---|---|---|---|
| `da_cq_price` | -100.0 | 1500.0 | 元/MWh; negative prices real |
| `rt_cq_price` | -100.0 | 1500.0 | 元/MWh; negative prices real |

Both targets share the same hard bounds `[-100, 1500]`. No out-of-range or
obvious unit-error spikes observed at the extremes (the ±100 / 1500 caps are
consistent with Shandong market price limits).

---

## 5. Negative-Price Prevalence (feeds native classifier plan)

| Condition | `da_cq_price` | `rt_cq_price` |
|---|---|---|
| `<= 0` | 19,590 periods (12.3%) | 23,385 periods (14.7%) |
| `<= -50` | — | **22,288 periods (14.0%)** |
| `== 0` | — | 522 periods |

The native 96-point negative-price classifier (task §19) uses base label
`realtime_price <= -50`, which qualifies **22,288** of 159,360 RT periods
(~14%). This is substantial enough to warrant a dedicated native classifier
rather than relying on the regression alone.

---

## 6. Audit / Provenance Columns

`create_time` on `epf_market_data_96` ranges **2026-04-14 → 2026-07-28** — i.e.
the remote rows were bulk back-filled in spring/summer 2026. `update_time`
similar. These columns are **crawler provenance only** and must be excluded from
features (they would leak ingestion timing and are not available at prediction
time).

---

## 7. Late-Start Features

In the synchronized data, **every** `fcast_*` and `actual_*` column has its first
non-null date on **2022-01-01** — no late-starting feature was observed. The
adapter still keeps a `late_start` guard (skip a feature until its first valid
date) so a future backfill that adds a column mid-history will not silently
train on `NaN`-padded history.

---

## 8. Quality Verdict

| Dimension | Verdict |
|---|---|
| Daily completeness | ✅ Perfect (96/day, 0 incomplete) |
| Key uniqueness | ✅ 0 duplicates |
| Target nulls | ✅ 0% |
| Forecast nulls | ✅ 0% |
| Actual nulls | ⚠️ 0.12% (impute) |
| Numeric coercion | ⚠️ Required (varchar→float) |
| Multi-unit | ⚠️ Only 1 unit live (join must stay unit-aware) |
| Negative prices | ⚠️ Real, ~14% RT ≤ -50 (classifier needed) |

The dataset is **production-ready for a first 96-point adapter** after the two
mechanical fixes: (a) `object→float64` coercion, (b) `~0.12%` actual-null
imputation.
