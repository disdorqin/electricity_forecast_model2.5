# DATA_CONTRACT_96 — Formal 96-Point (15-Minute) Data Contract

> **Status:** Design document (analysis artifact). Generated from the locally synchronized
> mirror produced by `sync_data_96_core` on 2026-07-28 (full sync, source = remote DB).
> No model code was modified to produce this document.
>
> Companion docs: `DATA_QUALITY_96.md`, `LEAKAGE_AUDIT_96.md`,
> `PLAN_96_POINT_MODEL_COMPATIBILITY_AFTER_LOCAL_SYNC.md`, `SYNC_DATASET_96.md`.

---

## 1. Grain & Time Semantics

| Rule | Definition |
|---|---|
| One row | One 15-minute interval (one "period") |
| 96 rows | One business day |
| `period_no` | Integer `1..96`, increasing within a business day |
| `data_time` | **Interval-end** timestamp (period 1 ends 00:15, period 96 ends next-day 00:00) |
| `market_date` | Trading/business day the interval *belongs to* (period 96 belongs to day D even though its `data_time` is D+1 00:00) |
| `hour_business` | `ceil(period_no / 4)` → `1..24` |
| `quarter_in_hour` | `((period_no - 1) % 4) + 1` → `1..4` |
| `p96` | `D+1 00:00` but semantically belongs to business day D |

This contract is the native 96-point equivalent of the existing hourly
`hour_business 1..24` contract used by `utils/business_day.py`. The 96-point
layer must **not** silently aggregate prices to hourly.

---

## 2. Source Tables (required_core)

Local mirror paths (from `outputs/data_sync_96/sync_manifest.json`):

```
data/remote_96/parquet/epf_market_data_96.parquet
data/remote_96/parquet/epf_unit_data_96.parquet
```

### 2.1 `epf_market_data_96` — market-level grid features (NO price)

Shape: **160,416 rows × 32 columns**. Date range **2022-01-01 → 2026-07-29**,
**1,671** complete 96-days, 0 duplicate `(market_date, period_no)` keys.

| Column | Chinese | pandas dtype (local) | Role | Null % (synced) |
|---|---|---|---|---|
| `id` | — | int64 | PK (drop) | 0 |
| `market_date` | 交易日 | object→date | Join key | 0 |
| `period_no` | 时段号(1-96) | int64 | Join key | 0 |
| `data_time` | 时刻(区间结束) | datetime64 | Time index | 0 |
| `fcast_local_plant` | 地方电厂出力预测 | object (varchar) | Feature (forecast) | 0.00 |
| `fcast_tie_line` | 外电预测 | object | Feature | 0.00 |
| `fcast_wind` | 风电预测 | object | Feature | 0.00 |
| `fcast_solar` | 光伏预测 | object | Feature | 0.00 |
| `fcast_nuclear` | 核电预测 | object | Feature | 0.00 |
| `fcast_self_owned` | 自备电厂预测 | object | Feature | 0.00 |
| `fcast_test_unit` | 试验机组预测 | object | Feature | 0.00 |
| `fcast_direct_load` | 直调负荷预测 | object | Feature | 0.00 |
| `fcast_unit_maintenance` | 机组检修预测 | object | Feature | 0.00 |
| `fcast_pos_reserve` | 正备用预测 | object | Feature | 0.00 |
| `fcast_neg_reserve` | 负备用预测 | object | Feature | 0.00 |
| `fcast_bidding_space` | 竞价空间预测 | object | Feature | 0.00 |
| `fcast_new_energy` | 新能源预测 | object | Feature | 0.00 |
| `actual_direct_load` | 直调负荷实际 | object | Feature (actual) | 0.12 |
| `actual_local_plant` | 地方电厂出力实际 | object | Feature | 0.12 |
| `actual_tie_line` | 外电实际 | object | Feature | 0.12 |
| `actual_wind` | 风电实际 | object | Feature | 0.12 |
| `actual_solar` | 光伏实际 | object | Feature | 0.12 |
| `actual_nuclear` | 核电实际 | object | Feature | 0.12 |
| `actual_self_owned` | 自备电厂实际 | object | Feature | 0.12 |
| `actual_test_unit` | 试验机组实际 | object | Feature | 0.12 |
| `actual_unit_maintenance` | 机组检修实际 | object | Feature | 0.12 |
| `actual_pos_reserve` | 正备用实际 | object | Feature | 0.12 |
| `actual_neg_reserve` | 负备用实际 | object | Feature | 0.12 |
| `actual_bidding_space` | 竞价空间实际 | object | Feature | 0.12 |
| `actual_new_energy` | 新能源实际 | object | Feature | 0.12 |
| `create_time` | 入库时间(审计) | datetime64 | **Exclude (audit)** | 0 |
| `update_time` | 更新时间(审计) | datetime64 | **Exclude (audit)** | 0 |

> **Coercion note:** every `fcast_*` / `actual_*` column arrives as MySQL `varchar`
> and is read locally as pandas `object`. The dataset adapter MUST coerce to
> `float64` (`pd.to_numeric(errors="coerce")`) before feature use.

### 2.2 `epf_unit_data_96` — unit-level clearing prices (THE TARGETS)

Shape: **159,360 rows × 15 columns**. Date range **2022-01-01 → 2026-07-18**,
**1,660** complete 96-days, 0 duplicate `(market_date, period_no, unit_id)` keys.

| Column | Chinese | dtype | Role | Null % |
|---|---|---|---|---|
| `id` | — | int64 | PK (drop) | 0 |
| `market_date` | 交易日 | object→date | Join key | 0 |
| `period_no` | 时段号(1-96) | int64 | Join key | 0 |
| `data_time` | 时刻(区间结束) | datetime64 | Time index | 0 |
| `unit_id` | 机组标识 | object | Join key / entity | 0 |
| `da_cq_price` | 日前出清价格(元/MWh) | object | **TARGET (DA)** | 0.00 |
| `da_power` | 日前出力 | object | Feature (aux) | 0.00 |
| `da_energy` | 日前电量 | object | Feature (aux) | 0.00 |
| `da_status` | 日前开机状态 | object | Feature (categorical) | 0.00 |
| `rt_cq_price` | 实时出清价格(元/MWh) | object | **TARGET (RT)** | 0.00 |
| `rt_power` | 实时出力 | object | Feature (aux) | 0.00 |
| `rt_energy` | 实时电量 | object | Feature (aux) | 0.00 |

> **Unit field behavior (live DB):** exactly **one** unit is present —
> `7B2B5622A6FA5E9BE0531001C10A211E`. The join key on this table is therefore
> `(market_date, period_no)` in practice, but the schema supports multiple units;
> the adapter must join on `(market_date, period_no, unit_id)` and **never** filter
> to a single configured crawler unit (per task §10: "All units are downloaded
> without filtering"). If the live DB later contains more units, they must be kept.

---

## 3. Targets (native 96-point price pair — owner-approved)

```text
epf_unit_data_96.da_cq_price   # 日前 (day-ahead) clearing price, 元/MWh
epf_unit_data_96.rt_cq_price   # 实时 (realtime) clearing price, 元/MWh
```

Both are **unit-level** under the current 96-point project scope. They are the
only price columns in the entire 96-point ecosystem (the market table carries
no price). Synced statistics:

| Target | Min | Max | Null % | First | Last |
|---|---|---|---|---|---|
| `da_cq_price` | -100.0 | 1500.0 | 0.00 | 2022-01-01 | 2026-07-18 |
| `rt_cq_price` | -100.0 | 1500.0 | 0.00 | 2022-01-01 | 2026-07-18 |

---

## 4. Field Availability at Prediction Time

| Field group | Available at prediction time? | Notes |
|---|---|---|
| `fcast_*` (market) | ✅ Yes | Market forecast made *before* the interval; safe for both DA and RT. |
| `da_cq_price` (target D+1) | ❌ No (it is the target) | Past days' DA price is history; D+1 DA is the label. |
| `rt_cq_price` (target D+1) | ❌ No (label only) | The **target day's** `rt_cq_price` is entirely unknown at prediction time and is the **prediction target only** — it must **never** be filled with true post-cutoff values and must **never** be used as an input feature. Only the **decision day's** realized RT up to `realtime_cutoff_period` (=56, i.e. 14:00) is a visible *historical* input frame, reused via the existing DA-substitution / cutoff-mask logic (LEAKAGE_AUDIT_96 §2 Trap B). |
| `actual_*` (market) | ⚠️ History-only | Realized *after* the interval. Valid only as **lag** features for PAST periods. Using `actual_*` of a target period = leakage. |
| `da_power/da_energy/da_status` | ✅ Yes (DA stage) | Known at day-ahead clearing. |
| `rt_power/rt_energy` | ⚠️ Partial | Same cutoff rule as `rt_cq_price`. |
| `create_time/update_time` | ❌ Exclude | Audit columns; leak crawler timing. |

---

## 5. Forecast-Time Filtering & Target-Day Placeholders

- For a D+1 forecast made at D-day cutoff, the **target-day** rows
  (`market_date == D+1`) must exist as placeholder rows (all features filled,
  targets `NaN`) so the model can score them. The adapter emits one placeholder
  row per `(period_no)` (and per `unit_id` if multi-unit) for the target day.
- Historical `actual_*` of target day are **not** filled (they do not exist yet).
- **`actual_*` imputation scope (clarification):** the "missing-value fill" for
  `actual_*` referenced in this contract and in `DATA_QUALITY_96` applies **only**
  to the market *actual value* fields in `epf_market_data_96` (e.g. `actual_wind`,
  `actual_solar`, `actual_direct_load`, …), which carry a tiny ~0.12% null gap
  (`DATA_QUALITY_96` §2). It does **NOT** mean filling `rt_cq_price` post-cutoff
  truth — `rt_cq_price` is a label and its post-cutoff values are never imputed.
- Complete-day validation: a scored day must contain exactly 96 non-placeholder
  target rows to count as a `complete_96_day`.

---

## 6. Splitting Rules (recommended, to be implemented in R1)

- **Chronological only.** No random shuffle (time-series leakage).
- Default: train ≤ a fixed `train_end`, validate = next N days, test = latest held-out window.
- Every split boundary must align to **business-day** boundaries (midnight 00:00),
  never inside a 96-period day.
- Keep a frozen `train/val/test` day list in the processed-data manifest so
  R4–R7 comparisons are reproducible against the 24-point golden baseline.

---

## 7. Owner-Set Decisions & Open Items

### 7.1 Fixed decisions (2026-07-28, project lead)

1. **Realtime cutoff is fixed at 14:00.**
   - `realtime_cutoff_hour = 14`, `realtime_cutoff_period = 56`.
   - Periods **p1..p56 are visible**; **p57..p96 are masked / substituted** per the
     existing project logic (DA substitution / cutoff mask, see LEAKAGE_AUDIT_96 §2
     Trap B).
   - The legacy `15:00` figure (`SGDFNet/src/sgdfnet/protocol_b_cutoff.py` default
     `decision_hour=15`, and some older docs) is recorded **only as a historical /
     legacy ambiguity** — it is **no longer a candidate option**.
2. **`actual_*` imputation scope.** The `actual_*` missing-value fill applies
   **only** to market actual-value fields in `epf_market_data_96` (e.g.
   `actual_wind`, `actual_solar`, `actual_direct_load`). It does **not** apply to
   `rt_cq_price`, which is a prediction target whose post-cutoff truth is never
   filled.
3. **Model legs are fixed** (5 families, 7 task legs) — see LEAKAGE_AUDIT_96 §1.1
   and `PLAN_96_POINT_MODEL_COMPATIBILITY_AFTER_LOCAL_SYNC.md` §3:
   - **Day-ahead:** TimesFM DA, LightGBM DA, TimeMixer DA.
   - **Realtime:** RT916 RT, TimeMixer RT, SGDFNet RT, TimesFM RT.
4. **Minimal-change principle.** All 96-point compatibility changes must be
   minimal-change *parametrizations* of the existing 2.5 local code, grounded in
   real source (file:line), not a rewrite or new model. Enforced as an acceptance
   gate (`PLAN_96_POINT_MODEL_COMPATIBILITY_AFTER_LOCAL_SYNC.md` §5).

### 7.2 Still open (owner decision required)

1. **Single-unit scope:** confirm `epf_unit_data_96` unit-level price is to be
   treated as the *market* price for delivery, or whether a province-level 96-point
   price must still be sourced (the market table has none).
