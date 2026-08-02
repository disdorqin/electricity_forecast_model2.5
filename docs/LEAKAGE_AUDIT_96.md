# LEAKAGE_AUDIT_96 — 96-Point Leakage Audit

> **Status:** Design/analysis document. Grounded in the 2026-07-28 local sync. No
> model code modified. Companion: `DATA_CONTRACT_96.md`, `DATA_QUALITY_96.md`.

The leakage surface for 96-point is the **same shape** as the hourly pipeline
(`protocol_b_cutoff.py`): day-ahead is fully known at prediction time; realtime
is only known up to the D-day cutoff. The only change is the resolution of the
cutoff boundary and the addition of `actual_*` columns that must be treated as
strictly historical.

---

## 1. Field Classification

| Field | Class | Why |
|---|---|---|
| `fcast_*` (market, 13 cols) | **Safe (future-known)** | Market forecast issued before the interval; available when predicting DA and RT. |
| `da_cq_price` | **Target (DA)** | The D+1 day-ahead price is the label; past DA prices are history. |
| `da_power/da_energy/da_status` | **Safe (DA stage)** | Known at day-ahead clearing. |
| `rt_cq_price`, `rt_power`, `rt_energy` | **Target/aux (RT), cutoff-gated** | Periods after D-day cutoff are unknown at prediction time. |
| `actual_*` (market, 13 cols) | **Historical-only (leakage-unsafe for target period)** | Realized *after* the interval. Usable only as **lag** features over PAST periods. |
| `create_time`, `update_time` | **Exclude** | Audit provenance; leaks crawl timing; not available at prediction time. |
| `id` | **Exclude** | Surrogate PK. |

### 1.1 Fixed Model-Leg Set (owner-approved, 2026-07-28)

The 96-point compatibility covers **5 model families / 7 task legs**. This list is
**fixed** for the minimal-change round — do not add, drop, or merge legs.

- **Day-ahead (3 legs):** TimesFM DA · LightGBM DA · TimeMixer DA
- **Realtime (4 legs):** RT916 RT · TimeMixer RT · SGDFNet RT · TimesFM RT

All cutoff handling across these legs shares the single business parameter
`realtime_cutoff_hour = 14` / `realtime_cutoff_period = 56` (§3); each leg keeps
its **own** existing cutoff implementation (no unified rewrite).

---

## 2. The Three Leakage Traps

### Trap A — `actual_*` of a target period
Using any `actual_*` column (e.g. `actual_wind`) for the **same** `(market_date,
period_no)` whose `rt_cq_price` is being predicted injects the realized future.
These columns must only ever enter as **lag features** (same period on prior
days, e.g. `actual_wind_d1`, `actual_wind_d7`). The adapter's
`forecast_time_filter` must drop same-period `actual_*`.

### Trap B — future `rt_cq_price` past cutoff
For a D+1 RT forecast made at D-day cutoff, the RT price of every post-cutoff
period on D+1 is unknown. The hourly `protocol_b_cutoff._build_protocol_b_visible_frame`
masks same-day post-cutoff RT truth by substituting the DA price (anti-leak).
The 96-point adapter must replicate this: build a `visible_rt` frame where
periods > cutoff on the prediction day are masked, and RT training labels for
those periods are only available in the *historical* portion (not at scoring
time).

### Trap C — `da_cq_price` as a same-day feature for RT
Using the *target-day* `da_cq_price` as an RT feature is legitimate **because DA
is known before RT**. For the **SGDFNet** leg specifically, this is the
`da_anchor` design: `rt_hat = da_anchor + delta_hat` (see
`PLAN_96_POINT_MODEL_COMPATIBILITY_AFTER_LOCAL_SYNC.md` §3.7). Other RT legs
(TimesFM RT, TimeMixer RT) do **not** use a DA-anchor+delta formulation — they may
still consume `da_cq_price` as an ordinary known feature, but their modelling is
**not** "anchor + delta". What is leakage in **all** legs is using target-day
`rt_cq_price` or target-day `actual_*` as features.

---

## 3. Cutoff Boundary in 96-Point Resolution (FIXED: 14:00)

**Owner decision (2026-07-28, final):** the realtime cutoff is **fixed at 14:00**.
This is the canonical value in `cli/parser.py` (`--realtime-cutoff-hour` default =
14). It is **no longer an open question**.

- `realtime_cutoff_hour = 14`
- `realtime_cutoff_period = 56`
- Periods **p1..p56 are visible** (known RT history up to 14:00).
- Periods **p57..p96 are masked / substituted** per the existing project logic
  (DA substitution / cutoff mask, see Trap B).

`data_time` is interval-end, so:

```
period p ends at  p * 15 minutes after 00:00
14:00  = 840 min  ->  period 56  (ends exactly 14:00)
```

| Cutoff (FIXED) | Known RT periods (≤ cutoff) | Masked RT periods (> cutoff) |
|---|---|---|
| **14:00** (`realtime_cutoff_period = 56`) | `period_no` 1..56 | 57..96 |

The single configurable constant `REALTIME_CUTOFF_PERIOD = 56` is defined **once**
and reused by all 7 legs. No leg may hardcode its own cutoff hour.

> **Historical / legacy note (NOT a candidate):** the hourly
> `SGDFNet/src/sgdfnet/protocol_b_cutoff.py` default `decision_hour=15`
> (→ period 60) and some older docs state 15:00. That figure is recorded here
> *only* as a legacy ambiguity from the hourly layer. It is **no longer a
> selectable option** — the 96-point contract is 14:00 / period 56.

---

## 4. Anti-Leakage Guards (to be enforced in R1 adapter)

1. **Drop same-period `actual_*` at feature build time** — only lag/rolling
   windows over strictly-past periods are allowed.
2. **Cutoff mask** on `rt_cq_price` / `rt_power` / `rt_energy` for the prediction
   day; replicate `protocol_b_cutoff` masked-DA substitution.
3. **Exclude** `id`, `create_time`, `update_time` from any feature matrix.
4. **Chronological split only** (DATA_CONTRACT_96 §6) — no shuffling.
5. **Target-day placeholder rows** carry `NaN` targets (DATA_CONTRACT_96 §5) so a
   model cannot read its own label.
6. **Re-run the dedicated leakage test** (`96-point smoke test` + `24-point
   golden-baseline regression`) after every leg is adapted (task §20).

---

## 5. Leakage Test Checklist (R4–R7 acceptance)

- [ ] No `actual_*` column appears in the feature matrix for the target period.
- [ ] `rt_*` labels for post-cutoff periods are absent from the RT scoring frame.
- [ ] `da_cq_price` (target day) is **not** in the DA feature matrix (it is the label).
- [ ] `id` / `create_time` / `update_time` absent from features.
- [ ] Walk-forward: train window strictly precedes test window.
- [ ] Negative-price classifier uses only pre-cutoff / historical info.

---

## 6. Verdict

The 96-point data introduces **no new leakage class** beyond the hourly one — it
only (a) adds `actual_*` columns that must obey Trap A, and (b) shifts the cutoff
to a 15-minute-period boundary. The existing `protocol_b_cutoff` logic is
resolution-agnostic and can be reused once `REALTIME_CUTOFF_PERIOD` is defined.
