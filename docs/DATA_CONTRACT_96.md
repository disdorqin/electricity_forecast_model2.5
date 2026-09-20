# DATA_CONTRACT_96 — Formal 96-Point (15-Minute) Data Contract

> **Status:** active. Production source was consolidated on 2026-09-17 to the
> single remote table `epf_pmos_96_full`; `sync_data_96_core` and the old two-table
> mirror are historical compatibility paths only.
>
> Companion doc: `LEAKAGE_AUDIT_96.md`.
> Historical source material is retained under `docs/archive/`; current operation commands are maintained in `RUNBOOK.md`.

> **Current serving contract (Dynamic-v1, production-accepted 2026-09-20):** formal `--96` first
> synchronizes `epf_pmos_96_full`, then persists an immutable D/T snapshot and
> builds one routed `FeatureViewBuilder` view.  Visibility is **not** a fixed
> hour/period rule: all RT cells present in the snapshot are retained and each
> missing cell follows the approved same-day/history fallback chain.  Target-day
> truth is always masked.  The legacy fixed-cutoff helper is compatibility-only.

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

## 2. Production Source and Three-Layer Local Contract

The only production remote source is:

```text
epf_pmos_96_full
```

`main.py --pipeline sync_dataset --resolution 15min` performs a read-only sync
from this table. The old `epf_market_data_96` / `epf_unit_data_96` mirrors are
historical compatibility artifacts and are not production inputs.

The synchronized data has one persistent model-format store:

```text
data/96/authoritative/pmos_96_全量.csv
    faithful business-column mirror of the selected production unit;
    may contain the latest partial or forecast-only day.

data/96/model_input/shandong_pmos_96_model_input_full.parquet
    single persistent canonical model store; closed history plus partial/forecast-only tail.
```

Closed history is selected logically from the full store when needed. The old clean parquet remains a compatibility artifact only and is not rebuilt by the production sync path.

The complete remote DB row, including database audit metadata, is also retained
at `data/96/remote/parquet/epf_pmos_96_full.parquet`.

Current DB schema keeps `market_date`, `时段`, forecast/actual grid fields,
`日前出清价格`, `实时出清价格`, unit fields, reserve forecasts, `unit_id`, and
audit timestamps. Database metadata (`id`, `source_captured_at`, `create_time`,
`update_time`) is never a model feature.

If the table contains exactly one `unit_id`, sync selects it automatically. If
multiple units appear, production sync fails closed until `--sync-unit-id` or
`PMOS_96_UNIT_ID` selects the intended unit explicitly.

Historical 96 `核电预测` contains gaps. The validated model-store builder may
fill **forecast-only** missing cells from the 24-point canonical forecast; no
actual value and no price target is ever filled from 24-point data.

---

## 3. Targets (native 96-point price pair — owner-approved)

Production targets come from the selected `epf_pmos_96_full` unit:

```text
日前出清价格   -> model canonical `日前电价`
实时出清价格   -> model canonical `实时电价`
```

Both are 96-point unit-level clearing prices under the current project scope.
The authoritative/full stores may contain a latest partial day; target-day
truth is never exposed to a model run and is used only for later actual-ledger
settlement/evaluation when it becomes available.

---

## 4. Field Availability at Prediction Time

| Field group | Available at prediction time? | Notes |
|---|---|---|
| D target-day grid forecasts | ✅ Yes, if 96/96 present | Required target-day exogenous information. Missing target forecasts fail closed as `TARGET_FORECAST_NOT_READY`. |
| D target-day `日前电价` | ❌ No | Prediction target; outer runner mask sets all 96 slots to `NaN`. |
| D target-day `实时电价` | ❌ No | Prediction target; outer runner mask sets all 96 slots to `NaN`. |
| D target-day actual grid fields | ❌ No | Labels/realizations only; all 96 slots are masked. |
| D-1 `日前电价` | ✅ Complete 96-point history | Visible historical price input. |
| D-1 `实时电价` and actual grid fields | ✅ Exactly the cells present in the immutable snapshot | FeatureViewBuilder routes missing cells; no second physical-time trim is applied. |
| D-2 and earlier realized history | ✅ Historical | May be used according to each model's lag/training contract. Formal96 fusion learner currently uses a conservative unified lag=2 for both DA/RT, so only full-day truth through D-2 enters weight learning. DA may eventually support a fresher lag after the final information-boundary review, but that is not enabled now. |
| DB audit timestamps | ❌ Exclude | `source_captured_at/create_time/update_time` are audit metadata, not model features. |

---

## 5. Forecast-Time Filtering & Dynamic FeatureView

For every formal 96 target `D`, the runner resolves a Snapshot before any model starts. Except for `--finish`, the façade first performs DB sync and obtains `latest_closed_day`; wall-clock date is not the historical/live authority.

| Route | Condition | Snapshot semantics |
|---|---|---|
| `STORED_LIVE_SNAPSHOT_REPLAY` | `D` is historical/closed and a successful canonical LIVE snapshot is bound by run provenance | Reuse the exact historical LIVE values/manifest; never rebuild from today's DB and never choose the newest attempt by directory time. |
| `HISTORICAL_PROXY_V1` | `D` is historical/closed and no valid LIVE snapshot exists | D-1 DA remains 96/96; D-1 final actual and RT are exposed only through p56 (14:00 proxy), RealityTmp is absent, tail cells are masked and routed by the normal FeatureView fallback. This is operational proxy evidence only. |
| `LIVE_DYNAMIC` | current/unclosed formal target | Freeze exactly the cells present in the synchronized DB at forecast origin; no fixed-hour second trim. |

All three routes converge on the same `FeatureViewBuilder`. The snapshot is retained in an immutable attempt slot at `outputs/96/runs/<D>/snapshot/attempt_<id>/`; a successful LIVE canonical snapshot is a durable historical replay asset and is preserved by formal96 reruns/force. The shared routed view is attempt-owned scratch at `runtime/.../feature_view/input.parquet` and is deleted after NORMAL delivery. The original authoritative/model-store files are never modified.

The FeatureView contract is:

```text
actual: final → RealityTmp → same-field forecast → latest closed same-period → recent median
RT:     snapshot RT → same-day same-period DA → latest closed same-period RT → recent median
D target actual/DA/RT truth: mask all 96 slots
```

Routing is cell-local and uses every RT cell visible in the synchronized snapshot;
it never applies a second fixed-hour trim.  The view audit records route counts,
remaining NaN, snapshot_id and `target_truth_mask=true`.  Critical source loss or
DB sync failure is fail-closed.  A NORMAL invocation removes the transient view;
the immutable snapshot and prediction provenance remain for cache/finish recovery.

Any model that retrains during prediction must also keep supervised fit/validation off the routed decision-day effective rows: those rows may contain forecast/DA/history fallbacks and are serving context, not truth. Dynamic-v1 TimeMixer restricts supervised sample days to strictly before the decision day; RT916 ends its dynamic supervised window at decision-day midnight (the prior business day's p96); SGDFNet already uses train/validation rows strictly before the decision day.

Operational warm-start history is separate from serving input. `outputs/96/ledger/`
may be migrated so `ledger_weight` can start immediately; selector history still
uses only complete closed days.  A migrated row preserves its original
`data_cutoff`, `serving_protocol` and source provenance and is never relabeled as
current Dynamic-v1 evidence.

A target day is scoreable only when all 96 target forecast slots required by
the canonical model contract are present. Live prediction does **not** require
D-day actual prices; partial target actual may be appended for settlement/diagnostics without changing model-serving legality. Post-run prediction audit follows the same rule: partial/absent target actual is legal, but any present rows must be valid unique finite p1..p96 subsets. Backtest/settlement code opts in to the explicit `--require-target-actual` gate when exact target-day 96/96 truth is required.

A live Stage1 may therefore finish as `complete_with_warnings` solely because target-day actual is not yet closed. That status remains reusable by `--finish` only after the strict Dynamic provenance validator proves all canonical DA3/RT4 model files, 96 slots, snapshot/protocol identity, SGDFNet anchor and RT916 stride contracts.

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

### 7.1 Fixed decisions (updated 2026-09-19, project lead)

1. **Dynamic-v1 snapshot/FeatureView is the sole formal visibility boundary.**
   The historical `realtime_cutoff_hour`/p60 values remain only in legacy or
   experiment fixtures; formal production does not re-trim RT by a fixed hour.
2. **Single persistent model-store contract is fixed.**
   - authoritative = faithful synchronized DB business columns;
   - `model_input_full.parquet` = the only persistent 96-point model store, containing closed history plus partial/forecast-only tail;
   - closed history is selected logically from the full store and is not materialized as a second production parquet;
   - the FeatureView parquet is task-local scratch only and is deleted after the invocation completes; the immutable snapshot remains auditable.
3. **No realized-value imputation across the forecast boundary.** Historical
   missing **forecast-only** cells may use the validated 24-point forecast
   fallback; actual values and price targets are never filled from future truth.
4. **Model legs are fixed** (5 families, 7 task legs) — see LEAKAGE_AUDIT_96 §1.1
   and `docs/archive/historical-audits-2026-07/PLAN_96_POINT_MODEL_COMPATIBILITY_AFTER_LOCAL_SYNC.md` §3:
   - **Day-ahead:** TimesFM DA, LightGBM DA, TimeMixer DA.
- **Realtime:** TimesFM RT, RT916 RT, TimeMixer RT, SGDFNet RT. LightGBM RT is disabled from the production candidate pool.
   Formal SGDFNet rows identify the source as
   `sgdfnet_decision_day_da_anchor` and record `anchor_source_day`,
   `anchor_source_type`, `anchor_rows=96`, and `fallback_used`.
5. **Minimal-change principle.** All 96-point compatibility changes must be
   minimal-change *parametrizations* of the existing 2.5 local code, grounded in
   real source (file:line), not a rewrite or new model. Enforced as an acceptance
   gate (`docs/archive/historical-audits-2026-07/PLAN_96_POINT_MODEL_COMPATIBILITY_AFTER_LOCAL_SYNC.md` §5).

### 7.2 Still open (owner decision required)

1. **Single-unit scope:** current production DB contains one selected `unit_id`.
   If multiple units appear, sync fails closed and requires an explicit unit selection;
   any future move to a province-level 96-point target is a separate business decision.
2. **Historical forecast version provenance:** current `epf_pmos_96_full` is an
   upserted latest-state table. Older historical dates do not, by themselves, prove
   that the stored forecast values are exactly the version published at the historical
   D-1 15:00 origin. Future crawler `next_forecast` archives preserve capture-time
   snapshots; legacy history must be marked accordingly when strict historical
   publication-version evidence is required. 2026-09-18 local audit of
   2026-08-15..2026-09-15 found 0/32 days with an independent D-1 snapshot;
   all 32 are therefore `UNVERIFIED_LEGACY_VINTAGE`. This does not invalidate
   production mechanics testing, but it blocks any claim of strict historical
   forecast-publication equivalence for that range.

## 8. 24/96对照与数据质量摘要

本节收敛原 `docs/archive/historical/24_VS_96_FEATURE_COMPARISON.md` 和 `docs/archive/historical/DATA_QUALITY_96.md` 的长期有效结论：

- 24 点是小时级正式交付口径，96 点是 15 分钟级完整口径；两者不能混用 ledger、runs 或目标列；
- 96 点 `period_no=1..96`，`p96` 属于业务日 D 的最后一个点；
- 96 点 `actual_*` 必须和 `fcast_*` 做独立性检查，历史/目标日 actual 只能按防泄漏规则使用；
- 每个业务日必须具备完整 96 个 period，禁止用重复、插值或 24 点复制值伪造完整性；
- 24/96 的比较必须先统一 business day、边界点和聚合口径，再计算 MAD、相关系数或 MAPE；
- 详细历史测量和旧字段对照保留在 `docs/archive/historical-audits-2026-07/`，本节只维护当前契约。

当前快照验证（2026-08-16）：`data/96/authoritative/pmos_96_全量.csv` 与
`data/24/canonical/shandong_pmos_hourly.xlsx` 的共同 actual 字段已完成按小时
聚合对照；主字段直调负荷 MAD < 0.001、相关系数 1.000，交叉验证通过。该 96 点
文件仍只承担 actual authority，不替代 `data/96/model_input/`。

## 8. Historical replay and proxy routes (2026-09-20)

Formal96 route resolution is explicit and manifest-bound: a closed historical day with a validated canonical LIVE snapshot uses `STORED_LIVE_SNAPSHOT_REPLAY`; a closed day without one uses `HISTORICAL_PROXY_V1` (`formal96_historical_proxy_v1`, p56/14:00 proxy metadata); the current target uses `LIVE_DYNAMIC`. All three routes converge on the same FeatureViewBuilder and preserve `snapshot_id`, route, and cache provenance. A successful LIVE snapshot is retained under its attempt-bound canonical path and is never selected by directory recency.

The immutable server archive can be audited read-only with `--history-scope full-source`; source rows are staged first and current production rows are applied second, so current production wins overlapping keys. `--apply` remains an explicit, separate promotion step.
