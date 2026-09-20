# LEAKAGE_AUDIT_96 — 96-Point Leakage Audit

> **Status:** active. Grounded in the 2026-07-28 local sync and current pipeline checks.
> Companion: `DATA_CONTRACT_96.md`.

The formal 96 leakage boundary is the immutable D/T snapshot plus one
`FeatureViewBuilder`.  Day-ahead and realtime visibility are routed from that
snapshot; there is no second fixed-hour/p60 trim in serving.  `actual_*` fields
remain strictly historical and target-day truth is always masked.  Legacy
`protocol_b_cutoff.py`/fixed-cutoff paths remain compatibility-only.

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
- **Realtime (4 production legs):** TimesFM RT · RT916 RT · TimeMixer RT · SGDFNet RT.

All formal 96-point visibility handling across these legs is supplied by the
shared Dynamic-v1 FeatureView; each leg keeps only mathematical feature logic
and required assertions.  Fixed-hour parameters remain internal/legacy only.

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

### Trap C — DA anchor semantics for RT
Target-day DA may be available as a normal RT feature, but the formal **SGDFNet**
contract is stricter: `da_anchor` for target D must be the complete D-1 DA curve
(`D-1 p1..p96 → D p1..p96`). Target-day DA/RT/actual values must never become the
SGDFNet anchor; a historical median is permitted only for an exceptional missing
source slot and must be audited as `fallback_used=true`. Other RT legs do not use
an anchor-plus-delta formulation. Target-day `rt_cq_price` and target-day
`actual_*` remain forbidden features for every leg.

The production provenance label is `sgdfnet_decision_day_da_anchor`; the old
`sgdfnet_config_da_fill` label is retained only in historical fixtures.

---

## 3. Dynamic-v1 serving boundary

Before any formal model starts, `--96` performs DB sync, writes an immutable
attempt-scoped snapshot under `outputs/96/runs/<target>/snapshot/attempt_<id>/`
(protocol `formal96_dynamic_snapshot_v1`) and builds an attempt-owned FeatureView.  The
FeatureView routes actuals as `final → RealityTmp → same-field forecast → latest
closed same-period → recent median`; RT routes as `snapshot RT → same-day DA →
latest closed RT → recent median`.  It uses every RT cell present in the
snapshot, records route counts, and masks all target-day actual/DA/RT truth.

The snapshot manifest and FeatureView audit are the leakage evidence.  Missing
critical sources or failed sync are fail-closed with source/field/row details;
there is no stale-data or degraded-model fallback.  Models must not recreate a
serving cutoff from `realtime_cutoff_hour`, `decision_hour`, or `asof_ts`.

**Training/serving separation is a second independent leakage wall.**  The routed
D-day `actual`/RT cells may contain forecast/DA/history fallbacks and therefore are
valid inference context but are **not supervised truth**.  Any model that retrains
inside `predict()` must cap fit/validation before decision day D for RT/full-actual
labels.  In the current Dynamic-v1 implementation TimeMixer training days are
restricted to `< D`, RT916's dynamic training window ends at D 00:00 (the p96
timestamp of business day D-1), and SGDFNet already trains/validates strictly
before D.  A controlled serving smoke that replaces model execution with test
doubles is not sufficient evidence for this training-boundary rule.

**Real-model acceptance evidence (2026-09-20):** the production Dynamic FeatureView was exercised by all seven formal model legs with real model computation. TimeMixer DA/RT and RT916 completed their real training/inference paths; TimesFM DA/RT and LightGBM DA produced exact 96-slot outputs; SGDFNet produced exact 96 slots with decision-day DA anchor `rows=96` and `fallback_used=false`. The final standard `main.py --96 2026-09-20` run completed NORMAL with postflight PASS and no emergency fallback. This proves the implemented serving/training boundary is executable end-to-end; it does **not** turn legacy latest-state historical forecasts into strict historical publication vintages.

---

## 4. Anti-Leakage Guards (to be enforced in R1 adapter)

1. **Drop same-period `actual_*` at feature build time** — only lag/rolling
   windows over strictly-past periods are allowed.
2. **FeatureView route/mask audit** proves target-day RT/actual truth is absent
   and records every fallback source; fixed-hour cutoff substitution is not used
   by formal Dynamic-v1.
3. **Exclude** `id`, `create_time`, `update_time` from any feature matrix.
4. **Chronological split only** (DATA_CONTRACT_96 §6) — no shuffling.
5. **Target-day placeholder rows** carry `NaN` targets (DATA_CONTRACT_96 §5) so a
   model cannot read its own label.
6. **Routed decision-day values cannot become labels** — TimeMixer/RT916/any future
   retraining model must prove its fit boundary excludes D synthetic effective RT/actual.
7. **Re-run the dedicated leakage test** (`96-point smoke test` + `24-point
   golden-baseline regression`) after every leg is adapted (task §20).

---

## 5. Leakage Test Checklist (R4–R7 acceptance)

- [ ] No `actual_*` column appears in the feature matrix for the target period.
- [ ] `rt_*` labels for post-cutoff periods are absent from the RT scoring frame.
- [ ] `da_cq_price` (target day) is **not** in the DA feature matrix (it is the label).
- [ ] `id` / `create_time` / `update_time` absent from features.
- [ ] Walk-forward: train window strictly precedes test window.
- [ ] Legacy/shadow negative-price classifier (when explicitly run) uses only
      pre-cutoff / historical info; formal `--96` must record
      `classifier_policy=disabled_by_production_policy` and consume uncorrected RT fuse.

---

## 6. Verdict

The 96-point data introduces **no new leakage class** beyond the hourly one — it
adds `actual_*` columns that must obey Trap A.  Dynamic-v1 makes the snapshot
and FeatureView the single auditable information wall; the legacy
`protocol_b_cutoff` logic is retained only for compatibility.

## 7. 运行时信息可得性门控

本节收敛原 `docs/archive/agent-research-2026-08/信息可得性自动化门控设计_20260816.md`。长期设计只在本文件维护，专项旧稿归档。

### 7.1 统一上下文

生产模型最终应共享只读 Dynamic-v1 FeatureView metadata：

```text
target_day, decision_day, resolution, snapshot_id,
serving_protocol, target_truth_mask, route_audit
```

正式 96 下游模型不得自行产生固定小时可见性口径；24 点
legacy/strict-spread 可保留其独立 14:00 contract。

### 7.2 特征注册与审计

新增特征必须记录：`name`、`source_table`、`role`（forecast/actual/target/lag）、`available_at`、`allowed_tasks` 和 `required_shift_slots`。模型推理前检查：

1. target-day `actual_*` 是否直接进入特征；
2. cutoff 之后的 RT 真值是否仍存在；
3. 数据可得时间是否不晚于 `cutoff_ts`；
4. lag 是否按 hourly/15min 正确 shift；
5. business day、slot 和 resolution 是否一致。

### 7.3 阻断等级

| 等级 | 条件 | 行为 |
|---|---|---|
| P0 | target-day actual/RT truth 直接进入特征 | 阻断模型 |
| P1 | cutoff 后数据未遮蔽或来源不明 | 阻断 RT 模型 |
| P2 | 特征缺失、降级填充或非生产来源 | 仅允许经过批准的特征级异常 fallback，并写入 manifest；strict history/provenance/模型契约失败仍必须 formal96 fail-closed，不得 emergency/degraded fallback |

完整接入顺序为：共享上下文 → 特征注册表 → TimesFM/SGDFNet/TimeMixer/RT916 逐模型接入 → CI 边界测试。

## 8. Historical replay and proxy routes (2026-09-20)

Formal96 now uses a three-route information boundary, but only one downstream serving policy:

1. **Stored LIVE replay.** If a closed historical target has a successful canonical `formal96_dynamic_snapshot_v1` bound by run/Stage1 provenance, replay that exact snapshot. Do not rebuild from today's DB and do not select an attempt by filesystem recency.
2. **Historical Proxy v1.** If no valid LIVE snapshot exists, build `formal96_historical_proxy_v1`: D-1 DA remains 96/96; D-1 final actual and final RT are visible only through p56; historical RealityTmp/provisional RT are not fabricated; p57..p96 are masked and filled only by the existing FeatureView routes. T-day truth remains fully masked. Metadata must stay `UNVERIFIED_LEGACY_VINTAGE / OPERATIONAL_PROXY` and is not strict publication-vintage evidence.
3. **LIVE Dynamic.** For an unclosed target, use the synchronized DB exactly as visible at forecast origin. No model may impose an additional fixed 14/15-hour serving cutoff.

Route selection is based on synchronized `latest_closed_day` plus validated stored provenance, not UTC/local wall-clock date. `--finish` is excluded from re-resolution and reuses its original Stage1 snapshot. The shared FeatureViewBuilder remains the sole serving visibility authority for all three routes; TimeMixer, SGDFNet and RT916 stay on `dynamic_serving=true`, while their fixed-hour parameters remain training/legacy compatibility only.

**Historical Proxy real-model acceptance:** `python main.py --96 2026-08-17` completed NORMAL with the proxy p56 boundary materially present in `values.parquet` (actual/RT p1..p56 only, p57 masked), zero target-truth visibility, seven 96-slot model legs, SGDFNet D=2026-08-16 DA96 anchor/fallback=false, RT916 stride24, 30-day learner capped at T-2=2026-08-15, weight/fuse/final/postflight PASS. This validates the operational proxy mechanism; it still does not prove historical temporary-value or forecast publication vintage.
