"""Iteration 9: strict regime-conditioned training-sample selection for P6 direct-spread.

Literature rationale:
- Electricity-price calibration-window studies show that "more/most-recent data" is not always best.
- Similar-day / recurring-drift studies suggest selecting historical regimes that resemble the target state.

Information contract (hard):
- forecast origin for target day D = D-1 14:00;
- target-day RT/spread/actual are labels only; target-day DA is NOT a feature;
- target-day forecast fundamentals are allowed;
- D-1 realized spread contributes only p1..p14 as visible context;
- complete supervised labels used for fitting are from D-2 and earlier only;
- candidate historical regime days are <= D-2;
- fresh 2026-08-15..2026-08-21 holdout is sealed.

Two branches:
A) Forecast-profile regime selection: choose top-K historical days by target forecast-profile distance.
B) Information-equivalent state selection: distance also contains the visible D-1 p1..p14 context
   that would have been available at each historical day's own origin.

We compare pure top-K selection, recent+similar unions, and soft similarity weighting.
Production code is untouched; outputs are experiment-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    CORE_SD_TOKENS,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    dedupe,
    direction_metrics,
    p6_features,
)
from scripts.experiments.spread_direction_24.goal70_engineering.run_iteration6_visible_d1_context import (
    add_visible_context,
)


def make_model(seed: int, balanced: bool = True) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced" if balanced else None,
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def complete_days(slot: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    day_map: dict[str, pd.DataFrame] = {}
    for d, g in slot.groupby("target_day", sort=True):
        gg = g.sort_values("hour_business")
        if len(gg) == 24 and gg.target_spread.notna().all():
            day_map[str(d)] = gg
    return sorted(day_map), day_map


def profile_columns(groups: dict[str, list[str]]) -> list[str]:
    cols = [c for c in groups["F2"] if any(t in c for t in CORE_SD_TOKENS)]
    cols += [
        c for c in groups["F3"]
        if c in {"residual_load_renew", "renewable_share", "bidding_space_ratio", "interconnect_share"}
    ]
    return dedupe(cols)


def day_descriptor(g: pd.DataFrame, pcols: list[str], context_cols: list[str], include_context: bool) -> np.ndarray:
    # Forecast profile is slot-preserving, all known at origin.
    v = [g[pcols].to_numpy(float).reshape(-1)]
    if include_context:
        # d1 raw context columns are broadcast within target day; take row 1 once.
        row = g.iloc[0]
        ctx = np.asarray([pd.to_numeric(row.get(c), errors="coerce") for c in context_cols], dtype=float)
        v.append(ctx)
    return np.concatenate(v)


def distances(candidate_desc: np.ndarray, query: np.ndarray) -> np.ndarray:
    med = np.nanmedian(candidate_desc, axis=0)
    X = np.where(np.isfinite(candidate_desc), candidate_desc, med)
    q = np.where(np.isfinite(query), query, med)
    scale = np.nanstd(X, axis=0)
    scale = np.where((~np.isfinite(scale)) | (scale < 1e-6), 1.0, scale)
    return np.sqrt(np.mean(((X - q) / scale) ** 2, axis=1))


def build_selection_map(
    slot: pd.DataFrame,
    groups: dict[str, list[str]],
    context_cols: list[str],
    target_days: list[str],
    pool_days: int,
) -> tuple[dict[tuple[str, str], tuple[list[str], np.ndarray]], pd.DataFrame]:
    days, dmap = complete_days(slot)
    pcols = profile_columns(groups)
    desc_profile = {d: day_descriptor(dmap[d], pcols, context_cols, False) for d in days}
    desc_state = {d: day_descriptor(dmap[d], pcols, context_cols, True) for d in days}
    result: dict[tuple[str, str], tuple[list[str], np.ndarray]] = {}
    audit_rows: list[dict] = []

    for day in target_days:
        if day not in dmap:
            continue
        cutoff = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        cand = [d for d in days if d <= cutoff][-pool_days:]
        if len(cand) < 120:
            raise RuntimeError(f"{day}: insufficient strict candidate pool {len(cand)}")
        for mode, desc in (("profile", desc_profile), ("state", desc_state)):
            X = np.stack([desc[d] for d in cand])
            q = desc[day]
            dd = distances(X, q)
            result[(day, mode)] = (cand, dd)
            audit_rows.append({
                "target_day": day,
                "mode": mode,
                "latest_candidate_day": max(cand),
                "required_latest_candidate_le": cutoff,
                "candidate_count": len(cand),
                "causal_ok": max(cand) <= cutoff,
            })
    audit = pd.DataFrame(audit_rows)
    if audit.empty or not audit.causal_ok.all():
        raise RuntimeError("regime selection causal audit failed")
    return result, audit


def fit_one(
    slot: pd.DataFrame,
    base_features: list[str],
    day: str,
    selected_days: list[str],
    name: str,
    seed: int,
    sample_weight_by_day: dict[str, float] | None = None,
    balanced: bool = True,
) -> pd.DataFrame:
    tr = slot[slot.target_day.isin(selected_days)].copy()
    te = slot[slot.target_day.eq(day)].sort_values("hour_business").copy()
    if len(te) != 24:
        raise RuntimeError(f"{day}: target slots={len(te)}")
    if not selected_days:
        raise RuntimeError(f"{day}: no selected days")
    cutoff = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    if max(selected_days) > cutoff:
        raise RuntimeError(f"{day}: leakage selected latest {max(selected_days)} > {cutoff}")
    y = (tr.target_spread.to_numpy(float) > 0).astype(int)
    sw = None
    if sample_weight_by_day is not None:
        sw = tr.target_day.map(sample_weight_by_day).to_numpy(float)
    m = make_model(seed, balanced=balanced)
    m.fit(tr[base_features], y, sample_weight=sw)
    p = m.predict_proba(te[base_features])[:, 1].astype(float)
    out = te[["target_day", "hour_business", "period", "target_spread"]].copy()
    out["variant"] = name
    out["prob_positive"] = p
    out["predicted_direction"] = np.where(p >= 0.5, 1, -1)
    out["training_last_day"] = max(selected_days)
    out["training_days"] = len(selected_days)
    out["training_min_day"] = min(selected_days)
    return out


def select_top(cand: list[str], dd: np.ndarray, k: int) -> list[str]:
    idx = np.argsort(dd)[: min(k, len(cand))]
    return [cand[i] for i in idx]


def weighted_days(cand: list[str], dd: np.ndarray, top_k: int, temp: float) -> tuple[list[str], dict[str, float]]:
    idx = np.argsort(dd)[: min(top_k, len(cand))]
    ds = [cand[i] for i in idx]
    d = dd[idx]
    scale = max(float(np.nanmedian(d)), 1e-6)
    w = np.exp(-d / (temp * scale))
    w = 0.10 + 0.90 * (w / max(float(w.max()), 1e-9))
    return ds, dict(zip(ds, w.astype(float)))


def summarize(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = ledger.copy(); x["month"] = x.target_day.str[:7]
    rows = []
    for (v, mo), g in x.groupby(["variant", "month"], sort=True):
        rows.append({"variant": v, "month": mo, "days": g.target_day.nunique(), **direction_metrics(g.target_spread, g.predicted_direction)})
    monthly = pd.DataFrame(rows)
    robust_rows = []
    for v, g in monthly.groupby("variant", sort=False):
        acc = g.direction_accuracy.astype(float); bal = g.balanced_direction_accuracy.astype(float)
        gain = acc - g.all_negative_accuracy.astype(float)
        robust_rows.append({
            "variant": v,
            "months": len(g),
            "mean_month_acc": acc.mean(),
            "median_month_acc": acc.median(),
            "min_month_acc": acc.min(),
            "max_month_acc": acc.max(),
            "std_month_acc": acc.std(ddof=0),
            "months_ge_065": int((acc >= .65).sum()),
            "months_ge_070": int((acc >= .70).sum()),
            "mean_month_bal": bal.mean(),
            "min_month_bal": bal.min(),
            "months_beating_all_negative": int((gain > 0).sum()),
            "mean_gain_vs_all_negative": gain.mean(),
        })
    robust = pd.DataFrame(robust_rows).sort_values(["mean_month_acc", "mean_month_bal"], ascending=False)
    return monthly, robust


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube")
    ap.add_argument("--raw-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--pool-days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration9_regime_sample_selection_strictD2")
    args = ap.parse_args()
    if args.end >= "2026-08-15":
        raise RuntimeError("fresh final holdout remains sealed")

    cube = Path(args.cube_root)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    base = p6_features(groups)
    # Add origin-equivalent raw visible context to descriptors only. The forecasting model remains P6 unless named otherwise.
    slot_ctx, context_cols_all, ctx_audit = add_visible_context(slot, args.raw_path)
    context_cols = [c for c in context_cols_all if c.startswith("d1_spread_p")]
    target_days = [str(d) for d in sorted(slot_ctx.target_day.dropna().astype(str).unique()) if args.start <= str(d) <= args.end]
    selections, sel_audit = build_selection_map(slot_ctx, groups, context_cols, target_days, args.pool_days)

    ledger_parts = []
    selection_rows = []
    for i, day in enumerate(target_days, 1):
        cutoff = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        recent90 = [d for d in sorted(slot_ctx.target_day.dropna().astype(str).unique()) if d <= cutoff][-90:]
        ledger_parts.append(fit_one(slot_ctx, base, day, recent90, "P6_recent90", args.seed))

        for mode in ("profile", "state"):
            cand, dd = selections[(day, mode)]
            for k in (45, 60, 90):
                ds = select_top(cand, dd, k)
                name = f"{mode}_top{k}"
                ledger_parts.append(fit_one(slot_ctx, base, day, ds, name, args.seed))
                selection_rows.append({"target_day": day, "variant": name, "latest_selected_day": max(ds), "n_days": len(ds), "strict_ok": max(ds) <= cutoff})

        # Hybrid: preserve adaptation to recent drift while recovering recurring regimes.
        cand, dd = selections[(day, "state")]
        sim45 = select_top(cand, dd, 45)
        recent45 = [d for d in cand if d <= cutoff][-45:]
        union = sorted(set(sim45 + recent45))
        ledger_parts.append(fit_one(slot_ctx, base, day, union, "state_top45_plus_recent45", args.seed))
        selection_rows.append({"target_day": day, "variant": "state_top45_plus_recent45", "latest_selected_day": max(union), "n_days": len(union), "strict_ok": max(union) <= cutoff})

        for top_k, temp in ((120, 0.50), (120, 1.00), (180, 0.75)):
            ds, wm = weighted_days(cand, dd, top_k, temp)
            name = f"state_weight{top_k}_t{temp:.2f}"
            ledger_parts.append(fit_one(slot_ctx, base, day, ds, name, args.seed, wm))
            selection_rows.append({"target_day": day, "variant": name, "latest_selected_day": max(ds), "n_days": len(ds), "strict_ok": max(ds) <= cutoff})

        if i % 20 == 0:
            print(f"sample-selection {i}/{len(target_days)} {day}", flush=True)

    ledger = pd.concat(ledger_parts, ignore_index=True)
    audit = pd.DataFrame(selection_rows)
    if audit.empty or not audit.strict_ok.all():
        raise RuntimeError("selected-training-day strict audit failed")
    if not ledger.training_last_day.le((pd.to_datetime(ledger.target_day) - pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")).all():
        raise RuntimeError("ledger training_last_day strict audit failed")

    monthly, robust = summarize(ledger)
    out = Path(args.output_root)
    atomic_parquet(out / "ledger.parquet", ledger)
    atomic_csv(out / "monthly.csv", monthly)
    atomic_csv(out / "robustness.csv", robust)
    atomic_csv(out / "selection_audit.csv", audit)
    atomic_csv(out / "descriptor_candidate_audit.csv", sel_audit)
    atomic_csv(out / "visible_context_causal_audit.csv", ctx_audit)
    atomic_json(out / "manifest.json", {
        "status": "complete",
        "experiment": "iteration9_regime_sample_selection",
        "forecast_origin": "D-1 14:00",
        "training_labels": "selected complete historical days <= D-2 only",
        "descriptor_profile": "target-day forecast profiles known at origin",
        "descriptor_state": "forecast profile + D-1 p1-p14 visible spread, information-equivalent for historical candidates",
        "target_day_DA_as_feature": False,
        "target_day_actual_as_feature": False,
        "d1_post14_realized_as_feature": False,
        "fresh_final_holdout_reserved": ["2026-08-15", "2026-08-21"],
        "final_holdout_touched": False,
        "literature_basis": [
            "Selection of Calibration Windows for Day-Ahead Electricity Price Forecasting: fixed windows are risky",
            "Shandong/similar-day forecasting: representative historical days can outperform indiscriminate history",
            "Recurring-vs-emergent drift: recurring states may require non-recent historical regimes",
        ],
    })
    print("\nROBUSTNESS\n", robust.to_string(index=False), flush=True)
    top = robust.iloc[0].variant
    print("\nTOP MONTHLY\n", monthly[monthly.variant.eq(top)].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
