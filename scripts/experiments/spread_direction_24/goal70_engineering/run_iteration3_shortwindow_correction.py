"""Goal-70 iteration 3: short-window / calibration-window ensemble + forecast correction.

Strict experiment-only runner built on the cutoff-safe P6 Feature Cube.

Branch A (deepening current best):
- causal similar-day P6 with short 45/60/75/90-day windows;
- k sensitivity around the iteration-1 k=20 winner;
- prequential calibration-window ensembles using only prior out-of-sample errors.

Branch B (new direction):
- explicit bias-corrected target-day fundamentals using target forecasts plus D-2-or-earlier
  historical forecast errors, inspired by work showing that enhanced load/wind/solar
  forecasts improve downstream spot/intraday market choice.

No target-day actual/DA/RT/spread enters features. Fresh 2026-08-15..08-21 holdout is reserved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    add_similar_day_features,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    dedupe,
    direction_metrics,
    p6_features,
    strict_train_days,
)


def _model(seed=42):
    return lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=180, learning_rate=0.04,
        num_leaves=31, min_child_samples=35, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=-1,
    )


def add_corrected_fundamentals(slot: pd.DataFrame, groups: dict[str, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """Explicit safe forecast+bias features; all error terms in the cube are D-2 or earlier."""
    out = slot.copy()
    fcasts = [c for c in groups["F2"] if c.startswith("fcast_")]
    new = []
    for f in fcasts:
        base = f[len("fcast_"):]
        for w in (7, 28):
            err = f"err_{base}_{w}d_mean"
            if err in out.columns:
                name = f"corrected_{base}_{w}d"
                out[name] = pd.to_numeric(out[f], errors="coerce") + pd.to_numeric(out[err], errors="coerce")
                new.append(name)
    # Corrected physical relationships for the most important market-tightness variables.
    for w in (7, 28):
        load = f"corrected_直调负荷_{w}d"
        renew = f"corrected_新能源总加_{w}d"
        wind = f"corrected_风电总加_{w}d"
        solar = f"corrected_光伏总加_{w}d"
        space = f"corrected_竞价空间_{w}d"
        if load in out and renew in out:
            n = f"corrected_residual_load_{w}d"; out[n] = out[load] - out[renew]; new.append(n)
            n = f"corrected_renewable_share_{w}d"; out[n] = out[renew] / out[load].replace(0, np.nan); new.append(n)
        if load in out and wind in out and solar in out:
            n = f"corrected_residual_ws_{w}d"; out[n] = out[load] - out[wind] - out[solar]; new.append(n)
        if load in out and space in out:
            n = f"corrected_bidding_space_ratio_{w}d"; out[n] = out[space] / out[load].replace(0, np.nan); new.append(n)
    return out, dedupe(new)


def predict(slot: pd.DataFrame, features: list[str], *, name: str, target_days: list[str], training_days: int, seed: int) -> pd.DataFrame:
    all_days = sorted(slot.target_day.dropna().astype(str).unique())
    rows = []
    for day in target_days:
        train_days = strict_train_days(all_days, day, training_days)
        if len(train_days) < min(45, training_days):
            raise RuntimeError(f"{name} {day}: insufficient train days")
        tr = slot[slot.target_day.isin(train_days)]
        te = slot[slot.target_day.eq(day)].sort_values("hour_business")
        if len(te) != 24:
            raise RuntimeError(f"{name} {day}: expected 24 slots")
        y = (tr.target_spread.to_numpy(float) > 0).astype(int)
        m = _model(seed)
        m.fit(tr[features], y)
        p = m.predict_proba(te[features])[:, 1].astype(float)
        o = te[["target_day", "hour_business", "period", "target_spread"]].copy()
        o["variant"] = name
        o["prob_positive"] = p
        o["predicted_direction"] = np.where(p >= 0.5, 1, -1)
        o["training_days"] = len(train_days)
        o["training_last_day"] = train_days[-1]
        rows.append(o)
    return pd.concat(rows, ignore_index=True)


def _combine_fixed(ledger: pd.DataFrame, names: list[str], name: str, weights=None) -> pd.DataFrame:
    keys = ["target_day", "hour_business", "period", "target_spread"]
    p = ledger[ledger.variant.isin(names)].pivot_table(index=keys, columns="variant", values="prob_positive").reset_index()
    if weights is None:
        weights = np.ones(len(names), float) / len(names)
    prob = sum(float(w) * p[n].to_numpy(float) for w, n in zip(weights, names))
    o = p[keys].copy(); o["variant"] = name; o["prob_positive"] = prob; o["predicted_direction"] = np.where(prob >= .5, 1, -1)
    return o


def _prequential_ensemble(ledger: pd.DataFrame, names: list[str], *, name: str, history_days: int, eta: float, period_specific: bool) -> pd.DataFrame:
    keys = ["target_day", "hour_business", "period", "target_spread"]
    base = ledger[ledger.variant.isin(names)].copy()
    days = sorted(base.target_day.unique())
    rows = []
    for day in days:
        target = base[base.target_day.eq(day)].pivot_table(index=keys, columns="variant", values="prob_positive").reset_index()
        max_hist_day = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        past_days = [d for d in days if d <= max_hist_day][-history_days:]
        weights_by_period = {}
        periods = ["1_8", "9_16", "17_24"] if period_specific else ["all"]
        for period in periods:
            hist = base[base.target_day.isin(past_days)]
            if period_specific:
                hist = hist[hist.period.eq(period)]
            losses = []
            for n in names:
                g = hist[hist.variant.eq(n)]
                if g.empty:
                    losses.append(0.5); continue
                acc = float((np.where(g.prob_positive.to_numpy(float) >= .5, 1, -1) == np.sign(g.target_spread.to_numpy(float))).mean())
                losses.append(1.0 - acc)
            ww = np.exp(-eta * np.asarray(losses, float))
            ww = ww / ww.sum() if ww.sum() > 0 else np.ones(len(names)) / len(names)
            weights_by_period[period] = ww
        probs = []
        for _, r in target.iterrows():
            period = str(r.period) if period_specific else "all"
            ww = weights_by_period[period]
            probs.append(float(sum(ww[i] * float(r[n]) for i, n in enumerate(names))))
        o = target[keys].copy(); o["variant"] = name; o["prob_positive"] = probs; o["predicted_direction"] = np.where(np.asarray(probs) >= .5, 1, -1)
        o["weight_history_days"] = len(past_days)
        rows.append(o)
    return pd.concat(rows, ignore_index=True)


def summarize(ledger: pd.DataFrame) -> pd.DataFrame:
    splits = {
        "prehistory30": ("2026-06-16", "2026-07-15"),
        "selection15": ("2026-07-16", "2026-07-30"),
        "validation15": ("2026-07-31", "2026-08-14"),
        "combined30": ("2026-07-16", "2026-08-14"),
    }
    rows = []
    for split, (a, b) in splits.items():
        p = ledger[(ledger.target_day >= a) & (ledger.target_day <= b)]
        for n, g in p.groupby("variant", sort=False):
            rows.append({"split": split, "variant": n, "days": int(g.target_day.nunique()), **direction_metrics(g.target_spread.to_numpy(), g.predicted_direction.to_numpy())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube")
    ap.add_argument("--output-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration3_shortwindow_correction")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(); t0 = time.perf_counter()
    cube = Path(args.cube_root); out = Path(args.output_root)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    cube_manifest = json.loads((cube / "manifest.json").read_text(encoding="utf-8"))
    base = p6_features(groups)
    slot, sd = add_similar_day_features(slot, groups, k_values=(15, 20, 25, 30), lookback_days=365)
    atomic_csv(out / "similar_day_causal_audit.csv", sd["audit"])
    slot, corrected = add_corrected_fundamentals(slot, groups)
    target_days = [d for d in sorted(slot.target_day.unique()) if "2026-06-16" <= d <= "2026-08-14"]

    ledgers = []
    specs = [
        ("A_sd20_w45", 45, 20, False),
        ("A_sd20_w60", 60, 20, False),
        ("A_sd20_w75", 75, 20, False),
        ("A_sd20_w90", 90, 20, False),
        ("A_sd15_w60", 60, 15, False),
        ("A_sd25_w60", 60, 25, False),
        ("A_sd30_w60", 60, 30, False),
        ("B_sd20_corrected_w60", 60, 20, True),
        ("B_p6_corrected_w60", 60, 0, True),
    ]
    for i, (name, w, k, corr) in enumerate(specs, 1):
        feats = list(base)
        if k:
            feats += [c for c in sd["features"] if c.startswith(f"sd{k}_")]
        if corr:
            feats += corrected
        feats = dedupe(feats)
        print(f"[{i}/{len(specs)}] {name} features={len(feats)}", flush=True)
        ledgers.append(predict(slot, feats, name=name, target_days=target_days, training_days=w, seed=args.seed))
    base_ledger = pd.concat(ledgers, ignore_index=True)

    # Calibration-window integration: fixed and strictly prequential alternatives.
    core = ["A_sd20_w45", "A_sd20_w60", "A_sd20_w90"]
    ens = [
        _combine_fixed(base_ledger, core, "A_window_equal_45_60_90"),
        _combine_fixed(base_ledger, ["A_sd20_w60", "A_sd20_w90"], "A_window_equal_60_90"),
        _combine_fixed(base_ledger, ["A_sd20_w60", "B_sd20_corrected_w60"], "AB_equal_base_corrected"),
    ]
    for h, eta in ((7, 3.0), (14, 3.0), (14, 6.0), (28, 3.0)):
        ens.append(_prequential_ensemble(base_ledger, core, name=f"A_preq_global_h{h}_e{eta:g}", history_days=h, eta=eta, period_specific=False))
        ens.append(_prequential_ensemble(base_ledger, core, name=f"A_preq_period_h{h}_e{eta:g}", history_days=h, eta=eta, period_specific=True))
    ledger = pd.concat([base_ledger, *ens], ignore_index=True)
    atomic_parquet(out / "ledger.parquet", ledger)
    summary = summarize(ledger); atomic_csv(out / "summary.csv", summary)
    val = summary[summary.split.eq("validation15")].sort_values(["direction_accuracy", "balanced_direction_accuracy"], ascending=False)
    atomic_csv(out / "validation_ranking.csv", val)
    atomic_json(out / "manifest.json", {
        "status": "complete", "experiment": "goal70_iteration3_shortwindow_correction",
        "forecast_origin": "D-1 14:00", "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False, "similar_day_latest": "D-2",
        "screen_range": ["2026-06-16", "2026-08-14"], "fresh_final_holdout_reserved": ["2026-08-15", "2026-08-21"], "final_holdout_touched": False,
        "branch_A": "short-window DSA + calibration-window ensemble/prequential drift adaptation",
        "branch_B": "explicit correction of target forecast fundamentals using D-2-or-earlier forecast errors",
        "literature_basis": [
            "Huang et al. 2024 Applied Energy: Shandong DSA and rolling-window analysis",
            "Michalakopoulos et al. 2025: LightGBM robust with short 45/60d training windows",
            "Liu et al. 2024 CSEE: calibration-window ensemble via BMA",
            "Maciejowska et al.: enhancing load/wind/solar forecasts improves electricity-price market choice",
        ],
        "cube_information_boundary": cube_manifest.get("information_boundary"),
        "runtime_seconds": time.perf_counter() - t0,
    })
    print("\nVALIDATION RANKING\n", val.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
