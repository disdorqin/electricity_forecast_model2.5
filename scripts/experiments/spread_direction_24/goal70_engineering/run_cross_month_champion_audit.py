"""Cross-month robustness audit for the strict P6 direct-spread route.

This experiment intentionally evaluates the current champion across many calendar
months before any fresh final holdout is touched.  It reuses the cutoff-safe
Feature Cube and the causal similar-day builder from run_model_screen.py.

Contract:
- forecast origin = D-1 14:00;
- target-day DA/RT/spread/actual grid values are labels only;
- target-day forecast fundamentals are allowed;
- D-1 spread is limited to p1-p14 context;
- complete historical spread / forecast-error state is D-2 or earlier;
- similar-day neighbors are <= D-2.

The fresh 2026-08-15..2026-08-21 block remains untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    Variant,
    add_similar_day_features,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    direction_metrics,
    p6_features,
    predict_variant,
)


def month_label(day: str) -> str:
    return str(day)[:7]


def summarize_monthly(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    work = ledger.copy()
    work["month"] = work["target_day"].map(month_label)
    for (variant, month), g in work.groupby(["variant", "month"], sort=True):
        rows.append({
            "variant": variant,
            "month": month,
            "days": int(g["target_day"].nunique()),
            **direction_metrics(g["target_spread"].to_numpy(), g["predicted_direction"].to_numpy()),
        })
    return pd.DataFrame(rows)


def aggregate_robustness(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, g in monthly.groupby("variant", sort=False):
        acc = pd.to_numeric(g["direction_accuracy"], errors="coerce")
        bal = pd.to_numeric(g["balanced_direction_accuracy"], errors="coerce")
        gain = acc - pd.to_numeric(g["all_negative_accuracy"], errors="coerce")
        rows.append({
            "variant": variant,
            "months": int(g["month"].nunique()),
            "mean_month_acc": float(acc.mean()),
            "median_month_acc": float(acc.median()),
            "min_month_acc": float(acc.min()),
            "max_month_acc": float(acc.max()),
            "std_month_acc": float(acc.std(ddof=0)),
            "months_ge_065": int((acc >= 0.65).sum()),
            "months_ge_070": int((acc >= 0.70).sum()),
            "mean_month_bal": float(bal.mean()),
            "min_month_bal": float(bal.min()),
            "mean_gain_vs_all_negative": float(gain.mean()),
            "months_beating_all_negative": int((gain > 0).sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["mean_month_acc", "mean_month_bal"], ascending=False
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cube-root",
        default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube",
    )
    ap.add_argument(
        "--output-root",
        default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/cross_month_champion_audit",
    )
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.perf_counter()
    cube = Path(args.cube_root)
    out = Path(args.output_root)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    cube_manifest = json.loads((cube / "manifest.json").read_text(encoding="utf-8"))

    base = p6_features(groups)
    slot, sd = add_similar_day_features(slot, groups, k_values=(20,), lookback_days=365)
    atomic_csv(out / "similar_day_causal_audit.csv", sd["audit"])
    if sd["audit"].empty or not sd["audit"]["causal_ok"].all():
        raise RuntimeError("similar-day causal audit failed")

    target_days = [
        str(d) for d in sorted(slot["target_day"].dropna().astype(str).unique())
        if args.start <= str(d) <= args.end
    ]
    if any("2026-08-15" <= d <= "2026-08-21" for d in target_days):
        raise RuntimeError("fresh final holdout must remain untouched")

    variants = [
        Variant("P6_w90", "lgbm", 90, False, False, "balanced"),
        Variant("P6_SD20_w90", "lgbm", 90, True, False, "balanced", 20),
    ]
    ledgers = []
    for i, v in enumerate(variants, 1):
        feats = list(base)
        if v.use_sd:
            feats += [c for c in sd["features"] if c.startswith("sd20_")]
        print(f"[{i}/{len(variants)}] {v.name}: {len(target_days)} days, {len(feats)} features")
        ledgers.append(predict_variant(slot, list(dict.fromkeys(feats)), v, target_days, args.seed))

    ledger = pd.concat(ledgers, ignore_index=True)
    atomic_parquet(out / "ledger.parquet", ledger)
    monthly = summarize_monthly(ledger)
    atomic_csv(out / "monthly.csv", monthly)
    robustness = aggregate_robustness(monthly)
    atomic_csv(out / "robustness.csv", robustness)

    paired = monthly.pivot(index="month", columns="variant", values=["direction_accuracy", "balanced_direction_accuracy", "all_negative_accuracy"])
    paired_rows = []
    for month in sorted(monthly["month"].unique()):
        def get(metric: str, variant: str) -> float:
            try:
                return float(paired.loc[month, (metric, variant)])
            except Exception:
                return math.nan
        base_acc = get("direction_accuracy", "P6_w90")
        sd_acc = get("direction_accuracy", "P6_SD20_w90")
        paired_rows.append({
            "month": month,
            "p6_acc": base_acc,
            "sd20_acc": sd_acc,
            "delta_acc": sd_acc - base_acc,
            "p6_bal": get("balanced_direction_accuracy", "P6_w90"),
            "sd20_bal": get("balanced_direction_accuracy", "P6_SD20_w90"),
            "all_negative": get("all_negative_accuracy", "P6_SD20_w90"),
        })
    atomic_csv(out / "paired_monthly.csv", pd.DataFrame(paired_rows))

    manifest = {
        "status": "complete",
        "experiment": "cross_month_champion_audit",
        "range": [args.start, args.end],
        "fresh_final_holdout_reserved": ["2026-08-15", "2026-08-21"],
        "final_holdout_touched": False,
        "forecast_origin": "D-1 14:00",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "similar_day_contract": "neighbors <= D-2",
        "variants": [v.__dict__ for v in variants],
        "days": len(target_days),
        "runtime_seconds": time.perf_counter() - t0,
        "cube_information_boundary": cube_manifest.get("information_boundary"),
    }
    atomic_json(out / "manifest.json", manifest)
    print("\nROBUSTNESS\n", robustness.to_string(index=False))
    print("\nMONTHLY\n", pd.DataFrame(paired_rows).to_string(index=False))


if __name__ == "__main__":
    main()
