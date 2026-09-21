"""Cycle 09 A pilot: compress the legally visible D-1 p1-p14 context.

The current A route exposes several correlated summaries of the visible context. This
pilot compares the full F1 summary package with a smaller, state-oriented package and
adds only causal transformations (momentum, recent-vs-baseline pressure, sign-switch
proxy). It reuses the audited causal Similar-Day builder and the single strict training
helper; no target-day label is used as a feature.

This is experiment-only and never writes production ledgers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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


COMPACT_CONTEXT = [
    "ctx_spread_mean14", "ctx_spread_std14", "ctx_spread_last",
    "ctx_spread_mean3", "ctx_spread_range14", "ctx_spread_absmean14",
    "ctx_spread_positive_rate14", "ctx_spread_slope14",
]
DROP_CONTEXT = [
    "ctx_spread_mean14", "ctx_spread_std14", "ctx_spread_median14", "ctx_spread_last",
    "ctx_spread_mean3", "ctx_spread_min14", "ctx_spread_max14", "ctx_spread_range14",
    "ctx_spread_absmean14", "ctx_spread_positive_rate14", "ctx_spread_negative_rate14",
    "ctx_spread_slope14",
]
SD20 = [
    "sd20_spread_mean", "sd20_spread_median", "sd20_positive_rate",
    "sd20_weighted_positive_rate", "sd20_spread_std", "sd20_mean_distance",
    "sd20_day_positive_rate", "sd20_1_8_positive_rate", "sd20_9_16_positive_rate",
    "sd20_17_24_positive_rate",
]


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def add_context_transforms(slot: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create only algebraic transforms of D-1 p1-p14 summaries."""
    out = slot.copy()
    new: list[str] = []
    pairs = [
        ("ctx_momentum_last14", "ctx_spread_last", "ctx_spread_mean14", "sub"),
        ("ctx_recent_shift3_14", "ctx_spread_mean3", "ctx_spread_mean14", "sub"),
        ("ctx_pressure_to_vol", "ctx_spread_mean14", "ctx_spread_std14", "ratio"),
        ("ctx_abs_pressure_to_vol", "ctx_spread_absmean14", "ctx_spread_std14", "ratio"),
    ]
    for name, left, right, op in pairs:
        a = pd.to_numeric(out[left], errors="coerce")
        b = pd.to_numeric(out[right], errors="coerce")
        if op == "sub":
            out[name] = a - b
        else:
            out[name] = a / b.replace(0, np.nan)
        new.append(name)
    out["ctx_positive_minus_negative14"] = (
        pd.to_numeric(out["ctx_spread_positive_rate14"], errors="coerce")
        - pd.to_numeric(out["ctx_spread_negative_rate14"], errors="coerce")
    )
    new.append("ctx_positive_minus_negative14")
    return out, new


def score(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, month), group in ledger.assign(month=ledger["target_day"].str[:7]).groupby(["variant", "month"], sort=True):
        rows.append({"variant": variant, "month": month, **direction_metrics(
            group["target_spread"].to_numpy(float), group["predicted_direction"].to_numpy(int)
        )})
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    cube = Path(args.cube_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    if pd.Timestamp(args.end) >= pd.Timestamp("2026-08-15"):
        raise RuntimeError("fresh final holdout remains sealed")
    slot["target_day"] = slot["target_day"].astype(str)
    slot, ctx_new = add_context_transforms(slot)
    slot, sd = add_similar_day_features(slot, groups, k_values=(20,), lookback_days=args.lookback_days)
    audit = sd["audit"]
    if audit.empty or not audit["causal_ok"].all():
        raise RuntimeError("Similar-Day causal audit failed")
    latest = pd.to_datetime(audit["latest_candidate_day"])
    required = pd.to_datetime(audit["required_latest_candidate_le"])
    if (latest > required).any():
        raise RuntimeError("Similar-Day candidate exceeds D-2")

    base = p6_features(groups)
    base_no_context = [c for c in base if c not in DROP_CONTEXT]
    compact = _dedupe(base_no_context + COMPACT_CONTEXT + SD20)
    compact_momentum = _dedupe(compact + ctx_new)
    full = _dedupe(base + SD20)
    packages = {
        "A_p6_sd20_full_context": full,
        "A_p6_sd20_compact_context": compact,
        "A_p6_sd20_compact_momentum": compact_momentum,
    }
    all_days = sorted(slot["target_day"].unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    variants = [
        Variant(name=name, family="lgbm", training_days=args.training_days, use_sd=False, segment=False, class_mode="balanced")
        for name in packages
    ]
    ledgers = []
    for variant in variants:
        features = packages[variant.name]
        missing = [c for c in features if c not in slot.columns]
        if missing:
            raise RuntimeError(f"{variant.name} missing features: {missing}")
        ledger = predict_variant(slot, features, variant, target_days, args.seed)
        ledgers.append(ledger)
    ledger = pd.concat(ledgers, ignore_index=True)
    monthly = score(ledger)
    robustness = monthly.groupby("variant", as_index=False).agg(
        months=("month", "nunique"), mean_month_acc=("direction_accuracy", "mean"),
        mean_month_bal=("balanced_direction_accuracy", "mean"),
        mean_positive_recall=("positive_accuracy", "mean"),
        mean_negative_recall=("negative_accuracy", "mean"),
        mean_all_negative=("all_negative_accuracy", "mean"),
    )
    robustness["mean_gain_vs_all_negative"] = robustness["mean_month_acc"] - robustness["mean_all_negative"]
    atomic_parquet(output / "ledger.parquet", ledger)
    atomic_csv(output / "monthly.csv", monthly)
    atomic_csv(output / "robustness.csv", robustness)
    atomic_csv(output / "similar_day_causal_audit.csv", audit)
    feature_rows = [{"variant": name, "feature_count": len(cols), "features": json.dumps(cols, ensure_ascii=False)} for name, cols in packages.items()]
    atomic_csv(output / "feature_packages.csv", pd.DataFrame(feature_rows))
    manifest = {
        "status": "STRICT/PASS",
        "route": "A_context_compression",
        "forecast_origin": "D-1 14:00",
        "training_last_day": "strict_train_days helper, per target day <= D-2",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "similar_day_latest_candidate": "per target day <= D-2",
        "final_holdout_touched": False,
        "screen_range": [args.start, args.end],
        "training_days": args.training_days,
        "lookback_days": args.lookback_days,
        "context_design": "compact F1 summaries plus algebraic momentum/pressure transforms",
        "feature_source": str(cube),
        "pilot_only": True,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "manifest.json", manifest)
    print(robustness.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-root", default="outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--training-days", type=int, default=90)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=20260823)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
