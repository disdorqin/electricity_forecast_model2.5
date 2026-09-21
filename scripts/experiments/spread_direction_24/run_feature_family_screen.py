"""Forward screen small, physically meaningful feature families on top of F0+F1.

The base is the strongest cheap feature configuration found so far:
F0 cutoff-safe spread history + F1 D-1 p1-p14 current-regime summaries.
Each candidate family is added independently and evaluated by a balanced
LightGBM direction classifier in strict daily walk-forward fashion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GROUP_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _clf(args) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )


def _metrics(frame: pd.DataFrame) -> dict:
    true = np.sign(frame["y_true_spread"].to_numpy(float))
    pred = frame["pred_direction"].to_numpy(int)
    eligible = true != 0
    correct = eligible & (true == pred)
    pos = true > 0
    neg = true < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "days": int(frame["target_day"].nunique()),
        "n_slots": int(len(frame)),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
    }


def _prefixed(columns: set[str], prefixes: tuple[str, ...]) -> list[str]:
    return sorted([c for c in columns if c.startswith(prefixes)])


def build_families(slot: pd.DataFrame) -> dict[str, list[str]]:
    cols = set(slot.columns.astype(str))
    def exact(*names: str) -> list[str]:
        return [x for x in names if x in cols]
    families = {
        "hour_numeric": exact("hour_business"),
        "raw_space": exact("fcast_竞价空间"),
        "raw_local": exact("fcast_地方电厂总加"),
        "raw_interconnect": exact("fcast_联络线受电负荷"),
        "raw_nuclear": exact("fcast_核电总加"),
        "raw_self": exact("fcast_自备机组总加"),
        "raw_trial": exact("fcast_试验机组总加"),
        "raw_conventional_core": exact("fcast_地方电厂总加", "fcast_核电总加", "fcast_自备机组总加"),
        "raw_load": exact("fcast_直调负荷"),
        "raw_renew": exact("fcast_风电总加", "fcast_光伏总加", "fcast_新能源总加"),
        "raw_conventional": exact("fcast_地方电厂总加", "fcast_联络线受电负荷", "fcast_核电总加", "fcast_自备机组总加", "fcast_试验机组总加"),
        "physical_residual": exact("residual_load_ws", "residual_load_renew"),
        "physical_shares": exact("renewable_share", "wind_share", "solar_share"),
        "physical_tightness": exact("bidding_space_ratio", "renewable_minus_space", "interconnect_share"),
        "ramp_load_residual": exact("ramp_load", "ramp2_load", "ramp_residual_load", "ramp2_residual_load"),
        "ramp_renew": exact("ramp_wind", "ramp2_wind", "ramp_solar", "ramp2_solar", "ramp_renewable", "ramp2_renewable"),
        "err_load": _prefixed(cols, ("err_直调负荷_",)),
        "err_renew": _prefixed(cols, ("err_风电总加_", "err_光伏总加_", "err_新能源总加_")),
        "err_space": _prefixed(cols, ("err_竞价空间_",)),
        "err_net_load": _prefixed(cols, ("err_net_load_",)),
        "uncert_core": _prefixed(cols, ("uncert_直调负荷_", "uncert_风电总加_", "uncert_光伏总加_", "uncert_新能源总加_", "uncert_竞价空间_")),
        "regime_core": exact(
            "regime_z_residual_load_renew", "regime_z_renewable_share", "regime_z_bidding_space_ratio",
            "regime_high_residual_load_renew", "regime_low_residual_load_renew",
            "regime_high_renewable_share", "regime_low_renewable_share",
            "regime_high_bidding_space_ratio", "regime_low_bidding_space_ratio",
        ),
        "regime_supply": exact(
            "regime_z_fcast_地方电厂总加", "regime_z_fcast_联络线受电负荷", "regime_z_fcast_核电总加",
            "regime_z_fcast_自备机组总加", "regime_z_fcast_试验机组总加",
        ),
        "regime_renew": exact("regime_z_fcast_风电总加", "regime_z_fcast_光伏总加", "regime_z_fcast_新能源总加"),
        "regime_space_load": exact("regime_z_fcast_竞价空间", "regime_z_fcast_直调负荷"),
    }
    return {k: v for k, v in families.items() if v}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    p.add_argument("--output-root", required=True)
    p.add_argument("--start", default="2026-06-16")
    p.add_argument("--end", default="2026-07-30")
    p.add_argument("--training-days", type=int, default=365)
    p.add_argument("--min-training-days", type=int, default=180)
    p.add_argument("--families", default="")
    p.add_argument("--base-groups", default="F0,F1", help="comma-separated cube groups used as the base")
    p.add_argument("--base-families", default="", help="comma-separated candidate families to bake into the base before screening")
    p.add_argument("--n-estimators", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--min-child-samples", type=int, default=40)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    started = time.perf_counter()
    cube = Path(args.cube_root)
    out = Path(args.output_root)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    base_groups = [x.strip() for x in args.base_groups.split(",") if x.strip()]
    bad_groups = [x for x in base_groups if x not in groups]
    if bad_groups:
        raise ValueError(f"unknown base groups {bad_groups}")
    base = [f for g in base_groups for f in groups[g]]
    families = build_families(slot)
    families.update({
        "context_all": list(groups["F1"]),
        "context_level": [x for x in ["ctx_spread_mean14", "ctx_spread_median14", "ctx_spread_last", "ctx_spread_mean3"] if x in slot.columns],
        "context_volatility": [x for x in ["ctx_spread_std14", "ctx_spread_min14", "ctx_spread_max14", "ctx_spread_range14", "ctx_spread_absmean14"] if x in slot.columns],
        "context_sign_shape": [x for x in ["ctx_spread_positive_rate14", "ctx_spread_negative_rate14", "ctx_spread_slope14"] if x in slot.columns],
        "context_core": [x for x in ["ctx_spread_median14", "ctx_spread_std14", "ctx_spread_min14", "ctx_spread_max14", "ctx_spread_range14", "ctx_spread_absmean14", "ctx_spread_last"] if x in slot.columns],
        "context_top4": [x for x in ["ctx_spread_median14", "ctx_spread_std14", "ctx_spread_min14", "ctx_spread_max14"] if x in slot.columns],
    })
    base_families = [x.strip() for x in args.base_families.split(",") if x.strip()]
    bad_base = [x for x in base_families if x not in families]
    if bad_base:
        raise ValueError(f"unknown base families {bad_base}; available={sorted(families)}")
    for family in base_families:
        base.extend(families[family])
    base = list(dict.fromkeys(base))
    requested = [x.strip() for x in args.families.split(",") if x.strip()] or [x for x in families if x not in base_families]
    bad = [x for x in requested if x not in families]
    if bad:
        raise ValueError(f"unknown families {bad}; available={sorted(families)}")

    all_days = sorted(slot["target_day"].astype(str).unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    rows = []
    timing = []
    for family in ["BASE", *requested]:
        extra = [] if family == "BASE" else families[family]
        features = list(dict.fromkeys([*base, *extra]))
        t0 = time.perf_counter()
        for target_day in target_days:
            idx = all_days.index(target_day)
            train_days = all_days[max(0, idx - args.training_days): idx]
            if len(train_days) < args.min_training_days:
                raise ValueError(f"{target_day}: insufficient training days")
            train = slot[slot["target_day"].isin(train_days)]
            test = slot[slot["target_day"].eq(target_day)].sort_values("hour_business")
            y = train["target_spread"].to_numpy(float)
            eligible = y != 0
            model = _clf(args)
            model.fit(train.loc[eligible, features], (y[eligible] > 0).astype(int))
            prob = model.predict_proba(test[features])[:, 1]
            pred = np.where(prob >= 0.5, 1, -1)
            for td, hour, period, yt, pp, pdirection in zip(
                test["target_day"], test["hour_business"], test["period"], test["target_spread"], prob, pred
            ):
                rows.append({
                    "family": family,
                    "target_day": td,
                    "hour_business": int(hour),
                    "period": period,
                    "y_true_spread": float(yt),
                    "prob_positive": float(pp),
                    "pred_direction": int(pdirection),
                    "n_features": len(features),
                })
        timing.append({"family": family, "n_features": len(features), "elapsed_seconds": time.perf_counter() - t0})
        print(f"{family}: +{len(extra)} => {len(features)} features, {timing[-1]['elapsed_seconds']:.2f}s", flush=True)

    ledger = pd.DataFrame(rows)
    _atomic_csv(out / "timing.csv", pd.DataFrame(timing))
    splits = {
        "overall45": (args.start, args.end),
        "development30": (args.start, "2026-07-15"),
        "confirmation15": ("2026-07-16", args.end),
    }
    summary = []
    for split, (lo, hi) in splits.items():
        s = ledger[ledger["target_day"].between(lo, hi)]
        for family, g in s.groupby("family", sort=False):
            summary.append({"split": split, "family": family, "n_features": int(g["n_features"].iloc[0]), **_metrics(g)})
    summary = pd.DataFrame(summary)
    _atomic_csv(out / "summary.csv", summary)
    # Keep the ledger compact but auditable.
    out.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(out / "ledger.parquet", index=False)

    base_dev = summary[(summary["split"].eq("development30")) & (summary["family"].eq("BASE"))].iloc[0]
    base_conf = summary[(summary["split"].eq("confirmation15")) & (summary["family"].eq("BASE"))].iloc[0]
    compare = summary[summary["split"].isin(["development30", "confirmation15"])].pivot(index="family", columns="split", values="balanced_direction_accuracy")
    compare["delta_dev_vs_base"] = compare["development30"] - float(base_dev["balanced_direction_accuracy"])
    compare["delta_conf_vs_base"] = compare["confirmation15"] - float(base_conf["balanced_direction_accuracy"])
    compare = compare.sort_values(["delta_dev_vs_base", "delta_conf_vs_base"], ascending=False).reset_index()
    _atomic_csv(out / "family_comparison.csv", compare)

    manifest = {
        "pipeline": "spread_feature_family_forward_screen",
        "status": "complete",
        "base": "+".join(base_groups) + ("+" + "+".join(base_families) if base_families else ""),
        "base_groups": base_groups,
        "base_families": base_families,
        "families": {k: families[k] for k in requested},
        "start": args.start,
        "end": args.end,
        "training_days": args.training_days,
        "selection_rule": "prefer positive development30 delta and non-negative confirmation15 delta; holdout untouched",
        "runtime_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(out / "manifest.json", manifest)
    print(compare.to_string(index=False))


if __name__ == "__main__":
    main()
