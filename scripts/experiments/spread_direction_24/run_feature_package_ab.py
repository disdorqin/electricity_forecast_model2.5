"""Semantic feature-package ablation for cutoff-safe spread direction forecasting.

This runner consumes the prebuilt Feature Cube and evaluates cheap LightGBM
classifiers under the same walk-forward contract.  It is intentionally focused
on *input organization*, not model tuning.

Every package is defined from auditable Feature Cube groups.  No target-day
actual values are used as features; the cube already enforces D-1 14:00 origin,
D-2-or-earlier historical spread/error state, and target-day forecast-only grid
features.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


KEY_ERR_TOKENS = (
    "风电总加",
    "光伏总加",
    "直调负荷",
    "竞价空间",
    "新能源总加",
)


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


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _metrics(frame: pd.DataFrame) -> dict:
    true = frame["y_true_spread"].to_numpy(float)
    pred = frame["predicted_direction"].to_numpy(int)
    sign = np.sign(true)
    eligible = sign != 0
    correct = eligible & (sign == pred)
    pos = sign > 0
    neg = sign < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "days": int(frame["target_day"].nunique()),
        "n_slots": int(len(frame)),
        "n_positive_actual": int(pos.sum()),
        "n_negative_actual": int(neg.sum()),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
    }


def _dedupe(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))


def _feature_packages(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    base = groups["F0"] + groups["F1"]
    phys = groups["F3"] + groups["F4"]
    raw = groups["F2"]
    err7 = [c for c in groups["F5"] if "_7d_" in c]
    err28 = [c for c in groups["F5"] if "_28d_" in c]
    err_core = [
        c for c in groups["F5"]
        if c.startswith("err_net_load_") or any(token in c for token in KEY_ERR_TOKENS)
    ]
    uncert_core = [c for c in groups["F6"] if any(token in c for token in KEY_ERR_TOKENS)]
    regime = groups["F7"]
    packages = {
        "P0_base": base,
        "P1_base_phys_ramp": base + phys,
        "P2_base_raw_phys_ramp": base + raw + phys,
        "P3_phys_err7_all": base + phys + err7,
        "P4_phys_err28_all": base + phys + err28,
        "P5_phys_err_core": base + phys + err_core,
        "P6_raw_phys_err_core": base + raw + phys + err_core,
        "P7_phys_err_core_uncert": base + phys + err_core + uncert_core,
        "P8_phys_err_core_regime": base + phys + err_core + regime,
    }
    return {k: _dedupe(v) for k, v in packages.items()}


def _classifier(args) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=-1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    p.add_argument("--output-root", default="outputs/experiments/spread_feature_package_ab_20260821")
    p.add_argument("--start", default="2026-06-16")
    p.add_argument("--end", default="2026-07-30")
    p.add_argument("--training-days", type=int, default=90)
    p.add_argument("--min-training-days", type=int, default=60)
    p.add_argument("--packages", default="all")
    p.add_argument("--n-estimators", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--max-depth", type=int, default=-1)
    p.add_argument("--min-child-samples", type=int, default=40)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    started = time.perf_counter()
    cube = Path(args.cube_root)
    out = Path(args.output_root)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    cube_manifest = json.loads((cube / "manifest.json").read_text(encoding="utf-8"))
    packages = _feature_packages(groups)
    if args.packages != "all":
        requested = [x.strip() for x in args.packages.split(",") if x.strip()]
        unknown = [x for x in requested if x not in packages]
        if unknown:
            raise ValueError(f"unknown packages: {unknown}")
        packages = {k: packages[k] for k in requested}

    all_days = sorted(slot["target_day"].dropna().astype(str).unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    if not target_days:
        raise ValueError("no target days")

    ledger_rows: list[pd.DataFrame] = []
    timing_rows: list[dict] = []
    importance_rows: list[dict] = []

    for package_name, features in packages.items():
        missing = [c for c in features if c not in slot.columns]
        if missing:
            raise ValueError(f"{package_name}: missing cube features {missing[:8]}")
        package_started = time.perf_counter()
        last_clf = None
        for target_day in target_days:
            target_idx = all_days.index(target_day)
            train_days = all_days[max(0, target_idx - args.training_days): target_idx]
            if len(train_days) < args.min_training_days:
                raise ValueError(f"{target_day}: only {len(train_days)} training days")
            train = slot[slot["target_day"].isin(train_days)]
            test = slot[slot["target_day"].eq(target_day)].sort_values("hour_business")
            if len(test) != 24:
                raise ValueError(f"{target_day}: expected 24 slots, got {len(test)}")
            X_train = train[features]
            X_test = test[features]
            y_train = train["target_spread"].to_numpy(float)
            y_test = test["target_spread"].to_numpy(float)
            eligible_train = y_train != 0
            clf = _classifier(args)
            clf.fit(X_train.loc[eligible_train], (y_train[eligible_train] > 0).astype(int))
            prob = clf.predict_proba(X_test)[:, 1].astype(float)
            pred = np.where(prob >= 0.5, 1, -1)
            day = test[["target_day", "时刻", "hour_business", "period"]].copy()
            day["package"] = package_name
            day["n_features"] = len(features)
            day["y_true_spread"] = y_test
            day["direction_probability_positive"] = prob
            day["predicted_direction"] = pred
            day["actual_direction"] = np.sign(y_test).astype(int)
            day["direction_correct"] = day["predicted_direction"].eq(day["actual_direction"])
            ledger_rows.append(day)
            last_clf = clf
        elapsed = time.perf_counter() - package_started
        timing_rows.append({"package": package_name, "n_features": len(features), "elapsed_seconds": elapsed})
        if last_clf is not None:
            gains = last_clf.booster_.feature_importance(importance_type="gain")
            total = float(gains.sum())
            for feature, gain in zip(features, gains):
                importance_rows.append({
                    "package": package_name,
                    "feature": feature,
                    "gain": float(gain),
                    "gain_share": float(gain / total) if total > 0 else 0.0,
                })
        print(f"{package_name}: {len(features)} features, {elapsed:.2f}s")

    ledger = pd.concat(ledger_rows, ignore_index=True)
    _atomic_parquet(out / "ledger.parquet", ledger)
    split_ranges = {
        "overall45": (args.start, args.end),
        "development30": (args.start, "2026-07-15"),
        "confirmation15": ("2026-07-16", args.end),
    }
    summary_rows = []
    for split, (start, end) in split_ranges.items():
        part = ledger[(ledger["target_day"] >= start) & (ledger["target_day"] <= end)]
        if part.empty:
            continue
        for package_name, g in part.groupby("package", sort=False):
            summary_rows.append({"split": split, "package": package_name, **_metrics(g)})
    summary = pd.DataFrame(summary_rows)
    _atomic_csv(out / "summary.csv", summary)
    _atomic_csv(out / "timing.csv", pd.DataFrame(timing_rows))
    importance = pd.DataFrame(importance_rows).sort_values(["package", "gain"], ascending=[True, False])
    _atomic_csv(out / "feature_importance_last_day.csv", importance)

    manifest = {
        "pipeline": "spread_feature_package_ab",
        "status": "complete",
        "cube_root": str(cube),
        "cube_schema": cube_manifest.get("schema"),
        "start": args.start,
        "end": args.end,
        "training_days": args.training_days,
        "min_training_days": args.min_training_days,
        "packages": {k: v for k, v in packages.items()},
        "model": {
            "type": "LGBMClassifier",
            "class_weight": "balanced",
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
        },
        "information_boundary": cube_manifest.get("information_boundary"),
        "runtime_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(out / "manifest.json", manifest)
    overall = summary[summary["split"].eq("overall45")].sort_values("balanced_direction_accuracy", ascending=False)
    print(overall[["package", "direction_accuracy", "positive_accuracy", "negative_accuracy", "balanced_direction_accuracy"]].to_string(index=False))
    print(json.dumps({"runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
