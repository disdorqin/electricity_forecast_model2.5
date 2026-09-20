"""Fast cumulative feature-group ablation on the cutoff-safe spread Feature Cube.

Uses two cheap LightGBM probes per group:
- regression probe predicts signed spread;
- balanced direction classifier predicts sign, with regression magnitude used only
  for an optional hybrid numeric diagnostic.

All target-day features come from the prebuilt cube; the runner never touches
production model files, ledgers, weights, fusion, classifier, or delivery stages.
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

from scripts.experiments.spread_direction_24.spread_metrics import smape_percent  # noqa: E402
from utils.resolution import HOURLY  # noqa: E402

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


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    true = frame["y_true_spread"].to_numpy(float)
    pred = frame[pred_col].to_numpy(float)
    true_sign = np.sign(true)
    pred_sign = np.sign(pred)
    eligible = true_sign != 0
    correct = eligible & (true_sign == pred_sign)
    pos = true_sign > 0
    neg = true_sign < 0
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
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "spread_smape_pct": smape_percent(true, pred),
    }


def _classifier_metrics(frame: pd.DataFrame) -> dict:
    true_sign = np.sign(frame["y_true_spread"].to_numpy(float))
    pred_sign = frame["classifier_direction"].to_numpy(int)
    eligible = true_sign != 0
    correct = eligible & (true_sign == pred_sign)
    pos = true_sign > 0
    neg = true_sign < 0
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


def _regressor(args) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
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


def _cumulative_features(groups: dict[str, list[str]], group_name: str) -> list[str]:
    idx = GROUP_ORDER.index(group_name)
    return [f for g in GROUP_ORDER[: idx + 1] for f in groups[g]]


def _period_metrics(ledger: pd.DataFrame, pred_col: str, model_kind: str) -> pd.DataFrame:
    rows = []
    for (feature_group, period), g in ledger.groupby(["feature_group", "period"], sort=False):
        m = _classifier_metrics(g) if model_kind == "classifier" else _metrics(g, pred_col)
        rows.append({"feature_group": feature_group, "period": period, "model_kind": model_kind, **m})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    p.add_argument("--output-root", default="outputs/experiments/spread_feature_group_ab_20260821")
    p.add_argument("--start", default="2026-06-16")
    p.add_argument("--end", default="2026-07-30")
    p.add_argument("--training-days", type=int, default=365)
    p.add_argument("--min-training-days", type=int, default=180)
    p.add_argument("--groups", default=",".join(GROUP_ORDER))
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
    requested_groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    invalid = [g for g in requested_groups if g not in GROUP_ORDER]
    if invalid:
        raise ValueError(f"unknown groups: {invalid}")

    all_days = sorted(slot["target_day"].dropna().astype(str).unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    if not target_days:
        raise ValueError("no target days")

    ledger_rows: list[pd.DataFrame] = []
    timing_rows: list[dict] = []
    importance_rows: list[dict] = []
    for feature_group in requested_groups:
        features = _cumulative_features(groups, feature_group)
        group_started = time.perf_counter()
        for target_day in target_days:
            target_idx = all_days.index(target_day)
            train_days = all_days[max(0, target_idx - args.training_days): target_idx]
            if len(train_days) < args.min_training_days:
                raise ValueError(f"{target_day}: only {len(train_days)} training days")
            train = slot[slot["target_day"].isin(train_days)]
            test = slot[slot["target_day"].eq(target_day)].sort_values("hour_business")
            if len(test) != HOURLY.slots_per_day:
                raise ValueError(f"{target_day}: expected 24 slots, got {len(test)}")
            X_train = train[features]
            X_test = test[features]
            y_train = train["target_spread"].to_numpy(float)
            y_test = test["target_spread"].to_numpy(float)

            reg = _regressor(args)
            reg.fit(X_train, y_train)
            reg_pred = reg.predict(X_test).astype(float)

            eligible_train = y_train != 0
            clf = _classifier(args)
            clf.fit(X_train.loc[eligible_train], (y_train[eligible_train] > 0).astype(int))
            prob = clf.predict_proba(X_test)[:, 1].astype(float)
            cls_dir = np.where(prob >= 0.5, 1, -1)
            hybrid = np.abs(reg_pred) * cls_dir

            day = test[["target_day", "时刻", "hour_business", "period"]].copy()
            day["feature_group"] = feature_group
            day["n_features"] = len(features)
            day["y_true_spread"] = y_test
            day["regression_prediction"] = reg_pred
            day["classifier_prob_positive"] = prob
            day["classifier_direction"] = cls_dir
            day["hybrid_prediction"] = hybrid
            ledger_rows.append(day)

            # Final target day importance is enough for fast diagnosis; average over all days is unnecessary overhead.
            if target_day == target_days[-1]:
                for model_kind, model in (("regression", reg), ("classifier", clf)):
                    for fname, gain in zip(features, model.booster_.feature_importance(importance_type="gain")):
                        importance_rows.append({
                            "feature_group": feature_group,
                            "model_kind": model_kind,
                            "feature": fname,
                            "gain": float(gain),
                        })
        timing_rows.append({
            "feature_group": feature_group,
            "n_features": len(features),
            "elapsed_seconds": time.perf_counter() - group_started,
        })
        print(f"{feature_group}: {len(features)} features, {timing_rows[-1]['elapsed_seconds']:.2f}s", flush=True)

    ledger = pd.concat(ledger_rows, ignore_index=True)
    _atomic_parquet(out / "ledger.parquet", ledger)

    split_defs = {
        "overall45": (args.start, args.end),
        "development30": (args.start, "2026-07-15"),
        "confirmation15": ("2026-07-16", args.end),
    }
    summary_rows = []
    for split, (lo, hi) in split_defs.items():
        frame = ledger[ledger["target_day"].between(lo, hi)]
        if frame.empty:
            continue
        for feature_group, g in frame.groupby("feature_group", sort=False):
            reg_m = _metrics(g, "regression_prediction")
            cls_m = _classifier_metrics(g)
            hyb_m = _metrics(g, "hybrid_prediction")
            summary_rows.extend([
                {"split": split, "feature_group": feature_group, "model_kind": "regression", **reg_m},
                {"split": split, "feature_group": feature_group, "model_kind": "classifier", **cls_m, "mae": math.nan, "rmse": math.nan, "spread_smape_pct": math.nan},
                {"split": split, "feature_group": feature_group, "model_kind": "hybrid", **hyb_m},
            ])
    summary = pd.DataFrame(summary_rows)
    _atomic_csv(out / "summary.csv", summary)
    _atomic_csv(out / "timing.csv", pd.DataFrame(timing_rows))

    period_frames = []
    for kind, pred_col in (("regression", "regression_prediction"), ("classifier", "classifier_direction"), ("hybrid", "hybrid_prediction")):
        period_frames.append(_period_metrics(ledger, pred_col, kind))
    _atomic_csv(out / "period_metrics.csv", pd.concat(period_frames, ignore_index=True))

    imp = pd.DataFrame(importance_rows)
    if not imp.empty:
        imp["gain_share"] = imp.groupby(["feature_group", "model_kind"])["gain"].transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)
        _atomic_csv(out / "feature_importance_last_day.csv", imp.sort_values(["feature_group", "model_kind", "gain"], ascending=[True, True, False]))

    # Cheap non-model baselines on the same evaluation range.
    baseline_frame = slot[slot["target_day"].between(args.start, args.end)].copy()
    baseline_rows = []
    if "spread_same_slot_28d_median" in baseline_frame.columns:
        tmp = baseline_frame.rename(columns={"target_spread": "y_true_spread", "spread_same_slot_28d_median": "baseline_pred"})
        baseline_rows.append({"baseline": "rolling_same_slot_median28", **_metrics(tmp, "baseline_pred")})
    tmp = baseline_frame.rename(columns={"target_spread": "y_true_spread"}).copy()
    tmp["always_negative"] = -1.0
    baseline_rows.append({"baseline": "always_negative_sign", **_metrics(tmp, "always_negative")})
    _atomic_csv(out / "baselines.csv", pd.DataFrame(baseline_rows))

    manifest = {
        "pipeline": "spread_feature_group_ab",
        "status": "complete",
        "cube_schema": cube_manifest.get("schema"),
        "cube_source": cube_manifest.get("source"),
        "information_boundary": cube_manifest.get("information_boundary"),
        "start": args.start,
        "end": args.end,
        "days": len(target_days),
        "training_days": args.training_days,
        "groups": requested_groups,
        "model_config": vars(args),
        "runtime_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(out / "manifest.json", manifest)

    overall = summary[summary["split"].eq("overall45")].sort_values(
        ["model_kind", "balanced_direction_accuracy"], ascending=[True, False]
    )
    print(overall[["feature_group", "model_kind", "direction_accuracy", "positive_accuracy", "negative_accuracy", "balanced_direction_accuracy", "mae", "spread_smape_pct"]].to_string(index=False))
    print(json.dumps({"runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
