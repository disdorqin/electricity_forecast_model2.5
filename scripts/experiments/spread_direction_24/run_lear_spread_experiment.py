"""Causal hourly LEAR and shared-trunk baselines for signed spread forecasting.

This is an experiment-only runner.  It never writes production ledgers or
model directories.  The LEAR variants use a LASSO-regularized autoregression
with target-day forecast grid features and cutoff-safe historical spread
features.  ``lear_shared_lasso`` fits one shared model across all 24 hourly
slots with hour encodings; ``lear_segmented_lasso`` fits the existing three
8-slot blocks.  ``shared_trunk_mlp`` is a small shared nonlinear challenger,
not a production model.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_masked_spread_experiment import (
    prepare_source_cache,
)
from scripts.experiments.spread_direction_24.spread_metrics import smape_percent
from utils.resolution import HOURLY


MODEL_NAMES = ("lear_shared_lasso", "lear_segmented_lasso", "shared_trunk_mlp")
SEGMENTS = (("1_8", 1, 8), ("9_16", 9, 16), ("17_24", 17, 24))
DA_COL = "日前电价"
RT_COL = "实时电价"
SPREAD_COL = "价差"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _daily_map(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for day, group in raw.groupby("_business_day", sort=True):
        ordered = group.sort_values("_business_period").copy()
        if len(ordered) != HOURLY.slots_per_day or ordered["_business_period"].nunique() != HOURLY.slots_per_day:
            continue
        result[str(day)] = ordered.set_index("_business_period")
    return result


def _date_list(day_map: dict[str, pd.DataFrame], start: str, end: str) -> list[str]:
    dates = sorted(day_map)
    return [d for d in dates if start <= d <= end]


def _safe_spread_features(
    raw: pd.DataFrame,
    day_map: dict[str, pd.DataFrame],
    target_day: str,
    forecast_cols: list[str],
    history_by_period: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build one day's features without reading target-day actual prices."""

    target = day_map[target_day]
    target_ts = pd.Timestamp(target_day)
    cutoff = target_ts - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    rows: list[dict] = []
    source_max: list[pd.Timestamp] = []

    for period in range(1, HOURLY.slots_per_day + 1):
        target_row = target.loc[period]
        d1 = (target_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        d2 = (target_ts - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        d7 = (target_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        d1_row = day_map.get(d1, pd.DataFrame()).loc[period] if d1 in day_map else None
        d2_row = day_map.get(d2, pd.DataFrame()).loc[period] if d2 in day_map else None
        d7_row = day_map.get(d7, pd.DataFrame()).loc[period] if d7 in day_map else None

        d1_ts = pd.Timestamp(d1) + pd.Timedelta(hours=period - 1)
        # Business-period timestamps are resolved from the canonical rows,
        # rather than assuming a wall-clock offset at the midnight boundary.
        if d1_row is not None:
            d1_ts = pd.Timestamp(d1_row["时刻"])
        d2_ts = pd.Timestamp(d2_row["时刻"]) if d2_row is not None else pd.NaT
        d7_ts = pd.Timestamp(d7_row["时刻"]) if d7_row is not None else pd.NaT

        d1_visible = period <= 14 and d1_row is not None and pd.Timestamp(d1_row["时刻"]) <= cutoff
        lag1 = float(d1_row[SPREAD_COL]) if d1_visible else math.nan
        lag2 = float(d2_row[SPREAD_COL]) if d2_row is not None else math.nan
        lag7 = float(d7_row[SPREAD_COL]) if d7_row is not None else math.nan
        safe_lag = lag1 if d1_visible and np.isfinite(lag1) else lag2
        safe_source = d1_ts if d1_visible and np.isfinite(lag1) else d2_ts

        historical = history_by_period[period]
        historical = historical[
            (historical["时刻"] <= cutoff)
            & (historical["_business_day"].astype(str) < target_day)
            & historical[SPREAD_COL].notna()
        ].tail(28)
        rolling_median = float(pd.to_numeric(historical[SPREAD_COL], errors="coerce").median()) if not historical.empty else math.nan
        rolling_source = pd.Timestamp(historical["时刻"].max()) if not historical.empty else pd.NaT

        row = {
            "target_day": target_day,
            "hour_business": period,
            "period": HOURLY.infer_period(period),
            "ds": pd.Timestamp(target_row["时刻"]),
            "y_true_spread": float(target_row[SPREAD_COL]),
            "spread_lag1_visible": lag1,
            "spread_lag2": lag2,
            "spread_lag7": lag7,
            "spread_safe_lag": safe_lag,
            "spread_rolling_median": rolling_median,
            "spread_visible_d1": float(d1_visible),
            "spread_source_lag_days": 1.0 if d1_visible and np.isfinite(lag1) else 2.0,
            "hour_sin": math.sin(2 * math.pi * (period - 1) / HOURLY.slots_per_day),
            "hour_cos": math.cos(2 * math.pi * (period - 1) / HOURLY.slots_per_day),
            "dow_sin": math.sin(2 * math.pi * target_ts.dayofweek / 7),
            "dow_cos": math.cos(2 * math.pi * target_ts.dayofweek / 7),
            "month_sin": math.sin(2 * math.pi * (target_ts.month - 1) / 12),
            "month_cos": math.cos(2 * math.pi * (target_ts.month - 1) / 12),
        }
        for col in forecast_cols:
            row[f"fcast::{col}"] = pd.to_numeric(target_row[col], errors="coerce")
        for slot in range(1, HOURLY.slots_per_day + 1):
            row[f"slot_onehot::{slot}"] = float(period == slot)
        rows.append(row)
        source_candidates = [x for x in (safe_source, rolling_source) if pd.notna(x)]
        source_max.append(max(source_candidates) if source_candidates else pd.NaT)

    frame = pd.DataFrame(rows).sort_values("hour_business").reset_index(drop=True)
    if len(frame) != HOURLY.slots_per_day or not np.isfinite(frame["y_true_spread"]).all():
        raise ValueError(f"{target_day}: invalid feature/label rows")
    source_series = pd.Series(source_max, index=frame.index, dtype="datetime64[ns]")
    if source_series.notna().any() and (source_series.dropna() > cutoff).any():
        raise RuntimeError(f"{target_day}: a LEAR feature source exceeds cutoff {cutoff}")
    return frame, source_series


def _model(feature_count: str, *, alpha_grid: np.ndarray, random_state: int, mlp: bool = False):
    if mlp:
        estimator = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=2e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )
    else:
        estimator = LassoCV(alphas=alpha_grid, cv=3, max_iter=20000, n_jobs=1)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def _metrics(group: pd.DataFrame, model_name: str) -> dict:
    true = group["y_true_spread"].to_numpy(float)
    pred = group["y_pred_spread"].to_numpy(float)
    true_sign = np.sign(true)
    pred_sign = np.sign(pred)
    eligible = true_sign != 0
    correct = eligible & (true_sign == pred_sign)
    pos = true_sign > 0
    neg = true_sign < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "model_name": model_name,
        "days": int(group["target_day"].nunique()),
        "n_slots": int(len(group)),
        "n_direction_eligible": int(eligible.sum()),
        "n_positive_actual": int(pos.sum()),
        "n_negative_actual": int(neg.sum()),
        "n_zero_actual": int((true_sign == 0).sum()),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "spread_smape_pct": smape_percent(true, pred),
    }


def run(args) -> dict:
    output_root = Path(args.output_root)
    source = Path(args.data_path)
    cache_root = Path(args.cache_root) if args.cache_root else None
    _, raw, source_info = prepare_source_cache(source, output_root, cache_root)
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw[SPREAD_COL] = pd.to_numeric(raw[SPREAD_COL], errors="coerce")
    forecast_cols = [c for c in raw.columns if str(c).endswith("预测值")]
    if not forecast_cols:
        raise ValueError("no forecast grid columns found")
    day_map = _daily_map(raw)
    dates = _date_list(day_map, args.start, args.end)[:: args.date_step]
    if not dates:
        raise ValueError("no target dates")
    feature_cache: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    history_by_period = {
        period: raw[raw["_business_period"].eq(period)].sort_values("时刻").copy()
        for period in range(1, HOURLY.slots_per_day + 1)
    }
    feature_start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=args.training_days + 10)).strftime("%Y-%m-%d")
    for day in sorted(day_map):
        if feature_start <= day <= dates[-1]:
            feature_cache[day] = _safe_spread_features(
                raw, day_map, day, forecast_cols, history_by_period
            )

    alpha_grid = np.asarray(args.alpha_grid, dtype=float)
    all_rows: list[pd.DataFrame] = []
    daily_metrics: list[dict] = []
    started_all = time.perf_counter()
    for target_day in dates:
        day_start = time.perf_counter()
        target_frame, source_max = feature_cache[target_day]
        earlier = [d for d in sorted(feature_cache) if d < target_day]
        train_days = earlier[-args.training_days:]
        if len(train_days) < args.min_training_days:
            raise ValueError(f"{target_day}: only {len(train_days)} training days")
        train = pd.concat([feature_cache[d][0] for d in train_days], ignore_index=True)
        feature_cols = [c for c in target_frame.columns if c not in {"target_day", "hour_business", "period", "ds", "y_true_spread"}]
        # In the late segment, D-1 same-slot spread is structurally unavailable.
        # Drop such all-missing columns before SimpleImputer rather than letting
        # sklearn silently discard them with a warning.
        feature_cols = [c for c in feature_cols if train[c].notna().any()]
        X_train = train[feature_cols]
        y_train = train["y_true_spread"].to_numpy(float)
        X_target = target_frame[feature_cols]
        predictions: dict[str, np.ndarray] = {}

        shared = _model("shared", alpha_grid=alpha_grid, random_state=args.seed)
        shared.fit(X_train, y_train)
        predictions["lear_shared_lasso"] = shared.predict(X_target)

        segmented = np.full(HOURLY.slots_per_day, np.nan, dtype=float)
        for name, start, end in SEGMENTS:
            mask = train["period"].eq(name)
            segment_feature_cols = [c for c in feature_cols if train.loc[mask, c].notna().any()]
            model = _model(name, alpha_grid=alpha_grid, random_state=args.seed)
            model.fit(X_train.loc[mask, segment_feature_cols], y_train[mask.to_numpy()])
            target_mask = target_frame["period"].eq(name)
            segmented[target_mask.to_numpy()] = model.predict(X_target.loc[target_mask, segment_feature_cols])
        predictions["lear_segmented_lasso"] = segmented

        if not args.skip_shared_mlp:
            trunk = _model("shared_trunk", alpha_grid=alpha_grid, random_state=args.seed, mlp=True)
            trunk.fit(X_train, y_train)
            predictions["shared_trunk_mlp"] = trunk.predict(X_target)

        for model_name, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError(f"{target_day}/{model_name}: non-finite prediction")
            out = target_frame[["target_day", "ds", "hour_business", "period", "y_true_spread"]].copy()
            out["model_name"] = model_name
            out["prediction_mode"] = "causal_lear_shared_or_segmented"
            out["y_pred_spread"] = pred.astype(float)
            out["segment_model_id"] = out["model_name"] + "_" + out["period"].astype(str)
            out["segment_training"] = True
            out["source_max_ds"] = source_max.to_numpy()
            out["information_cutoff"] = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
            true_sign = np.sign(out["y_true_spread"].to_numpy(float))
            pred_sign = np.sign(pred)
            out["actual_direction"] = true_sign.astype(int)
            out["predicted_direction"] = pred_sign.astype(int)
            out["direction_eligible"] = true_sign != 0
            out["direction_correct"] = out["direction_eligible"] & (true_sign == pred_sign)
            diff = out["y_pred_spread"].to_numpy(float) - out["y_true_spread"].to_numpy(float)
            denom = np.abs(out["y_pred_spread"].to_numpy(float)) + np.abs(out["y_true_spread"].to_numpy(float))
            out["spread_smape_contribution_pct"] = np.divide(2 * np.abs(diff), denom, out=np.zeros_like(denom), where=denom != 0) * 100
            all_rows.append(out)
            daily_metrics.append(_metrics(out, model_name) | {"target_day": target_day, "elapsed_seconds": time.perf_counter() - day_start})

    ledger = pd.concat(all_rows, ignore_index=True)
    _atomic_parquet(output_root / "ledger" / "evaluation_ledger.parquet", ledger)
    _atomic_parquet(output_root / "ledger" / "prediction_ledger.parquet", ledger.drop(columns=["y_true_spread"]))
    summary = pd.DataFrame([_metrics(group, name) for name, group in ledger.groupby("model_name")]).sort_values(
        ["balanced_direction_accuracy", "spread_smape_pct"], ascending=[False, True]
    )
    _atomic_csv(output_root / "summary" / "model_summary.csv", summary)
    _atomic_csv(output_root / "summary" / "daily_model_metrics.csv", pd.DataFrame(daily_metrics))
    active_models = [m for m in MODEL_NAMES if not (args.skip_shared_mlp and m == "shared_trunk_mlp")]
    manifest = {
        "pipeline": "spread_direction_24_lear_experiment",
        "status": "complete",
        "resolution": HOURLY.label,
        "hourly_only": True,
        "target_definition": "realtime_actual - dayahead_actual",
        "direction_definition": "same strict sign; actual zero excluded; predicted zero is wrong",
        "models": active_models,
        "start": dates[0],
        "end": dates[-1],
        "days": len(dates),
        "rows": int(len(ledger)),
        "training_days": args.training_days,
        "min_training_days": args.min_training_days,
        "alpha_grid": alpha_grid.tolist(),
        "seed": args.seed,
        "source": source_info,
        "information_boundary": {
            "forecast_origin": "D-1 14:00",
            "target_day_dayahead_realtime_spread": "masked from features; label only",
            "target_day_grid": "forecast columns only",
            "d1_spread": "p1-p14 visible; p15-p24 replaced by D-2 same-slot spread",
            "source_max_ds": "must be <= cutoff for every feature row",
        },
        "shared_trunk": {
            "lear_shared_lasso": "one LASSO model over all slots with slot encodings",
            "lear_segmented_lasso": "independent LASSO models for 1_8/9_16/17_24",
            "shared_trunk_mlp": "small one-shared-trunk MLP challenger; experiment only",
        },
        "elapsed_seconds": time.perf_counter() - started_all,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_root / "range_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--cache-root", default="outputs/experiments/01_spread_24/invalid_leakage/legacy_cutoff_leakage_spread_direction_24_shared_cache_opt_20260820")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--date-step", type=int, default=1)
    parser.add_argument("--training-days", type=int, default=365)
    parser.add_argument("--min-training-days", type=int, default=60)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-shared-mlp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=str))
