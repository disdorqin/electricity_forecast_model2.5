"""Run existing spread models with the winning rolling-fill input view.

This is the post-structure migration experiment only.  It keeps SGDFNet and
LightGBM unchanged, but gives them the same causal input used by the winning
TimeMixer structure: D-1 p1-p14 spread plus same-slot rolling median for
p15-p24, with D-2 same-slot fallback.  It never touches production model
directories or delivery stages.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_masked_spread_experiment import (  # noqa: E402
    actual_for_day,
    prepare_source_cache,
    predict_lightgbm,
    predict_sgdfnet,
    score_predictions,
)
from scripts.experiments.spread_direction_24.spread_contract import (  # noqa: E402
    DA_ALIASES,
    RT_ALIASES,
    first_existing,
    materialize_asof_input,
)
from utils.resolution import HOURLY  # noqa: E402


MODEL_RUNNERS = {"sgdfnet": predict_sgdfnet, "lightgbm": predict_lightgbm}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _date_list(raw: pd.DataFrame, start: str, end: str) -> list[str]:
    days = sorted(raw["_business_day"].astype(str).unique())
    return [d for d in days if start <= d <= end]


def _rolling_fill_input(raw: pd.DataFrame, target_day: str, output_path: Path, da_col: str, rt_col: str) -> dict:
    audit = materialize_asof_input(
        raw,
        target_day,
        da_col,
        rt_col,
        output_path,
        input_scheme="safe_mixed_lag",
        cutoff_hour=14,
        business_day_fn=HOURLY.business_day_from_timestamp,
        business_period_fn=HOURLY.business_period_from_timestamp,
    )
    view = pd.read_parquet(output_path)
    view["时刻"] = pd.to_datetime(view["时刻"], errors="raise")
    view["_business_day"] = view["时刻"].map(HOURLY.business_day_from_timestamp).astype(str)
    view["_business_period"] = view["时刻"].map(HOURLY.business_period_from_timestamp).astype(int)
    cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    d1 = (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw_work = raw.copy()
    raw_work["时刻"] = pd.to_datetime(raw_work["时刻"], errors="raise")
    values: list[float] = []
    sources: list[pd.Timestamp] = []
    for period in range(15, 25):
        hist = raw_work[
            raw_work["时刻"].le(cutoff)
            & raw_work["_business_period"].eq(period)
            & raw_work["价差"].notna()
        ].sort_values("时刻").tail(28)
        value = float(pd.to_numeric(hist["价差"], errors="coerce").median()) if not hist.empty else math.nan
        source = pd.Timestamp(hist["时刻"].max()) if not hist.empty else pd.NaT
        mask = view["_business_day"].eq(d1) & view["_business_period"].eq(period)
        if mask.sum() != 1:
            raise ValueError(f"{target_day}: expected one D-1 row for p{period}")
        view.loc[mask, "价差"] = value
        view.loc[mask, "价差来源滞后日"] = ((pd.Timestamp(target_day) - source).days if pd.notna(source) else np.nan)
        values.append(value)
        sources.append(source)
    if not np.isfinite(values).all():
        raise ValueError(f"{target_day}: rolling fill contains missing values")
    view = view.drop(columns=["_business_day", "_business_period"], errors="ignore")
    tmp = output_path.with_name(output_path.name + ".rolling.tmp")
    view.to_parquet(tmp, index=False)
    tmp.replace(output_path)
    audit.update({
        "input_scheme": "safe_rolling_median",
        "rolling_fill_slots": 10,
        "rolling_fill_source_max": max(sources),
        "rolling_fill_cutoff": cutoff,
    })
    return audit


def _split_name(index: int) -> str:
    if index < 30:
        return "development_30d"
    if index < 45:
        return "confirmation_15d"
    return "holdout_15d"


def _args_for_runner(args):
    return SimpleNamespace(
        training_months=args.training_months,
        val_ratio=args.val_ratio,
        realtime_cutoff_hour=14,
        seed=args.seed,
        deterministic=True,
        device="cpu",
    )


def _aggregate(frame: pd.DataFrame, model: str, split: str) -> dict:
    true = frame["y_true_spread"].to_numpy(float)
    pred = frame["y_pred_spread"].to_numpy(float)
    ts, ps = np.sign(true), np.sign(pred)
    eligible = ts != 0
    correct = eligible & (ts == ps)
    pos, neg = ts > 0, ts < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "model_name": model,
        "split": split,
        "days": int(frame["target_day"].nunique()),
        "n_slots": int(len(frame)),
        "direction_accuracy": float(correct[eligible].mean()),
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
        "mae": float(np.abs(pred - true).mean()),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "spread_smape_pct": float(np.mean(200.0 * np.abs(pred - true) / np.where(np.abs(pred) + np.abs(true) == 0, 1.0, np.abs(pred) + np.abs(true)))),
    }


def run(args) -> dict:
    out_root = Path(args.output_root)
    source = Path(args.data_path)
    _, raw, source_info = prepare_source_cache(source, out_root, Path(args.cache_root) if args.cache_root else None)
    da_col = first_existing(raw, DA_ALIASES, "dayahead actual")
    rt_col = first_existing(raw, RT_ALIASES, "realtime actual")
    dates = _date_list(raw, args.start, args.end)
    if len(dates) != 60 and not args.allow_short:
        raise ValueError("migration experiment expects exactly 60 target days")
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = sorted(set(models) - set(MODEL_RUNNERS))
    if unknown:
        raise ValueError(f"unknown models={unknown}; allowed={sorted(MODEL_RUNNERS)}")
    runner_args = _args_for_runner(args)
    rows: list[pd.DataFrame] = []
    daily: list[dict] = []
    audit: list[dict] = []
    started = time.perf_counter()
    for idx, target_day in enumerate(dates):
        input_path = out_root / "inputs" / f"{target_day}.parquet"
        audit.append(_rolling_fill_input(raw, target_day, input_path, da_col, rt_col))
        actual = actual_for_day(raw, target_day, da_col, rt_col)
        for model in models:
            model_root = out_root / "model_runs" / model / target_day
            prediction = MODEL_RUNNERS[model](input_path, target_day, model_root, runner_args)
            joined, metrics = score_predictions(prediction, actual)
            joined["input_fill_scheme"] = "d1_p1_p14_plus_rolling_median_p15_p24_fallback_lag48"
            joined["information_cutoff"] = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
            joined["training_months"] = args.training_months
            rows.append(joined)
            daily.append(metrics | {"split": _split_name(idx)})
    ledger = pd.concat(rows, ignore_index=True)
    _atomic_parquet(out_root / "ledger" / "evaluation_ledger.parquet", ledger)
    _atomic_csv(out_root / "ledger" / "evaluation_ledger.csv", ledger)
    daily_df = pd.DataFrame(daily)
    _atomic_csv(out_root / "summary" / "daily_metrics.csv", daily_df)
    split_rows = []
    for (model, split), group in ledger.assign(split=ledger["target_day"].map({d: _split_name(i) for i, d in enumerate(dates)})).groupby(["model_name", "split"]):
        split_rows.append(_aggregate(group, model, split))
    split_df = pd.DataFrame(split_rows)
    _atomic_csv(out_root / "summary" / "metrics_by_split.csv", split_df)
    _atomic_csv(out_root / "summary" / "model_summary.csv", split_df[split_df["split"].eq("holdout_15d")].copy())
    _atomic_json(out_root / "input_audit.json", {"days": audit})
    manifest = {
        "pipeline": "spread_direction_24_model_migration_experiment",
        "status": "complete",
        "resolution": "hourly",
        "hourly_only": True,
        "models": models,
        "start": dates[0], "end": dates[-1], "days": len(dates),
        "training_months": args.training_months,
        "evaluation_splits": {"development": 30, "confirmation": 15, "holdout": 15},
        "input_fill_scheme": "d1_p1_p14_plus_rolling_median_p15_p24_fallback_lag48",
        "information_boundary": {"forecast_origin": "D-1 14:00", "target_day_actual_labels": "label only", "source_max_ds": "<= cutoff"},
        "source": source_info,
        "runtime": {"python": sys.version, "platform": platform.platform(), "seed": args.seed},
        "rows": int(len(ledger)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(out_root / "range_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    p.add_argument("--cache-root", default="outputs/experiments/01_spread_24/invalid_leakage/legacy_cutoff_leakage_spread_direction_24_shared_cache_opt_20260820")
    p.add_argument("--output-root", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--allow-short", action="store_true", help="smoke-test only; does not represent the registered 60-day evaluation")
    p.add_argument("--models", default="sgdfnet,lightgbm")
    p.add_argument("--training-months", type=int, default=12)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
