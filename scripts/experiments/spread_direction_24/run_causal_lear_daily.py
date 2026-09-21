"""Classic LEAR-style daily sparse regression adapted to the D-1 14:00 spread contract.

Unlike the earlier row-wise shared LASSO probe, this runner follows the classic
EPF LEAR sample organization: one row per business day, full lagged daily
curves + target-day forecast exogenous curves, and 24 independent sparse
regressions.  D-1 spread p15-p24 is never read; it is represented by the safe
rolling proxy already used by the spread experiments.

Reference design: epftoolbox LEAR (Lago et al.), adapted rather than copied.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_lear_spread_experiment import (
    SPREAD_COL,
    _daily_map,
    _date_list,
    _metrics,
    _safe_spread_features,
    prepare_source_cache,
)
from utils.resolution import HOURLY

KEY_FORECAST_COLS = (
    "风电总加预测值",
    "光伏总加预测值",
    "新能源总加预测值",
    "竞价空间预测值",
    "直调负荷预测值",
)
ALPHAS = np.asarray([0.003, 0.01, 0.03, 0.1, 0.3], dtype=float)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


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


def _day_curve(day_map: dict[str, pd.DataFrame], day: str, col: str) -> np.ndarray:
    frame = day_map[day]
    arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
    if len(arr) != HOURLY.slots_per_day:
        raise ValueError(f"{day}/{col}: incomplete curve")
    return arr


def _build_daily_feature(
    target_day: str,
    *,
    day_map: dict[str, pd.DataFrame],
    safe_frame: pd.DataFrame,
    use_exog_history: bool,
    proxy_dropout: float,
) -> tuple[np.ndarray, np.ndarray, list[int], pd.Timestamp]:
    target = pd.Timestamp(target_day)
    cutoff = target - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    parts: list[np.ndarray] = []
    proxy_idx: list[int] = []

    # LEAR-like spread lags D-1, D-2, D-3, D-7. D-1 tail is strictly proxied.
    observed = pd.to_numeric(safe_frame["spread_lag1_visible"], errors="coerce").to_numpy(float)
    rolling = pd.to_numeric(safe_frame["spread_rolling_median"], errors="coerce").to_numpy(float)
    lag2 = pd.to_numeric(safe_frame["spread_lag2"], errors="coerce").to_numpy(float)
    d1_safe = observed.copy()
    d1_safe[14:] = rolling[14:]
    d1_safe = np.where(np.isfinite(d1_safe), d1_safe, lag2)
    start = 0
    parts.append(d1_safe)
    proxy_idx = list(range(start + 14, start + 24))
    for lag in (2, 3, 7):
        day = (target - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        parts.append(_day_curve(day_map, day, SPREAD_COL))

    # Target-day forecast exogenous curves are known at inference.
    for col in KEY_FORECAST_COLS:
        parts.append(pd.to_numeric(safe_frame[f"fcast::{col}"], errors="coerce").to_numpy(float))

    # Canonical LEAR also uses exogenous D-1 and D-7 curves. We keep forecast
    # versions only, avoiding any D-1 actual-grid ambiguity.
    if use_exog_history:
        for lag in (1, 7):
            day = (target - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            for col in KEY_FORECAST_COLS:
                parts.append(_day_curve(day_map, day, col))

    # Weekday dummies, matching the classic LEAR spirit.
    dow = np.zeros(7, dtype=float)
    dow[target.dayofweek] = 1.0
    parts.append(dow)
    y = pd.to_numeric(safe_frame["y_true_spread"], errors="coerce").to_numpy(float)
    return np.concatenate(parts), y, proxy_idx, cutoff


def _apply_dropout(Xs: np.ndarray, proxy_idx: list[int], rate: float, seed: int) -> np.ndarray:
    if rate <= 0:
        return Xs
    out = Xs.copy()
    rng = np.random.default_rng(seed)
    cols = np.asarray(proxy_idx, dtype=int)
    mask = rng.random((len(out), len(cols))) < rate
    block = out[:, cols]
    block[mask] = 0.0
    out[:, cols] = block
    return out


def _fit_predict(X: np.ndarray, y: np.ndarray, xt: np.ndarray, proxy_idx: list[int], dropout: float, seed: int):
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    xti = imp.transform(xt[None, :])
    xs = StandardScaler().fit(Xi)
    Xs = xs.transform(Xi)
    xts = xs.transform(xti)
    Xs = _apply_dropout(Xs, proxy_idx, dropout, seed)
    ys = StandardScaler().fit(y)
    yt = ys.transform(y)
    cv = TimeSeriesSplit(n_splits=3)
    pred_scaled = np.zeros((1, HOURLY.slots_per_day), dtype=float)
    selected = []
    nonzero = []
    for h in range(HOURLY.slots_per_day):
        model = LassoCV(
            alphas=ALPHAS, cv=cv, max_iter=5000, tol=1e-4,
            fit_intercept=True, n_jobs=1, selection="cyclic",
        )
        model.fit(Xs, yt[:, h])
        pred_scaled[0, h] = model.predict(xts)[0]
        selected.append(float(model.alpha_))
        nonzero.append(int(np.count_nonzero(np.abs(model.coef_) > 1e-10)))
    return ys.inverse_transform(pred_scaled)[0], selected, nonzero


def run(args) -> dict:
    out_root = Path(args.output_root)
    _, raw, source_info = prepare_source_cache(Path(args.data_path), out_root, Path(args.cache_root) if args.cache_root else None)
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw[SPREAD_COL] = pd.to_numeric(raw[SPREAD_COL], errors="coerce")
    missing = [c for c in KEY_FORECAST_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing forecast columns={missing}")
    day_map = _daily_map(raw)
    dates = _date_list(day_map, args.start, args.end)
    history_by_period = {p: raw[raw["_business_period"].eq(p)].sort_values("时刻").copy() for p in range(1, 25)}
    cache_start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=args.training_days + 20)).strftime("%Y-%m-%d")
    safe_cache = {
        day: _safe_spread_features(raw, day_map, day, list(KEY_FORECAST_COLS), history_by_period)[0]
        for day in sorted(day_map) if cache_start <= day <= dates[-1]
    }
    variants = [
        ("lear_daily_target_exog", False, 0.0),
        ("lear_daily_target_hist_exog", True, 0.0),
        ("lear_daily_target_hist_exog_drop25", True, 0.25),
    ]
    if args.variants:
        requested = {x.strip() for x in args.variants.split(",") if x.strip()}
        variants = [v for v in variants if v[0] in requested]
    all_rows, audits = [], []
    started = time.perf_counter()
    for name, use_hist_exog, drop in variants:
        vectors = {}
        for day in sorted(safe_cache):
            try:
                vectors[day] = _build_daily_feature(
                    day, day_map=day_map, safe_frame=safe_cache[day],
                    use_exog_history=use_hist_exog, proxy_dropout=drop,
                )
            except (KeyError, ValueError):
                continue
        for target_day in dates:
            train_days = [d for d in sorted(vectors) if d < target_day][-args.training_days:]
            if len(train_days) < args.min_training_days:
                continue
            X = np.stack([vectors[d][0] for d in train_days])
            y = np.stack([vectors[d][1] for d in train_days])
            xt, yt, proxy_idx, cutoff = vectors[target_day]
            t0 = time.perf_counter()
            pred, alphas, nonzero = _fit_predict(
                X, y, xt, proxy_idx, drop, args.seed + pd.Timestamp(target_day).dayofyear
            )
            frame = pd.DataFrame({
                "target_day": target_day,
                "hour_business": np.arange(1, 25),
                "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                "y_true_spread": yt,
                "y_pred_spread": pred,
                "model_name": name,
                "information_cutoff": cutoff,
            })
            all_rows.append(frame)
            audits.append({
                "target_day": target_day, "model_name": name, "training_days": len(train_days),
                "feature_count": X.shape[1], "median_alpha": float(np.median(alphas)),
                "median_nonzero_features": float(np.median(nonzero)), "fit_seconds": time.perf_counter() - t0,
            })
    ledger = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame([_metrics(g, name) for name, g in ledger.groupby("model_name", sort=False)])
    summary = summary.sort_values(["balanced_direction_accuracy", "direction_accuracy"], ascending=[False, False])
    _atomic_parquet(out_root / "ledger" / "evaluation_ledger.parquet", ledger)
    _atomic_csv(out_root / "summary" / "model_summary.csv", summary)
    _atomic_csv(out_root / "summary" / "training_audit.csv", pd.DataFrame(audits))
    manifest = {
        "pipeline": "spread_direction_24_causal_lear_daily",
        "status": "complete",
        "reference": "epftoolbox LEAR daily ARX/LASSO sample organization, adapted to cutoff-safe spread",
        "reference_url": "https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/models/_lear.py",
        "start": dates[0], "end": dates[-1], "days": len(dates),
        "training_days": args.training_days,
        "key_forecast_columns": list(KEY_FORECAST_COLS),
        "alphas": ALPHAS.tolist(),
        "information_boundary": {"forecast_origin": "D-1 14:00", "d1_spread_tail": "rolling proxy, never truth", "target_exog": "forecast only", "historical_exog": "forecast only"},
        "historical_preflight_override": "24-point freshness failed; retrospective range is earlier and integrity/leakage checks passed",
        "source": source_info,
        "runtime": {"platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started},
        "top_models": summary.to_dict(orient="records"),
    }
    _atomic_json(out_root / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    p.add_argument("--cache-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_shared_cache_opt_20260820")
    p.add_argument("--output-root", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--training-days", type=int, default=365)
    p.add_argument("--min-training-days", type=int, default=180)
    p.add_argument("--variants", default="")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
