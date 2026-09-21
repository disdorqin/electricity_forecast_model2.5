"""Fast causal linear probes for hourly spread direction forecasting.

Experiment-only.  Uses completed historical spread days, D-1 14:00-visible
context, safe proxy channels, and target-day forecast-grid covariates.  Fits
cheap Ridge regression and RidgeClassifier probes in a strict walk-forward
loop.  No production ledgers/models are touched.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeClassifier
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

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class ProbeConfig:
    name: str
    history_days: int
    use_context: bool
    use_proxy: bool
    proxy_dropout: float = 0.0


CONFIGS = (
    ProbeConfig("r0_hist7_future", 7, False, False, 0.0),
    ProbeConfig("r1_hist7_context_future", 7, True, False, 0.0),
    ProbeConfig("r2_hist7_context_proxy_future", 7, True, True, 0.0),
    ProbeConfig("r3_hist7_context_proxydrop25_future", 7, True, True, 0.25),
    ProbeConfig("r4_hist14_context_proxydrop25_future", 14, True, True, 0.25),
)


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


def _completed_history(day_map: dict[str, pd.DataFrame], target_day: str, history_days: int) -> tuple[np.ndarray, pd.Timestamp]:
    target = pd.Timestamp(target_day)
    values = []
    latest = pd.NaT
    # Strictly completed days only: D-2, D-3, ...
    for lag in range(history_days + 1, 1, -1):
        day = (target - pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
        if day not in day_map:
            raise KeyError(day)
        frame = day_map[day]
        spread = pd.to_numeric(frame[SPREAD_COL], errors="coerce").to_numpy(float)
        if len(spread) != HOURLY.slots_per_day:
            raise ValueError(f"{day}: incomplete history")
        values.extend(spread.tolist())
        ds_max = pd.to_datetime(frame["时刻"], errors="coerce").max()
        latest = ds_max if pd.isna(latest) else max(latest, ds_max)
    return np.asarray(values, dtype=float), pd.Timestamp(latest)


def _build_vector(
    target_day: str,
    *,
    cfg: ProbeConfig,
    day_map: dict[str, pd.DataFrame],
    safe_frame: pd.DataFrame,
    forecast_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], pd.Timestamp]:
    hist, hist_max = _completed_history(day_map, target_day, cfg.history_days)
    target_ts = pd.Timestamp(target_day)
    cutoff = target_ts - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    parts: list[np.ndarray] = [hist]
    proxy_indices: list[int] = []

    if cfg.use_context:
        context = np.zeros(HOURLY.slots_per_day, dtype=float)
        visible = pd.to_numeric(safe_frame["spread_lag1_visible"], errors="coerce").to_numpy(float)
        context[:14] = visible[:14]
        parts.append(context)

    if cfg.use_proxy:
        observed = pd.to_numeric(safe_frame["spread_lag1_visible"], errors="coerce").to_numpy(float)
        rolling = pd.to_numeric(safe_frame["spread_rolling_median"], errors="coerce").to_numpy(float)
        lag2 = pd.to_numeric(safe_frame["spread_lag2"], errors="coerce").to_numpy(float)
        lag7 = pd.to_numeric(safe_frame["spread_lag7"], errors="coerce").to_numpy(float)
        safe_input = observed.copy()
        safe_input[14:] = rolling[14:]
        safe_input = np.where(np.isfinite(safe_input), safe_input, lag2)
        start = sum(len(x) for x in parts)
        parts.extend([safe_input, rolling, lag2, lag7])
        proxy_indices = list(range(start + 14, start + HOURLY.slots_per_day))

    future = []
    for col in forecast_cols:
        values = pd.to_numeric(safe_frame[f"fcast::{col}"], errors="coerce").to_numpy(float)
        future.extend(values.tolist())
    parts.append(np.asarray(future, dtype=float))

    # Calendar terms are target-known and cheap.
    dow = target_ts.dayofweek
    month = target_ts.month - 1
    parts.append(np.asarray([
        math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
        math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
    ], dtype=float))

    y = pd.to_numeric(safe_frame["y_true_spread"], errors="coerce").to_numpy(float)
    source_max = max(hist_max, cutoff if cfg.use_context else hist_max)
    return np.concatenate(parts), y, proxy_indices, source_max


def _preprocess_fit(X: np.ndarray):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return imp, scaler, X_scaled


def _transform(imp, scaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(imp.transform(X))


def _apply_proxy_dropout(X: np.ndarray, proxy_indices: list[int], rate: float, seed: int) -> np.ndarray:
    if rate <= 0 or not proxy_indices:
        return X
    out = X.copy()
    rng = np.random.default_rng(seed)
    mask = rng.random((len(out), len(proxy_indices))) < rate
    cols = np.asarray(proxy_indices, dtype=int)
    block = out[:, cols]
    block[mask] = 0.0  # standardized zero == training mean, matching the prior neural probe semantics
    out[:, cols] = block
    return out


def _dir_stats(y_true: np.ndarray, y_pred_sign: np.ndarray) -> tuple[float, float, float, float]:
    true_sign = np.sign(y_true)
    eligible = true_sign != 0
    pred_sign = np.sign(y_pred_sign)
    correct = eligible & (true_sign == pred_sign)
    pos, neg = true_sign > 0, true_sign < 0
    acc = float(correct[eligible].mean()) if eligible.any() else math.nan
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    bal = float(np.nanmean([pos_acc, neg_acc]))
    return acc, pos_acc, neg_acc, bal


def _choose_alpha_reg(X: np.ndarray, y: np.ndarray, proxy_idx: list[int], dropout: float, seed: int) -> float:
    split = max(30, int(len(X) * 0.8))
    split = min(split, len(X) - 1)
    imp, scaler, Xtr = _preprocess_fit(X[:split])
    Xtr = _apply_proxy_dropout(Xtr, proxy_idx, dropout, seed)
    Xv = _transform(imp, scaler, X[split:])
    best = None
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(Xtr, y[:split])
        pred = model.predict(Xv)
        _, _, _, bal = _dir_stats(y[split:], pred)
        mae = float(np.mean(np.abs(pred - y[split:])))
        key = (bal, -mae)
        if best is None or key > best[0]:
            best = (key, alpha)
    return float(best[1])


def _choose_alpha_cls(X: np.ndarray, y: np.ndarray, proxy_idx: list[int], dropout: float, seed: int) -> float:
    split = max(30, int(len(X) * 0.8))
    split = min(split, len(X) - 1)
    imp, scaler, Xtr = _preprocess_fit(X[:split])
    Xtr = _apply_proxy_dropout(Xtr, proxy_idx, dropout, seed)
    Xv = _transform(imp, scaler, X[split:])
    best = None
    y_sign = np.sign(y)
    for alpha in ALPHAS:
        preds = np.zeros_like(y[split:])
        for h in range(HOURLY.slots_per_day):
            yt = y_sign[:split, h]
            if len(np.unique(yt)) < 2:
                preds[:, h] = yt[-1]
                continue
            clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
            clf.fit(Xtr, yt)
            preds[:, h] = clf.predict(Xv)
        acc, pos, neg, bal = _dir_stats(y[split:], preds)
        key = (bal, acc)
        if best is None or key > best[0]:
            best = (key, alpha)
    return float(best[1])


def _fit_predict_reg(X: np.ndarray, y: np.ndarray, xt: np.ndarray, proxy_idx: list[int], cfg: ProbeConfig, seed: int) -> tuple[np.ndarray, float]:
    alpha = _choose_alpha_reg(X, y, proxy_idx, cfg.proxy_dropout, seed)
    imp, scaler, Xs = _preprocess_fit(X)
    Xs = _apply_proxy_dropout(Xs, proxy_idx, cfg.proxy_dropout, seed + 17)
    xts = _transform(imp, scaler, xt[None, :])
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)
    return model.predict(xts)[0], alpha


def _fit_predict_cls(X: np.ndarray, y: np.ndarray, xt: np.ndarray, proxy_idx: list[int], cfg: ProbeConfig, seed: int) -> tuple[np.ndarray, float]:
    alpha = _choose_alpha_cls(X, y, proxy_idx, cfg.proxy_dropout, seed)
    imp, scaler, Xs = _preprocess_fit(X)
    Xs = _apply_proxy_dropout(Xs, proxy_idx, cfg.proxy_dropout, seed + 31)
    xts = _transform(imp, scaler, xt[None, :])
    y_sign = np.sign(y)
    pred = np.zeros(HOURLY.slots_per_day, dtype=float)
    for h in range(HOURLY.slots_per_day):
        yt = y_sign[:, h]
        if len(np.unique(yt)) < 2:
            pred[h] = yt[-1]
            continue
        clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
        clf.fit(Xs, yt)
        pred[h] = clf.predict(xts)[0]
    return pred, alpha


def run(args) -> dict:
    out_root = Path(args.output_root)
    _, raw, source_info = prepare_source_cache(Path(args.data_path), out_root, Path(args.cache_root) if args.cache_root else None)
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw[SPREAD_COL] = pd.to_numeric(raw[SPREAD_COL], errors="coerce")
    forecast_cols = [c for c in raw.columns if str(c).endswith("预测值")]
    day_map = _daily_map(raw)
    dates = _date_list(day_map, args.start, args.end)
    if not dates:
        raise ValueError("no target dates")
    history_by_period = {p: raw[raw["_business_period"].eq(p)].sort_values("时刻").copy() for p in range(1, HOURLY.slots_per_day + 1)}
    max_hist = max(c.history_days for c in CONFIGS)
    cache_start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=args.training_days + max_hist + 20)).strftime("%Y-%m-%d")
    safe_cache = {}
    for day in sorted(day_map):
        if cache_start <= day <= dates[-1]:
            safe_cache[day] = _safe_spread_features(raw, day_map, day, forecast_cols, history_by_period)[0]

    rows = []
    audits = []
    started = time.perf_counter()
    active_names = {x.strip() for x in args.configs.split(",") if x.strip()} if args.configs else {c.name for c in CONFIGS}
    active_configs = [c for c in CONFIGS if c.name in active_names]
    unknown = active_names - {c.name for c in CONFIGS}
    if unknown:
        raise ValueError(f"unknown configs: {sorted(unknown)}")
    for cfg in active_configs:
        vector_cache = {}
        for day in sorted(safe_cache):
            try:
                vector_cache[day] = _build_vector(day, cfg=cfg, day_map=day_map, safe_frame=safe_cache[day], forecast_cols=forecast_cols)
            except (KeyError, ValueError):
                continue
        for target_day in dates:
            earlier = [d for d in sorted(vector_cache) if d < target_day]
            train_days = earlier[-args.training_days:]
            if len(train_days) < args.min_training_days:
                continue
            X = np.stack([vector_cache[d][0] for d in train_days])
            y = np.stack([vector_cache[d][1] for d in train_days])
            xt, yt, proxy_idx, source_max = vector_cache[target_day]
            cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
            if source_max > cutoff:
                raise RuntimeError(f"{target_day}/{cfg.name}: source_max {source_max} > cutoff {cutoff}")
            seed = args.seed + pd.Timestamp(target_day).dayofyear
            reg_pred, reg_alpha = _fit_predict_reg(X, y, xt, proxy_idx, cfg, seed)
            candidates = [("ridge_reg", reg_pred, reg_alpha)]
            cls_alpha = math.nan
            if not args.skip_classifier:
                cls_pred, cls_alpha = _fit_predict_cls(X, y, xt, proxy_idx, cfg, seed)
                candidates.append(("ridge_cls", cls_pred, cls_alpha))
            for model_kind, pred, alpha in candidates:
                frame = pd.DataFrame({
                    "target_day": target_day,
                    "hour_business": np.arange(1, HOURLY.slots_per_day + 1),
                    "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                    "y_true_spread": yt,
                    "y_pred_spread": pred,
                    "model_name": f"{cfg.name}_{model_kind}",
                    "probe_config": cfg.name,
                    "model_kind": model_kind,
                    "alpha": alpha,
                    "history_days": cfg.history_days,
                    "proxy_dropout": cfg.proxy_dropout,
                    "information_cutoff": cutoff,
                })
                rows.append(frame)
            audits.append({
                "target_day": target_day,
                "config": cfg.name,
                "training_days": len(train_days),
                "feature_count": len(xt),
                "proxy_dropout": cfg.proxy_dropout,
                "source_max_ds": source_max,
                "cutoff": cutoff,
                "reg_alpha": reg_alpha,
                "cls_alpha": cls_alpha,
            })

    ledger = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame([_metrics(g, name) for name, g in ledger.groupby("model_name", sort=False)])
    summary = summary.sort_values(["balanced_direction_accuracy", "direction_accuracy"], ascending=[False, False])
    _atomic_parquet(out_root / "ledger" / "evaluation_ledger.parquet", ledger)
    _atomic_csv(out_root / "summary" / "model_summary.csv", summary)
    _atomic_csv(out_root / "summary" / "audit.csv", pd.DataFrame(audits))
    manifest = {
        "pipeline": "spread_direction_24_fast_linear_probe",
        "status": "complete",
        "resolution": "hourly",
        "start": dates[0], "end": dates[-1], "days": len(dates),
        "training_days": args.training_days,
        "configs": [c.__dict__ for c in active_configs],
        "alphas": list(ALPHAS),
        "forecast_columns": forecast_cols,
        "information_boundary": {
            "forecast_origin": "D-1 14:00",
            "completed_history": "D-2 and earlier only",
            "d1_context": "p1-p14 only when enabled",
            "target_day_grid": "forecast columns only",
            "target_day_actual": "label only",
        },
        "historical_preflight_override": "24-point freshness failed because source ends 2026-08-17; experiment end is earlier and all leakage/data-integrity checks passed",
        "source": source_info,
        "runtime": {"python": sys.version, "platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started},
        "top_models": summary.head(10).to_dict(orient="records"),
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
    p.add_argument("--configs", default="", help="comma-separated probe configs; empty means all")
    p.add_argument("--skip-classifier", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
