"""Build a cutoff-safe hourly spread Feature Cube for fast model experiments.

Experiment-only. Outputs live under outputs/experiments and production code/ledgers
are untouched.

Information contract for predicting business day D at D-1 14:00:
- target-day actual prices / actual grid values are labels only, never features;
- target-day forecast grid values are allowed;
- D-1 spread is only used through p1-p14 current-regime summaries;
- historical spread / forecast-error statistics use D-2 and earlier only.

The cube exposes three reusable representations:
1) slot_table.parquet: one row per target day/hour for tree/linear probes;
2) day_matrix.npz: one row per target day with all slot features flattened;
3) sequence_cube.npz: completed-history + D-1 partial context + target forecasts.
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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_lear_spread_experiment import prepare_source_cache  # noqa: E402
from utils.resolution import HOURLY  # noqa: E402

SPREAD_COL = "价差"
DA_COL = "日前电价"
RT_COL = "实时电价"

FORECAST_SUFFIX = "预测值"
ACTUAL_SUFFIX = "实际值"

GROUP_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    a = pd.to_numeric(num, errors="coerce")
    b = pd.to_numeric(den, errors="coerce")
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _rolling_same_slot(frame: pd.DataFrame, source_col: str, *, shift_days: int, window: int, stat: str) -> pd.Series:
    shifted = frame.groupby("_business_period", sort=False)[source_col].shift(shift_days)
    grouped = shifted.groupby(frame["_business_period"], sort=False)
    min_periods = min(max(3, window // 4), window)
    if stat == "mean":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    if stat == "std":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).std(ddof=0))
    if stat == "median":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).median())
    if stat == "q10":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).quantile(0.10))
    if stat == "q90":
        return grouped.transform(lambda s: s.rolling(window, min_periods=min_periods).quantile(0.90))
    if stat == "positive_rate":
        return grouped.transform(lambda s: s.gt(0).astype(float).rolling(window, min_periods=min_periods).mean())
    raise ValueError(stat)


def _daily_context(frame: pd.DataFrame) -> pd.DataFrame:
    partial = frame[frame["_business_period"].between(1, 14)].copy()
    records = []
    for day, g in partial.groupby("_business_day", sort=True):
        g = g.sort_values("_business_period")
        x = pd.to_numeric(g[SPREAD_COL], errors="coerce").to_numpy(float)
        t = np.arange(len(x), dtype=float)
        valid = np.isfinite(x)
        xv = x[valid]
        slope = float(np.polyfit(t[valid], xv, 1)[0]) if valid.sum() >= 3 else math.nan
        max_ts = pd.to_datetime(g.loc[valid, "时刻"], errors="coerce").max() if valid.any() else pd.NaT
        record = {
            "context_day": str(day),
            "target_day": (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "ctx_spread_mean14": float(np.nanmean(x)) if valid.any() else math.nan,
            "ctx_spread_std14": float(np.nanstd(x)) if valid.any() else math.nan,
            "ctx_spread_median14": float(np.nanmedian(x)) if valid.any() else math.nan,
            "ctx_spread_last": float(xv[-1]) if len(xv) else math.nan,
            "ctx_spread_mean3": float(np.nanmean(x[-3:])) if len(x) >= 3 else math.nan,
            "ctx_spread_min14": float(np.nanmin(x)) if valid.any() else math.nan,
            "ctx_spread_max14": float(np.nanmax(x)) if valid.any() else math.nan,
            "ctx_spread_range14": float(np.nanmax(x) - np.nanmin(x)) if valid.any() else math.nan,
            "ctx_spread_absmean14": float(np.nanmean(np.abs(x))) if valid.any() else math.nan,
            "ctx_spread_positive_rate14": float(np.nanmean(x > 0)) if valid.any() else math.nan,
            "ctx_spread_negative_rate14": float(np.nanmean(x < 0)) if valid.any() else math.nan,
            "ctx_spread_slope14": slope,
            "context_source_max_ds": max_ts,
        }
        # Preserve the complete allowable D-1 morning trajectory.  This is
        # deliberately separate from F1 summaries: the gate can learn a
        # transition pattern (e.g. a fast morning reversal) without ever
        # seeing D-1 p15-p24 or target-day realized prices.
        for period in range(1, 15):
            value = x[period - 1] if len(x) >= period else math.nan
            record[f"ctxraw_spread_p{period:02d}"] = float(value) if np.isfinite(value) else math.nan
        records.append(record)
    return pd.DataFrame(records)


def _add_feature(registry: list[dict], group: str, name: str, source: str, availability: str, transform: str) -> None:
    registry.append({
        "group": group,
        "feature": name,
        "source": source,
        "availability": availability,
        "transform": transform,
        "task": "spread",
        "leakage_status": "safe_by_builder_contract",
    })


def build_slot_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], list[dict], list[str], list[str]]:
    work = raw.copy()
    work["时刻"] = pd.to_datetime(work["时刻"], errors="raise")
    work = work.sort_values("时刻").reset_index(drop=True)
    if "_business_day" not in work.columns:
        work["_business_day"] = work["时刻"].map(HOURLY.business_day_from_timestamp)
    if "_business_period" not in work.columns:
        work["_business_period"] = work["时刻"].map(HOURLY.business_period_from_timestamp).astype(int)
    work[SPREAD_COL] = pd.to_numeric(work[RT_COL], errors="coerce") - pd.to_numeric(work[DA_COL], errors="coerce")
    work["target_day"] = work["_business_day"].astype(str)
    work["hour_business"] = work["_business_period"].astype(int)
    work["period"] = work["hour_business"].map(HOURLY.infer_period)
    work["target_spread"] = work[SPREAD_COL]
    work["target_direction"] = np.sign(work[SPREAD_COL]).astype(float)
    work["hour_sin"] = np.sin(2 * np.pi * (work["hour_business"] - 1) / HOURLY.slots_per_day)
    work["hour_cos"] = np.cos(2 * np.pi * (work["hour_business"] - 1) / HOURLY.slots_per_day)
    target_dates = pd.to_datetime(work["target_day"])
    work["dow_sin"] = np.sin(2 * np.pi * target_dates.dt.dayofweek / 7)
    work["dow_cos"] = np.cos(2 * np.pi * target_dates.dt.dayofweek / 7)

    forecast_cols = [c for c in work.columns if str(c).endswith(FORECAST_SUFFIX)]
    actual_cols = [c for c in work.columns if str(c).endswith(ACTUAL_SUFFIX)]
    actual_by_base = {c[: -len(ACTUAL_SUFFIX)]: c for c in actual_cols}
    paired = [(c[: -len(FORECAST_SUFFIX)], c, actual_by_base[c[: -len(FORECAST_SUFFIX)]]) for c in forecast_cols if c[: -len(FORECAST_SUFFIX)] in actual_by_base]
    if not paired:
        raise ValueError("no forecast/actual grid pairs")

    registry: list[dict] = []
    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}

    # F0: cutoff-safe spread history. All full-history values use D-2 or earlier.
    for lag in (2, 3, 7):
        name = f"spread_lag{lag}d"
        work[name] = work.groupby("_business_period", sort=False)[SPREAD_COL].shift(lag)
        groups["F0"].append(name)
        _add_feature(registry, "F0", name, SPREAD_COL, "D-2 or earlier", f"same-slot lag {lag}d")
    for stat in ("mean", "std", "median", "positive_rate"):
        name = f"spread_same_slot_28d_{stat}"
        work[name] = _rolling_same_slot(work, SPREAD_COL, shift_days=2, window=28, stat=stat)
        groups["F0"].append(name)
        _add_feature(registry, "F0", name, SPREAD_COL, "D-2 or earlier", f"same-slot rolling28 {stat}")
    for name in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        groups["F0"].append(name)
        _add_feature(registry, "F0", name, "calendar/business slot", "known", "cyclical encoding")

    # F1: D-1 p1-p14 current market state, broadcast to target D slots.
    context = _daily_context(work)
    ctx_cols = [c for c in context.columns if c.startswith("ctx_")]
    raw_ctx_cols = [c for c in context.columns if c.startswith("ctxraw_")]
    work = work.merge(context[["target_day", "context_source_max_ds", *ctx_cols, *raw_ctx_cols]], on="target_day", how="left")
    for name in ctx_cols:
        groups["F1"].append(name)
        _add_feature(registry, "F1", name, "D-1 spread p1-p14", "D-1 <=14:00", "daily current-regime summary")
    for name in raw_ctx_cols:
        groups["F8"].append(name)
        _add_feature(registry, "F8", name, "D-1 spread p1-p14", "D-1 <=14:00", "raw visible morning trajectory")

    # F2: target-day forecast fundamentals, slot-preserving.
    fcast_alias: dict[str, str] = {}
    for base, fcol, _ in paired:
        name = f"fcast_{base}"
        work[name] = pd.to_numeric(work[fcol], errors="coerce")
        fcast_alias[base] = name
        groups["F2"].append(name)
        _add_feature(registry, "F2", name, fcol, "target D forecast known at origin", "identity")

    def col(base: str) -> pd.Series:
        key = fcast_alias.get(base)
        if key is None:
            return pd.Series(np.nan, index=work.index, dtype=float)
        return work[key]

    # F3: physical relationships.
    load = col("直调负荷")
    wind = col("风电总加")
    solar = col("光伏总加")
    renew = col("新能源总加")
    space = col("竞价空间")
    inter = col("联络线受电负荷")
    physical = {
        "residual_load_ws": load - wind - solar,
        "residual_load_renew": load - renew,
        "renewable_share": _safe_div(renew, load),
        "wind_share": _safe_div(wind, load),
        "solar_share": _safe_div(solar, load),
        "bidding_space_ratio": _safe_div(space, load),
        "interconnect_share": _safe_div(inter, load),
        "renewable_minus_space": renew - space,
    }
    for name, values in physical.items():
        work[name] = values
        groups["F3"].append(name)
        _add_feature(registry, "F3", name, "target D forecast fundamentals", "known", "physical ratio/difference")

    # F4: within-target-day ramps, preserving hourly alignment.
    ramp_sources = {
        "load": fcast_alias.get("直调负荷"),
        "wind": fcast_alias.get("风电总加"),
        "solar": fcast_alias.get("光伏总加"),
        "renewable": fcast_alias.get("新能源总加"),
        "bidding_space": fcast_alias.get("竞价空间"),
        "residual_load": "residual_load_renew",
    }
    for short, source in ramp_sources.items():
        if source is None:
            continue
        name = f"ramp_{short}"
        work[name] = work.groupby("target_day", sort=False)[source].diff().fillna(0.0)
        groups["F4"].append(name)
        _add_feature(registry, "F4", name, source, "target D forecast known at origin", "within-day first difference")
        name2 = f"ramp2_{short}"
        work[name2] = work.groupby("target_day", sort=False)[name].diff().fillna(0.0)
        groups["F4"].append(name2)
        _add_feature(registry, "F4", name2, name, "target D forecast known at origin", "within-day second difference")

    # F5/F6: historical forecast-error state and pseudo probabilistic bands.
    for base, fcol, acol in paired:
        err = f"__err_{base}"
        work[err] = pd.to_numeric(work[acol], errors="coerce") - pd.to_numeric(work[fcol], errors="coerce")
        safe_stats = {}
        for window in (7, 28):
            for stat in ("mean", "std"):
                name = f"err_{base}_{window}d_{stat}"
                work[name] = _rolling_same_slot(work, err, shift_days=2, window=window, stat=stat)
                groups["F5"].append(name)
                safe_stats[(window, stat)] = name
                _add_feature(registry, "F5", name, f"{acol}-{fcol}", "D-2 or earlier", f"same-slot rolling{window} {stat}")
        q10 = f"err_{base}_28d_q10"
        q50 = f"err_{base}_28d_q50"
        q90 = f"err_{base}_28d_q90"
        work[q10] = _rolling_same_slot(work, err, shift_days=2, window=28, stat="q10")
        work[q50] = _rolling_same_slot(work, err, shift_days=2, window=28, stat="median")
        work[q90] = _rolling_same_slot(work, err, shift_days=2, window=28, stat="q90")
        for name, stat in ((q10, "q10"), (q50, "q50"), (q90, "q90")):
            groups["F5"].append(name)
            _add_feature(registry, "F5", name, f"{acol}-{fcol}", "D-2 or earlier", f"same-slot rolling28 {stat}")
        f_alias = fcast_alias[base]
        for suffix, err_name in (("low", q10), ("mid", q50), ("high", q90)):
            name = f"uncert_{base}_{suffix}"
            work[name] = work[f_alias] + work[err_name]
            groups["F6"].append(name)
            _add_feature(registry, "F6", name, f"{fcol}+historical error", "target forecast + D-2 or earlier errors", f"pseudo quantile {suffix}")
        width = f"uncert_{base}_width"
        work[width] = work[q90] - work[q10]
        groups["F6"].append(width)
        _add_feature(registry, "F6", width, "historical forecast error", "D-2 or earlier", "q90-q10 uncertainty width")

    # Aggregate net forecast-error state from load minus renewable.
    if "__err_直调负荷" in work.columns and "__err_新能源总加" in work.columns:
        work["__err_net_load"] = work["__err_直调负荷"] - work["__err_新能源总加"]
        for stat in ("mean", "std", "q10", "q90"):
            name = f"err_net_load_28d_{stat}"
            work[name] = _rolling_same_slot(work, "__err_net_load", shift_days=2, window=28, stat=stat)
            groups["F5"].append(name)
            _add_feature(registry, "F5", name, "load error - renewable error", "D-2 or earlier", f"same-slot rolling28 {stat}")

    # F7: anomaly/regime features relative to safe historical forecast distributions.
    regime_sources = [*groups["F2"], *groups["F3"]]
    for source in regime_sources:
        hist_mean = _rolling_same_slot(work, source, shift_days=2, window=28, stat="mean")
        hist_std = _rolling_same_slot(work, source, shift_days=2, window=28, stat="std")
        z = f"regime_z_{source}"
        work[z] = (work[source] - hist_mean) / hist_std.replace(0, np.nan)
        groups["F7"].append(z)
        _add_feature(registry, "F7", z, source, "target forecast vs D-2 or earlier forecast history", "same-slot rolling zscore")
    for source in ("residual_load_renew", "renewable_share", "bidding_space_ratio"):
        if source not in work.columns:
            continue
        z = f"regime_z_{source}"
        if z in work.columns:
            hi = f"regime_high_{source}"
            lo = f"regime_low_{source}"
            work[hi] = work[z].gt(1.5).astype(float)
            work[lo] = work[z].lt(-1.5).astype(float)
            groups["F7"].extend([hi, lo])
            _add_feature(registry, "F7", hi, z, "known", "z > 1.5")
            _add_feature(registry, "F7", lo, z, "known", "z < -1.5")

    # Contract checks: current context must not exceed D-1 14:00.
    target_ts = pd.to_datetime(work["target_day"])
    cutoff = target_ts - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    ctx_source = pd.to_datetime(work["context_source_max_ds"], errors="coerce")
    bad = ctx_source.notna() & (ctx_source > cutoff)
    if bad.any():
        row = work.loc[bad, ["target_day", "context_source_max_ds"]].head(1).to_dict("records")[0]
        raise RuntimeError(f"context leakage detected: {row}")

    feature_cols = [f for g in GROUP_ORDER for f in groups[g]]
    keep = [
        "target_day", "时刻", "hour_business", "period", "target_spread", "target_direction",
        "context_source_max_ds", *feature_cols,
    ]
    slot = work[keep].copy()
    slot = slot[slot["target_spread"].notna()].reset_index(drop=True)
    return slot, groups, registry, forecast_cols, [a for _, _, a in paired]


def build_day_matrix(slot: pd.DataFrame, feature_cols: list[str]) -> dict[str, np.ndarray]:
    days = sorted(slot["target_day"].unique())
    X, Y, kept = [], [], []
    for day in days:
        g = slot[slot["target_day"].eq(day)].sort_values("hour_business")
        if len(g) != HOURLY.slots_per_day:
            continue
        X.append(g[feature_cols].to_numpy(np.float32).reshape(-1))
        Y.append(g["target_spread"].to_numpy(np.float32))
        kept.append(day)
    return {
        "X": np.asarray(X, dtype=np.float32),
        "Y": np.asarray(Y, dtype=np.float32),
        "dates": np.asarray(kept),
        "feature_names": np.asarray([f"h{h:02d}::{f}" for h in range(1, HOURLY.slots_per_day + 1) for f in feature_cols]),
    }


def build_sequence_cube(raw: pd.DataFrame, forecast_cols: list[str], actual_cols: list[str]) -> dict[str, np.ndarray]:
    work = raw.copy()
    work["时刻"] = pd.to_datetime(work["时刻"], errors="raise")
    if "_business_day" not in work.columns:
        work["_business_day"] = work["时刻"].map(HOURLY.business_day_from_timestamp)
    if "_business_period" not in work.columns:
        work["_business_period"] = work["时刻"].map(HOURLY.business_period_from_timestamp).astype(int)
    work[SPREAD_COL] = pd.to_numeric(work[RT_COL], errors="coerce") - pd.to_numeric(work[DA_COL], errors="coerce")
    day_map = {str(d): g.sort_values("_business_period").copy() for d, g in work.groupby("_business_day", sort=True) if len(g) == HOURLY.slots_per_day}
    days = sorted(day_map)
    hist_cols = [SPREAD_COL, DA_COL, RT_COL, *forecast_cols, *actual_cols]
    X_hist, X_ctx, X_future, Y, kept = [], [], [], [], []
    for day in days:
        d = pd.Timestamp(day)
        hist_days = [(d - pd.Timedelta(days=k)).strftime("%Y-%m-%d") for k in range(8, 1, -1)]  # D-8..D-2
        d1 = (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if any(x not in day_map for x in [*hist_days, d1, day]):
            continue
        hist = pd.concat([day_map[x] for x in hist_days], ignore_index=True)
        if len(hist) != 7 * HOURLY.slots_per_day:
            continue
        target = day_map[day]
        prev = day_map[d1]
        ctx = np.zeros((HOURLY.slots_per_day, 3), dtype=np.float32)
        spread_prev = pd.to_numeric(prev[SPREAD_COL], errors="coerce").to_numpy(float)
        ctx[:14, 0] = np.nan_to_num(spread_prev[:14], nan=0.0)
        ctx[:14, 1] = 1.0
        # third channel is a safe same-slot D-2 proxy for the unavailable tail and as a separate proxy channel
        d2 = (d - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        d2_spread = pd.to_numeric(day_map[d2][SPREAD_COL], errors="coerce").to_numpy(float)
        ctx[:, 2] = np.nan_to_num(d2_spread, nan=0.0)
        hist_arr = hist[hist_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        fut_arr = target[forecast_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        y = pd.to_numeric(target[SPREAD_COL], errors="coerce").to_numpy(np.float32)
        if not np.isfinite(y).all():
            continue
        X_hist.append(hist_arr)
        X_ctx.append(ctx)
        X_future.append(fut_arr)
        Y.append(y)
        kept.append(day)
    return {
        "X_hist": np.asarray(X_hist, dtype=np.float32),
        "X_context": np.asarray(X_ctx, dtype=np.float32),
        "X_future": np.asarray(X_future, dtype=np.float32),
        "Y": np.asarray(Y, dtype=np.float32),
        "dates": np.asarray(kept),
        "hist_feature_names": np.asarray(hist_cols),
        "context_feature_names": np.asarray(["d1_visible_spread", "d1_visibility_mask", "d2_same_slot_proxy"]),
        "future_feature_names": np.asarray(forecast_cols),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    p.add_argument("--cache-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_shared_cache_opt_20260820")
    p.add_argument("--output-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    args = p.parse_args()
    started = time.perf_counter()
    out = Path(args.output_root)
    source = Path(args.data_path)
    _, raw, source_info = prepare_source_cache(source, out, Path(args.cache_root))
    slot, groups, registry, forecast_cols, actual_cols = build_slot_table(raw)
    feature_cols = [f for g in GROUP_ORDER for f in groups[g]]
    _atomic_parquet(out / "slot_table.parquet", slot)

    day = build_day_matrix(slot, feature_cols)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "day_matrix.npz", **day)
    seq = build_sequence_cube(raw, forecast_cols, actual_cols)
    np.savez_compressed(out / "sequence_cube.npz", **seq)

    _atomic_json(out / "feature_groups.json", groups)
    _atomic_json(out / "feature_registry.json", {"schema": "spread_feature_cube_v1", "features": registry})
    manifest = {
        "pipeline": "spread_feature_cube",
        "schema": "spread_feature_cube_v1",
        "status": "complete",
        "resolution": HOURLY.label,
        "source": source_info,
        "information_boundary": {
            "forecast_origin": "D-1 14:00",
            "target_actual": "labels only",
            "target_forecast_grid": "allowed, slot-preserving",
            "d1_spread": "p1-p14 only via current-regime/context",
            "historical_spread": "D-2 or earlier",
            "historical_forecast_error": "D-2 or earlier",
        },
        "slot_table": {"rows": int(len(slot)), "days": int(slot["target_day"].nunique()), "features": len(feature_cols)},
        "day_matrix": {"samples": int(day["X"].shape[0]), "shape_X": list(day["X"].shape), "shape_Y": list(day["Y"].shape)},
        "sequence_cube": {
            "samples": int(seq["Y"].shape[0]),
            "shape_hist": list(seq["X_hist"].shape),
            "shape_context": list(seq["X_context"].shape),
            "shape_future": list(seq["X_future"].shape),
            "shape_Y": list(seq["Y"].shape),
        },
        "group_counts": {g: len(groups[g]) for g in GROUP_ORDER},
        "forecast_columns": forecast_cols,
        "actual_columns": actual_cols,
        "runtime_seconds": time.perf_counter() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
