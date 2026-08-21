"""Isolated 24-point direct spread forecasting experiment.

The production dayahead/realtime ledgers are intentionally untouched.  This
runner reuses the existing model implementations, but changes the supervised
target to::

    spread = realtime_price - dayahead_price

Every target day gets an as-of-safe parquet view at D-1 14:00: realtime,
spread, and actual-side grid fields strictly after that cutoff are masked,
while target-day dayahead and target-day forecast-side grid fields remain
available.  Actual spread is joined only after prediction for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS  # noqa: E402
from utils.feature_store import FeatureStore  # noqa: E402
from utils.resolution import HOURLY  # noqa: E402


SPREAD_COL = "价差"
EXPERIMENT_SCHEMA_VERSION = "spread_cutoff_dminus1_14_v2"
MODEL_POOL = tuple(dict.fromkeys((*DAYAHEAD_MODELS, *REALTIME_MODELS)))
SAFE_BASELINES = (
    "spread_asof_lag",
    "spread_lag48",
    "spread_weekly",
    "spread_rolling_median",
)
UNSAFE_BASELINES = {"spread_lag24"}
ALL_CANDIDATES = (*MODEL_POOL, *SAFE_BASELINES)
DA_ALIASES = ("日前电价", "日前出清电价", "日前出清价格")
RT_ALIASES = ("实时电价", "实时出清电价", "实时出清价格")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_existing(frame: pd.DataFrame, aliases: Iterable[str], label: str) -> str:
    for name in aliases:
        if name in frame.columns:
            return name
    raise ValueError(f"missing {label} column; tried={list(aliases)}")


def _business_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "时刻" not in out.columns:
        raise ValueError("hourly source must contain 时刻")
    out["时刻"] = pd.to_datetime(out["时刻"], errors="coerce")
    out = out.dropna(subset=["时刻"]).sort_values("时刻").reset_index(drop=True)
    out["_business_day"] = out["时刻"].map(HOURLY.business_day_from_timestamp)
    out["_business_period"] = out["时刻"].map(HOURLY.business_period_from_timestamp).astype(int)
    return out


def prepare_source_cache(source: Path, output_root: Path) -> tuple[Path, pd.DataFrame, dict]:
    """Build/reuse a resolution-isolated parquet cache without touching source data."""
    fs = FeatureStore(
        resolution=HOURLY.label,
        source=source,
        root=output_root / "cache" / "feature_store",
    )
    base_path = fs.ensure_base()
    raw = _business_columns(pd.read_parquet(base_path))
    da_col = _first_existing(raw, DA_ALIASES, "dayahead actual")
    rt_col = _first_existing(raw, RT_ALIASES, "realtime actual")
    raw[SPREAD_COL] = pd.to_numeric(raw[rt_col], errors="coerce") - pd.to_numeric(
        raw[da_col], errors="coerce"
    )
    info = {
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "feature_store_base": str(base_path.resolve()),
        "feature_store_version": fs.version,
        "rows": int(len(raw)),
        "dayahead_column": da_col,
        "realtime_column": rt_col,
    }
    return base_path, raw, info


def actual_for_day(raw: pd.DataFrame, target_day: str, da_col: str, rt_col: str) -> pd.DataFrame:
    day = raw[raw["_business_day"].eq(target_day)].copy()
    if len(day) != HOURLY.slots_per_day:
        raise ValueError(f"{target_day}: actual rows={len(day)}, expected=24")
    day["y_true_dayahead"] = pd.to_numeric(day[da_col], errors="coerce")
    day["y_true_realtime"] = pd.to_numeric(day[rt_col], errors="coerce")
    day["y_true_spread"] = day["y_true_realtime"] - day["y_true_dayahead"]
    if day[["y_true_dayahead", "y_true_realtime", "y_true_spread"]].isna().any().any():
        raise ValueError(f"{target_day}: actual price/spread contains NaN")
    out = pd.DataFrame(
        {
            "target_day": target_day,
            "ds": day["时刻"].to_numpy(),
            "hour_business": day["_business_period"].to_numpy(dtype=int),
            "period": day["_business_period"].map(HOURLY.infer_period).to_numpy(),
            "y_true_dayahead": day["y_true_dayahead"].to_numpy(float),
            "y_true_realtime": day["y_true_realtime"].to_numpy(float),
            "y_true_spread": day["y_true_spread"].to_numpy(float),
        }
    )
    return out.sort_values("hour_business").reset_index(drop=True)


def build_asof_input(
    raw: pd.DataFrame,
    target_day: str,
    da_col: str,
    rt_col: str,
    output_root: Path,
) -> tuple[Path, dict]:
    """Materialize the exact D-1 14:00 information set for target business day D."""
    out = raw.drop(columns=["_business_day", "_business_period"], errors="ignore").copy()
    business_day = raw["_business_day"]
    timestamps = pd.to_datetime(raw["时刻"], errors="raise")
    cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    after_cutoff = timestamps.gt(cutoff)
    future_only = business_day.gt(target_day)

    # RT/spread/actual-side fields are labels after the decision timestamp,
    # including D-1 15:00..24:00.  Masking only target_day would leak ten
    # post-cutoff hours into lag features and model context.
    out[SPREAD_COL] = raw[SPREAD_COL].mask(after_cutoff)
    out[rt_col] = pd.to_numeric(raw[rt_col], errors="coerce").mask(after_cutoff)

    # Target-day DA is a known anchor for the RT/spread decision; later DA is
    # not available and is masked.  Target/future actual grid fields are never
    # model inputs in this experiment.
    out[da_col] = pd.to_numeric(raw[da_col], errors="coerce").mask(future_only)
    actual_cols = [c for c in out.columns if c.endswith("实际值")]
    if actual_cols:
        out.loc[after_cutoff, actual_cols] = np.nan
    forecast_cols = [c for c in out.columns if c.endswith("预测值")]
    if forecast_cols:
        out.loc[future_only, forecast_cols] = np.nan

    day_mask = business_day.eq(target_day)
    audit = {
        "target_day": target_day,
        "rows": int(len(out)),
        "target_rows": int(day_mask.sum()),
        "target_spread_non_null": int(out.loc[day_mask, SPREAD_COL].notna().sum()),
        "target_realtime_non_null": int(out.loc[day_mask, rt_col].notna().sum()),
        "target_dayahead_non_null": int(out.loc[day_mask, da_col].notna().sum()),
        "target_actual_feature_non_null": int(out.loc[day_mask, actual_cols].notna().sum().sum())
        if actual_cols
        else 0,
        "post_cutoff_spread_non_null": int(out.loc[after_cutoff, SPREAD_COL].notna().sum()),
        "post_cutoff_realtime_non_null": int(out.loc[after_cutoff, rt_col].notna().sum()),
        "post_cutoff_actual_feature_non_null": int(
            out.loc[after_cutoff, actual_cols].notna().sum().sum()
        )
        if actual_cols
        else 0,
        "future_forecast_feature_non_null": int(
            out.loc[future_only, forecast_cols].notna().sum().sum()
        )
        if forecast_cols
        else 0,
        "forecast_feature_columns": forecast_cols,
        "actual_feature_columns_masked": actual_cols,
        "cutoff": str(cutoff),
    }
    expected = {
        "target_rows": 24,
        "target_spread_non_null": 0,
        "target_realtime_non_null": 0,
        "target_dayahead_non_null": 24,
        "target_actual_feature_non_null": 0,
        "post_cutoff_spread_non_null": 0,
        "post_cutoff_realtime_non_null": 0,
        "post_cutoff_actual_feature_non_null": 0,
        "future_forecast_feature_non_null": 0,
    }
    bad = {k: (audit[k], v) for k, v in expected.items() if audit[k] != v}
    if bad:
        raise RuntimeError(f"{target_day}: as-of masking audit failed: {bad}")

    path = output_root / "cache" / "asof_inputs" / f"{target_day}.parquet"
    _atomic_parquet(path, out)
    return path, audit


def _normalize_prediction(
    frame: pd.DataFrame,
    pred_col: str,
    target_day: str,
    model_name: str,
    timestamp_candidates: tuple[str, ...] = ("时刻", "timestamp", "ds"),
) -> pd.DataFrame:
    ts_col = next((c for c in timestamp_candidates if c in frame.columns), None)
    if ts_col is None:
        raise ValueError(f"{model_name}: no timestamp column in {list(frame.columns)}")
    if pred_col not in frame.columns:
        raise ValueError(f"{model_name}: missing prediction column {pred_col}; cols={list(frame.columns)}")
    work = frame.copy()
    work[ts_col] = pd.to_datetime(work[ts_col], errors="coerce")
    work = work.dropna(subset=[ts_col])
    work["target_day"] = work[ts_col].map(HOURLY.business_day_from_timestamp)
    work = work[work["target_day"].eq(target_day)].copy()
    work["hour_business"] = work[ts_col].map(HOURLY.business_period_from_timestamp).astype(int)
    work["y_pred_spread"] = pd.to_numeric(work[pred_col], errors="coerce")
    if len(work) != 24 or work["hour_business"].nunique() != 24:
        raise ValueError(
            f"{model_name}/{target_day}: rows={len(work)}, unique_hours={work['hour_business'].nunique()}"
        )
    if not np.isfinite(work["y_pred_spread"].to_numpy(float)).all():
        raise ValueError(f"{model_name}/{target_day}: prediction contains NaN/inf")
    out = pd.DataFrame(
        {
            "target_day": target_day,
            "ds": work[ts_col].to_numpy(),
            "hour_business": work["hour_business"].to_numpy(dtype=int),
            "period": work["hour_business"].map(HOURLY.infer_period).to_numpy(),
            "model_name": model_name,
            "prediction_mode": "direct_spread",
            "y_pred_spread": work["y_pred_spread"].to_numpy(float),
        }
    )
    return out.sort_values("hour_business").reset_index(drop=True)


def predict_lightgbm(input_path: Path, target_day: str, model_root: Path, args) -> pd.DataFrame:
    from lightGBM.main_fix import run_lgbm_pipeline

    # The legacy trainer persists its fitted model through this environment
    # template.  Redirect it into the experiment run so production artifacts
    # under models/LightGBM are never touched.
    model_root.mkdir(parents=True, exist_ok=True)
    env_key = "LightGBM_MODEL_PATH"
    old_model_path = os.environ.get(env_key)
    os.environ[env_key] = str((model_root / "best_model_{}.pkl").resolve())
    try:
        result = run_lgbm_pipeline(
            data_path=str(input_path),
            forecast_start=target_day,
            forecast_end=target_day,
            target=SPREAD_COL,
            use_predicted_temp=False,
            training_months=args.training_months,
            val_ratio=args.val_ratio,
            resolution=HOURLY.label,
        )
    finally:
        if old_model_path is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = old_model_path
    if result is None or result.empty:
        raise RuntimeError("lightgbm returned no spread predictions")
    pred_col = next((c for c in ("pred_y", "预测价差", "预测值") if c in result.columns), None)
    if pred_col is None:
        raise ValueError(f"lightgbm unsupported output columns: {list(result.columns)}")
    return _normalize_prediction(result, pred_col, target_day, "lightgbm")


def predict_timesfm(input_path: Path, target_day: str, model_root: Path, args) -> pd.DataFrame:
    from TimesFMBackend import price_forecast_copy_分时段预测 as timesfm_core
    from TimesFMBackend.infer import predict_price_for_date

    # The backend's dataset/backtest path already supports ``spread`` but its
    # forecast lookup table historically omitted the alias.  Register the
    # experiment-only materialized target without modifying backend source.
    timesfm_core.TARGET_CFG.setdefault(
        "spread", {"keywords": [SPREAD_COL], "exclude": ["日前", "实时"]}
    )
    result = predict_price_for_date(
        data_path=str(input_path),
        forecast_date=target_day,
        target="spread",
        segment_count=3,
        seed=args.seed,
        deterministic=args.deterministic,
        resolution=HOURLY.label,
    )
    pred_col = "预测值" if "预测值" in result.columns else result.columns[-1]
    return _normalize_prediction(result, pred_col, target_day, "timesfm")


def predict_sgdfnet(input_path: Path, target_day: str, model_root: Path, args) -> pd.DataFrame:
    from SGDFNet.pipeline import ModelPipeline
    from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment

    pipeline = ModelPipeline()
    # SGDFNet emits several deeply named audit files.  On Windows the project
    # path can exceed MAX_PATH, so execute its native audit in a short temp
    # directory and copy the auditable summaries back into the experiment.
    with tempfile.TemporaryDirectory(prefix="efm3_sg_spread_") as tmp:
        tmp_config = pipeline._build_temp_config(
            data_path=str(input_path.resolve()),
            start_day=target_day,
            end_day=target_day,
            output_root=tmp,
            decision_hour=args.realtime_cutoff_hour,
            resolution=24,
            seed=args.seed,
            deterministic=args.deterministic,
        )
        run_dir = Path(run_protocol_b_cutoff_experiment(tmp_config))
        result = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")
        audit_root = model_root / "native_audit"
        audit_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "metrics_summary.json",
            "split_audit.json",
            "split_audit.csv",
            "feature_manifest.csv",
            "run_config_snapshot.json",
        ):
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, audit_root / name)
    return _normalize_prediction(result, "delta_hat", target_day, "sgdfnet")


def _timemixer_usable_days(tm, frame: pd.DataFrame, days: list[pd.Timestamp], args) -> list[pd.Timestamp]:
    usable: list[pd.Timestamp] = []
    for day in days:
        try:
            tm.make_sample(
                frame,
                day,
                target_col="realtime_price",
                seq_len=args.timemixer_seq_len,
                cutoff_hour=args.realtime_cutoff_hour,
                target_mode="direct",
                inference_mode=False,
                resolution=24,
            )
            usable.append(day)
        except Exception:
            continue
    return usable


def predict_timemixer(input_path: Path, target_day: str, model_root: Path, args) -> pd.DataFrame:
    import torch
    from TimeMixer import repro_pipeline as tm

    frame = tm.load_data(str(input_path))
    # The experiment parquet masks target-day RT.  Replacing the RT target in
    # this private in-memory frame preserves the existing RT feature path
    # (including known DA anchor) while supervising on historical spread.
    frame["realtime_price"] = frame["realtime_price"] - frame["day_ahead_clearing_price"]
    target_ts = pd.Timestamp(target_day)
    train_start = max(
        frame["ds"].min().normalize() + pd.Timedelta(days=8),
        target_ts - pd.DateOffset(months=args.training_months),
    )
    candidate_days = tm.date_range_days(train_start, target_ts)
    train_days = _timemixer_usable_days(tm, frame, candidate_days, args)
    if len(train_days) < 30:
        raise ValueError(f"timemixer has only {len(train_days)} usable training days")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    cfg = tm.RunConfig(
        data_path=str(input_path),
        output_dir=str(model_root),
        month=target_ts.strftime("%Y-%m"),
        test_start=target_day,
        test_end_exclusive=(target_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        train_months=args.training_months,
        val_ratio=args.val_ratio,
        resolution=24,
        seq_len=args.timemixer_seq_len,
        epochs=args.timemixer_epochs,
        batch_size=args.timemixer_batch_size,
        patience=args.timemixer_patience,
        seed=args.seed,
        deterministic=args.deterministic,
        device=device_name,
        cutoff_hour_rt=args.realtime_cutoff_hour,
        target_mode="direct",
        rt_target_mode="direct",
        rt_loss_mode="l1",
        segment_training=True,
        append_leaderboard=False,
    )
    tm.set_seed(args.seed, args.deterministic)
    day_preds = np.zeros(24, dtype=float)
    history: dict[str, list[dict]] = {}
    for segment_name, start_idx, end_idx in tm._segments(24):
        train_past, train_future, train_y, _ = tm.build_segment_arrays(
            frame,
            train_days,
            "realtime_price",
            args.timemixer_seq_len,
            args.realtime_cutoff_hour,
            start_idx,
            end_idx,
            target_mode="direct",
            resolution=24,
        )
        bundle = tm.train_model(
            train_past,
            train_future,
            train_y,
            cfg,
            device,
            task="rt",
            segment_name=segment_name,
        )
        test_past, test_future, _, _ = tm.build_segment_arrays(
            frame,
            [target_ts],
            "realtime_price",
            args.timemixer_seq_len,
            args.realtime_cutoff_hour,
            start_idx,
            end_idx,
            target_mode="direct",
            inference_mode=True,
            resolution=24,
        )
        pred = tm.predict_model(bundle, test_past, test_future, device, args.timemixer_batch_size, 24)[0]
        day_preds[start_idx:end_idx] = pred
        history[segment_name] = bundle["history"]

    actual_ds = pd.date_range(target_ts + pd.Timedelta(hours=1), periods=24, freq="h")
    raw = pd.DataFrame({"时刻": actual_ds, "预测值": day_preds})
    _atomic_json(model_root / "training_history.json", history)
    return _normalize_prediction(raw, "预测值", target_day, "timemixer")


def predict_rt916(input_path: Path, target_day: str, model_root: Path, args) -> pd.DataFrame:
    from RT916_SpikeFusionNet.src.rt916_spikefusionnet import core

    core.RAW_DF_PATH = str(input_path.resolve())
    core.PACKAGE_OUT_ROOT = model_root / "native_runs"
    core.set_resolution(24)
    core.CONFIG["SEED"] = args.seed
    os.environ["RT916_TRAIN_STEPS"] = str(args.rt916_train_steps)
    os.environ["SPIKE_TRAIN_MONTHS"] = str(args.training_months)
    os.environ["OPTIM_AMP"] = "1" if args.device != "cpu" else "0"
    target_ts = pd.Timestamp(target_day)
    result = core.run(
        target=SPREAD_COL,
        start_end_list=[
            (target_ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            (target_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
        ],
        mod="all",
        asof_ts=target_ts - pd.Timedelta(days=1) + pd.Timedelta(hours=args.realtime_cutoff_hour),
        enforce_asof_cutoff=True,
    )
    if result is None or result.empty:
        raise RuntimeError("rt916 returned no spread predictions")
    return _normalize_prediction(result, f"预测{SPREAD_COL}", target_day, "rt916")


MODEL_RUNNERS: dict[str, Callable] = {
    "lightgbm": predict_lightgbm,
    "timesfm": predict_timesfm,
    "timemixer": predict_timemixer,
    "sgdfnet": predict_sgdfnet,
    "rt916": predict_rt916,
}


def predict_safe_baseline(
    raw: pd.DataFrame,
    actual: pd.DataFrame,
    target_day: str,
    model_name: str,
) -> pd.DataFrame:
    """Predict from values observable no later than D-1 14:00."""
    cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    work = raw.copy()
    work["时刻"] = pd.to_datetime(work["时刻"], errors="raise")
    spread_by_ds = work.set_index("时刻")[SPREAD_COL]
    predictions: list[float] = []
    source_max: list[pd.Timestamp] = []

    for row in actual.itertuples(index=False):
        target_ts = pd.Timestamp(row.ds)
        period = int(row.hour_business)
        if model_name == "spread_asof_lag":
            source_ts = target_ts - pd.Timedelta(days=1)
            if source_ts > cutoff:
                source_ts -= pd.Timedelta(days=1)
            value = spread_by_ds.get(source_ts, np.nan)
            used_max = source_ts
        elif model_name == "spread_lag48":
            source_ts = target_ts - pd.Timedelta(days=2)
            value = spread_by_ds.get(source_ts, np.nan)
            used_max = source_ts
        elif model_name == "spread_weekly":
            source_ts = target_ts - pd.Timedelta(days=7)
            value = spread_by_ds.get(source_ts, np.nan)
            used_max = source_ts
        elif model_name == "spread_rolling_median":
            history = work[
                work["时刻"].le(cutoff)
                & work["_business_period"].eq(period)
                & work[SPREAD_COL].notna()
            ].sort_values("时刻").tail(28)
            if history.empty:
                value = np.nan
                used_max = pd.NaT
            else:
                value = float(pd.to_numeric(history[SPREAD_COL], errors="coerce").median())
                used_max = pd.Timestamp(history["时刻"].max())
        else:
            raise ValueError(f"unknown safe baseline: {model_name}")
        if pd.notna(used_max) and used_max > cutoff:
            raise RuntimeError(
                f"{target_day}/{model_name}/p{period}: source {used_max} exceeds cutoff {cutoff}"
            )
        predictions.append(value)
        source_max.append(used_max)

    out = actual[["target_day", "ds", "hour_business", "period"]].copy()
    out["model_name"] = model_name
    out["prediction_mode"] = "cutoff_safe_historical_baseline"
    out["y_pred_spread"] = pd.to_numeric(pd.Series(predictions), errors="coerce").to_numpy()
    out["source_max_ds"] = source_max
    out["information_cutoff"] = cutoff
    if not np.isfinite(out["y_pred_spread"].to_numpy(float)).all():
        raise ValueError(f"{target_day}: {model_name} baseline contains NaN/inf")
    return out


def score_predictions(predictions: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    joined = predictions.merge(
        actual,
        on=["target_day", "ds", "hour_business", "period"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 24:
        raise ValueError(f"evaluation join rows={len(joined)}, expected=24")
    true = joined["y_true_spread"].to_numpy(float)
    pred = joined["y_pred_spread"].to_numpy(float)
    true_sign = np.sign(true)
    pred_sign = np.sign(pred)
    directional = true_sign != 0
    correct = directional & (true_sign == pred_sign)
    joined["actual_direction"] = true_sign.astype(int)
    joined["predicted_direction"] = pred_sign.astype(int)
    joined["direction_eligible"] = directional
    joined["direction_correct"] = correct

    pos = true_sign > 0
    neg = true_sign < 0
    direction_accuracy = float(correct[directional].mean()) if directional.any() else math.nan
    pos_accuracy = float(correct[pos].mean()) if pos.any() else math.nan
    neg_accuracy = float(correct[neg].mean()) if neg.any() else math.nan
    balanced = float(np.nanmean([pos_accuracy, neg_accuracy]))
    weights = np.abs(true[directional])
    weighted = (
        float(np.average(correct[directional].astype(float), weights=weights))
        if directional.any() and weights.sum() > 0
        else math.nan
    )
    metrics = {
        "model_name": str(predictions["model_name"].iloc[0]),
        "target_day": str(actual["target_day"].iloc[0]),
        "n_slots": 24,
        "n_direction_eligible": int(directional.sum()),
        "n_zero_actual": int((true_sign == 0).sum()),
        "direction_accuracy": direction_accuracy,
        "positive_accuracy": pos_accuracy,
        "negative_accuracy": neg_accuracy,
        "balanced_direction_accuracy": balanced,
        "abs_spread_weighted_direction_accuracy": weighted,
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "near_zero_abs_le_1_accuracy": float(correct[np.abs(true) <= 1].mean())
        if (np.abs(true) <= 1).any()
        else math.nan,
        "near_zero_abs_le_5_accuracy": float(correct[np.abs(true) <= 5].mean())
        if (np.abs(true) <= 5).any()
        else math.nan,
    }
    return joined, metrics


def _resolve_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_CANDIDATES)
    models = [x.strip().lower() for x in value.split(",") if x.strip()]
    unsafe = sorted(set(models) & UNSAFE_BASELINES)
    if unsafe:
        raise ValueError(
            "spread_lag24 is forbidden: p15-p24 would use D-1 post-14:00 realtime; "
            "use spread_asof_lag or spread_lag48"
        )
    unknown = sorted(set(models) - set(ALL_CANDIDATES))
    if unknown:
        raise ValueError(f"unknown candidates={unknown}; allowed={list(ALL_CANDIDATES)}")
    return models


def _date_list(start: str, end: str) -> list[str]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if e < s:
        raise ValueError("--end must be >= --start")
    return [(s + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range((e - s).days + 1)]


def run(args) -> dict:
    source = Path(args.data_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _, raw, source_info = prepare_source_cache(source, output_root)
    da_col = source_info["dayahead_column"]
    rt_col = source_info["realtime_column"]
    models = _resolve_models(args.models)
    if args.date_step <= 0:
        raise ValueError("--date-step must be positive")
    dates = _date_list(args.start, args.end)[:: args.date_step]
    range_manifest = {
        "pipeline": "spread_direction_24_experiment",
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "status": "running",
        "resolution": HOURLY.label,
        "slots_per_day": HOURLY.slots_per_day,
        "target_definition": "realtime_actual - dayahead_actual",
        "direction_definition": "same strict sign; actual zero excluded; predicted zero is wrong",
        "models": models,
        "start": args.start,
        "end": args.end,
        "date_step": args.date_step,
        "selected_dates": dates,
        "source": source_info,
        "information_boundary": {
            "forecast_origin": "D-1 14:00",
            "decision_day_realtime": "visible through 14:00 inclusive; later rows masked",
            "decision_day_actual_features": "visible through 14:00 inclusive; later rows masked",
            "target_day_dayahead": "allowed known anchor",
            "target_day_realtime": "masked label",
            "target_day_actual_features": "masked",
            "target_day_forecast_features": "allowed",
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "seed": args.seed,
            "deterministic": args.deterministic,
            "training_months": args.training_months,
            "timemixer_epochs": args.timemixer_epochs,
            "rt916_train_steps": args.rt916_train_steps,
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "daily": [],
    }
    _atomic_json(output_root / "range_manifest.json", range_manifest)

    all_metrics: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    all_evaluations: list[pd.DataFrame] = []
    for target_day in dates:
        day_start = time.perf_counter()
        run_dir = output_root / "runs" / target_day
        prior_manifest_path = run_dir / "run_manifest.json"
        prior_manifest = (
            json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            if prior_manifest_path.exists()
            else {}
        )
        actual = actual_for_day(raw, target_day, da_col, rt_col)
        _atomic_parquet(run_dir / "actual" / "actual_spread.parquet", actual)
        input_path, mask_audit = build_asof_input(raw, target_day, da_col, rt_col, output_root)
        day_manifest = {
            "target_day": target_day,
            "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
            "status": "running",
            "resolution": HOURLY.label,
            "cutoff": mask_audit["cutoff"],
            "asof_input": str(input_path),
            "mask_audit": mask_audit,
            "models": {},
            "errors": [],
        }
        for model_name in models:
            started = time.perf_counter()
            try:
                pred_path = run_dir / "prediction" / f"{model_name}_predictions.csv"
                cached = (
                    pred_path.exists()
                    and not args.force
                    and prior_manifest.get("experiment_schema_version")
                    == EXPERIMENT_SCHEMA_VERSION
                )
                if cached:
                    pred = pd.read_csv(pred_path)
                    pred["ds"] = pd.to_datetime(pred["ds"], errors="coerce")
                    if (
                        len(pred) != 24
                        or pred["hour_business"].nunique() != 24
                        or pred["target_day"].astype(str).nunique() != 1
                        or str(pred["target_day"].iloc[0]) != target_day
                        or not np.isfinite(pd.to_numeric(pred["y_pred_spread"], errors="coerce")).all()
                    ):
                        raise ValueError(f"invalid cached prediction: {pred_path}")
                elif model_name in SAFE_BASELINES:
                    pred = predict_safe_baseline(raw, actual, target_day, model_name)
                else:
                    pred = MODEL_RUNNERS[model_name](
                        input_path,
                        target_day,
                        run_dir / "model_runtime" / model_name,
                        args,
                    )
                if cached:
                    elapsed = float(
                        prior_manifest.get("models", {})
                        .get(model_name, {})
                        .get("elapsed_seconds", 0.0)
                    )
                else:
                    elapsed = time.perf_counter() - started
                joined, metrics = score_predictions(pred, actual)
                metrics["elapsed_seconds"] = elapsed
                if not cached:
                    _atomic_csv(pred_path, pred)
                _atomic_csv(run_dir / "evaluation" / f"{model_name}_evaluation.csv", joined)
                day_manifest["models"][model_name] = {
                    "status": "cached" if cached else "ok",
                    "elapsed_seconds": round(elapsed, 3),
                    "rows": int(len(pred)),
                    "metrics": metrics,
                }
                all_predictions.append(pred)
                all_evaluations.append(joined)
                all_metrics.append(metrics)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                day_manifest["models"][model_name] = {
                    "status": "failed",
                    "elapsed_seconds": round(elapsed, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                day_manifest["errors"].append(f"{model_name}: {type(exc).__name__}: {exc}")
                if args.fail_fast:
                    _atomic_json(run_dir / "run_manifest.json", day_manifest)
                    raise

        ok = [
            m
            for m in models
            if day_manifest["models"].get(m, {}).get("status") in {"ok", "cached"}
        ]
        day_manifest["elapsed_seconds"] = round(time.perf_counter() - day_start, 3)
        day_manifest["status"] = "complete" if len(ok) == len(models) else "partial"
        _atomic_json(run_dir / "run_manifest.json", day_manifest)
        range_manifest["daily"].append(
            {
                "target_day": target_day,
                "status": day_manifest["status"],
                "ok_models": ok,
                "failed_models": sorted(set(models) - set(ok)),
                "elapsed_seconds": day_manifest["elapsed_seconds"],
            }
        )
        _atomic_json(output_root / "range_manifest.json", range_manifest)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        _atomic_csv(output_root / "summary" / "daily_model_metrics.csv", metrics_df)
        evaluation_df = pd.concat(all_evaluations, ignore_index=True)
        _atomic_parquet(output_root / "ledger" / "evaluation_ledger.parquet", evaluation_df)
        summary_rows: list[dict] = []
        elapsed_mean = metrics_df.groupby("model_name")["elapsed_seconds"].mean().to_dict()
        for model_name, group in evaluation_df.groupby("model_name"):
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
            weights = np.abs(true[eligible])
            summary_rows.append(
                {
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
                    "abs_spread_weighted_direction_accuracy": float(
                        np.average(correct[eligible].astype(float), weights=weights)
                    )
                    if eligible.any() and weights.sum() > 0
                    else math.nan,
                    "mae": float(np.mean(np.abs(pred - true))),
                    "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
                    "elapsed_seconds": float(elapsed_mean.get(model_name, math.nan)),
                }
            )
        summary = pd.DataFrame(summary_rows)
        summary = summary.sort_values(
            ["direction_accuracy", "balanced_direction_accuracy"], ascending=False
        ).reset_index(drop=True)
        _atomic_csv(output_root / "summary" / "model_summary.csv", summary)
    if all_predictions:
        _atomic_parquet(
            output_root / "ledger" / "prediction_ledger.parquet",
            pd.concat(all_predictions, ignore_index=True),
        )
    range_manifest["status"] = (
        "complete"
        if all(x["status"] == "complete" for x in range_manifest["daily"])
        else "partial"
    )
    range_manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_root / "range_manifest.json", range_manifest)
    return range_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--output-root", default="outputs/experiments/spread_direction_24")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--date-step",
        type=int,
        default=1,
        help="run every Nth date from --start; useful for predeclared challenger sampling",
    )
    parser.add_argument("--models", default="all", help="comma-separated candidates or all")
    parser.add_argument("--training-months", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--realtime-cutoff-hour", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--timemixer-seq-len", type=int, default=168)
    parser.add_argument("--timemixer-epochs", type=int, default=10)
    parser.add_argument("--timemixer-patience", type=int, default=5)
    parser.add_argument("--timemixer-batch-size", type=int, default=16)
    parser.add_argument("--rt916-train-steps", type=int, default=24)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore valid prediction caches")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
