"""
Ledger predict pipeline.

Runs all models for a single target day D, producing 24-hour predictions
per model, standardized to the ledger format.

Day-ahead models:  lightgbm, timesfm, timemixer   (3 models × 24 = 72 rows)
Real-time models:  timesfm, sgdfnet, timemixer, rt916  (4 models × 24 = 96 rows)

Phase 1 only: no validation, no weight learning. Just predictions.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from pipelines.prediction_ledger import (
    append_predictions_to_ledger,
    update_actual_ledger,
)
from runtime.resource_scheduler import (
    ResourceScheduler,
    ScheduleTask,
    ScheduleResult,
)
from utils.business_day import (
    standardize_business_columns,
    validate_daily_predictions,
    infer_period,
    business_day_from_timestamp,
    hour_business_from_timestamp,
    business_period_from_timestamp,
    business_day_res,
)

logger = logging.getLogger(__name__)

# Model sets
DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
REALTIME_MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]


# ===========================================================================
# Main entry point
# ===========================================================================

def run_ledger_predict(args: Any) -> dict:
    """
    Main entry for --pipeline ledger_predict.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: date, data_path, output_root (optional),
        ledger_root, runs_root, max_cpu_workers, max_gpu_workers,
        allow_missing_models, force, realtime_cutoff_hour,
        epf_v1_mode. epf_v1_root is optional (legacy compatibility).

    Returns
    -------
    dict with manifest-like status.
    """
    target_date = args.date
    if not target_date:
        raise ValueError("--date is required for ledger_predict")

    from utils.resolution import resolve_resolution
    res = resolve_resolution(getattr(args, "resolution", "hourly"))
    data_path = args.data_path
    epf_root = getattr(args, "epf_v1_root", None)
    allow_v2_fb = getattr(args, "allow_v2_fallback", False)
    epf_v1_mode = getattr(args, "epf_v1_mode", "exact")
    # 96 点用独立 ledger_96/runs_96；24 点保持 outputs/ledger + outputs/runs
    default_ledger = "outputs/ledger_96" if res.label == "15min" else "outputs/ledger"
    default_runs = "outputs/runs_96" if res.label == "15min" else "outputs/runs"
    ledger_root = Path(getattr(args, "ledger_root", default_ledger))
    runs_root = Path(getattr(args, "runs_root", default_runs))
    max_cpu = getattr(args, "max_cpu_workers", 2)
    max_gpu = getattr(args, "max_gpu_workers", 1)
    allow_missing = getattr(args, "allow_missing_models", False)
    force = getattr(args, "force", False)
    rt_cutoff_hour = getattr(args, "realtime_cutoff_hour", 14)

    # Read model tuning parameters from args
    training_months = getattr(args, "training_months", 12)
    val_ratio = getattr(args, "val_ratio", 0.2)
    timemixer_epochs = getattr(args, "timemixer_epochs", 80)
    timemixer_patience = getattr(args, "timemixer_patience", 15)
    timemixer_batch_size = getattr(args, "timemixer_batch_size", 16)
    timemixer_full_refit = getattr(args, "timemixer_full_refit", True)
    timemixer_seeds = getattr(args, "timemixer_seeds", 42)
    seed = getattr(args, "seed", 42)
    deterministic = getattr(args, "deterministic", False)
    realtime_cutoff_hour = getattr(args, "realtime_cutoff_hour", 14)

    # Validate EPF v1 root — optional; warn if provided but missing
    if epf_root and not Path(epf_root).exists():
        logger.warning(
            "Provided --epf-v1-root does not exist: %s. "
            "Ignoring it and using local bundled implementations.",
            epf_root,
        )
        epf_root = None

    # Validate bundled implementations are available
    required_local_paths = [
        Path("lightGBM"),
        Path("TimesFMBackend"),
    ]
    missing = [str(p) for p in required_local_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing bundled model directories required for ledger pipeline: "
            + ", ".join(missing)
        )

    # Setup directories
    run_dir = runs_root / target_date
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging for this run
    _setup_run_logging(logs_dir)

    logger.info(f"=== ledger_predict: {target_date} ===")

    manifest = {
        "pipeline": "ledger_predict",
        "target_date": target_date,
        "realtime_cutoff_hour": rt_cutoff_hour,
        "seed": seed,
        "deterministic": deterministic,
        "epf_v1_mode": epf_v1_mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "results": {},
        "warnings": [],
        "errors": [],
    }

    # Determine cutoffs
    da_cutoff_date = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rt_cutoff_date = f"{da_cutoff_date} {rt_cutoff_hour:02d}:00:00"
    manifest["data_cutoff_dayahead"] = da_cutoff_date
    manifest["data_cutoff_realtime"] = rt_cutoff_date

    # Add model_runtime_config to manifest
    manifest["model_runtime_config"] = {
        "timemixer": {
            "cutoff_hour_rt": rt_cutoff_hour,
            "epochs": timemixer_epochs,
            "patience": timemixer_patience,
            "batch_size": timemixer_batch_size,
            "full_refit": timemixer_full_refit,
            "seed": seed,
            "legacy_timemixer_seeds": timemixer_seeds,
            "deterministic": deterministic,
        },
        "rt916": {
            "asof_hour": rt_cutoff_hour,
            "seed": seed,
            "deterministic": deterministic,
        },
        "sgdfnet": {
            "decision_hour": rt_cutoff_hour,
            "seed": seed,
            "deterministic": deterministic,
        },
        "timesfm": {
            "device": "cpu",
            "epf_v1_mode": epf_v1_mode,
            "seed": seed,
            "deterministic": deterministic,
        },
        "lightgbm": {
            "epf_v1_mode": epf_v1_mode,
            "seed": seed,
            "deterministic": deterministic,
        },
    }
    manifest["model_source"] = {
        "lightgbm": "bundled:lightGBM/",
        "timesfm": "bundled:TimesFMBackend/",
        "timemixer": "bundled:TimeMixer/",
        "sgdfnet": "bundled:SGDFNet/",
        "rt916": "bundled:RT916_SpikeFusionNet/",
        "external_epf_v1_root": str(epf_root) if epf_root else None,
    }

    try:
        # --- Dayahead predictions (must run BEFORE realtime) ---
        logger.info("\n>>> Dayahead models starting...")
        da_results = _run_model_set(
            target_date=target_date,
            task="dayahead",
            models=DAYAHEAD_MODELS,
            data_path=data_path,
            epf_root=epf_root,
            allow_v2_fallback=allow_v2_fb,
            epf_v1_mode=epf_v1_mode,
            cutoff_date=da_cutoff_date,
            realtime_cutoff_hour=rt_cutoff_hour,
            training_months=training_months,
            val_ratio=val_ratio,
            timemixer_epochs=timemixer_epochs,
            timemixer_patience=timemixer_patience,
            timemixer_batch_size=timemixer_batch_size,
            timemixer_full_refit=timemixer_full_refit,
            timemixer_seeds=timemixer_seeds,
            seed=seed,
            deterministic=deterministic,
            resolution=res.label,
            run_dir=run_dir,
            max_cpu=max_cpu,
            max_gpu=max_gpu,
            force=force,
        )
        manifest["results"]["dayahead"] = da_results

        # Write dayahead long table immediately
        _write_long_table_single(run_dir, target_date, "dayahead", manifest, resolution=res)

        # --- Realtime predictions (after dayahead complete) ---
        logger.info("\n>>> Realtime models starting...")
        rt_results = _run_model_set(
            target_date=target_date,
            task="realtime",
            models=REALTIME_MODELS,
            data_path=data_path,
            epf_root=epf_root,
            allow_v2_fallback=allow_v2_fb,
            epf_v1_mode=epf_v1_mode,
            cutoff_date=rt_cutoff_date,
            realtime_cutoff_hour=rt_cutoff_hour,
            training_months=training_months,
            val_ratio=val_ratio,
            timemixer_epochs=timemixer_epochs,
            timemixer_patience=timemixer_patience,
            timemixer_batch_size=timemixer_batch_size,
            timemixer_full_refit=timemixer_full_refit,
            timemixer_seeds=timemixer_seeds,
            seed=seed,
            deterministic=deterministic,
            resolution=res.label,
            run_dir=run_dir,
            max_cpu=max_cpu,
            max_gpu=max_gpu,
            force=force,
        )
        manifest["results"]["realtime"] = rt_results

        # Write realtime long table
        _write_long_table_single(run_dir, target_date, "realtime", manifest, resolution=res)

        # --- Append to prediction ledger ---
        _append_all_to_ledger(run_dir, target_date, ledger_root, manifest)

        # --- Extract and update actual ledger ---
        _extract_actuals(data_path, target_date, ledger_root, manifest, resolution=res)

        # --- Validate final status ---
        manifest = _finalize_manifest(manifest, allow_missing)

        logger.info(f"ledger_predict {target_date}: {manifest['status']}")

    except Exception as e:
        manifest["status"] = "error"
        manifest["errors"].append(str(e))
        logger.exception(f"ledger_predict failed: {e}")

    # Write manifest
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    return manifest


# ===========================================================================
# Model execution
# ===========================================================================

def _run_model_set(
    target_date: str,
    task: str,
    models: list[str],
    data_path: str,
    epf_root: Optional[str],
    allow_v2_fallback: bool,
    epf_v1_mode: str,
    cutoff_date: str,
    realtime_cutoff_hour: int,
    training_months: int = 12,
    val_ratio: float = 0.2,
    timemixer_epochs: int = 80,
    timemixer_patience: int = 15,
    timemixer_batch_size: int = 16,
    timemixer_full_refit: bool = True,
    timemixer_seeds: int = 42,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
    run_dir: Path = None,
    max_cpu: int = 2,
    max_gpu: int = 1,
    force: bool = False,
) -> dict:
    """Run all models for a given task (dayahead or realtime)."""
    results = {}

    # Build tasks
    tasks: list[ScheduleTask] = []
    for model_name in models:
        pred_dir = run_dir / task / "prediction"
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Check cache
        output_path = pred_dir / f"{model_name}_predictions.csv"
        if output_path.exists() and not force:
            logger.info(f"[{task}/{model_name}] Cache hit: {output_path}")
            try:
                cached_df = pd.read_csv(output_path)
                # Validate cached output
                errors = validate_daily_predictions(cached_df, target_date, model_name, task, resolution=resolution)
                if not errors:
                    results[model_name] = {
                        "status": "cached",
                        "output_path": str(output_path),
                        "rows": len(cached_df),
                    }
                    continue
                else:
                    logger.warning(f"Cache invalid for {model_name}, re-running: {errors}")
            except Exception:
                logger.warning(f"Cache read failed, re-running {model_name}")

        # Create task — pass ALL model params through kwargs
        task_spec = ScheduleTask(
            model_name=model_name,
            task_name=task,
            target_date=target_date,
            fn=_predict_model,
            kwargs={
                "model_name": model_name,
                "task": task,
                "target_date": target_date,
                "data_path": data_path,
                "epf_root": epf_root,
                "allow_v2_fallback": allow_v2_fallback,
                "epf_v1_mode": epf_v1_mode,
                "cutoff_date": cutoff_date,
                "realtime_cutoff_hour": realtime_cutoff_hour,
                "training_months": training_months,
                "val_ratio": val_ratio,
                "timemixer_epochs": timemixer_epochs,
                "timemixer_patience": timemixer_patience,
                "timemixer_batch_size": timemixer_batch_size,
                "timemixer_full_refit": timemixer_full_refit,
                "timemixer_seeds": timemixer_seeds,
                "seed": seed,
                "deterministic": deterministic,
                "resolution": resolution,
                "output_path": str(output_path),
            },
        )
        tasks.append(task_spec)

    if not tasks:
        logger.info(f"[{task}] All models cached, nothing to run")
        return results

    # Run through scheduler
    scheduler = ResourceScheduler(
        max_cpu_workers=max_cpu,
        max_gpu_workers=max_gpu,
    )
    schedule_results = scheduler.run(tasks)

    for sr in schedule_results:
        if sr.success:
            results[sr.model_name] = {
                "status": "ok",
                "output_path": str(run_dir / task / "prediction" / f"{sr.model_name}_predictions.csv"),
                "elapsed_seconds": sr.elapsed_seconds,
            }
        else:
            results[sr.model_name] = {
                "status": "failed",
                "error": sr.error,
                "elapsed_seconds": sr.elapsed_seconds,
            }

    return results


def _predict_model(
    model_name: str,
    task: str,
    target_date: str,
    data_path: str,
    epf_root: Optional[str],
    allow_v2_fallback: bool,
    epf_v1_mode: str,
    cutoff_date: str,
    realtime_cutoff_hour: int,
    training_months: int = 12,
    val_ratio: float = 0.2,
    timemixer_epochs: int = 80,
    timemixer_patience: int = 15,
    timemixer_batch_size: int = 16,
    timemixer_full_refit: bool = True,
    timemixer_seeds: int = 42,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
    output_path: str = "",
) -> pd.DataFrame:
    """
    Run a single model prediction and save to CSV.
    Fails fast if validation errors are detected.
    """
    logger.info(f"Predicting: {model_name}/{task} on {target_date} (res={resolution})")

    if model_name == "lightgbm":
        df = _predict_lightgbm(task, target_date, data_path, epf_root, allow_v2_fallback, epf_v1_mode, cutoff_date, seed=seed, deterministic=deterministic, resolution=resolution)
    elif model_name == "timesfm":
        df = _predict_timesfm(task, target_date, data_path, epf_root, allow_v2_fallback, epf_v1_mode, cutoff_date, seed=seed, deterministic=deterministic, resolution=resolution)
    elif model_name == "timemixer":
        df = _predict_timemixer(
            task, target_date, data_path, cutoff_date,
            realtime_cutoff_hour, training_months, val_ratio,
            timemixer_epochs, timemixer_patience, timemixer_batch_size,
            timemixer_full_refit, timemixer_seeds,
            seed=seed, deterministic=deterministic, resolution=resolution,
        )
    elif model_name == "sgdfnet":
        df = _predict_sgdfnet(
            task, target_date, data_path, cutoff_date, realtime_cutoff_hour,
            seed=seed, deterministic=deterministic, resolution=resolution,
        )
    elif model_name == "rt916":
        df = _predict_rt916(
            task, target_date, data_path, cutoff_date,
            realtime_cutoff_hour, training_months,
            seed=seed, deterministic=deterministic, resolution=resolution,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Validate — FAIL FAST on errors
    errors = validate_daily_predictions(df, target_date, model_name, task, resolution=resolution)
    # Add additional checks
    if df["y_pred"].isna().all():
        errors.append(f"{model_name}/{task}: all y_pred values are NaN")
    if "business_day" in df.columns and (df["business_day"] != target_date).any():
        errors.append(f"{model_name}/{task}: business_day mismatch for target {target_date}")

    if errors:
        err_msg = f"Validation FAILED for {model_name}/{task} on {target_date}: {'; '.join(errors)}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Save
    df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path} ({len(df)} rows)")

    return df


# ===========================================================================
# Per-model prediction implementations
# ===========================================================================

def _predict_lightgbm(
    task: str,
    target_date: str,
    data_path: str,
    epf_root: Optional[str],
    allow_v2_fallback: bool,
    epf_v1_mode: str,
    cutoff_date: str,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
) -> pd.DataFrame:
    """LightGBM prediction via bundled adapter (local lightGBM/ by default)."""
    from runners.adapters.lightgbm_v1 import LightGBMV1Adapter
    adapter = LightGBMV1Adapter(epf_root=epf_root, mode=epf_v1_mode)
    return adapter.predict(
        target_date=target_date,
        target=task,
        data_path=data_path,
        cutoff_date=cutoff_date,
        seed=seed,
        deterministic=deterministic,
        resolution=resolution,
    )


def _predict_timesfm(
    task: str,
    target_date: str,
    data_path: str,
    epf_root: Optional[str],
    allow_v2_fallback: bool,
    epf_v1_mode: str,
    cutoff_date: str,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
) -> pd.DataFrame:
    """TimesFM prediction via bundled adapter (local TimesFMBackend/ by default)."""
    from runners.adapters.timesfm_v1 import TimesFMV1Adapter
    adapter = TimesFMV1Adapter(epf_root=epf_root, mode=epf_v1_mode)
    return adapter.predict(
        target_date=target_date,
        target=task,
        data_path=data_path,
        cutoff_date=cutoff_date,
        seed=seed,
        deterministic=deterministic,
        resolution=resolution,
    )


def _predict_timemixer(
    task: str,
    target_date: str,
    data_path: str,
    cutoff_date: str,
    realtime_cutoff_hour: int = 14,
    training_months: int = 12,
    val_ratio: float = 0.2,
    timemixer_epochs: int = 80,
    timemixer_patience: int = 15,
    timemixer_batch_size: int = 16,
    timemixer_full_refit: bool = True,
    timemixer_seeds: int = 42,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
) -> pd.DataFrame:
    """TimeMixer prediction using 2.0 model (GPU preferred)."""
    return _predict_via_registry(
        "timemixer", task, target_date, data_path, cutoff_date,
        realtime_cutoff_hour=realtime_cutoff_hour,
        training_months=training_months,
        val_ratio=val_ratio,
        timemixer_epochs=timemixer_epochs,
        timemixer_patience=timemixer_patience,
        timemixer_batch_size=timemixer_batch_size,
        timemixer_full_refit=timemixer_full_refit,
        timemixer_seeds=timemixer_seeds,
        seed=seed,
        deterministic=deterministic,
        resolution=resolution,
    )


def _predict_sgdfnet(
    task: str,
    target_date: str,
    data_path: str,
    cutoff_date: str,
    realtime_cutoff_hour: int = 14,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
) -> pd.DataFrame:
    """SGDFNet prediction using 2.0 model (CPU)."""
    return _predict_via_registry(
        "sgdfnet", task, target_date, data_path, cutoff_date,
        realtime_cutoff_hour=realtime_cutoff_hour,
        seed=seed,
        deterministic=deterministic,
        resolution=resolution,
    )


def _predict_rt916(
    task: str,
    target_date: str,
    data_path: str,
    cutoff_date: str,
    realtime_cutoff_hour: int = 14,
    training_months: int = 12,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
) -> pd.DataFrame:
    """RT916 prediction using 2.0 model (GPU)."""
    return _predict_via_registry(
        "rt916", task, target_date, data_path, cutoff_date,
        realtime_cutoff_hour=realtime_cutoff_hour,
        training_months=training_months,
        seed=seed,
        deterministic=deterministic,
        resolution=resolution,
    )


def _predict_via_registry(
    model_name: str,
    task: str,
    target_date: str,
    data_path: str,
    cutoff_date: str,
    realtime_cutoff_hour: int = 14,
    training_months: int = 12,
    val_ratio: float = 0.2,
    timemixer_epochs: int = 80,
    timemixer_patience: int = 15,
    timemixer_batch_size: int = 16,
    timemixer_full_refit: bool = True,
    timemixer_seeds: int = 42,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
) -> pd.DataFrame:
    """
    Run prediction via the existing 2.0 model registry.

    ALL model tuning parameters are forwarded to pipeline.predict_range()
    so that realtime_cutoff_hour, timemixer-*, resolution etc. actually reach the model.
    """
    from runners.registry import get_model_pipeline

    pipeline = get_model_pipeline(model_name)

    result = pipeline.predict_range(
        target=task,
        data_path=data_path,
        predict_date=target_date,
        start=target_date,
        end=target_date,
        resolution=resolution,
        # Forward ALL tuning parameters
        realtime_cutoff_hour=realtime_cutoff_hour,
        cutoff_date=cutoff_date,
        training_months=training_months,
        val_ratio=val_ratio,
        timemixer_epochs=timemixer_epochs,
        timemixer_patience=timemixer_patience,
        timemixer_batch_size=timemixer_batch_size,
        timemixer_full_refit=timemixer_full_refit,
        timemixer_seeds=timemixer_seeds,
        # Generic reproducibility pass-through
        seed=seed,
        deterministic=deterministic,
    )

    if result is None or result.frame is None:
        raise RuntimeError(f"{model_name}/{task} returned None")

    df = result.frame.copy()

    # Record da_feature_source for realtime models
    da_source = _get_da_feature_source(model_name, task)

    # Standardize
    df = standardize_business_columns(
        df,
        ds_col="时刻",
        task_label=task,
        model_name=model_name,
        forecast_date=target_date,
        target_day=target_date,
        data_cutoff=cutoff_date,
        run_id=f"{model_name}_v2_{target_date}",
        model_version="v2.0",
        resolution=resolution,
    )

    # Add da_feature_source if available
    if da_source:
        df["da_feature_source"] = da_source

    # Keep required columns（96 点含 business_period，账本去重键与校验依赖它）
    keep_cols = [
        "task", "model_name", "forecast_date", "target_day",
        "business_day", "ds", "business_period", "hour_business", "period", "y_pred",
        "data_cutoff", "run_id", "model_version", "da_feature_source",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def _get_da_feature_source(model_name: str, task: str) -> str:
    """Return the day-ahead feature source string for a given model+task."""
    if task == "dayahead":
        return "none"  # dayahead models don't use DA features
    return {
        "timemixer": "timemixer_internal_dayahead_prediction",
        "rt916": "rt916_internal_joint_dayahead_prediction",
        "sgdfnet": "sgdfnet_config_da_fill",
        "timesfm": "timesfm_none",
    }.get(model_name, "unknown")


def _write_long_table_single(
    run_dir: Path,
    target_date: str,
    task: str,
    manifest: dict,
    resolution=None,
):
    """Write all_model_predictions_long.csv for a single task.

    期望行数 = 模型数 × 每日本槽数：24 点 DA 3×24=72 / RT 4×24=96；
    96 点 DA 3×96=288 / RT 4×96=384（不写死 "96"=RT 4×24 的旧陷阱）。
    """
    from utils.resolution import HOURLY

    _res = resolution or HOURLY
    pred_dir = run_dir / task / "prediction"
    if not pred_dir.exists():
        return

    pieces = []
    for csv_file in sorted(pred_dir.glob("*_predictions.csv")):
        if csv_file.name == "all_model_predictions_long.csv":
            continue
        try:
            df = pd.read_csv(csv_file)
            pieces.append(df)
        except Exception as e:
            manifest["warnings"].append(f"Failed to read {csv_file}: {e}")

    if pieces:
        long_df = pd.concat(pieces, ignore_index=True)
        long_path = pred_dir / "all_model_predictions_long.csv"
        long_df.to_csv(long_path, index=False)

        n_rows = len(long_df)
        n_models = {"dayahead": len(DAYAHEAD_MODELS), "realtime": len(REALTIME_MODELS)}[task]
        expected = n_models * _res.slots_per_day
        manifest["results"][f"{task}_long_rows"] = n_rows
        if n_rows != expected:
            manifest["warnings"].append(
                f"{task} long table: expected {expected} rows, got {n_rows}"
            )
        logger.info(f"{task} long table: {n_rows} rows → {long_path}")


# ===========================================================================
# Output aggregation
# ===========================================================================


def _append_all_to_ledger(
    run_dir: Path,
    target_date: str,
    ledger_root: Path,
    manifest: dict,
):
    """Append all predictions to the prediction ledger."""
    for task in ["dayahead", "realtime"]:
        long_path = run_dir / task / "prediction" / "all_model_predictions_long.csv"
        if not long_path.exists():
            manifest["warnings"].append(f"No long table for {task}, skipping ledger append")
            continue

        df = pd.read_csv(long_path)
        result = append_predictions_to_ledger(
            df=df,
            ledger_root=ledger_root,
            task=task,
            source_file=str(long_path),
        )
        manifest["results"][f"{task}_ledger"] = result


def _extract_actuals(
    data_path: str,
    target_date: str,
    ledger_root: Path,
    manifest: dict,
    resolution=None,
):
    """
    Extract actual prices from the raw data file for target_date
    and append to the actual ledger.
    """
    from utils.resolution import HOURLY

    _res = resolution or HOURLY
    if not data_path or not Path(data_path).exists():
        manifest["warnings"].append(f"Data file not found: {data_path}")
        return

    try:
        ext = os.path.splitext(data_path)[1].lower()
        if ext in (".xlsx", ".xls"):
            raw = pd.read_excel(data_path)
        else:
            raw = pd.read_csv(data_path)

        # Find timestamp column
        ts_col = None
        for c in ["时刻", "ds", "timestamp", "time", "datetime"]:
            if c in raw.columns:
                ts_col = c
                break

        if ts_col is None:
            manifest["warnings"].append("No timestamp column in data file")
            return

        raw["ds"] = pd.to_datetime(raw[ts_col], errors="coerce")

        # Filter to target_date's business hours
        target_dt = pd.Timestamp(target_date)
        # Business day D spans D 01:00..D+1 00:00（24 点）/ D 00:15..D+1 00:00（96 点）
        if _res.label == "15min":
            start_ts = target_dt.replace(hour=0, minute=15, second=0)
        else:
            start_ts = target_dt.replace(hour=1, minute=0, second=0)
        end_ts = (target_dt + pd.Timedelta(days=1)).replace(hour=0, minute=0, second=0)

        mask = (raw["ds"] >= start_ts) & (raw["ds"] <= end_ts)
        day_data = raw[mask].copy()

        if len(day_data) == 0:
            manifest["warnings"].append(f"No actual data for {target_date}")
            return

        logger.info(f"Extracted {len(day_data)} actual rows for {target_date}")

        # Standardize（96 点加 business_period 列）
        if _res.label == "15min":
            day_data["business_period"] = day_data["ds"].apply(
                lambda ts: business_period_from_timestamp(ts, _res)
            )
            day_data["hour_business"] = (
                (day_data["business_period"].astype(int) - 1) // (_res.slots_per_day // 24) + 1
            ).astype(int)
            day_data["period"] = day_data["business_period"].apply(
                lambda p: infer_period(int(p), _res)
            )
            day_data["business_day"] = day_data["ds"].apply(lambda ts: business_day_res(ts, _res))
        else:
            day_data["business_day"] = day_data["ds"].apply(business_day_from_timestamp)
            day_data["hour_business"] = day_data["ds"].apply(hour_business_from_timestamp)
            day_data["period"] = day_data["hour_business"].apply(infer_period)

        # Find actual price columns with extended aliases
        dayahead_aliases = [
            "日前电价", "日前出清电价", "day_ahead_clearing_price",
            "dayahead_price", "da_price",
        ]
        realtime_aliases = [
            "实时电价", "realtime_price", "rt_price",
        ]

        for task, col_names in [
            ("dayahead", dayahead_aliases),
            ("realtime", realtime_aliases),
        ]:
            y_col = None
            for cn in col_names:
                if cn in day_data.columns:
                    y_col = cn
                    break

            if y_col is None:
                manifest["warnings"].append(
                    f"Actual column not found for {task}. Tried: {col_names}"
                )
                continue

            # 96 点 actual 也带 business_period，保证账本去重键（含 period）不塌缩
            act_cols = ["ds", "business_day", "hour_business", "period", y_col]
            if "business_period" in day_data.columns:
                act_cols.append("business_period")
            act_df = day_data[act_cols].copy()
            act_df["y_true"] = pd.to_numeric(day_data[y_col], errors="coerce")
            act_df["task"] = task
            act_df["target_day"] = target_date

            act_df = act_df.dropna(subset=["y_true"])

            result = update_actual_ledger(
                df=act_df,
                ledger_root=ledger_root,
                task=task,
                source_file=data_path,
            )
            manifest["results"][f"{task}_actual_ledger"] = result

    except Exception as e:
        manifest["warnings"].append(f"Actual extraction failed: {e}")
        logger.warning(f"Actual extraction error: {e}")


def _finalize_manifest(manifest: dict, allow_missing: bool) -> dict:
    """Determine final status and add completion timestamp."""
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()

    errors = manifest.get("errors", [])
    warnings = manifest.get("warnings", [])

    # Check model failures
    failed_models = []
    for task in ["dayahead", "realtime"]:
        task_results = manifest.get("results", {}).get(task, {})
        for model, info in task_results.items():
            if isinstance(info, dict) and info.get("status") == "failed":
                failed_models.append(f"{task}/{model}")

    if failed_models:
        manifest["failed_models"] = failed_models
        if allow_missing:
            manifest["status"] = "complete_with_warnings"
            warnings.append(f"Missing models: {failed_models}")
        else:
            manifest["status"] = "failed"
            errors.append(f"Required models failed: {failed_models}")
    elif errors:
        manifest["status"] = "failed"
    elif warnings:
        manifest["status"] = "complete_with_warnings"
    else:
        manifest["status"] = "complete"

    return manifest


def _setup_run_logging(logs_dir: Path):
    """Add a file handler for this pipeline run."""
    handler = logging.FileHandler(logs_dir / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(handler)
