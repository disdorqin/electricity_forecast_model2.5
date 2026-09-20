"""
Ledger predict pipeline.

Runs all models for a single target day D, producing 24-hour predictions
per model, standardized to the ledger format.

Day-ahead models:  lightgbm, timesfm, timemixer
Real-time models:  timesfm, sgdfnet, timemixer, rt916

Phase 1 only: no validation, no weight learning. Just predictions.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
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
    classify_model_device,
)
from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS, tasks_for_target
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


def _resolve_requested_models(value: str | None) -> list[str] | None:
    """Parse an optional smoke/debug subset without redefining the pool."""
    if value is None or str(value).strip().lower() in {"", "all"}:
        return None
    return [name.strip().lower() for name in str(value).split(",") if name.strip()]


def _select_models(pool: tuple[str, ...], requested: list[str] | None) -> tuple[str, ...]:
    """Select from the canonical pool; never add an ad-hoc production model."""
    if requested is None:
        return pool
    unknown = sorted(set(requested) - set(DAYAHEAD_MODELS) - set(REALTIME_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}; canonical pool is managed in fusion/model_pool.py")
    return tuple(name for name in pool if name in requested)


def _resolve_runtime_root(
    runs_root: Path,
    *,
    resolution_label: str,
    output_profile: str,
    domain: str,
) -> Path:
    """Resolve scratch ownership without adding another public CLI option."""
    if resolution_label == "15min" and output_profile == "production":
        return runs_root.parent / "runtime"
    return Path("outputs") / domain / "runtime"


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
    source_data_path = data_path
    actual_data_path = getattr(args, "actual_data_path", None) or data_path
    epf_root = getattr(args, "epf_v1_root", None)
    allow_v2_fb = getattr(args, "allow_v2_fallback", False)
    epf_v1_mode = getattr(args, "epf_v1_mode", "exact")
    # Production defaults are domain-scoped. main.py normally resolves these
    # through utils.output_layout; these fallbacks keep direct API calls safe.
    domain = "96" if res.label == "15min" else "24"
    default_ledger = "outputs/96/ledger" if res.label == "15min" else "outputs/ledger"
    default_runs = "outputs/96/runs" if res.label == "15min" else "outputs/runs"
    ledger_root = Path(getattr(args, "ledger_root", None) or default_ledger)
    runs_root = Path(getattr(args, "runs_root", None) or default_runs)
    output_profile = str(getattr(args, "output_profile", "production"))
    formal_96_output = res.label == "15min" and output_profile == "production"
    max_cpu = getattr(args, "max_cpu_workers", 2)
    max_gpu = getattr(args, "max_gpu_workers", 1)
    allow_missing = getattr(args, "allow_missing_models", False)
    force = getattr(args, "force", False)
    rt_cutoff_hour = getattr(
        args, "realtime_cutoff_hour", 15 if res.label == "15min" else 14
    )

    # Read model tuning parameters from args
    training_months = getattr(args, "training_months", 12)
    lgbm_training_months_candidates = getattr(args, "lgbm_training_months_candidates", None)
    lgbm_window_selection_metric = getattr(args, "lgbm_window_selection_metric", "smape")
    lgbm_window_mae_weight = getattr(args, "lgbm_window_mae_weight", 0.25)
    val_ratio = getattr(args, "val_ratio", 0.2)
    timemixer_epochs = getattr(args, "timemixer_epochs", 80)
    timemixer_patience = getattr(args, "timemixer_patience", 15)
    timemixer_batch_size = getattr(args, "timemixer_batch_size", 16)
    timemixer_full_refit = getattr(args, "timemixer_full_refit", True)
    timemixer_seeds = getattr(args, "timemixer_seeds", 42)
    seed = getattr(args, "seed", 42)
    deterministic = getattr(args, "deterministic", False)
    realtime_cutoff_hour = getattr(
        args, "realtime_cutoff_hour", 15 if res.label == "15min" else 14
    )

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
        Path("models/timesFM/config.json"),
        Path("models/timesFM/model.safetensors"),
    ]
    missing = [str(p) for p in required_local_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model/code assets for ledger pipeline: "
            + ", ".join(missing)
            + ". Static TimesFM weights are deployment artifacts and are not "
              "assumed to be present after a bare git clone."
        )

    # Setup directories
    run_dir = runs_root / target_date
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Model-internal outputs are scratch. Formal96 derives runtime as the
    # sibling of the already-resolved runs root, so an isolated/custom runs
    # root automatically isolates scratch as well. 24/legacy retains the
    # historical domain runtime path.
    runtime_root = _resolve_runtime_root(
        runs_root,
        resolution_label=res.label,
        output_profile=output_profile,
        domain=domain,
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    stale_before = time.time() - 24 * 3600
    stale_patterns = ("attempt_*", "models_*") if formal_96_output else ("models_*",)
    for pattern in stale_patterns:
        for stale_dir in runtime_root.glob(pattern):
            try:
                if stale_dir.is_dir() and stale_dir.stat().st_mtime < stale_before:
                    shutil.rmtree(stale_dir, ignore_errors=True)
            except OSError:
                pass

    if formal_96_output:
        attempt_id = str(
            getattr(args, "_attempt_id", None)
            or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}"
        )
        attempt_root = (
            runtime_root / f"attempt_{target_date.replace('-', '')}_{attempt_id}"
        ).resolve()
        model_runtime_dir = attempt_root / "models"
        setattr(args, "_runtime_attempt_root", str(attempt_root))
    else:
        attempt_root = None
        model_runtime_dir = (
            runtime_root / f"models_{target_date.replace('-', '')}_{os.getpid()}"
        ).resolve()

    # Setup file logging for this run
    _setup_run_logging(logs_dir)

    logger.info(f"=== ledger_predict: {target_date} ===")

    manifest = {
        "pipeline": "ledger_predict",
        "target_date": target_date,
        # Keep resolution at the manifest top level so range-level audits can
        # reject mixed 24/96 artifacts before reading model outputs.
        "resolution": res.label,
        "realtime_cutoff_hour": "dynamic_snapshot" if formal_96_output else rt_cutoff_hour,
        "seed": seed,
        "deterministic": deterministic,
        "epf_v1_mode": epf_v1_mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "results": {},
        "warnings": [],
        "errors": [],
    }
    manifest["output_profile"] = output_profile
    manifest["output_roots"] = {
        "ledger_root": str(ledger_root),
        "runs_root": str(runs_root),
        "feature_store_root": str(
            getattr(args, "feature_store_root", None) or f"outputs/{domain}/cache"
        ),
        "runtime_root": str(runtime_root),
        "runtime_attempt_root": str(attempt_root) if attempt_root is not None else None,
    }
    manifest["actual_source"] = {
        "path": str(actual_data_path),
        "separate_from_model_source": str(actual_data_path) != str(source_data_path),
    }
    manifest["model_pool"] = {
        "dayahead": list(DAYAHEAD_MODELS),
        "realtime": list(REALTIME_MODELS),
    }
    requested_models = _resolve_requested_models(getattr(args, "models", "all"))
    requested_tasks = tasks_for_target(getattr(args, "target", "both"))
    manifest["requested_models"] = requested_models or "all"
    manifest["requested_tasks"] = list(requested_tasks)
    resource_mode = getattr(args, "resource_mode", "legacy")
    if resource_mode == "split_process" and res.label != "15min":
        raise ValueError("--resource-mode split_process is currently enabled only for 96-point runs")
    manifest["resource_mode"] = resource_mode
    split_process_96 = res.label == "15min" and resource_mode == "split_process"
    dynamic_96 = formal_96_output
    manifest["scheduler"] = {
        "mode": resource_mode,
        "cpu_workers": 2 if split_process_96 else int(max_cpu),
        "gpu_workers": 1 if split_process_96 else int(max_gpu),
        "cpu_dag_aware": bool(split_process_96),
        "gpu_serial": bool(split_process_96),
    }
    manifest["production_config"] = {
        "formal_96": dynamic_96,
        "rt916_train_steps": 24 if res.label == "15min" else None,
        "rt916_train_steps_role": "production_stride" if res.label == "15min" else "legacy/internal",
        "serving_visibility_source": "FeatureViewBuilder" if dynamic_96 else "model_legacy",
        "serving_cutoff_policy": "dynamic_snapshot" if dynamic_96 else "fixed_legacy_cutoff",
    }
    selected_da_models = (
        _select_models(DAYAHEAD_MODELS, requested_models)
        if "dayahead" in requested_tasks else ()
    )
    selected_rt_models = (
        _select_models(REALTIME_MODELS, requested_models)
        if "realtime" in requested_tasks else ()
    )
    manifest["selected_model_pool"] = {
        "dayahead": list(selected_da_models),
        "realtime": list(selected_rt_models),
    }
    if requested_models and not (selected_da_models or selected_rt_models):
        raise ValueError(f"None of --models={requested_models} is in the canonical production pool")

    # Dynamic-v1 serving boundary: freeze a small D/T snapshot after sync,
    # then route one shared FeatureView for every model leg.  The legacy
    # fixed-cutoff helper remains available to compatibility callers only.
    if dynamic_96:
        from utils.asof_view_96 import (
            build_dynamic_feature_view_96,
            resolve_formal96_snapshot_route,
        )
        # Same-day reruns must never overwrite the snapshot owned by an older
        # successful prediction.  Keep each attempt in its own immutable slot;
        # --finish follows the exact paths captured in Stage1 provenance.
        snapshot_slot = (
            f"attempt_{attempt_id}"
            if formal_96_output
            else f"run_{os.getpid()}"
        )
        snapshot_dir = run_dir / "snapshot" / snapshot_slot
        route_result = resolve_formal96_snapshot_route(
            target_day=target_date,
            model_store_path=source_data_path,
            authoritative_path=actual_data_path,
            output_dir=snapshot_dir,
            run_dir=run_dir,
            latest_closed_day=getattr(args, "_formal96_latest_closed_day", None),
            current_target_day=getattr(args, "_formal96_current_target_day", None),
        )
        feature_view_path = (
            (attempt_root / "feature_view" / "input.parquet")
            if attempt_root is not None
            else (run_dir / "runtime" / "feature_view" / "input.parquet")
        )
        _, feature_audit = build_dynamic_feature_view_96(
            model_store_path=source_data_path,
            snapshot_values=route_result["values_path"],
            snapshot_manifest=route_result["manifest"],
            target_day=target_date,
            output_path=feature_view_path,
        )
        data_path = str(feature_view_path)
        # Propagate the exact same routed view and snapshot provenance to all
        # model legs and later ledger stages.
        setattr(args, "data_path", data_path)
        setattr(args, "full_model_input_path", str(source_data_path))
        setattr(args, "_transient_asof_path", str(feature_view_path))
        setattr(args, "_snapshot_id", route_result["snapshot_id"])
        setattr(args, "_snapshot_dir", str(Path(route_result["values_path"]).parent))
        manifest["serving_protocol"] = route_result["manifest"].get("protocol", FORMAL96_PREDICTION_CONTRACT)
        manifest["snapshot_id"] = route_result["snapshot_id"]
        manifest["dynamic_snapshot"] = route_result["manifest"]
        manifest["snapshot_route"] = route_result["route"]
        manifest["run_mode"] = route_result["run_mode"]
        manifest["snapshot_kind"] = route_result["snapshot_kind"]
        for key in ("proxy_policy_version", "proxy_cutoff_period", "historical_vintage", "strict_historical_vintage_proven"):
            if key in route_result["manifest"]:
                manifest[key] = route_result["manifest"][key]
        manifest["feature_view"] = feature_audit
        manifest["model_input_source"] = str(source_data_path)
        manifest["model_input_effective"] = data_path

    # Prepare the candidate raw cache once per prediction chain.  The default
    # remains off, so the legacy chain is unchanged until the candidate has
    # passed its full-chain gates.
    feature_store_mode = getattr(args, "feature_store_mode", "off")
    if dynamic_96 and feature_store_mode != "off":
        raise ValueError(
            "FORMAL96_DYNAMIC_FEATURE_STORE_FORBIDDEN: "
            f"feature_store_mode={feature_store_mode!r}; "
            "Dynamic-v1 requires the shared FeatureView as the sole serving input"
        )
    feature_view_paths: dict[str, str] = {}
    if feature_store_mode in {"raw", "materialized"}:
        from utils.feature_store import FeatureStore

        feature_store = FeatureStore(
            resolution=res.label,
            source=data_path,
            root=getattr(args, "feature_store_root", None),
        )
        feature_store.ensure()
        if feature_store_mode == "materialized":
            feature_store.ensure_base()
            view_manifest = {}
            for model_name, task_name in [
                *((m, "dayahead") for m in selected_da_models),
                *((m, "realtime") for m in selected_rt_models),
            ]:
                view_manifest[f"{task_name}/{model_name}"] = str(
                    feature_store.ensure_view(model_name, task_name)
                )
            feature_view_paths = view_manifest
            data_path = str(feature_store.base_path)
        else:
            view_manifest = {}
            data_path = str(feature_store.raw_path)
        manifest["feature_store"] = {
            "mode": feature_store_mode,
            "source_path": str(source_data_path),
            "masked_source_path": str(data_path),
            "effective_data_path": data_path,
            "cache_root": str(feature_store.feature_root),
            "cache_dir": str(feature_store.dir),
            "raw_cache_path": str(feature_store.raw_path),
            "feature_matrix_path": str(feature_store.matrix_path),
            "feature_matrix_rows": int(len(feature_store._da)) if feature_store._da is not None else 0,
            "base_path": str(getattr(feature_store, "base_path", feature_store.raw_path)),
            "views": view_manifest,
        }
    else:
        manifest["feature_store"] = {"mode": "off"}

    # Determine cutoffs
    da_cutoff_date = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rt_cutoff_date = f"{da_cutoff_date} {rt_cutoff_hour:02d}:00:00"
    manifest["data_cutoff_dayahead"] = da_cutoff_date
    manifest["data_cutoff_realtime"] = "dynamic_snapshot" if dynamic_96 else rt_cutoff_date

    # Add model_runtime_config to manifest
    manifest["model_runtime_config"] = {
        "timemixer": {
            "cutoff_hour_rt": rt_cutoff_hour,
            "cutoff_role": "training_sample_compat_only" if dynamic_96 else "legacy_serving",
            "serving_visibility_source": "FeatureViewBuilder" if dynamic_96 else "model_local",
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
            "asof_role": "training_window_compat_only" if dynamic_96 else "legacy_serving",
            "serving_visibility_source": "FeatureViewBuilder" if dynamic_96 else "model_local",
            "train_steps": 24 if res.label == "15min" else None,
            "train_steps_role": "production_stride" if res.label == "15min" else "legacy/internal",
            "seed": seed,
            "deterministic": deterministic,
        },
        "sgdfnet": {
            "decision_hour": rt_cutoff_hour,
            "decision_role": "anchor/training_compat_only" if dynamic_96 else "legacy_serving",
            "serving_visibility_source": "FeatureViewBuilder" if dynamic_96 else "model_local",
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
            "training_months_candidates": lgbm_training_months_candidates,
            "window_selection_metric": lgbm_window_selection_metric,
            "window_mae_weight": lgbm_window_mae_weight,
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
        common_predict_kwargs = {
            "data_path": data_path,
            "epf_root": epf_root,
            "allow_v2_fallback": allow_v2_fb,
            "epf_v1_mode": epf_v1_mode,
            "realtime_cutoff_hour": rt_cutoff_hour,
            "training_months": training_months,
            "lgbm_training_months_candidates": lgbm_training_months_candidates,
            "lgbm_window_selection_metric": lgbm_window_selection_metric,
            "lgbm_window_mae_weight": lgbm_window_mae_weight,
            "val_ratio": val_ratio,
            "timemixer_epochs": timemixer_epochs,
            "timemixer_patience": timemixer_patience,
            "timemixer_batch_size": timemixer_batch_size,
            "timemixer_full_refit": timemixer_full_refit,
            "timemixer_seeds": timemixer_seeds,
            "seed": seed,
            "deterministic": deterministic,
            "resolution": res.label,
            "model_output_root": str(model_runtime_dir),
            "production_mode": bool(dynamic_96),
            "dynamic_serving": bool(dynamic_96),
            "snapshot_id": getattr(args, "_snapshot_id", None),
            "serving_protocol": manifest.get("serving_protocol", FORMAL96_PREDICTION_CONTRACT),
            "run_mode": manifest.get("run_mode", "LIVE_DYNAMIC"),
            "snapshot_kind": manifest.get("snapshot_kind", "live"),
            "proxy_policy_version": manifest.get("proxy_policy_version"),
            "proxy_cutoff_period": manifest.get("proxy_cutoff_period"),
            "rt916_train_steps": 24 if res.label == "15min" else None,
        }

        if resource_mode == "split_process":
            logger.info("\n>>> Unified CPU/GPU split-process model DAG starting...")
            unified_results = _run_unified_model_plan(
                target_date=target_date,
                selected_da_models=selected_da_models,
                selected_rt_models=selected_rt_models,
                common_kwargs=common_predict_kwargs,
                feature_view_paths=feature_view_paths,
                da_cutoff_date=da_cutoff_date,
                rt_cutoff_date=rt_cutoff_date,
                run_dir=run_dir,
                max_cpu=max_cpu,
                max_gpu=max_gpu,
                force=force,
            )
            if "dayahead" in requested_tasks:
                manifest["results"]["dayahead"] = unified_results["dayahead"]
            if "realtime" in requested_tasks:
                manifest["results"]["realtime"] = unified_results["realtime"]
            for task_name in requested_tasks:
                if not _result_set_complete(
                    unified_results[task_name],
                    selected_da_models if task_name == "dayahead" else selected_rt_models,
                ):
                    raise RuntimeError(
                        f"{task_name} production model set incomplete; refusing ledger append"
                    )
                _write_long_table_single(run_dir, target_date, task_name, manifest, resolution=res)
        else:
            if "dayahead" in requested_tasks:
                logger.info("\n>>> Dayahead models starting...")
                da_results = _run_model_set(
                    target_date=target_date, task="dayahead", models=selected_da_models,
                    data_path=data_path, epf_root=epf_root,
                    allow_v2_fallback=allow_v2_fb, epf_v1_mode=epf_v1_mode,
                    cutoff_date=da_cutoff_date, realtime_cutoff_hour=rt_cutoff_hour,
                    training_months=training_months, val_ratio=val_ratio,
                    lgbm_training_months_candidates=lgbm_training_months_candidates,
                    lgbm_window_selection_metric=lgbm_window_selection_metric,
                    lgbm_window_mae_weight=lgbm_window_mae_weight,
                    timemixer_epochs=timemixer_epochs, timemixer_patience=timemixer_patience,
                    timemixer_batch_size=timemixer_batch_size,
                    timemixer_full_refit=timemixer_full_refit, timemixer_seeds=timemixer_seeds,
                    seed=seed, deterministic=deterministic, resolution=res.label,
                    run_dir=run_dir, model_output_root=str(model_runtime_dir),
                    max_cpu=max_cpu, max_gpu=max_gpu, force=force,
                    dynamic_serving=bool(dynamic_96), snapshot_id=getattr(args, "_snapshot_id", None),
                    serving_protocol=manifest.get("serving_protocol", FORMAL96_PREDICTION_CONTRACT),
                    run_mode=manifest.get("run_mode", "LIVE_DYNAMIC"),
                    snapshot_kind=manifest.get("snapshot_kind", "live"),
                    proxy_policy_version=manifest.get("proxy_policy_version"),
                    proxy_cutoff_period=manifest.get("proxy_cutoff_period"),
                )
                manifest["results"]["dayahead"] = da_results
                _write_long_table_single(run_dir, target_date, "dayahead", manifest, resolution=res)

            if "realtime" in requested_tasks:
                logger.info("\n>>> Realtime models starting...")
                rt_results = _run_model_set(
                    target_date=target_date, task="realtime", models=selected_rt_models,
                    data_path=data_path, epf_root=epf_root,
                    allow_v2_fallback=allow_v2_fb, epf_v1_mode=epf_v1_mode,
                    cutoff_date=rt_cutoff_date, realtime_cutoff_hour=rt_cutoff_hour,
                    training_months=training_months, val_ratio=val_ratio,
                    lgbm_training_months_candidates=lgbm_training_months_candidates,
                    lgbm_window_selection_metric=lgbm_window_selection_metric,
                    lgbm_window_mae_weight=lgbm_window_mae_weight,
                    timemixer_epochs=timemixer_epochs, timemixer_patience=timemixer_patience,
                    timemixer_batch_size=timemixer_batch_size,
                    timemixer_full_refit=timemixer_full_refit, timemixer_seeds=timemixer_seeds,
                    seed=seed, deterministic=deterministic, resolution=res.label,
                    run_dir=run_dir, model_output_root=str(model_runtime_dir),
                    max_cpu=max_cpu, max_gpu=max_gpu, force=force,
                    dynamic_serving=bool(dynamic_96), snapshot_id=getattr(args, "_snapshot_id", None),
                    serving_protocol=manifest.get("serving_protocol", FORMAL96_PREDICTION_CONTRACT),
                    run_mode=manifest.get("run_mode", "LIVE_DYNAMIC"),
                    snapshot_kind=manifest.get("snapshot_kind", "live"),
                    proxy_policy_version=manifest.get("proxy_policy_version"),
                    proxy_cutoff_period=manifest.get("proxy_cutoff_period"),
                )
                manifest["results"]["realtime"] = rt_results
                _write_long_table_single(run_dir, target_date, "realtime", manifest, resolution=res)

        # --- Append to prediction ledger ---
        _append_all_to_ledger(run_dir, target_date, ledger_root, manifest, tasks=requested_tasks)

        # --- Extract and update actual ledger ---
        _extract_actuals(
            actual_data_path, target_date, ledger_root, manifest,
            resolution=res, tasks=requested_tasks,
        )
        _check_target_actual_readiness(
            manifest=manifest,
            tasks=requested_tasks,
            target_date=target_date,
            slots_per_day=res.slots_per_day,
            require_target_actual=bool(getattr(args, "require_target_actual", False)),
        )

        # --- Validate final status ---
        manifest = _finalize_manifest(manifest, allow_missing)

        logger.info(f"ledger_predict {target_date}: {manifest['status']}")

    except Exception as e:
        manifest["status"] = "error"
        manifest["errors"].append(str(e))
        logger.exception(f"ledger_predict failed: {e}")

    # Scratch lifecycle: a direct formal --predict owns the whole attempt and
    # removes it on success. A ledger_full child keeps only the as-of input
    # until the parent finishes; model scratch is still removed immediately.
    # Failures retain one attempt directory for diagnosis/TTL recovery.
    if manifest.get("status") in {"complete", "complete_with_warnings"}:
        root_owned = bool(getattr(args, "_root_manifest_owned", False))
        cleanup_path = model_runtime_dir
        if formal_96_output and attempt_root is not None and not root_owned:
            cleanup_path = attempt_root
        if cleanup_path.exists():
            shutil.rmtree(cleanup_path, ignore_errors=True)
        manifest["model_runtime_cleanup"] = {
            "path": str(cleanup_path),
            "attempt_root": str(attempt_root) if attempt_root is not None else None,
            "persistent": False,
            "removed": not cleanup_path.exists(),
            "scope": "whole_attempt" if cleanup_path == attempt_root else "model_scratch",
        }
    elif (attempt_root is not None and attempt_root.exists()) or model_runtime_dir.exists():
        retained = attempt_root if attempt_root is not None else model_runtime_dir
        manifest["model_runtime_cleanup"] = {
            "path": str(retained),
            "attempt_root": str(attempt_root) if attempt_root is not None else None,
            "persistent": False,
            "removed": False,
            "reason": "retained_after_failure_for_diagnosis",
        }

    # A formal ledger_full attempt owns the root manifest.  Its prediction
    # child writes a stage-specific manifest so a late child completion cannot
    # resurrect an older root ``status=complete`` record.  Direct --predict
    # callers retain the historical per-run manifest location.
    manifest_path = Path(
        getattr(args, "_stage_manifest_path", None) or (run_dir / "run_manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    return manifest


# ===========================================================================
# Model execution
# ===========================================================================

def _result_set_complete(results: dict, expected_models: tuple[str, ...]) -> bool:
    """Strictly validate one production task's model result set."""
    for model_name in expected_models:
        item = results.get(model_name, {})
        if item.get("status") not in {"ok", "cached"}:
            return False
        output_path = item.get("output_path")
        if not output_path or not Path(output_path).exists():
            return False
    return True


FORMAL96_PREDICTION_CONTRACT = "formal96_dynamic_snapshot_v1"


def _validate_formal96_prediction_cache(
    df: pd.DataFrame,
    *,
    target_date: str,
    model_name: str,
    task: str,
    expected_cutoff: str,
    snapshot_id: str | None = None,
    serving_protocol: str = FORMAL96_PREDICTION_CONTRACT,
    run_mode: str | None = None,
    snapshot_kind: str | None = None,
    proxy_policy_version: str | None = None,
    proxy_cutoff_period: int | None = None,
) -> list[str]:
    """Reject stale or unknown caches before formal 96 ledger append."""
    errors = validate_daily_predictions(
        df, target_date, model_name, task, resolution="15min"
    )
    if "y_pred" not in df.columns or df["y_pred"].isna().any():
        errors.append(f"{task}/{model_name}: cached y_pred contains NaN or is missing")

    expected = {
        "production_contract": serving_protocol,
        "serving_protocol": serving_protocol,
        "production_resolution": "15min",
        "production_resource_mode": "split_process",
        "task": task,
        "model_name": model_name,
        "target_day": target_date,
    }
    if snapshot_id:
        expected["snapshot_id"] = snapshot_id
    if run_mode:
        expected["run_mode"] = run_mode
    if snapshot_kind:
        expected["snapshot_kind"] = snapshot_kind
    if proxy_policy_version:
        expected["proxy_policy_version"] = proxy_policy_version
    if proxy_cutoff_period is not None:
        expected["proxy_cutoff_period"] = proxy_cutoff_period
    for column, value in expected.items():
        if column not in df.columns:
            errors.append(f"{task}/{model_name}: cache missing {column}")
            continue
        observed = {str(v).strip() for v in df[column].dropna().unique()}
        if observed != {str(value)}:
            errors.append(
                f"{task}/{model_name}: cache {column}={sorted(observed)} expected={value}"
            )

    expected_da_source = _get_da_feature_source(model_name, task)
    if "da_feature_source" not in df.columns:
        errors.append(f"{task}/{model_name}: cache missing da_feature_source")
    else:
        observed_source = {
            str(v).strip() for v in df["da_feature_source"].dropna().unique()
        }
        if observed_source != {expected_da_source}:
            errors.append(
                f"{task}/{model_name}: cache da_feature_source={sorted(observed_source)} "
                f"expected={expected_da_source}"
            )

    if model_name == "rt916":
        if "production_rt916_train_steps" not in df.columns:
            errors.append("realtime/rt916: cache missing production_rt916_train_steps")
        else:
            steps = set(
                pd.to_numeric(
                    df["production_rt916_train_steps"], errors="coerce"
                ).dropna().astype(int)
            )
            if steps != {24}:
                errors.append(
                    f"realtime/rt916: cache train_steps={sorted(steps)} expected=24"
                )

    if model_name == "sgdfnet" and task == "realtime":
        source_day = (
            pd.Timestamp(target_date) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        for column, value in {
            "anchor_source_day": source_day,
            "anchor_source_type": "decision_day_da",
        }.items():
            if column not in df.columns:
                errors.append(f"realtime/sgdfnet: cache missing {column}")
                continue
            observed = {
                (str(v).strip()[:10] if column == "anchor_source_day" else str(v).strip())
                for v in df[column].dropna().unique()
            }
            if observed != {value}:
                errors.append(
                    f"realtime/sgdfnet: cache {column}={sorted(observed)} expected={value}"
                )
        if "anchor_rows" not in df.columns:
            errors.append("realtime/sgdfnet: cache missing anchor_rows")
        else:
            rows = set(
                pd.to_numeric(df["anchor_rows"], errors="coerce")
                .dropna()
                .astype(int)
            )
            if rows != {96}:
                errors.append(
                    f"realtime/sgdfnet: cache anchor_rows={sorted(rows)} expected=96"
                )
        if "fallback_used" not in df.columns:
            errors.append("realtime/sgdfnet: cache missing fallback_used")
        else:
            fallback = df["fallback_used"].astype(str).str.strip().str.lower()
            if fallback.isin({"true", "1", "yes"}).any():
                errors.append("realtime/sgdfnet: cache fallback_used=true")

    return errors


def _build_cached_result_payload(
    cached_df: pd.DataFrame,
    *,
    output_path: Path,
    model_name: str,
    task_name: str,
) -> dict:
    """Build manifest metadata for a cache that already passed strict validation."""
    result = {
        "status": "cached",
        "output_path": str(output_path),
        "rows": len(cached_df),
    }
    if model_name == "sgdfnet" and task_name == "realtime":
        result["anchor_contract"] = {
            "anchor_source_day": str(
                cached_df["anchor_source_day"].dropna().iloc[0]
            )[:10],
            "source_type": str(
                cached_df["anchor_source_type"].dropna().iloc[0]
            ),
            "rows": int(
                pd.to_numeric(
                    cached_df["anchor_rows"], errors="coerce"
                ).dropna().iloc[0]
            ),
            "fallback_used": False,
        }
    return result


def _run_unified_model_plan(
    *,
    target_date: str,
    selected_da_models: tuple[str, ...],
    selected_rt_models: tuple[str, ...],
    common_kwargs: dict,
    feature_view_paths: dict[str, str],
    da_cutoff_date: str,
    rt_cutoff_date: str,
    run_dir: Path,
    max_cpu: int,
    max_gpu: int,
    force: bool,
) -> dict[str, dict]:
    """Run the 96-point production DAG using ``(model, task/node)`` identity.

    Realtime-only runs may execute DA prerequisite nodes for TimesFM and
    TimeMixer, but only the explicitly requested RT legs are published to the
    result set and ledger.
    """
    ordered = [
        ("lightgbm", "dayahead", "cpu"),
        ("timesfm", "dayahead", "cpu"),
        ("timesfm", "realtime", "cpu"),
        ("sgdfnet", "anchor_prepare", "cpu"),
        ("sgdfnet", "realtime", "cpu"),
        ("timemixer", "dayahead", "gpu"),
        ("timemixer", "realtime", "gpu"),
        ("rt916", "realtime", "gpu"),
    ]
    direct_nodes = {
        *((model, "dayahead") for model in selected_da_models),
        *((model, "realtime") for model in selected_rt_models),
    }
    # Explicit internal prerequisites.  RT916 performs its own DA→RT work;
    # it must not depend on TimeMixer RT.  SGDFNet's anchor preparation is an
    # internal node, not a fifth public model leg.
    requested_nodes = set(direct_nodes)
    if ("timesfm", "realtime") in requested_nodes:
        requested_nodes.add(("timesfm", "dayahead"))
    if ("timemixer", "realtime") in requested_nodes:
        requested_nodes.add(("timemixer", "dayahead"))
    if ("sgdfnet", "realtime") in requested_nodes:
        requested_nodes.add(("sgdfnet", "anchor_prepare"))
    cpu_tasks: list[ScheduleTask] = []
    gpu_tasks: list[ScheduleTask] = []
    results = {"dayahead": {}, "realtime": {}}
    output_paths: dict[tuple[str, str], Path] = {}
    cached_nodes: set[str] = set()

    for model_name, task_name, _ in ordered:
        node_key = (model_name, task_name)
        if node_key not in requested_nodes:
            continue
        if task_name == "anchor_prepare":
            task_spec = ScheduleTask(
                model_name=model_name,
                task_name=task_name,
                target_date=target_date,
                fn=_prepare_sgdfnet_anchor,
                kwargs={"data_path": common_kwargs["data_path"], "target_date": target_date},
                device="cpu",
                node_id="sgdfnet/anchor_prepare",
            )
            cpu_tasks.append(task_spec)
            continue
        output_path = run_dir / task_name / "prediction" / f"{model_name}_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_paths[(model_name, task_name)] = output_path
        if output_path.exists() and not force:
            try:
                cached_df = pd.read_csv(output_path)
                formal96_cache = (
                    bool(common_kwargs.get("production_mode"))
                    and str(common_kwargs.get("resolution")) == "15min"
                )
                if formal96_cache:
                    expected_cutoff = (
                        da_cutoff_date if task_name == "dayahead" else rt_cutoff_date
                    )
                    errors = _validate_formal96_prediction_cache(
                        cached_df,
                        target_date=target_date,
                        model_name=model_name,
                        task=task_name,
                        expected_cutoff=expected_cutoff,
                        snapshot_id=common_kwargs.get("snapshot_id"),
                        serving_protocol=common_kwargs.get("serving_protocol", FORMAL96_PREDICTION_CONTRACT),
                        run_mode=common_kwargs.get("run_mode"),
                        snapshot_kind=common_kwargs.get("snapshot_kind"),
                        proxy_policy_version=common_kwargs.get("proxy_policy_version"),
                        proxy_cutoff_period=common_kwargs.get("proxy_cutoff_period"),
                    )
                else:
                    errors = validate_daily_predictions(
                        cached_df, target_date, model_name, task_name,
                        resolution=common_kwargs["resolution"],
                    )
                    if cached_df["y_pred"].isna().any():
                        errors.append(f"{task_name}/{model_name}: cached y_pred contains NaN")
                if not errors:
                    # --finish validates SGDFNet's decision-day anchor from the
                    # prediction manifest.  A strict cache hit must therefore
                    # preserve the same audit metadata as a freshly executed
                    # leg, not just the CSV rows.
                    results[task_name][model_name] = _build_cached_result_payload(
                        cached_df,
                        output_path=output_path,
                        model_name=model_name,
                        task_name=task_name,
                    )
                    cached_nodes.add(f"{model_name}/{task_name}")
                    continue
                logger.info(
                    "Rejecting stale/unknown cache %s: %s",
                    output_path,
                    "; ".join(errors[:4]),
                )
            except Exception as exc:
                logger.warning("Invalid split-process cache %s: %s", output_path, exc)

        kwargs = dict(common_kwargs)
        kwargs.update(
            {
                "model_name": model_name,
                "task": task_name,
                "target_date": target_date,
                "data_path": feature_view_paths.get(
                    f"{task_name}/{model_name}", common_kwargs["data_path"]
                ),
                "cutoff_date": da_cutoff_date if task_name == "dayahead" else rt_cutoff_date,
                "output_path": str(output_path),
            }
        )
        dependencies: tuple[str, ...] = ()
        graph_dependencies: tuple[str, ...] = ()
        if (model_name, task_name) == ("timesfm", "realtime"):
            graph_dependencies = ("timesfm/dayahead",)
        elif (model_name, task_name) == ("sgdfnet", "realtime"):
            graph_dependencies = ("sgdfnet/anchor_prepare",)
        elif (model_name, task_name) == ("timemixer", "realtime"):
            graph_dependencies = ("timemixer/dayahead",)
        graph_dependencies = tuple(
            dependency for dependency in graph_dependencies
            if dependency not in cached_nodes
        )

        task_spec = ScheduleTask(
            model_name=model_name,
            task_name=task_name,
            target_date=target_date,
            fn=_predict_model,
            kwargs=kwargs,
            device=classify_model_device(model_name),
            dependencies=dependencies,
            node_id=f"{model_name}/{task_name}",
            depends_on=graph_dependencies,
        )
        (cpu_tasks if task_spec.device == "cpu" else gpu_tasks).append(task_spec)

    scheduler = ResourceScheduler(
        max_cpu_workers=2,
        max_gpu_workers=1,
        resource_mode="split_process",
        log_path=str(run_dir / "logs" / "pipeline.log"),
    )
    schedule_results = scheduler.run(cpu_tasks + gpu_tasks)
    for sr in schedule_results:
        if (sr.model_name, sr.task_name) not in direct_nodes:
            continue
        task_result = results[sr.task_name]
        if sr.success:
            output_path = run_dir / sr.task_name / "prediction" / f"{sr.model_name}_predictions.csv"
            task_result[sr.model_name] = {
                "status": "ok",
                "output_path": str(output_path),
                "elapsed_seconds": sr.elapsed_seconds,
            }
            if sr.model_name == "sgdfnet" and sr.task_name == "realtime" and output_path.exists():
                audit_df = pd.read_csv(output_path)
                fallback = audit_df.get("fallback_used", pd.Series(False, index=audit_df.index))
                task_result[sr.model_name]["anchor_contract"] = {
                    "anchor_source_day": str(audit_df.get("anchor_source_day", pd.Series([pd.NaT])).dropna().iloc[0])
                    if audit_df.get("anchor_source_day", pd.Series(dtype=object)).notna().any() else None,
                    "source_type": str(audit_df.get("anchor_source_type", pd.Series(["unknown"])).dropna().iloc[0])
                    if audit_df.get("anchor_source_type", pd.Series(dtype=object)).notna().any() else "unknown",
                    "rows": int(audit_df.get("anchor_rows", pd.Series([0])).max()),
                    "fallback_used": bool(pd.Series(fallback).astype(bool).any()),
                }
                anchor_contract = task_result[sr.model_name]["anchor_contract"]
                expected_anchor_day = (
                    pd.Timestamp(target_date) - pd.Timedelta(days=1)
                ).date().isoformat()
                if common_kwargs.get("production_mode") and anchor_contract["anchor_source_day"] != expected_anchor_day:
                    task_result[sr.model_name]["status"] = "failed"
                    task_result[sr.model_name]["error"] = (
                        f"formal SGDFNet anchor source day={anchor_contract['anchor_source_day']} "
                        f"expected D-1={expected_anchor_day}"
                    )
                if common_kwargs.get("production_mode") and anchor_contract["rows"] != 96:
                    task_result[sr.model_name]["status"] = "failed"
                    task_result[sr.model_name]["error"] = (
                        f"formal SGDFNet anchor rows={anchor_contract['rows']} expected 96"
                    )
                if common_kwargs.get("production_mode") and anchor_contract["fallback_used"]:
                    # A median fallback is retained for abnormal/internal
                    # replay, but a normal formal production run must not
                    # silently publish an anchor that is not D-1 complete.
                    task_result[sr.model_name]["status"] = "failed"
                    task_result[sr.model_name]["error"] = (
                        "formal SGDFNet anchor used historical fallback; "
                        "production requires fallback_used=false"
                    )
        else:
            task_result[sr.model_name] = {
                "status": "failed",
                "error": sr.error,
                "elapsed_seconds": sr.elapsed_seconds,
            }
    return results


def _prepare_sgdfnet_anchor(*, data_path: str, target_date: str) -> None:
    """Validate the shared Dynamic-v1 FeatureView before SGDFNet anchor use.

    The SGDFNet protocol performs the actual D-1 DA p1..p96 materialization
    inside its prediction call.  This explicit internal DAG node prevents RT
    submission until the same routed FeatureView is available and keeps the anchor
    contract visible in scheduling/manifest audits.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"SGDFNet anchor preparation has no shared as-of input for {target_date}: {data_path}"
        )


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
    lgbm_training_months_candidates=None,
    lgbm_window_selection_metric: str = "smape",
    lgbm_window_mae_weight: float = 0.25,
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
    model_output_root: str | None = None,
    max_cpu: int = 2,
    max_gpu: int = 1,
    force: bool = False,
    dynamic_serving: bool = False,
    snapshot_id: str | None = None,
    serving_protocol: str = "formal96_dynamic_snapshot_v1",
    run_mode: str = "LIVE_DYNAMIC",
    snapshot_kind: str = "live",
    proxy_policy_version: str | None = None,
    proxy_cutoff_period: int | None = None,
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
                "lgbm_training_months_candidates": lgbm_training_months_candidates,
                "lgbm_window_selection_metric": lgbm_window_selection_metric,
                "lgbm_window_mae_weight": lgbm_window_mae_weight,
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
                "model_output_root": model_output_root,
                "production_mode": resolution == "15min",
                "dynamic_serving": dynamic_serving,
                "snapshot_id": snapshot_id,
                "serving_protocol": serving_protocol,
                "run_mode": run_mode,
                "snapshot_kind": snapshot_kind,
                "proxy_policy_version": proxy_policy_version,
                "proxy_cutoff_period": proxy_cutoff_period,
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
    lgbm_training_months_candidates=None,
    lgbm_window_selection_metric: str = "smape",
    lgbm_window_mae_weight: float = 0.25,
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
    model_output_root: str | None = None,
    production_mode: bool = False,
    dynamic_serving: bool = False,
    snapshot_id: str | None = None,
    serving_protocol: str = "formal96_dynamic_snapshot_v1",
    run_mode: str = "LIVE_DYNAMIC",
    snapshot_kind: str = "live",
    proxy_policy_version: str | None = None,
    proxy_cutoff_period: int | None = None,
    rt916_train_steps: int | None = None,
) -> pd.DataFrame:
    """
    Run a single model prediction and save to CSV.
    Fails fast if validation errors are detected.
    """
    logger.info(f"Predicting: {model_name}/{task} on {target_date} (res={resolution})")

    if task == "realtime" and model_name == "lightgbm":
        raise ValueError(
            f"{model_name}/realtime is disabled: it is not in the production "
            "realtime candidate pool"
        )

    if model_name == "lightgbm":
        df = _predict_lightgbm(task, target_date, data_path, epf_root, allow_v2_fallback, epf_v1_mode, cutoff_date, seed=seed, deterministic=deterministic, resolution=resolution, training_months_candidates=lgbm_training_months_candidates, window_selection_metric=lgbm_window_selection_metric, window_mae_weight=lgbm_window_mae_weight, model_output_root=model_output_root)
    elif model_name == "timesfm":
        df = _predict_timesfm(task, target_date, data_path, epf_root, allow_v2_fallback, epf_v1_mode, cutoff_date, seed=seed, deterministic=deterministic, resolution=resolution)
    elif model_name == "timemixer":
        df = _predict_timemixer(
            task, target_date, data_path, cutoff_date,
            realtime_cutoff_hour, training_months, val_ratio,
            timemixer_epochs, timemixer_patience, timemixer_batch_size,
            timemixer_full_refit, timemixer_seeds,
            seed=seed, deterministic=deterministic, resolution=resolution,
            model_output_root=model_output_root,
            dynamic_serving=dynamic_serving,
        )
    elif model_name == "sgdfnet":
        df = _predict_sgdfnet(
            task, target_date, data_path, cutoff_date, realtime_cutoff_hour,
            seed=seed, deterministic=deterministic, resolution=resolution,
            model_output_root=model_output_root,
            dynamic_serving=dynamic_serving,
        )
    elif model_name == "rt916":
        df = _predict_rt916(
            task, target_date, data_path, cutoff_date,
            realtime_cutoff_hour, training_months,
            seed=seed, deterministic=deterministic, resolution=resolution,
            model_output_root=model_output_root,
            production_mode=production_mode,
            dynamic_serving=dynamic_serving,
            rt916_train_steps=rt916_train_steps,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Normalize provenance at the common model exit.  Registry-backed models
    # already provide this field; bundled v1 adapters (LightGBM/TimesFM) do not.
    if "da_feature_source" not in df.columns:
        df["da_feature_source"] = _get_da_feature_source(model_name, task)

    # Stamp formal 96 outputs with the contract required for future cache reuse.
    # This is deliberately stored in the prediction CSV itself so a cache can
    # prove how it was produced without trusting a mutable day-level manifest.
    if production_mode and str(resolution) == "15min":
        # The historical cutoff timestamp is retained only as an internal
        # training-window argument.  Formal serving provenance is the shared
        # immutable snapshot, never a reconstructed fixed-hour timestamp.
        if "data_cutoff" in df.columns:
            df["data_cutoff"] = "dynamic_snapshot"
        df["production_contract"] = serving_protocol
        df["serving_protocol"] = serving_protocol
        df["snapshot_id"] = snapshot_id
        df["run_mode"] = run_mode
        df["snapshot_kind"] = snapshot_kind
        df["proxy_policy_version"] = proxy_policy_version
        df["proxy_cutoff_period"] = proxy_cutoff_period
        df["production_resolution"] = "15min"
        df["production_resource_mode"] = "split_process"
        df["production_rt_cutoff_hour"] = "dynamic_snapshot"
        df["production_rt916_train_steps"] = (
            int(rt916_train_steps or 24) if model_name == "rt916" else pd.NA
        )

    # Validate — FAIL FAST on errors
    errors = validate_daily_predictions(df, target_date, model_name, task, resolution=resolution)
    # Add additional checks
    if df["y_pred"].isna().all():
        errors.append(f"{model_name}/{task}: all y_pred values are NaN")
    elif production_mode and str(resolution) == "15min" and df["y_pred"].isna().any():
        errors.append(f"{model_name}/{task}: formal96 y_pred contains NaN")
    if "business_day" in df.columns and (df["business_day"] != target_date).any():
        errors.append(f"{model_name}/{task}: business_day mismatch for target {target_date}")

    if errors:
        err_msg = f"Validation FAILED for {model_name}/{task} on {target_date}: {'; '.join(errors)}"
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Save atomically so a killed worker can never leave a valid-looking
    # partial prediction file for the range runner to reuse.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(output.name + f".tmp-{os.getpid()}")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, output)
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
    training_months_candidates=None,
    window_selection_metric: str = "smape",
    window_mae_weight: float = 0.25,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
    model_output_root: str | None = None,
) -> pd.DataFrame:
    """LightGBM prediction via bundled adapter (local lightGBM/ by default)."""
    from runners.adapters.lightgbm_v1 import LightGBMV1Adapter
    adapter = LightGBMV1Adapter(epf_root=epf_root, mode=epf_v1_mode)
    previous_model_path = os.environ.get("LightGBM_MODEL_PATH")
    if model_output_root:
        runtime_model_dir = Path(model_output_root) / "lightgbm" / task
        runtime_model_dir.mkdir(parents=True, exist_ok=True)
        os.environ["LightGBM_MODEL_PATH"] = str(
            runtime_model_dir / "best_model_{}.pkl"
        )
    try:
        return adapter.predict(
            target_date=target_date,
            target=task,
            data_path=data_path,
            cutoff_date=cutoff_date,
            seed=seed,
            deterministic=deterministic,
            resolution=resolution,
            training_months_candidates=training_months_candidates,
            window_selection_metric=window_selection_metric,
            window_mae_weight=window_mae_weight,
        )
    finally:
        if model_output_root:
            if previous_model_path is None:
                os.environ.pop("LightGBM_MODEL_PATH", None)
            else:
                os.environ["LightGBM_MODEL_PATH"] = previous_model_path


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
    realtime_cutoff_hour: int = 15,
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
    model_output_root: str | None = None,
    dynamic_serving: bool = False,
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
        model_output_root=model_output_root,
        dynamic_serving=dynamic_serving,
    )


def _predict_sgdfnet(
    task: str,
    target_date: str,
    data_path: str,
    cutoff_date: str,
    realtime_cutoff_hour: int = 15,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
    model_output_root: str | None = None,
    dynamic_serving: bool = False,
) -> pd.DataFrame:
    """SGDFNet prediction using 2.0 model (CPU)."""
    return _predict_via_registry(
        "sgdfnet", task, target_date, data_path, cutoff_date,
        realtime_cutoff_hour=realtime_cutoff_hour,
        seed=seed,
        deterministic=deterministic,
        resolution=resolution,
        model_output_root=model_output_root,
        dynamic_serving=dynamic_serving,
    )


def _predict_rt916(
    task: str,
    target_date: str,
    data_path: str,
    cutoff_date: str,
    realtime_cutoff_hour: int = 15,
    training_months: int = 12,
    seed: int = 42,
    deterministic: bool = False,
    resolution: str = "hourly",
    model_output_root: str | None = None,
    production_mode: bool = False,
    dynamic_serving: bool = False,
    rt916_train_steps: int | None = None,
) -> pd.DataFrame:
    """RT916 prediction using 2.0 model (GPU)."""
    return _predict_via_registry(
        "rt916", task, target_date, data_path, cutoff_date,
        realtime_cutoff_hour=realtime_cutoff_hour,
        training_months=training_months,
        seed=seed,
        deterministic=deterministic,
        resolution=resolution,
        model_output_root=model_output_root,
        production_mode=production_mode,
        dynamic_serving=dynamic_serving,
        rt916_train_steps=rt916_train_steps,
    )


def _predict_via_registry(
    model_name: str,
    task: str,
    target_date: str,
    data_path: str,
    cutoff_date: str,
    realtime_cutoff_hour: int = 15,
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
    model_output_root: str | None = None,
    production_mode: bool = False,
    dynamic_serving: bool = False,
    rt916_train_steps: int | None = None,
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
        output_root=model_output_root,
        production_mode=production_mode,
        dynamic_serving=dynamic_serving,
        rt916_train_steps=rt916_train_steps,
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
    # SGDFNet production anchor audit fields survive normalization so the
    # ledger run manifest can prove source day/type/row count/fallback state.
    keep_cols.extend([
        "anchor_source_day", "anchor_source_type", "anchor_rows", "fallback_used",
    ])
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def _get_da_feature_source(model_name: str, task: str) -> str:
    """Return the day-ahead feature source string for a given model+task."""
    if task == "dayahead":
        return "none"  # dayahead models don't use DA features
    return {
        "timemixer": "timemixer_internal_dayahead_prediction",
        "rt916": "rt916_internal_joint_dayahead_prediction",
        "sgdfnet": "sgdfnet_decision_day_da_anchor",
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

    selected_pool = manifest.get("selected_model_pool", {})
    selected_models = tuple(selected_pool.get(task, {
        "dayahead": DAYAHEAD_MODELS,
        "realtime": REALTIME_MODELS,
    }[task]))
    selected_set = set(selected_models)

    pieces = []
    for csv_file in sorted(pred_dir.glob("*_predictions.csv")):
        if csv_file.name == "all_model_predictions_long.csv":
            continue
        try:
            df = pd.read_csv(csv_file)
            if "model_name" not in df.columns:
                manifest["warnings"].append(
                    f"Skipping {csv_file}: model_name column missing"
                )
                continue
            # A partial --models run publishes only its selected models.
            # Stale sibling CSVs from older attempts are never swept into the
            # current production ledger append.
            df = df[df["model_name"].astype(str).isin(selected_set)].copy()
            if not df.empty:
                pieces.append(df)
        except Exception as e:
            manifest["warnings"].append(f"Failed to read {csv_file}: {e}")

    if pieces:
        long_df = pd.concat(pieces, ignore_index=True)
        actual_models = set(long_df["model_name"].astype(str).unique())
        n_rows = len(long_df)
        expected = len(selected_models) * _res.slots_per_day
        if actual_models != selected_set or n_rows != expected:
            raise RuntimeError(
                f"{task} long table publication mismatch: "
                f"models={sorted(actual_models)} expected_models={sorted(selected_set)} "
                f"rows={n_rows} expected_rows={expected}"
            )

        long_path = pred_dir / "all_model_predictions_long.csv"
        long_df.to_csv(long_path, index=False)
        manifest["results"][f"{task}_long_rows"] = n_rows
        logger.info(f"{task} long table: {n_rows} rows → {long_path}")


# ===========================================================================
# Output aggregation
# ===========================================================================


def _append_all_to_ledger(
    run_dir: Path,
    target_date: str,
    ledger_root: Path,
    manifest: dict,
    tasks=("dayahead", "realtime"),
):
    """Append selected-task predictions to the prediction ledger."""
    for task in tasks:
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
            fragmented=(manifest.get("output_profile") == "feature_store"),
        )
        manifest["results"][f"{task}_ledger"] = result


def _extract_actuals(
    data_path: str,
    target_date: str,
    ledger_root: Path,
    manifest: dict,
    resolution=None,
    tasks=("dayahead", "realtime"),
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
        from utils.data_loader import load_table

        raw = load_table(data_path)

        # Find timestamp column.  A clean 96-point source may instead carry
        # business-day plus market-slot columns; normalize that form here so
        # actual extraction uses the same business-period contract as model
        # predictions.
        ts_col = None
        for c in ["时刻", "ds", "timestamp", "time", "datetime"]:
            if c in raw.columns:
                ts_col = c
                break

        if ts_col is not None:
            raw["ds"] = pd.to_datetime(raw[ts_col], errors="coerce")
        elif {"market_date", "时段"}.issubset(raw.columns):
            base = pd.to_datetime(raw["market_date"], errors="coerce").dt.normalize()
            token = raw["时段"].astype(str).str.strip()
            parts = token.str.split(":", n=1, expand=True)
            numeric_slot = pd.to_numeric(token, errors="coerce")
            has_clock = parts.shape[1] > 1
            if has_clock:
                hours = pd.to_numeric(parts[0], errors="coerce")
                minutes = pd.to_numeric(parts[1], errors="coerce")
                raw["ds"] = base + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")
                # 00:00 is the terminal point p96/h24, not the beginning of
                # the calendar day, and is handled by business-day mapping.
                raw.loc[hours.eq(0) & minutes.eq(0), "ds"] = base + pd.Timedelta(days=1)
            else:
                if numeric_slot.isna().any():
                    manifest["warnings"].append(
                        "No usable timestamp or market_date/时段 values in actual source"
                    )
                    return
                raw["ds"] = [
                    _res.timestamp_from_business(str(day.date()), int(slot))
                    if pd.notna(day) else pd.NaT
                    for day, slot in zip(base, numeric_slot)
                ]
        else:
            manifest["warnings"].append("No timestamp or market_date/时段 columns in data file")
            return

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
            "日前电价", "日前出清电价", "日前出清价格",
            "day_ahead_clearing_price",
            "dayahead_price", "da_price",
        ]
        realtime_aliases = [
            "实时电价", "实时出清电价", "实时出清价格",
            "realtime_price", "rt_price",
        ]

        task_aliases = {
            "dayahead": dayahead_aliases,
            "realtime": realtime_aliases,
        }
        for task in tasks:
            col_names = task_aliases[task]
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
                fragmented=(manifest.get("output_profile") == "feature_store"),
            )
            # rows_after is the persistent ledger size and grows across dates.
            # Keep the current target-day count for readiness gates.
            result["target_day_rows"] = int(len(act_df))
            manifest["results"][f"{task}_actual_ledger"] = result

    except Exception as e:
        manifest["warnings"].append(f"Actual extraction failed: {e}")
        logger.warning(f"Actual extraction error: {e}")


def settle_closed_actuals(
    data_path: str,
    target_date: str,
    ledger_root: Path,
    *,
    resolution,
    tasks=("dayahead", "realtime"),
    output_profile: str = "production",
    settlement_lag_days: int = 2,
) -> dict:
    """Append the latest *fully closed* historical actual day before weighting.

    For formal 96 serving, target T is predicted on T-1 before that business
    day is complete.  Therefore T-1 full-day truth must not be used as learner
    history.  The latest causally closed full day is T-2.  The selector may
    still skip this day if the authoritative source is incomplete.
    """
    closed_day = (
        pd.Timestamp(target_date) - pd.Timedelta(days=int(settlement_lag_days))
    ).strftime("%Y-%m-%d")
    scratch = {
        "output_profile": output_profile,
        "results": {},
        "warnings": [],
    }
    _extract_actuals(
        data_path,
        closed_day,
        ledger_root,
        scratch,
        resolution=resolution,
        tasks=tasks,
    )

    rows = {}
    expected = int(getattr(resolution, "slots_per_day", 0) or 0)
    for task in tasks:
        result = scratch["results"].get(f"{task}_actual_ledger", {})
        rows[task] = int(result.get("target_day_rows", 0) or 0)

    complete = bool(expected) and all(rows.get(task) == expected for task in tasks)
    return {
        "status": "complete" if complete else "partial",
        "closed_day": closed_day,
        "settlement_lag_days": int(settlement_lag_days),
        "expected_rows_per_task": expected,
        "rows": rows,
        "warnings": list(scratch.get("warnings", [])),
    }


def _check_target_actual_readiness(
    *,
    manifest: dict,
    tasks,
    target_date: str,
    slots_per_day: int,
    require_target_actual: bool,
) -> None:
    """Apply the same target-truth gate in every resource execution mode."""
    for task_name in tasks:
        actual_result = manifest.get("results", {}).get(f"{task_name}_actual_ledger", {})
        actual_rows = int(actual_result.get("target_day_rows", 0) or 0)
        if require_target_actual and actual_rows != slots_per_day:
            raise RuntimeError(
                f"{task_name} actual ledger incomplete for {target_date}: "
                f"rows={actual_rows} expected={slots_per_day}"
            )
        if not require_target_actual and actual_rows != slots_per_day:
            manifest.setdefault("warnings", []).append(
                f"{task_name} target-day actual not complete ({actual_rows}/{slots_per_day}); "
                "allowed in live prediction mode"
            )


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
