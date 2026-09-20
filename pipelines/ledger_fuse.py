"""
Ledger fuse pipeline.

For a target day D, reads predictions from the prediction ledger
(or daily run outputs) and learned weights, then produces fused
predictions via weighted averaging.

Output:
  outputs/runs/{D}/{task}/fuse/
    fused_predictions.csv
    fused_debug.csv
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fusion.apply_daily_ledger_weights import apply_daily_ledger_weights
from fusion.model_pool import tasks_for_target

logger = logging.getLogger(__name__)


def run_ledger_fuse(args: Any) -> dict:
    """
    Main entry for --pipeline ledger_fuse.
    """
    from utils.resolution import resolve_resolution

    target_date = args.date
    if not target_date:
        raise ValueError("--date is required for ledger_fuse")

    res = resolve_resolution(getattr(args, "resolution", "hourly"))
    domain = "96" if res.label == "15min" else "24"
    default_ledger = "outputs/96/ledger" if res.label == "15min" else "outputs/ledger"
    default_runs = "outputs/96/runs" if res.label == "15min" else "outputs/runs"
    ledger_root = Path(getattr(args, "ledger_root", None) or default_ledger)
    runs_root = Path(getattr(args, "runs_root", None) or default_runs)
    allow_eq_w = getattr(args, "allow_equal_weight_fallback", False)
    weight_prune_threshold = float(getattr(args, "weight_prune_threshold", 0.05))
    min_active_models = int(getattr(args, "weight_min_active_models", 1))
    requested_tasks = tasks_for_target(getattr(args, "target", "both"))

    logger.info(
        f"=== ledger_fuse: {target_date} (res={res.label}, tasks={','.join(requested_tasks)}) ==="
    )

    manifest = {
        "pipeline": "ledger_fuse",
        "target_date": target_date,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "results": {},
        "warnings": [],
        "errors": [],
        "requested_tasks": list(requested_tasks),
    }

    try:
        failed_tasks = []
        for task in requested_tasks:
            task_result = _fuse_for_task(
                task=task,
                target_date=target_date,
                ledger_root=ledger_root,
                runs_root=runs_root,
                allow_equal_weight_fallback=allow_eq_w,
                weight_prune_threshold=weight_prune_threshold,
                min_active_models=min_active_models,
                resolution=res,
            )
            manifest["results"][task] = task_result
            if task_result.get("status") != "complete":
                failed_tasks.append(
                    f"{task}: {task_result.get('error', task_result.get('status'))}"
                )

        if failed_tasks:
            manifest["status"] = "failed"
            manifest["errors"].extend(failed_tasks)
        else:
            manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as e:
        manifest["status"] = "failed"
        manifest["errors"].append(str(e))
        logger.exception(f"ledger_fuse failed: {e}")

    # Write manifest
    manifest_path = Path(
        getattr(args, "_fuse_manifest_path", None)
        or (runs_root / target_date / "run_manifest.json")
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.name != "run_manifest.json":
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        return manifest
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing["fuse_stage"] = manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    return manifest


def _fuse_for_task(
    task: str,
    target_date: str,
    ledger_root: Path,
    runs_root: Path,
    allow_equal_weight_fallback: bool = False,
    weight_prune_threshold: float = 0.05,
    min_active_models: int = 1,
    resolution=None,
) -> dict:
    """Fuse predictions for a single task."""
    result = {"task": task, "status": "running", "errors": [], "warnings": []}

    # Find predictions
    pred_path = runs_root / target_date / task / "prediction" / "all_model_predictions_long.csv"

    if not pred_path.exists():
        # Try loading from ledger
        from pipelines.prediction_ledger import load_prediction_ledger
        predictions_long = load_prediction_ledger(ledger_root, task, [target_date])
        if predictions_long.empty:
            result["status"] = "failed"
            result["error"] = f"No predictions found for {task} on {target_date}"
            return result
    else:
        predictions_long = pd.read_csv(pred_path)

    # Find weights
    weight_path = runs_root / target_date / task / "weight" / "weights.csv"
    if not weight_path.exists():
        result["status"] = "failed"
        result["error"] = f"No weights found at {weight_path}"
        return result

    weights = pd.read_csv(weight_path)

    logger.info(
        f"[{task}] Fusing: {len(predictions_long)} predictions, "
        f"{len(weights)} weight entries"
    )

    # Apply weights (strict mode)
    fused_df, debug_df = apply_daily_ledger_weights(
        predictions_long=predictions_long,
        weights=weights,
        target_day=target_date,
        task=task,
        allow_equal_weight_fallback=allow_equal_weight_fallback,
        strict=True,
        weight_prune_threshold=weight_prune_threshold,
        min_active_models=min_active_models,
        resolution=resolution,
    )

    # Save
    fuse_dir = runs_root / target_date / task / "fuse"
    fuse_dir.mkdir(parents=True, exist_ok=True)

    fused_df.to_csv(fuse_dir / "fused_predictions.csv", index=False)
    debug_df.to_csv(fuse_dir / "fused_debug.csv", index=False)
    result["output_paths"] = {
        "fused": str(fuse_dir / "fused_predictions.csv"),
        "debug": str(fuse_dir / "fused_debug.csv"),
    }
    gate_cols = [
        "task", "period", "weight_gate_threshold", "pruned_models",
        "active_models", "gate_fallback_used", "gate_fallback_model",
    ]
    if set(gate_cols).issubset(debug_df.columns):
        debug_df[gate_cols].drop_duplicates().to_csv(
            fuse_dir / "model_quality_gate.csv", index=False
        )
        result["output_paths"]["quality_gate"] = str(fuse_dir / "model_quality_gate.csv")
    else:
        result["errors"].append(
            "fused_debug.csv is missing the model-quality gate columns"
        )

    result["fused_rows"] = len(fused_df)
    result["fuse_dir"] = str(fuse_dir)
    result["weight_prune_threshold"] = weight_prune_threshold
    if "pruned_models" in debug_df.columns:
        result["pruned_models"] = sorted({
            model
            for value in debug_df["pruned_models"].fillna("")
            for model in str(value).split(",")
            if model
        })

    # Verify
    _verify_fuse_output(fused_df, debug_df, task, result, resolution)
    if result["errors"]:
        result["status"] = "failed"
    else:
        result["status"] = "complete"

    logger.info(f"[{task}] Fused: {len(fused_df)} rows")

    return result


def _verify_fuse_output(
    fused_df: pd.DataFrame,
    debug_df: pd.DataFrame,
    task: str,
    result: dict,
    resolution=None,
):
    """Verify fused output integrity."""
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    slot_col = res.slot_column
    n_expected = res.slots_per_day
    errors = []
    warnings = []

    # Check N rows
    if len(fused_df) != n_expected:
        errors.append(f"Expected {n_expected} rows, got {len(fused_df)}")

    # Check slots 1..N
    if slot_col in fused_df.columns:
        hours = fused_df[slot_col].values
        expected = set(range(1, n_expected + 1))
        actual = set(int(h) for h in hours)
        if actual != expected:
            missing = expected - actual
            if missing:
                errors.append(f"Missing slots: {sorted(missing)}")
            extra = actual - expected
            if extra:
                errors.append(f"Unexpected slots: {sorted(extra)}")

        # Check no duplicate slots
        if fused_df[slot_col].duplicated().any():
            errors.append("Duplicate slots detected")

    if "y_fused" not in fused_df.columns:
        errors.append("fused output is missing y_fused")
    elif not pd.to_numeric(fused_df["y_fused"], errors="coerce").notna().all():
        errors.append("fused output contains NaN/non-numeric y_fused")

    if "period" in fused_df.columns and fused_df["period"].isna().any():
        errors.append("fused output contains NaN period")

    # Check no fillna(0)
    if (fused_df["y_fused"] == 0).any():
        warnings.append("Zero values in fused predictions (suspect fillna(0))")

    # Check renormalization info
    if "renormalized" in debug_df.columns:
        n_renorm = debug_df["renormalized"].sum()
        if n_renorm > 0:
            result["renormalized_slots"] = int(n_renorm)

    if warnings:
        result["warnings"].extend(warnings)
        for w in warnings:
            logger.warning(f"[{task}] {w}")
    if errors:
        result["errors"].extend(errors)
        for error in errors:
            logger.error(f"[{task}] {error}")
