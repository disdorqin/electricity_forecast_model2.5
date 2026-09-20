"""Bootstrap a formal 96-point production ledger from an audited history ledger.

This is an operational state migration tool, not a backtest scorer.  It copies
only a bounded historical window into the current production ledger, preserves
the source cutoffs/provenance columns, validates the canonical DA3/RT4 pools and
96 slots, stages the result, verifies the weight learner can select the required
history, and only then promotes the staged ledger.

Typical first deployment / warm-start:

    python scripts/server/bootstrap_96_production_ledger.py \
      --source-ledger outputs/experiments/04_pipeline_audits/re_prediction_96_merged_improved_20251218_20260814_v2/ledger \
      --target-date 2026-08-16

Add --apply only after the dry-run audit is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS  # noqa: E402
from pipelines.ledger_weight import select_complete_training_days  # noqa: E402
from pipelines.prediction_ledger import (  # noqa: E402
    append_predictions_to_ledger,
    update_actual_ledger,
)
from utils.resolution import QUARTER  # noqa: E402

CONTRACT = "formal96_ledger_warm_start_v1"
FULL_SOURCE_CONTRACT = "formal96_ledger_full_source_merge_v1"
FULL_SOURCE_START = "2025-12-18"
FULL_SOURCE_END = "2026-08-14"
TASK_MODELS = {
    "dayahead": tuple(DAYAHEAD_MODELS),
    "realtime": tuple(REALTIME_MODELS),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _history_windows(
    target_date: str,
    complete_days: int,
    *,
    actual_end_lag_days: int = 2,
) -> tuple[list[str], list[str]]:
    """Return prediction and actual warm-start windows for live serving.

    Predictions may exist through T-1, while full-day actuals are only causal
    through T-2 under the current D-1 intraday serving design.  Keeping the
    open T-1 prediction rows lets that day become trainable automatically once
    its actuals are settled on the next invocation.
    """
    target = pd.Timestamp(target_date)
    actual_end = target - pd.Timedelta(days=int(actual_end_lag_days))
    actual_start = actual_end - pd.Timedelta(days=int(complete_days) - 1)
    prediction_end = target - pd.Timedelta(days=1)
    prediction_days = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(actual_start, prediction_end, freq="D")
    ]
    actual_days = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(actual_start, actual_end, freq="D")
    ]
    return prediction_days, actual_days


def _canonical_paths(root: Path, task: str) -> tuple[Path, Path]:
    return (
        root / task / "prediction" / "prediction_ledger.parquet",
        root / task / "actual" / "actual_ledger.parquet",
    )


def _read_window(
    source_root: Path,
    task: str,
    prediction_days: list[str],
    actual_days: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path, actual_path = _canonical_paths(source_root, task)
    if not pred_path.exists():
        raise FileNotFoundError(f"source prediction ledger missing: {pred_path}")
    if not actual_path.exists():
        raise FileNotFoundError(f"source actual ledger missing: {actual_path}")

    pred = pd.read_parquet(pred_path)
    actual = pd.read_parquet(actual_path)
    prediction_day_set = set(prediction_days)
    actual_day_set = set(actual_days)
    pred = pred[pred["target_day"].astype(str).isin(prediction_day_set)].copy()
    actual = actual[actual["target_day"].astype(str).isin(actual_day_set)].copy()
    return pred, actual


def _safe_cutoff(task: str, target_day: str, value: Any) -> bool:
    """Current formal boundary: history may be earlier, but never later."""
    try:
        cutoff = pd.Timestamp(value)
    except Exception:
        return False
    day = pd.Timestamp(target_day)
    if task == "dayahead":
        return cutoff <= day - pd.Timedelta(days=1) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return cutoff <= day - pd.Timedelta(days=1) + pd.Timedelta(hours=15)


def _audit_window(
    source_root: Path,
    target_date: str,
    days: int,
    *,
    actual_end_lag_days: int = 2,
    max_lookback_days: int = 90,
) -> tuple[dict[str, Any], dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Audit a warm-start window using the same selector as production."""
    expected_slots = set(range(1, QUARTER.slots_per_day + 1))
    target = pd.Timestamp(target_date)
    open_prediction_day = (target - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    audit: dict[str, Any] = {
        "status": "PASS",
        "contract": CONTRACT,
        "target_date": target_date,
        "selection_mode": "adaptive_complete_days",
        "actual_end_lag_days": int(actual_end_lag_days),
        "required_days": days,
        "max_lookback_days": int(max_lookback_days),
        "resolution": QUARTER.label,
        "tasks": {},
        "errors": [],
        "policy": (
            "operational warm-start only; use the same adaptive complete-day "
            "selector as production; preserve source cutoff/provenance; never "
            "relabel legacy history as current-model replay evidence"
        ),
    }
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_prediction_days: set[str] = set()
    all_actual_days: set[str] = set()

    for task, expected_models in TASK_MODELS.items():
        selector = select_complete_training_days(
            task=task,
            target_date=target_date,
            ledger_root=source_root,
            expected_models=list(expected_models),
            required_days=days,
            max_lookback_days=max_lookback_days,
            resolution=QUARTER,
            history_lag_days=actual_end_lag_days,
        )
        task_errors: list[str] = []
        if selector.get("status") != "PASS":
            task_errors.extend(str(err) for err in selector.get("errors", []))
            if not task_errors:
                task_errors.append(
                    f"adaptive source readiness selected "
                    f"{selector.get('selected_count', 0)}/{days} complete days"
                )

        selected_days = sorted(set(str(day) for day in selector.get("selected_days", [])))
        pred_path, actual_path = _canonical_paths(source_root, task)
        pred_all = pd.read_parquet(pred_path) if pred_path.exists() else pd.DataFrame()
        actual_all = pd.read_parquet(actual_path) if actual_path.exists() else pd.DataFrame()
        if pred_all.empty:
            task_errors.append(f"source prediction ledger missing/empty: {pred_path}")
        if actual_all.empty:
            task_errors.append(f"source actual ledger missing/empty: {actual_path}")

        include_open_prediction = False
        open_reason = "absent"
        if not pred_all.empty and "target_day" in pred_all.columns:
            open_part = pred_all[
                pred_all["target_day"].astype(str) == open_prediction_day
            ].copy()
            if not open_part.empty:
                open_errors: list[str] = []
                for model in expected_models:
                    model_rows = open_part[
                        open_part["model_name"].astype(str) == model
                    ]
                    if len(model_rows) != QUARTER.slots_per_day:
                        open_errors.append(f"{model}:rows={len(model_rows)} expected=96")
                        continue
                    slots = set(
                        pd.to_numeric(
                            model_rows["business_period"], errors="coerce"
                        ).dropna().astype(int)
                    )
                    if slots != expected_slots:
                        open_errors.append(f"{model}:slot mismatch")
                    if model_rows["y_pred"].isna().any():
                        open_errors.append(f"{model}:NaN prediction")
                    if (
                        "data_cutoff" not in model_rows.columns
                        or model_rows["data_cutoff"].isna().any()
                    ):
                        open_errors.append(f"{model}:missing data_cutoff")
                    else:
                        bad = [
                            str(v)
                            for v in model_rows["data_cutoff"].unique()
                            if not _safe_cutoff(task, open_prediction_day, v)
                        ]
                        if bad:
                            open_errors.append(
                                f"{model}:cutoff later than formal boundary"
                            )
                if not open_errors:
                    include_open_prediction = True
                    open_reason = "included"
                else:
                    open_reason = "omitted_invalid:" + ";".join(open_errors[:5])

        prediction_days = list(selected_days)
        if include_open_prediction and open_prediction_day not in prediction_days:
            prediction_days.append(open_prediction_day)
        prediction_days = sorted(prediction_days)
        prediction_day_set = set(prediction_days)
        actual_day_set = set(selected_days)

        pred = (
            pred_all[pred_all["target_day"].astype(str).isin(prediction_day_set)].copy()
            if not pred_all.empty
            else pred_all.copy()
        )
        actual = (
            actual_all[actual_all["target_day"].astype(str).isin(actual_day_set)].copy()
            if not actual_all.empty
            else actual_all.copy()
        )
        frames[task] = (pred, actual)

        pred_days_present = (
            set(pred["target_day"].astype(str))
            if not pred.empty and "target_day" in pred.columns
            else set()
        )
        actual_days_present = (
            set(actual["target_day"].astype(str))
            if not actual.empty and "target_day" in actual.columns
            else set()
        )
        if pred_days_present != prediction_day_set:
            task_errors.append(
                f"prediction day set mismatch missing="
                f"{sorted(prediction_day_set - pred_days_present)} "
                f"extra={sorted(pred_days_present - prediction_day_set)}"
            )
        if actual_days_present != actual_day_set:
            task_errors.append(
                f"actual day set mismatch missing="
                f"{sorted(actual_day_set - actual_days_present)} "
                f"extra={sorted(actual_days_present - actual_day_set)}"
            )

        present_models = (
            set(pred["model_name"].astype(str))
            if not pred.empty and "model_name" in pred.columns
            else set()
        )
        if present_models != set(expected_models):
            task_errors.append(
                f"model pool mismatch actual={sorted(present_models)} "
                f"expected={sorted(expected_models)}"
            )

        for day in prediction_days:
            day_pred = pred[pred["target_day"].astype(str) == day]
            for model in expected_models:
                model_rows = day_pred[
                    day_pred["model_name"].astype(str) == model
                ]
                if len(model_rows) != QUARTER.slots_per_day:
                    task_errors.append(
                        f"{day}/{model}: rows={len(model_rows)} expected=96"
                    )
                    continue
                slots = set(
                    pd.to_numeric(
                        model_rows["business_period"], errors="coerce"
                    ).dropna().astype(int)
                )
                if slots != expected_slots:
                    task_errors.append(f"{day}/{model}: business_period set mismatch")
                if model_rows["y_pred"].isna().any():
                    task_errors.append(f"{day}/{model}: y_pred contains NaN")
                if (
                    "data_cutoff" not in model_rows.columns
                    or model_rows["data_cutoff"].isna().any()
                ):
                    task_errors.append(f"{day}/{model}: data_cutoff missing")
                else:
                    bad = [
                        str(v)
                        for v in model_rows["data_cutoff"].unique()
                        if not _safe_cutoff(task, day, v)
                    ]
                    if bad:
                        task_errors.append(
                            f"{day}/{model}: cutoff later than formal boundary {bad[:3]}"
                        )

        for day in selected_days:
            day_actual = actual[actual["target_day"].astype(str) == day]
            if len(day_actual) != QUARTER.slots_per_day:
                task_errors.append(f"{day}/actual: rows={len(day_actual)} expected=96")
            else:
                slots = set(
                    pd.to_numeric(
                        day_actual["business_period"], errors="coerce"
                    ).dropna().astype(int)
                )
                if slots != expected_slots:
                    task_errors.append(f"{day}/actual: business_period set mismatch")
                if day_actual["y_true"].isna().any():
                    task_errors.append(f"{day}/actual: y_true contains NaN")

        cutoff_values = (
            sorted(pred["data_cutoff"].dropna().astype(str).unique().tolist())
            if not pred.empty and "data_cutoff" in pred.columns
            else []
        )
        audit["tasks"][task] = {
            "status": "PASS" if not task_errors else "FAIL",
            "models": list(expected_models),
            "selection_mode": "adaptive_complete_days",
            "selected_days": selected_days,
            "selected_count": len(selected_days),
            "open_prediction_day": open_prediction_day,
            "open_prediction_status": open_reason,
            "prediction_rows": int(len(pred)),
            "actual_rows": int(len(actual)),
            "prediction_days": int(len(pred_days_present)),
            "actual_days": int(len(actual_days_present)),
            "cutoff_min": cutoff_values[0] if cutoff_values else None,
            "cutoff_max": cutoff_values[-1] if cutoff_values else None,
            "errors": task_errors,
        }
        audit["errors"].extend(f"{task}: {error}" for error in task_errors)
        all_prediction_days.update(prediction_day_set)
        all_actual_days.update(actual_day_set)

    audit["prediction_window_start"] = min(all_prediction_days) if all_prediction_days else None
    audit["prediction_window_end"] = max(all_prediction_days) if all_prediction_days else None
    audit["prediction_days"] = len(all_prediction_days)
    audit["actual_window_start"] = min(all_actual_days) if all_actual_days else None
    audit["actual_window_end"] = max(all_actual_days) if all_actual_days else None
    audit["actual_days"] = len(all_actual_days)

    if audit["errors"]:
        audit["status"] = "FAIL"
    return audit, frames


def _audit_full_source(
    source_root: Path,
    *,
    source_start: str = FULL_SOURCE_START,
    source_end: str = FULL_SOURCE_END,
) -> tuple[dict[str, Any], dict[str, tuple[pd.DataFrame, pd.DataFrame]]]:
    """Audit the complete historical source without the 90-day selector.

    The server archive is an immutable historical artifact.  Its legacy
    provenance/cutoff is preserved verbatim; this audit only checks canonical
    shape, model pools, resolution and finite values before staging it behind
    current production rows.
    """
    expected_slots = set(range(1, QUARTER.slots_per_day + 1))
    expected_days = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(source_start, source_end, freq="D")
    ]
    expected_day_set = set(expected_days)
    audit: dict[str, Any] = {
        "status": "PASS",
        "contract": FULL_SOURCE_CONTRACT,
        "selection_mode": "full_source_exact_range",
        "source_range": {"start": source_start, "end": source_end},
        "source_days": len(expected_days),
        "resolution": QUARTER.label,
        "tasks": {},
        "errors": [],
        "policy": (
            "source provenance is preserved; current production rows are staged "
            "after source rows and therefore win on duplicate canonical keys"
        ),
    }
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for task, expected_models in TASK_MODELS.items():
        pred_path, actual_path = _canonical_paths(source_root, task)
        errors: list[str] = []
        try:
            pred = pd.read_parquet(pred_path)
        except Exception as exc:
            pred = pd.DataFrame()
            errors.append(f"prediction read failed: {pred_path}: {exc}")
        try:
            actual = pd.read_parquet(actual_path)
        except Exception as exc:
            actual = pd.DataFrame()
            errors.append(f"actual read failed: {actual_path}: {exc}")
        pred_days = set(pred.get("target_day", pd.Series(dtype=str)).astype(str))
        actual_days = set(actual.get("target_day", pd.Series(dtype=str)).astype(str))
        if pred_days != expected_day_set:
            errors.append(
                f"prediction day range mismatch missing={sorted(expected_day_set - pred_days)} "
                f"extra={sorted(pred_days - expected_day_set)}"
            )
        if actual_days != expected_day_set:
            errors.append(
                f"actual day range mismatch missing={sorted(expected_day_set - actual_days)} "
                f"extra={sorted(actual_days - expected_day_set)}"
            )
        models = set(pred.get("model_name", pd.Series(dtype=str)).astype(str))
        if models != set(expected_models):
            errors.append(f"model pool mismatch actual={sorted(models)} expected={list(expected_models)}")
        if "resolution" in pred.columns:
            bad_resolution = set(pred["resolution"].dropna().astype(str)) - {QUARTER.label, "15min", "quarter"}
            if bad_resolution:
                errors.append(f"prediction resolution mismatch={sorted(bad_resolution)}")
        for day in expected_days:
            day_pred = pred[pred.get("target_day", pd.Series(dtype=str)).astype(str).eq(day)] if not pred.empty else pred
            for model in expected_models:
                rows = day_pred[day_pred.get("model_name", pd.Series(dtype=str)).astype(str).eq(model)] if not day_pred.empty else day_pred
                if len(rows) != QUARTER.slots_per_day:
                    errors.append(f"{day}/{model}: rows={len(rows)} expected=96")
                    continue
                if "business_period" not in rows.columns:
                    errors.append(f"{day}/{model}: business_period missing")
                    continue
                slots = set(pd.to_numeric(rows["business_period"], errors="coerce").dropna().astype(int))
                if slots != expected_slots:
                    errors.append(f"{day}/{model}: business_period set mismatch")
                if "y_pred" not in rows or rows["y_pred"].isna().any():
                    errors.append(f"{day}/{model}: y_pred contains NaN/missing")
        for day in expected_days:
            rows = actual[actual.get("target_day", pd.Series(dtype=str)).astype(str).eq(day)] if not actual.empty else actual
            if len(rows) != QUARTER.slots_per_day:
                errors.append(f"{day}/actual: rows={len(rows)} expected=96")
                continue
            if "business_period" not in rows.columns:
                errors.append(f"{day}/actual: business_period missing")
                continue
            slots = set(pd.to_numeric(rows["business_period"], errors="coerce").dropna().astype(int))
            if slots != expected_slots:
                errors.append(f"{day}/actual: business_period set mismatch")
            if "y_true" not in rows or rows["y_true"].isna().any():
                errors.append(f"{day}/actual: y_true contains NaN/missing")
        frames[task] = (pred.copy(), actual.copy())
        audit["tasks"][task] = {
            "status": "PASS" if not errors else "FAIL",
            "models": list(expected_models),
            "resolution": QUARTER.label,
            "resolution_source": "96_slot_contract" if "resolution" not in pred.columns else "source_column",
            "prediction_rows": int(len(pred)),
            "actual_rows": int(len(actual)),
            "prediction_days": len(pred_days),
            "actual_days": len(actual_days),
            "errors": errors,
        }
        audit["errors"].extend(f"{task}: {error}" for error in errors)
    if audit["errors"]:
        audit["status"] = "FAIL"
    return audit, frames


def _append_existing_target(staging_root: Path, target_root: Path, *, preserve_provenance: bool = False) -> None:
    """Let existing production rows win over imported history on duplicate keys."""
    def _without_target_conflicts(staged: pd.DataFrame, current: pd.DataFrame, kind: str) -> pd.DataFrame:
        if staged.empty or current.empty:
            return staged
        base = (
            ["task", "model_name", "forecast_date", "target_day", "business_day", "hour_business"]
            if kind == "prediction"
            else ["task", "target_day", "business_day", "hour_business"]
        )
        key_cols = [c for c in base if c in staged.columns and c in current.columns]
        if "business_period" in staged.columns and "business_period" in current.columns:
            key_cols.append("business_period")
        if not key_cols:
            return staged
        left = staged[key_cols].astype(str)
        right = current[key_cols].astype(str).drop_duplicates()
        conflict = left.merge(right.assign(__conflict=True), on=key_cols, how="left")["__conflict"].eq(True)
        return staged.loc[~conflict.to_numpy()].copy()

    for task in TASK_MODELS:
        pred_path, actual_path = _canonical_paths(target_root, task)
        if pred_path.exists():
            staged_path = staging_root / task / "prediction" / "prediction_ledger.parquet"
            staged_csv = staging_root / task / "prediction" / "prediction_ledger.csv"
            staged = pd.read_parquet(staged_path) if staged_path.exists() else pd.DataFrame()
            current = pd.read_parquet(pred_path)
            staged = _without_target_conflicts(staged, current, "prediction")
            staged_path.unlink(missing_ok=True)
            staged_csv.unlink(missing_ok=True)
            if not staged.empty:
                append_predictions_to_ledger(staged, staging_root, task, source_file="historical-source-filtered", preserve_provenance=preserve_provenance)
            append_predictions_to_ledger(
                current,
                staging_root,
                task,
                source_file=str(pred_path),
                preserve_provenance=preserve_provenance,
            )
        if actual_path.exists():
            staged_path = staging_root / task / "actual" / "actual_ledger.parquet"
            staged_csv = staging_root / task / "actual" / "actual_ledger.csv"
            staged = pd.read_parquet(staged_path) if staged_path.exists() else pd.DataFrame()
            current = pd.read_parquet(actual_path)
            staged = _without_target_conflicts(staged, current, "actual")
            staged_path.unlink(missing_ok=True)
            staged_csv.unlink(missing_ok=True)
            if not staged.empty:
                update_actual_ledger(staged, staging_root, task, source_file="historical-source-filtered", preserve_provenance=preserve_provenance)
            update_actual_ledger(
                current,
                staging_root,
                task,
                source_file=str(actual_path),
                preserve_provenance=preserve_provenance,
            )


def _stage(
    staging_root: Path,
    source_root: Path,
    target_root: Path,
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    preserve_provenance: bool = False,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # Imported history first; current production ledger second so current rows
    # win if the migration is rerun or windows overlap.
    for task, (pred, actual) in frames.items():
        results[f"{task}_prediction_import"] = append_predictions_to_ledger(
            pred,
            staging_root,
            task,
            source_file=str(source_root / task / "prediction" / "prediction_ledger.parquet"),
            preserve_provenance=preserve_provenance,
        )
        results[f"{task}_actual_import"] = update_actual_ledger(
            actual,
            staging_root,
            task,
            source_file=str(source_root / task / "actual" / "actual_ledger.parquet"),
            preserve_provenance=preserve_provenance,
        )
    _append_existing_target(staging_root, target_root, preserve_provenance=preserve_provenance)
    return results


def _readiness(root: Path, target_date: str, days: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for task, models in TASK_MODELS.items():
        results[task] = select_complete_training_days(
            task=task,
            target_date=target_date,
            ledger_root=root,
            expected_models=list(models),
            required_days=days,
            max_lookback_days=max(90, days),
            resolution=QUARTER,
            history_lag_days=2,
        )
    return {
        "status": "PASS"
        if all(result["status"] == "PASS" for result in results.values())
        else "FAIL",
        "tasks": results,
    }


def _promote(staging_root: Path, target_root: Path) -> None:
    """Promote all canonical ledger files with rollback on any failure."""
    targets: list[tuple[Path, Path]] = []
    for task in TASK_MODELS:
        for kind, stem in (("prediction", "prediction_ledger"), ("actual", "actual_ledger")):
            for suffix in (".parquet", ".csv"):
                staged = staging_root / task / kind / f"{stem}{suffix}"
                final = target_root / task / kind / f"{stem}{suffix}"
                if not staged.exists():
                    raise FileNotFoundError(f"staged ledger missing: {staged}")
                targets.append((staged, final))

    backup_root = staging_root / "_backup"
    previous: dict[Path, Path | None] = {}
    promoted: list[Path] = []
    try:
        for _, final in targets:
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                backup = backup_root / final.relative_to(target_root)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(final, backup)
                previous[final] = backup
            else:
                previous[final] = None

        for staged, final in targets:
            temp_final = final.with_name(final.name + f".bootstrap-{os.getpid()}")
            shutil.copy2(staged, temp_final)
            os.replace(temp_final, final)
            promoted.append(final)
    except Exception:
        for final in reversed(promoted):
            backup = previous.get(final)
            if backup is None:
                final.unlink(missing_ok=True)
            elif backup.exists():
                restore_tmp = final.with_name(final.name + f".restore-{os.getpid()}")
                shutil.copy2(backup, restore_tmp)
                os.replace(restore_tmp, final)
        raise


def migrate(
    *,
    source_root: Path,
    target_root: Path,
    target_date: str,
    days: int = 30,
    apply: bool = False,
    runtime_root: Path | None = None,
    history_scope: str = "warm-start",
) -> dict[str, Any]:
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    runtime_root = (runtime_root or target_root.parent / "runtime").resolve()

    if history_scope not in {"warm-start", "full-source"}:
        raise ValueError(f"unsupported history_scope={history_scope!r}")
    if history_scope == "full-source":
        audit, frames = _audit_full_source(source_root)
    else:
        audit, frames = _audit_window(source_root, target_date, days)
    source_hashes = {}
    for task in TASK_MODELS:
        pred_path, actual_path = _canonical_paths(source_root, task)
        source_hashes[f"{task}_prediction"] = _sha256(pred_path) if pred_path.exists() else None
        source_hashes[f"{task}_actual"] = _sha256(actual_path) if actual_path.exists() else None

    result: dict[str, Any] = {
        "status": "AUDIT_PASS" if audit["status"] == "PASS" else "AUDIT_FAIL",
        "applied": False,
        "contract": CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "target_root": str(target_root),
        "target_date": target_date,
        "days": days,
        "history_scope": history_scope,
        "audit": audit,
        "source_sha256": source_hashes,
    }
    if audit["status"] != "PASS":
        return result

    runtime_root.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix="bootstrap_96_ledger_", dir=str(runtime_root))
    )
    staging_root = staging_parent / "ledger"
    try:
        result["stage_results"] = _stage(
            staging_root, source_root, target_root, frames,
            preserve_provenance=(history_scope == "full-source"),
        )
        readiness = _readiness(staging_root, target_date, days)
        result["readiness"] = readiness
        if readiness["status"] != "PASS":
            result["status"] = "STAGING_READINESS_FAIL"
            return result

        # Full-source mode is intentionally useful as a read-only migration
        # proof.  Compute the exact overlap/new-day accounting and validate the
        # staged result, but never touch target_root unless --apply is explicit.
        if history_scope == "full-source":
            source_days = set()
            target_days = set()
            for task, (pred, actual) in frames.items():
                source_days.update(pred.get("target_day", pd.Series(dtype=str)).astype(str))
                source_days.update(actual.get("target_day", pd.Series(dtype=str)).astype(str))
                pred_path, actual_path = _canonical_paths(target_root, task)
                for path in (pred_path, actual_path):
                    if path.exists():
                        existing = pd.read_parquet(path)
                        target_days.update(existing.get("target_day", pd.Series(dtype=str)).astype(str))
            overlap = sorted(source_days & target_days)
            imported = sorted(source_days - target_days)
            result["source_range"] = {
                "start": min(source_days) if source_days else None,
                "end": max(source_days) if source_days else None,
            }
            result["source_days"] = len(source_days)
            result["imported_days"] = imported
            result["imported_day_count"] = len(imported)
            result["skipped_overlap_days"] = overlap
            result["skipped_overlap_day_count"] = len(overlap)
            result["conflicts"] = {
                "overlap_days": overlap,
                "overlap_day_count": len(overlap),
                "rows_by_task": {},
                "current_production_wins": True,
            }
            for task in TASK_MODELS:
                task_rows = {}
                for kind, stem in (("prediction", "prediction"), ("actual", "actual")):
                    path = target_root / task / kind / f"{stem}_ledger.parquet"
                    if path.exists():
                        frame = pd.read_parquet(path)
                        task_rows[kind] = int(frame.get("target_day", pd.Series(dtype=str)).astype(str).isin(overlap).sum())
                    else:
                        task_rows[kind] = 0
                result["conflicts"]["rows_by_task"][task] = task_rows
            staged_days = set()
            for task in TASK_MODELS:
                for kind in ("prediction", "actual"):
                    path = staging_root / task / kind / f"{'prediction' if kind == 'prediction' else 'actual'}_ledger.parquet"
                    if path.exists():
                        staged = pd.read_parquet(path)
                        staged_days.update(staged.get("target_day", pd.Series(dtype=str)).astype(str))
            result["final_range"] = {
                "start": min(staged_days) if staged_days else None,
                "end": max(staged_days) if staged_days else None,
                "days": len(staged_days),
            }
            result["final_readiness"] = readiness

        if not apply:
            result["status"] = "AUDIT_PASS"
            result["applied"] = False
            return result

        _promote(staging_root, target_root)
        final_readiness = _readiness(target_root, target_date, days)
        result["final_readiness"] = final_readiness
        if final_readiness["status"] != "PASS":
            raise RuntimeError("promoted ledger failed final readiness audit")

        # Staging paths are intentionally temporary and are removed below.
        # Keep stage statistics, but publish only durable production paths and
        # hashes in the migration manifest so deployment audit records never
        # point at deleted runtime files.
        for stage_result in result.get("stage_results", {}).values():
            if isinstance(stage_result, dict):
                stage_result.pop("parquet_path", None)
                stage_result.pop("csv_path", None)

        promoted_paths: dict[str, str] = {}
        promoted_sha256: dict[str, str] = {}
        for task in TASK_MODELS:
            pred_path, actual_path = _canonical_paths(target_root, task)
            for key, path in (
                (f"{task}_prediction", pred_path),
                (f"{task}_actual", actual_path),
            ):
                promoted_paths[key] = str(path)
                promoted_sha256[key] = _sha256(path)
        result["promoted_paths"] = promoted_paths
        result["promoted_sha256"] = promoted_sha256

        result["status"] = "COMPLETE"
        result["applied"] = True
        manifest_path = target_root / "bootstrap_manifest.json"
        manifest_tmp = manifest_path.with_name(
            manifest_path.name + f".tmp-{os.getpid()}"
        )
        manifest_tmp.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest_path)
        result["manifest_path"] = str(manifest_path)
        return result
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ledger", required=True)
    parser.add_argument("--target-ledger", default="outputs/96/ledger")
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--history-scope",
        choices=("warm-start", "full-source"),
        default="warm-start",
        help="warm-start uses the production 30-day selector; full-source audits "
        "the immutable 2025-12-18..2026-08-14 server archive",
    )
    parser.add_argument("--runtime-root", default="outputs/96/runtime")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Promote the audited staging ledger into the production ledger. "
        "Without this flag the command is read-only.",
    )
    args = parser.parse_args()

    result = migrate(
        source_root=Path(args.source_ledger),
        target_root=Path(args.target_ledger),
        target_date=args.target_date,
        days=args.days,
        apply=args.apply,
        runtime_root=Path(args.runtime_root),
        history_scope=args.history_scope,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result["status"] in {"AUDIT_PASS", "COMPLETE"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
