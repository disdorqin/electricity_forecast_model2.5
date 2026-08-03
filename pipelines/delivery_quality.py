"""
Delivery quality checks — ledger window validation, submission validation,
next-day readiness.

All checks are pure validation: no GPU, no model inference, no ledger writes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SUBMISSION_COLUMNS = [
    "business_day", "ds", "hour_business", "period",
    "dayahead_price", "realtime_price",
]
# 96 点提交契约（计划 §12）：business_period(1..96) 取代 hour_business
SUBMISSION_COLUMNS_96 = [
    "business_day", "ds", "business_period", "period",
    "dayahead_price", "realtime_price",
]

# ---------------------------------------------------------------------------
# Expected grid builder
# ---------------------------------------------------------------------------


def build_expected_ledger_grid(start_date: str, days: int, task: str, resolution=None) -> pd.DataFrame:
    """Build the full expected row grid for a task's ledger window.

    Parameters
    ----------
    start_date : str
        The target date (D). Window is D-30 .. D-1.
    days : int
        Number of days in the window (expected 30).
    task : str
        ``"dayahead"`` or ``"realtime"``.
    resolution : Resolution, optional
        Default HOURLY（24 行/天）。96 点用 QUARTER。

    Returns
    -------
    pd.DataFrame with columns [business_day, model_name, slot_column]
    representing every row that must exist.
    """
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    slot_col = res.slot_column
    if task == "dayahead":
        models = ["lightgbm", "timesfm", "timemixer"]
    else:
        models = ["timesfm", "sgdfnet", "timemixer", "rt916"]

    start_dt = pd.Timestamp(start_date)
    window_end = start_dt - pd.Timedelta(days=1)
    window_start = start_dt - pd.Timedelta(days=days)

    date_range = pd.date_range(start=window_start, end=window_end, freq="D")
    rows = []
    for d in date_range:
        d_str = d.strftime("%Y-%m-%d")
        for model in models:
            for h in range(1, res.slots_per_day + 1):
                rows.append({
                    "business_day": d_str,
                    "model_name": model,
                    slot_col: h,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ledger window validation  (section 3 in design)
# ---------------------------------------------------------------------------


def validate_ledger_window(
    target_date: str,
    ledger_root: str | Path,
    days: int = 30,
    resolution=None,
) -> dict:
    """Strictly validate D-30..D-1 ledger coverage.

    Checks all four ledger files for complete daily coverage:
      - dayahead prediction  (3 models x 24h)
      - realtime prediction  (4 models x 24h)
      - dayahead actual      (24h)
      - realtime actual      (24h)

    Returns a dict with status PASS/FAIL, errors, warnings, and summary.
    """
    ledger_root = Path(ledger_root)
    errors: list[dict] = []
    warnings: list[str] = []

    ledger_paths = {
        "dayahead prediction": ledger_root / "dayahead" / "prediction" / "prediction_ledger.parquet",
        "dayahead actual": ledger_root / "dayahead" / "actual" / "actual_ledger.parquet",
        "realtime prediction": ledger_root / "realtime" / "prediction" / "prediction_ledger.parquet",
        "realtime actual": ledger_root / "realtime" / "actual" / "actual_ledger.parquet",
    }

    # Build expected grid
    da_pred_grid = build_expected_ledger_grid(target_date, days, "dayahead", resolution)
    rt_pred_grid = build_expected_ledger_grid(target_date, days, "realtime", resolution)
    actual_grid = _build_actual_grid(target_date, days, resolution)

    # Dayahead prediction
    _check_prediction_ledger(
        ledger_paths["dayahead prediction"],
        "dayahead prediction",
        da_pred_grid,
        errors,
        resolution=resolution,
    )

    # Realtime prediction
    _check_prediction_ledger(
        ledger_paths["realtime prediction"],
        "realtime prediction",
        rt_pred_grid,
        errors,
        resolution=resolution,
    )

    # Dayahead actual
    _check_actual_ledger(
        ledger_paths["dayahead actual"],
        "dayahead actual",
        actual_grid,
        errors,
        resolution=resolution,
    )

    # Realtime actual
    _check_actual_ledger(
        ledger_paths["realtime actual"],
        "realtime actual",
        actual_grid,
        errors,
        resolution=resolution,
    )

    # Build summary counts
    summary = _build_summary_counts(
        ledger_paths, target_date, days, da_pred_grid, rt_pred_grid,
        resolution,
    )

    result: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "target_date": target_date,
        "window_start": (pd.Timestamp(target_date) - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
        "window_end": (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "missing": errors,
    }

    return result


def _build_actual_grid(start_date: str, days: int, resolution=None) -> pd.DataFrame:
    """Build expected grid for actual ledger (no model dimension)."""
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    slot_col = res.slot_column
    start_dt = pd.Timestamp(start_date)
    window_end = start_dt - pd.Timedelta(days=1)
    window_start = start_dt - pd.Timedelta(days=days)

    date_range = pd.date_range(start=window_start, end=window_end, freq="D")
    rows = []
    for d in date_range:
        d_str = d.strftime("%Y-%m-%d")
        for h in range(1, res.slots_per_day + 1):
            rows.append({"business_day": d_str, slot_col: h})
    return pd.DataFrame(rows)


def _check_prediction_ledger(
    path: Path,
    label: str,
    expected_grid: pd.DataFrame,
    errors: list,
    resolution=None,
) -> None:
    """Check prediction ledger against expected grid."""
    if not path.exists():
        errors.append({
            "ledger": label,
            "error": f"file not found: {path}",
        })
        return

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        errors.append({
            "ledger": label,
            "error": f"cannot read parquet: {exc}",
        })
        return

    _check_ledger_against_grid(df, label, expected_grid, errors, is_prediction=True, resolution=resolution)


def _check_actual_ledger(
    path: Path,
    label: str,
    expected_grid: pd.DataFrame,
    errors: list,
    resolution=None,
) -> None:
    """Check actual ledger against expected grid."""
    if not path.exists():
        errors.append({
            "ledger": label,
            "error": f"file not found: {path}",
        })
        return

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        errors.append({
            "ledger": label,
            "error": f"cannot read parquet: {exc}",
        })
        return

    _check_ledger_against_grid(df, label, expected_grid, errors, is_prediction=False, resolution=resolution)


def _check_ledger_against_grid(
    df: pd.DataFrame,
    label: str,
    expected_grid: pd.DataFrame,
    errors: list,
    is_prediction: bool,
    resolution=None,
) -> None:
    """Compare actual ledger counts against expected grid."""
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    slot_col = res.slot_column
    # Determine date column
    date_col = "target_day" if "target_day" in df.columns else "business_day"
    if date_col not in df.columns:
        errors.append({
            "ledger": label,
            "error": f"no '{date_col}' column found (columns: {list(df.columns)})",
        })
        return

    # Normalise date column to string
    df = df.copy()
    df["_date_str"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    # Normalise slot column to int
    if slot_col in df.columns:
        df[slot_col] = df[slot_col].astype(int)

    # Count existing rows
    if is_prediction:
        if "model_name" not in df.columns:
            errors.append({
                "ledger": label,
                "error": f"no 'model_name' column in prediction ledger",
            })
            return

        counts = (
            df.groupby(["_date_str", "model_name"])[slot_col]
            .nunique()
            .reset_index(name="n_hours")
        )
    else:
        # 96 点按 slot_col=business_period 计 96 行；hourly 按 hour_business 计 24 行
        _act_slot = slot_col if slot_col in df.columns else "hour_business"
        counts = (
            df.groupby(["_date_str"])[_act_slot]
            .nunique()
            .reset_index(name="n_hours")
        )

    # Check each expected row（96 点按 slot_col=business_period 期望 96 行，不是 hour_business=24）
    if is_prediction:
        grid_slot = slot_col if slot_col in expected_grid.columns else "hour_business"
        for (day, model), grp in expected_grid.groupby(["business_day", "model_name"]):
            n_expected = len(grp[grid_slot].unique())
            match = counts[(counts["_date_str"] == day) & (counts["model_name"] == model)]
            if match.empty:
                errors.append({
                    "ledger": label,
                    "day": day,
                    "model": model,
                    "hour_business": "all",
                    "error": f"missing all {int(n_expected)} rows — model completely absent",
                    "detail": f"0/{int(n_expected)}",
                })
            else:
                n_actual = int(match.iloc[0]["n_hours"])
                if n_actual < n_expected:
                    errors.append({
                        "ledger": label,
                        "day": day,
                        "model": model,
                        "hour_business": f"only {n_actual}/{int(n_expected)} hours",
                        "error": f"incomplete coverage: {n_actual}/{int(n_expected)} hours",
                        "detail": f"{n_actual}/{int(n_expected)}",
                    })
    else:
        for day in expected_grid["business_day"].unique():
            n_expected = res.slots_per_day
            match = counts[counts["_date_str"] == day]
            if match.empty:
                errors.append({
                    "ledger": label,
                    "day": day,
                    "error": f"missing all {n_expected} rows — day completely absent",
                    "detail": f"0/{n_expected}",
                })
            else:
                n_actual = int(match.iloc[0]["n_hours"])
                if n_actual < n_expected:
                    errors.append({
                        "ledger": label,
                        "day": day,
                        "error": f"incomplete coverage: {n_actual}/{n_expected} hours",
                        "detail": f"{n_actual}/{n_expected}",
                    })


def _build_summary_counts(
    ledger_paths: dict[str, Path],
    target_date: str,
    days: int,
    da_pred_grid: pd.DataFrame,
    rt_pred_grid: pd.DataFrame,
    resolution=None,
) -> dict:
    """Build summary of expected vs actual row counts for each ledger."""
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    summary: dict[str, Any] = {}

    for label_key, label in [
        ("dayahead_prediction_expected_rows", "dayahead prediction"),
    ]:
        summary[label_key] = len(da_pred_grid)

    summary["realtime_prediction_expected_rows"] = len(rt_pred_grid)
    summary["dayahead_actual_expected_rows"] = days * res.slots_per_day
    summary["realtime_actual_expected_rows"] = days * res.slots_per_day

    ledger_labels = {
        "dayahead prediction": "dayahead_prediction_actual_rows",
        "realtime prediction": "realtime_prediction_actual_rows",
        "dayahead actual": "dayahead_actual_actual_rows",
        "realtime actual": "realtime_actual_actual_rows",
    }

    for ll, key in ledger_labels.items():
        path = ledger_paths.get(ll)
        if path and path.exists():
            try:
                df = pd.read_parquet(path)
                summary[key] = len(df)
            except Exception:
                summary[key] = 0
        else:
            summary[key] = 0

    return summary


# ---------------------------------------------------------------------------
# Daily submission validation  (section 3 in design)
# ---------------------------------------------------------------------------


def validate_daily_submission(
    runs_root: str | Path,
    target_date: str,
    allow_degraded: bool = False,
    resolution=None,
) -> dict:
    """Validate a single day's submission_ready.csv and run_manifest.json.

    Parameters
    ----------
    runs_root : str | Path
        Root directory for run outputs (e.g. ``outputs/runs``).
    target_date : str
        Business day YYYY-MM-DD.
    allow_degraded : bool
        If True, DEGRADED_DELIVERED delivery_status is accepted as PASS.
    resolution : Resolution, optional
        Default HOURLY（24 行）。96 点用 QUARTER。

    Returns
    -------
    dict with status, errors, warnings.
    """
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    n_expected = res.slots_per_day
    slot_col = res.slot_column
    runs_root = Path(runs_root)
    errors: list[str] = []
    warnings: list[str] = []

    run_dir = runs_root / target_date
    sub_path = run_dir / "final" / "submission_ready.csv"
    manifest_path = run_dir / "run_manifest.json"

    # 1. File existence
    if not sub_path.exists():
        errors.append(f"submission_ready.csv not found: {sub_path}")
        return _submission_result("FAIL", errors, warnings, sub_path, manifest_path)

    try:
        df = pd.read_csv(sub_path)
    except Exception as exc:
        errors.append(f"cannot read {sub_path}: {exc}")
        return _submission_result("FAIL", errors, warnings, sub_path, manifest_path)

    # 2. Columns exact match（96 点用 business_period 契约，hourly 保持 6 列不变）
    expected_cols = SUBMISSION_COLUMNS_96 if res.label == "15min" else SUBMISSION_COLUMNS
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        errors.append(
            f"column mismatch: expected {expected_cols}, got {actual_cols}"
        )

    # 3. Row count
    if len(df) != n_expected:
        errors.append(f"row count: expected {n_expected}, got {len(df)}")

    # 4. slot 1..N
    if slot_col in df.columns:
        df[slot_col] = pd.to_numeric(df[slot_col], errors="coerce")
        slots = sorted(df[slot_col].dropna().unique())
        if slots != list(range(1, n_expected + 1)):
            errors.append(f"{slot_col} range: expected 1..{n_expected}, got {slots}")
    else:
        errors.append(f"column {slot_col} missing")

    # 5. No duplicate slots
    if slot_col in df.columns:
        dups = df[df[slot_col].duplicated()][slot_col].tolist()
        if dups:
            errors.append(f"duplicate {slot_col}: {dups}")

    # 6. business_day all match target_date
    if "business_day" in df.columns:
        bdays = df["business_day"].unique()
        if len(bdays) != 1 or str(bdays[0]) != target_date:
            errors.append(
                f"business_day mismatch: expected {target_date}, got {bdays}"
            )
    else:
        errors.append("column business_day missing")

    # 7. Last-slot ds is target_date + 1 day 00:00:00
    if slot_col in df.columns and "ds" in df.columns:
        last_slot = n_expected
        h24 = df[df[slot_col] == last_slot]
        if not h24.empty:
            next_day = pd.Timestamp(target_date) + pd.Timedelta(days=1)
            expected_ds_prefix = next_day.strftime("%Y-%m-%d 00:00:00")
            actual_ds = str(h24.iloc[0]["ds"])
            if not actual_ds.startswith(expected_ds_prefix):
                errors.append(
                    f"last-slot ds: expected '{expected_ds_prefix}', got '{actual_ds}'"
                )

    # 8. Price non-null and numeric
    for col in ("dayahead_price", "realtime_price"):
        if col in df.columns:
            null_mask = df[col].isna()
            if null_mask.any():
                bad_hours = df.loc[null_mask, "hour_business"].tolist()
                errors.append(f"{col}: null in hours {bad_hours}")
            try:
                pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError) as exc:
                errors.append(f"{col}: non-numeric — {exc}")
        else:
            errors.append(f"column {col} missing")

    # 9. No _x/_y suffixes
    for col in df.columns:
        if col.endswith("_x") or col.endswith("_y"):
            errors.append(f"suffix column detected: '{col}'")

    # --- Manifest checks ---
    if not manifest_path.exists():
        errors.append(f"run_manifest.json not found: {manifest_path}")
        return _submission_result("FAIL", errors, warnings, sub_path, manifest_path)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        errors.append(f"cannot read manifest: {exc}")
        return _submission_result("FAIL", errors, warnings, sub_path, manifest_path)

    # Check delivery_status
    delivery_status = manifest.get("delivery_status")

    if delivery_status == "FAILED_NO_DELIVERY":
        errors.append(f"delivery_status is FAILED_NO_DELIVERY — no usable output")
        return _submission_result("FAIL", errors, warnings, sub_path, manifest_path)

    if delivery_status == "DEGRADED_DELIVERED":
        if not allow_degraded:
            errors.append(
                f"delivery_status is DEGRADED_DELIVERED — degraded output, "
                f"pass allow_degraded=True to accept"
            )
            return _submission_result("FAIL", errors, warnings, sub_path, manifest_path)

        # DEGRADED_DELIVERED with allow_degraded=True:
        #   - submission_ready.csv structure already checked above (must PASS)
        #   - skip five-stage complete check
        #   - skip manifest errors check
        #   - verify fallback_report exists
        #   - verify manifest.fallback.fallback_used == True
        fb_json = run_dir / "final" / "fallback_report.json"
        fb_md = run_dir / "final" / "fallback_report.md"
        if not fb_json.exists() and not fb_md.exists():
            errors.append(
                f"fallback_report.json/md not found in {run_dir / 'final'}/"
            )

        fb = manifest.get("fallback", {})
        if not fb.get("fallback_used", False):
            errors.append(
                f"manifest.fallback.fallback_used is False — "
                f"expected True for DEGRADED_DELIVERED"
            )

        status = "PASS" if not errors else "FAIL"
        return _submission_result(status, errors, warnings, sub_path, manifest_path)

    # NORMAL or unset delivery_status → strict checks
    stages = manifest.get("stages", {})
    expected_stages = [
        "ledger_predict", "ledger_weight", "ledger_fuse",
        "ledger_classifier", "final_outputs",
    ]

    for stage_name in expected_stages:
        stage = stages.get(stage_name, {})
        stage_status = stage.get("status", "missing")
        # classifier 允许降级：分类器失败时官方输出回退未修正值（计划 §11），
        # complete_with_warnings 视为可接受交付。
        if stage_status != "complete" and not (
            stage_name == "ledger_classifier" and stage_status == "complete_with_warnings"
        ):
            errors.append(
                f"stage '{stage_name}' status={stage_status}, "
                f"expected 'complete'"
            )

    # Manifest errors
    manifest_errors = manifest.get("errors", [])
    if manifest_errors:
        errors.append(f"manifest has {len(manifest_errors)} error(s): {manifest_errors}")

    status = "PASS" if not errors else "FAIL"
    return _submission_result(status, errors, warnings, sub_path, manifest_path)


def _submission_result(
    status: str,
    errors: list,
    warnings: list,
    sub_path: Path,
    manifest_path: Path,
) -> dict:
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "submission_ready_path": str(sub_path),
        "manifest_path": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# Next-day readiness  (section 3 in design)
# ---------------------------------------------------------------------------


def validate_next_day_readiness(
    target_date: str,
    ledger_root: str | Path,
    days: int = 30,
) -> dict:
    """Check whether tomorrow's D-30..D-1 ledger window is already complete.

    Called after today's run completes, to warn if the next day lacks
    sufficient ledger coverage.
    """
    next_date = pd.Timestamp(target_date) + pd.Timedelta(days=1)
    next_date_str = next_date.strftime("%Y-%m-%d")

    result = validate_ledger_window(next_date_str, ledger_root, days=days)
    return result
