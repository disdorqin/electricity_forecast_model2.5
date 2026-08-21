"""
Ledger backfill pipeline.

Iterates over a date range [start, end], running ledger_predict
for each day to pre-populate the prediction ledger and actual ledger.

This generates the 30-day companion prediction ledger needed for
weight learning on the first production day.

Cutoff safety: even for historical dates, uses only data up to D-1.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipelines.ledger_predict import run_ledger_predict

logger = logging.getLogger(__name__)


def run_ledger_backfill(args: Any) -> dict:
    """
    Main entry for --pipeline ledger_backfill.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: start, end, data_path, epf_v1_root (optional),
        ledger_root, runs_root, max_cpu_workers, max_gpu_workers,
        allow_missing_models, force.

    Returns
    -------
    dict with summary of backfill results.
    """
    start_date = args.start
    end_date = args.end

    if not start_date or not end_date:
        raise ValueError("--start and --end are required for ledger_backfill")

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    if start_dt > end_dt:
        raise ValueError(f"start ({start_date}) must be <= end ({end_date})")

    logger.info(f"=== ledger_backfill: {start_date} → {end_date} ===")

    # Generate date range
    date_range = []
    current = start_dt
    while current <= end_dt:
        date_range.append(current.strftime("%Y-%m-%d"))
        current += pd.Timedelta(days=1)

    total_days = len(date_range)
    logger.info(f"Backfill will process {total_days} days")

    summary = {
        "pipeline": "ledger_backfill",
        "start_date": start_date,
        "end_date": end_date,
        "total_days": total_days,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "daily_results": [],
        "completed_days": 0,
        "failed_days": 0,
        "errors": [],
    }

    ledger_root = Path(getattr(args, "ledger_root", None) or "outputs/ledger")
    runs_root = Path(getattr(args, "runs_root", None) or "outputs/runs")

    for i, day in enumerate(date_range):
        logger.info(f"\n--- Backfill day {i+1}/{total_days}: {day} ---")

        # Create a modified args for this day
        import copy
        day_args = copy.copy(args)
        day_args.date = day

        try:
            day_result = run_ledger_predict(day_args)
            summary["daily_results"].append({
                "day": day,
                "status": day_result.get("status", "unknown"),
                "warnings": day_result.get("warnings", []),
            })
            if day_result.get("status") in ("complete", "complete_with_warnings"):
                summary["completed_days"] += 1
            else:
                summary["failed_days"] += 1
                summary["errors"].append(f"{day}: {day_result.get('status')}")

        except Exception as e:
            logger.exception(f"Backfill failed for {day}: {e}")
            summary["daily_results"].append({
                "day": day,
                "status": "error",
                "error": str(e),
            })
            summary["failed_days"] += 1
            summary["errors"].append(f"{day}: {e}")

    # Finalize
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    if summary["failed_days"] == 0:
        summary["status"] = "complete"
    elif summary["completed_days"] > 0:
        summary["status"] = "partial"
    else:
        summary["status"] = "failed"

    # Write summary manifest
    summary_path = runs_root / "backfill_manifest.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Log ledger stats
    _log_ledger_stats(ledger_root, start_date, end_date)

    logger.info(
        f"Backfill complete: {summary['completed_days']}/{total_days} days OK, "
        f"{summary['failed_days']} failed"
    )

    return summary


def _log_ledger_stats(ledger_root: Path, start_date: str, end_date: str):
    """Log current ledger statistics."""
    for task in ["dayahead", "realtime"]:
        pq_path = ledger_root / task / "prediction" / "prediction_ledger.parquet"
        if pq_path.exists():
            df = pd.read_parquet(pq_path)
            n_rows = len(df)
            n_days = df["target_day"].nunique() if "target_day" in df.columns else 0
            n_models = df["model_name"].nunique() if "model_name" in df.columns else 0
            logger.info(
                f"Ledger [{task}]: {n_rows} rows, {n_days} days, {n_models} models"
            )
        else:
            logger.warning(f"Ledger [{task}]: not found")

        act_path = ledger_root / task / "actual" / "actual_ledger.parquet"
        if act_path.exists():
            act_df = pd.read_parquet(act_path)
            logger.info(
                f"Actual [{task}]: {len(act_df)} rows, "
                f"{act_df['target_day'].nunique() if 'target_day' in act_df.columns else 0} days"
            )
