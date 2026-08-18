#!/usr/bin/env python3
"""Run the expensive, prediction-only 96-point backtest on a GPU server.

This runner deliberately does not run ledger_weight, ledger_fuse, the
classifier, or final_outputs.  It creates a frozen model prediction ledger
and an authoritative actual ledger that can later be copied back for cheap,
repeatable learner experiments.

Typical server usage::

    python scripts/server/run_96_prediction_backtest.py \
      --data-path data/96/model_input/pmos_96_model_input_clean.xlsx \
      --actual-data-path data/96/authoritative/pmos_96_全量.csv \
      --report-start 2026-01-01 --end 2026-08-15

For a one-day timing smoke test, omit the prewarm window::

    python scripts/server/run_96_prediction_backtest.py \
      --data-path ... --actual-data-path ... \
      --report-start 2026-01-01 --end 2026-01-01 --no-prewarm

The script is resumable.  It skips a day only when both DA and RT contain the
complete canonical model pool, exactly 96 slots, and no NaN predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS  # noqa: E402
from pipelines.ledger_full_range import _prediction_day_audit  # noqa: E402
from pipelines.prediction_ledger import load_actual_ledger  # noqa: E402
from utils.data_loader import load_table  # noqa: E402
from utils.resolution import QUARTER  # noqa: E402


PREDICTION_START = "2025-12-18"
REPORT_START = "2026-01-01"
END_DATE = "2026-08-15"
POLLUTION_NAMES = {
    "shandong_pmos_96_model_input.xlsx",
    "shandong_pmos_96_full_v2.xlsx",
}
PRICE_ALIASES = {
    "dayahead": [
        "日前电价", "日前出清电价", "日前出清价格",
        "day_ahead_clearing_price", "dayahead_price", "da_price",
    ],
    "realtime": [
        "实时电价", "实时出清电价", "实时出清价格",
        "realtime_price", "rt_price",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": _sha256(path),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": _git_commit(),
        "timesfm_device": os.environ.get("TIMESFM_DEVICE", ""),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def _find_timestamp(df: pd.DataFrame) -> str:
    for column in ("时刻", "ds", "timestamp", "time", "datetime"):
        if column in df.columns:
            return column
    if {"market_date", "时段"}.issubset(df.columns):
        return "__market_slot__"
    raise ValueError("96-point model/actual source has no timestamp column")


def _timestamp_series(df: pd.DataFrame, ts_col: str) -> pd.Series:
    if ts_col != "__market_slot__":
        return pd.to_datetime(df[ts_col], errors="coerce")
    base = pd.to_datetime(df["market_date"], errors="coerce").dt.normalize()
    parts = df["时段"].astype(str).str.split(":", n=1, expand=True)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    return base + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def _business_day_series(df: pd.DataFrame, ts_col: str) -> pd.Series:
    values = _timestamp_series(df, ts_col)
    if values.isna().any():
        raise ValueError(f"timestamp column {ts_col!r} contains NaN/unparseable values")
    # Native 96-point business days are p1=00:15 .. p96=24:00.
    return values.dt.normalize().where(
        ~(values.dt.hour.eq(0) & values.dt.minute.eq(0)),
        values.dt.normalize() - pd.Timedelta(days=1),
    ).dt.strftime("%Y-%m-%d")


def _validate_source(path: Path, label: str, *, require_prices: bool) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} source does not exist: {path}")
    resolved = path.resolve()
    if resolved.name.lower() in {name.lower() for name in POLLUTION_NAMES}:
        raise ValueError(
            f"Refusing known historical-invalid source for {label}: {resolved}. "
            "Build/copy a clean 96-point source before starting the server run."
        )

    df = load_table(path)
    ts_col = _find_timestamp(df)
    df = df.copy()
    df["_business_day"] = _business_day_series(df, ts_col)
    if "business_period" in df.columns:
        slots = pd.to_numeric(df["business_period"], errors="coerce")
    elif "period_no" in df.columns:
        slots = pd.to_numeric(df["period_no"], errors="coerce")
    else:
        ts = _timestamp_series(df, ts_col)
        minutes = ts.dt.hour * 60 + ts.dt.minute
        slots = ((minutes - 15) // 15) + 1
        slots = slots.where(minutes != 0, 96)
    df["_slot"] = slots

    counts = df.groupby("_business_day")["_slot"].nunique()
    bad_days = counts[counts != 96]
    if not bad_days.empty:
        raise ValueError(
            f"{label} source is not native 96-point data; bad day counts: "
            f"{bad_days.head(5).to_dict()}"
        )
    if not df["_slot"].between(1, 96).all():
        raise ValueError(f"{label} source has slots outside 1..96")
    duplicate_slots = df.groupby(["_business_day", "_slot"]).size()
    duplicate_slots = duplicate_slots[duplicate_slots > 1]
    if not duplicate_slots.empty:
        raise ValueError(
            f"{label} source has duplicate rows for business-day/slot: "
            f"{duplicate_slots.head(5).to_dict()}"
        )

    price_columns: dict[str, str] = {}
    for task, aliases in PRICE_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                price_columns[task] = alias
                break
        if require_prices and task not in price_columns:
            raise ValueError(
                f"{label} source has no {task} actual-price column; checked {aliases}"
            )

    # If the source contains actual/forecast feature pairs, reject the known
    # copy pattern.  The clean target-day input must not have actual==forecast
    # across a material fraction of rows.
    duplicate_pairs: list[str] = []
    columns = set(df.columns)
    for col in list(columns):
        if col.endswith("实际值"):
            forecast = col[:-3] + "预测值"
        elif col.startswith("actual_"):
            forecast = "fcast_" + col[len("actual_"):]
        else:
            continue
        if forecast not in columns:
            continue
        actual = pd.to_numeric(df[col], errors="coerce")
        predicted = pd.to_numeric(df[forecast], errors="coerce")
        # A constant zero feature (for example the test-unit series in the
        # authoritative table) is a real degenerate signal, not evidence that
        # the crawler copied a forecast into the actual column.  Exclude rows
        # where both sides are zero from the copy-pattern audit, while still
        # checking every non-zero observed value.
        mask = (
            actual.notna()
            & predicted.notna()
            & ((actual.abs() > 1e-12) | (predicted.abs() > 1e-12))
        )
        if mask.any() and float((actual[mask] == predicted[mask]).mean()) > 0.01:
            duplicate_pairs.append(f"{col}=={forecast}")
    if duplicate_pairs:
        raise ValueError(
            f"{label} source fails actual!=forecast authenticity gate: {duplicate_pairs}"
        )

    summary = {
        "label": label,
        "fingerprint": _file_fingerprint(path),
        "timestamp_column": ts_col,
        "rows": int(len(df)),
        "days": int(df["_business_day"].nunique()),
        "min_business_day": str(df["_business_day"].min()),
        "max_business_day": str(df["_business_day"].max()),
        "price_columns": price_columns,
    }
    return summary


def _run_day(args: argparse.Namespace, target_day: str, log_path: Path) -> tuple[int, float]:
    ledger_root = Path(args.output_root) / "ledger"
    runs_root = Path(args.output_root) / "runs"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--pipeline", "ledger_predict",
        "--date", target_day,
        "--resolution", "15min",
        "--data-path", str(Path(args.data_path).resolve()),
        "--actual-data-path", str(Path(args.actual_data_path).resolve()),
        "--output-profile", "feature_store",
        "--feature-store-mode", "raw",
        "--ledger-root", str(ledger_root),
        "--runs-root", str(runs_root),
        "--realtime-cutoff-hour", "14",
        "--max-cpu-workers", str(args.max_cpu_workers),
        "--max-gpu-workers", str(args.max_gpu_workers),
        "--training-months", str(args.training_months),
        "--seed", str(args.seed),
    ]
    if args.deterministic:
        command.append("--deterministic")
    if args.force:
        command.append("--force")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        log.write(f"ENV: RT916_TRAIN_STEPS={args.rt916_train_steps}\n\n")
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "TIMESFM_DEVICE": os.environ.get("TIMESFM_DEVICE", "cpu"),
                "RT916_TRAIN_STEPS": str(args.rt916_train_steps),
            },
            check=False,
        )
    return result.returncode, time.perf_counter() - started


def _actual_day_audit(ledger_root: Path, target_date: str) -> tuple[bool, list[str]]:
    """Require a complete, non-null 96-point actual ledger for both tasks."""
    reasons: list[str] = []
    for task in ("dayahead", "realtime"):
        frame = load_actual_ledger(ledger_root, task, [target_date])
        if frame.empty:
            reasons.append(f"{task}: actual ledger empty")
            continue
        day_col = "target_day" if "target_day" in frame.columns else "business_day"
        frame = frame[frame[day_col].astype(str) == str(target_date)].copy()
        slot_col = "business_period" if "business_period" in frame.columns else "hour_business"
        if frame.empty:
            reasons.append(f"{task}: actual target day {target_date} absent")
            continue
        if slot_col not in frame.columns:
            reasons.append(f"{task}: actual slot column missing")
            continue
        if len(frame) != 96:
            reasons.append(f"{task}: actual rows={len(frame)} expected=96")
        if frame[slot_col].isna().any() or frame[slot_col].nunique() != 96:
            reasons.append(
                f"{task}: actual slots={frame[slot_col].nunique()} expected=96"
            )
        if frame[slot_col].duplicated().any():
            reasons.append(f"{task}: duplicate actual slots detected")
        if frame["y_true"].isna().any():
            reasons.append(f"{task}: y_true contains NaN")
    return not reasons, reasons


def _date_list(start: str, end: str, no_prewarm: bool) -> tuple[list[str], str]:
    report_start = pd.Timestamp(start)
    effective_start = report_start if no_prewarm else report_start - pd.Timedelta(days=14)
    dates = pd.date_range(effective_start, pd.Timestamp(end), freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates], effective_start.strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Clean 96-point model-input source")
    parser.add_argument("--actual-data-path", required=True, help="Authoritative 96-point price/actual source")
    parser.add_argument("--report-start", default=REPORT_START)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--output-root", default="outputs/96/feature_store")
    parser.add_argument("--no-prewarm", action="store_true", help="Do not add the 14-day learner prewarm window")
    parser.add_argument("--limit-days", type=int, default=0, help="Run only the first N dates; useful for smoke timing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cpu-workers", type=int, default=2)
    parser.add_argument("--max-gpu-workers", type=int, default=1)
    parser.add_argument("--training-months", type=int, default=12)
    parser.add_argument(
        "--rt916-train-steps",
        type=int,
        default=24,
        help="RT916 training stride for the server run (24 is the validated speed/quality setting).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    if pd.Timestamp(args.report_start) > pd.Timestamp(args.end):
        parser.error("--report-start must be <= --end")
    if args.max_gpu_workers != 1:
        parser.error("This project currently supports one GPU worker; use --max-gpu-workers 1")
    if args.rt916_train_steps <= 0:
        parser.error("--rt916-train-steps must be positive")

    data_summary = _validate_source(Path(args.data_path), "model", require_prices=False)
    actual_summary = _validate_source(Path(args.actual_data_path), "actual", require_prices=True)
    dates, effective_start = _date_list(args.report_start, args.end, args.no_prewarm)
    if args.limit_days > 0:
        dates = dates[:args.limit_days]

    out_root = Path(args.output_root)
    range_dir = out_root / f"prediction_range_{effective_start}_to_{args.end}"
    log_dir = range_dir / "logs"
    range_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = range_dir / "prediction_range_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "stage": "prediction_only",
        "resolution": "15min",
        "report_start": args.report_start,
        "effective_start": effective_start,
        "end": args.end,
        "total_dates": len(dates),
        "completed_dates": 0,
        "skipped_dates": 0,
        "failed_dates": 0,
        "models": {"dayahead": list(DAYAHEAD_MODELS), "realtime": list(REALTIME_MODELS)},
        "cutoff": {"realtime_hour": 14, "realtime_slot": 56},
        "training": {"training_months": args.training_months, "rt916_train_steps": args.rt916_train_steps},
        "data": {"model": data_summary, "actual": actual_summary},
        "runtime": _runtime_info(),
        "daily": [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger_root = out_root / "ledger"
    for index, target_day in enumerate(dates, start=1):
        print(f"[{index}/{len(dates)}] prediction {target_day}", flush=True)
        prediction_ok, prediction_reasons = _prediction_day_audit(ledger_root, target_day, QUARTER)
        actual_ok, actual_reasons = _actual_day_audit(ledger_root, target_day)
        if prediction_ok and actual_ok and not args.force:
            entry = {
                "date": target_day,
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "prediction_audit": "PASS",
                "actual_audit": "PASS",
            }
            manifest["skipped_dates"] += 1
            manifest["daily"].append(entry)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        return_code, elapsed = _run_day(args, target_day, log_dir / f"{target_day}.log")
        prediction_ok, prediction_reasons = _prediction_day_audit(ledger_root, target_day, QUARTER)
        actual_ok, actual_reasons = _actual_day_audit(ledger_root, target_day)
        audit_reasons = prediction_reasons + actual_reasons
        ok = return_code == 0 and prediction_ok and actual_ok
        entry = {
            "date": target_day,
            "status": "complete" if ok else "failed",
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 2),
            "prediction_audit": "PASS" if prediction_ok else "FAIL",
            "actual_audit": "PASS" if actual_ok else "FAIL",
            "audit_reasons": audit_reasons,
            "log": str(log_dir / f"{target_day}.log"),
        }
        manifest["daily"].append(entry)
        if ok:
            manifest["completed_dates"] += 1
        else:
            manifest["failed_dates"] += 1
            manifest["status"] = "failed"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"FAILED {target_day}: {audit_reasons}", flush=True)
            if not args.force:
                break

        observed = [x["elapsed_seconds"] for x in manifest["daily"] if x.get("status") == "complete" and x["elapsed_seconds"] > 0]
        if observed:
            manifest["observed_seconds_per_day"] = float(np.mean(observed))
            remaining = len(dates) - manifest["completed_dates"] - manifest["skipped_dates"]
            manifest["estimated_remaining_hours"] = float(np.mean(observed) * max(remaining, 0) / 3600.0)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if manifest["failed_dates"] == 0 and manifest["completed_dates"] + manifest["skipped_dates"] == len(dates):
        manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "effective_start": effective_start,
        "report_start": args.report_start,
        "end": args.end,
        "completed": manifest["completed_dates"],
        "skipped": manifest["skipped_dates"],
        "failed": manifest["failed_dates"],
        "observed_seconds_per_day": manifest.get("observed_seconds_per_day"),
        "estimated_remaining_hours": manifest.get("estimated_remaining_hours"),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
