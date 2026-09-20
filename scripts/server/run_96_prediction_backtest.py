#!/usr/bin/env python3
"""Run the expensive, prediction-only 96-point backtest on a GPU server.

This runner deliberately does not run ledger_weight, ledger_fuse, the
classifier, or final_outputs.  It creates a frozen model prediction ledger
and an authoritative actual ledger that can later be copied back for cheap,
repeatable learner experiments.

Typical server usage::

    python scripts/server/run_96_prediction_backtest.py \
      --report-start 2026-08-15 --end 2026-09-15

The production defaults are the single persistent model store
``data/96/model_input/shandong_pmos_96_model_input_full.parquet`` and the
canonical authoritative truth table. Each day is synced, frozen into an
immutable D/T snapshot, and routed by FeatureViewBuilder; no materialized clean
input or fixed-hour second trim is required.

For a one-day timing smoke test (prewarm defaults to zero)::

    python scripts/server/run_96_prediction_backtest.py \
      --report-start 2026-08-16 --end 2026-08-16

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


from utils.asof_view_96 import DYNAMIC_PROTOCOL, HISTORICAL_PROXY_PROTOCOL
FORMAL96_PROTOCOLS = {DYNAMIC_PROTOCOL, HISTORICAL_PROXY_PROTOCOL}


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
        "--output-profile", "production",
        "--feature-store-mode", "off",
        "--resource-mode", args.resource_mode,
        "--ledger-root", str(ledger_root),
        "--runs-root", str(runs_root),
        "--realtime-cutoff-hour", "15",
        "--require-target-actual",
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


def _protocol_manifest_audit(
    runs_root: Path,
    target_date: str,
    *,
    resource_mode: str,
) -> tuple[bool, list[str]]:
    """Prove an existing daily run was produced by the current strict 96 protocol.

    Ledger completeness alone is not enough for resume: historical ledgers can
    contain complete predictions produced under an older cutoff/as-of policy.
    """
    reasons: list[str] = []
    manifest_path = runs_root / target_date / "run_manifest.json"
    if not manifest_path.exists():
        return False, [f"protocol manifest missing: {manifest_path}"]
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"protocol manifest unreadable: {exc}"]

    # A later replay-only ledger_full run preserves the original prediction
    # manifest under prediction_provenance. Full single-day runs may instead
    # retain it directly under stages.ledger_predict.  The formal runner
    # accepts Dynamic-v1 and the explicit historical-proxy protocol; the old
    # fixed-cutoff/as-of manifest remains valid only for legacy readers.
    candidates = [raw_manifest]
    nested = raw_manifest.get("prediction_provenance")
    if isinstance(nested, dict):
        candidates.append(nested)
    stage = raw_manifest.get("stages", {}).get("ledger_predict", {})
    if isinstance(stage, dict):
        candidates.append(stage)
    manifest = next(
        (candidate for candidate in candidates
         if candidate.get("serving_protocol") in FORMAL96_PROTOCOLS),
        None,
    )
    if manifest is None:
        manifest = next(
            (candidate for candidate in candidates
             if isinstance(candidate.get("dynamic_snapshot"), dict)
             and candidate["dynamic_snapshot"].get("protocol") in FORMAL96_PROTOCOLS),
            raw_manifest,
        )

    if manifest.get("status") not in {"complete", "complete_with_warnings"}:
        reasons.append(f"manifest status={manifest.get('status')!r}")
    if manifest.get("resolution") != "15min":
        reasons.append(f"resolution={manifest.get('resolution')!r}")
    if manifest.get("serving_protocol") not in FORMAL96_PROTOCOLS:
        reasons.append(f"serving_protocol={manifest.get('serving_protocol')!r}")
    if manifest.get("resource_mode") != resource_mode:
        reasons.append(
            f"resource_mode={manifest.get('resource_mode')!r} expected={resource_mode!r}"
        )

    source = Path(str(manifest.get("model_input_source", "")))
    if source.name != "shandong_pmos_96_model_input_full.parquet":
        reasons.append(f"model_input_source={manifest.get('model_input_source')!r}")

    selected = manifest.get("selected_model_pool", {})
    if list(selected.get("dayahead", [])) != list(DAYAHEAD_MODELS):
        reasons.append("selected dayahead model pool is not canonical")
    if list(selected.get("realtime", [])) != list(REALTIME_MODELS):
        reasons.append("selected realtime model pool is not canonical")

    snapshot = manifest.get("dynamic_snapshot", {})
    if not isinstance(snapshot, dict) or snapshot.get("protocol") not in FORMAL96_PROTOCOLS:
        reasons.append(f"dynamic_snapshot.protocol={snapshot.get('protocol') if isinstance(snapshot, dict) else None!r}")
    if not manifest.get("snapshot_id"):
        reasons.append("snapshot_id missing")
    elif isinstance(snapshot, dict) and snapshot.get("snapshot_id") != manifest.get("snapshot_id"):
        reasons.append("snapshot_id does not match dynamic_snapshot")
    values_raw = snapshot.get("values_path") if isinstance(snapshot, dict) else None
    manifest_raw = snapshot.get("manifest_path") if isinstance(snapshot, dict) else None
    # Do not let Path("") resolve to the current directory: a missing
    # provenance path must fail closed instead of accidentally passing the
    # filesystem existence check.
    snapshot_values = Path(str(values_raw)) if values_raw else None
    snapshot_manifest = Path(str(manifest_raw)) if manifest_raw else None
    if snapshot_values is None or not snapshot_values.exists():
        reasons.append(f"snapshot values missing: {values_raw!r}")
    if snapshot_manifest is None or not snapshot_manifest.exists():
        reasons.append(f"snapshot manifest missing: {manifest_raw!r}")
    view = manifest.get("feature_view", {})
    if not isinstance(view, dict) or view.get("status") != "PASS":
        reasons.append(f"feature_view status={view.get('status') if isinstance(view, dict) else None!r}")
    if isinstance(view, dict) and view.get("target_truth_mask") is not True:
        reasons.append(f"feature_view target_truth_mask={view.get('target_truth_mask')!r}")
    # The dynamic snapshot is the sole visibility contract.  Do not recreate
    # a p60 assertion here: the database may expose any number of RT cells.
    if isinstance(snapshot, dict):
        grid_rows = snapshot.get("grid_rows")
        if grid_rows is not None and int(grid_rows) != 192:
            reasons.append(f"dynamic_snapshot.grid_rows={grid_rows!r} expected=192")
    expected_decision = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if str(snapshot.get("decision_day")) != expected_decision:
        reasons.append(f"snapshot decision_day={snapshot.get('decision_day')!r} expected={expected_decision}")
    if str(snapshot.get("target_day")) != target_date:
        reasons.append(f"snapshot target_day={snapshot.get('target_day')!r}")

    return not reasons, reasons


def _date_list(start: str, end: str, prewarm_days: int = 0) -> tuple[list[str], str]:
    report_start = pd.Timestamp(start)
    prewarm_days = max(0, int(prewarm_days))
    effective_start = report_start - pd.Timedelta(days=prewarm_days)
    dates = pd.date_range(effective_start, pd.Timestamp(end), freq="D")
    return [d.strftime("%Y-%m-%d") for d in dates], effective_start.strftime("%Y-%m-%d")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="data/96/model_input/shandong_pmos_96_model_input_full.parquet",
        help="Single persistent 96-point model store (default: production full parquet)",
    )
    parser.add_argument(
        "--actual-data-path",
        default="data/96/authoritative/pmos_96_全量.csv",
        help="Authoritative 96-point price/actual source",
    )
    parser.add_argument("--report-start", default=REPORT_START)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--output-root", default="outputs/96")
    parser.add_argument(
        "--prewarm-days", type=int, default=0,
        help="Optional extra prediction days before --report-start (default 0).",
    )
    parser.add_argument(
        "--no-prewarm", action="store_true", help=argparse.SUPPRESS,
    )
    parser.add_argument("--limit-days", type=int, default=0, help="Run only the first N dates; useful for smoke timing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--resource-mode",
        choices=["legacy", "split_process"],
        default="legacy",
        help=(
            "Execution mode for the formal replay. legacy is the accepted default; "
            "use split_process only for the server A/B until equivalence and stability are proven."
        ),
    )
    parser.add_argument("--max-cpu-workers", type=int, default=1)
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
    expected_cpu_workers = 2 if args.resource_mode == "split_process" else 1
    if args.max_cpu_workers != expected_cpu_workers:
        parser.error(
            f"resource mode {args.resource_mode!r} requires "
            f"--max-cpu-workers {expected_cpu_workers}"
        )
    if args.rt916_train_steps <= 0:
        parser.error("--rt916-train-steps must be positive")

    data_summary = _validate_source(Path(args.data_path), "model", require_prices=False)
    actual_summary = _validate_source(Path(args.actual_data_path), "actual", require_prices=True)
    prewarm_days = 0 if args.no_prewarm else args.prewarm_days
    dates, effective_start = _date_list(args.report_start, args.end, prewarm_days)
    if args.limit_days > 0:
        dates = dates[:args.limit_days]

    out_root = Path(args.output_root)
    range_dir = out_root / "runs" / f"range_{effective_start}_to_{args.end}_predict"
    log_dir = range_dir / "logs"
    range_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = range_dir / "prediction_range_manifest.json"
    manifest: dict[str, Any] = {
        "status": "running",
        "stage": "prediction_only",
        "resolution": "15min",
        "report_start": args.report_start,
        "effective_start": effective_start,
        "prewarm_days": int(prewarm_days),
        "end": args.end,
        "total_dates": len(dates),
        "completed_dates": 0,
        "skipped_dates": 0,
        "failed_dates": 0,
        "models": {"dayahead": list(DAYAHEAD_MODELS), "realtime": list(REALTIME_MODELS)},
        "serving_protocol": DYNAMIC_PROTOCOL,
        "serving_visibility_source": "FeatureViewBuilder",
        "training": {"training_months": args.training_months, "rt916_train_steps": args.rt916_train_steps},
        "execution": {
            "resource_mode": args.resource_mode,
            "feature_store_mode": "off",
            "cpu_workers": int(args.max_cpu_workers),
            "gpu_workers": int(args.max_gpu_workers),
            "dag_aware": args.resource_mode == "split_process",
            "gpu_serial": True,
        },
        "data": {"model": data_summary, "actual": actual_summary},
        "runtime": _runtime_info(),
        "model_input_contract": "DB sync -> immutable D/T snapshot -> FeatureViewBuilder -> models",
        "forecast_vintage": {
            "status": "UNVERIFIED_LEGACY_VINTAGE",
            "strict_historical_vintage_proven": False,
            "reason": (
                "epf_pmos_96_full is an upserted latest-state table; historical target-day "
                "forecast revisions are not versioned in the canonical table"
            ),
            "audit": "scripts/tests/check_forecast_vintage_96.py",
            "scope": "mechanical production replay only; not strict publication-vintage evidence",
        },
        "daily": [],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger_root = out_root / "ledger"
    for index, target_day in enumerate(dates, start=1):
        print(f"[{index}/{len(dates)}] prediction {target_day}", flush=True)
        prediction_ok, prediction_reasons = _prediction_day_audit(ledger_root, target_day, QUARTER)
        actual_ok, actual_reasons = _actual_day_audit(ledger_root, target_day)
        protocol_ok, protocol_reasons = _protocol_manifest_audit(
            out_root / "runs", target_day, resource_mode=args.resource_mode
        )
        if prediction_ok and actual_ok and protocol_ok and not args.force:
            entry = {
                "date": target_day,
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "prediction_audit": "PASS",
                "actual_audit": "PASS",
                "protocol_audit": "PASS",
            }
            manifest["skipped_dates"] += 1
            manifest["daily"].append(entry)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        return_code, elapsed = _run_day(args, target_day, log_dir / f"{target_day}.log")
        prediction_ok, prediction_reasons = _prediction_day_audit(ledger_root, target_day, QUARTER)
        actual_ok, actual_reasons = _actual_day_audit(ledger_root, target_day)
        protocol_ok, protocol_reasons = _protocol_manifest_audit(
            out_root / "runs", target_day, resource_mode=args.resource_mode
        )
        audit_reasons = prediction_reasons + actual_reasons + protocol_reasons
        ok = return_code == 0 and prediction_ok and actual_ok and protocol_ok
        entry = {
            "date": target_day,
            "status": "complete" if ok else "failed",
            "return_code": return_code,
            "elapsed_seconds": round(elapsed, 2),
            "prediction_audit": "PASS" if prediction_ok else "FAIL",
            "actual_audit": "PASS" if actual_ok else "FAIL",
            "protocol_audit": "PASS" if protocol_ok else "FAIL",
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

    # Daily prediction writes use immutable parts. Compact only after the
    # complete range so replay/learner stages get one canonical ledger file.
    if manifest["failed_dates"] == 0:
        from pipelines.prediction_ledger import compact_ledger
        manifest["ledger_compaction"] = {
            "dayahead_prediction": compact_ledger(ledger_root, "dayahead", "prediction"),
            "realtime_prediction": compact_ledger(ledger_root, "realtime", "prediction"),
            "dayahead_actual": compact_ledger(ledger_root, "dayahead", "actual"),
            "realtime_actual": compact_ledger(ledger_root, "realtime", "actual"),
        }

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
