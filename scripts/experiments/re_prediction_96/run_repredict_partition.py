"""Run isolated LightGBM/TimesFM 96-point re-prediction partitions.

This experiment intentionally writes only below ``outputs/experiments``.  It
does not modify the production ledger and does not run weight/fuse stages.
Each worker owns a LightGBM model path to avoid cross-process overwrite.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable


def dates_between(start: str, end: str):
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while current <= stop:
        yield current.isoformat()
        current += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-store-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    worker = str(args.worker)
    log_dir = output_root / "worker_logs"
    status_dir = output_root / "worker_status"
    model_dir = output_root / "models" / f"worker_{worker}"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    common = [
        PYTHON,
        str(ROOT / "main.py"),
        "--pipeline", "ledger_predict",
        "--resolution", "15min",
        "--models", "lightgbm,timesfm",
        "--data-path", "data/96/model_input/shandong_pmos_96_model_input_clean.parquet",
        "--actual-data-path", "data/96/authoritative/pmos_96_全量.csv",
        "--output-profile", "feature_store",
        "--ledger-root", str(output_root / "ledger"),
        "--runs-root", str(output_root / "runs"),
        "--feature-store-root", args.feature_store_root,
        "--feature-store-mode", "materialized",
        "--resource-mode", "split_process",
        "--max-cpu-workers", "1",
        "--max-gpu-workers", "1",
        "--training-months", "12",
        "--lgbm-training-months-candidates", "6,9,12",
        "--lgbm-window-selection-metric", "composite",
        "--lgbm-window-mae-weight", "0.25",
        "--force",
    ]

    env = os.environ.copy()
    env.update({
        "EFM3_CPU_THREAD_BUDGET": "24",
        "OMP_NUM_THREADS": "24",
        "MKL_NUM_THREADS": "24",
        "OPENBLAS_NUM_THREADS": "24",
        "NUMEXPR_NUM_THREADS": "24",
        "LightGBM_MODEL_PATH": str(model_dir / "best_model_{}.pkl"),
    })

    statuses = []
    started = time.time()
    for target_day in dates_between(args.start, args.end):
        command = common + ["--date", target_day]
        log_path = log_dir / f"{target_day}.log"
        day_started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row = {
            "target_day": target_day,
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.time() - day_started, 3),
            "log": str(log_path),
        }
        statuses.append(row)
        (status_dir / f"{target_day}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if completed.returncode != 0:
            break

    summary = {
        "worker": worker,
        "start": args.start,
        "end": args.end,
        "status": "complete" if statuses and all(x["returncode"] == 0 for x in statuses) and len(statuses) == len(list(dates_between(args.start, args.end))) else "failed",
        "days_completed": len(statuses),
        "elapsed_seconds": round(time.time() - started, 3),
        "days": statuses,
    }
    (status_dir / f"worker_{worker}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
