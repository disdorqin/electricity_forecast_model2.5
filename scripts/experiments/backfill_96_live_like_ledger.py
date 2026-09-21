"""Backfill a strict live-like 96-point prediction ledger.

For every target day D this driver builds the same serving snapshot used by a
live run (closed history + D-1 partial as-of + D forecast-only), then invokes
``ledger_predict``.  The resulting prediction/actual ledger can be used by the
30-day fusion weight learner without relying on historical target-day labels as
model inputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sync.build_96_asof_snapshot import build_snapshot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill leak-safe 96-point prediction ledger")
    parser.add_argument("--start", required=True, help="first target day YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last target day YYYY-MM-DD")
    parser.add_argument("--base", default="data/96/model_input/shandong_pmos_96_model_input_clean.parquet")
    parser.add_argument("--remote-full", default="data/96/remote/parquet/epf_pmos_96_full.parquet")
    parser.add_argument("--actual-data-path", default="data/96/authoritative/pmos_96_全量.csv")
    parser.add_argument("--ledger-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--rt-cutoff-hour", type=int, default=15)
    parser.add_argument("--training-months", type=int, default=3)
    parser.add_argument("--timemixer-epochs", type=int, default=80)
    parser.add_argument("--timemixer-patience", type=int, default=15)
    parser.add_argument("--max-cpu-workers", type=int, default=2)
    parser.add_argument("--max-gpu-workers", type=int, default=1)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="build/audit snapshots but do not execute models")
    parser.add_argument("--unit-id", default=None)
    args = parser.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if start > end:
        parser.error("--start must be <= --end")

    base = (PROJECT_ROOT / args.base).resolve()
    remote = (PROJECT_ROOT / args.remote_full).resolve()
    actual = (PROJECT_ROOT / args.actual_data_path).resolve()
    ledger_root = (PROJECT_ROOT / args.ledger_root).resolve()
    runs_root = (PROJECT_ROOT / args.runs_root).resolve()
    snapshot_root = runs_root / "asof_inputs"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    ledger_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "model_artifacts" / "LightGBM").mkdir(parents=True, exist_ok=True)

    failures: list[tuple[str, int]] = []
    for target in pd.date_range(start, end, freq="D"):
        target_str = target.strftime("%Y-%m-%d")
        decision_str = (target - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        snapshot = snapshot_root / f"{target_str}.parquet"
        audit = build_snapshot(
            base_path=base,
            remote_path=remote,
            decision_day=decision_str,
            target_day=target_str,
            rt_cutoff_hour=args.rt_cutoff_hour,
            output_path=snapshot,
            unit_id=args.unit_id,
        )
        if audit["target_price_actual_nonnull"] != 0:
            raise RuntimeError(f"Leakage audit failed for {target_str}: {audit}")

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--pipeline", "ledger_predict",
            "--target", "both",
            "--date", target_str,
            "--resolution", "15min",
            "--data-path", str(snapshot),
            "--actual-data-path", str(actual),
            "--ledger-root", str(ledger_root),
            "--runs-root", str(runs_root),
            "--realtime-cutoff-hour", str(args.rt_cutoff_hour),
            "--training-months", str(args.training_months),
            "--timemixer-epochs", str(args.timemixer_epochs),
            "--timemixer-patience", str(args.timemixer_patience),
            "--max-cpu-workers", str(args.max_cpu_workers),
            "--max-gpu-workers", str(args.max_gpu_workers),
        ]
        env = os.environ.copy()
        env.setdefault("OPTIM_NUM_WORKERS", "0")
        env.setdefault(
            "LightGBM_MODEL_PATH",
            str((runs_root / "model_artifacts" / "LightGBM" / "best_model_{}.pkl").resolve()),
        )
        print(f"\n=== live-like ledger day {target_str} (decision={decision_str}) ===", flush=True)
        if args.dry_run:
            print({"snapshot": str(snapshot), "audit": audit, "command": cmd})
            continue
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
        if completed.returncode != 0:
            failures.append((target_str, completed.returncode))
            if not args.continue_on_error:
                break

    if failures:
        print({"status": "FAIL", "failures": failures})
        return 2
    print({
        "status": "PASS",
        "start": args.start,
        "end": args.end,
        "ledger_root": str(ledger_root),
        "rt_cutoff_hour": args.rt_cutoff_hour,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
