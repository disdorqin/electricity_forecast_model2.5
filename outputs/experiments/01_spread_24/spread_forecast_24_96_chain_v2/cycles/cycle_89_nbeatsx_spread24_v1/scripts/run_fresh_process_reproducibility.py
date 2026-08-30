from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove same-day formal B0 reproducibility in two fresh processes.")
    parser.add_argument("--target-day", default="2026-06-01")
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1] / "runs/reproducibility_gate")
    args = parser.parse_args()
    cycle = Path(__file__).resolve().parents[1]
    args.run_root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("process_a", "process_b"):
        run_dir = args.run_root / suffix
        command = [
            sys.executable, str(cycle / "scripts/run_business_backtest.py"),
            "--target-day", args.target_day, "--run-dir", str(run_dir),
        ]
        subprocess.run(command, cwd=cycle, check=True)
        day_dir = run_dir / args.target_day
        rows = list(csv.DictReader((day_dir / "target_day_prediction.csv").open(encoding="utf-8")))
        manifest = json.loads((day_dir / "manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((day_dir / "provenance.json").read_text(encoding="utf-8"))
        outputs.append({
            "run_dir": str(run_dir),
            "prediction_sha256": file_hash(day_dir / "target_day_prediction.csv"),
            "split_sha256": file_hash(day_dir / "split_manifest.json"),
            "config_sha256": provenance["config_sha256"],
            "initial_state_sha256": provenance["model_initial_state_sha256"],
            "best_step": manifest["best_step"],
            "row_count": len(rows),
        })
    first, second = outputs
    prediction_equal = first["prediction_sha256"] == second["prediction_sha256"]
    passed = (
        prediction_equal
        and first["split_sha256"] == second["split_sha256"]
        and first["config_sha256"] == second["config_sha256"]
        and first["initial_state_sha256"] == second["initial_state_sha256"]
        and first["best_step"] == second["best_step"]
        and first["row_count"] == second["row_count"] == 24
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "target_day": args.target_day,
        "fresh_process_runs": outputs,
        "same_prediction_sha256": prediction_equal,
        "prediction_tolerance": 1e-7,
        "same_initial_state_hash": first["initial_state_sha256"] == second["initial_state_sha256"],
        "same_best_step": first["best_step"] == second["best_step"],
    }
    (args.run_root / "reproducibility_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
