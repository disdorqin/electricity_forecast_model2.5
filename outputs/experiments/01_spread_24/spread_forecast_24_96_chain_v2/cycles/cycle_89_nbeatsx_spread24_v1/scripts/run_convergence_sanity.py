from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
CYCLE = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CYCLE / "src"))

from nbeatsx_spread.audits import audit_holdout_registry, load_holdout_registry  # noqa: E402
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.model.factory import build_model  # noqa: E402
from nbeatsx_spread.training.config import training_config_from_business  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.trainer import Trainer  # noqa: E402
from nbeatsx_spread.losses.paper_mae import PaperMAE  # noqa: E402
from run_business_backtest import run_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-day", default="2026-06-01")
    ap.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    ap.add_argument("--run-dir", type=Path, default=CYCLE / "runs/convergence_sanity")
    args = ap.parse_args()
    config = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    gate = audit_holdout_registry([args.target_day], registry)
    if not gate.passed:
        raise RuntimeError(gate.detail)
    source = CanonicalHourlySource.from_csv(args.data)
    decision = select_device("cuda_if_deterministic_else_cpu", seed=int(config["training"]["seed"]))
    manifest = run_one(args.target_day, source, config, args.run_dir, registry,
                       {"max_steps": 200, "min_steps": 200, "eval_every": 25, "patience_checks": 8, "schedule_total_steps": 1200},
                       device=decision.device)
    run_dir = args.run_dir / args.target_day
    import pandas as pd
    grads = pd.read_csv(run_dir / "gradient_stats.csv")
    curve = pd.read_csv(run_dir / "training_curve.csv")
    clip_fraction = float(grads["clipped"].mean())
    train_start = float(curve["train_loss"].iloc[0])
    train_end = float(curve["train_loss"].iloc[-1])
    val_start = float(curve["validation_mae"].iloc[0])
    val_end = float(curve["validation_mae"].iloc[-1])
    summary = {
        "status": "PASS" if not (clip_fraction > 0.75 and train_end >= train_start and val_end >= val_start) else "B0_BLOCKED_GRADIENT_INSTABILITY",
        "median_grad_norm": float(np.median(grads["grad_norm_pre_clip"])),
        "p90_grad_norm": float(np.percentile(grads["grad_norm_pre_clip"], 90)),
        "max_grad_norm": float(grads["grad_norm_pre_clip"].max()),
        "clip_fraction": clip_fraction,
        "nonfinite_count": int((~np.isfinite(grads["grad_norm_pre_clip"])).sum()),
        "training_loss_trend": {"first": train_start, "last": train_end, "decreased": train_end < train_start},
        "validation_mae_trend": {"first": val_start, "last": val_end, "decreased": val_end < val_start},
        "schedule_total_steps": 1200,
        "actual_run_steps": int(manifest["final_step"]),
        "device_decision": decision.__dict__,
        "warning": "GRADIENT_INSTABILITY_WARNING" if clip_fraction > 0.25 else None,
    }
    (run_dir / "convergence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
