"""Strict 96-point production-ledger cold-start contract."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from pipelines.ledger_weight import select_complete_training_days
from utils.resolution import HOURLY, QUARTER


def _write_case(
    root: Path,
    *,
    days: int = 30,
    missing_da: str | None = None,
    missing_rt: str | None = None,
    missing_actual_p96: bool = False,
    actual_nan: bool = False,
    mixed_24: bool = False,
) -> None:
    """Create canonical parquet ledgers with one controlled defect."""
    target = pd.Timestamp("2026-03-15")
    quarter_slots = QUARTER.slots_per_day
    hourly_slots = HOURLY.slots_per_day
    for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
        pred_rows: list[dict] = []
        act_rows: list[dict] = []
        for offset in range(2, days + 2):
            day = (target - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            slots = list(range(1, hourly_slots + 1)) if (mixed_24 and offset == 2) else list(range(1, quarter_slots + 1))
            for model in models:
                if model == missing_da and task == "dayahead":
                    continue
                if model == missing_rt and task == "realtime":
                    continue
                for slot in slots:
                    pred_rows.append({
                        "task": task, "target_day": day, "model_name": model,
                        "business_period": slot, "y_pred": float(slot),
                        "resolution": QUARTER.label if len(slots) == quarter_slots else HOURLY.label,
                    })
            actual_slots = list(range(1, quarter_slots + 1))
            if mixed_24 and offset == 2:
                actual_slots = list(range(1, hourly_slots + 1))
            if missing_actual_p96 and offset == 2:
                actual_slots = list(range(1, quarter_slots))
            for slot in actual_slots:
                value = np.nan if actual_nan and offset == 2 and slot == 1 else float(slot)
                act_rows.append({
                    "task": task, "target_day": day, "business_period": slot,
                    "y_true": value,
                    "resolution": QUARTER.label if len(actual_slots) == quarter_slots else HOURLY.label,
                })
        pred_dir = root / task / "prediction"
        act_dir = root / task / "actual"
        pred_dir.mkdir(parents=True, exist_ok=True)
        act_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_rows).to_parquet(pred_dir / "prediction_ledger.parquet", index=False)
        pd.DataFrame(act_rows).to_parquet(act_dir / "actual_ledger.parquet", index=False)


def _select(root: Path, task: str, target_date: str = "2026-03-15") -> dict:
    models = list(DAYAHEAD_MODELS if task == "dayahead" else REALTIME_MODELS)
    return select_complete_training_days(
        task=task, target_date=target_date, ledger_root=root,
        expected_models=models, required_days=30, max_lookback_days=90,
        resolution=QUARTER, history_lag_days=2,
    )


def _assert_case(**kwargs) -> None:
    with tempfile.TemporaryDirectory(prefix="efm3-cold-start-") as tmp:
        root = Path(tmp)
        _write_case(root, **kwargs)
        results = {task: _select(root, task) for task in ("dayahead", "realtime")}
        if not kwargs:
            assert all(r["status"] == "PASS" and r["selected_count"] == 30 for r in results.values()), results
        else:
            assert any(r["status"] == "FAIL" for r in results.values()), results


def audit(root: Path, target_date: str) -> dict:
    """Audit the real production ledger for a concrete forecast target date."""
    results = {
        task: _select(root, task, target_date)
        for task in ("dayahead", "realtime")
    }
    return {
        "strict_ready": all(r["status"] == "PASS" for r in results.values()),
        "required_days": 30, "max_lookback_days": 90, "resolution": "15min",
        "tasks": results,
        "policy": "use_only_strict_production_ledger; never fallback to historical-invalid-features",
    }


def audit_legacy_candidate(root: Path) -> dict:
    """Explain why a numerically complete shadow ledger is not production-ready."""
    reasons: list[str] = []
    for task in ("dayahead", "realtime"):
        path = root / task / "prediction" / "prediction_ledger.csv"
        actual = root / task / "actual" / "actual_ledger.csv"
        if not path.exists() or not actual.exists():
            reasons.append(f"{task}:candidate_files_missing")
            continue
        pred = pd.read_csv(path)
        truth = pd.read_csv(actual)
        if len(set(pred.get("target_day", []))) != len(set(truth.get("target_day", []))):
            reasons.append(f"{task}:prediction_actual_day_sets_differ")
        required = {"provenance", "leakage_status", "resolution", "model_protocol"}
        missing_meta = sorted(required - set(pred.columns))
        if missing_meta:
            reasons.append(f"{task}:missing_metadata={','.join(missing_meta)}")
        # Historical candidate rows may carry an older fixed-hour cutoff.  It
        # is provenance only and is never promoted as the Dynamic-v1 serving
        # boundary; readiness is decided exclusively by the strict selector
        # above.
    return {
        "strict_ready": False, "root": str(root), "reasons": reasons,
        "policy": (
            "shadow/legacy candidates cannot be copied directly into production; "
            "an audited warm-start migration may import an explicitly bounded "
            "history window while preserving source cutoff/provenance"
        ),
    }


def _infer_target_date(root: Path) -> str:
    days: list[str] = []
    for task in ("dayahead", "realtime"):
        path = root / task / "prediction" / "prediction_ledger.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["target_day"])
        days.extend(df["target_day"].dropna().astype(str).tolist())
    if not days:
        raise RuntimeError(
            "cannot infer target date from an empty production ledger; "
            "pass --target-date explicitly"
        )
    return max(days)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-date",
        default=None,
        help="Forecast target D used by the real environment readiness audit. "
             "Defaults to the latest target_day already present in outputs/96/ledger.",
    )
    args = parser.parse_args()

    _assert_case()  # 30 complete days PASS
    _assert_case(days=29)
    _assert_case(missing_da=DAYAHEAD_MODELS[0])
    _assert_case(missing_rt=REALTIME_MODELS[0])
    _assert_case(missing_actual_p96=True)
    _assert_case(actual_nan=True)
    _assert_case(mixed_24=True)

    production_root = ROOT / "outputs" / "96" / "ledger"
    target_date = args.target_date or _infer_target_date(production_root)
    real = audit(production_root, target_date)
    real["target_date"] = target_date
    print("check_96_production_ledger_cold_start: PASS (synthetic contract regression)")
    print(
        "current_environment_readiness:",
        "READY" if real["strict_ready"] else "BLOCKED",
        real,
    )
    print("legacy_candidate_audit:", audit_legacy_candidate(ROOT / "outputs" / "96" / "feature_store" / "ledger"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
