"""Audit whether 96-point prediction and actual ledgers correspond.

This is a structural audit only.  Historical actual ledgers sourced from the
known contaminated ``shandong_pmos_96_full_v2.xlsx`` input are explicitly
reported as ``LEGACY-UNVERIFIED`` and are never promoted to an accuracy claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def audit_task(task: str, root: Path) -> dict:
    pred_path = root / task / "prediction" / "prediction_ledger.parquet"
    actual_path = root / task / "actual" / "actual_ledger.parquet"
    if not pred_path.exists() or not actual_path.exists():
        return {"task": task, "status": "MISSING", "prediction_path": str(pred_path), "actual_path": str(actual_path)}
    pred = pd.read_parquet(pred_path)
    actual = pd.read_parquet(actual_path)
    pred["target_day"] = pd.to_datetime(pred["target_day"]).dt.strftime("%Y-%m-%d")
    actual["target_day"] = pd.to_datetime(actual["target_day"]).dt.strftime("%Y-%m-%d")
    required_p = {"target_day", "business_period", "model_name", "y_pred"}
    required_a = {"target_day", "business_period", "y_true"}
    missing = sorted((required_p - set(pred.columns)) | (required_a - set(actual.columns)))
    if missing:
        return {"task": task, "status": "FAIL", "missing_columns": missing}
    pred["business_period"] = pd.to_numeric(pred["business_period"], errors="coerce")
    actual["business_period"] = pd.to_numeric(actual["business_period"], errors="coerce")
    pred_keys = ["target_day", "business_period", "model_name"]
    actual_keys = ["target_day", "business_period"]
    dup_pred = int(pred.duplicated(pred_keys).sum())
    dup_actual = int(actual.duplicated(actual_keys).sum())
    period_ok = bool(pred["business_period"].between(1, 96).all() and actual["business_period"].between(1, 96).all())
    actual_key_frame = actual[actual_keys].drop_duplicates()
    model_rows = []
    for model, group in pred.groupby("model_name", sort=True):
        keys = group[actual_keys].drop_duplicates()
        overlap = keys.merge(actual_key_frame, on=actual_keys, how="inner").shape[0]
        day_counts = group.groupby("target_day")["business_period"].nunique()
        model_rows.append({
            "model_name": str(model),
            "rows": int(len(group)),
            "days": int(group["target_day"].nunique()),
            "min_day": str(group["target_day"].min()),
            "max_day": str(group["target_day"].max()),
            "complete_96_days": int((day_counts == 96).sum()),
            "incomplete_days": int((day_counts != 96).sum()),
            "prediction_actual_key_overlap": int(overlap),
            "actual_keys_available": int(len(actual_key_frame)),
        })
    source_values = [str(x) for x in actual.get("source_file", pd.Series(dtype=str)).dropna().unique()]
    legacy = any(token in " ".join(source_values).lower() for token in ("full_v2", "quarantine", "legacy"))
    structural_pass = dup_pred == 0 and dup_actual == 0 and period_ok and bool(model_rows) and all(x["incomplete_days"] == 0 for x in model_rows)
    return {
        "task": task,
        "status": "STRUCTURAL_PASS_LEGACY_UNVERIFIED" if structural_pass and legacy else ("STRUCTURAL_PASS" if structural_pass else "FAIL"),
        "prediction_rows": int(len(pred)), "actual_rows": int(len(actual)),
        "prediction_min_day": str(pred["target_day"].min()), "prediction_max_day": str(pred["target_day"].max()),
        "actual_min_day": str(actual["target_day"].min()), "actual_max_day": str(actual["target_day"].max()),
        "prediction_models": model_rows, "duplicate_prediction_keys": dup_pred, "duplicate_actual_keys": dup_actual,
        "period_range_ok": period_ok, "actual_source_files": source_values,
        "quality_status": "LEGACY-UNVERIFIED" if legacy else "UNVERIFIED_SOURCE",
        "accuracy_claim_allowed": False if legacy else None,
        "final_holdout_touched_by_audit": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/ledger_96"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    results = [audit_task(t, args.root.resolve()) for t in ("dayahead", "realtime")]
    overall = "PASS_WITH_LEGACY_UNVERIFIED" if all(r["status"] == "STRUCTURAL_PASS_LEGACY_UNVERIFIED" for r in results) else ("PASS" if all(r["status"] == "STRUCTURAL_PASS" for r in results) else "FAIL")
    payload = {"status": overall, "resolution": "96", "structural_only": True, "accuracy_claim_allowed": False, "final_holdout_touched": False, "tasks": results}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([{"task": r["task"], "status": r["status"], "quality_status": r.get("quality_status"), "prediction_rows": r.get("prediction_rows"), "actual_rows": r.get("actual_rows"), "prediction_min_day": r.get("prediction_min_day"), "prediction_max_day": r.get("prediction_max_day"), "actual_min_day": r.get("actual_min_day"), "actual_max_day": r.get("actual_max_day")} for r in results]).to_csv(out / "task_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
