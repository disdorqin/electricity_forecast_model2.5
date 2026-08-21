#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare a legacy classifier XLSX with a range-runner Parquet ledger.

The comparison is intentionally output-focused: timestamps and binary
decisions must match exactly; floating-point probabilities are checked under a
small tolerance.  Results are written to the experiment directory only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NUMERIC_COLUMNS = (
    "p1_prob",
    "p2_prob",
    "final_prob",
    "gray_low",
    "gray_high",
    "threshold",
)
DECISION_COLUMNS = ("p1_pred", "p2_pred", "final_pred")


def _read(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        frame = pd.read_parquet(path)
    if "时刻" not in frame.columns:
        raise ValueError(f"missing 时刻: {path}")
    return frame.sort_values("时刻").reset_index(drop=True)


def compare(reference: Path, candidate: Path, tolerance: float) -> dict:
    old = _read(reference)
    new = _read(candidate)
    report: dict = {
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_rows": len(old),
        "candidate_rows": len(new),
        "timestamps_equal": bool(old["时刻"].equals(new["时刻"])),
        "decision_equal": {},
        "numeric_max_abs_diff": {},
        "numeric_within_tolerance": {},
    }
    if len(old) != len(new) or not report["timestamps_equal"]:
        report["status"] = "FAIL"
        return report

    for col in DECISION_COLUMNS:
        if col not in old.columns or col not in new.columns:
            report["decision_equal"][col] = False
            continue
        left = old[col].fillna(-999).to_numpy()
        right = new[col].fillna(-999).to_numpy()
        report["decision_equal"][col] = bool(np.array_equal(left, right))

    for col in NUMERIC_COLUMNS:
        if col not in old.columns or col not in new.columns:
            report["numeric_within_tolerance"][col] = False
            continue
        left = pd.to_numeric(old[col], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(new[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(left) & np.isnan(right)
        finite = np.isfinite(left) & np.isfinite(right)
        diff = np.zeros(len(left), dtype=float)
        diff[finite] = np.abs(left[finite] - right[finite])
        diff[~finite & ~both_nan] = np.inf
        report["numeric_max_abs_diff"][col] = float(np.max(diff)) if len(diff) else 0.0
        report["numeric_within_tolerance"][col] = bool(np.all(diff <= tolerance))

    report["status"] = "PASS" if (
        report["timestamps_equal"]
        and all(report["decision_equal"].values())
        and all(report["numeric_within_tolerance"].values())
    ) else "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare legacy classifier output with range runner output")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--report", help="JSON report path")
    args = parser.parse_args()

    report = compare(Path(args.reference), Path(args.candidate), args.tolerance)
    report_path = Path(args.report) if args.report else (
        PROJECT_ROOT / "outputs" / "experiments" / "classifier_range" / "parity_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

