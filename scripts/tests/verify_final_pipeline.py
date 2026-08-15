#!/usr/bin/env python
"""Verify existing full pipeline outputs without re-running models.

Usage:
    python scripts/verify_final_pipeline.py --date 2026-02-24 --runs-root outputs/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def check(condition: bool, label: str, issues: list) -> None:
    if not condition:
        issues.append(label)


def verify_pipeline(date: str, runs_root: str) -> bool:
    runs_dir = Path(runs_root) / date
    manifest_path = runs_dir / "run_manifest.json"
    issues: list[str] = []

    print(f"FINAL_VERIFY: {date}")
    print(f"  manifest: {manifest_path}")

    # 1. Manifest exists and parses
    if not manifest_path.exists():
        print(f"  FAIL: run_manifest.json not found at {manifest_path}")
        return False
    with open(manifest_path) as f:
        manifest = json.load(f)

    # 2. All stages complete
    stages = manifest.get("stages", manifest)
    stage_statuses = {
        "ledger_predict": stages.get("ledger_predict", {}).get("status", "missing"),
        "ledger_weight": stages.get("ledger_weight", {}).get("status", "missing"),
        "ledger_fuse": stages.get("ledger_fuse", {}).get("status", "missing"),
        "ledger_classifier": stages.get("ledger_classifier", {}).get("status", "missing"),
        "final_outputs": stages.get("final_outputs", {}).get("status", "missing"),
    }

    for stage, status in stage_statuses.items():
        check(status == "complete", f"{stage}: expected 'complete', got '{status}'", issues)
        print(f"  {stage}: {status}")

    # 3. Errors and warnings
    errors = manifest.get("errors", [])
    warnings = manifest.get("warnings", [])
    check(len(errors) == 0, f"errors: {errors}", issues)
    check(len(warnings) == 0, f"warnings: {warnings}", issues)
    print(f"  errors: {len(errors)}")
    print(f"  warnings: {len(warnings)}")

    # 4. Prediction data
    for task in ["dayahead", "realtime"]:
        long_csv = runs_dir / task / "prediction" / "all_model_predictions_long.csv"
        check(long_csv.exists(), f"Missing {task}/prediction/all_model_predictions_long.csv", issues)
        if long_csv.exists():
            df = pd.read_csv(long_csv)
            expected = 72 if task == "dayahead" else 96
            actual = len(df)
            check(actual == expected,
                  f"{task} long table: expected {expected} rows, got {actual}", issues)
            print(f"  {task}_long_rows: {actual} (expected {expected}) {'OK' if actual == expected else 'MISMATCH'}")

            # Per-model row check
            for model_name in df["model_name"].unique():
                model_rows = len(df[df["model_name"] == model_name])
                check(model_rows == 24,
                      f"{task}/{model_name}: expected 24 rows, got {model_rows}", issues)
                print(f"    {task}/{model_name}: {model_rows} rows {'OK' if model_rows == 24 else 'MISMATCH'}")

    # 5. Weight data
    for task in ["dayahead", "realtime"]:
        weight_result = stages.get("ledger_weight", {}).get("results", {}).get(task, {})
        training_rows = weight_result.get("training_rows", 0)
        expected_rows = 2160 if task == "dayahead" else 2880
        check(training_rows == expected_rows,
              f"{task} training_rows: expected {expected_rows}, got {training_rows}", issues)
        print(f"  {task}_training_rows: {training_rows} (expected {expected_rows}) {'OK' if training_rows == expected_rows else 'MISMATCH'}")

        # Weight sum check
        weights_csv = runs_dir / task / "weight" / "weights.csv"
        if weights_csv.exists():
            wdf = pd.read_csv(weights_csv)
            for period in wdf["period"].unique():
                sub = wdf[wdf["period"] == period]
                wsum = sub["weight"].sum()
                near1 = abs(wsum - 1.0) < 1e-4
                check(near1, f"{task}/{period} weight sum: {wsum:.6f} (not ~1.0)", issues)
                print(f"    {task}/{period} weight_sum: {wsum:.6f} {'OK' if near1 else 'NEAR1'}")

    # 6. Fuse data
    for task in ["dayahead", "realtime"]:
        fuse_csv = runs_dir / task / "fuse" / "fused_predictions.csv"
        check(fuse_csv.exists(), f"Missing {task}/fuse/fused_predictions.csv", issues)
        if fuse_csv.exists():
            fdf = pd.read_csv(fuse_csv)
            actual = len(fdf)
            check(actual == 24, f"{task} fused rows: expected 24, got {actual}", issues)
            print(f"  {task}_fused_rows: {actual} {'OK' if actual == 24 else 'MISMATCH'}")

            # hour_business range
            hb_range = (fdf["hour_business"].min(), fdf["hour_business"].max())
            check(hb_range == (1, 24), f"{task} hour_business range: {hb_range}", issues)
            print(f"    hour_business: {hb_range[0]}..{hb_range[1]} {'OK' if hb_range == (1, 24) else 'MISMATCH'}")

            # Hour 24 = D+1 00:00
            h24 = fdf[fdf["hour_business"] == 24]
            if len(h24) > 0:
                expected_ds = pd.Timestamp(date) + pd.Timedelta(days=1)
                actual_ds = pd.Timestamp(h24["ds"].values[0])
                check(actual_ds == expected_ds,
                      f"{task} hour 24 ds: expected {expected_ds}, got {actual_ds}", issues)
                print(f"    hour_24_ds: {actual_ds} {'OK' if actual_ds == expected_ds else 'MISMATCH'}")

    # 7. Classifier
    clf_result = stages.get("ledger_classifier", {}).get("results", {})
    corr_applied = clf_result.get("corrections_applied", -1)
    corr_rows = clf_result.get("corrected_rows", 0)
    check(corr_rows == 24, f"classifier corrected_rows: expected 24, got {corr_rows}", issues)
    print(f"  classifier_corrected_rows: {corr_rows} {'OK' if corr_rows == 24 else 'MISMATCH'}")
    print(f"  classifier_corrections_applied: {corr_applied}")

    clf_report = runs_dir / "realtime" / "final" / "classifier_report.json"
    check(clf_report.exists(), "classifier_report.json not found", issues)

    # 8. Final outputs
    final = stages.get("final_outputs", {})
    sub_rows = final.get("submission_ready_rows", 0)
    check(sub_rows == 24, f"submission_ready_rows: expected 24, got {sub_rows}", issues)
    print(f"  submission_ready_rows: {sub_rows} {'OK' if sub_rows == 24 else 'MISMATCH'}")

    # 9. submission_ready.csv details
    sub_csv = runs_dir / "final" / "submission_ready.csv"
    if sub_csv.exists():
        sdf = pd.read_csv(sub_csv)
        expected_cols = ["business_day", "ds", "hour_business", "period", "dayahead_price", "realtime_price"]
        actual_cols = list(sdf.columns)
        check(actual_cols == expected_cols,
              f"submission_ready columns: expected {expected_cols}, got {actual_cols}", issues)
        print(f"  submission_ready_columns: {actual_cols} {'OK' if actual_cols == expected_cols else 'MISMATCH'}")

        # No _x/_y suffix columns
        suffix_cols = [c for c in sdf.columns if c.endswith("_x") or c.endswith("_y")]
        check(len(suffix_cols) == 0, f"suffix _x/_y columns found: {suffix_cols}", issues)

        # hour_business range
        hb_range = (sdf["hour_business"].min(), sdf["hour_business"].max())
        check(hb_range == (1, 24), f"submission hour_business range: {hb_range}", issues)

        # hour 24 = D+1 00:00
        h24 = sdf[sdf["hour_business"] == 24]
        if len(h24) > 0:
            expected_ds = pd.Timestamp(date) + pd.Timedelta(days=1)
            actual_ds = pd.Timestamp(h24["ds"].values[0])
            check(actual_ds == expected_ds,
                  f"submission hour 24 ds: expected {expected_ds}, got {actual_ds}", issues)

        # Columns present
        check("dayahead_price" in sdf.columns, "missing dayahead_price column", issues)
        check("realtime_price" in sdf.columns, "missing realtime_price column", issues)

        print(f"    hour_business: {hb_range[0]}..{hb_range[1]} {'OK' if hb_range == (1, 24) else 'MISMATCH'}")
        print(f"    hour_24_ds: {actual_ds if len(h24) > 0 else 'N/A'} {'OK' if len(h24) > 0 and actual_ds == expected_ds else 'CHECK'}")
        print(f"    suffix_columns_x_y: {suffix_cols} {'OK' if len(suffix_cols) == 0 else 'FOUND'}")
        print(f"    has_dayahead_price: {'dayahead_price' in sdf.columns}")
        print(f"    has_realtime_price: {'realtime_price' in sdf.columns}")

    # Summary
    print()
    if issues:
        print(f"  FAIL — {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print(f"\nFINAL_STATUS: FAIL")
        return False
    else:
        print(f"  FINAL_STATUS: PASS")
        return True


def main():
    parser = argparse.ArgumentParser(description="Verify final pipeline outputs")
    parser.add_argument("--date", default=None, required=True, help="Target date YYYY-MM-DD")
    parser.add_argument("--runs-root", default="outputs/runs", help="Runs root directory")
    args = parser.parse_args()

    ok = verify_pipeline(args.date, args.runs_root)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
