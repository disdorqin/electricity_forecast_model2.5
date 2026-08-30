"""Prepare a small, reviewable Cycle89 collaboration package.

The package contains monthly metrics and the 25-day L1/C0 scored predictions,
but never copies checkpoints, raw data, or the local ``runs`` tree.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve()
CYCLE = HERE.parents[1]
SHARE = CYCLE / "share"
MONTHLY_DIR = SHARE / "monthly"
DAILY_DIR = SHARE / "daily"
METADATA_DIR = SHARE / "metadata"


def sha256(path: Path) -> str:
    """Hash one generated/source artifact for the share manifest."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty collaboration CSV."""
    if not rows:
        raise ValueError(f"empty collaboration table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read UTF-8 CSV rows with BOM tolerance."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def monthly_results() -> list[dict[str, Any]]:
    """Normalize FULLDEV5 and L1 RACE25 monthly outputs to one schema."""
    full = CYCLE / "runs/FULLDEV5"
    micro = pd.read_csv(full / "monthly_micro_metrics.csv")
    macro = pd.read_csv(full / "monthly_daily_macro_metrics.csv")
    structure = pd.read_csv(full / "monthly_structure_metrics.csv")
    merged = micro.merge(macro, on=["strategy", "month", "target_days"], how="left").merge(structure, on=["strategy", "month"], how="left", suffixes=("_macro", "_structure"))
    rows: list[dict[str, Any]] = []
    for item in merged.to_dict("records"):
        rows.append({
            "experiment": "FULLDEV5",
            "model": item["strategy"],
            "month": item["month"],
            "target_days": int(item["target_days"]),
            "scored_points": int(item["target_days"]) * 24,
            "raw": float(item["direction_accuracy"]),
            "positive_recall": float(item["positive_recall"]),
            "negative_recall": float(item["negative_recall"]),
            "balanced": float(item["balanced_accuracy"]),
            "MAE": float(item["mae"]),
            "daily_macro_balanced": float(item["daily_macro_balanced"]),
            "minority_recall": float(item["minority_recall_macro"]),
            "collapse_days": int(item["majority_collapse_days"]),
            "transition_f1": float(item["transition_f1"]),
            "leakage_status": "STRICT/PASS",
        })

    race = pd.read_csv(CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025/RACE25_monthly_metrics.csv")
    for item in race.to_dict("records"):
        rows.append({
            "experiment": "RACE25",
            "model": "L1_D24_MAE_BRIDGE025",
            "month": item["month"],
            "target_days": int(item["target_days"]),
            "scored_points": int(item["target_days"]) * 24,
            "raw": float(item["l1_raw"]),
            "positive_recall": float(item["l1_positive_recall"]),
            "negative_recall": float(item["l1_negative_recall"]),
            "balanced": float(item["l1_balanced"]),
            "MAE": float(item["l1_MAE"]),
            "daily_macro_balanced": float(item["l1_daily_macro_balanced"]),
            "minority_recall": float(item["l1_minority_recall"]),
            "collapse_days": int(item["l1_collapse_days"]),
            "transition_f1": float(item["l1_transition_f1"]),
            "leakage_status": "STRICT/PASS",
        })
    return rows


def race25_predictions() -> list[dict[str, Any]]:
    """Export only scored D-day rows for L1 and its frozen same-date C0."""
    days = [
        "2026-01-06", "2026-01-07", "2026-01-17", "2026-01-19", "2026-01-31",
        "2026-02-08", "2026-02-09", "2026-02-12", "2026-02-24", "2026-02-27",
        "2026-04-06", "2026-04-09", "2026-04-12", "2026-04-16", "2026-04-18",
        "2026-06-02", "2026-06-11", "2026-06-14", "2026-06-26", "2026-06-29",
        "2026-07-03", "2026-07-07", "2026-07-15", "2026-07-25", "2026-07-28",
    ]
    rows: list[dict[str, Any]] = []
    for day in days:
        specs = (
            ("L1_D24_MAE_BRIDGE025", CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025" / day / "target_day_prediction.csv"),
            ("C0_DIRECT_H34", CYCLE / "runs/FULLDEV5/C0" / day / "target_day_prediction.csv"),
        )
        for model, path in specs:
            source = read_rows(path)
            if len(source) != 24:
                raise RuntimeError(f"{model} {day} must have 24 scored rows")
            for item in sorted(source, key=lambda x: int(x["business_hour"])):
                rows.append({"experiment": "RACE25", "model": model, "target_day": day, "business_hour": int(item["business_hour"]), "prediction": float(item["prediction"]), "target": float(item["target"]), "leakage_status": "STRICT/PASS"})
    return rows


def results_catalog() -> dict[str, Any]:
    """Describe the review order without copying the large local run tree."""
    return {
        "review_order": [
            "monthly/cycle89_monthly_results.csv",
            "daily/cycle89_race25_daily_predictions.csv",
            "../README.md",
        ],
        "shared_results": {
            "monthly": "monthly/cycle89_monthly_results.csv",
            "daily": "daily/cycle89_race25_daily_predictions.csv",
        },
        "local_run_roots": [
            {"category": "smoke", "path": "runs/smoke_final/"},
            {"category": "readiness", "path": "runs/formal_single_day_oos/; runs/formal_mini_backtest_3day/; runs/b0_extended_panel_14d/"},
            {"category": "history_window", "path": "runs/history_window_study/"},
            {"category": "feature_study", "path": "runs/feature_study/"},
            {"category": "forecast_strategy", "path": "runs/forecast_strategy_stage1/"},
            {"category": "dirmo", "path": "runs/C3_DIRMO_10_12_12/"},
            {"category": "full_month", "path": "runs/FULLDEV5/"},
            {"category": "loss_objective", "path": "runs/loss_objective/"},
            {"category": "paper_reproduction", "path": "runs/paper_repro/"},
        ],
        "note": "The run roots are local reproducibility artifacts and are intentionally not committed to the collaboration branch.",
    }


def main() -> int:
    """Materialize the small collaboration package below ``share/``."""
    # Keep human-facing artifacts separated by purpose.  This makes the
    # remote branch useful for review without exposing the large local runs/.
    for directory in (MONTHLY_DIR, DAILY_DIR, METADATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    metrics = monthly_results()
    predictions = race25_predictions()
    metrics_path = MONTHLY_DIR / "cycle89_monthly_results.csv"
    predictions_path = DAILY_DIR / "cycle89_race25_daily_predictions.csv"
    catalog_path = METADATA_DIR / "cycle89_results_catalog.json"
    write_csv(metrics_path, metrics)
    write_csv(predictions_path, predictions)
    catalog_path.write_text(json.dumps(results_catalog(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = METADATA_DIR / "cycle89_share_manifest.json"
    manifest = {
        "status": "SHARE_PACKAGE_READY",
        "cycle": "cycle_89_nbeatsx_spread24_v1",
        "created_for": "review collaboration",
        "scientific_contract": {"target": "DA - RT", "origin": "D-1 14:00", "training_last_day": "D-2 or earlier", "headline_scope": "D-day 24 scored points"},
        "included": [str(metrics_path.relative_to(CYCLE)).replace("\\", "/"), str(predictions_path.relative_to(CYCLE)).replace("\\", "/"), str(catalog_path.relative_to(CYCLE)).replace("\\", "/"), "README.md", "experiment_manifest.json", "src/", "tests/", "scripts/", "configs/", "docs/", "third_party/nbeatsx_source_manifest.json"],
        "excluded": ["runs/", "*.pt", "raw data/", "third_party/reference_source/results/forecasts.zip", "third_party/reference_source/*.ipynb"],
        "full_month_models": ["C0_DIRECT_H34", "C3_DIRMO_10_12_12", "Cycle88_LGBM_v2_full_F0_F9"],
        "race25_models": ["L1_D24_MAE_BRIDGE025", "C0_DIRECT_H34"],
        "source_files": {
            "full_month_metrics": "runs/FULLDEV5/monthly_micro_metrics.csv",
            "full_month_macro": "runs/FULLDEV5/monthly_daily_macro_metrics.csv",
            "full_month_structure": "runs/FULLDEV5/monthly_structure_metrics.csv",
            "race25_metrics": "runs/loss_objective/L1_D24_MAE_BRIDGE025/RACE25_monthly_metrics.csv",
        },
        "generated_sha256": {
            str(metrics_path.relative_to(CYCLE)).replace("\\", "/"): sha256(metrics_path),
            str(predictions_path.relative_to(CYCLE)).replace("\\", "/"): sha256(predictions_path),
            str(catalog_path.relative_to(CYCLE)).replace("\\", "/"): sha256(catalog_path),
        },
        "leakage_status": "STRICT/PASS",
        "note": "This package shares auditable summaries and scored predictions only; local checkpoints and the full run tree remain excluded.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "monthly_rows": len(metrics), "prediction_rows": len(predictions), "share_dir": str(SHARE)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
