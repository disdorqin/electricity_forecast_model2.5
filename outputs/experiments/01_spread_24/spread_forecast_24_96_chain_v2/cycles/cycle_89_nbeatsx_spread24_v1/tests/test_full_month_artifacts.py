"""Post-run artifact checks for FULLDEV5."""

from __future__ import annotations

import csv
import json
from pathlib import Path


CYCLE = Path(__file__).resolve().parents[1]
RUN = CYCLE / "runs" / "FULLDEV5"


def test_fulldev5_manifest_is_complete() -> None:
    """All 150 registered dates and the strict pooled scope are present."""
    manifest = json.loads((RUN / "full_month_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FULLDEV5_COMPLETE"
    assert manifest["target_day_count"] == 150
    assert manifest["scored_hours_per_strategy"] == 3600
    assert manifest["pre_training_audit"] == "STRICT/PASS"
    assert manifest["scientific_contract"]["training_history_months"] == 36
    assert manifest["scientific_contract"]["validation_days"] == 28


def test_fulldev5_daily_prediction_shapes() -> None:
    """Every model strategy has exactly 24 scored and 34 total rows per day."""
    for strategy_dir in (RUN / "C0", RUN / "C3"):
        day_dirs = [path for path in strategy_dir.iterdir() if path.is_dir()]
        assert len(day_dirs) == 150
        for day_dir in day_dirs:
            with (day_dir / "target_day_prediction.csv").open(encoding="utf-8") as handle:
                assert sum(1 for _ in csv.DictReader(handle)) == 24
            with (day_dir / "predictions.csv").open(encoding="utf-8") as handle:
                assert sum(1 for _ in csv.DictReader(handle)) == 34
            audit = json.loads((day_dir / "leakage_audit.json").read_text(encoding="utf-8"))
            assert audit["leakage_status"] == "STRICT/PASS"


def test_fulldev5_required_summaries_exist() -> None:
    """Required cross-month and paired analysis layers are materialized."""
    names = (
        "monthly_micro_metrics.csv", "monthly_daily_macro_metrics.csv",
        "monthly_structure_metrics.csv", "cross_month_micro_metrics.json",
        "cross_month_daily_macro_metrics.json", "cross_month_month_macro_metrics.json",
        "paired_daily_deltas.csv", "paired_monthly_deltas.csv", "paired_bootstrap_ci.json",
        "majority_collapse_by_month.csv", "transition_by_month.csv",
        "horizon_by_month.csv", "runtime_cost_by_month.csv",
        "cycle88_comparator_manifest.json", "full_month_cross_month_review.md",
    )
    for name in names:
        assert (RUN / name).exists(), name
