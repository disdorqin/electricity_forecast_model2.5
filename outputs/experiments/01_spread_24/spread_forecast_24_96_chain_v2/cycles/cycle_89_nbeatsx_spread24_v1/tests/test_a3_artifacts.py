from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "runs/history_window_study/comparison_a3"


def test_a3_required_artifacts_and_result_scope() -> None:
    required = {
        "A3_daily_metrics.csv",
        "A3_micro_metrics.json",
        "A3_macro_metrics.json",
        "A3_majority_collapse.csv",
        "A3_transition_metrics.csv",
        "A3_h34_offset_metrics.csv",
        "A3_training_sample_counts.csv",
        "A3_gradient_summary.csv",
        "A3_paired_vs_A0.csv",
        "A3_paired_vs_A2.csv",
        "A3_paired_vs_Cycle88.csv",
        "A3_review.md",
    }
    assert required.issubset({path.name for path in COMPARISON.iterdir()})
    manifest = json.loads((ROOT / "runs/history_window_study/36m_val84/A3_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate"] == "A3_HISTORY36_VAL84"
    assert manifest["validation_history_days"] == 84
    assert manifest["leakage_status"] == "STRICT/PASS"
    assert manifest["decision_status"] in {"ROBUST_LONG_HISTORY_PASS", "LONG_HISTORY_MAGNITUDE_ONLY", "LONG_HISTORY_REGRESSION"}


def test_a3_has_14_target_days_and_24_scored_rows() -> None:
    with (COMPARISON / "A3_daily_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 14
    assert len({row["target_day"] for row in rows}) == 14
    assert all(float(row["sample_count"]) == 24 for row in rows)
