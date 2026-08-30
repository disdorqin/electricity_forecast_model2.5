from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "runs/history_window_study/comparison"
EXPECTED = {"A0_HISTORY_9M", "A1_HISTORY_24M", "A2_HISTORY_36M"}


def _rows(name: str) -> list[dict[str, str]]:
    with (COMPARISON / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_history_study_has_exact_preregistered_candidates_and_artifacts() -> None:
    required = {
        "history_window_daily_metrics.csv",
        "history_window_micro_metrics.csv",
        "history_window_macro_metrics.csv",
        "history_window_paired_deltas.csv",
        "history_window_training_sample_counts.csv",
        "history_window_gradient_summary.csv",
        "history_window_majority_collapse.csv",
        "history_window_transition_metrics.csv",
        "history_window_review.md",
    }
    assert required.issubset({path.name for path in COMPARISON.iterdir()})
    manifest = json.loads((ROOT / "runs/history_window_study/history_window_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "HISTORY_WINDOW_STUDY_COMPLETE"
    assert set(manifest["candidates"]) == EXPECTED
    assert manifest["leakage_status"] == "STRICT/PASS"


def test_history_study_has_14_days_and_24_scored_points_per_candidate() -> None:
    rows = _rows("history_window_daily_metrics.csv")
    assert len(rows) == 42
    for candidate in EXPECTED:
        candidate_rows = [row for row in rows if row["history_id"] == candidate]
        assert len(candidate_rows) == 14
        assert len({row["target_day"] for row in candidate_rows}) == 14
        assert all(float(row["sample_count"]) == 24 for row in candidate_rows)


def test_history_study_training_cutoff_is_d_minus_2_or_earlier() -> None:
    rows = _rows("history_window_training_sample_counts.csv")
    assert len(rows) == 42
    for row in rows:
        target = date.fromisoformat(row["target_day"])
        cutoff = date.fromisoformat(row["training_last_day"])
        assert cutoff <= target - timedelta(days=2)
        assert int(row["validation_count"]) == 28
        assert int(row["train_count"]) > 0


def test_history_study_did_not_start_forbidden_stages() -> None:
    manifest = json.loads((ROOT / "runs/history_window_study/history_window_manifest.json").read_text(encoding="utf-8"))
    assert manifest["forbidden_stages_not_run"] == [
        "feature_stage",
        "rollout_stage",
        "directional_loss",
        "hyperparameter_search",
    ]
