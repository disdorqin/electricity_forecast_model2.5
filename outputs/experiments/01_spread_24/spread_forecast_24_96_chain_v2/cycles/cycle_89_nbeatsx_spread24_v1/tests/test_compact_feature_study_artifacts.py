from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/feature_study"
EXPECTED = {"F0_A2_CORE5", "F1_A2_PHYSICAL_SHAPE", "F2_A2_CAUSAL_PRICE_STATE"}


def test_compact_feature_study_has_strict_dev14_artifacts() -> None:
    manifest = json.loads((STUDY / "feature_study_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPACT_FEATURE_STUDY_COMPLETE"
    assert manifest["leakage_status"] == "STRICT/PASS"
    assert set(manifest["candidates"]) == EXPECTED
    assert len(manifest["target_days"]) == 14
    for candidate in EXPECTED:
        day_dirs = [path for path in (STUDY / candidate).iterdir() if path.is_dir()]
        assert len(day_dirs) == 14
        for day_dir in day_dirs:
            run_manifest = json.loads((day_dir / "manifest.json").read_text(encoding="utf-8"))
            assert run_manifest["leakage_status"] == "STRICT/PASS"
            assert run_manifest["target_day_sample_count"] == 24
            rows = list(csv.DictReader((day_dir / "target_day_prediction.csv").open(encoding="utf-8")))
            assert len(rows) == 24
            audit = json.loads((day_dir / "leakage_audit.json").read_text(encoding="utf-8"))
            assert audit["leakage_status"] == "STRICT/PASS"
            assert all(item["status"] == "PASS" for item in audit["audits"])


def test_compact_feature_study_comparison_artifacts_are_complete() -> None:
    comparison = STUDY / "comparison"
    required = {
        "feature_daily_metrics.csv", "feature_micro_metrics.csv", "feature_macro_metrics.csv",
        "feature_majority_collapse.csv", "feature_transition_metrics.csv",
        "feature_h34_offset_metrics.csv", "feature_paired_vs_A2.csv",
        "feature_paired_vs_Cycle88.csv", "feature_review.md",
    }
    assert required.issubset({path.name for path in comparison.iterdir()})
    rows = list(csv.DictReader((comparison / "feature_daily_metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 42
    assert all(float(row["sample_count"]) == 24 for row in rows)
