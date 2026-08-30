from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "forecast_strategy_stage1"
COMPARISON = RUN / "comparison"
STRATEGIES = ("C0_DIRECT_H34", "C1_GAP_DIRECT_D24", "C2A_BRIDGE_TF")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stage1_comparison_is_complete_and_scored_on_24_points() -> None:
    manifest = json.loads((COMPARISON / "strategy_stage1_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FORECAST_STRATEGY_STAGE1_COMPLETE"
    assert manifest["leakage_status"] == "STRICT/PASS"
    assert tuple(manifest["candidates"]) == STRATEGIES
    assert len(manifest["target_days"]) == 14

    micro = _rows(COMPARISON / "strategy_stage1_micro_metrics.csv")
    assert {row["strategy"] for row in micro} == set(STRATEGIES)
    for row in micro:
        assert float(row["sample_count"]) == 14 * 24
        assert float(row["direction_sample_count"]) == 14 * 24


def test_stage1_target_artifacts_have_exactly_24_scored_rows() -> None:
    days = json.loads((COMPARISON / "strategy_stage1_manifest.json").read_text(encoding="utf-8"))["target_days"]
    for strategy in STRATEGIES:
        for day in days:
            path = RUN / strategy / day / "target_day_prediction.csv"
            rows = _rows(path)
            assert len(rows) == 24, path
            assert {row["section"] for row in rows} == {"D-day"}
            assert [int(row["business_hour"]) for row in rows] == list(range(1, 25))


def test_c2a_bridge_teacher_forcing_audit_is_explicit_and_strict() -> None:
    days = json.loads((COMPARISON / "strategy_stage1_manifest.json").read_text(encoding="utf-8"))["target_days"]
    for day in days:
        root = RUN / "C2A_BRIDGE_TF" / day
        leakage = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
        bridge = json.loads((root / "bridge_teacher_forcing_audit.json").read_text(encoding="utf-8"))
        assert leakage["leakage_status"] == "STRICT/PASS"
        assert leakage["target_day_sample_count"] == 24
        assert bridge["status"] == "LEGAL_TF_TRAIN_PREDICTED_BRIDGE_INFERENCE"
        assert bridge["true_bridge_inference"] is False
        assert bridge["inference_stage2_bridge_source"] == "STAGE1_PREDICTED_BRIDGE"
        latest_allowed = (date.fromisoformat(day) - timedelta(days=2)).isoformat()
        assert bridge["stage1_training_last_day"] <= latest_allowed
