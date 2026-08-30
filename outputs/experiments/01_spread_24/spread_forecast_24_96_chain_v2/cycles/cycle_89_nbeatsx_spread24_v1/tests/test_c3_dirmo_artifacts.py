from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "C3_DIRMO_10_12_12"
COMPARISON = RUN / "comparison"
BLOCKS = ("B0_BRIDGE10", "B1_DDAY_FIRST12", "B2_DDAY_LAST12")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_c3_manifest_and_required_comparison_outputs() -> None:
    manifest = json.loads((COMPARISON / "C3_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "C3_COMPLETE"
    assert manifest["strategy"] == "C3_DIRMO_10_12_12"
    assert manifest["leakage_status"] == "STRICT/PASS"
    assert manifest["recursive_feedback"] is False
    assert len(manifest["target_days"]) == 14
    required = {
        "C3_daily_metrics.csv",
        "C3_micro_metrics.json",
        "C3_macro_metrics.json",
        "C3_transition_metrics.csv",
        "C3_majority_collapse.csv",
        "C3_h34_offset_metrics.csv",
        "C3_block_metrics.csv",
        "C3_runtime_capacity.csv",
        "C3_paired_vs_C0.csv",
        "C3_paired_vs_Cycle88.csv",
        "C3_review.md",
    }
    assert required <= {path.name for path in COMPARISON.iterdir()}


def test_c3_has_three_strict_block_runs_per_target_day() -> None:
    days = json.loads((COMPARISON / "C3_manifest.json").read_text(encoding="utf-8"))["target_days"]
    for day in days:
        root = RUN / day
        leakage = json.loads((root / "leakage_audit.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert leakage["leakage_status"] == "STRICT/PASS"
        assert leakage["target_day_sample_count"] == 24
        assert leakage["block_predictions_not_reused_as_inputs"] is True
        assert manifest["recursive_feedback"] is False
        assert manifest["forward_passes"] == 3
        assert len(_rows(root / "target_day_prediction.csv")) == 24
        for block in BLOCKS:
            block_root = root / "blocks" / block
            assert (block_root / "checkpoint.pt").exists()
            assert (block_root / "config_execution_audit.json").exists()
            assert (block_root / "model_summary.json").exists()
            audits = json.loads((block_root / "config_execution_audit.json").read_text(encoding="utf-8"))
            assert all(row["status"] != "FAIL" for row in audits)


def test_c3_headline_and_block_counts_are_not_bridge_mixed() -> None:
    micro = json.loads((COMPARISON / "C3_micro_metrics.json").read_text(encoding="utf-8"))
    assert micro["sample_count"] == 14 * 24
    assert micro["direction_sample_count"] == 14 * 24
    block_rows = _rows(COMPARISON / "C3_block_metrics.csv")
    assert {row["block_id"] for row in block_rows} == set(BLOCKS)
    assert float(next(row["sample_count"] for row in block_rows if row["block_id"] == "B0_BRIDGE10")) == 14 * 10
    assert all(float(row["sample_count"]) == 14 * 12 for row in block_rows if row["block_id"] != "B0_BRIDGE10")
