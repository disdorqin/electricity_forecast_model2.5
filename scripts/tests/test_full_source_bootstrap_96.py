from __future__ import annotations

from pathlib import Path

import pandas as pd

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from pipelines.prediction_ledger import append_predictions_to_ledger, update_actual_ledger
from scripts.server.bootstrap_96_production_ledger import migrate
from scripts.tests.test_bootstrap_96_production_ledger import _actual_rows, _prediction_rows


def _write_history(root: Path, days: list[str], cutoff: int = 14, *, current_marker: float = 0.0) -> None:
    for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
        predictions = _prediction_rows(task, tuple(models), days, cutoff)
        if current_marker:
            predictions["y_pred"] = predictions["y_pred"] + current_marker
        append_predictions_to_ledger(
            predictions, root, task,
        )
        update_actual_ledger(_actual_rows(task, days), root, task)


def test_full_source_dry_run_imports_only_new_days_and_keeps_current(tmp_path: Path):
    source = tmp_path / "server_archive"
    target = tmp_path / "production"
    all_days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-12-18", "2026-08-14", freq="D")]
    current_days = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-07-16", "2026-08-16", freq="D")]
    _write_history(source, all_days)
    _write_history(target, current_days, cutoff=15, current_marker=10000.0)

    before = {
        task: (target / task / "prediction" / "prediction_ledger.parquet").read_bytes()
        for task in ("dayahead", "realtime")
    }
    result = migrate(
        source_root=source,
        target_root=target,
        target_date="2026-08-16",
        history_scope="full-source",
        apply=False,
        runtime_root=tmp_path / "runtime",
    )
    assert result["status"] == "AUDIT_PASS", result.get("audit", {}).get("errors")
    assert result["applied"] is False
    assert result["source_days"] == 240
    assert result["imported_day_count"] == 210
    assert result["skipped_overlap_day_count"] == 30
    assert result["final_readiness"]["status"] == "PASS"
    assert result["final_range"]["start"] == "2025-12-18"
    assert result["final_range"]["end"] == "2026-08-16"
    assert before["dayahead"] == (target / "dayahead" / "prediction" / "prediction_ledger.parquet").read_bytes()
    assert before["realtime"] == (target / "realtime" / "prediction" / "prediction_ledger.parquet").read_bytes()

    applied = migrate(
        source_root=source, target_root=target, target_date="2026-08-16",
        history_scope="full-source", apply=True, runtime_root=tmp_path / "runtime_apply",
    )
    assert applied["status"] == "COMPLETE"
    merged = pd.read_parquet(target / "dayahead" / "prediction" / "prediction_ledger.parquet")
    overlap = merged[(merged["target_day"] == "2026-07-16") & (merged["model_name"] == DAYAHEAD_MODELS[0])]
    assert float(overlap.iloc[0]["y_pred"]) >= 10000.0  # current production wins
    rerun = migrate(
        source_root=source, target_root=target, target_date="2026-08-16",
        history_scope="full-source", apply=True, runtime_root=tmp_path / "runtime_apply2",
    )
    assert rerun["status"] == "COMPLETE"
    merged_again = pd.read_parquet(target / "dayahead" / "prediction" / "prediction_ledger.parquet")
    assert len(merged_again) == len(merged)
