from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from scripts.server.bootstrap_96_production_ledger import CONTRACT, migrate
from pipelines.prediction_ledger import append_predictions_to_ledger, update_actual_ledger


def _period(slot: int) -> str:
    if slot <= 32:
        return "1_32"
    if slot <= 64:
        return "33_64"
    return "65_96"


def _prediction_rows(task: str, models: tuple[str, ...], days: list[str], cutoff_hour: int) -> pd.DataFrame:
    rows = []
    for day_text in days:
        day = pd.Timestamp(day_text)
        for model_idx, model in enumerate(models):
            for slot in range(1, 97):
                rows.append({
                    "task": task,
                    "model_name": model,
                    "forecast_date": day_text,
                    "target_day": day_text,
                    "business_day": day_text,
                    "ds": day + pd.Timedelta(minutes=15 * slot),
                    "business_period": slot,
                    "hour_business": (slot - 1) // 4 + 1,
                    "period": _period(slot),
                    "y_pred": 100.0 + slot + model_idx,
                    "data_cutoff": (
                        (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        if task == "dayahead"
                        else (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d") + f" {cutoff_hour:02d}:00:00"
                    ),
                    "run_id": f"{model}_{day_text}",
                    "model_version": "legacy-server",
                    "source_file": "server-history",
                })
    return pd.DataFrame(rows)


def _actual_rows(task: str, days: list[str]) -> pd.DataFrame:
    rows = []
    for day_text in days:
        day = pd.Timestamp(day_text)
        for slot in range(1, 97):
            rows.append({
                "task": task,
                "target_day": day_text,
                "business_day": day_text,
                "ds": day + pd.Timedelta(minutes=15 * slot),
                "business_period": slot,
                "hour_business": (slot - 1) // 4 + 1,
                "period": _period(slot),
                "y_true": 110.0 + slot,
                "source_file": "server-actual",
            })
    return pd.DataFrame(rows)


def _make_source(root: Path, target_date: str, days: int = 30, rt_cutoff_hour: int = 14) -> None:
    target = pd.Timestamp(target_date)
    history = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(
            target - pd.Timedelta(days=days + 1),
            target - pd.Timedelta(days=1),
            freq="D",
        )
    ]
    for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
        append_predictions_to_ledger(
            _prediction_rows(task, tuple(models), history, rt_cutoff_hour),
            root,
            task,
        )
        update_actual_ledger(_actual_rows(task, history), root, task)


def _make_current_target(root: Path, target_date: str) -> None:
    for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
        append_predictions_to_ledger(
            _prediction_rows(task, tuple(models), [target_date], 15),
            root,
            task,
        )
        update_actual_ledger(_actual_rows(task, [target_date]), root, task)


def test_warm_start_dry_run_is_read_only(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _make_source(source, "2026-03-15")

    result = migrate(
        source_root=source,
        target_root=target,
        target_date="2026-03-15",
        days=30,
        apply=False,
        runtime_root=tmp_path / "runtime",
    )

    assert result["status"] == "AUDIT_PASS"
    assert result["applied"] is False
    assert result["audit"]["tasks"]["dayahead"]["prediction_rows"] == 31 * 3 * 96
    assert result["audit"]["tasks"]["realtime"]["prediction_rows"] == 31 * 4 * 96
    assert result["audit"]["tasks"]["dayahead"]["actual_rows"] == 30 * 96
    assert result["audit"]["tasks"]["realtime"]["actual_rows"] == 30 * 96
    assert result["audit"]["prediction_window_end"] == "2026-03-14"
    assert result["audit"]["actual_window_end"] == "2026-03-13"
    assert not target.exists()


def test_warm_start_apply_preserves_current_and_is_idempotent(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _make_source(source, "2026-03-15", rt_cutoff_hour=14)
    _make_current_target(target, "2026-03-15")

    first = migrate(
        source_root=source,
        target_root=target,
        target_date="2026-03-15",
        days=30,
        apply=True,
        runtime_root=tmp_path / "runtime",
    )
    assert first["status"] == "COMPLETE"
    assert first["applied"] is True
    assert first["final_readiness"]["status"] == "PASS"

    da = pd.read_parquet(target / "dayahead" / "prediction" / "prediction_ledger.parquet")
    rt = pd.read_parquet(target / "realtime" / "prediction" / "prediction_ledger.parquet")
    assert len(da) == 32 * 3 * 96
    assert len(rt) == 32 * 4 * 96
    assert set(da["target_day"]) >= {"2026-02-12", "2026-03-14", "2026-03-15"}
    assert set(rt[rt["target_day"] != "2026-03-15"]["data_cutoff"].str[-8:]) == {"14:00:00"}

    for task in ("dayahead", "realtime"):
        actual = pd.read_parquet(
            target / task / "actual" / "actual_ledger.parquet"
        )
        # Current target truth is preserved from the pre-existing target ledger,
        # while T-1 full-day truth is intentionally not imported by warm-start.
        assert "2026-03-14" not in set(actual["target_day"].astype(str))
        assert "2026-03-13" in set(actual["target_day"].astype(str))
        assert "2026-03-15" in set(actual["target_day"].astype(str))

    manifest = json.loads((target / "bootstrap_manifest.json").read_text(encoding="utf-8"))
    assert manifest["contract"] == CONTRACT
    assert manifest["audit"]["prediction_window_end"] == "2026-03-14"
    assert manifest["audit"]["actual_window_end"] == "2026-03-13"
    assert all("parquet_path" not in item and "csv_path" not in item for item in manifest["stage_results"].values())
    assert Path(manifest["promoted_paths"]["dayahead_prediction"]).exists()
    assert Path(manifest["promoted_paths"]["realtime_actual"]).exists()
    assert len(manifest["promoted_sha256"]["dayahead_prediction"]) == 64

    second = migrate(
        source_root=source,
        target_root=target,
        target_date="2026-03-15",
        days=30,
        apply=True,
        runtime_root=tmp_path / "runtime",
    )
    assert second["status"] == "COMPLETE"
    da2 = pd.read_parquet(target / "dayahead" / "prediction" / "prediction_ledger.parquet")
    rt2 = pd.read_parquet(target / "realtime" / "prediction" / "prediction_ledger.parquet")
    assert len(da2) == len(da)
    assert len(rt2) == len(rt)


def test_warm_start_accepts_noncontiguous_complete_days_within_lookback(tmp_path: Path):
    source = tmp_path / "source"
    target_root = tmp_path / "target"
    target_date = "2026-03-15"
    target_ts = pd.Timestamp(target_date)

    # Build 40 possible historical days ending at T-2, then deliberately drop
    # several recent calendar days. Production readiness should still succeed
    # because the learner selects the most recent 30 complete days within the
    # 90-day lookback rather than requiring 30 contiguous calendar days.
    all_days = [
        d.strftime("%Y-%m-%d")
        for d in pd.date_range(
            target_ts - pd.Timedelta(days=45),
            target_ts - pd.Timedelta(days=2),
            freq="D",
        )
    ]
    dropped = set(all_days[-12:-7])
    complete_days = [day for day in all_days if day not in dropped]

    for task, models in (
        ("dayahead", DAYAHEAD_MODELS),
        ("realtime", REALTIME_MODELS),
    ):
        append_predictions_to_ledger(
            _prediction_rows(task, tuple(models), complete_days, 14),
            source,
            task,
        )
        update_actual_ledger(
            _actual_rows(task, complete_days),
            source,
            task,
        )

    result = migrate(
        source_root=source,
        target_root=target_root,
        target_date=target_date,
        days=30,
        apply=False,
        runtime_root=tmp_path / "runtime",
    )

    assert result["status"] == "AUDIT_PASS", result["audit"]["errors"]
    for task in ("dayahead", "realtime"):
        task_audit = result["audit"]["tasks"][task]
        assert task_audit["selection_mode"] == "adaptive_complete_days"
        assert task_audit["selected_count"] == 30
        assert dropped.isdisjoint(set(task_audit["selected_days"]))


def test_warm_start_rejects_cutoff_later_than_current_boundary(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    _make_source(source, "2026-03-15", rt_cutoff_hour=16)

    result = migrate(
        source_root=source,
        target_root=target,
        target_date="2026-03-15",
        days=30,
        apply=False,
        runtime_root=tmp_path / "runtime",
    )

    assert result["status"] == "AUDIT_FAIL"
    assert any("cutoff later than formal boundary" in error for error in result["audit"]["errors"])
    assert not target.exists()
