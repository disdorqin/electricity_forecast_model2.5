from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.server.maintenance_96 import build_retention_plan, main


def _touch_old(path: Path, when: datetime) -> None:
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "marker.txt"
    marker.write_text("x", encoding="utf-8")
    ts = when.timestamp()
    os.utime(marker, (ts, ts))
    os.utime(path, (ts, ts))


def test_retention_plan_marks_only_expired_rebuildable_assets(tmp_path: Path):
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    root = tmp_path / "outputs" / "96"

    success = root / "runs" / "2026-08-01"
    success.mkdir(parents=True)
    (success / "run_manifest.json").write_text(
        json.dumps({
            "status": "complete",
            "decision_snapshot": {
                "tasks": {
                    "dayahead": {
                        "weights": [{"model_name": "lightgbm", "weight": 1.0}],
                        "model_quality_gate": [{"period": "1_32", "status": "PASS"}],
                    },
                    "realtime": {
                        "weights": [{"model_name": "sgdfnet", "weight": 1.0}],
                        "model_quality_gate": [{"period": "1_32", "status": "PASS"}],
                    },
                }
            },
        }), encoding="utf-8"
    )
    _touch_old(success / "logs", now - timedelta(days=31))
    _touch_old(success / "dayahead" / "prediction", now - timedelta(days=31))
    _touch_old(success / "realtime" / "fuse", now - timedelta(days=31))
    _touch_old(success / "final", now - timedelta(days=400))

    failed = root / "runs" / "2026-05-01"
    failed.mkdir(parents=True)
    (failed / "run_manifest.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    _touch_old(failed / "logs", now - timedelta(days=91))

    range_dir = root / "runs" / "range_2026-08-15_to_2026-09-15_predict"
    range_dir.mkdir(parents=True)
    (range_dir / "prediction_range_manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    _touch_old(range_dir / "logs", now - timedelta(days=31))

    _touch_old(root / "runtime" / "models_stale", now - timedelta(days=2))
    _touch_old(root / "ledger", now - timedelta(days=400))
    _touch_old(root / "cache" / "classifier", now - timedelta(days=400))

    plan = build_retention_plan(root, now=now)
    paths = {Path(item["path"]) for item in plan["candidates"]}

    assert success / "logs" in paths
    assert success / "dayahead" / "prediction" in paths
    assert success / "realtime" / "fuse" in paths
    assert failed / "logs" in paths
    assert range_dir / "logs" in paths
    assert root / "runtime" / "models_stale" in paths

    assert success / "final" not in paths
    assert root / "ledger" not in paths
    assert root / "cache" / "classifier" not in paths
    assert plan["policy"]["destructive_cleanup_enabled"] is False
    assert plan["blocked_count"] == 0


def test_retention_blocks_success_intermediates_without_decision_snapshot(tmp_path: Path):
    now = datetime(2026, 9, 18, tzinfo=timezone.utc)
    root = tmp_path / "outputs" / "96"
    run_dir = root / "runs" / "2026-08-01"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    _touch_old(run_dir / "dayahead" / "weight", now - timedelta(days=31))
    _touch_old(run_dir / "realtime" / "fuse", now - timedelta(days=31))

    plan = build_retention_plan(root, now=now)
    candidate_paths = {Path(item["path"]) for item in plan["candidates"]}
    blocked_paths = {Path(item["path"]) for item in plan["blocked"]}

    assert run_dir / "dayahead" / "weight" not in candidate_paths
    assert run_dir / "realtime" / "fuse" not in candidate_paths
    assert run_dir / "dayahead" / "weight" in blocked_paths
    assert run_dir / "realtime" / "fuse" in blocked_paths
    assert plan["blocked_count"] == 2
    assert all(
        item["kind"] == "blocked_missing_decision_snapshot"
        for item in plan["blocked"]
    )


def test_destructive_apply_is_hard_disabled():
    with pytest.raises(SystemExit, match="DESTRUCTIVE_CLEANUP_DISABLED"):
        main(["--apply"])
