from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import main as app_main
from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from pipelines.ledger_full import prepare_daily_run_dir
from utils.asof_view_96 import (
    DYNAMIC_PROTOCOL,
    HISTORICAL_PROXY_PROTOCOL,
    HISTORICAL_PROXY_V1,
    LIVE_DYNAMIC,
    STORED_LIVE_SNAPSHOT_REPLAY,
    build_dynamic_feature_view_96,
    build_dynamic_snapshot_96,
    resolve_formal96_snapshot_route,
)


FORECAST = [
    "直调负荷预测", "地方电厂出力预测", "外电预测", "风电预测", "光伏预测",
    "核电预测", "自备电厂预测", "试验机组预测",
]
ACTUAL = [
    "直调负荷实际", "地方电厂出力实际", "外电实际", "风电实际", "光伏实际",
    "核电实际", "自备电厂实际", "试验机组实际",
]


def _source(path: Path, start: str = "2026-08-13", end: str = "2026-09-20") -> None:
    rows = []
    for day in pd.date_range(start, end, freq="D"):
        for period in range(1, 97):
            row = {"market_date": day, "period_no": period, "日前出清价格": 1000 + period, "实时出清价格": 900 + period}
            row.update({name: 500 + period for name in FORECAST})
            row.update({name: 400 + period for name in ACTUAL})
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_three_way_route_and_proxy_visibility(tmp_path: Path):
    source = tmp_path / "model.parquet"
    _source(source)
    target = "2026-08-15"

    proxy = resolve_formal96_snapshot_route(
        target_day=target, model_store_path=source, authoritative_path=source,
        output_dir=tmp_path / "proxy", current_target_day="2026-09-20",
    )
    assert proxy["route"] == HISTORICAL_PROXY_V1
    assert proxy["manifest"]["protocol"] == HISTORICAL_PROXY_PROTOCOL
    snap = pd.read_parquet(proxy["values_path"])
    decision = snap[snap.market_date.eq(pd.Timestamp("2026-08-14"))]
    assert decision[decision.period_no.le(56)]["实时电价"].notna().all()
    assert decision[decision.period_no.gt(56)]["实时电价"].isna().all()
    assert decision[decision.period_no.gt(56)]["直调负荷实际值"].isna().all()
    assert snap[snap.market_date.eq(pd.Timestamp(target))][["日前电价", "实时电价", "直调负荷实际值"]].isna().all().all()
    _, audit = build_dynamic_feature_view_96(
        model_store_path=source, snapshot_values=proxy["values_path"],
        snapshot_manifest=proxy["manifest"], target_day=target,
    )
    assert audit["protocol"] == HISTORICAL_PROXY_PROTOCOL
    assert audit["target_truth_mask"] is True

    live = resolve_formal96_snapshot_route(
        target_day="2026-09-20", model_store_path=source, authoritative_path=source,
        output_dir=tmp_path / "live", current_target_day="2026-09-20",
    )
    assert live["route"] == LIVE_DYNAMIC
    assert live["manifest"]["protocol"] == DYNAMIC_PROTOCOL

    run_dir = tmp_path / "runs" / target
    stored_dir = run_dir / "snapshot" / "canonical"
    stored = build_dynamic_snapshot_96(
        model_store_path=source, authoritative_path=source, target_day=target,
        output_dir=stored_dir,
    )
    payload = {
        "pipeline": "ledger_predict", "status": "complete", "target_date": target,
        "resolution": "15min", "output_profile": "production",
        "serving_protocol": DYNAMIC_PROTOCOL, "snapshot_id": stored["snapshot_id"],
        "selected_model_pool": {"dayahead": list(DAYAHEAD_MODELS), "realtime": list(REALTIME_MODELS)},
        "results": {
            "dayahead": {model: {"status": "ok"} for model in DAYAHEAD_MODELS},
            "realtime": {model: {"status": "ok"} for model in REALTIME_MODELS},
        },
        "dynamic_snapshot": stored["manifest"],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    replay = resolve_formal96_snapshot_route(
        target_day=target, model_store_path=source, authoritative_path=source,
        output_dir=tmp_path / "must-not-write", run_dir=run_dir,
        current_target_day="2026-09-20",
    )
    assert replay["route"] == STORED_LIVE_SNAPSHOT_REPLAY
    assert replay["snapshot_id"] == stored["snapshot_id"]


def test_route_prefers_latest_closed_day_over_wall_clock(tmp_path: Path):
    source = tmp_path / "model.parquet"
    _source(source, start="2026-09-17", end="2026-09-20")

    # 9/19 is earlier than the wall-clock reference but the synchronized DB
    # says only 9/18 is closed. It must therefore stay on LIVE_DYNAMIC.
    live = resolve_formal96_snapshot_route(
        target_day="2026-09-19",
        model_store_path=source,
        authoritative_path=source,
        output_dir=tmp_path / "live_not_closed",
        latest_closed_day="2026-09-18",
        current_target_day="2026-09-20",
    )
    assert live["route"] == LIVE_DYNAMIC

    proxy = resolve_formal96_snapshot_route(
        target_day="2026-09-18",
        model_store_path=source,
        authoritative_path=source,
        output_dir=tmp_path / "proxy_closed",
        latest_closed_day="2026-09-18",
        current_target_day="2026-09-20",
    )
    assert proxy["route"] == HISTORICAL_PROXY_V1


def test_formal_force_preserves_snapshot_subtree(tmp_path: Path):
    runs = tmp_path / "runs"
    run_dir = runs / "2026-09-20"
    snapshot = run_dir / "snapshot" / "attempt_ok"
    snapshot.mkdir(parents=True)
    (snapshot / "values.parquet").write_bytes(b"snapshot")
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "final").mkdir()
    (run_dir / "final" / "submission_ready.csv").write_text("x", encoding="utf-8")

    prepare_daily_run_dir(
        runs, "2026-09-20", force=True, preserve_snapshot=True
    )
    assert (snapshot / "values.parquet").read_bytes() == b"snapshot"
    assert not (run_dir / "run_manifest.json").exists()
    assert not (run_dir / "final").exists()

    legacy_dir = runs / "2026-09-21"
    legacy_snapshot = legacy_dir / "snapshot" / "legacy"
    legacy_snapshot.mkdir(parents=True)
    (legacy_snapshot / "marker").write_text("x", encoding="utf-8")
    prepare_daily_run_dir(
        runs, "2026-09-21", force=True, preserve_snapshot=False
    )
    assert not (legacy_dir / "snapshot").exists()


def test_formal96_historical_facade_syncs_and_passes_latest_closed_day(monkeypatch):
    observed = {}

    def fake_sync(args):
        observed["sync_called"] = True
        observed["sync_source"] = args.sync_source
        observed["force_sync"] = args.force_sync
        return {
            "status": "ok",
            "source_table": "epf_pmos_96_full",
            "authoritative_rows": 123,
            "latest_closed_day": "2026-09-19",
            "model_inputs": {"full_parquet": "data/96/model_input/fake.parquet"},
            "paths": {"authoritative_csv": "data/96/authoritative/fake.csv"},
        }

    def fake_full(args):
        observed["latest_closed_day"] = getattr(
            args, "_formal96_latest_closed_day", None
        )
        observed["data_path"] = args.data_path
        observed["actual_data_path"] = args.actual_data_path
        return {"delivery_status": "NORMAL"}

    monkeypatch.setattr(app_main, "run_sync_dataset_pipeline", fake_sync)
    monkeypatch.setattr(app_main, "run_ledger_full", fake_full)
    monkeypatch.setattr(sys, "argv", ["main.py", "--96", "2026-08-17"])

    assert app_main.main() == 0
    assert observed["sync_called"] is True
    assert observed["sync_source"] == "db"
    assert observed["force_sync"] is True
    assert observed["latest_closed_day"] == "2026-09-19"
    assert observed["data_path"] == "data/96/model_input/fake.parquet"
    assert observed["actual_data_path"] == "data/96/authoritative/fake.csv"
