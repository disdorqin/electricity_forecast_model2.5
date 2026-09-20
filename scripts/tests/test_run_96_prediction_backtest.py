from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipelines.ledger_full as ledger_full_module
import pipelines.ledger_predict as ledger_predict_module
from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from pipelines.ledger_full import (
    _extract_prediction_provenance,
    _validate_finish_prediction_provenance,
)
from scripts.server.run_96_prediction_backtest import _protocol_manifest_audit

DYNAMIC_PROTOCOL = "formal96_dynamic_snapshot_v1"


FORECASTS = {
    "直调负荷预测值": 96,
    "地方电厂总加预测值": 96,
    "联络线受电负荷预测值": 96,
    "风电总加预测值": 96,
    "光伏总加预测值": 96,
    "核电总加预测值": 96,
    "自备机组总加预测值": 96,
    "试验机组总加预测值": 96,
    "竞价空间预测值": 96,
    "新能源总加预测值": 96,
}

ACTUALS = {
    "直调负荷实际值": 60,
    "地方电厂总加实际值": 60,
    "联络线受电负荷实际值": 60,
    "风电总加实际值": 60,
    "光伏总加实际值": 60,
    "核电总加实际值": 60,
    "自备机组总加实际值": 60,
    "试验机组总加实际值": 60,
    "竞价空间实际值": 60,
    "新能源总加实际值": 60,
}


def _manifest(resource_mode: str = "legacy") -> dict:
    return {
        "status": "complete",
        "resolution": "15min",
        "serving_protocol": DYNAMIC_PROTOCOL,
        "resource_mode": resource_mode,
        "model_input_source": "data/96/model_input/shandong_pmos_96_model_input_full.parquet",
        "selected_model_pool": {
            "dayahead": list(DAYAHEAD_MODELS),
            "realtime": list(REALTIME_MODELS),
        },
        "snapshot_id": "snapshot-test",
        "dynamic_snapshot": {
            "protocol": DYNAMIC_PROTOCOL,
            "snapshot_id": "snapshot-test",
            "target_day": "2026-08-16",
            "decision_day": "2026-08-15",
            "grid_rows": 192,
        },
        "feature_view": {"status": "PASS", "target_truth_mask": True},
        "asof_view": {
            "status": "PASS",
            "target_day": "2026-08-16",
            "decision_day": "2026-08-15",
            "decision_day_da_visible": 96,
            "decision_day_rt_visible": 60,
            "decision_day_actual_visible": ACTUALS,
            "target_forecast_nonnull": FORECASTS,
            "target_realized_nonnull": 0,
        },
    }


def _write(runs_root: Path, payload: dict) -> None:
    run_dir = runs_root / "2026-08-16"
    run_dir.mkdir(parents=True)
    snapshot_dir = run_dir / "snapshot"
    snapshot_dir.mkdir()
    values_path = snapshot_dir / "values.parquet"
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    values_path.touch()
    manifest_path.write_text("{}", encoding="utf-8")
    snapshot_owner = payload.get("prediction_provenance") if isinstance(payload.get("prediction_provenance"), dict) else payload
    snapshot_owner.setdefault("dynamic_snapshot", {}).update({
        "values_path": str(values_path),
        "manifest_path": str(manifest_path),
    })
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_protocol_manifest_accepts_current_strict_contract(tmp_path: Path):
    runs = tmp_path / "runs"
    _write(runs, _manifest())
    ok, reasons = _protocol_manifest_audit(
        runs, "2026-08-16", resource_mode="legacy"
    )
    assert ok, reasons


def test_protocol_manifest_rejects_wrong_dynamic_contract_or_other_resource_mode(tmp_path: Path):
    runs = tmp_path / "runs"
    payload = _manifest(resource_mode="split_process")
    payload["serving_protocol"] = "formal96_prediction_v1"
    payload["dynamic_snapshot"]["protocol"] = "formal96_prediction_v1"
    _write(runs, payload)

    ok, reasons = _protocol_manifest_audit(
        runs, "2026-08-16", resource_mode="legacy"
    )
    assert not ok
    joined = " | ".join(reasons)
    assert "serving_protocol='formal96_prediction_v1'" in joined
    assert "resource_mode='split_process' expected='legacy'" in joined


def test_protocol_manifest_accepts_replay_manifest_with_preserved_prediction_provenance(tmp_path: Path):
    runs = tmp_path / "runs"
    replay = {
        "pipeline": "ledger_full",
        "status": "complete",
        "mode": "replay_only",
        "prediction_provenance": _manifest(),
        "stages": {"ledger_predict": {"status": "complete", "reused": True}},
    }
    _write(runs, replay)

    extracted = _extract_prediction_provenance(replay)
    assert extracted is not None and extracted["snapshot_id"] == "snapshot-test"
    ok, reasons = _protocol_manifest_audit(
        runs, "2026-08-16", resource_mode="legacy"
    )
    assert ok, reasons


def test_replay_only_production_96_without_prediction_provenance_fails_closed(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        ledger_full_module,
        "_finalize_delivery",
        lambda _args, manifest: manifest,
    )
    settlement_calls: list[str] = []
    monkeypatch.setattr(
        ledger_predict_module,
        "settle_closed_actuals",
        lambda *_args, **_kwargs: settlement_calls.append("called") or {"status": "complete"},
    )
    args = SimpleNamespace(
        date="2026-08-16",
        resolution="15min",
        target="both",
        ledger_root=str(tmp_path / "ledger"),
        runs_root=str(tmp_path / "runs"),
        force=False,
        replay_only=True,
        output_profile="production",
    )
    result = ledger_full_module.run_ledger_full(args)
    assert result["status"] == "failed"
    assert any("requires strict prediction provenance" in error for error in result["errors"])
    assert result["stages"]["ledger_predict"]["reason"] == "INCOMPLETE_PREDICTION_PROVENANCE"
    assert settlement_calls == []


def test_protocol_manifest_rejects_nonfull_source_or_incomplete_forecast(tmp_path: Path):
    runs = tmp_path / "runs"
    payload = _manifest()
    payload["model_input_source"] = "data/96/model_input/shandong_pmos_96_model_input_clean.parquet"
    payload["dynamic_snapshot"]["grid_rows"] = 191
    _write(runs, payload)

    ok, reasons = _protocol_manifest_audit(
        runs, "2026-08-16", resource_mode="legacy"
    )
    assert not ok
    assert any("model_input_source" in reason for reason in reasons)
    assert any("grid_rows=191" in reason for reason in reasons)


def test_finish_partial_prediction_provenance_fails_closed(tmp_path: Path):
    run_dir = tmp_path / "runs" / "2026-08-16"
    ledger = tmp_path / "ledger"
    run_dir.mkdir(parents=True)
    source = _manifest()
    source.update({
        "target_date": "2026-08-16",
        "output_profile": "production",
        "requested_tasks": ["dayahead", "realtime"],
        "production_config": {"rt916_train_steps": 24},
        "results": {"realtime": {"sgdfnet": {"anchor_contract": {
            "anchor_source_day": "2026-08-15", "rows": 96, "fallback_used": False,
        }}}},
    })
    # Deliberately create only a partial scope; no stage may be reused.
    (run_dir / "dayahead" / "prediction").mkdir(parents=True)
    result = _validate_finish_prediction_provenance(
        source, target_date="2026-08-16", run_dir=run_dir, ledger_root=ledger,
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "INCOMPLETE_PREDICTION_PROVENANCE"


def test_full_cold_start_failure_preserves_prior_prediction_provenance(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "2026-08-16"
    run_dir.mkdir(parents=True)
    source = _manifest(resource_mode="split_process")
    source.update({
        "pipeline": "ledger_predict",
        "target_date": "2026-08-16",
        "output_profile": "production",
        "requested_tasks": ["dayahead", "realtime"],
        "production_config": {"rt916_train_steps": 24},
        "status": "complete",
    })
    (run_dir / "run_manifest.json").write_text(
        json.dumps(source, ensure_ascii=False), encoding="utf-8"
    )

    args = SimpleNamespace(
        date="2026-08-16", resolution="15min", target="both",
        ledger_root=str(tmp_path / "ledger"), runs_root=str(runs_root),
        force=False, replay_only=False, output_profile="production",
        resource_mode="split_process", max_cpu_workers=2, max_gpu_workers=1,
        validation_days=30, weight_max_lookback_days=90,
        weight_learner="smape_reg", weight_granularity="period",
        weight_prune_threshold=0.05, data_path="unused.parquet",
    )
    result = ledger_full_module.run_ledger_full(args)
    assert result["delivery_status"] == "FAILED_NO_DELIVERY"
    assert "INSUFFICIENT_STRICT_HISTORY" in result["errors"]

    payload = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    preserved = _extract_prediction_provenance(payload)
    assert preserved is not None
    assert preserved["pipeline"] == "ledger_predict"
    assert preserved["status"] == "complete"
    assert preserved["target_date"] == "2026-08-16"


def test_interrupted_full_attempt_owns_running_manifest(tmp_path: Path, monkeypatch):
    def _ready(*_args, **_kwargs):
        return {"status": "PASS", "ready": True, "tasks": {}}

    def _interrupt(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(ledger_full_module, "_strict_history_readiness", _ready)
    monkeypatch.setattr("pipelines.ledger_predict.run_ledger_predict", _interrupt)
    args = SimpleNamespace(
        date="2026-08-16", resolution="15min", target="both",
        ledger_root=str(tmp_path / "ledger"), runs_root=str(tmp_path / "runs"),
        force=False, replay_only=False, output_profile="production",
        resource_mode="split_process", max_cpu_workers=2, max_gpu_workers=1,
        validation_days=30, weight_max_lookback_days=90,
        weight_learner="smape_reg", weight_granularity="period",
        weight_prune_threshold=0.05,
    )
    with pytest.raises(KeyboardInterrupt):
        ledger_full_module.run_ledger_full(args)
    payload = json.loads((tmp_path / "runs" / "2026-08-16" / "run_manifest.json").read_text())
    assert payload["pipeline"] == "ledger_full"
    assert payload["status"] == "interrupted"
    assert payload["delivery_status"] == "FAILED_NO_DELIVERY"
