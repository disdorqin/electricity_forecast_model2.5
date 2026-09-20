from __future__ import annotations

import json
from pathlib import Path

from scripts.server.audit_96_artifacts import (
    EXPECTED,
    _audit_prediction_ledger,
    _audit_prediction_range_manifest,
    _range_manifest_required,
)


def _write_range(root: Path, *, resource_mode: str = "legacy", status: str = "complete") -> Path:
    path = (
        root
        / "runs"
        / "range_2026-08-15_to_2026-09-15_predict"
        / "prediction_range_manifest.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "effective_start": "2026-08-15",
                "report_start": "2026-08-15",
                "end": "2026-09-15",
                "serving_protocol": "formal96_dynamic_snapshot_v1",
                "serving_visibility_source": "FeatureViewBuilder",
                "execution": {
                    "resource_mode": resource_mode,
                    "cpu_workers": 2 if resource_mode == "split_process" else 1,
                    "gpu_workers": 1,
                    "dag_aware": resource_mode == "split_process",
                },
                "model_input_contract": "DB sync -> immutable D/T snapshot -> FeatureViewBuilder -> models",
                "forecast_vintage": {
                    "status": "UNVERIFIED_LEGACY_VINTAGE",
                    "strict_historical_vintage_proven": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_daily_audit_does_not_require_range_manifest_by_default():
    assert _range_manifest_required("2026-08-16", "2026-08-16") is False
    assert _range_manifest_required(
        "2026-08-16", "2026-08-16", explicit=True
    ) is True
    assert _range_manifest_required("2026-08-16", "2026-08-17") is True


def test_prediction_range_manifest_accepts_current_production_contract(tmp_path: Path):
    _write_range(tmp_path)
    errors: list[str] = []
    _audit_prediction_range_manifest(
        tmp_path, "2026-08-15", "2026-09-15", "legacy", errors
    )
    assert errors == []


def test_prediction_range_manifest_strict_vintage_gate_rejects_latest_state_history(tmp_path: Path):
    _write_range(tmp_path)
    errors: list[str] = []
    _audit_prediction_range_manifest(
        tmp_path,
        "2026-08-15",
        "2026-09-15",
        "legacy",
        errors,
        require_strict_forecast_vintage=True,
    )
    assert any("strict historical forecast vintage is not proven" in error for error in errors)


def test_live_prediction_audit_accepts_partial_target_actual(tmp_path: Path):
    ledger_root = tmp_path / "ledger"
    target = "2026-09-20"
    for task, models in EXPECTED.items():
        pred_rows = []
        for model in models:
            for slot in range(1, 97):
                pred_rows.append({
                    "task": task,
                    "target_day": target,
                    "model_name": model,
                    "business_period": slot,
                    "y_pred": 100.0 + slot,
                    "resolution": "15min",
                })
        pred_dir = ledger_root / task / "prediction"
        pred_dir.mkdir(parents=True)
        import pandas as pd
        pd.DataFrame(pred_rows).to_parquet(
            pred_dir / "prediction_ledger.parquet", index=False
        )

        actual_rows = [
            {
                "task": task,
                "target_day": target,
                "business_period": slot,
                "y_true": 90.0 + slot,
                "resolution": "15min",
            }
            for slot in range(1, 45)
        ]
        act_dir = ledger_root / task / "actual"
        act_dir.mkdir(parents=True)
        pd.DataFrame(actual_rows).to_parquet(
            act_dir / "actual_ledger.parquet", index=False
        )

    errors: list[str] = []
    _audit_prediction_ledger(
        ledger_root, [target], errors, require_target_actual=False
    )
    assert errors == []

    strict_errors: list[str] = []
    _audit_prediction_ledger(
        ledger_root, [target], strict_errors, require_target_actual=True
    )
    assert any("expected exactly p1..p96" in error for error in strict_errors)


def test_prediction_range_manifest_rejects_unaccepted_resource_mode(tmp_path: Path):
    _write_range(tmp_path, resource_mode="split_process")
    errors: list[str] = []
    _audit_prediction_range_manifest(
        tmp_path, "2026-08-15", "2026-09-15", "legacy", errors
    )
    assert any("resource_mode='split_process' expected='legacy'" in error for error in errors)
