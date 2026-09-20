from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from utils.data_layout import DATA, data_path
from utils.output_layout import resolve_output_layout
from pipelines.ledger_predict import _resolve_runtime_root


def test_production_output_roots_are_phased_without_breaking_hourly_state():
    hourly = resolve_output_layout("production", "hourly")
    quarter = resolve_output_layout("production", "15min")

    assert hourly.ledger_root == Path("outputs/ledger")
    assert hourly.runs_root == Path("outputs/runs")
    assert hourly.feature_store_root == Path("outputs/24/cache")

    assert quarter.ledger_root == Path("outputs/96/ledger")
    assert quarter.runs_root == Path("outputs/96/runs")
    assert quarter.feature_store_root == Path("outputs/96/cache")


def test_formal96_runtime_is_sibling_of_resolved_runs_root(tmp_path):
    runs_root = tmp_path / "isolated" / "runs"
    runtime = _resolve_runtime_root(
        runs_root,
        resolution_label="15min",
        output_profile="production",
        domain="96",
    )
    assert runtime == tmp_path / "isolated" / "runtime"

    legacy = _resolve_runtime_root(
        runs_root,
        resolution_label="15min",
        output_profile="legacy",
        domain="96",
    )
    assert legacy == Path("outputs/96/runtime")


def test_96_training_uses_the_single_persistent_full_store():
    assert data_path("15min", kind="training") == DATA.model_96_full_parquet
    assert data_path("15min") == DATA.model_96_full_parquet
    assert DATA.model_96_clean_parquet != DATA.model_96_full_parquet


def test_lightgbm_model_artifact_isolated_to_runtime(tmp_path, monkeypatch):
    import pipelines.ledger_predict as lp
    import runners.adapters.lightgbm_v1 as adapter_mod

    seen = {}

    class FakeAdapter:
        def __init__(self, **kwargs):
            pass

        def predict(self, **kwargs):
            seen["model_path"] = os.environ.get("LightGBM_MODEL_PATH")
            return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(adapter_mod, "LightGBMV1Adapter", FakeAdapter)
    monkeypatch.delenv("LightGBM_MODEL_PATH", raising=False)

    out = lp._predict_lightgbm(
        "dayahead",
        "2026-08-16",
        "dummy.parquet",
        None,
        False,
        "exact",
        "2026-08-15",
        resolution="15min",
        model_output_root=str(tmp_path),
    )

    assert not out.empty
    assert seen["model_path"] == str(
        tmp_path / "lightgbm" / "dayahead" / "best_model_{}.pkl"
    )
    assert "LightGBM_MODEL_PATH" not in os.environ


def test_registry_models_receive_runner_scratch_root(tmp_path, monkeypatch):
    import pipelines.ledger_predict as lp
    import runners.registry as registry

    captured = {}

    class FakePipeline:
        def predict_range(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("capture-only")

    monkeypatch.setattr(registry, "get_model_pipeline", lambda name: FakePipeline())

    with pytest.raises(RuntimeError, match="capture-only"):
        lp._predict_via_registry(
            "timemixer",
            "dayahead",
            "2026-08-16",
            "dummy.parquet",
            "2026-08-15",
            resolution="15min",
            model_output_root=str(tmp_path),
        )

    assert captured["output_root"] == str(tmp_path)
