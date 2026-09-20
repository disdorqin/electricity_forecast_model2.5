from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import utils.asof_view_96 as asofmod
import pipelines.ledger_full as ledger_full_module
from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from pipelines.ledger_predict import (
    FORMAL96_PREDICTION_CONTRACT,
    _build_cached_result_payload,
    _check_target_actual_readiness,
    _validate_formal96_prediction_cache,
    settle_closed_actuals,
    _write_long_table_single,
)
from utils.data_layout import DATA, data_path
from utils.output_layout import resolve_output_layout
from utils.resolution import QUARTER


def test_96_production_layout_is_domain_scoped():
    layout = resolve_output_layout("production", "15min")
    assert layout.ledger_root == Path("outputs/96/ledger")
    assert layout.runs_root == Path("outputs/96/runs")
    assert layout.feature_store_root == Path("outputs/96/cache")


def test_96_training_uses_single_full_store():
    assert data_path("15min") == DATA.model_96_full_parquet
    assert data_path("15min", "training") == DATA.model_96_full_parquet
    assert data_path("15min", "clean") == DATA.model_96_clean_parquet


def test_transient_asof_is_cleanup_only(tmp_path, monkeypatch):
    runtime_root = tmp_path / "outputs" / "96" / "runtime"
    monkeypatch.setattr(asofmod, "RUNTIME_ROOT", runtime_root)

    path = asofmod.transient_asof_path_96("2026-08-16")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"scratch")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(b"partial")

    assert path.exists()
    assert tmp.exists()

    asofmod.cleanup_transient_asof_96(path)

    assert not path.exists()
    assert not tmp.exists()


def test_formal96_selector_ignores_complete_t_minus_1_truth(tmp_path):
    ledger_root, _ = _write_formal96_full_chain_fixture(
        tmp_path, target="2026-03-15"
    )
    # Add a fully complete D-1 day on purpose. Formal96 must still start at D-2.
    day = pd.Timestamp("2026-03-14")
    for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
        pred_rows = []
        actual_rows = []
        for slot in range(1, 97):
            ds = day + pd.Timedelta(minutes=15 * slot)
            actual_rows.append({
                "task": task, "target_day": "2026-03-14",
                "business_day": "2026-03-14", "business_period": slot,
                "hour_business": (slot - 1) // 4 + 1,
                "period": _period_label(slot), "ds": ds,
                "y_true": 200.0 + slot, "resolution": "15min",
            })
            for idx, model in enumerate(models):
                pred_rows.append({
                    "task": task, "model_name": model,
                    "target_day": "2026-03-14", "business_day": "2026-03-14",
                    "business_period": slot, "hour_business": (slot - 1) // 4 + 1,
                    "period": _period_label(slot), "ds": ds,
                    "y_pred": 200.0 + slot + idx, "resolution": "15min",
                })
        from pipelines.prediction_ledger import (
            append_predictions_to_ledger,
            update_actual_ledger,
        )
        append_predictions_to_ledger(pd.DataFrame(pred_rows), ledger_root, task)
        update_actual_ledger(pd.DataFrame(actual_rows), ledger_root, task)

    from pipelines.ledger_weight import select_complete_training_days
    for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
        selected = select_complete_training_days(
            task=task,
            target_date="2026-03-15",
            ledger_root=ledger_root,
            expected_models=list(models),
            required_days=30,
            max_lookback_days=90,
            resolution=QUARTER,
            history_lag_days=2,
        )
        assert selected["status"] == "PASS"
        assert selected["anchor_start"] == "2026-03-13"
        assert selected["selected_days"][0] == "2026-03-13"
        assert "2026-03-14" not in selected["selected_days"]


def test_formal96_closed_actual_settlement_uses_t_minus_2_not_t_minus_1(tmp_path):
    source = tmp_path / "authoritative.csv"
    rows = []
    for day in ("2026-08-14", "2026-08-15"):
        for slot in range(1, 97):
            rows.append({
                "market_date": day,
                "时段": slot,
                "日前出清价格": 300.0 + slot,
                "实时出清价格": 280.0 + slot,
            })
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")

    ledger_root = tmp_path / "ledger"
    result = settle_closed_actuals(
        str(source),
        "2026-08-16",
        ledger_root,
        resolution=QUARTER,
        tasks=("dayahead", "realtime"),
        output_profile="production",
        settlement_lag_days=2,
    )

    assert result["status"] == "complete"
    assert result["closed_day"] == "2026-08-14"
    for task in ("dayahead", "realtime"):
        actual = pd.read_parquet(
            ledger_root / task / "actual" / "actual_ledger.parquet"
        )
        assert set(actual["target_day"].astype(str)) == {"2026-08-14"}
        assert len(actual) == 96


def test_target_actual_gate_uses_current_target_rows_not_ledger_total():
    manifest = {
        "results": {
            "dayahead_actual_ledger": {"rows_after": 192, "target_day_rows": 96},
            "realtime_actual_ledger": {"rows_after": 192, "target_day_rows": 96},
        },
        "warnings": [],
    }
    _check_target_actual_readiness(
        manifest=manifest,
        tasks=("dayahead", "realtime"),
        target_date="2026-08-16",
        slots_per_day=96,
        require_target_actual=True,
    )
    assert manifest["warnings"] == []

    incomplete = {
        "results": {
            "realtime_actual_ledger": {"rows_after": 960, "target_day_rows": 60},
        },
        "warnings": [],
    }
    with pytest.raises(RuntimeError, match="rows=60 expected=96"):
        _check_target_actual_readiness(
            manifest=incomplete,
            tasks=("realtime",),
            target_date="2026-08-16",
            slots_per_day=96,
            require_target_actual=True,
        )

    _check_target_actual_readiness(
        manifest=incomplete,
        tasks=("realtime",),
        target_date="2026-08-16",
        slots_per_day=96,
        require_target_actual=False,
    )
    assert any("allowed in live prediction mode" in w for w in incomplete["warnings"])


def _formal_cache_frame(model_name: str, task: str) -> pd.DataFrame:
    target = "2026-08-16"
    start = pd.Timestamp(target)
    rows = []
    for period in range(1, 97):
        ds = start + pd.Timedelta(minutes=15 * period)
        row = {
            "task": task,
            "model_name": model_name,
            "target_day": target,
            "business_day": target,
            "ds": ds,
            "business_period": period,
            "y_pred": float(period),
            "data_cutoff": "dynamic_snapshot",
            "da_feature_source": (
                "none" if task == "dayahead" else {
                    "timesfm": "timesfm_none",
                    "sgdfnet": "sgdfnet_decision_day_da_anchor",
                    "timemixer": "timemixer_internal_dayahead_prediction",
                    "rt916": "rt916_internal_joint_dayahead_prediction",
                }[model_name]
            ),
            "production_contract": FORMAL96_PREDICTION_CONTRACT,
            "production_resolution": "15min",
            "production_resource_mode": "split_process",
            "production_rt_cutoff_hour": "dynamic_snapshot",
            "serving_protocol": FORMAL96_PREDICTION_CONTRACT,
            "snapshot_id": "snapshot-test",
        }
        if model_name == "rt916":
            row["production_rt916_train_steps"] = 24
        if model_name == "sgdfnet":
            row.update({
                "anchor_source_day": "2026-08-15",
                "anchor_source_type": "decision_day_da",
                "anchor_rows": 96,
                "fallback_used": False,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def test_formal96_cache_requires_current_contract():
    df = _formal_cache_frame("rt916", "realtime")
    assert _validate_formal96_prediction_cache(
        df,
        target_date="2026-08-16",
        model_name="rt916",
        task="realtime",
        expected_cutoff="dynamic_snapshot",
    ) == []

    stale = df.drop(columns=["production_contract"])
    errors = _validate_formal96_prediction_cache(
        stale,
        target_date="2026-08-16",
        model_name="rt916",
        task="realtime",
        expected_cutoff="dynamic_snapshot",
    )
    assert any("cache missing production_contract" in error for error in errors)


def test_formal96_sgdfnet_cache_requires_strict_anchor():
    df = _formal_cache_frame("sgdfnet", "realtime")
    assert _validate_formal96_prediction_cache(
        df,
        target_date="2026-08-16",
        model_name="sgdfnet",
        task="realtime",
        expected_cutoff="dynamic_snapshot",
    ) == []

    leaked = df.copy()
    leaked["anchor_source_day"] = "2026-08-16"
    leaked["fallback_used"] = True
    errors = _validate_formal96_prediction_cache(
        leaked,
        target_date="2026-08-16",
        model_name="sgdfnet",
        task="realtime",
        expected_cutoff="dynamic_snapshot",
    )
    assert any("anchor_source_day" in error for error in errors)
    assert any("fallback_used=true" in error for error in errors)


def test_formal96_sgdfnet_cache_payload_preserves_anchor_contract(tmp_path):
    df = _formal_cache_frame("sgdfnet", "realtime")
    payload = _build_cached_result_payload(
        df,
        output_path=tmp_path / "sgdfnet_predictions.csv",
        model_name="sgdfnet",
        task_name="realtime",
    )
    assert payload["status"] == "cached"
    assert payload["rows"] == 96
    assert payload["anchor_contract"] == {
        "anchor_source_day": "2026-08-15",
        "source_type": "decision_day_da",
        "rows": 96,
        "fallback_used": False,
    }


def _period_label(slot: int) -> str:
    if slot <= 32:
        return "1_32"
    if slot <= 64:
        return "33_64"
    return "65_96"


def _write_formal96_full_chain_fixture(root: Path, target: str = "2026-03-15") -> tuple[Path, Path]:
    ledger_root = root / "ledger"
    runs_root = root / "runs"
    target_ts = pd.Timestamp(target)

    for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
        pred_rows: list[dict] = []
        actual_rows: list[dict] = []
        # 30 causal historical days for formal96 smape_reg.  Dynamic-v1 keeps
        # the learner's closed-history lag, so the window is D-31..D-2.
        # Target-day predictions are appended below.
        for offset in range(31, 1, -1):
            day = (target_ts - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            for slot in range(1, 97):
                ds = pd.Timestamp(day) + pd.Timedelta(minutes=15 * slot)
                hour = (slot - 1) // 4 + 1
                y_true = 300.0 + 0.7 * slot + 0.2 * offset
                actual_rows.append({
                    "task": task,
                    "target_day": day,
                    "business_day": day,
                    "business_period": slot,
                    "hour_business": hour,
                    "period": _period_label(slot),
                    "ds": ds,
                    "y_true": y_true,
                    "resolution": "15min",
                })
                for model_idx, model in enumerate(models):
                    pred_rows.append({
                        "task": task,
                        "model_name": model,
                        "target_day": day,
                        "business_day": day,
                        "business_period": slot,
                        "hour_business": hour,
                        "period": _period_label(slot),
                        "ds": ds,
                        "y_pred": y_true + (model_idx - 1) * 2.0 + (slot % 5) * 0.1,
                        "resolution": "15min",
                    })

        # Current target-day prediction ledger append and per-model run files.
        for slot in range(1, 97):
            ds = target_ts + pd.Timedelta(minutes=15 * slot)
            hour = (slot - 1) // 4 + 1
            for model_idx, model in enumerate(models):
                row = {
                    "task": task,
                    "model_name": model,
                    "forecast_date": target,
                    "target_day": target,
                    "business_day": target,
                    "business_period": slot,
                    "hour_business": hour,
                    "period": _period_label(slot),
                    "ds": ds,
                    "y_pred": 320.0 + 0.6 * slot + model_idx,
                    "data_cutoff": "dynamic_snapshot",
                    "run_id": f"{model}_formal96_fixture",
                    "model_version": "test",
                    "da_feature_source": (
                        "none" if task == "dayahead" else {
                            "timesfm": "timesfm_none",
                            "sgdfnet": "sgdfnet_decision_day_da_anchor",
                            "timemixer": "timemixer_internal_dayahead_prediction",
                            "rt916": "rt916_internal_joint_dayahead_prediction",
                        }[model]
                    ),
                    "production_contract": FORMAL96_PREDICTION_CONTRACT,
                    "production_resolution": "15min",
                    "production_resource_mode": "split_process",
                    "production_rt_cutoff_hour": "dynamic_snapshot",
                    "serving_protocol": FORMAL96_PREDICTION_CONTRACT,
                    "snapshot_id": "snapshot-test",
                    "resolution": "15min",
                }
                if model == "rt916":
                    row["production_rt916_train_steps"] = 24
                if model == "sgdfnet":
                    row.update({
                        "anchor_source_day": (target_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        "anchor_source_type": "decision_day_da",
                        "anchor_rows": 96,
                        "fallback_used": False,
                    })
                pred_rows.append(row)

        pred_dir = ledger_root / task / "prediction"
        act_dir = ledger_root / task / "actual"
        pred_dir.mkdir(parents=True, exist_ok=True)
        act_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_rows).to_parquet(pred_dir / "prediction_ledger.parquet", index=False)
        pd.DataFrame(actual_rows).to_parquet(act_dir / "actual_ledger.parquet", index=False)

        run_pred_dir = runs_root / target / task / "prediction"
        run_pred_dir.mkdir(parents=True, exist_ok=True)
        target_df = pd.DataFrame([row for row in pred_rows if row["target_day"] == target])
        for model in models:
            target_df[target_df["model_name"] == model].to_csv(
                run_pred_dir / f"{model}_predictions.csv", index=False
            )
        target_df.to_csv(run_pred_dir / "all_model_predictions_long.csv", index=False)

    source_manifest = {
        "pipeline": "ledger_predict",
        "status": "complete",
        "target_date": target,
        "resolution": "15min",
        "output_profile": "production",
        "resource_mode": "split_process",
        "requested_tasks": ["dayahead", "realtime"],
        "serving_protocol": FORMAL96_PREDICTION_CONTRACT,
        "snapshot_id": "snapshot-test",
        "selected_model_pool": {
            "dayahead": list(DAYAHEAD_MODELS),
            "realtime": list(REALTIME_MODELS),
        },
        "production_config": {"formal_96": True, "rt916_train_steps": 24},
        "dynamic_snapshot": {
            "status": "PASS",
            "protocol": FORMAL96_PREDICTION_CONTRACT,
            "snapshot_id": "snapshot-test",
            "target_day": target,
            "decision_day": (target_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "values_path": str(runs_root / target / "snapshot" / "values.parquet"),
            "manifest_path": str(runs_root / target / "snapshot" / "snapshot_manifest.json"),
        },
        "feature_view": {"status": "PASS", "target_truth_mask": True},
        "results": {
            "realtime": {
                "sgdfnet": {
                    "anchor_contract": {
                        "anchor_source_day": (target_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        "rows": 96,
                        "fallback_used": False,
                    }
                }
            }
        },
    }
    run_dir = runs_root / target
    snapshot_dir = run_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_values = snapshot_dir / "values.parquet"
    snapshot_manifest = snapshot_dir / "snapshot_manifest.json"
    snapshot_values.touch()
    snapshot_manifest.write_text(
        json.dumps({
            "protocol": FORMAL96_PREDICTION_CONTRACT,
            "snapshot_id": "snapshot-test",
            "target_day": target,
            "decision_day": (target_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "values_path": str(snapshot_values),
            "manifest_path": str(snapshot_manifest),
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False), encoding="utf-8"
    )
    return ledger_root, runs_root


def test_formal96_replay_runs_weight_fuse_final_postflight(tmp_path):
    ledger_root, runs_root = _write_formal96_full_chain_fixture(tmp_path)

    actual_source = tmp_path / "authoritative_actual.csv"
    actual_rows = []
    for slot in range(1, 97):
        actual_rows.append({
            "market_date": "2026-03-13",
            "时段": slot,
            "日前出清价格": 310.0 + slot,
            "实时出清价格": 290.0 + slot,
        })
    pd.DataFrame(actual_rows).to_csv(
        actual_source, index=False, encoding="utf-8-sig"
    )

    args = SimpleNamespace(
        date="2026-03-15",
        resolution="15min",
        target="both",
        ledger_root=str(ledger_root),
        runs_root=str(runs_root),
        force=False,
        replay_only=True,
        output_profile="production",
        resource_mode="split_process",
        max_cpu_workers=2,
        max_gpu_workers=1,
        validation_days=30,
        weight_max_lookback_days=90,
        weight_learner="smape_reg",
        weight_granularity="period",
        weight_prune_threshold=0.05,
        weight_min_active_models=1,
        allow_equal_weight_fallback=False,
        allow_missing_models=False,
        recent_week_boost=True,
        recent_week_max_gate=0.85,
        realtime_cutoff_hour=15,
        rt916_train_steps=24,
        data_path="unused-formal96-fixture.parquet",
        actual_data_path=str(actual_source),
    )

    result = ledger_full_module.run_ledger_full(args)

    assert result["status"] == "complete"
    assert result["delivery_status"] == "NORMAL"
    assert result["stages"]["ledger_predict"]["status"] == "complete"
    assert result["stages"]["ledger_weight"]["status"] in {"complete", "complete_with_warnings"}
    assert result["stages"]["ledger_fuse"]["status"] == "complete"
    assert result["stages"]["ledger_classifier"]["status"] == "disabled_by_production_policy"
    assert result["stages"]["final_outputs"]["status"] == "complete"
    assert result["postflight"]["status"] == "PASS"
    assert result["decision_snapshot"]["tasks"]["dayahead"]["weights"]
    assert result["decision_snapshot"]["tasks"]["realtime"]["weights"]
    assert not (runs_root / "2026-03-15" / "runtime" / "stage_manifests").exists()

    submission = pd.read_csv(runs_root / "2026-03-15" / "final" / "submission_ready.csv")
    assert len(submission) == 96
    assert submission[["dayahead_price", "realtime_price"]].notna().all().all()


def test_formal96_postflight_accepts_complete_with_warnings(tmp_path):
    ledger_root, runs_root = _write_formal96_full_chain_fixture(tmp_path)
    run_dir = runs_root / "2026-03-15"

    actual_source = tmp_path / "authoritative_actual_warn.csv"
    pd.DataFrame([
        {
            "market_date": "2026-03-13",
            "时段": slot,
            "日前出清价格": 310.0 + slot,
            "实时出清价格": 290.0 + slot,
        }
        for slot in range(1, 97)
    ]).to_csv(actual_source, index=False, encoding="utf-8-sig")

    args = SimpleNamespace(
        date="2026-03-15", resolution="15min", target="both",
        ledger_root=str(ledger_root), runs_root=str(runs_root),
        force=False, replay_only=True, output_profile="production",
        resource_mode="split_process", max_cpu_workers=2, max_gpu_workers=1,
        validation_days=30, weight_max_lookback_days=90,
        weight_learner="smape_reg", weight_granularity="period",
        weight_prune_threshold=0.05, weight_min_active_models=1,
        allow_equal_weight_fallback=False, allow_missing_models=False,
        recent_week_boost=True, recent_week_max_gate=0.85,
        realtime_cutoff_hour=15, rt916_train_steps=24,
        data_path="unused-formal96-fixture.parquet",
        actual_data_path=str(actual_source),
    )
    result = ledger_full_module.run_ledger_full(args)
    assert result["delivery_status"] == "NORMAL"

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["ledger_predict"]["status"] = "complete_with_warnings"
    manifest["stages"]["ledger_predict"]["warnings"] = [
        "realtime target-day actual not complete (44/96); allowed in live prediction mode"
    ]
    manifest["delivery_status"] = "NORMAL"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    from pipelines.delivery_quality import validate_daily_submission
    check = validate_daily_submission(
        runs_root, "2026-03-15", resolution=QUARTER
    )
    assert check["status"] == "PASS", check["errors"]


def test_formal96_postflight_failure_never_uses_emergency_fallback(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "2026-03-15"
    run_dir.mkdir(parents=True)
    manifest = {
        "pipeline": "ledger_full",
        "target_date": "2026-03-15",
        "status": "complete",
        "production_config": {"formal_96": True},
        "stages": {
            "ledger_predict": {"status": "complete"},
            "ledger_weight": {"status": "complete"},
            "ledger_fuse": {"status": "complete"},
            "ledger_classifier": {"status": "disabled_by_production_policy"},
            "final_outputs": {"status": "complete"},
        },
        "warnings": [],
        "errors": [],
    }
    args = SimpleNamespace(
        resolution="15min", target="both",
        runs_root=str(tmp_path / "runs"),
        ledger_root=str(tmp_path / "ledger"),
        data_path=str(tmp_path / "data.parquet"),
    )

    import pipelines.delivery_quality as dq
    import pipelines.emergency_fallback as ef
    monkeypatch.setattr(
        dq, "validate_daily_submission",
        lambda *a, **k: {"status": "FAIL", "errors": ["synthetic postflight failure"], "warnings": []},
    )
    monkeypatch.setattr(
        dq, "validate_next_day_readiness",
        lambda *a, **k: {"status": "SKIPPED"},
    )
    monkeypatch.setattr(
        ef, "try_emergency_fallback",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("formal96 emergency fallback must not be called")
        ),
    )

    result = ledger_full_module._finalize_delivery(args, manifest)
    assert result["delivery_status"] == "FAILED_NO_DELIVERY"
    assert result["fallback"]["fallback_used"] is False
    assert result["fallback"]["policy"] == "disabled_by_production_policy"


def test_finish_accepts_complete_with_warnings_prediction_provenance(tmp_path):
    ledger_root, runs_root = _write_formal96_full_chain_fixture(tmp_path)
    run_dir = runs_root / "2026-03-15"
    source = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    source["status"] = "complete_with_warnings"
    source["warnings"] = [
        "realtime target-day actual not complete (44/96); allowed in live prediction mode"
    ]

    result = ledger_full_module._validate_finish_prediction_provenance(
        source,
        target_date="2026-03-15",
        run_dir=run_dir,
        ledger_root=ledger_root,
    )
    assert result["status"] == "PASS", result["errors"]


def test_finish_rejects_persisted_snapshot_manifest_mismatch(tmp_path):
    ledger_root, runs_root = _write_formal96_full_chain_fixture(tmp_path)
    run_dir = runs_root / "2026-03-15"
    source = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))

    snapshot_manifest_path = Path(source["dynamic_snapshot"]["manifest_path"])
    persisted = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    persisted["snapshot_id"] = "tampered-snapshot-id"
    snapshot_manifest_path.write_text(
        json.dumps(persisted, ensure_ascii=False), encoding="utf-8"
    )

    result = ledger_full_module._validate_finish_prediction_provenance(
        source,
        target_date="2026-03-15",
        run_dir=run_dir,
        ledger_root=ledger_root,
    )
    assert result["status"] == "FAIL"
    assert any(
        "persisted dynamic snapshot_id mismatch" in err
        for err in result["errors"]
    )


def test_same_day_rerun_overwrites_previous_delivery_slot(tmp_path):
    run_dir = tmp_path / "runs" / "2026-08-16"
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "submission_ready.csv").write_text("version\nfirst\n", encoding="utf-8")
    (run_dir / "delivery_report.json").write_text('{"version": 1}', encoding="utf-8")

    ledger_full_module._isolate_stale_delivery_artifacts(run_dir, "attempt-one")
    previous = run_dir / "runtime" / "diagnostics" / "stale_delivery_previous"
    assert (previous / "final" / "submission_ready.csv").read_text(encoding="utf-8").endswith("first\n")

    final_dir.mkdir(parents=True)
    (final_dir / "submission_ready.csv").write_text("version\nsecond\n", encoding="utf-8")
    (run_dir / "delivery_report.json").write_text('{"version": 2}', encoding="utf-8")
    ledger_full_module._isolate_stale_delivery_artifacts(run_dir, "attempt-two")

    stale_dirs = list((run_dir / "runtime" / "diagnostics").glob("stale_delivery*"))
    assert stale_dirs == [previous]
    assert (previous / "final" / "submission_ready.csv").read_text(encoding="utf-8").endswith("second\n")
    meta = json.loads((previous / "archive_manifest.json").read_text(encoding="utf-8"))
    assert meta["replaced_by_attempt_id"] == "attempt-two"


def test_partial_model_run_does_not_publish_stale_sibling_csv(tmp_path):
    pred_dir = tmp_path / "realtime" / "prediction"
    pred_dir.mkdir(parents=True)
    selected = _formal_cache_frame("sgdfnet", "realtime")
    stale = _formal_cache_frame("timesfm", "realtime")
    selected.to_csv(pred_dir / "sgdfnet_predictions.csv", index=False)
    stale.to_csv(pred_dir / "timesfm_predictions.csv", index=False)

    manifest = {
        "selected_model_pool": {"dayahead": [], "realtime": ["sgdfnet"]},
        "results": {},
        "warnings": [],
    }
    from utils.resolution import QUARTER

    _write_long_table_single(
        tmp_path, "2026-08-16", "realtime", manifest, resolution=QUARTER
    )
    long_df = pd.read_csv(pred_dir / "all_model_predictions_long.csv")
    assert len(long_df) == 96
    assert set(long_df["model_name"]) == {"sgdfnet"}
    assert manifest["results"]["realtime_long_rows"] == 96
    assert manifest["warnings"] == []
