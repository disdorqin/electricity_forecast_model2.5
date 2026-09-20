"""Controlled formal96 Dynamic-v1 smoke without starting heavyweight models.

The real model scheduler is replaced only inside this test with a seven-leg
writer.  Snapshot creation, FeatureView routing, cache provenance, ledger
append and manifest contracts still execute through ``run_ledger_predict``.
"""
from __future__ import annotations

import tempfile
import sys
import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pipelines.ledger_predict as lp
from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from utils.asof_view_96 import PRIMITIVE_ACTUAL_COLUMNS, PRIMITIVE_FORECAST_COLUMNS

PROTOCOL = "formal96_dynamic_snapshot_v1"
TARGET = "2026-09-20"
DECISION = "2026-09-19"


def _source(path: Path) -> None:
    rows: list[dict] = []
    for day in ("2026-09-18", DECISION, TARGET):
        for period in range(1, 97):
            row = {"market_date": day, "period_no": period}
            for idx, col in enumerate(PRIMITIVE_FORECAST_COLUMNS):
                row[col] = 1000.0 + idx * 10 + period
            for idx, col in enumerate(PRIMITIVE_ACTUAL_COLUMNS):
                row[col] = 500.0 + idx * 5 + period
            row["日前电价"] = 300.0 + period
            row["实时电价"] = 310.0 + period
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _fake_plan(*, target_date, selected_da_models, selected_rt_models,
               run_dir, common_kwargs, **_kwargs):
    snapshot_id = common_kwargs["snapshot_id"]
    results = {"dayahead": {}, "realtime": {}}
    for task, models in (("dayahead", selected_da_models), ("realtime", selected_rt_models)):
        out_dir = run_dir / task / "prediction"
        out_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            rows = []
            for period in range(1, 97):
                row = {
                    "task": task, "model_name": model,
                    "forecast_date": target_date, "target_day": target_date,
                    "business_day": target_date,
                    "business_period": period,
                    "hour_business": (period - 1) // 4 + 1,
                    "period": ("1_32" if period <= 32 else "33_64" if period <= 64 else "65_96"),
                    "ds": pd.Timestamp(target_date) + pd.Timedelta(minutes=15 * period),
                    "y_pred": float(200 + period),
                    "data_cutoff": DECISION,
                    "run_id": "controlled-dynamic-smoke",
                    "model_version": "test",
                    "production_contract": PROTOCOL,
                    "serving_protocol": PROTOCOL,
                    "snapshot_id": snapshot_id,
                    "production_resolution": "15min",
                    "production_resource_mode": "split_process",
                    "da_feature_source": "none" if task == "dayahead" else {
                        "timesfm": "timesfm_none",
                        "sgdfnet": "sgdfnet_decision_day_da_anchor",
                        "timemixer": "timemixer_internal_dayahead_prediction",
                        "rt916": "rt916_internal_joint_dayahead_prediction",
                    }[model],
                }
                if model == "rt916":
                    row["production_rt916_train_steps"] = 24
                if model == "sgdfnet":
                    row.update({
                        "anchor_source_day": DECISION,
                        "anchor_source_type": "decision_day_da",
                        "anchor_rows": 96,
                        "fallback_used": False,
                    })
                rows.append(row)
            output = out_dir / f"{model}_predictions.csv"
            pd.DataFrame(rows).to_csv(output, index=False)
            results[task][model] = {"status": "ok", "output_path": str(output), "rows": 96}
    return results


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efm3_dynamic_smoke_") as tmp:
        root = Path(tmp)
        model_store = root / "model_store.csv"
        authority = root / "authority.csv"
        _source(model_store)
        _source(authority)
        args = SimpleNamespace(
            date=TARGET, target="both", models="all", resolution="15min",
            data_path=str(model_store), actual_data_path=str(authority),
            ledger_root=str(root / "ledger"), runs_root=str(root / "runs"),
            output_profile="production", resource_mode="split_process",
            max_cpu_workers=2, max_gpu_workers=1, force=True,
            allow_missing_models=False, allow_v2_fallback=False,
            feature_store_mode="off", realtime_cutoff_hour=15,
            require_target_actual=False, _attempt_id="controlled-smoke",
        )
        original = lp._run_unified_model_plan
        lp._run_unified_model_plan = _fake_plan
        try:
            result = lp.run_ledger_predict(args)
        finally:
            lp._run_unified_model_plan = original

        assert result["status"] in {"complete", "complete_with_warnings"}, result
        manifest = result
        assert manifest["serving_protocol"] == PROTOCOL
        assert manifest["feature_view"]["target_truth_mask"] is True
        assert manifest["dynamic_snapshot"]["protocol"] == PROTOCOL
        assert manifest["snapshot_id"]
        for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
            ledger = pd.read_parquet(root / "ledger" / task / "prediction" / "prediction_ledger.parquet")
            target = ledger[ledger["target_day"].eq(TARGET)]
            assert set(target["model_name"]) == set(models)
            assert len(target) == len(models) * 96
            assert set(target["serving_protocol"]) == {PROTOCOL}
            assert set(target["snapshot_id"]) == {manifest["snapshot_id"]}
        assert Path(manifest["dynamic_snapshot"]["values_path"]).exists()
        assert Path(manifest["dynamic_snapshot"]["manifest_path"]).exists()
        assert Path(manifest["dynamic_snapshot"]["values_path"]).parent.name.startswith("attempt_")

        # Even direct production API callers cannot re-enable FeatureStore on
        # top of the Dynamic-v1 shared FeatureView.
        forbidden = SimpleNamespace(**vars(args))
        forbidden._attempt_id = "feature-store-forbidden"
        forbidden.feature_store_mode = "raw"
        forbidden.data_path = str(model_store)
        forbidden.actual_data_path = str(authority)
        try:
            lp.run_ledger_predict(forbidden)
        except ValueError as exc:
            assert "FORMAL96_DYNAMIC_FEATURE_STORE_FORBIDDEN" in str(exc)
        else:
            raise AssertionError("formal Dynamic-v1 accepted FeatureStore override")
        logging.shutdown()
    print("check_dynamic_formal_smoke_96: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
