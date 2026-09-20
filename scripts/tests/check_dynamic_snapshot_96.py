"""D1/D2 synthetic contract checks; writes only to the OS temporary directory."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.asof_view_96 import (
    PRIMITIVE_ACTUAL_COLUMNS,
    PRIMITIVE_FORECAST_COLUMNS,
    build_dynamic_feature_view_96,
    build_dynamic_snapshot_96,
)
from pipelines.ledger_predict import FORMAL96_PREDICTION_CONTRACT, _validate_formal96_prediction_cache
from TimeMixer.repro_pipeline import restrict_dynamic_training_days
from RT916_SpikeFusionNet.src.rt916_spikefusionnet.core import _serving_training_end


def _fixture() -> pd.DataFrame:
    rows = []
    for day in pd.to_datetime(["2026-08-14", "2026-08-15"]):
        for period in range(1, 97):
            row = {
                "market_date": day,
                "period_no": period,
                "时刻": day + pd.Timedelta(minutes=15 * period),
                "日前电价": 200.0 + period,
                "实时电价": 180.0 + period,
            }
            for col in PRIMITIVE_FORECAST_COLUMNS:
                row[col] = 1000.0 + period
            for col in PRIMITIVE_ACTUAL_COLUMNS:
                row[col] = 900.0 + period
            rows.append(row)
    return pd.DataFrame(rows)


def _authority(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        **dict(zip(PRIMITIVE_FORECAST_COLUMNS, ["直调负荷预测", "地方电厂出力预测", "外电预测", "风电预测", "光伏预测", "核电预测", "自备电厂预测", "试验机组预测"])),
        **dict(zip(PRIMITIVE_ACTUAL_COLUMNS, ["直调负荷实际", "地方电厂出力实际", "外电实际", "风电实际", "光伏实际", "核电实际", "自备电厂实际", "试验机组实际"])),
        "日前电价": "日前出清价格",
        "实时电价": "实时出清价格",
    }
    return frame.rename(columns=mapping)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efm3_dynamic_snapshot_", dir=tempfile.gettempdir()) as tmp:
        root = Path(tmp)
        model = root / "model.parquet"
        authority = root / "authority.csv"
        frame = _fixture()
        frame.to_parquet(model, index=False)
        _authority(frame).to_csv(authority, index=False, encoding="utf-8-sig")

        first = build_dynamic_snapshot_96(
            model_store_path=model,
            authoritative_path=authority,
            target_day="2026-08-15",
            output_dir=root / "snapshot1",
        )
        second = build_dynamic_snapshot_96(
            model_store_path=model,
            authoritative_path=authority,
            target_day="2026-08-15",
            output_dir=root / "snapshot2",
        )
        assert first["snapshot_id"] == second["snapshot_id"]
        snap = pd.read_parquet(first["values_path"])
        assert len(snap) == 192

        # Dynamic training must stop before decision day D.  For target T=8/15,
        # the latest supervised sample day is 8/13 and RT916's latest training
        # timestamp is decision-day midnight (the p96 timestamp of 8/13).
        candidate_days = list(pd.date_range("2026-08-10", "2026-08-14", freq="D"))
        safe_days = restrict_dynamic_training_days(
            candidate_days, pd.Timestamp("2026-08-15"), True
        )
        assert max(safe_days) == pd.Timestamp("2026-08-13")
        assert _serving_training_end(
            pd.Timestamp("2026-08-15"), asof_hour=15, dynamic_serving=True
        ) == pd.Timestamp("2026-08-14 00:00:00")

        view, audit = build_dynamic_feature_view_96(
            model_store_path=model,
            snapshot_values=first["values_path"],
            snapshot_manifest=first["manifest"],
            target_day="2026-08-15",
            output_path=root / "feature_view.parquet",
        )
        assert audit["status"] == "PASS"
        assert audit["target_truth_mask"] is True
        target = view[view["market_date"].eq(pd.Timestamp("2026-08-15"))]
        assert target[["日前电价", "实时电价", *PRIMITIVE_ACTUAL_COLUMNS]].notna().sum().sum() == 0

        # D/T forecast facts must come from the frozen snapshot, not from a
        # model store that changes after snapshot capture.
        mutated_model = frame.copy()
        target_model_mask = mutated_model["market_date"].eq(pd.Timestamp("2026-08-15"))
        mutated_model.loc[target_model_mask, PRIMITIVE_FORECAST_COLUMNS[0]] = 888888.0
        mutated_model.to_parquet(model, index=False)
        frozen_view, _ = build_dynamic_feature_view_96(
            model_store_path=model,
            snapshot_values=first["values_path"],
            snapshot_manifest=first["manifest"],
            target_day="2026-08-15",
        )
        frozen_target = frozen_view[
            frozen_view["market_date"].eq(pd.Timestamp("2026-08-15"))
        ]
        assert (frozen_target[PRIMITIVE_FORECAST_COLUMNS[0]] != 888888.0).all()
        # Restore the base store for the remaining route checks.
        frame.to_parquet(model, index=False)

        # Target-day truth mutation must not change the routed model view.
        mutated = snap.copy()
        target_mask = mutated["market_date"].eq(pd.Timestamp("2026-08-15"))
        mutated.loc[target_mask, PRIMITIVE_ACTUAL_COLUMNS[0]] = 999999.0
        view2, _ = build_dynamic_feature_view_96(
            model_store_path=model,
            snapshot_values=mutated,
            snapshot_manifest=first["manifest"],
            target_day="2026-08-15",
        )
        assert view[PRIMITIVE_ACTUAL_COLUMNS[0]].equals(view2[PRIMITIVE_ACTUAL_COLUMNS[0]])

        # Cell-level route contract: p1 uses tmp, p2 uses same-field
        # ForecastData, p41 remains a hole-local fallback (p42 is untouched).
        authority_gap = _authority(frame)
        authority_gap.loc[(authority_gap["market_date"] == "2026-08-14") & (authority_gap["period_no"] == 1), "直调负荷实际"] = None
        authority_gap.loc[(authority_gap["market_date"] == "2026-08-14") & (authority_gap["period_no"] == 1), "直调负荷临时实际"] = 777.0
        authority_gap.loc[(authority_gap["market_date"] == "2026-08-14") & (authority_gap["period_no"] == 41), "直调负荷实际"] = None
        gap_path = root / "authority_gap.csv"
        authority_gap.to_csv(gap_path, index=False, encoding="utf-8-sig")
        gap_snapshot = build_dynamic_snapshot_96(
            model_store_path=model,
            authoritative_path=gap_path,
            target_day="2026-08-15",
            output_dir=root / "snapshot_gap",
        )

        # A complete outage of target ForecastData must not be hidden by the
        # model store / 24-point forecast fallback.
        missing_forecast = _authority(frame)
        target_missing_mask = missing_forecast["market_date"].eq(
            pd.Timestamp("2026-08-15")
        )
        for raw_col in [
            "直调负荷预测", "地方电厂出力预测", "外电预测", "风电预测",
            "光伏预测", "核电预测", "自备电厂预测", "试验机组预测",
        ]:
            missing_forecast.loc[target_missing_mask, raw_col] = None
        missing_forecast_path = root / "authority_missing_forecast.csv"
        missing_forecast.to_csv(
            missing_forecast_path, index=False, encoding="utf-8-sig"
        )
        try:
            build_dynamic_snapshot_96(
                model_store_path=model,
                authoritative_path=missing_forecast_path,
                target_day="2026-08-15",
                output_dir=root / "snapshot_missing_forecast",
            )
        except ValueError as exc:
            assert "MISSING_CRITICAL_SOURCE source=target_day_forecast" in str(exc)
        else:
            raise AssertionError("target ForecastData outage was hidden by fallback")

        # Decision-day DA is a critical source and must be present in the
        # synchronized authoritative state, not recovered from old model data.
        missing_da = _authority(frame)
        decision_missing_mask = missing_da["market_date"].eq(
            pd.Timestamp("2026-08-14")
        )
        missing_da.loc[decision_missing_mask, "日前出清价格"] = None
        missing_da_path = root / "authority_missing_da.csv"
        missing_da.to_csv(missing_da_path, index=False, encoding="utf-8-sig")
        try:
            build_dynamic_snapshot_96(
                model_store_path=model,
                authoritative_path=missing_da_path,
                target_day="2026-08-15",
                output_dir=root / "snapshot_missing_da",
            )
        except ValueError as exc:
            assert "MISSING_CRITICAL_SOURCE source=decision_day_day_ahead" in str(exc)
        else:
            raise AssertionError("decision-day DA outage was hidden by fallback")
        gap_view, gap_audit = build_dynamic_feature_view_96(
            model_store_path=model,
            snapshot_values=gap_snapshot["values_path"],
            snapshot_manifest=gap_snapshot["manifest"],
            target_day="2026-08-15",
        )
        decision = gap_view[gap_view["market_date"].eq(pd.Timestamp("2026-08-14"))].set_index("period_no")
        assert decision.loc[1, PRIMITIVE_ACTUAL_COLUMNS[0]] == 777.0
        assert decision.loc[41, PRIMITIVE_ACTUAL_COLUMNS[0]] != decision.loc[42, PRIMITIVE_ACTUAL_COLUMNS[0]]
        assert gap_audit["routes"][PRIMITIVE_ACTUAL_COLUMNS[0]]["tmp_cells"] >= 1

        cache_rows = []
        for period in range(1, 97):
            cache_rows.append({
                "task": "realtime", "model_name": "rt916", "target_day": "2026-08-15",
                "business_day": "2026-08-15", "business_period": period,
                "ds": pd.Timestamp("2026-08-15") + pd.Timedelta(minutes=15 * period),
                "hour_business": (period - 1) // 4 + 1, "period": "1_32" if period <= 32 else ("33_64" if period <= 64 else "65_96"),
                "y_pred": float(period), "da_feature_source": "rt916_internal_joint_dayahead_prediction",
                "production_contract": FORMAL96_PREDICTION_CONTRACT,
                "serving_protocol": FORMAL96_PREDICTION_CONTRACT,
                "production_resolution": "15min", "production_resource_mode": "split_process",
                "snapshot_id": first["snapshot_id"], "production_rt916_train_steps": 24,
            })
        cache = pd.DataFrame(cache_rows)
        assert _validate_formal96_prediction_cache(
            cache, target_date="2026-08-15", model_name="rt916", task="realtime",
            expected_cutoff="legacy-ignored", snapshot_id=first["snapshot_id"],
        ) == []
        stale = cache.copy(); stale["snapshot_id"] = "changed"
        assert any("snapshot_id" in e for e in _validate_formal96_prediction_cache(
            stale, target_date="2026-08-15", model_name="rt916", task="realtime",
            expected_cutoff="legacy-ignored", snapshot_id=first["snapshot_id"],
        ))
        old = cache.copy()
        old["production_contract"] = "formal96_prediction_v1"
        old["serving_protocol"] = "formal96_prediction_v1"
        assert any("production_contract" in e for e in _validate_formal96_prediction_cache(
            old, target_date="2026-08-15", model_name="rt916", task="realtime",
            expected_cutoff="legacy-ignored", snapshot_id=first["snapshot_id"],
        ))
    print("check_dynamic_snapshot_96: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
