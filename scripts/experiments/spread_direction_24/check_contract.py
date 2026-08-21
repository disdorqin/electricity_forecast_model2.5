"""Fast contract checks for the isolated 24-point spread experiment."""

from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_spread_experiment import (  # noqa: E402
    SPREAD_COL,
    _business_columns,
    actual_for_day,
    build_asof_input,
    predict_safe_baseline,
    score_predictions,
    _resolve_models,
)
from scripts.experiments.spread_direction_24.analyze_30day import (  # noqa: E402
    prequential_fusions,
)


def _fixture() -> pd.DataFrame:
    n = 240
    ds = pd.date_range("2026-01-01 01:00:00", periods=n, freq="h")
    frame = pd.DataFrame(
        {
            "时刻": ds,
            "日前电价": np.arange(n, dtype=float) + 100,
            "实时电价": np.arange(n, dtype=float) + 101,
            "直调负荷预测值": 50000.0,
            "直调负荷实际值": 49900.0,
        }
    )
    frame = _business_columns(frame)
    frame[SPREAD_COL] = frame["实时电价"] - frame["日前电价"]
    return frame


def main() -> None:
    raw = _fixture()
    day = "2026-01-09"
    actual = actual_for_day(raw, day, "日前电价", "实时电价")
    assert len(actual) == 24
    assert actual["hour_business"].tolist() == list(range(1, 25))

    with tempfile.TemporaryDirectory() as tmp:
        path, audit = build_asof_input(raw, day, "日前电价", "实时电价", Path(tmp))
        masked = pd.read_parquet(path)
        masked = _business_columns(masked)
        target = masked[masked["_business_day"].eq(day)]
        cutoff = pd.Timestamp("2026-01-08 14:00:00")
        at_cutoff = masked[masked["时刻"].eq(cutoff)]
        after_cutoff = masked[masked["时刻"].gt(cutoff)]
        assert target[SPREAD_COL].isna().all()
        assert target["实时电价"].isna().all()
        assert target["日前电价"].notna().all()
        assert target["直调负荷实际值"].isna().all()
        assert at_cutoff["实时电价"].notna().all()
        assert at_cutoff["直调负荷实际值"].notna().all()
        assert after_cutoff[SPREAD_COL].isna().all()
        assert after_cutoff["实时电价"].isna().all()
        assert after_cutoff["直调负荷实际值"].isna().all()
        target_forecast = masked[masked["_business_day"].eq(day)]["直调负荷预测值"]
        future_forecast = masked[masked["_business_day"].gt(day)]["直调负荷预测值"]
        assert target_forecast.notna().all()
        assert future_forecast.isna().all()
        assert audit["target_rows"] == 24
        assert audit["post_cutoff_realtime_non_null"] == 0

    for baseline in ("spread_asof_lag", "spread_lag48", "spread_weekly", "spread_rolling_median"):
        baseline_pred = predict_safe_baseline(raw, actual, day, baseline)
        assert len(baseline_pred) == 24
        assert pd.to_datetime(baseline_pred["source_max_ds"]).max() <= pd.Timestamp(
            "2026-01-08 14:00:00"
        )
    try:
        _resolve_models("spread_lag24")
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("unsafe spread_lag24 must be rejected")

    pred = actual[["target_day", "ds", "hour_business", "period"]].copy()
    pred["model_name"] = "fixture"
    pred["prediction_mode"] = "direct_spread"
    pred["y_pred_spread"] = actual["y_true_spread"]
    joined, metrics = score_predictions(pred, actual)
    assert len(joined) == 24
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["n_direction_eligible"] == 24

    # A target day's truth cannot affect that same day's fusion decision.
    fusion_rows = []
    models = ["sgdfnet", "timemixer", "spread_lag24"]
    for day in pd.date_range("2026-01-01", periods=12, freq="D"):
        for hour in range(1, 25):
            truth = 1.0 if hour % 5 == 0 else -1.0
            for model_idx, model in enumerate(models):
                prediction = truth if (hour + model_idx) % 4 else -truth
                fusion_rows.append(
                    {
                        "target_day": day.strftime("%Y-%m-%d"),
                        "ds": day + pd.Timedelta(hours=hour),
                        "hour_business": hour,
                        "period": "1_8" if hour <= 8 else ("9_16" if hour <= 16 else "17_24"),
                        "model_name": model,
                        "y_pred_spread": prediction,
                        "y_true_spread": truth,
                    }
                )
    fusion_input = pd.DataFrame(fusion_rows)
    kwargs = dict(
        primary="sgdfnet",
        helper="timemixer",
        baseline="spread_lag24",
        warmup_days=10,
        min_gate_support=2,
        gate_margin=0.1,
    )
    fused_before = prequential_fusions(fusion_input, **kwargs)
    changed = fusion_input.copy()
    changed.loc[changed["target_day"].eq("2026-01-11"), "y_true_spread"] *= -1
    fused_after = prequential_fusions(changed, **kwargs)
    day11_before = fused_before[fused_before["target_day"].eq("2026-01-11")]
    day11_after = fused_after[fused_after["target_day"].eq("2026-01-11")]
    assert day11_before["y_pred_spread"].tolist() == day11_after["y_pred_spread"].tolist()
    assert fused_before["target_day"].min() == "2026-01-11"
    print("spread_direction_24 contract checks: PASS")


if __name__ == "__main__":
    main()
