from __future__ import annotations

from pathlib import Path

import pandas as pd

from RT916_SpikeFusionNet.pipeline import ModelPipeline, core


def _fake_rt_result() -> pd.DataFrame:
    ds = pd.date_range("2026-09-20 00:15:00", periods=96, freq="15min")
    return pd.DataFrame(
        {
            "时刻": ds,
            "预测实时电价": [100.0] * 96,
            "实时电价": [float("nan")] * 96,
        }
    )


def test_dynamic_rt916_uses_shallow_runtime_root(monkeypatch, tmp_path: Path):
    seen = {}

    def fake_joint(*, start_end_list, mod, asof_hour, dynamic_serving):
        seen["package_out_root"] = Path(core.PACKAGE_OUT_ROOT)
        seen["dynamic_serving"] = dynamic_serving
        return _fake_rt_result()

    monkeypatch.setattr(core, "run_joint_da_rt_daily_backtest", fake_joint)

    base = tmp_path / ("very_long_attempt_component_" * 4) / "models"
    result = ModelPipeline().predict_range(
        target="realtime",
        predict_date="2026-09-20",
        resolution="15min",
        data_path=str(tmp_path / "input.parquet"),
        output_root=str(base),
        dynamic_serving=True,
        production_mode=True,
        rt916_train_steps=24,
    )

    assert seen["dynamic_serving"] is True
    assert seen["package_out_root"] == base / "r9"
    assert result.output_path == base / "r9" / "predictions.csv"
    assert result.output_path.exists()
    assert len(result.frame) == 96
    assert core.CONFIG["TRAIN_STEPS"] == 24


def test_legacy_rt916_keeps_historical_runtime_layout(monkeypatch, tmp_path: Path):
    seen = {}

    def fake_joint(*, start_end_list, mod, asof_hour, dynamic_serving):
        seen["package_out_root"] = Path(core.PACKAGE_OUT_ROOT)
        return _fake_rt_result()

    monkeypatch.setattr(core, "run_joint_da_rt_daily_backtest", fake_joint)

    base = tmp_path / "models"
    result = ModelPipeline().predict_range(
        target="realtime",
        predict_date="2026-09-20",
        resolution="15min",
        data_path=str(tmp_path / "input.parquet"),
        output_root=str(base),
        dynamic_serving=False,
        production_mode=False,
        rt916_train_steps=24,
    )

    assert seen["package_out_root"] == base / "rt916" / "realtime" / "core"
    assert result.output_path == base / "rt916" / "realtime" / "predictions.csv"
