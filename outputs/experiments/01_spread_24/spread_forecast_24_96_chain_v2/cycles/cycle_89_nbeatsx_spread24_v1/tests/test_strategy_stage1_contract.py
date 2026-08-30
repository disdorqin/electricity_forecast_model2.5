from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nbeatsx_spread.data.business_dataset import build_inference_sample  # noqa: E402
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.data.origin_index import build_origin_window  # noqa: E402
from nbeatsx_spread.data.strategy_dataset import build_strategy_split  # noqa: E402
from run_forecast_strategy_stage1 import STAGE_CANDIDATES, dev14_days, load_matrix  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = next(p for p in ROOT.parents if (p / "utils" / "resolution.py").exists()) / "data/24/canonical/shandong_pmos_hourly.csv"


def test_strategy_matrix_is_exactly_stage1_and_dev14() -> None:
    matrix = load_matrix()
    assert STAGE_CANDIDATES == ("C0_DIRECT_H34", "C1_GAP_DIRECT_D24", "C2A_BRIDGE_TF")
    assert matrix["development_panel"] == "DEV14"
    assert len(dev14_days()) == 14
    assert "C2B_BRIDGE_OOF_MATCHED" in matrix["forbidden_in_stage1"]
    assert "C3_DIRMO" in matrix["forbidden_in_stage1"]
    assert "CONFIRM21" in matrix["forbidden_in_stage1"]


def test_gap_and_bridge_views_have_frozen_shapes_and_alignment() -> None:
    source = CanonicalHourlySource.from_csv(DATA)
    c1_train, _, c1_split = build_strategy_split(source, "2026-06-01", strategy="C1_GAP_DIRECT_D24", stage="direct")
    c2s1_train, _, _ = build_strategy_split(source, "2026-06-01", strategy="C2A_BRIDGE_TF", stage="stage1")
    c2s2_train, _, c2_split = build_strategy_split(source, "2026-06-01", strategy="C2A_BRIDGE_TF", stage="stage2")
    assert (c1_train.horizon, c1_train.n_features) == (24, 9)
    assert (c2s1_train.horizon, c2s1_train.n_features) == (10, 9)
    assert (c2s2_train.horizon, c2s2_train.n_features) == (24, 19)
    window = build_origin_window("2026-06-01")
    assert c1_split["target_slice"] == {"start": 10, "stop": 34}
    assert c2_split["target_slice"] == {"start": 10, "stop": 34}
    # Canonical hourly rows label business h1 at 01:00; the strategy offset
    # is still D-day h1 after the ten D-1 bridge rows.
    assert window.forecast_timestamps[10].strftime("%Y-%m-%d %H:%M") == "2026-06-01 01:00"


def test_c2a_teacher_forcing_context_is_historical_only() -> None:
    source = CanonicalHourlySource.from_csv(DATA)
    train, _, split = build_strategy_split(source, "2026-06-01", strategy="C2A_BRIDGE_TF", stage="stage2")
    assert split["training_last_day"] <= "2026-05-30"
    item = train[0]
    assert item["x_backcast"].shape == (168, 19)
    assert item["x_future"].shape == (24, 19)
    assert np.isfinite(item["x_future"].numpy()).all()
    inference = build_inference_sample(source, "2026-06-01")
    assert not hasattr(inference, "y_future")
