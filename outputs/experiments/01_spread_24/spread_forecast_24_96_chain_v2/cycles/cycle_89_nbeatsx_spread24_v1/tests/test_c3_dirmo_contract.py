from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nbeatsx_spread.data.business_dataset import build_inference_sample  # noqa: E402
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.data.dirmo_dataset import DIRMO_BLOCKS, build_dirmo_split, validate_dirmo_partition  # noqa: E402
from run_c3_dirmo_stage1 import dev14_days, load_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = next(path for path in ROOT.parents if (path / "utils" / "resolution.py").exists()) / "data/24/canonical/shandong_pmos_hourly.csv"


def test_c3_config_and_partition_are_frozen() -> None:
    config = load_config()
    validate_dirmo_partition()
    assert config["strategy"]["id"] == "C3_DIRMO_10_12_12"
    assert config["strategy"]["recursive_feedback"] is False
    assert [(block.start, block.stop) for block in DIRMO_BLOCKS] == [(0, 10), (10, 22), (22, 34)]
    assert len(dev14_days()) == 14
    assert "alternative_block_sizes" in config["forbidden"]
    assert "RecMO" in config["forbidden"]
    assert "CONFIRM21" in config["forbidden"]


def test_each_direct_block_uses_same_legal_origin_and_own_horizon() -> None:
    source = CanonicalHourlySource.from_csv(DATA)
    expected = {"B0_BRIDGE10": 10, "B1_DDAY_FIRST12": 12, "B2_DDAY_LAST12": 12}
    for block in DIRMO_BLOCKS:
        train, validation, split = build_dirmo_split(source, "2026-06-01", block.block_id)
        assert len(train) > 0 and len(validation) == 28
        assert train.horizon == expected[block.block_id]
        assert train.n_features == 9
        assert split["recursive_feedback"] is False
        assert split["training_last_day"] <= "2026-05-30"
        assert train[0]["x_backcast"].shape == (168, 9)
        assert train[0]["x_future"].shape == (expected[block.block_id], 9)


def test_dirmo_inference_has_no_target_and_no_feedback_channels() -> None:
    source = CanonicalHourlySource.from_csv(DATA)
    inference = build_inference_sample(source, "2026-06-01")
    assert not hasattr(inference, "y_future")
    assert inference.x_future.shape == (34, 9)
    assert np.isfinite(inference.x_future).all()
    assert all(block.start >= 0 and block.stop <= 34 for block in DIRMO_BLOCKS)
