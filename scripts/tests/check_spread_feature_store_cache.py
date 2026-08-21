"""Regression test for the shared hourly spread FeatureStore base."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_store import FeatureStore  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "hourly.parquet"
        frame = pd.DataFrame(
            {
                "时刻": pd.date_range("2026-01-01 01:00:00", periods=48, freq="h"),
                "日前电价": np.arange(48, dtype=float),
                "实时电价": np.arange(48, dtype=float) + 2.5,
            }
        )
        frame.to_parquet(source, index=False)

        first = FeatureStore("hourly", source=source, root=root / "cache")
        first_path = first.ensure_spread_base("日前电价", "实时电价")
        first_frame = pd.read_parquet(first_path)

        second = FeatureStore("hourly", source=source, root=root / "cache")
        second_path = second.ensure_spread_base("日前电价", "实时电价")
        second_frame = pd.read_parquet(second_path)

        assert first_path == second_path
        assert second.spread_cache_hit is True
        assert {"_business_day", "_business_period", "价差"}.issubset(first_frame.columns)
        assert np.allclose(first_frame["价差"].to_numpy(), 2.5)
        pd.testing.assert_frame_equal(first_frame, second_frame)

    print("PASS: shared hourly spread FeatureStore builds once and reuses the signed parquet base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
