"""Regression test: matrix construction must reuse raw.parquet."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_store import FeatureStore


def _fixture() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01 01:00:00", periods=96, freq="h")
    return pd.DataFrame({
        "时刻": ts,
        "日前电价": np.linspace(100, 195, len(ts)),
        "直调负荷预测值": 1000.0,
        "风电总加预测值": 100.0,
        "光伏总加预测值": 50.0,
        "联络线受电负荷预测值": 200.0,
    })


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "input.xlsx"
        _fixture().to_excel(source, index=False)
        calls = {"read_excel": 0}
        original = pd.read_excel

        def counted(*args, **kwargs):
            calls["read_excel"] += 1
            return original(*args, **kwargs)

        with patch.object(pd, "read_excel", side_effect=counted):
            first = FeatureStore("hourly", source=source, root=root / "cache").ensure()
            first_raw = first.load_raw()
            first.ensure_base()
            first_view = first.ensure_view("lightgbm", "dayahead")
            second = FeatureStore("hourly", source=source, root=root / "cache").ensure()
            second_raw = second.load_raw()

        assert calls["read_excel"] == 1, calls
        assert first.raw_path == second.raw_path
        assert first.matrix_path == second.matrix_path
        assert first.base_path.exists()
        assert first_view.exists()
        assert first_view.stat().st_size == first.base_path.stat().st_size
        assert len(first_raw) == len(second_raw) == 96
        assert first.matrix_path.exists()
        print("PASS: FeatureStore first build reads source once; subsequent matrix/raw loads hit parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
