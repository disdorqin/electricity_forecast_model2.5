"""Lightweight contract checks for the reusable 24/96 classifier runner.

This test never trains a classifier and never touches production outputs.  It
checks the two resolution adapters and cache namespace isolation so it can run
in the CPU development environment and in server preflight.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ExtremPriceClf.merge_model.core.range_runner import (
    ClassifierRangeSpec,
    normalize_classifier_input,
)
from utils.classifier_cache import ClassifierCacheSpec, classifier_cache_layout


BASE_COLUMNS = {
    "日前电价": 100.0,
    "实时电价": 90.0,
    "地方电厂总加实际值": 10.0,
    "核电总加实际值": 10.0,
    "自备机组总加实际值": 10.0,
    "试验机组总加实际值": 10.0,
    "直调负荷实际值": 100.0,
    "联络线受电负荷实际值": 20.0,
    "风电总加实际值": 20.0,
    "光伏总加实际值": 10.0,
    "新能源总加实际值": 30.0,
    "竞价空间实际值": 50.0,
    "地方电厂总加预测值": 10.0,
    "核电总加预测值": 10.0,
    "自备机组总加预测值": 10.0,
    "试验机组总加预测值": 10.0,
    "直调负荷预测值": 100.0,
    "联络线受电负荷预测值": 20.0,
    "风电总加预测值": 20.0,
    "光伏总加预测值": 10.0,
    "新能源总加预测值": 30.0,
    "竞价空间预测值": 50.0,
}


def _fixture(freq: str, periods: int, start: str) -> pd.DataFrame:
    frame = pd.DataFrame({"时刻": pd.date_range(start, periods=periods, freq=freq)})
    for col, value in BASE_COLUMNS.items():
        frame[col] = value
    return frame


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="classifier_contract_") as tmp:
        root = Path(tmp)
        hourly = root / "hourly.parquet"
        quarter = root / "quarter.parquet"
        hourly_frame = _fixture("h", 48, "2026-01-01 01:00")
        hourly_frame = hourly_frame.drop(columns=["新能源总加预测值", "新能源总加实际值", "竞价空间预测值", "竞价空间实际值"])
        hourly_frame.to_parquet(hourly, index=False)
        _fixture("15min", 192, "2026-01-01 00:15").to_parquet(quarter, index=False)

        hourly_out = normalize_classifier_input(
            hourly,
            ClassifierRangeSpec(start_date="2026-01-02", end_date="2026-01-02", resolution="hourly"),
        )
        quarter_out = normalize_classifier_input(
            quarter,
            ClassifierRangeSpec(start_date="2026-01-02", end_date="2026-01-02", resolution="15min"),
        )
        assert len(hourly_out) == 48
        # Preserve the legacy 15-minute -> floor-hour aggregation semantics.
        assert len(quarter_out) == 49

        cache_root = root / "feature_store"
        common = dict(
            task="realtime",
            target_name="实时电价",
            price_threshold=-50.0,
            train_start="2022-01-01",
            stage2_train_start="2024-01-01",
            oof_cutoff="2024-12-31",
        )
        cache_24 = classifier_cache_layout(
            project_root=root,
            source=hourly,
            spec=ClassifierCacheSpec(resolution="hourly", **common),
            feature_store_root=cache_root,
        )
        cache_96 = classifier_cache_layout(
            project_root=root,
            source=quarter,
            spec=ClassifierCacheSpec(resolution="15min", **common),
            feature_store_root=cache_root,
        )
        assert cache_24.root != cache_96.root

    print("classifier range contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
