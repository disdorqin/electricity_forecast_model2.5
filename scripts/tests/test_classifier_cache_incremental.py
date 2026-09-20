from __future__ import annotations

from pathlib import Path

import pandas as pd

from ExtremPriceClf.merge_model.core.range_runner import (
    ClassifierRangeSpec,
    _semantic_p1_prefix_match,
)
from utils.classifier_cache import ClassifierCacheSpec, classifier_cache_layout


FORECASTS = [
    "直调负荷预测值", "地方电厂总加预测值", "联络线受电负荷预测值",
    "风电总加预测值", "光伏总加预测值", "核电总加预测值",
    "自备机组总加预测值", "试验机组总加预测值", "竞价空间预测值",
    "新能源总加预测值",
]


def _normalized() -> pd.DataFrame:
    ts = pd.date_range("2026-08-10 00:00", "2026-08-15 23:00", freq="h")
    df = pd.DataFrame({"时刻": ts, "日前电价": 300.0, "实时电价": 280.0})
    for i, col in enumerate(FORECASTS):
        df[col] = 1000.0 + i + pd.Series(range(len(df)), dtype=float) * 0.01
    return df


def _p1() -> pd.DataFrame:
    return pd.DataFrame({
        "时刻": pd.date_range("2026-08-10 00:00", "2026-08-15 23:00", freq="h"),
        "p1_prob_OOF": 0.2,
        "prob_source": "predict",
    })


def test_semantic_prefix_allows_next_day_asof_label_visibility_change():
    spec = ClassifierRangeSpec(start_date="2026-08-16", end_date="2026-08-16", resolution="15min")
    previous = _normalized()
    current = previous.copy()
    # The prior cached day becomes fully realized on the next run. These labels
    # were not used to train the cached day's p1 model and must not invalidate it.
    previous.loc[previous["时刻"] >= pd.Timestamp("2026-08-15 16:00"), "实时电价"] = float("nan")
    ok, detail = _semantic_p1_prefix_match(previous, current, _p1(), spec)
    assert ok, detail
    assert detail["label_validated_until"] == "2026-08-13 23:00:00"


def test_semantic_prefix_rejects_forecast_or_training_label_mutation():
    spec = ClassifierRangeSpec(start_date="2026-08-16", end_date="2026-08-16", resolution="15min")
    reference = _normalized()

    changed_forecast = reference.copy()
    changed_forecast.loc[10, "风电总加预测值"] += 1.0
    ok, detail = _semantic_p1_prefix_match(reference, changed_forecast, _p1(), spec)
    assert not ok and "forecast prefix changed" in detail["reason"]

    changed_label = reference.copy()
    changed_label.loc[10, "实时电价"] += 1.0
    ok, detail = _semantic_p1_prefix_match(reference, changed_label, _p1(), spec)
    assert not ok and detail["reason"] == "historical target labels changed"


def test_cache_namespace_no_longer_depends_on_source_mtime_or_size(tmp_path: Path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    a.write_bytes(b"a")
    b.write_bytes(b"different-size")
    spec = ClassifierCacheSpec(
        resolution="15min",
        task="realtime",
        target_name="实时电价",
        price_threshold=-50.0,
        train_start="2022-01-01",
        stage2_train_start="2024-01-01",
        oof_cutoff="2024-12-31",
    )
    la = classifier_cache_layout(project_root=tmp_path, source=a, spec=spec, feature_store_root=tmp_path / "cache")
    lb = classifier_cache_layout(project_root=tmp_path, source=b, spec=spec, feature_store_root=tmp_path / "cache")
    assert la.root == lb.root
