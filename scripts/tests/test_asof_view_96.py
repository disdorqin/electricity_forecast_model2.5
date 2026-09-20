from __future__ import annotations

from pathlib import Path
import os
import time

import pandas as pd
import pytest

import utils.asof_view_96 as asofmod
from utils.asof_view_96 import FORECAST_COLUMNS, build_asof_view_96


def _full_frame() -> pd.DataFrame:
    rows = []
    for day in pd.to_datetime(["2026-08-13", "2026-08-14", "2026-08-15"]):
        for p in range(1, 97):
            ts = day + pd.Timedelta(minutes=15 * p)
            if p == 96:
                ts = day + pd.Timedelta(days=1)
            row = {
                "时刻": ts,
                "market_date": day,
                "period_no": p,
                "日前电价": 200.0 + p,
                "实时电价": 180.0 + p,
            }
            for col in FORECAST_COLUMNS:
                row[col] = 1000.0 + p
            for col in [
                "直调负荷实际值", "地方电厂总加实际值", "联络线受电负荷实际值",
                "风电总加实际值", "光伏总加实际值", "核电总加实际值",
                "自备机组总加实际值", "试验机组总加实际值",
                "竞价空间实际值", "新能源总加实际值",
            ]:
                row[col] = 900.0 + p
            rows.append(row)
    return pd.DataFrame(rows)


def test_one_shared_asof_view_masks_only_runtime_copy(tmp_path: Path):
    """Legacy fixed-cutoff helper regression; formal serving uses Dynamic-v1."""
    source = tmp_path / "full.parquet"
    output = tmp_path / "asof.parquet"
    original = _full_frame()
    original.to_parquet(source, index=False)

    _, audit = build_asof_view_96(
        source_path=source,
        target_day="2026-08-15",
        cutoff_hour=15,
        output_path=output,
    )
    masked = pd.read_parquet(output)
    source_after = pd.read_parquet(source)
    masked["market_date"] = pd.to_datetime(masked["market_date"])

    decision = masked[masked["market_date"].eq(pd.Timestamp("2026-08-14"))]
    target = masked[masked["market_date"].eq(pd.Timestamp("2026-08-15"))]
    actual_cols = [c for c in masked.columns if c.endswith("实际值")]

    assert audit["status"] == "PASS"
    assert audit["decision_day_rt_visible"] == 60
    assert len(decision) == 96 and len(target) == 96
    assert decision.loc[decision["时刻"] > pd.Timestamp("2026-08-14 15:00"), ["实时电价", *actual_cols]].notna().sum().sum() == 0
    assert target[["日前电价", "实时电价", *actual_cols]].notna().sum().sum() == 0
    assert all(target[c].notna().sum() == 96 for c in FORECAST_COLUMNS)
    pd.testing.assert_frame_equal(original.reset_index(drop=True), source_after.reset_index(drop=True))


def test_transient_asof_can_live_inside_attempt_owned_root(tmp_path: Path):
    attempt_asof = tmp_path / "attempt_20260816_test" / "asof"
    path = asofmod.transient_asof_path_96(
        "2026-08-16", runtime_root=attempt_asof
    )
    assert path == attempt_asof / "input.parquet"
    assert attempt_asof.exists()


def test_stale_asof_scratch_is_reclaimed_but_recent_file_is_kept(tmp_path: Path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    stale = runtime / "asof_96_20260815_111.parquet"
    recent = runtime / "asof_96_20260816_222.parquet"
    stale.write_bytes(b"old")
    recent.write_bytes(b"new")
    old_ts = time.time() - 8 * 3600
    os.utime(stale, (old_ts, old_ts))

    monkeypatch.setattr(asofmod, "RUNTIME_ROOT", runtime)
    removed = asofmod.cleanup_stale_asof_96(max_age_hours=6)

    assert removed == 1
    assert not stale.exists()
    assert recent.exists()


def test_asof_rejects_decision_day_rt_before_cutoff(tmp_path: Path):
    """Legacy compatibility assertion retained outside the formal façade."""
    source = tmp_path / "full.parquet"
    frame = _full_frame()
    decision_mask = (
        pd.to_datetime(frame["market_date"]).eq(pd.Timestamp("2026-08-14"))
        & frame["period_no"].eq(60)
    )
    frame.loc[decision_mask, "实时电价"] = None
    frame.to_parquet(source, index=False)

    with pytest.raises(RuntimeError, match="DECISION_DAY_RT_NOT_READY"):
        build_asof_view_96(
            source_path=source,
            target_day="2026-08-15",
            cutoff_hour=15,
        )


def test_asof_allows_partial_decision_actual_with_explicit_warning(tmp_path: Path):
    source = tmp_path / "full.parquet"
    frame = _full_frame()
    decision_mask = (
        pd.to_datetime(frame["market_date"]).eq(pd.Timestamp("2026-08-14"))
        & frame["period_no"].eq(60)
    )
    frame.loc[decision_mask, "直调负荷实际值"] = None
    frame.to_parquet(source, index=False)

    _, audit = build_asof_view_96(
        source_path=source,
        target_day="2026-08-15",
        cutoff_hour=15,
    )
    assert audit["status"] == "PASS"
    assert any("DECISION_DAY_ACTUAL_PARTIAL" in w for w in audit["warnings"])


def test_asof_rejects_incomplete_target_forecast(tmp_path: Path):
    source = tmp_path / "full.parquet"
    frame = _full_frame()
    target_mask = (
        pd.to_datetime(frame["market_date"]).eq(pd.Timestamp("2026-08-15"))
        & frame["period_no"].eq(1)
    )
    frame.loc[target_mask, FORECAST_COLUMNS[0]] = None
    frame.to_parquet(source, index=False)

    with pytest.raises(RuntimeError, match="TARGET_FORECAST_NOT_READY"):
        build_asof_view_96(
            source_path=source,
            target_day="2026-08-15",
            cutoff_hour=15,
        )
