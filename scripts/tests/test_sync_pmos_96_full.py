from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

import scripts.sync.sync_pmos_96_full as syncmod
import scripts.sync.build_96_model_input_from_authoritative as modelmod


def _frame(days: list[str], *, partial_latest: bool = True) -> pd.DataFrame:
    rows = []
    for day_index, day in enumerate(days):
        for p in range(1, 97):
            minutes = p * 15
            label = "24:00" if p == 96 else f"{minutes // 60:02d}:{minutes % 60:02d}"
            actual = float(1000 + p)
            rt = float(300 + p)
            if partial_latest and day_index == len(days) - 1 and p > 60:
                actual = None
                rt = None
            rows.append(
                {
                    "id": len(rows) + 1,
                    "market_date": day,
                    "时段": label,
                    "直调负荷预测": 50000.0,
                    "地方电厂出力预测": 6000.0,
                    "外电预测": 16000.0,
                    "风电预测": 7000.0,
                    "光伏预测": 3000.0,
                    "核电预测": 1200.0,
                    "自备电厂预测": 4000.0,
                    "试验机组预测": 0.0,
                    "全网负荷预测": 52000.0,
                    "直调负荷实际": actual,
                    "地方电厂出力实际": actual,
                    "外电实际": actual,
                    "风电实际": actual,
                    "光伏实际": actual,
                    "核电实际": actual,
                    "自备电厂实际": actual,
                    "试验机组实际": actual,
                    "抽蓄实际": 0.0,
                    "全网负荷实际": actual,
                    "直调负荷临时实际": actual,
                    "地方电厂出力临时实际": actual,
                    "外电临时实际": actual,
                    "风电临时实际": actual,
                    "光伏临时实际": actual,
                    "核电临时实际": actual,
                    "自备电厂临时实际": actual,
                    "试验机组临时实际": actual,
                    "抽蓄临时实际": 0.0,
                    "全网负荷临时实际": actual,
                    "边界全网负荷预测": 51500.0,
                    "边界直调负荷预测": 49500.0,
                    "边界外电预测": 15800.0,
                    "边界风电预测": 6800.0,
                    "边界光伏预测": 2900.0,
                    "边界核电预测": 1200.0,
                    "日前一次出清价格": 250.0,
                    "日前出清价格": 260.0,
                    "日前出力": 100.0,
                    "日前电量": 25.0,
                    "日前开机状态": "ON",
                    "日前电源类型": "TEST",
                    "实时出清价格": rt,
                    "实时出力": 100.0,
                    "实时电量": 25.0,
                    "实时开机状态": "ON",
                    "实时电源类型": "TEST",
                    "正备用预测": 1000.0,
                    "负备用预测": 1000.0,
                    "unit_id": "UNIT_A",
                    "source_captured_at": f"{day} 15:00:00",
                    "create_time": f"{day} 15:00:00",
                    "update_time": f"{day} 15:00:00",
                }
            )
    return pd.DataFrame(rows)


def test_full_sync_creates_fresh_checkout_paths_and_keeps_partial_tail(tmp_path, monkeypatch):
    remote = _frame(["2026-09-15", "2026-09-16"])
    remote_parquet = tmp_path / "data" / "96" / "remote" / "parquet" / "epf_pmos_96_full.parquet"
    remote_raw = tmp_path / "data" / "96" / "remote" / "raw" / "epf_pmos_96_full.csv.gz"
    authority = tmp_path / "data" / "96" / "authoritative" / "pmos_96_全量.csv"
    manifest = tmp_path / "outputs" / "96" / "sync" / "sync_manifest.json"

    monkeypatch.setattr(syncmod, "REMOTE_PARQUET", remote_parquet)
    monkeypatch.setattr(syncmod, "REMOTE_RAW", remote_raw)
    monkeypatch.setattr(syncmod, "AUTHORITATIVE_CSV", authority)
    monkeypatch.setattr(syncmod, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(syncmod, "ensure_data_directories", lambda: None)
    monkeypatch.setattr(syncmod, "fetch_96_table", lambda *args, **kwargs: remote.copy())
    monkeypatch.setattr(
        syncmod,
        "fetch_96_table_summary",
        lambda table: {"d_min": "2026-09-15", "d_max": "2026-09-16", "rows_total": len(remote)},
    )
    monkeypatch.setattr(syncmod, "get_db_server_version", lambda: "test-db")
    monkeypatch.setattr(syncmod, "_ensure_hourly_fallback_source", lambda: {"status": "existing", "output_xlsx": "test.xlsx"})
    model_dir = tmp_path / "data" / "96" / "model_input"
    full_model = model_dir / "shandong_pmos_96_model_input_full.parquet"
    monkeypatch.setattr(
        modelmod,
        "refresh_96_model_input_full",
        lambda **kwargs: {
            "status": "ok",
            "full_parquet": str(full_model),
            "latest_closed_day": "2026-09-15",
            "full_end_date": "2026-09-16",
            "mode": "incremental",
            "persistent_store_count": 1,
            "closed_view": "logical_only",
        },
    )

    args = Namespace(sync_source="db", sync_mode="full", sync_overlap_days=7, sync_unit_id=None)
    result = syncmod.sync_pmos_96_full(args)

    assert result["status"] == "ok"
    assert result["source_table"] == "epf_pmos_96_full"
    assert result["latest_closed_day"] == "2026-09-15"
    assert result["latest_contiguous_rt_period"] == 60
    assert authority.exists()
    assert remote_parquet.exists()
    assert remote_raw.exists()
    assert manifest.exists()

    local = pd.read_csv(authority, encoding="utf-8-sig")
    assert len(local) == 192
    assert list(local.columns) == syncmod.AUTHORITATIVE_COLUMNS
    latest = local[local["market_date"].astype(str) == "2026-09-16"]
    assert latest["实时出清价格"].notna().sum() == 60


def test_multiple_units_require_explicit_selection():
    frame = _frame(["2026-09-15"], partial_latest=False)
    other = frame.copy()
    other["unit_id"] = "UNIT_B"
    both = pd.concat([frame, other], ignore_index=True)
    try:
        syncmod._resolve_unit_id(both, None)
    except ValueError as exc:
        assert "multiple units" in str(exc)
    else:
        raise AssertionError("multiple units must require explicit selection")
