"""
96-point (15-min) FULL WIDE TABLE builder.

Merges the two synced 96-point mirror tables into ONE wide table that mirrors
the 24-point canonical dataset ``data/shandong_pmos_hourly.xlsx``:

  epf_market_data_96  (market features)  +  epf_unit_data_96  (unit prices)
          join on data_time  ->  data/shandong_pmos_96_full.xlsx(.csv)

Output columns
--------------
  时刻              data_time (15-min interval end)
  market_date       logical trade day
  period_no         1..96
  日前电价           unit-level day-ahead clearing price (da_cq_price, 元/MWh)
  实时电价           unit-level realtime clearing price (rt_cq_price, 元/MWh)
  日前出力/实时出力   unit output (MW)
  日前开机/实时开机   unit on/off status
  ... market feature columns (直调负荷实际/预测, 风电实际/预测, ...)

NOTE: the prices here are the UNIT clearing prices (the single configured
unit), NOT the provincial market-average prices found in the 24-point hourly
dataset. This is exactly the difference the user wants to verify.

Usage
-----
  # Build from the synced local mirror (parquet). Syncs first if missing.
  python build_96_full_table.py

  # Force a fresh DB sync before building
  python build_96_full_table.py --force-sync
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REMOTE_96_ROOT = DATA_DIR / "remote_96"
PARQUET_DIR = REMOTE_96_ROOT / "parquet"
MARKET_PQ = PARQUET_DIR / "epf_market_data_96.parquet"
UNIT_PQ = PARQUET_DIR / "epf_unit_data_96.parquet"
OUT_XLSX = DATA_DIR / "shandong_pmos_96_full.xlsx"
OUT_CSV = DATA_DIR / "shandong_pmos_96_full.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "data_sync_96"

sys.stdout.reconfigure(encoding="utf-8")


# 24-point hourly column names (for cross-reference in report only)
HOURLY_COLS = [
    "时刻", "日前电价", "实时电价",
    "地方电厂总加预测值", "联络线受电负荷预测值", "风电总加预测值", "光伏总加预测值",
    "核电总加预测值", "自备机组总加预测值", "试验机组总加预测值", "直调负荷预测值",
    "竞价空间预测值", "新能源总加预测值",
    "地方电厂总加实际值", "联络线受电负荷实际值", "风电总加实际值", "光伏总加实际值",
    "核电总加实际值", "自备机组总加实际值", "试验机组总加实际值", "直调负荷实际值",
    "竞价空间实际值", "新能源总加实际值",
]

# market feature column aliases — 统一用与 24 点 `shandong_pmos_hourly.xlsx`
# 相同的长列名（"直调负荷预测值"等），使各模型无需改列映射即可复用。
MARKET_ALIASES: dict[str, str] = {
    "actual_direct_load": "直调负荷实际值",
    "actual_local_plant": "地方电厂总加实际值",
    "actual_tie_line": "联络线受电负荷实际值",
    "actual_wind": "风电总加实际值",
    "actual_solar": "光伏总加实际值",
    "actual_nuclear": "核电总加实际值",
    "actual_self_owned": "自备机组总加实际值",
    "actual_test_unit": "试验机组总加实际值",
    "actual_unit_maintenance": "机组检修实际值",
    "actual_pos_reserve": "正备用实际值",
    "actual_neg_reserve": "负备用实际值",
    "actual_bidding_space": "竞价空间实际值",
    "actual_new_energy": "新能源总加实际值",
    "fcast_direct_load": "直调负荷预测值",
    "fcast_local_plant": "地方电厂总加预测值",
    "fcast_tie_line": "联络线受电负荷预测值",
    "fcast_wind": "风电总加预测值",
    "fcast_solar": "光伏总加预测值",
    "fcast_nuclear": "核电总加预测值",
    "fcast_self_owned": "自备机组总加预测值",
    "fcast_test_unit": "试验机组总加预测值",
    "fcast_unit_maintenance": "机组检修预测值",
    "fcast_pos_reserve": "正备用预测值",
    "fcast_neg_reserve": "负备用预测值",
    "fcast_bidding_space": "竞价空间预测值",
    "fcast_new_energy": "新能源总加预测值",
}


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing local mirror parquet: {path}")
    return pd.read_parquet(path)


def build_merged_frame() -> tuple[pd.DataFrame, dict]:
    """Merge unit prices + market features into one wide 96-point frame."""
    market = _load_parquet(MARKET_PQ)
    unit = _load_parquet(UNIT_PQ)

    # --- normalize dtypes ---
    for df in (market, unit):
        df["data_time"] = pd.to_datetime(df["data_time"], errors="coerce")
        df["market_date"] = pd.to_datetime(df["market_date"], errors="coerce").dt.date

    # --- unit ids present in the unit table ---
    unit_ids = sorted({str(x) for x in unit["unit_id"].dropna().unique()}) if "unit_id" in unit.columns else []

    # --- unit table: pick the price columns (single unit in current data) ---
    unit_cols = [
        "data_time", "market_date", "period_no",
        "da_cq_price", "rt_cq_price",
        "da_power", "rt_power", "da_status", "rt_status",
    ]
    unit = unit[unit_cols].copy()

    # --- market table: keep data_time + feature columns ---
    market_cols = ["data_time", "market_date", "period_no"] + list(MARKET_ALIASES.keys())
    market = market[market_cols].copy()

    # --- join on data_time (unique per market row; single unit) ---
    merged = market.merge(unit, on="data_time", how="left",
                          suffixes=("", "_unit"), validate="one_to_one")

    # --- build final column order + Chinese labels ---
    out = pd.DataFrame()
    out["时刻"] = merged["data_time"]
    out["market_date"] = pd.to_datetime(merged["market_date"])
    out["period_no"] = merged["period_no"]

    # unit price/power/status columns
    out["日前电价"] = merged["da_cq_price"]
    out["实时电价"] = merged["rt_cq_price"]
    out["日前出力"] = merged["da_power"]
    out["实时出力"] = merged["rt_power"]
    out["日前开机状态"] = merged["da_status"]
    out["实时开机状态"] = merged["rt_status"]

    # market feature columns (Chinese aliases)
    for remote_col, zh in MARKET_ALIASES.items():
        out[zh] = merged[remote_col]

    out = out.sort_values("时刻").reset_index(drop=True)

    meta = {
        "rows": int(len(out)),
        "min_ts": str(out["时刻"].min()),
        "max_ts": str(out["时刻"].max()),
        "distinct_days": int(out["market_date"].nunique()),
        "columns": list(out.columns),
        "unit_ids": unit_ids,
    }
    return out, meta


def _save_frame(df: pd.DataFrame, xlsx: Path, csv: Path) -> None:
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(xlsx, index=False)
    try:
        df.to_csv(csv, index=False, encoding="gbk")
    except Exception:
        df.to_csv(csv, index=False, encoding="utf-8-sig")


def _write_report(meta: dict, unit_ids: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    lines = [
        "# 96-Point Full Wide Table Report",
        "",
        f"- **Built at:** {ts}",
        f"- **Output XLSX:** `{OUT_XLSX}`",
        f"- **Output CSV:** `{OUT_CSV}`",
        f"- **Rows:** {meta['rows']}",
        f"- **Distinct trade days:** {meta['distinct_days']}",
        f"- **Min 时刻:** {meta['min_ts']}",
        f"- **Max 时刻:** {meta['max_ts']}",
        f"- **Unit IDs:** {', '.join(unit_ids) if unit_ids else 'N/A'}",
        "",
        "## 价格列来源（与 24 点的重要区别）",
        "",
        "- `日前电价` / `实时电价` 来自 `epf_unit_data_96`（机组级出清价 "
          "`da_cq_price` / `rt_cq_price`），是**单机组的出清价格**。",
        "- 24 点 hourly 数据集的 `日前电价` / `实时电价` 来自 `epf_market_data`"
          "（`price_dayahead` / `price_realtime`），是**全省市场出清均价**。",
        "",
        "## 列清单",
        "",
    ]
    lines.append("| 列名 | 来源 |")
    lines.append("|---|---|")
    for c in meta["columns"]:
        src = "unit price" if c in ("日前电价", "实时电价", "日前出力", "实时出力",
                                    "日前开机状态", "实时开机状态") else "market feature"
        lines.append(f"| {c} | {src} |")
    lines.append("")
    lines.append("---")
    lines.append("_Generated by build_96_full_table.py_")

    path = REPORT_DIR / "sync_96_full_table_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build 96-point full wide table")
    parser.add_argument("--force-sync", action="store_true", default=False,
                        help="Run a DB sync before building (re-pull latest)")
    args = parser.parse_args()

    if args.force_sync or not (MARKET_PQ.exists() and UNIT_PQ.exists()):
        print("Syncing 96-point data from DB first...")
        sys.path.insert(0, str(PROJECT_ROOT))
        from sync_data_96_core import sync_96
        manifest = sync_96(args=None)  # default full sync
        status = manifest.get("status")
        print(f"  sync_96 status: {status}")
        if status not in ("ok", "partial"):
            print(json.dumps(manifest.get("errors", []), ensure_ascii=False, indent=2))
            return 1

    df, meta = build_merged_frame()
    if df.empty:
        print("ERROR: merged frame is empty")
        return 1

    _save_frame(df, OUT_XLSX, OUT_CSV)
    _write_report(meta, meta["unit_ids"])
    print(f"OK: {len(df)} rows -> {OUT_XLSX}")
    print(f"     {len(df)} rows -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
