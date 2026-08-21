"""Cross-check authoritative 96-point actuals against the 24-point table.

The 96-point file is an actual-data authority only.  It is deliberately not
used as the price/model-input wide table.  Every common actual field is
checked after aggregating p1..p4, p5..p8, ... into hourly interval ends.

Usage::

    python scripts/tests/check_96_vs_24_actual.py
    python scripts/tests/check_96_vs_24_actual.py <96-csv> <24-xlsx>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from utils.data_layout import DATA  # noqa: E402


ACTUAL_MAP = {
    "直调负荷实际": "直调负荷实际值",
    "地方电厂出力实际": "地方电厂总加实际值",
    "外电实际": "联络线受电负荷实际值",
    "风电实际": "风电总加实际值",
    "光伏实际": "光伏总加实际值",
    "核电实际": "核电总加实际值",
    "自备电厂实际": "自备机组总加实际值",
    "试验机组实际": "试验机组总加实际值",
}


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"cannot decode {path}")


def load_96(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    required = {"market_date", "时段", *ACTUAL_MAP.keys()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"96点权威表缺列: {sorted(missing)}")
    df["business_day"] = pd.to_datetime(df["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    parts = df["时段"].astype(str).str.split(":", n=1, expand=True)
    minute = pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")
    # 00:15..01:00 -> hour 1, ..., 23:15..24:00 -> hour 24.
    df["hour_business"] = ((minute + 59) // 60).astype("Int64")
    df.loc[df["hour_business"] == 0, "hour_business"] = 24
    for col in ACTUAL_MAP:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_24(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    timestamp = pd.to_datetime(df["时刻"], errors="coerce")
    df["business_day"] = timestamp.dt.normalize()
    midnight = timestamp.dt.hour.eq(0) & timestamp.dt.minute.eq(0)
    df.loc[midnight, "business_day"] = df.loc[midnight, "business_day"] - pd.Timedelta(days=1)
    df["business_day"] = df["business_day"].dt.strftime("%Y-%m-%d")
    df["hour_business"] = timestamp.dt.hour.replace(0, 24)
    return df


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path96 = Path(argv[0]) if argv else DATA.authoritative_96_actual_csv
    path24 = Path(argv[1]) if len(argv) > 1 else DATA.hourly_xlsx
    if not path96.exists() or not path24.exists():
        print(f"❌ 数据不存在: 96={path96} 24={path24}")
        return 1

    print(f"读取权威 96 点实际: {path96}")
    df96 = load_96(path96)
    print(f"  行数: {len(df96)}, 日期: {df96['business_day'].min()} ~ {df96['business_day'].max()}")
    print(f"读取 24 点表: {path24}")
    df24 = load_24(path24)
    print(f"  行数: {len(df24)}, 日期: {df24['business_day'].min()} ~ {df24['business_day'].max()}")

    rows = []
    for c96, c24 in ACTUAL_MAP.items():
        if c24 not in df24.columns:
            continue
        agg = (
            df96.groupby(["business_day", "hour_business"], as_index=False)[c96]
            .mean()
            .rename(columns={c96: "actual_96"})
        )
        h24 = df24[["business_day", "hour_business", c24]].rename(columns={c24: "actual_24"})
        merged = agg.merge(h24, on=["business_day", "hour_business"], how="inner")
        merged["actual_96"] = pd.to_numeric(merged["actual_96"], errors="coerce")
        merged["actual_24"] = pd.to_numeric(merged["actual_24"], errors="coerce")
        merged = merged.dropna(subset=["actual_96", "actual_24"])
        if merged.empty:
            continue
        diff = (merged["actual_96"] - merged["actual_24"]).abs()
        if merged["actual_96"].nunique() <= 1 or merged["actual_24"].nunique() <= 1:
            corr = 1.0 if diff.max() == 0 else 0.0
        else:
            corr = merged["actual_96"].corr(merged["actual_24"])
        mape = (diff / merged["actual_24"].abs().replace(0, pd.NA)).mean() * 100
        rows.append({
            "actual_96_column": c96,
            "actual_24_column": c24,
            "matched_rows": int(len(merged)),
            "matched_days": int(merged["business_day"].nunique()),
            "mad": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "corr": None if pd.isna(corr) else float(corr),
            "mape_pct": None if pd.isna(mape) else float(mape),
        })

    if not rows:
        print("❌ 没有任何共同实际列完成匹配")
        return 1
    report = pd.DataFrame(rows)
    print("\n" + "=" * 88)
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 88)

    direct = report.loc[report["actual_96_column"] == "直调负荷实际"]
    if direct.empty:
        print("❌ 缺少直调负荷实际这一主判定列")
        return 1
    d = direct.iloc[0]
    ok = float(d["mad"]) < 500 and float(d["corr"]) > 0.99
    report_dir = DATA.sync_96_root
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass" if ok else "review",
        "authoritative_96": str(path96),
        "hourly_24": str(path24),
        "criteria": {"direct_load_mad_lt": 500, "direct_load_corr_gt": 0.99},
        "fields": rows,
    }
    (report_dir / "actual_crosscheck.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结论: {'✅ 通过 — 权威 96 点实际可作为交叉验证基准' if ok else '❌ 存疑 — 暂停接入生产链路'}")
    print(f"报告: {report_dir / 'actual_crosscheck.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
