"""
96点 vs 24点 数据对照验证 —— 甲方爬取数据搬回后运行

目标：验证甲方电脑爬的 96 点「实际值」与 24 点表实际值一致（相差不多）。
方法：
  1. 读甲方合并总表 output_96/pmos_96_全量.csv（直调负荷实际 等列）
  2. 按小时聚合（hour=ceil(period/4) 或 时刻向上取整）
  3. 与 24 点表 shandong_pmos_hourly.xlsx 的「直调负荷实际值」对比
  4. 输出差异统计（MAD/MAPE/相关系数/每日异常报告）

判定标准：
  - 直调负荷实际：96点小时均值 vs 24点实际值，MAD 应 < 500MW（同源）
  - 相关系数 > 0.99 视为一致

用法：
  python scripts/tests/check_96_vs_24_actual.py [96点CSV路径]
  （默认读 output_96/pmos_96_全量.csv；也可指定甲方拷贝回来的路径）
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

DATA = PROJECT_ROOT / "data"
H24 = DATA / "shandong_pmos_hourly.xlsx"
DEFAULT_96 = Path(r"D:\爬虫电网\output_96\pmos_96_全量.csv")
if not DEFAULT_96.exists():
    DEFAULT_96 = Path(r"output_96\pmos_96_全量.csv")


def load_96(path: Path) -> pd.DataFrame:
    """读甲方合并总表，返回 market_date/时段/直调负荷实际 等。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 时段 "00:15".."24:00" → 小时
    def _hour(pid):
        hh, mm = pid.split(":")
        h = int(hh) + (1 if int(mm) > 0 else 0)  # 区间末：00:15 属于 hour 1
        return 24 if h == 0 or (h == 25) else h
    df["hour_business"] = df["时段"].apply(_hour)
    df["market_date"] = pd.to_datetime(df["market_date"]).dt.date
    return df


def load_24() -> pd.DataFrame:
    df = pd.read_excel(H24)
    df["时刻"] = pd.to_datetime(df["时刻"], errors="coerce")
    df = df.dropna(subset=["时刻"])
    df["market_date"] = df["时刻"].dt.date
    # 24点表时刻即区间末：h24 = 次日00:00；hour_business = 1..24
    df["hour_business"] = df["时刻"].dt.hour.replace(0, 24)
    return df


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_96
    if not path.exists():
        print(f"❌ 96点总表不存在: {path}")
        print("用法: python scripts/tests/check_96_vs_24_actual.py [96点CSV路径]")
        return 1

    print(f"读取 96 点总表: {path}")
    df96 = load_96(path)
    if "直调负荷实际" not in df96.columns:
        print("❌ 96点表缺少 '直调负荷实际' 列")
        print("实际列:", [c for c in df96.columns if "实际" in c])
        return 1
    print(f"  96点行数: {len(df96)}, 日期: {df96['market_date'].min()} ~ {df96['market_date'].max()}")

    h24 = load_24()
    print(f"24点行数: {len(h24)}, 日期: {h24['market_date'].min()} ~ {h24['market_date'].max()}")

    # 96点按 (market_date, hour_business) 聚合实际值均值
    agg = df96.groupby(["market_date", "hour_business"])["直调负荷实际"].mean().reset_index()
    agg = agg.rename(columns={"直调负荷实际": "q96_actual"})

    # 24点实际值列（尝试多个候选列名）
    col24 = None
    for c in ["直调负荷实际值", "直调负荷实际", "系统负荷实际值"]:
        if c in h24.columns:
            col24 = c
            break
    if col24 is None:
        print("❌ 24点表缺少实际负荷列")
        return 1

    h24_sel = h24[["market_date", "hour_business", col24]].rename(columns={col24: "h24_actual"})

    merged = agg.merge(h24_sel, on=["market_date", "hour_business"], how="inner")
    merged = merged.dropna(subset=["q96_actual", "h24_actual"])
    print(f"\n成功匹配: {len(merged)} 行 ({merged['market_date'].nunique()} 天)")

    if merged.empty:
        print("❌ 无重叠日期，无法对照")
        return 1

    # 差异统计
    mad = (merged["q96_actual"] - merged["h24_actual"]).abs().mean()
    corr = merged["q96_actual"].corr(merged["h24_actual"])
    mape = ((merged["q96_actual"] - merged["h24_actual"]).abs() / merged["h24_actual"].abs().replace(0, pd.NA)).mean() * 100

    print("\n" + "=" * 60)
    print(f"MAD (平均绝对差): {mad:.2f} MW")
    print(f"相关系数: {corr:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("=" * 60)

    # 每日异常
    daily = merged.groupby("market_date").apply(
        lambda g: pd.Series({"mad": (g["q96_actual"] - g["h24_actual"]).abs().mean()}), include_groups=False
    )
    bad_days = daily[daily["mad"] > 500]
    if len(bad_days):
        print(f"\n⚠️ 差异>500MW 的天数: {len(bad_days)}")
        print(bad_days.head(10).to_string())
    else:
        print(f"\n✅ 全部 {len(daily)} 天差异 < 500MW，96点实际与24点实际一致")

    ok = corr > 0.99 and mad < 500
    print(f"\n结论: {'✅ 通过 — 96点实际值可信，可开跑实验' if ok else '❌ 存疑 — 需检查96点爬取正确性'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
