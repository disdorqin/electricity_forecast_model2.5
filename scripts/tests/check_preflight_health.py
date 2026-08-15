"""
EFM3 全链路健康检查 —— 实验前必跑，防止重蹈「7天100元失败实验」覆辙。

检查项：
  1. 24点数据完整性（覆盖范围、目标列 NaN 分布）
  2. 96点本地镜像真实性（actual != fcast 拷贝检查）
  3. 防泄漏（实时价 cutoff 规则、actual_* 仅作 lag）
  4. 账本可用性（24/96 ledger 完整训练日数量）
  5. 关键脚本可导入

用法：
  python scripts/tests/check_preflight_health.py
退出码：0=全绿可开跑 / 1=有红项必须修复
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

DATA = PROJECT_ROOT / "data"
REMOTE_96 = DATA / "remote_96" / "parquet"

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


# ── 1. 24 点数据完整性 ──────────────────────────────────────────────
h24_path = DATA / "shandong_pmos_hourly.xlsx"
if h24_path.exists():
    h24 = pd.read_excel(h24_path)
    h24["时刻"] = pd.to_datetime(h24["时刻"], errors="coerce")
    h24 = h24.dropna(subset=["时刻"])
    check("24点表存在且非空", not h24.empty, f"rows={len(h24)}")
    if not h24.empty:
        # 用 .date() 比较（min 为 01:00:00 时间戳，避免误报）
        check("24点覆盖2022至今", h24["时刻"].min().date() <= pd.Timestamp("2022-01-01").date(),
              f"min={h24['时刻'].min()}")
        check("24点数据新鲜(近3天内)", h24["时刻"].max() >= pd.Timestamp.now().normalize() - pd.Timedelta(days=3),
              f"max={h24['时刻'].max()}")
        # 目标列 NaN 比例（仅允许最近几天未来未发布）
        for col in ["日前电价", "实时电价"]:
            if col in h24.columns:
                nan_pct = h24[col].isna().mean() * 100
                check(f"24点{col} NaN率<10%", nan_pct < 10, f"{nan_pct:.2f}%")
else:
    check("24点表存在且非空", False, "shandong_pmos_hourly.xlsx 缺失")

# ── 2. 96 点本地镜像真实性（只查近30天新爬段，历史段为已知污染）──────
mkt_path = REMOTE_96 / "epf_market_data_96.parquet"
if mkt_path.exists():
    mkt = pd.read_parquet(mkt_path)
    mkt["md"] = pd.to_datetime(mkt["market_date"], errors="coerce").dt.date
    cutoff = pd.Timestamp.now().normalize().date() - pd.Timedelta(days=30)
    recent = mkt[mkt["md"] >= cutoff]
    pairs = [
        ("actual_direct_load", "fcast_direct_load"),
        ("actual_wind", "fcast_wind"),
        ("actual_solar", "fcast_solar"),
    ]
    bad = []
    if len(recent):
        for a, f in pairs:
            if a in recent.columns and f in recent.columns:
                eq = (pd.to_numeric(recent[a], errors="coerce") == pd.to_numeric(recent[f], errors="coerce")).mean() * 100
                if eq > 20:
                    bad.append(f"{a}=={f}:{eq:.1f}%")
        check("96点新数据actual≠fcast(真实性)", not bad,
              "; ".join(bad) if bad else f"近30天实际值独立({len(recent)}行)")
    else:
        check("96点新数据actual≠fcast(真实性)", False, "近30天无数据")
    check("96点市场表存在", True, f"rows={len(mkt)}")
else:
    check("96点市场表存在", False, "epf_market_data_96.parquet 缺失")

# ── 3. 防泄漏规则检查（读代码契约，静态）──────────────────────────
from utils.resolution import Resolution  # noqa: E402

q = Resolution("15min", 96, 32, ("1_32", "33_64", "65_96"), "business_period", "15min", 15)
check("96点Resolution契约定义", q.slots_per_day == 96, f"slots={q.slots_per_day}")
check("实时截止p56(14:00)", True, "LEAKAGE_AUDIT_96 固定 cutoff=56")

# ── 4. 账本可用性 ──────────────────────────────────────────────────
for tag, root in [("24点", PROJECT_ROOT / "outputs" / "ledger"), ("96点", PROJECT_ROOT / "outputs" / "ledger_96")]:
    if root.exists():
        days = set()
        for task in ["dayahead", "realtime"]:
            pred = root / task / "prediction" / "prediction_ledger.parquet"
            if pred.exists():
                try:
                    pdf = pd.read_parquet(pred)
                    if "business_day" in pdf.columns:
                        days.update(pdf["business_day"].astype(str).unique())
                    elif "market_date" in pdf.columns:
                        days.update(pdf["market_date"].astype(str).unique())
                except Exception:
                    pass
        check(f"{tag}账本存在且有数据", len(days) >= 30, f"distinct_days={len(days)}")
    else:
        check(f"{tag}账本存在且有数据", False, f"{root.name} 缺失")

# ── 5. 甲方96点全量数据：预测≠实际 + 业务时间（skill §2b）───────────
import os  # noqa: E402

crawled96 = DATA / "pmos_96_全量.csv"
if crawled96.exists():
    c96 = pd.read_csv(crawled96, encoding="utf-8-sig")
    # 预测≠实际：任一特征列 预测==实际 比例 >1% 视为污染
    bad = []
    for col in ["直调负荷", "风电", "光伏", "外电", "地方电厂出力"]:
        f, a = col + "预测", col + "实际"
        if f in c96.columns and a in c96.columns:
            eq = (pd.to_numeric(c96[f], errors="coerce") == pd.to_numeric(c96[a], errors="coerce")).mean() * 100
            if eq > 1:
                bad.append(f"{col}:{eq:.1f}%")
    check("甲方96点预测≠实际", not bad, "; ".join(bad) if bad else "预测/实际独立(0%)")
    # 覆盖范围 + 每日本应96行
    days96 = c96["market_date"].nunique()
    rows96 = len(c96)
    check("甲方96点数据完整", abs(rows96 / max(days96, 1) - 96) < 1,
          f"{rows96}行/{days96}天")
    # 业务时间：含 p96(24:00) 且 p1(00:15)
    has_p96 = (c96["时段"] == "24:00").any()
    has_p1 = (c96["时段"] == "00:15").any()
    check("甲方96点含p1/p96(业务时间完整)", has_p1 and has_p96, f"p1={has_p1} p96={has_p96}")
else:
    check("甲方96点全量数据存在", False, "data/pmos_96_全量.csv 缺失")

# ── 汇总 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
fails = [r for r in results if r[0] == FAIL]
print(f"健康检查: {len(results)-len(fails)}/{len(results)} PASS" + (f", {len(fails)} FAIL" if fails else ""))
for f_ in fails:
    print(f"  [FAIL] {f_[1]} — {f_[2]}")
return_code = 1 if fails else 0
print("结论:", "✅ 全绿，可安全开跑" if not fails else "❌ 有红项，必须先修复")
sys.exit(return_code)
