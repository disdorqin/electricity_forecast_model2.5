#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极端价修正 V0-V4 对照实验 — 离线重放

在实验专区 outputs/experiments/classifier_ab/ 下跑，不污染生产链路。
用历史 fused_predictions.csv + 分类器 _clf.xlsx，重放 5 种修正公式，
用「真实极值标签」算事件级指标（M1-M4），回答：
  - 修正公式放哪、怎么搭配更好
  - 现状 y_fused<=100 门槛挡掉了多少真负价

用法：
  python scripts/experiments/classifier_ab/run_ab.py --days 2026-02-24,2026-02-25,2026-02-26
  python scripts/experiments/classifier_ab/run_ab.py --fused-dir <dir> --clf-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

EXP_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "classifier_ab"
RESULTS = EXP_ROOT / "results"
LOGS = EXP_ROOT / "logs"


# ── 修正公式（5 方案）───────────────────────────────────────────────
def _h(p: float, p_lo: float = 0.25, p_hi: float = 0.80) -> float:
    """分级软修正强度：p<=p_lo→0，p>=p_hi→1，中间线性。"""
    if p <= p_lo:
        return 0.0
    if p >= p_hi:
        return 1.0
    return (p - p_lo) / (p_hi - p_lo)


def scheme_v0(p: float, y_fused: float, theta: float = 0.55) -> float:
    """现状：p>=θ ∧ y_fused<=100 → -80。"""
    return -80.0 if (p >= theta and y_fused <= 100) else y_fused


def scheme_v1(p: float, y_fused: float, theta: float = 0.55) -> float:
    """硬前置：p>=θ → -80（无 y_fused 门槛）。"""
    return -80.0 if p >= theta else y_fused


def scheme_v2(p: float, y_fused: float, p_lo: float = 0.25, p_hi: float = 0.80) -> float:
    """软修正：y_fused + h(p)(-80-y_fused)，无门槛。"""
    return y_fused + _h(p, p_lo, p_hi) * (-80.0 - y_fused)


def scheme_v3(p: float, y_fused: float, p_lo: float = 0.25, p_hi: float = 0.80) -> float:
    """混合：软修正 + 高置信(p>=p_hi)直接触底。"""
    return scheme_v2(p, y_fused, p_lo, p_hi)


def scheme_v4(p: float, y_fused: float, min_m: float, theta: float = 0.55, p_lo: float = 0.25, p_hi: float = 0.80) -> float:
    """一致性：软修正 + p>=θ ∧ min模型输出<=30 → -80。"""
    base = scheme_v2(p, y_fused, p_lo, p_hi)
    return -80.0 if (p >= theta and min_m <= 30) else base


# ── 指标计算（事件级，按事件数加权）───────────────────────────────
def compute_metrics(df: pd.DataFrame, pred_col: str, label_col: str = "真实极值标签") -> dict:
    """事件级指标：M1 门槛挡住真事件率 / M2 分类器召回 / M3 事件SMAPE / M4 误报率。"""
    df = df.dropna(subset=["y_true", pred_col]).copy()
    if df.empty:
        return {}

    neg_event = df[label_col] == 1  # 真实极值（负价）事件
    corrected = df[pred_col] <= -70  # 修正到 -80 附近

    n_neg = int(neg_event.sum())
    # M2: 分类器召回 = p>=θ 且为真事件 / 真事件总数
    m2 = float((df[neg_event]["p"].fillna(0) >= 0.55).mean()) if n_neg else None
    # M1: 现状门槛挡住真事件率 = p>=θ 且 y_fused>100 且真事件 / 真事件
    m1 = float(((df[neg_event]["p"].fillna(0) >= 0.55) & (df[neg_event]["y_fused"] > 100)).mean()) if n_neg else None
    # M3: 事件加权 SMAPE（floor50，与交付口径一致）
    smape = float((abs(df[pred_col] - df["y_true"]) / ((abs(df[pred_col]) + abs(df["y_true"])) / 2 + 50)).mean())
    # M4: 误报率 = 被修正但非真事件 / 被修正总数
    n_corr = int(corrected.sum())
    m4 = float((df[corrected & ~neg_event].shape[0]) / n_corr) if n_corr else None

    return {
        "n_events": n_neg, "n_corrected": n_corr,
        "M1_gate_blocked_neg": m1, "M2_clf_recall": m2,
        "M3_event_smape": smape, "M4_false_pos_rate": m4,
    }


# ── 主流程 ───────────────────────────────────────────────────────────
def load_day(fused_dir: Path, clf_dir: Path, day: str) -> pd.DataFrame:
    """加载某天 fused 预测 + 分类器输出，合并为一行带 p/标签/真值。"""
    # fused 候选路径
    candidates = [
        fused_dir / day / "realtime" / "fuse" / "fused_predictions.csv",
        fused_dir / day / "realtime" / "compat_fusion" / "realtime" / "fused_predictions.csv",
        fused_dir / f"{day}.csv",
        fused_dir / day / "fused_predictions.csv",
    ]
    fused_path = next((p for p in candidates if p.exists()), None)
    if fused_path is None:
        raise FileNotFoundError(f"fused 未找到 for {day}")

    fused = pd.read_csv(fused_path)
    fused["ds"] = pd.to_datetime(fused["ds"], errors="coerce")

    # 分类器输出候选路径
    clf_candidates = [
        clf_dir / day / "realtime" / "compat_fusion" / "classifier" / f"{day}_{day}_clf.xlsx",
        clf_dir / day / "classifier" / f"{day}_{day}_clf.xlsx",
        clf_dir / f"{day}_{day}_clf.xlsx",
        clf_dir / day / f"{day}_{day}_clf.xlsx",
        clf_dir / f"{day}.xlsx",
    ]
    clf_path = next((p for p in clf_candidates if p.exists()), None)
    if clf_path is None:
        raise FileNotFoundError(f"clf 未找到 for {day}")

    clf = pd.read_excel(clf_path, engine="openpyxl")
    clf = clf.rename(columns={"时刻": "ds"})
    clf["ds"] = pd.to_datetime(clf["ds"], errors="coerce")

    # p = p1_prob（stage1 概率作为软修正用；final_pred 阈值来自它）
    p_col = "p1_prob" if "p1_prob" in clf.columns else ("final_prob" if "final_prob" in clf.columns else None)
    if p_col is None:
        raise ValueError(f"clf 无 p1_prob/final_prob: {list(clf.columns)}")

    label_col = "真实极值标签" if "真实极值标签" in clf.columns else "label"
    merged = fused.merge(clf[["ds", p_col, label_col]].rename(columns={p_col: "p", label_col: "真实极值标签"}),
                         on="ds", how="left")
    merged["p"] = pd.to_numeric(merged["p"], errors="coerce").fillna(0.0)
    merged["真实极值标签"] = pd.to_numeric(merged["真实极值标签"], errors="coerce").fillna(0)
    # 真实电价（用于 y_true；无则留空）
    if "实时电价" in clf.columns:
        clf_true = clf[["ds", "实时电价"]].copy()
        merged = merged.merge(clf_true, on="ds", how="left")
    if "y_true" not in merged.columns:
        merged["y_true"] = merged.get("实时电价", pd.NA)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="极端价修正 V0-V4 对照（离线重放）")
    parser.add_argument("--days", help="逗号分隔日期，如 2026-02-24,2026-02-25,2026-02-26")
    parser.add_argument("--fused-dir", default=str(PROJECT_ROOT / "fixtures" / "repro_bundle" / "sample_runs"))
    parser.add_argument("--clf-dir", default=str(PROJECT_ROOT / "fixtures" / "repro_bundle" / "sample_runs"))
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    days = [d.strip() for d in args.days.split(",") if d.strip()] if args.days else []
    if not days:
        # 默认用 sample_runs 下所有日期
        days = [p.name for p in Path(args.fused_dir).iterdir() if p.is_dir() and p.name[:4].isdigit()]

    fused_dir = Path(args.fused_dir)
    clf_dir = Path(args.clf_dir)

    all_rows = []
    for day in days:
        try:
            df = load_day(fused_dir, clf_dir, day)
            all_rows.append(df)
            print(f"加载 {day}: {len(df)} 行")
        except Exception as e:
            print(f"⚠ {day}: {e}")

    if not all_rows:
        print("❌ 无数据，检查 --fused-dir/--clf-dir")
        return 1

    full = pd.concat(all_rows, ignore_index=True)

    # 需要 y_true 列
    if "y_true" not in full or full["y_true"].isna().all():
        print("⚠ 无真实电价对照，只能算修正影响无法算指标")
        # 至少输出每个方案的修正结果
        full["y_true"] = full.get("真实电价", pd.NA)

    # 跑 5 方案
    schemes = {
        "V0_现状": lambda r: scheme_v0(r["p"], r["y_fused"]),
        "V1_硬前置": lambda r: scheme_v1(r["p"], r["y_fused"]),
        "V2_软修正": lambda r: scheme_v2(r["p"], r["y_fused"]),
        "V3_混合": lambda r: scheme_v3(r["p"], r["y_fused"]),
        "V4_一致性": lambda r: scheme_v4(r["p"], r["y_fused"], r.get("min_m", 999)),
    }

    summary = {}
    for name, fn in schemes.items():
        full[name] = full.apply(fn, axis=1)
        m = compute_metrics(full, name)
        if m:
            summary[name] = m
            print(f"\n[{name}] {json.dumps(m, ensure_ascii=False, indent=2)}")

    # 保存
    full.to_csv(RESULTS / "ab_full.csv", index=False, encoding="utf-8-sig")
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
