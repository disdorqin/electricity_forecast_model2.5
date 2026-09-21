#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24 点预测指标报告 —— 数据源: AI电力交易平台复盘爬虫数据集
    outputs/platform_review/电价预测复盘_详细数据.csv (218 天 x 24h)

口径与 docs/metrics_calculation.md 完全一致:
  SMAPE   : floor50 裁剪后对称百分比误差 (改良标准), 小数 (0.123 = 12.3%)
  accuracy: 1 - SMAPE
  MAE / MSE / MAPE / R2
时段划分 (24 点小时级): 1-8h / 9-16h / 17-24h

模型列 = 平台两个版本模型:
  1.0 模型   —— 上一代融合
  2.0 模型   —— 当前展示的融合 (我们展示的就是 2.0)

用法:
  python scripts/metrics_24_platform_report.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "outputs" / "platform_review" / "电价预测复盘_详细数据.csv"


# ---------------- 指标 (口径与 docs/metrics_calculation.md 对齐) ----------------

def smape_floor50(y_true, y_pred) -> float:
    yt = np.maximum(np.asarray(y_true, dtype=float), 50.0)
    yp = np.maximum(np.asarray(y_pred, dtype=float), 50.0)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom == 0, 0.0, np.abs(yp - yt) / denom)
    return float(np.mean(s))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def mse(y_true, y_pred) -> float:
    return float(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2))


def mape(y_true, y_pred) -> float:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(yt == 0, 0.0, np.abs(yp - yt) / np.abs(yt))
    return float(np.mean(s))


def r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def accuracy(y_true, y_pred) -> float:
    return 1.0 - smape_floor50(y_true, y_pred)


# ---------------- 时段划分 (24 点小时级) ----------------

def split_segments(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = [("总", df)]
    for seg, cond in [("1-8h", df["hour"] <= 8),
                      ("9-16h", df["hour"].between(9, 16)),
                      ("17-24h", df["hour"] >= 17)]:
        out.append((seg, df[cond]))
    return out


# ---------------- 主流程 ----------------

def main() -> int:
    if not CSV.exists():
        print(f"[错误] 数据集不存在: {CSV}")
        print("请先运行: python scripts/crawler/archive/legacy/platform_review_update.py")
        return 1

    df = pd.read_csv(CSV, dtype={"time": str})
    df["hour"] = df["小时"].astype(int)
    print(f"数据集: {len(df)} 行, 日期 {df['日期'].min()} ~ {df['日期'].max()}\n")

    tables = [
        ("日前 (DA)", "日前电价",
         ["1.0模型预测日前电价", "2.0模型预测日前电价"]),
        ("实时 (RT)", "实时电价",
         ["1.0模型预测实时电价", "2.0模型预测实时电价"]),
    ]

    for title, yt_col, pred_cols in tables:
        print(f"#### {title} —— {len(pred_cols)} 列")
        header = ["指标"] + [c.split("模型")[0] + "模型" for c in pred_cols]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * (len(pred_cols) + 1))

        # 每列模型的指标按 总/1-8h/9-16h/17-24h 组织
        per_model = {}
        for pc in pred_cols:
            d = df.dropna(subset=[yt_col, pc])
            per_model[pc] = {
                seg: (g[yt_col].values, g[pc].values)
                for seg, g in split_segments(d)
            }

        # 行: SMAPE 总 / 1-8h / 9-16h / 17-24h / 准确率总 / MAE总 / MSE总 / MAPE总 / R2总
        rows = []
        for label, fn, fmt in [
            ("SMAPE 总", lambda yt, yp: smape_floor50(yt, yp) * 100, "{:.2f}%"),
            ("SMAPE 1-8h", lambda yt, yp: smape_floor50(yt, yp) * 100, "{:.2f}%"),
            ("SMAPE 9-16h", lambda yt, yp: smape_floor50(yt, yp) * 100, "{:.2f}%"),
            ("SMAPE 17-24h", lambda yt, yp: smape_floor50(yt, yp) * 100, "{:.2f}%"),
            ("准确率 总", lambda yt, yp: accuracy(yt, yp) * 100, "{:.2f}%"),
            ("MAE 总", mae, "{:.1f}"),
            ("MSE 总", mse, "{:.0f}"),
            ("MAPE 总", lambda yt, yp: mape(yt, yp) * 100, "{:.2f}%"),
            ("R2 总", r2, "{:.3f}"),
        ]:
            seg = label.split(" ")[-1] if label != "准确率 总" else "总"
            vals = []
            for pc in pred_cols:
                yt, yp = per_model[pc][seg]
                if len(yt) == 0:
                    vals.append("-")
                else:
                    v = fn(yt, yp)
                    vals.append(fmt.format(v))
            print(f"| {label} | " + " | ".join(vals) + " |")

        # 样本量
        ns = []
        for pc in pred_cols:
            yt, _ = per_model[pc]["总"]
            ns.append(f"n={len(yt)}")
        print(f"| 样本量 | " + " | ".join(ns) + " |")
        print()

    # ---------------- 系统级指标 (DA+RT 联合, 融合系统 = 2.0 模型) ----------------
    print("#### 系统级指标 (DA+RT 联合, 融合系统 2.0)")
    sys_cols = ["融合系统 2.0"]
    print("| 指标 | " + " | ".join(sys_cols) + " |")
    print("|" + "---|" * (len(sys_cols) + 1))

    def sys_arbitrage(df: pd.DataFrame) -> dict[str, float]:
        """按 metrics_calculation.md §3.3/§3.4 计算度电套利(基础/改良)。"""
        pda = df["日前电价"].values
        prt = df["实时电价"].values
        pda_hat = df["2.0模型预测日前电价"].values
        prt_hat = df["2.0模型预测实时电价"].values
        q_base = (pda_hat > pda).astype(int)
        q_imp = ((prt_hat > pda_hat) & (pda_hat > pda)).astype(int)
        spread = prt - pda
        v_base = int(q_base.sum())
        v_imp = int(q_imp.sum())
        prof_base = float((q_base * spread).sum())
        prof_imp = float((q_imp * spread).sum())
        return {
            "套利基础_度电": prof_base / v_base if v_base else float("nan"),
            "套利改良_度电": prof_imp / v_imp if v_imp else float("nan"),
            "套利基础_总利": prof_base,
            "套利改良_总利": prof_imp,
            "量基础": v_base,
            "量改良": v_imp,
        }

    sysd = df.dropna(subset=["日前电价", "实时电价",
                             "2.0模型预测日前电价", "2.0模型预测实时电价"])
    print(f"样本: {len(sysd)} 行 (DA+RT 实际值与 2.0 融合预测均非空), "
          f"日期 {sysd['日期'].min()} ~ {sysd['日期'].max()}")

    segs = [("合计", sysd)] + [(s, sysd[cond]) for s, cond in [
        ("1-8h", sysd["hour"] <= 8),
        ("9-16h", sysd["hour"].between(9, 16)),
        ("17-24h", sysd["hour"] >= 17)]]

    rows = [
        ("度电套利·基础版(元/MWh)", "套利基础_度电", "{:.2f}"),
        ("度电套利·改良版(元/MWh)", "套利改良_度电", "{:.2f}"),
        ("总套利·基础版(元)", "套利基础_总利", "{:.0f}"),
        ("总套利·改良版(元)", "套利改良_总利", "{:.0f}"),
    ]
    header = ["指标"] + [s for s, _ in segs]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for label, key, fmt in rows:
        vals = []
        for s, g in segs:
            r = sys_arbitrage(g)
            v = r[key]
            vals.append(fmt.format(v) if v == v else "-")
        print(f"| {label} | " + " | ".join(vals) + " |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
