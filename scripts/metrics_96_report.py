"""
96 点回测指标报告：按 计算文档(docs/metrics_calculation.md) 口径汇总。

输出一张宽表 CSV（task 列区分 日前/实时）+ 文本摘要：

  横轴(行) = 各模型 + 融合模型
  纵轴(列) = 各指标 × 时段(1-8h / 9-16h / 17-24h / 总)
  时段映射(96点15min)：1-8h → business_period 1-32；9-16h → 33-64；17-24h → 65-96

指标（口径与 metrics_calculation.md 一致）：
  SMAPE   : floor50 裁剪后对称百分比误差，小数（0.123 = 12.3%）
  accuracy: 1 - SMAPE
  MAE / MSE / MAPE / R2
系统级（需 DA+RT 联合）：
  SCR     : 价差方向准确率，sgn(P_rt - P_da) == sgn(P̂_rt - P̂_da)
  度电套利: 基础版 + 改良版（单位元/兆瓦时）

数据源：
  outputs/ledger_96/{task}/prediction/prediction_ledger.parquet   —— 模型预测账本
  outputs/ledger_96/{task}/actual/actual_ledger.parquet            —— 实际账本
  outputs/runs_96/{date}/{task}/fuse/fused_predictions.csv         —— 融合预测（y_fused）

用法：
  python scripts/metrics_96_report.py --ledger-root outputs/ledger_96 --runs-root outputs/runs_96 \
      --out-csv outputs/metrics_96_report.csv
可选：
  --min-date YYYY-MM-DD  只统计该日期之后（含）
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------- 指标（口径与 docs/metrics_calculation.md 对齐） ----------------

def smape_floor50(y_true, y_pred) -> float:
    """SMAPE，floor50 裁剪，返回小数（0.123 = 12.3%）。"""
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


def scr(y_true_da, y_pred_da, y_true_rt, y_pred_rt) -> float:
    """价差方向准确率：sgn(P_rt-P_da) == sgn(P̂_rt-P̂_da) 的占比。"""
    real_spread = np.asarray(y_true_rt, float) - np.asarray(y_true_da, float)
    pred_spread = np.asarray(y_pred_rt, float) - np.asarray(y_pred_da, float)
    return float(np.mean(np.sign(real_spread) == np.sign(pred_spread)))


# ---------------- 时段 ----------------

def segment_map() -> dict[int, str]:
    """business_period → 时段标签。"""
    segs = {}
    for p in range(1, 97):
        if p <= 32:
            segs[p] = "1-8h"
        elif p <= 64:
            segs[p] = "9-16h"
        else:
            segs[p] = "17-24h"
    return segs


SEGMENTS = ["1-8h", "9-16h", "17-24h", "总"]


def metric_row(model: str, df: pd.DataFrame) -> dict:
    """对一个模型(或融合)的样本集，计算全部指标。df 必须含 y_true/y_pred。"""
    yt = df["y_true"].values
    yp = df["y_pred"].values
    n = len(df)
    return {
        "model": model,
        "n": n,
        "SMAPE": smape_floor50(yt, yp),
        "accuracy": 1.0 - smape_floor50(yt, yp),
        "MAE": mae(yt, yp),
        "MSE": mse(yt, yp),
        "MAPE": mape(yt, yp),
        "R2": r2(yt, yp),
    }


# ---------------- 主流程 ----------------

def load_task_data(ledger_root: Path, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读某任务(dayahead/realtime)的 预测账本 + 实际账本，按 (business_day, business_period) 对齐。"""
    pred = pd.read_parquet(ledger_root / task / "prediction" / "prediction_ledger.parquet")
    act = pd.read_parquet(ledger_root / task / "actual" / "actual_ledger.parquet")

    key = ["business_day", "business_period"]
    act = act[key + ["y_true"]].drop_duplicates(subset=key)
    df = pred.merge(act, on=key, how="inner")
    df = df.dropna(subset=["y_pred", "y_true"])
    df = df[df["business_period"].between(1, 96)]
    df = df[["model_name", "business_day", "business_period", "y_pred", "y_true"]]
    return df, act


def load_fusion_data(runs_root: Path, task: str, min_date: str | None) -> pd.DataFrame:
    """读该任务全部日期的融合预测，返回 (business_day, business_period, y_pred)。"""
    rows = []
    pat = str(runs_root / "????-??-??" / task / "fuse" / "fused_predictions.csv")
    for f in sorted(glob.glob(pat)):
        day = Path(f).parts[-3]
        if min_date and day < min_date:
            continue
        d = pd.read_csv(f)
        if "y_fused" in d.columns and "business_period" in d.columns and "business_day" in d.columns:
            rows.append(d[["business_day", "business_period", "y_fused"]].rename(columns={"y_fused": "y_pred"}))
    if not rows:
        return pd.DataFrame(columns=["business_day", "business_period", "y_pred"])
    return pd.concat(rows, ignore_index=True)


def run(ledger_root: Path, runs_root: Path, out_csv: Path, min_date: str | None) -> int:
    seg_map = segment_map()
    all_rows: list[dict] = []

    for task in ["dayahead", "realtime"]:
        df, act = load_task_data(ledger_root, task)
        df = df[df["business_day"] >= min_date] if min_date else df
        if df.empty:
            print(f"[warn] {task}: 无对齐数据")
            continue

        # 各模型
        for model, g in df.groupby("model_name"):
            for seg_name, gseg in split_by_segment(g, seg_map):
                all_rows.append({"task": task, "segment": seg_name, **metric_row(model, gseg)})

        # 融合模型
        fused = load_fusion_data(runs_root, task, min_date)
        if not fused.empty:
            fused = fused.merge(act, on=["business_day", "business_period"], how="inner")
            fused = fused.dropna(subset=["y_pred", "y_true"])
            if min_date:
                fused = fused[fused["business_day"] >= min_date]
            if not fused.empty:
                for seg_name, gseg in split_by_segment(fused, seg_map):
                    all_rows.append({"task": task, "segment": seg_name, **metric_row("融合模型", gseg)})

    # ---------------- 系统级（SCR + 度电套利）：需要 DA 与 RT 融合预测联合 ----------------
    sys_rows: list[dict] = []
    da_fused = load_fusion_data(runs_root, "dayahead", min_date)
    rt_fused = load_fusion_data(runs_root, "realtime", min_date)
    da_act = load_task_data(ledger_root, "dayahead")[1]
    rt_act = load_task_data(ledger_root, "realtime")[1]
    if not da_fused.empty and not rt_fused.empty:
        key = ["business_day", "business_period"]
        da = da_fused.merge(da_act, on=key, how="inner").rename(
            columns={"y_pred": "y_pred_da", "y_true": "y_true_da"})
        rt = rt_fused.merge(rt_act, on=key, how="inner").rename(
            columns={"y_pred": "y_pred_rt", "y_true": "y_true_rt"})
        sys_df = da.merge(rt, on=key, how="inner")
        sys_df = sys_df.dropna(subset=["y_pred_da", "y_pred_rt", "y_true_da", "y_true_rt"])
        if min_date:
            sys_df = sys_df[sys_df["business_day"] >= min_date]
        if not sys_df.empty:
            for seg_name, gseg in split_by_segment(sys_df, seg_map):
                row = {
                    "task": "系统级(日前+实时)", "segment": seg_name, "model": "融合系统",
                    "n": len(gseg),
                    "SCR": scr(gseg["y_true_da"], gseg["y_pred_da"], gseg["y_true_rt"], gseg["y_pred_rt"]),
                    "SMAPE": smape_floor50(gseg["y_true_da"], gseg["y_pred_da"]),
                    "accuracy": 1.0 - smape_floor50(gseg["y_true_da"], gseg["y_pred_da"]),
                    "MAE": mae(gseg["y_true_da"], gseg["y_pred_da"]),
                    "MSE": mse(gseg["y_true_da"], gseg["y_pred_da"]),
                    "MAPE": mape(gseg["y_true_da"], gseg["y_pred_da"]),
                    "R2": r2(gseg["y_true_da"], gseg["y_pred_da"]),
                }
                # 度电套利（基础 + 改良），按计算文档 §3.3/§3.4
                q_base = (gseg["y_pred_da"] > gseg["y_true_da"]).astype(int)
                q_imp = ((gseg["y_pred_rt"] > gseg["y_pred_da"]) & (gseg["y_pred_da"] > gseg["y_true_da"])).astype(int)
                spread = gseg["y_true_rt"] - gseg["y_true_da"]
                v_base = int(q_base.sum())
                v_imp = int(q_imp.sum())
                row["套利基础_度电"] = float((q_base * spread).sum() / v_base) if v_base else float("nan")
                row["套利改良_度电"] = float((q_imp * spread).sum() / v_imp) if v_imp else float("nan")
                row["套利基础_总利"] = float((q_base * spread).sum())
                row["套利改良_总利"] = float((q_imp * spread).sum())
                sys_rows.append(row)

    # ---------------- 汇总成宽表 ----------------
    # 每个 (task, model, segment) 一行 → pivot 成 (task, model) × 指标×时段
    frame = pd.DataFrame(all_rows + sys_rows)
    if frame.empty:
        print("无数据")
        return 1

    metrics = ["SMAPE", "accuracy", "MAE", "MSE", "MAPE", "R2"]
    if "SCR" in frame.columns:
        metrics += ["SCR"]
    arbitrage_cols = [c for c in frame.columns if c.startswith("套利")]

    # 顺序：总 → 1-8h → 9-16h → 17-24h
    order = ["总", "1-8h", "9-16h", "17-24h"]

    values = metrics + arbitrage_cols
    piv = frame.pivot_table(index=["task", "model"], columns="segment", values=values, aggfunc="first")
    cols = []
    for m in values:
        for seg in order:
            cols.append((m, seg))
    piv = piv.reindex(columns=cols)
    piv.columns = [f"{m}_{s}" for m, s in cols]
    piv = piv.reset_index()
    # 附加样本数 n_总
    n_total = frame[frame["segment"] == "总"].set_index(["task", "model"])["n"]
    piv = piv.merge(n_total.rename("n_总"), on=["task", "model"], how="left")
    # task 排序：dayahead → realtime → 系统级
    task_order = {"dayahead": 0, "realtime": 1, "系统级(日前+实时)": 2}
    piv["_t"] = piv["task"].map(task_order)
    piv = piv.sort_values(["_t", "model"]).drop(columns="_t").reset_index(drop=True)
    # 列排序：task, model, n_总, 各指标
    first_cols = ["task", "model", "n_总"]
    other = [c for c in piv.columns if c not in first_cols]
    piv = piv[first_cols + other]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    piv.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"CSV 已写出: {out_csv}")
    print(f"  行数={len(piv)}  任务={sorted(piv['task'].unique())}")
    return 0


def split_by_segment(df: pd.DataFrame, seg_map: dict) -> list[tuple[str, pd.DataFrame]]:
    """按时段拆分，返回 [(总, df), (1-8h, df), (9-16h, df), (17-24h, df)]。"""
    df = df.copy()
    df["_seg"] = df["business_period"].map(seg_map)
    out = [("总", df)]
    for seg in ["1-8h", "9-16h", "17-24h"]:
        out.append((seg, df[df["_seg"] == seg]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", default="outputs/ledger_96")
    parser.add_argument("--runs-root", default="outputs/runs_96")
    parser.add_argument("--out-csv", default="outputs/metrics_96_report.csv")
    parser.add_argument("--min-date", default=None, help="只统计该日期及之后（含），如 2026-01-01")
    args = parser.parse_args()
    return run(Path(args.ledger_root), Path(args.runs_root), Path(args.out_csv), args.min_date)


if __name__ == "__main__":
    raise SystemExit(main())
