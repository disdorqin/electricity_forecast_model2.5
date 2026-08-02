"""
96 点账本分析：为调参 + 论文创新点挖掘提供证据。

读取 outputs/ledger_96 的预测账本 + actual 账本，按 (task, business_day, business_period)
对齐真实值，输出分层证据：

  A. 总体概览     —— 各模型整体 SMAPE/MAE/R2，跨模型排序
  B. 时段分析     —— 每模型 × 时段(1_32/33_64/65_96) 误差，找最弱时段
  C. 极端价格专题  —— 真实价 ≤ -50 / 尖峰(>400) 时各模型误差，支撑「尖峰修补」论文
  D. 逐日趋势     —— 每日误差，找异常日（节假日、突变日）
  E. 偏差方向     —— 每模型系统性高估/低估（有偏性）

用法（预热/回测完成后）:
  python scripts/analyze_96_ledger.py --ledger-root outputs/ledger_96
  可选: --top-k 5  每个专题只看最差的 N 个
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def smape(y_true, y_pred):
    y_true = np.maximum(np.asarray(y_true, dtype=float), 50.0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 50.0)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom == 0, 0.0, np.abs(y_pred - y_true) / denom)
    return float(np.mean(s))


def _metric_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in df.groupby("model_name"):
        yt = g["y_true"].values
        yp = g["y_pred"].values
        rows.append({
            "model": model,
            "n": len(g),
            "MAE": float(np.mean(np.abs(yp - yt))),
            "SMAPE": smape(yt, yp),
            "R2": float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)) if np.var(yt) > 0 else np.nan,
            "bias": float(np.mean(yp - yt)),  # >0 高估，<0 低估
            "corr": float(np.corrcoef(yt, yp)[0, 1]),
        })
    return pd.DataFrame(rows).sort_values("SMAPE")


def run(ledger_root: Path, top_k: int = 5) -> None:
    pred_path = ledger_root / "realtime" / "prediction" / "prediction_ledger.parquet"
    act_path = ledger_root / "realtime" / "actual" / "actual_ledger.parquet"
    if not pred_path.exists() or not act_path.exists():
        raise FileNotFoundError(
            f"需要账本: {pred_path} 和 {act_path}。请先跑完预热/回测。"
        )
    pred = pd.read_parquet(pred_path)
    act = pd.read_parquet(act_path)

    key = ["business_day", "business_period"]
    act = act[["business_day", "business_period", "y_true"]].drop_duplicates(subset=key)
    df = pred.merge(act, on=key, how="inner")
    df = df.dropna(subset=["y_pred", "y_true"])
    df = df[df["business_period"].between(1, 96)]

    if df.empty:
        print("无对齐数据（预测与 actual 无交集）")
        return

    print("=" * 70)
    print("96 点账本分析 | 对齐样本:", len(df),
          "| 天数:", df["business_day"].nunique())
    print("=" * 70)

    # A. 总体概览
    print("\n### A. 各模型总体误差（SMAPE 升序）")
    print(_metric_table(df).to_string(index=False))

    # B. 时段分析
    print("\n### B. 时段误差（时段 × 模型 SMAPE）")
    seg_map = {1: "1_32", 2: "33_64", 3: "65_96"}
    df["segment"] = df["business_period"].apply(lambda p: seg_map.get((p - 1) // 32 + 1, "?"))
    piv = df.pivot_table(index="segment", columns="model_name",
                         values="y_pred", aggfunc=lambda x: np.nan)  # placeholder
    sm_by_seg = df.groupby(["segment", "model_name"]).apply(
        lambda g: smape(g["y_true"], g["y_pred"]), include_groups=False
    ).reset_index(name="SMAPE")
    print(sm_by_seg.pivot(index="segment", columns="model_name", values="SMAPE")
          .round(4).to_string())

    # C. 极端价格专题（论文核心）
    print("\n### C. 极端价格误差（|真实价| 分桶，支撑尖峰修补模块）")
    df["price_bucket"] = pd.cut(
        df["y_true"],
        bins=[-np.inf, -50, 50, 200, 400, np.inf],
        labels=["极负(<-50)", "负/低(<50)", "正常(50-200)", "偏高(200-400)", "尖峰(>400)"],
    )
    sm_ext = df.groupby(["price_bucket", "model_name"], observed=True).apply(
        lambda g: smape(g["y_true"], g["y_pred"]), include_groups=False
    ).reset_index(name="SMAPE")
    sm_ext["n"] = df.groupby(["price_bucket", "model_name"], observed=True).size().values
    print(sm_ext.pivot(index="price_bucket", columns="model_name", values="SMAPE")
          .round(4).to_string())
    print("\n各价格桶样本数:")
    print(df.groupby("price_bucket", observed=True).size().to_string())
    # 极端桶最差模型
    extreme = df[df["y_true"].abs() > 400]
    if not extreme.empty:
        print(f"\n尖峰/极负样本共 {len(extreme)} 条，最差模型:")
        print(_metric_table(extreme).head(top_k).to_string(index=False))

    # D. 逐日趋势（找异常日）
    print(f"\n### D. 逐日误差（最差 {top_k} 天，可能为节假日/突变日）")
    daily = df.groupby("business_day").apply(
        lambda g: pd.Series({"MAE": np.mean(np.abs(g["y_pred"] - g["y_true"])),
                             "SMAPE": smape(g["y_true"], g["y_pred"]),
                             "n": len(g)}),
        include_groups=False,
    ).reset_index().sort_values("SMAPE", ascending=False)
    print(daily.head(top_k).to_string(index=False))

    # E. 偏差方向
    print("\n### E. 系统性偏差（bias>0 高估, <0 低估）")
    bias = df.groupby("model_name")["y_pred"].apply(
        lambda yp: float(np.mean(yp - df.loc[yp.index, "y_true"]))
    ).reset_index(name="bias")
    print(bias.sort_values("bias").to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", default="outputs/ledger_96")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(Path(args.ledger_root), args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
