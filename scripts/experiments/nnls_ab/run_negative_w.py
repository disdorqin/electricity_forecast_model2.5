#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
负权重融合实验 — 让强模型(sgdfnet)权重 >1，弱模型配负权重"纠错"
==========================================================
用户洞察：若 sgdfnet 一直最优，非负凸组合永远无法超越它。
允许负权重 = 线性回归系数（OLS/BLS），融合可外推超出凸包。

设计矩阵 X（每行=一个 15min 点，4 列=4 模型 y_pred），y = actual。
权重约束：sum(w)=1, w_i ∈ [lo, hi]（lo 可负，hi 可>1）。

搜索网格：
  lo ∈ {0, -0.2, -0.5, -1.0}
  hi ∈ {1.0, 1.5, 2.0, 3.0}
  是否正则：无 / L2 小正则
评估：融合 vs sgdfnet 单模型的超越率（目标>=70%），以及整体 loss。

用法：python scripts/experiments/nnls_ab/run_negative_w.py [--quick]
"""
import argparse, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fusion.learners.daily_ledger_gef import compute_daily_loss
from utils.resolution import resolve_resolution

PROJECT = Path(__file__).resolve().parents[3]
RES = resolve_resolution("15min")
MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
PRIOR = {m: (1.0 if m == "sgdfnet" else 0.0) for m in MODELS}


def load():
    pred = pd.read_parquet(PROJECT / "outputs/ledger_96/realtime/prediction/prediction_ledger.parquet")
    act = pd.read_parquet(PROJECT / "outputs/ledger_96/realtime/actual/actual_ledger.parquet")
    pred = pred[pred["task"] == "realtime"][["target_day", "business_period", "period", "model_name", "y_pred"]]
    act = act[act["task"] == "realtime"][["target_day", "business_period", "period", "y_true"]]
    return pred, act


def day_matrix(pred, act, day):
    """返回 {period: (y_true, {model: y_pred})}"""
    a = act[act["target_day"] == day]
    p = pred[pred["target_day"] == day]
    out = {}
    for per in RES.period_names:
        ap = a[a["period"] == per].sort_values("business_period")
        yt = ap["y_true"].values
        pm = {}
        for m in MODELS:
            mp = p[(p["model_name"] == m) & (p["period"] == per)].sort_values("business_period")
            if len(mp) == len(yt) and not mp["y_pred"].isna().any() and not np.isnan(yt).any():
                pm[m] = mp["y_pred"].values
        out[per] = (yt, pm)
    return out


def bls_weights(X, y, lo, hi, ridge=0.0):
    """有界最小二乘：min ||Xw-y||^2 + ridge*||w-w0||^2, s.t. sum(w)=1, lo<=w<=hi
    w0 用强模型先验（sgdfnet=1, 其余=0），ridge 把解拉向该先验防过拟合。"""
    n = X.shape[1]
    w0 = np.array([PRIOR[m] for m in MODELS])

    def obj(w):
        err = X @ w - y
        return float(np.sum(err ** 2)) + ridge * np.sum((w - w0) ** 2)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(lo, hi)] * n
    r = minimize(obj, np.ones(n) / n, method="SLSQP", bounds=bnds, constraints=cons,
                 options={"maxiter": 400, "ftol": 1e-10})
    if r.success and np.all(np.isfinite(r.x)):
        return r.x
    return None


def run_grid(pred, act, days, lo_grid, hi_grid, ridge_list, window=21, step=3):
    results = []
    n_days = len(days)
    for i in range(window, n_days, step):
        D = days[i]
        win = days[i - window:i]
        dm_D = day_matrix(pred, act, D)
        # 训练窗口 OOF per period
        per_win = {per: {"X": [], "y": []} for per in RES.period_names}
        for d in win:
            dm = day_matrix(pred, act, d)
            for per in RES.period_names:
                yt, pm = dm[per]
                if len(pm) != len(MODELS) or len(yt) == 0:
                    continue
                cols = [pm[m] for m in MODELS]
                if np.isnan(np.column_stack(cols)).any() or np.isnan(yt).any():
                    continue
                per_win[per]["X"].append(np.column_stack(cols))
                per_win[per]["y"].append(yt)
        # 评估每个配置
        for lo in lo_grid:
            for hi in hi_grid:
                for ridge in ridge_list:
                    for per in RES.period_names:
                        ent = per_win[per]
                        if len(ent["X"]) < 10:
                            continue
                        X = np.vstack(ent["X"]); y = np.concatenate(ent["y"])
                        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
                        w = bls_weights(Xs, y, lo, hi, ridge)
                        yt, pm = dm_D[per]
                        if len(yt) == 0 or len(pm) < 4:
                            continue
                        # 融合
                        if w is not None:
                            y_fus = np.sum([w[k] * pm[m] for k, m in enumerate(MODELS)], axis=0)
                            l_fus = compute_daily_loss(yt, y_fus, "composite")
                        else:
                            l_fus = np.nan
                        l_sgd = compute_daily_loss(yt, pm["sgdfnet"], "composite")
                        results.append({"day": D, "period": per, "lo": lo, "hi": hi,
                                        "ridge": ridge, "fuse": l_fus, "sgdfnet": l_sgd,
                                        "w": "" if w is None else ",".join(f"{v:.2f}" for v in w)})
    return pd.DataFrame(results)


def summarize(df):
    print(f"n单元={len(df)}")
    for (lo, hi, ridge), g in df.groupby(["lo", "hi", "ridge"]):
        g = g.dropna(subset=["fuse"])
        if len(g) == 0:
            continue
        win = (g["fuse"] < g["sgdfnet"]).mean()
        rel = ((g["sgdfnet"] - g["fuse"]) / g["sgdfnet"]).mean()
        print(f"  lo={lo:+.1f} hi={hi:.1f} ridge={ridge:.2f}: fuse={g['fuse'].mean():.2f} "
              f"sgdfnet={g['sgdfnet'].mean():.2f} | 超越sgdfnet={win:.1%} 相对提升={rel:+.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    pred, act = load()
    days = sorted(act["target_day"].unique())
    print(f"数据: {len(days)} 天")

    # 先看约束为 0 的 NNLS 基线 + 负权重扫描
    print("\n===== 负权重扫描 (窗口21d, step=3) =====")
    lo_grid = [0.0, -0.2, -0.5, -1.0]
    hi_grid = [1.0, 1.5, 2.0, 3.0]
    ridge_list = [0.0, 0.01, 0.1]
    df = run_grid(pred, act, days, lo_grid, hi_grid, ridge_list, window=21, step=3)
    summarize(df)
    out = PROJECT / "outputs/experiments/nnls_ab"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "negative_w_grid.csv", index=False)
    print(f"\n已存 negative_w_grid.csv | 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
