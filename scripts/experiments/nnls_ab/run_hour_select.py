#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小时级模型选择/条件权重实验
==========================
实证发现：sgdfnet 在 96 点 RT 上整体最强，但 10-14 时（光伏大发段）timesfm 更强。
假设：按小时块独立学权重/选模型，能显著提升"超越 sgdfnet 单模型"的比例。

策略对比（每 (day, hour) 单元）：
  1. equal       等权融合
  2. nnls_hour   该小时 21 天 OOF 学 NNLS 权重
  3. best_hour   该小时 21 天历史最强单模型（简单选择器）
  4. oracle      当天该小时最优单模型（理论下界）
  vs sgdfnet 单模型（目标超越率>=70%）

用法：python scripts/experiments/nnls_ab/run_hour_select.py [--quick]
"""
import argparse, sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import nnls as scipy_nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fusion.learners.daily_ledger_gef import compute_daily_loss
from utils.resolution import resolve_resolution

PROJECT = Path(__file__).resolve().parents[3]
RES = resolve_resolution("15min")
MODELS = None  # set in main by task


def load(task="realtime"):
    pred = pd.read_parquet(PROJECT / f"outputs/ledger_96/{task}/prediction/prediction_ledger.parquet")
    act = pd.read_parquet(PROJECT / f"outputs/ledger_96/{task}/actual/actual_ledger.parquet")
    pred = pred[pred["task"] == task][["target_day", "business_period", "hour_business", "model_name", "y_pred"]]
    act = act[act["task"] == task][["target_day", "business_period", "hour_business", "y_true"]]
    return pred, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--task", default="realtime", choices=["realtime", "dayahead"])
    args = ap.parse_args()
    t0 = time.time()
    pred, act = load(args.task)
    days = sorted(act["target_day"].unique())
    step = 5 if args.quick else 2
    global MODELS
    MODELS = sorted(pred["model_name"].unique())
    print(f"数据: {len(days)} 天 ({days[0]}~{days[-1]}) | step={step}")

    rows = []
    n_days = len(days)
    for i in range(21, n_days, step):
        D = days[i]
        win = days[i - 21:i]
        # 训练窗按小时统计: {hour: {model: mean loss}}
        hist = {}
        for h in range(1, 25):
            lv = {m: [] for m in MODELS}
            for d in win:
                a = act[(act["target_day"] == d) & (act["hour_business"] == h)]
                for m in MODELS:
                    mp = pred[(pred["target_day"] == d) & (pred["hour_business"] == h) & (pred["model_name"] == m)].sort_values("business_period")
                    if len(mp) == len(a) and len(a) > 0 and not mp["y_pred"].isna().any() and not a["y_true"].isna().any():
                        lv[m].append(compute_daily_loss(a["y_true"].values, mp["y_pred"].values, "composite"))
            if all(len(v) >= 10 for v in lv.values()):
                hist[h] = {m: float(np.mean(v)) for m, v in lv.items()}
        # 目标日评估
        for h in range(1, 25):
            aD = act[(act["target_day"] == D) & (act["hour_business"] == h)].sort_values("business_period")
            yt = aD["y_true"].values
            if len(yt) == 0:
                continue
            pm = {}
            for m in MODELS:
                mp = pred[(pred["target_day"] == D) & (pred["hour_business"] == h) & (pred["model_name"] == m)].sort_values("business_period")
                if len(mp) == len(yt) and not mp["y_pred"].isna().any():
                    pm[m] = mp["y_pred"].values
            if len(pm) < len(MODELS) or np.isnan(yt).any():
                continue
            # 等权
            y_eq = np.mean([pm[m] for m in MODELS], axis=0)
            # NNLS 21d（该小时）
            l_nnls = np.nan
            if h in hist:
                Xs, ys = [], []
                for d in win:
                    a = act[(act["target_day"] == d) & (act["hour_business"] == h)]
                    cols = []
                    for m in MODELS:
                        mp = pred[(pred["target_day"] == d) & (pred["hour_business"] == h) & (pred["model_name"] == m)].sort_values("business_period")
                        if len(mp) == len(a) and len(a) > 0 and not mp["y_pred"].isna().any() and not a["y_true"].isna().any():
                            cols.append(mp["y_pred"].values)
                    if len(cols) == len(MODELS):
                        Xs.append(np.column_stack(cols)); ys.append(a["y_true"].values)
                if len(Xs) >= 10:
                    X = np.vstack(Xs); y = np.concatenate(ys)
                    Xs2 = (X - X.mean(0)) / (X.std(0) + 1e-8)
                    sol, _ = scipy_nnls(Xs2, y)
                    s = sol.sum()
                    if s > 1e-9:
                        w = sol / s
                        y_fus = np.sum([w[k] * pm[m] for k, m in enumerate(MODELS)], axis=0)
                        l_nnls = compute_daily_loss(yt, y_fus, "composite")
            # best_hour：该小时历史最强单模型
            l_besth = np.nan
            if h in hist:
                bm = min(hist[h], key=hist[h].get)
                l_besth = compute_daily_loss(yt, pm[bm], "composite")
            # oracle
            l_orc = min(compute_daily_loss(yt, pm[m], "composite") for m in pm)
            # best_hist：该单元历史最强单模型（可达到的"最优单模型"基准）
            l_besthist = np.nan
            if h in hist:
                bm = min(hist[h], key=hist[h].get)
                l_besthist = compute_daily_loss(yt, pm[bm], "composite")
            rows.append({"day": D, "hour": h, "best_hist": l_besthist, "oracle": l_orc,
                         "equal": compute_daily_loss(yt, y_eq, "composite"),
                         "nnls_hour": l_nnls, "best_hour": l_besth})

    r = pd.DataFrame(rows)
    print(f"n单元: {len(r)}")
    for col in ["equal", "nnls_hour", "best_hour"]:
        sub = r.dropna(subset=[col, "best_hist"])
        if len(sub) == 0:
            continue
        win = (sub[col] < sub["best_hist"]).mean()
        rel = ((sub["best_hist"] - sub[col]) / sub["best_hist"]).mean()
        print(f"{col:12s}: loss={sub[col].mean():.2f} best_hist={sub['best_hist'].mean():.2f} | "
              f"超越历史最优单模型={win:.1%} 相对提升={rel:+.1%} (n={len(sub)})")
    sub = r.dropna(subset=["oracle"])
    print(f"{'oracle':12s}: loss={sub['oracle'].mean():.2f} | 超越当天最优单模型={(sub['oracle']<sub['oracle'].min()).mean():.1%}")

    # 分时段看 nnls_hour 表现（光伏 vs 其他）
    r["is_pv"] = r["hour"].between(10, 14)
    print("\n=== 光伏时段(10-14h) vs 其他 (nnls_hour) ===")
    for seg, mask in [("光伏10-14h", r["is_pv"]), ("其他", ~r["is_pv"])]:
        sub = r[mask].dropna(subset=["nnls_hour", "best_hist"])
        if len(sub):
            print(f"{seg}: nnls_hour 超越历史最优单模型={(sub['nnls_hour']<sub['best_hist']).mean():.1%} | "
                  f"nnls_hour={sub['nnls_hour'].mean():.2f} best_hist={sub['best_hist'].mean():.2f} (n={len(sub)})")

    out = PROJECT / "outputs/experiments/03_fusion_weighting/nnls_ab"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "hour_select.csv", index=False)
    print(f"\n已存 hour_select.csv | 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()


