#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一基准：所有融合方法在同一采样/口径下重跑，输出 composite + SMAPE 双指标。
解决文档中 NNLS 37.46/35.66/37.08 等数字混乱问题。

方法：bgew / nnls / slsqp_smape / equal / best_single(oracle)
采样：30d 窗, step=3, 同 201 单元（DA/RT 各跑）
口径：composite(0.7*SMAPE%+0.3*MAE%) 与 纯 SMAPE%，两列都给。
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
from pipelines.prediction_ledger import build_ledger_training_table
from fusion.learners.daily_ledger_gef import NNLSGEF, NNLSConfig, DailyLedgerGEF, GEFConfig, compute_daily_loss, smape_floor50
from fusion.weights import fit_weights_from_long_table
from utils.resolution import resolve_resolution

res = resolve_resolution("15min")


def run_task(task):
    pred = pd.read_parquet(rf"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\{task}\prediction\prediction_ledger.parquet")
    act = pd.read_parquet(rf"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\{task}\actual\actual_ledger.parquet")
    pred = pred[pred["task"] == task]
    act = act[act["task"] == task]
    MODELS = sorted(pred["model_name"].unique())
    days = sorted(act["target_day"].unique())

    rows = []
    for i in range(30, len(days), 1):
        D = days[i]
        win = days[i - 30:i]
        tt = build_ledger_training_table(pred[pred["target_day"].isin(win)], act[act["target_day"].isin(win)],
                                         target_day=D, window_days=len(win), window_days_list=win)
        if tt.empty:
            continue
        # 各 learner 学权重
        w_bgew = DailyLedgerGEF(GEFConfig(window_days=len(win), resolution=res)).fit(tt)
        w_nnls = NNLSGEF(NNLSConfig(window_days=21, resolution=res, granularity="period")).fit(tt)
        try:
            wdf_s, _ = fit_weights_from_long_table(tt, reg=0.2, lower_bound=0.0, upper_bound=1.0, resolution=res)
            w_sl = {}
            for _, r in wdf_s.iterrows():
                w_sl.setdefault((r["task"], r["period"]), {})[r["model_name"]] = r["weight"]
        except Exception:
            w_sl = {}

        aD = act[act["target_day"] == D]
        pD = pred[pred["target_day"] == D]
        for per in res.period_names:
            ap = aD[aD["period"] == per].sort_values("business_period")
            yt = ap["y_true"].values
            cols = {}
            for m in MODELS:
                mp = pD[(pD["model_name"] == m) & (pD["period"] == per)].sort_values("business_period")
                if len(mp) == len(yt) and not mp["y_pred"].isna().any():
                    cols[m] = mp["y_pred"].values
            if len(cols) != len(MODELS) or np.isnan(yt).any():
                continue
            Xp = np.column_stack([cols[m] for m in MODELS])

            def evalfn(name, yp):
                rows.append({"day": D, "period": per, "method": name,
                             "c": compute_daily_loss(yt, yp, "composite"),
                             "s": smape_floor50(yt, yp)})

            evalfn("equal", np.mean(Xp, axis=1))
            wm_b = w_bgew.get((task, per))
            if wm_b:
                evalfn("bgew", np.sum([wm_b.get(m, 0) * cols[m] for m in MODELS], axis=0))
            wm_n = w_nnls.get((task, per))
            if wm_n:
                evalfn("nnls", np.sum([wm_n.get(m, 0) * cols[m] for m in MODELS], axis=0))
            wm_s = w_sl.get((task, per))
            if wm_s:
                evalfn("slsqp_smape", np.sum([wm_s.get(m, 0) * cols[m] for m in MODELS], axis=0))
            # oracle: 当天最优单模型
            best_m = min(MODELS, key=lambda m: compute_daily_loss(yt, cols[m], "composite"))
            evalfn("oracle_best", cols[best_m])

    r = pd.DataFrame(rows)
    print(f"\n===== {task} =====")
    print(f"n单元: {len(r)} (同 201 采样, step=3, 30d窗)")
    piv = r.pivot_table(index="method", values=["c", "s"], aggfunc="mean")
    piv.columns = ["composite", "SMAPE%"]
    order = ["bgew", "nnls", "slsqp_smape", "equal", "oracle_best"]
    piv = piv.reindex([m for m in order if m in piv.index])
    print(piv.round(2).to_string())
    # win rate vs equal
    print("\n赢等权占比 (composite):")
    for m in ["bgew", "nnls", "slsqp_smape"]:
        sub = r[r["method"].isin([m, "equal"])].pivot_table(index=["day", "period"], columns="method", values="c").dropna()
        if m in sub and "equal" in sub:
            print(f"  {m}: {(sub[m] < sub['equal']).mean():.1%}")
    # 输出 csv 供复核
    r.to_csv(rf"C:\Users\37813\AppData\Local\Temp\opencode\unified_bench_{task}.csv", index=False)
    print(f"已存 unified_bench_{task}.csv")
    return r


for task in ["realtime", "dayahead"]:
    run_task(task)

