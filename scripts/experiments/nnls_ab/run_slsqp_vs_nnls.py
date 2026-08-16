"""融合实验：NNLS(现有生产) vs SLSQP 软门控(weights.py, smape+reg, 允许负权) vs 等权。
滚动 30 天窗，230 天数据，DA 和 RT 都测。
输出每个策略的整体 composite loss + SMAPE% + 赢等权占比。
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
from pipelines.prediction_ledger import build_ledger_training_table
from fusion.learners.daily_ledger_gef import NNLSGEF, NNLSConfig, compute_daily_loss, smape_floor50, mae_percent
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
    print(f"\n===== {task} ({MODELS}) =====")

    rows = []
    for i in range(30, len(days), 3):
        D = days[i]
        win = days[i - 30:i]
        tt = build_ledger_training_table(pred[pred["target_day"].isin(win)], act[act["target_day"].isin(win)],
                                         target_day=D, window_days=len(win), window_days_list=win)
        if tt.empty:
            continue
        # NNLS
        g = NNLSGEF(NNLSConfig(window_days=21, resolution=res, granularity="period"))
        w_nnls = g.fit(tt)
        # SLSQP 软门控 (weights.py)
        try:
            wdf, report = fit_weights_from_long_table(tt, reg=0.2, lower_bound=0.0, upper_bound=1.0, resolution=res)
            w_slsqp = {}
            for _, r in wdf.iterrows():
                w_slsqp.setdefault((r["task"], r["period"]), {})[r["model_name"]] = r["weight"]
        except Exception as e:
            w_slsqp = {}
            print(f"  SLSQP 失败: {e}")

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
            l_eq = compute_daily_loss(yt, np.mean(Xp, axis=1), "composite")
            # NNLS
            wm_n = w_nnls.get((task, per))
            y_nnls = np.sum([wm_n.get(m, 0) * cols[m] for m in MODELS], axis=0) if wm_n else np.mean(Xp, axis=1)
            l_nnls = compute_daily_loss(yt, y_nnls, "composite")
            s_nnls = smape_floor50(yt, y_nnls)
            # SLSQP
            wm_s = w_slsqp.get((task, per))
            y_sl = np.sum([wm_s.get(m, 0) * cols[m] for m in MODELS], axis=0) if wm_s else np.mean(Xp, axis=1)
            l_sl = compute_daily_loss(yt, y_sl, "composite")
            s_sl = smape_floor50(yt, y_sl)
            # 只用最优单模型(该单元历史最优, 简化用整体最强)
            best_m = min(MODELS, key=lambda m: compute_daily_loss(yt, cols[m], "composite"))
            l_best = compute_daily_loss(yt, cols[best_m], "composite")
            s_best = smape_floor50(yt, cols[best_m])
            rows.append({"day": D, "period": per, "equal": l_eq,
                         "nnls_c": l_nnls, "nnls_s": s_nnls,
                         "slsqp_c": l_sl, "slsqp_s": s_sl,
                         "best_c": l_best, "best_s": s_best})

    r = pd.DataFrame(rows)
    print(f"n单元: {len(r)}")
    print("=== composite loss (越小越好) ===")
    for col, label in [("nnls_c", "NNLS(现有)"), ("slsqp_c", "SLSQP软门控"), ("equal", "等权"), ("best_c", "最优单模型")]:
        print(f"  {label:12s}: {r[col].mean():.2f}")
    print("=== SMAPE% (越小越好) ===")
    for col, label in [("nnls_s", "NNLS(现有)"), ("slsqp_s", "SLSQP软门控"), ("best_s", "最优单模型")]:
        print(f"  {label:12s}: {r[col].mean():.2f}")
    sub = r.dropna(subset=["nnls_c", "slsqp_c"])
    print(f"=== 对比 ===")
    print(f"  NNLS 赢等权: {(sub['nnls_c']<sub['equal']).mean():.1%} | SLSQP 赢等权: {(sub['slsqp_c']<sub['equal']).mean():.1%}")
    print(f"  SLSQP 赢 NNLS: {(sub['slsqp_c']<sub['nnls_c']).mean():.1%} | SLSQP-NNLS 差: {(sub['slsqp_c']-sub['nnls_c']).mean():+.2f}")
    print(f"  SLSQP 赢最优单模型: {(sub['slsqp_c']<sub['best_c']).mean():.1%} | NNLS 赢最优: {(sub['nnls_c']<sub['best_c']).mean():.1%}")
    return r


for task in ["realtime", "dayahead"]:
    run_task(task)

