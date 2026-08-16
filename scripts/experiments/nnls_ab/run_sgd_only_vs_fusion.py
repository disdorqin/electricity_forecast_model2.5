"""决策实验：只用 sgdfnet vs 现有 NNLS 融合 vs 等权。用 build_ledger_training_table。"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
from pipelines.prediction_ledger import build_ledger_training_table
from fusion.learners.daily_ledger_gef import NNLSGEF, NNLSConfig, compute_daily_loss
from utils.resolution import resolve_resolution

res = resolve_resolution("15min")
pred = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\prediction\prediction_ledger.parquet")
act = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\actual\actual_ledger.parquet")
pred = pred[pred["task"] == "realtime"]
act = act[act["task"] == "realtime"]
MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
days = sorted(act["target_day"].unique())

rows = []
for i in range(30, len(days), 3):
    D = days[i]
    win = days[i-30:i]
    tt = build_ledger_training_table(pred[pred["target_day"].isin(win)], act[act["target_day"].isin(win)],
                                     target_day=D, window_days=len(win), window_days_list=win)
    if tt.empty:
        continue
    g = NNLSGEF(NNLSConfig(window_days=21, resolution=res, granularity="period"))
    w = g.fit(tt)

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
        l_sgd = compute_daily_loss(yt, cols["sgdfnet"], "composite")
        wm = w.get(("realtime", per))
        if wm is None:
            continue
        y_nnls = np.sum([wm.get(m, 0) * cols[m] for m in MODELS], axis=0)
        l_nnls = compute_daily_loss(yt, y_nnls, "composite")
        l_eq = compute_daily_loss(yt, np.mean([cols[m] for m in MODELS], axis=0), "composite")
        l_orc = min(compute_daily_loss(yt, cols[m], "composite") for m in MODELS)
        rows.append({"day": D, "period": per, "sgd": l_sgd, "nnls": l_nnls, "equal": l_eq, "oracle": l_orc})

r = pd.DataFrame(rows)
print(f"n单元: {len(r)}")
print("=== 每单元平均 loss (越小越好) ===")
for col, label in [("sgd", "只用sgdfnet"), ("nnls", "NNLS融合"), ("equal", "等权"), ("oracle", "oracle上限")]:
    print(f"  {label:14s}: {r[col].mean():.2f}")
print()
print("=== 对比 ===")
print(f"  nnls赢sgd: {(r['nnls']<r['sgd']).mean():.1%} | nnls-sgd差: {(r['nnls']-r['sgd']).mean():+.2f}")
print(f"  equal赢sgd: {(r['equal']<r['sgd']).mean():.1%} | equal-sgd差: {(r['equal']-r['sgd']).mean():+.2f}")
print(f"  oracle赢sgd: {(r['oracle']<r['sgd']).mean():.1%} (理论上限)")
