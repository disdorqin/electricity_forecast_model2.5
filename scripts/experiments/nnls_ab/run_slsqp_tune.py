"""SLSQP 软门控超参扫描 (RT)。找 reg / bound 最优配置。"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
from pipelines.prediction_ledger import build_ledger_training_table
from fusion.learners.daily_ledger_gef import compute_daily_loss, smape_floor50
from fusion.weights import fit_weights_from_long_table
from utils.resolution import resolve_resolution

res = resolve_resolution("15min")
task = "realtime"
pred = pd.read_parquet(rf"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\{task}\prediction\prediction_ledger.parquet")
act = pd.read_parquet(rf"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\{task}\actual\actual_ledger.parquet")
pred = pred[pred["task"] == task]
act = act[act["task"] == task]
MODELS = sorted(pred["model_name"].unique())
days = sorted(act["target_day"].unique())

configs = [
    {"reg": 0.05, "lb": -0.5, "ub": 1.2},
    {"reg": 0.1, "lb": -0.5, "ub": 1.2},
    {"reg": 0.2, "lb": -0.5, "ub": 1.2},
    {"reg": 0.5, "lb": -0.5, "ub": 1.2},
    {"reg": 0.2, "lb": -0.2, "ub": 1.5},
    {"reg": 0.2, "lb": 0.0, "ub": 1.0},
    {"reg": 0.1, "lb": -0.2, "ub": 1.5},
    {"reg": 0.1, "lb": -0.3, "ub": 1.3},
]
agg = {cfg["reg"]: {cfg["lb"]: {cfg["ub"]: []}} for cfg in configs}
# 简化: 直接按 tuple 存
results = {f"reg={c['reg']}_lb={c['lb']}_ub={c['ub']}": [] for c in configs}

for i in range(30, len(days), 3):
    D = days[i]
    win = days[i - 30:i]
    tt = build_ledger_training_table(pred[pred["target_day"].isin(win)], act[act["target_day"].isin(win)],
                                     target_day=D, window_days=len(win), window_days_list=win)
    if tt.empty:
        continue
    aD = act[act["target_day"] == D]
    pD = pred[pred["target_day"] == D]
    for cfg in configs:
        name = f"reg={cfg['reg']}_lb={cfg['lb']}_ub={cfg['ub']}"
        try:
            wdf, _ = fit_weights_from_long_table(tt, reg=cfg["reg"], lower_bound=cfg["lb"], upper_bound=cfg["ub"], resolution=res)
            w_s = {}
            for _, r in wdf.iterrows():
                w_s.setdefault((r["task"], r["period"]), {})[r["model_name"]] = r["weight"]
        except Exception:
            continue
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
            wm = w_s.get(("realtime", per))
            y_f = np.sum([wm.get(m, 0) * cols[m] for m in MODELS], axis=0) if wm else np.mean([cols[m] for m in MODELS], axis=0)
            results[name].append({"c": compute_daily_loss(yt, y_f, "composite"), "s": smape_floor50(yt, y_f)})

print("=== SLSQP 超参扫描 (RT, 201 单元) ===")
print(f"{'配置':30s} {'composite':>10s} {'SMAPE%':>10s}")
for name, rows in results.items():
    r = pd.DataFrame(rows)
    if r.empty:
        print(f"{name:30s} no data")
        continue
    print(f"{name:30s} {r['c'].mean():>10.2f} {r['s'].mean():>10.2f}")
