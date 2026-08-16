"""实证：sgdfnet 是否全面最优？按 regime 细分 + 误差相关性分析。
决定"单用 sgdfnet" vs "继续融合"的实证依据。
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
from fusion.learners.daily_ledger_gef import compute_daily_loss
from utils.resolution import resolve_resolution

res = resolve_resolution("15min")
pred = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\prediction\prediction_ledger.parquet")
act = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\actual\actual_ledger.parquet")
pred = pred[pred["task"] == "realtime"]
act = act[act["task"] == "realtime"]
MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]

# 合并宽表
piv = pred.pivot_table(index=["target_day", "business_period"], columns="model_name", values="y_pred")
a = act[["target_day", "business_period", "y_true", "hour_business"]].copy()
a["dow"] = pd.to_datetime(a["target_day"]).dt.dayofweek
a["month"] = pd.to_datetime(a["target_day"]).dt.month
m = a.merge(piv, on=["target_day", "business_period"]).dropna()

# regime 划分
m["is_weekend"] = m["dow"] >= 5
m["is_pv"] = m["hour_business"].between(10, 14)  # 光伏段
# 用实际负荷(直调) 无, 用价格波动率近似 regime
m["price"] = m["y_true"]
m["month_grp"] = np.where(m["month"].isin([12, 1, 2]), "冬",
                  np.where(m["month"].isin([3, 4, 5]), "春",
                  np.where(m["month"].isin([6, 7, 8]), "夏", "秋")))

print("=== 各 regime 下谁最优 (composite loss) ===")
for name, sub in [
    ("工作日", m[~m["is_weekend"]]), ("周末", m[m["is_weekend"]]),
    ("光伏段10-14h", m[m["is_pv"]]), ("非光伏", m[~m["is_pv"]]),
]:
    print(f"\n--- {name} (n={len(sub)}) ---")
    for mm in MODELS:
        l = compute_daily_loss(sub["y_true"].values, sub[mm].values, "composite")
        print(f"  {mm}: {l:.2f}")

print("\n=== 分月谁最优 ===")
for mg, sub in m.groupby("month_grp"):
    lv = {mm: compute_daily_loss(sub["y_true"].values, sub[mm].values, "composite") for mm in MODELS}
    print(f"  {mg}: " + ", ".join(f"{mm}={lv[mm]:.2f}" for mm in MODELS) + f" | 最优={min(lv, key=lv.get)}")

print("\n=== 误差相关性 (Pearson) ===")
err = {mm: (m[mm].values - m["y_true"].values) for mm in MODELS}
for i, mm in enumerate(MODELS):
    for nn in MODELS[i+1:]:
        c = np.corrcoef(err[mm], err[nn])[0, 1]
        print(f"  {mm} vs {nn}: {c:.3f}")

print("\n=== sgdfnet 胜率 (按 day+period 单元 vs 其他模型) ===")
# 每单元谁赢
m["period"] = np.where(m["business_period"] <= 32, "1_32", np.where(m["business_period"] <= 64, "33_64", "65_96"))
units = []
for (d, per), g in m.groupby(["target_day", "period"]):
    lv = {mm: compute_daily_loss(g["y_true"].values, g[mm].values, "composite") for mm in MODELS}
    best = min(lv, key=lv.get)
    units.append({"day": d, "period": per, "best": best, **{f"l_{mm}": lv[mm] for mm in MODELS}})
u = pd.DataFrame(units)
print("单元数:", len(u))
print("sgdfnet 是单元最优的比例:", (u["best"] == "sgdfnet").mean())
print("其他模型最优的比例:", u[u["best"] != "sgdfnet"]["best"].value_counts().to_dict())
print("sgdfnet 非最优的单元(按period):")
sub = u[u["best"] != "sgdfnet"]
print(sub.groupby(["period", "best"]).size())
