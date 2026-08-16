"""负权重稳健实现对比：OLS解析解+投影 vs SLSQP。看是否 n 覆盖更全、更稳。"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls as scipy_nnls
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
from fusion.learners.daily_ledger_gef import compute_daily_loss
from utils.resolution import resolve_resolution

res = resolve_resolution("15min")
pred = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\prediction\prediction_ledger.parquet")
act = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\actual\actual_ledger.parquet")
pred = pred[pred["task"] == "realtime"]
act = act[act["task"] == "realtime"]
MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
WEAK = [2, 3]  # timemixer, rt916 index

# 有界最小二乘：先无约束OLS，再投影到边界，再归一化(投影法)
def ols_proj(X, y, lo, hi):
    """OLS 解 w_ols, 然后投影到 [lo,hi] + 归一化 sum=1。lo 可负。"""
    n = X.shape[1]
    w_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    # 投影到 [lo, hi]
    w_proj = np.clip(w_ols, lo, hi)
    s = w_proj.sum()
    if abs(s) < 1e-9:
        return None
    w_proj = w_proj / s
    # 再 clip(因为归一化可能再越界) → 迭代投影(简单2次)
    w_proj = np.clip(w_proj, lo, hi)
    s2 = w_proj.sum()
    if abs(s2) < 1e-9:
        return None
    return w_proj / s2


def slsqp(X, y, lo, hi, w0=None):
    n = X.shape[1]
    w0 = np.ones(n) / n if w0 is None else np.array(w0)

    def obj(w):
        return float(np.sum((X @ w - y) ** 2))
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(lo, hi)] * n
    r = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
    if r.success and np.all(np.isfinite(r.x)):
        return r.x
    return None


def collect(pred, act, win):
    out = {}
    for d in win:
        a = act[act["target_day"] == d]
        p = pred[pred["target_day"] == d]
        for per in res.period_names:
            ap = a[a["period"] == per].sort_values("business_period")
            cols = []
            for m in MODELS:
                mp = p[(p["model_name"] == m) & (p["period"] == per)].sort_values("business_period")
                if len(mp) == len(ap) and not mp["y_pred"].isna().any():
                    cols.append(mp["y_pred"].values)
            if len(cols) == len(MODELS) and not ap["y_true"].isna().any():
                ent = out.setdefault(per, {"X": [], "y": []})
                ent["X"].append(np.column_stack(cols))
                ent["y"].append(ap["y_true"].values)
    return {k: (np.vstack(v["X"]), np.concatenate(v["y"])) for k, v in out.items()}


days = sorted(act["target_day"].unique())
configs = {
    "nnls(纯学)": ("nnls", None, None),
    "ols_proj_neg01": ("proj", -0.1, 1.5),
    "ols_proj_neg02": ("proj", -0.2, 1.5),
    "ols_proj_neg05": ("proj", -0.5, 2.0),
    "slsqp_neg01": ("slsqp", -0.1, 1.5),
}
# 注意：slsqp 的 hi 对 sgdfnet 也限制 1.5, 但投影法 per-model 界更细。简化统一。
# 更精细: 对弱模型负权, 对强模型 hi 大
fine_bounds = {
    "proj_neg01": ([-0.1, 0.0, -0.1, -0.1], [1.0, 1.5, 1.0, 1.0]),
    "proj_neg02": ([-0.2, 0.0, -0.2, -0.2], [1.0, 1.5, 1.0, 1.0]),
    "proj_neg05": ([-0.5, 0.0, -0.5, -0.5], [1.0, 2.0, 1.0, 1.0]),
}

all_res = {name: [] for name in list(configs.keys()) + [f"fine_{k}" for k in fine_bounds]}
for i in range(30, len(days), 2):
    D = days[i]
    win = days[i - 30:i]
    pool = collect(pred, act, win)
    aD = act[act["target_day"] == D].sort_values("business_period")
    pD = pred[pred["target_day"] == D]
    yt = aD["y_true"].values
    cols = {}
    for m in MODELS:
        mp = pD[pD["model_name"] == m].sort_values("business_period")
        if len(mp) == len(yt) and not mp["y_pred"].isna().any():
            cols[m] = mp["y_pred"].values
    if len(cols) != len(MODELS) or np.isnan(yt).any():
        continue
    per_col = aD["period"].values

    # 计算各策略权重 per period
    strategies = {}
    for name, (kind, lo, hi) in configs.items():
        w_p = {}
        for per, (X, y) in pool.items():
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
            if kind == "nnls":
                sol, _ = scipy_nnls(Xs, y)
                s = sol.sum()
                w = sol / s if s > 1e-9 else None
            elif kind == "proj":
                w = ols_proj(Xs, y, lo, hi)
            else:
                w = slsqp(Xs, y, lo, hi)
            if w is not None:
                w_p[per] = w
        strategies[name] = w_p
    for k, (lo, hi) in fine_bounds.items():
        w_p = {}
        for per, (X, y) in pool.items():
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
            w = ols_proj(Xs, y, lo, hi) if False else None
            # fine: 用带 per-model 界的投影
            w_ols, *_ = np.linalg.lstsq(Xs, y, rcond=None)
            w_proj = np.clip(w_ols, lo, hi)
            s = w_proj.sum()
            if abs(s) > 1e-9:
                w_proj = w_proj / s
                w_p[per] = w_proj
        strategies[f"fine_{k}"] = w_p

    for name, w_p in strategies.items():
        for per in res.period_names:
            mask = per_col == per
            if not mask.any() or per not in w_p:
                continue
            yt_p = yt[mask]
            Xp = np.column_stack([cols[m][mask] for m in MODELS])
            y_fus = Xp @ w_p[per]
            all_res[name].append({"day": D, "period": per,
                                  "fuse": compute_daily_loss(yt_p, y_fus, "composite"),
                                  "sgd": compute_daily_loss(yt_p, cols["sgdfnet"][mask], "composite"),
                                  "eq": compute_daily_loss(yt_p, np.mean(Xp, axis=1), "composite")})

print("===== 策略对比 (统一样本? 否, 各策略独立 n) =====")
for name, rows in all_res.items():
    r = pd.DataFrame(rows)
    if r.empty:
        print(f"{name:18s}: no data")
        continue
    n = len(r)
    win_sgd = (r["fuse"] < r["sgd"]).mean()
    win_eq = (r["fuse"] < r["eq"]).mean()
    print(f"{name:18s}: fuse={r['fuse'].mean():.2f} sgd={r['sgd'].mean():.2f} | 超越sgd={win_sgd:.1%} 超越等权={win_eq:.1%} (n={n})")
