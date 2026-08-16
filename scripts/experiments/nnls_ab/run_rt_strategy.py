"""RT 融合策略实验：sgdfnet 优先 + 差模型(timemixer/rt916)限权/负权。

策略对比（每 (day, period) 单元，30d 窗口，滚动 230 天）：
  A. nnls 纯学（现状）
  B. nnls + 初始先验 sgdfnet=0.7（ridge 拉向先验, 先验强度扫描）
  C. 差模型权重 cap：timemixer/rt916 <= cap_val, sgdfnet 无上限
  D. 差模型负权：timemixer/rt916 允许负到 neg_val, sgdfnet 到 1.5

评估：整体 composite loss + 超越 sgdfnet 单模型占比 + 超越等权占比。
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5")
from fusion.learners.daily_ledger_gef import NNLSGEF, NNLSConfig, compute_daily_loss
from utils.resolution import resolve_resolution

res = resolve_resolution("15min")
pred = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\prediction\prediction_ledger.parquet")
act = pd.read_parquet(r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5\outputs\ledger_96\realtime\actual\actual_ledger.parquet")
pred = pred[pred["task"] == "realtime"]
act = act[act["task"] == "realtime"]
MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]
WEAK = ["timemixer", "rt916"]  # 持续差模型


def bls(X, y, bounds, prior=None, ridge=0.0, sum_eq=1.0):
    """有界最小二乘。bounds: list[(lo,hi)] per model. prior: 可选先验。"""
    n = X.shape[1]
    w0 = np.ones(n) / n if prior is None else np.array(prior)

    def obj(w):
        err = X @ w - y
        reg = ridge * np.sum((w - w0) ** 2) if ridge > 0 else 0.0
        return float(np.sum(err ** 2)) + reg

    cons = [{"type": "eq", "fun": lambda w: w.sum() - sum_eq}]
    r = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 400, "ftol": 1e-10})
    if r.success and np.all(np.isfinite(r.x)):
        return r.x
    return None


def collect_win(pred, act, win):
    """窗口内 OOF: {period: (X, y)}"""
    out = {}
    for d in win:
        a = act[act["target_day"] == d]
        p = pred[pred["target_day"] == d]
        for per in res.period_names:
            ap = a[a["period"] == per].sort_values("business_period")
            yt = ap["y_true"].values
            cols = []
            for m in MODELS:
                mp = p[(p["model_name"] == m) & (p["period"] == per)].sort_values("business_period")
                if len(mp) == len(yt) and not mp["y_pred"].isna().any():
                    cols.append(mp["y_pred"].values)
            if len(cols) == len(MODELS) and not np.isnan(yt).any():
                ent = out.setdefault(per, {"X": [], "y": []})
                ent["X"].append(np.column_stack(cols))
                ent["y"].append(yt)
    return {k: (np.vstack(v["X"]), np.concatenate(v["y"])) for k, v in out.items()}


def eval_day(pred, act, D, weights_per, weights_hour=None):
    """评估目标日 D。返回 per-period 融合 loss。"""
    aD = act[act["target_day"] == D].sort_values("business_period")
    pD = pred[pred["target_day"] == D]
    yt = aD["y_true"].values
    cols = {}
    for m in MODELS:
        mp = pD[pD["model_name"] == m].sort_values("business_period")
        if len(mp) == len(yt) and not mp["y_pred"].isna().any():
            cols[m] = mp["y_pred"].values
    if len(cols) != len(MODELS) or np.isnan(yt).any():
        return None
    rows = []
    per_col = aD["period"].values
    for per in res.period_names:
        mask = per_col == per
        if not mask.any():
            continue
        yt_p = yt[mask]
        Xp = np.column_stack([cols[m][mask] for m in MODELS])
        w = weights_per.get(per)
        if w is None:
            continue
        y_fus = Xp @ w
        l_fus = compute_daily_loss(yt_p, y_fus, "composite")
        l_sgd = compute_daily_loss(yt_p, cols["sgdfnet"][mask], "composite")
        l_eq = compute_daily_loss(yt_p, np.mean(Xp, axis=1), "composite")
        rows.append({"day": D, "period": per, "fuse": l_fus, "sgdfnet": l_sgd, "equal": l_eq})
    return rows


days = sorted(act["target_day"].unique())
print(f"RT: {len(days)} 天, step=3")

configs = {
    "A_nnls_纯学": lambda Xs, y, n: bls(Xs, y, [(0.0, 1.0)] * n, ridge=0.0),
    "B_prior_sgd07": lambda Xs, y, n: bls(Xs, y, [(0.0, 1.0)] * n, prior=[0.1, 0.7, 0.1, 0.1], ridge=0.05),
    "B_prior_sgd09": lambda Xs, y, n: bls(Xs, y, [(0.0, 1.0)] * n, prior=[0.03, 0.9, 0.03, 0.03], ridge=0.05),
    "C_cap_weak05": lambda Xs, y, n: bls(Xs, y, [(0.0, 0.5), (0.0, 1.0), (0.0, 0.5), (0.0, 0.5)]),
    "C_cap_weak02": lambda Xs, y, n: bls(Xs, y, [(0.0, 0.2), (0.0, 1.0), (0.0, 0.2), (0.0, 0.2)]),
    "D_neg_weak_neg01": lambda Xs, y, n: bls(Xs, y, [(-0.1, 1.0), (0.0, 1.5), (-0.1, 1.0), (-0.1, 1.0)]),
    "D_neg_weak_neg02": lambda Xs, y, n: bls(Xs, y, [(-0.2, 1.0), (0.0, 1.5), (-0.2, 1.0), (-0.2, 1.0)]),
}

all_res = {name: [] for name in configs}
for i in range(30, len(days), 2):
    D = days[i]
    win = days[i - 30:i]
    pool = collect_win(pred, act, win)
    for name, fn in configs.items():
        w_per = {}
        for per, (X, y) in pool.items():
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
            w = fn(Xs, y, len(MODELS))
            if w is not None:
                w_per[per] = w
        rows = eval_day(pred, act, D, w_per)
        if rows:
            all_res[name].extend(rows)

print("\n===== 策略对比 =====")
for name, rows in all_res.items():
    r = pd.DataFrame(rows)
    if r.empty:
        continue
    n = len(r)
    win_sgd = (r["fuse"] < r["sgdfnet"]).mean()
    win_eq = (r["fuse"] < r["equal"]).mean()
    rel_sgd = ((r["sgdfnet"] - r["fuse"]) / r["sgdfnet"]).mean()
    print(f"{name:20s}: fuse={r['fuse'].mean():.2f} sgd={r['sgdfnet'].mean():.2f} eq={r['equal'].mean():.2f} | "
          f"超越sgdfnet={win_sgd:.1%} 超越等权={win_eq:.1%} 相对sgd={rel_sgd:+.1%} (n={n})")

