#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NNLSGEF 消融实验 — 窗口长度 / period 粒度 / 参数调优
==================================================
回答三个问题（用户关切）：
  1. 96 点数据级更细、量更大，能否让融合超越最优单模型？
  2. 窗口长度从 30 天改更多（45/60）能否有效提升收敛？
  3. 学习器参数（weight_floor / 时间衰减加权）如何影响 loss 曲线？

评估指标（每个 (day, period) 单元）：
  - composite loss（生产公式 0.7*SMAPE_floor50 + 0.3*MAE%）
  - vs equal：相对提升、赢率
  - vs oracle（当天最优单模型）：超越率（用户目标 >=70%）

用法：python scripts/experiments/nnls_ab/run_ab.py [--quick]
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
MODELS = ["timesfm", "sgdfnet", "timemixer", "rt916"]


def load_ledger():
    pred = pd.read_parquet(PROJECT / "outputs/ledger_96/realtime/prediction/prediction_ledger.parquet")
    act = pd.read_parquet(PROJECT / "outputs/ledger_96/realtime/actual/actual_ledger.parquet")
    pred = pred[pred["task"] == "realtime"][["target_day", "business_period", "period", "hour_business", "model_name", "y_pred"]]
    act = act[act["task"] == "realtime"][["target_day", "business_period", "period", "hour_business", "y_true"]]
    return pred, act


def build_day_matrix(pred, act, day, bucket):
    """按 bucket 分桶的一天数据。bucket='period' 或 'hour' 或 'point'。
    返回 {bucket_key: (y_true, {model: y_pred})}
    """
    a = act[act["target_day"] == day]
    p = pred[pred["target_day"] == day]
    keys = []
    if bucket == "period":
        keys = sorted(a["period"].unique())
    elif bucket == "hour":
        keys = sorted(a["hour_business"].unique())
    else:  # point = 96 点，每 business_period 一组
        keys = sorted(a["business_period"].unique())
    out = {}
    for k in keys:
        if bucket == "period":
            ap = a[a["period"] == k].sort_values("business_period")
            pp = p[p["period"] == k]
        elif bucket == "hour":
            ap = a[a["hour_business"] == k].sort_values("business_period")
            pp = p[p["hour_business"] == k]
        else:
            ap = a[a["business_period"] == k]
            pp = p[p["business_period"] == k]
        yt = ap["y_true"].values
        pm = {}
        for m in MODELS:
            mp = pp[pp["model_name"] == m].sort_values("business_period")
            if len(mp) == len(yt) and not mp["y_pred"].isna().any() and not np.isnan(yt).any():
                pm[m] = mp["y_pred"].values
        out[k] = (yt, pm)
    return out


def collect_oof(pred, act, win_days, bucket):
    """把窗口内每天的 (bucket, 模型) 预测/实际拼成 OOF 池。
    返回 {bucket_key: {"X": np.ndarray, "y": np.ndarray, "day_idx": np.ndarray}}
    """
    pool = {}
    for d in win_days:
        dm = build_day_matrix(pred, act, d, bucket)
        for k, (yt, pm) in dm.items():
            if len(pm) != len(MODELS) or len(yt) == 0:
                continue
            # 按 MODELS 顺序排 X
            cols = [pm[m] for m in MODELS if m in pm]
            if len(cols) != len(MODELS):
                continue
            X = np.column_stack(cols)
            if len(X) != len(yt) or np.isnan(X).any() or np.isnan(yt).any():
                continue
            ent = pool.setdefault(k, {"X": [], "y": [], "day": []})
            ent["X"].append(X)
            ent["y"].append(yt)
            ent["day"].append(np.full(len(yt), win_days.index(d)))
    return pool


def fit_nnls(pool, weight_floor=0.02, decay=0.0):
    """从 OOF 池学每 bucket 权重。decay>0 时给旧样本指数衰减（时间自适应）。"""
    weights = {}
    for k, ent in pool.items():
        if len(ent["X"]) < 3:
            continue
        X = np.vstack(ent["X"])
        y = np.concatenate(ent["y"])
        # 时间衰减加权：day_idx 越大（越新）权重越大
        if decay > 0 and len(ent["day"]) > 0:
            day_idx = np.concatenate(ent["day"])
            n_days = len(set(day_idx))
            wts = np.exp(decay * day_idx / max(n_days, 1))
            Xw = X * wts[:, None]
            yw = y * wts
        else:
            Xw, yw = X, y
        # 标准化列
        Xs = (Xw - Xw.mean(axis=0)) / (Xw.std(axis=0) + 1e-8)
        sol, _ = scipy_nnls(Xs, yw)
        s = sol.sum()
        if s < 1e-9:
            continue
        w = sol / s
        if weight_floor > 0:
            w = np.maximum(w, weight_floor)
            w = w / w.sum()
        weights[k] = dict(zip(MODELS, w))
    return weights


def evaluate(pred, act, D, bucket, weights, use_equal=False):
    """在目标日 D 上评估融合 vs 等权 vs oracle。返回 (day, bucket, method, loss) 行。"""
    rows = []
    dm = build_day_matrix(pred, act, D, bucket)
    for k, (yt, pm) in dm.items():
        if len(yt) == 0 or len(pm) < 2:
            continue
        # oracle = 当天最优单模型
        loss_m = {m: compute_daily_loss(yt, pm[m], "composite") for m in pm}
        best_m = min(loss_m, key=loss_m.get)
        rows.append((D, k, "oracle", loss_m[best_m]))
        # 等权
        y_eq = np.mean([pm[m] for m in MODELS if m in pm], axis=0)
        rows.append((D, k, "equal", compute_daily_loss(yt, y_eq, "composite")))
        # NNLS
        if weights and k in weights:
            wm = weights[k]
            wsum = sum(wm[m] for m in pm)
            if wsum > 0:
                y_nnls = np.sum([wm[m] * pm[m] for m in pm], axis=0) / wsum
                rows.append((D, k, "nnls", compute_daily_loss(yt, y_nnls, "composite")))
    return rows


def run_experiment(pred, act, days, bucket, window, weight_floor=0.02, decay=0.0, step=2):
    """滚动回测单一配置。days: 全部日期列表（升序）。"""
    all_rows = []
    n_days = len(days)
    for i in range(window, n_days, step):
        D = days[i]
        win_days = days[max(0, i - window):i]
        if len(win_days) < max(10, window * 0.6):
            continue
        pool = collect_oof(pred, act, win_days, bucket)
        weights = fit_nnls(pool, weight_floor=weight_floor, decay=decay)
        rows = evaluate(pred, act, D, bucket, weights)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows, columns=["day", "bucket", "method", "loss"])


def summarize(df, tag):
    piv = df.pivot_table(index=["day", "bucket"], columns="method", values="loss").dropna(subset=["nnls", "equal", "oracle"])
    nnls_win_equal = (piv["nnls"] < piv["equal"]).mean()
    nnls_win_oracle = (piv["nnls"] < piv["oracle"]).mean()
    rel = ((piv["equal"] - piv["nnls"]) / piv["equal"]).mean()
    print(f"[{tag}] n单元={len(piv)} | equal={piv['equal'].mean():.2f} nnls={piv['nnls'].mean():.2f} "
          f"oracle={piv['oracle'].mean():.2f} | nnls赢等权={nnls_win_equal:.1%} nnls超越oracle={nnls_win_oracle:.1%} | 相对提升={rel:+.1%}")
    return {"tag": tag, "n": len(piv), "equal": piv["equal"].mean(), "nnls": piv["nnls"].mean(),
            "oracle": piv["oracle"].mean(), "win_equal": nnls_win_equal,
            "win_oracle": nnls_win_oracle, "rel": rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速模式（每5天采样）")
    args = ap.parse_args()
    step = 5 if args.quick else 2
    t0 = time.time()

    pred, act = load_ledger()
    days = sorted(act["target_day"].unique())
    print(f"数据: {len(days)} 天 ({days[0]} ~ {days[-1]}) | 每{step}天采样回测")

    # 加载 OOF 一次（窗口无关，只依赖 bucket）—— 太慢则分次
    # 每个配置独立滚动（窗口影响 win_days），无法一次缓存。直接跑。

    print("\n========== 实验1: 窗口长度（period 3段基准, weight_floor=0.02）==========")
    results = []
    for w in [14, 21, 30, 45, 60]:
        df = run_experiment(pred, act, days, "period", w, weight_floor=0.02)
        results.append(summarize(df, f"win={w}d"))

    print("\n========== 实验2: period 粒度（窗口21天基准）==========")
    for b in ["period", "hour", "point"]:
        df = run_experiment(pred, act, days, b, 21, weight_floor=0.02)
        results.append(summarize(df, f"bucket={b}"))

    print("\n========== 实验3: 参数调优（period 3段, 窗口30天）==========")
    for wf in [0.0, 0.02, 0.05, 0.10]:
        df = run_experiment(pred, act, days, "period", 30, weight_floor=wf)
        results.append(summarize(df, f"floor={wf}"))
    for dec in [0.0, 0.5, 1.0]:
        df = run_experiment(pred, act, days, "period", 30, weight_floor=0.02, decay=dec)
        results.append(summarize(df, f"decay={dec}"))

    print("\n========== 汇总 ==========")
    rs = pd.DataFrame(results)
    print(rs.to_string(index=False))
    out = PROJECT / "outputs/experiments/nnls_ab"
    out.mkdir(parents=True, exist_ok=True)
    rs.to_csv(out / "summary.csv", index=False)
    print(f"\n汇总已存 outputs/experiments/nnls_ab/summary.csv | 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
