"""Iteration 8: strict dynamic expert competence estimation.

Two literature-inspired branches:
A) Parametric competence: one classifier per expert predicts P(expert is correct | legal state).
B) Local competence (DES-style): find a legal-state Region of Competence in historical OOS
   samples <= D-2, estimate each expert's local accuracy, and select/weight experts.

All base expert predictions are STRICT-D2 OOS outputs. Router/competence labels for target D
use only OOS rows from days <= D-2. Fresh 2026-08-15..08-21 holdout stays sealed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_iteration7_recurring_emergent_router import (
    ROOT, atomic_csv, atomic_json, atomic_parquet, build_meta_features, load_experts, metrics,
)

EXPERTS = ["p6", "sd", "ctx"]


def make_competence(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=100,
        learning_rate=0.035, num_leaves=15, max_depth=4, min_child_samples=35,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=3.0,
        random_state=seed, n_jobs=4, verbosity=-1,
    )


def feature_columns() -> list[str]:
    # Keep the gate deliberately smaller than the forecasting model to reduce meta-overfit.
    return [
        "p6_p", "sd_p", "ctx_p", "p6_conf", "sd_conf", "ctx_conf",
        "sd_ctx_gap", "sd_p6_gap", "ctx_p6_gap", "p6_votes_sd", "p6_votes_ctx",
        "ctx_spread_mean14", "ctx_spread_std14", "ctx_spread_last", "ctx_spread_mean3",
        "ctx_spread_range14", "ctx_spread_absmean14", "ctx_spread_positive_rate14", "ctx_spread_slope14",
        "residual_load_renew", "renewable_share", "bidding_space_ratio", "interconnect_share",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "sd20_positive_rate", "sd20_weighted_positive_rate", "sd20_spread_std", "sd20_mean_distance",
        "sd20_day_positive_rate", "sd20_1_8_positive_rate", "sd20_9_16_positive_rate", "sd20_17_24_positive_rate",
    ]


def correctness(frame: pd.DataFrame, expert: str) -> np.ndarray:
    return (frame[f"{expert}_d"].to_numpy(int) == np.sign(frame.target_spread.to_numpy(float))).astype(int)


def run(x: pd.DataFrame, start: str, end: str, history_days: int, seed: int, knn_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = feature_columns()
    all_days = sorted(x.target_day.unique())
    days = [d for d in all_days if start <= d <= end]
    rows = []; audits = []

    for i, day in enumerate(days, 1):
        max_hist_day = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        hist_days = [d for d in all_days if d <= max_hist_day][-history_days:]
        if len(hist_days) < 60:
            continue
        hist = x[x.target_day.isin(hist_days)].copy()
        q = x[x.target_day.eq(day)].copy()
        if len(q) != 24:
            raise RuntimeError(f"{day}: expected 24 rows")

        # A) Parametric per-expert competence probabilities.
        comp = []
        for j, e in enumerate(EXPERTS):
            y = correctness(hist, e)
            m = make_competence(seed + 13*j)
            m.fit(hist[feats], y)
            comp.append(m.predict_proba(q[feats])[:, 1])
        C = np.column_stack(comp)
        best_idx = np.argmax(C, axis=1)
        dirs = np.column_stack([q[f"{e}_d"].to_numpy(int) for e in EXPERTS])
        probs = np.column_stack([q[f"{e}_p"].to_numpy(float) for e in EXPERTS])
        pred_select = dirs[np.arange(24), best_idx]
        weights = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-9)
        soft_p = np.sum(weights * probs, axis=1)
        pred_soft = np.where(soft_p >= .5, 1, -1)

        # B) DES-style local competence region. Fit scaler/imputer on strict history only.
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xh = scaler.fit_transform(imp.fit_transform(hist[feats]))
        Xq = scaler.transform(imp.transform(q[feats]))
        k = min(knn_k, len(hist))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=4).fit(Xh)
        dist, idx = nn.kneighbors(Xq)
        local_C = np.zeros((24, 3), float)
        corr_hist = np.column_stack([correctness(hist, e) for e in EXPERTS]).astype(float)
        # inverse-distance competence; distances are query-specific and use only legal state features.
        for r in range(24):
            w = 1.0 / np.maximum(dist[r], 1e-6); w /= w.sum()
            local_C[r] = np.sum(corr_hist[idx[r]] * w[:, None], axis=0)
        local_best = np.argmax(local_C, axis=1)
        local_pred = dirs[np.arange(24), local_best]
        local_w = local_C / np.maximum(local_C.sum(axis=1, keepdims=True), 1e-9)
        local_p = np.sum(local_w * probs, axis=1)
        local_soft_pred = np.where(local_p >= .5, 1, -1)

        variants = [
            ("A_competence_select", pred_select, np.full(24, np.nan)),
            ("A_competence_soft", pred_soft, soft_p),
            (f"B_localDES_k{knn_k}_select", local_pred, np.full(24, np.nan)),
            (f"B_localDES_k{knn_k}_soft", local_soft_pred, local_p),
            ("P6_static", q.p6_d.to_numpy(int), q.p6_p.to_numpy(float)),
            ("SD20_static", q.sd_d.to_numpy(int), q.sd_p.to_numpy(float)),
            ("CTX_static", q.ctx_d.to_numpy(int), q.ctx_p.to_numpy(float)),
        ]
        for name, pred, pp in variants:
            o = q[["target_day", "hour_business", "period", "target_spread"]].copy()
            o["variant"] = name; o["predicted_direction"] = pred; o["prob_positive"] = pp
            o["router_training_last_day"] = hist_days[-1]
            rows.append(o)
        audits.append({
            "target_day": day, "router_training_last_day": hist_days[-1],
            "required_last_day_le": max_hist_day, "strict_ok": hist_days[-1] <= max_hist_day,
            "history_days": len(hist_days), "knn_k": k,
        })
        if i % 30 == 0:
            print(f"competence {i}/{len(days)} {day}")
    audit = pd.DataFrame(audits)
    if audit.empty or not audit.strict_ok.all():
        raise RuntimeError("strict competence audit failed")
    return pd.concat(rows, ignore_index=True), audit


def summarize(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z=ledger.copy(); z["month"]=z.target_day.str[:7]; mm=[]
    for (v,m),g in z.groupby(["variant","month"],sort=True): mm.append({"variant":v,"month":m,**metrics(g)})
    monthly=pd.DataFrame(mm); rr=[]
    for v,g in monthly.groupby("variant",sort=False):
        acc=g.direction_accuracy.astype(float); bal=g.balanced_direction_accuracy.astype(float); gain=acc-g.all_negative_accuracy.astype(float)
        rr.append({"variant":v,"months":len(g),"mean_month_acc":acc.mean(),"median_month_acc":acc.median(),"min_month_acc":acc.min(),"max_month_acc":acc.max(),"std_month_acc":acc.std(ddof=0),"months_ge_065":int((acc>=.65).sum()),"months_ge_070":int((acc>=.70).sum()),"mean_month_bal":bal.mean(),"min_month_bal":bal.min(),"months_beating_all_negative":int((gain>0).sum()),"mean_gain_vs_all_negative":gain.mean()})
    return monthly,pd.DataFrame(rr).sort_values(["mean_month_acc","mean_month_bal"],ascending=False)


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default=str(ROOT)); ap.add_argument("--start",default="2026-04-01"); ap.add_argument("--end",default="2026-08-14"); ap.add_argument("--history-days",type=int,default=120); ap.add_argument("--knn-k",type=int,default=160); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--output-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration8_dynamic_competence_strictD2"); args=ap.parse_args()
    if args.end >= "2026-08-15": raise RuntimeError("fresh final holdout remains sealed")
    root=Path(args.root); expert=load_experts(root); x,sd_audit=build_meta_features(root,expert); ledger,audit=run(x,args.start,args.end,args.history_days,args.seed,args.knn_k); monthly,robust=summarize(ledger); out=Path(args.output_root); atomic_parquet(out/"ledger.parquet",ledger); atomic_csv(out/"monthly.csv",monthly); atomic_csv(out/"robustness.csv",robust); atomic_csv(out/"router_training_audit.csv",audit); atomic_csv(out/"similar_day_causal_audit.csv",sd_audit)
    eval_x=x[(x.target_day>=args.start)&(x.target_day<=args.end)]; y=np.sign(eval_x.target_spread.to_numpy(float)); C=np.column_stack([eval_x[f"{e}_d"].to_numpy()==y for e in EXPERTS]); oracle=float(C.any(axis=1).mean())
    atomic_json(out/"manifest.json",{"status":"complete","forecast_origin":"D-1 14:00","router_training_labels":"historical OOS <= D-2","strict_training_audit_pass":bool(audit.strict_ok.all()),"fresh_final_holdout_reserved":["2026-08-15","2026-08-21"],"final_holdout_touched":False,"literature_basis":["Dynamic Ensemble Selection: estimate local classifier competence for each query","DynaME: dynamic specialized experts for recurring/emergent drift"],"oracle_any_expert_correct":"ORACLE_DIAGNOSTIC_ONLY","oracle_value":oracle}); print("\nROBUSTNESS\n",robust.to_string(index=False)); print("\nTOP MONTHLY\n",monthly[monthly.variant.isin(robust.head(3).variant)].to_string(index=False)); print("ORACLE",oracle)

if __name__=="__main__": main()
