"""Iteration 11: strict direct-spread Directional Loss / Focal Loss screen.

Motivation:
- Serafin & Weron (Energy Economics 2025) introduce a directional loss for spread
  regression that adds a penalty when predicted and realized spreads have opposite sign.
- TADLE (2026) uses Balanced Focal Loss for electricity-price trend classification.

This script tests whether optimizing the training objective for the actual engineering goal
(sign of RT-DA) closes the gap left by tree classifiers.

STRICT INFORMATION CONTRACT
- target day D forecast origin: D-1 14:00;
- target D RT/spread/actual are labels only; target D DA is forbidden as feature;
- target-day forecast fundamentals are allowed;
- D-1 realized spread is represented only by legal p1-p14 context already in P6;
- complete supervised fit labels end at D-2;
- Similar-Day candidates end at D-2;
- all imputation/scaling/target scaling/class weights are fit on the strict train window only;
- fresh 2026-08-15..2026-08-21 holdout remains sealed.

Experiment-only; production is untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    add_similar_day_features,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    dedupe,
    direction_metrics,
    p6_features,
    strict_train_days,
)


class MLP(nn.Module):
    def __init__(self, n_features: int, classification: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 48), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(48, 24), nn.GELU(),
            nn.Linear(24, 1),
        )
        self.classification = classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_transform(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    X = train[features].to_numpy(float)
    Q = test[features].to_numpy(float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    Q = np.where(np.isfinite(Q), Q, med)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where((~np.isfinite(sd)) | (sd < 1e-6), 1.0, sd)
    X = np.clip((X - mu) / sd, -8.0, 8.0).astype(np.float32)
    Q = np.clip((Q - mu) / sd, -8.0, 8.0).astype(np.float32)
    return X, Q, {"imputer":"train_median","scaler":"train_zscore_clip8"}


def directional_loss(pred: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    # Paper form: |y-yhat| + max(0, -lambda*y*yhat).
    return torch.mean(torch.abs(y - pred) + torch.relu(-lam * y * pred))


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float, alpha_pos: float | None) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, p, 1.0-p)
    focal = (1.0-pt).pow(gamma) * bce
    if alpha_pos is not None:
        a = torch.where(y > 0.5, torch.full_like(y, alpha_pos), torch.full_like(y, 1.0-alpha_pos))
        focal = a * focal
    return focal.mean()


def fit_predict(
    X: np.ndarray,
    Q: np.ndarray,
    y_spread: np.ndarray,
    *,
    mode: str,
    param: float,
    seed: int,
    device: torch.device,
    epochs: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    xt = torch.from_numpy(X).to(device)
    qt = torch.from_numpy(Q).to(device)
    y_sp = np.asarray(y_spread, float)
    classification = mode in {"bce", "focal", "focal_bal"}
    m = MLP(X.shape[1], classification=classification).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=0.008, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    if classification:
        y_np = (y_sp > 0).astype(np.float32)
        yt = torch.from_numpy(y_np).to(device)
        n_pos = max(1, int(y_np.sum())); n_neg = max(1, int(len(y_np)-n_pos))
        alpha_pos = n_neg / (n_pos+n_neg) if mode == "focal_bal" else None
        pos_weight = torch.tensor([n_neg/n_pos], dtype=torch.float32, device=device)
    else:
        # Robust train-only target scaling. Sign is invariant to positive scaling.
        scale = max(float(np.nanmedian(np.abs(y_sp))), 10.0)
        yt = torch.from_numpy((y_sp/scale).astype(np.float32)).to(device)
        alpha_pos = None; pos_weight = None

    best_state = None; best_loss = float("inf")
    patience = 18; stale = 0
    for _ in range(epochs):
        m.train(); opt.zero_grad(set_to_none=True); out = m(xt)
        if mode == "mae":
            loss = torch.mean(torch.abs(out-yt))
        elif mode == "dlf":
            loss = directional_loss(out, yt, param)
        elif mode == "bce":
            loss = F.binary_cross_entropy_with_logits(out, yt, pos_weight=pos_weight)
        elif mode == "focal":
            loss = focal_loss(out, yt, gamma=param, alpha_pos=None)
        elif mode == "focal_bal":
            loss = focal_loss(out, yt, gamma=param, alpha_pos=alpha_pos)
        else:
            raise ValueError(mode)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
        lv = float(loss.detach().cpu())
        if lv < best_loss - 1e-5:
            best_loss = lv; best_state = {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; stale=0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad(): out = m(qt).detach().cpu().numpy().astype(float)
    if classification:
        prob = 1.0/(1.0+np.exp(-np.clip(out,-30,30)))
        pred_dir = np.where(prob>=0.5,1,-1)
        return pred_dir, {"best_train_loss":best_loss,"epochs_max":epochs,"alpha_pos":alpha_pos}
    pred_dir = np.where(out>=0.0,1,-1)
    return pred_dir, {"best_train_loss":best_loss,"epochs_max":epochs,"target_scale":scale}


def summarize(ledger: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    z=ledger.copy(); z["month"]=z.target_day.str[:7]; mm=[]
    for (v,mo),g in z.groupby(["variant","month"],sort=True):
        mm.append({"variant":v,"month":mo,"days":g.target_day.nunique(),**direction_metrics(g.target_spread,g.predicted_direction)})
    monthly=pd.DataFrame(mm); rr=[]
    for v,g in monthly.groupby("variant",sort=False):
        acc=g.direction_accuracy.astype(float); bal=g.balanced_direction_accuracy.astype(float); gain=acc-g.all_negative_accuracy.astype(float)
        rr.append({"variant":v,"months":len(g),"mean_month_acc":acc.mean(),"median_month_acc":acc.median(),"min_month_acc":acc.min(),"max_month_acc":acc.max(),"std_month_acc":acc.std(ddof=0),"months_ge_065":int((acc>=.65).sum()),"months_ge_070":int((acc>=.70).sum()),"mean_month_bal":bal.mean(),"min_month_bal":bal.min(),"months_beating_all_negative":int((gain>0).sum()),"mean_gain_vs_all_negative":gain.mean()})
    return monthly,pd.DataFrame(rr).sort_values(["mean_month_acc","mean_month_bal"],ascending=False)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--cube-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube"); ap.add_argument("--start",default="2026-04-01"); ap.add_argument("--end",default="2026-08-14"); ap.add_argument("--epochs",type=int,default=100); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--output-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration11_directional_loss_strictD2"); args=ap.parse_args()
    if args.end >= "2026-08-15": raise RuntimeError("fresh final holdout remains sealed")
    cube=Path(args.cube_root); slot=pd.read_parquet(cube/"slot_table.parquet"); groups=json.loads((cube/"feature_groups.json").read_text(encoding="utf-8")); base=p6_features(groups); slot,sd=add_similar_day_features(slot,groups,k_values=(20,),lookback_days=365); sd20=[c for c in sd["features"] if c.startswith("sd20_")]; features=dedupe(base+sd20)
    if sd["audit"].empty or not sd["audit"].causal_ok.all(): raise RuntimeError("similar-day causal audit failed")
    days=[str(d) for d in sorted(slot.target_day.dropna().astype(str).unique()) if args.start<=str(d)<=args.end]
    all_days=sorted(slot.target_day.dropna().astype(str).unique())
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("DEVICE",device,flush=True)
    specs=[("MLP_MAE","mae",0.0),("MLP_DLF_l1","dlf",1.0),("MLP_DLF_l3","dlf",3.0),("MLP_DLF_l6","dlf",6.0),("MLP_BCE_bal","bce",0.0),("MLP_Focal_g2","focal",2.0),("MLP_FocalBal_g2","focal_bal",2.0)]
    led=[]; audits=[]
    for di,day in enumerate(days,1):
        tr_days=strict_train_days(all_days,day,90); tr=slot[slot.target_day.isin(tr_days)].copy(); te=slot[slot.target_day.eq(day)].sort_values("hour_business").copy(); X,Q,prep=train_transform(tr,te,features); y=tr.target_spread.to_numpy(float)
        for si,(name,mode,param) in enumerate(specs):
            pred,info=fit_predict(X,Q,y,mode=mode,param=param,seed=args.seed+si,device=device,epochs=args.epochs)
            o=te[["target_day","hour_business","period","target_spread"]].copy(); o["variant"]=name; o["predicted_direction"]=pred; o["training_last_day"]=tr_days[-1]; led.append(o); audits.append({"target_day":day,"variant":name,"training_last_day":tr_days[-1],**prep,**info})
        if di%15==0: print(f"DLF {di}/{len(days)} {day}",flush=True)
    ledger=pd.concat(led,ignore_index=True); audit=pd.DataFrame(audits); required=(pd.to_datetime(audit.target_day)-pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d");
    if not audit.training_last_day.le(required).all(): raise RuntimeError("strict training audit failed")
    monthly,robust=summarize(ledger); out=Path(args.output_root); atomic_parquet(out/"ledger.parquet",ledger); atomic_csv(out/"monthly.csv",monthly); atomic_csv(out/"robustness.csv",robust); atomic_csv(out/"training_audit.csv",audit); atomic_csv(out/"similar_day_causal_audit.csv",sd["audit"]); atomic_json(out/"manifest.json",{"status":"complete","experiment":"iteration11_directional_loss","forecast_origin":"D-1 14:00","training_labels":"D-2 and earlier only","preprocessing":"strict train-only median/zscore and target/class scaling","target_day_DA_as_feature":False,"target_day_actual_as_feature":False,"fresh_final_holdout_reserved":["2026-08-15","2026-08-21"],"final_holdout_touched":False,"literature_basis":["Serafin & Weron 2025 Directional Loss: MAE plus wrong-sign spread penalty","TADLE 2026 Balanced Focal Loss for electricity-price trend"]}); print("\nROBUSTNESS\n",robust.to_string(index=False)); top=robust.iloc[0].variant; print("\nTOP MONTHLY\n",monthly[monthly.variant.eq(top)].to_string(index=False))

if __name__=="__main__": main()
