"""Iteration 6: expose the full legally-visible D-1 p1-p14 spread context.

The base Feature Cube only broadcasts aggregate summaries of D-1 p1-p14.  Shandong
spread research and recent DART work identify lagged spread as a dominant predictor.
This experiment preserves the strict D-1 14:00 information boundary but adds:
- the raw 14-point D-1 spread vector, sign, and magnitude, broadcast to target D;
- same-slot D-1 spread/sign for target hours 1..14 only; hours 15..24 are missing;
- a visibility mask.

Training labels remain D-2 and earlier only. Production code is untouched.
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

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    add_similar_day_features, p6_features, strict_train_days
)
from utils.resolution import HOURLY


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+f".tmp-{os.getpid()}"); frame.to_csv(tmp,index=False,encoding="utf-8-sig"); os.replace(tmp,path)

def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+f".tmp-{os.getpid()}"); frame.to_parquet(tmp,index=False); os.replace(tmp,path)

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+f".tmp-{os.getpid()}"); tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); os.replace(tmp,path)


def add_visible_context(slot: pd.DataFrame, raw_path: str) -> tuple[pd.DataFrame,list[str],pd.DataFrame]:
    raw=pd.read_csv(raw_path, encoding="gb18030")
    raw["时刻"]=pd.to_datetime(raw["时刻"],errors="coerce")
    raw["business_day"]=raw["时刻"].map(HOURLY.business_day_from_timestamp).astype(str)
    raw["period_no"]=raw["时刻"].map(HOURLY.business_period_from_timestamp).astype(int)
    raw["spread"]=pd.to_numeric(raw["实时电价"],errors="coerce")-pd.to_numeric(raw["日前电价"],errors="coerce")
    partial=raw[raw.period_no.between(1,14)].copy()
    recs=[]; audits=[]
    for ctx_day,g in partial.groupby("business_day",sort=True):
        g=g.sort_values("period_no")
        if set(g.period_no.dropna().astype(int)) < set(range(1,15)):
            continue
        target=(pd.Timestamp(ctx_day)+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        rec={"target_day":target}
        latest=pd.to_datetime(g.loc[g.period_no<=14,"时刻"],errors="coerce").max()
        cutoff=pd.Timestamp(target)-pd.Timedelta(days=1)+pd.Timedelta(hours=14)
        for p in range(1,15):
            row=g[g.period_no.eq(p)].iloc[-1]
            s=float(row.spread) if pd.notna(row.spread) else math.nan
            rec[f"d1_spread_p{p:02d}"]=s
            rec[f"d1_sign_p{p:02d}"]=float(np.sign(s)) if np.isfinite(s) else math.nan
            rec[f"d1_abs_p{p:02d}"]=abs(s) if np.isfinite(s) else math.nan
        recs.append(rec)
        audits.append({"target_day":target,"context_day":ctx_day,"latest_source_ts":latest,"cutoff":cutoff,"causal_ok":bool(pd.isna(latest) or latest<=cutoff)})
    ctx=pd.DataFrame(recs); audit=pd.DataFrame(audits)
    if audit.empty or not audit.causal_ok.all(): raise RuntimeError("visible D-1 context causal audit failed")
    out=slot.merge(ctx,on="target_day",how="left")
    raw_cols=[c for c in ctx.columns if c!="target_day"]
    # Same-slot previous-day spread is legally visible only for target hours <=14.
    out["d1_same_slot_spread"]=np.nan; out["d1_same_slot_sign"]=np.nan; out["d1_same_slot_abs"]=np.nan; out["d1_same_slot_visible"]=(out.hour_business<=14).astype(float)
    for p in range(1,15):
        m=out.hour_business.eq(p); c=f"d1_spread_p{p:02d}"; out.loc[m,"d1_same_slot_spread"]=out.loc[m,c]; out.loc[m,"d1_same_slot_sign"]=np.sign(out.loc[m,c]); out.loc[m,"d1_same_slot_abs"]=np.abs(out.loc[m,c])
    raw_cols += ["d1_same_slot_spread","d1_same_slot_sign","d1_same_slot_abs","d1_same_slot_visible"]
    return out,raw_cols,audit


def make_model(class_mode: str, seed:int):
    return lgb.LGBMClassifier(objective="binary",class_weight=("balanced" if class_mode=="balanced" else None),n_estimators=180,learning_rate=.04,num_leaves=31,min_child_samples=35,subsample=.9,colsample_bytree=.9,reg_lambda=1.0,random_state=seed,n_jobs=4,verbosity=-1)


def predict(slot:pd.DataFrame,features:list[str],target_days:list[str],training_days:int,class_mode:str,seed:int,name:str)->pd.DataFrame:
    all_days=sorted(slot.target_day.dropna().astype(str).unique()); rows=[]
    for i,day in enumerate(target_days,1):
        tr_days=strict_train_days(all_days,day,training_days); tr=slot[slot.target_day.isin(tr_days)]; te=slot[slot.target_day.eq(day)].sort_values("hour_business")
        y=(tr.target_spread.to_numpy(float)>0).astype(int); model=make_model(class_mode,seed); model.fit(tr[features],y); prob=model.predict_proba(te[features])[:,1]
        o=te[["target_day","hour_business","period","target_spread"]].copy(); o["variant"]=name; o["prob_positive"]=prob; o["predicted_direction"]=np.where(prob>=.5,1,-1); o["training_last_day"]=tr_days[-1]; rows.append(o)
        if i%50==0: print(f"{name}: {i}/{len(target_days)} {day}")
    return pd.concat(rows,ignore_index=True)


def metr(g:pd.DataFrame)->dict:
    y=np.sign(g.target_spread.to_numpy(float)); p=g.predicted_direction.to_numpy(int); ok=y==p; pos=y>0; neg=y<0; pa=float(ok[pos].mean()); na=float(ok[neg].mean())
    return {"days":g.target_day.nunique(),"n":len(g),"direction_accuracy":float(ok.mean()),"positive_accuracy":pa,"negative_accuracy":na,"balanced_direction_accuracy":.5*(pa+na),"all_negative_accuracy":float(neg.mean())}

def summarize(ledger:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    x=ledger.copy(); x["month"]=x.target_day.str[:7]; m=[]
    for (v,mo),g in x.groupby(["variant","month"]): m.append({"variant":v,"month":mo,**metr(g)})
    m=pd.DataFrame(m); r=[]
    for v,g in m.groupby("variant"):
        acc=g.direction_accuracy.astype(float); bal=g.balanced_direction_accuracy.astype(float); gain=acc-g.all_negative_accuracy.astype(float)
        r.append({"variant":v,"months":len(g),"mean_month_acc":acc.mean(),"median_month_acc":acc.median(),"min_month_acc":acc.min(),"max_month_acc":acc.max(),"std_month_acc":acc.std(ddof=0),"months_ge_065":int((acc>=.65).sum()),"months_ge_070":int((acc>=.70).sum()),"mean_month_bal":bal.mean(),"min_month_bal":bal.min(),"months_beating_all_negative":int((gain>0).sum()),"mean_gain_vs_all_negative":gain.mean()})
    return m,pd.DataFrame(r).sort_values(["mean_month_acc","mean_month_bal"],ascending=False)


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--cube-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube"); ap.add_argument("--raw-path",default="data/24/canonical/shandong_pmos_hourly.csv"); ap.add_argument("--start",default="2026-01-01"); ap.add_argument("--end",default="2026-08-14"); ap.add_argument("--output-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration6_visible_d1_context_strictD2"); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    cube=Path(args.cube_root); slot=pd.read_parquet(cube/"slot_table.parquet"); groups=json.loads((cube/"feature_groups.json").read_text(encoding="utf-8")); base=p6_features(groups); slot,ctx,audit=add_visible_context(slot,args.raw_path); slot,sd=add_similar_day_features(slot,groups,k_values=(20,),lookback_days=365); sd20=[c for c in sd["features"] if c.startswith("sd20_")]
    days=[d for d in sorted(slot.target_day.unique()) if args.start<=d<=args.end];
    if any("2026-08-15"<=d<="2026-08-21" for d in days): raise RuntimeError("final holdout touched")
    specs=[("P6_bal_w90",base,"balanced"),("P6_CTX_bal_w90",base+ctx,"balanced"),("P6_CTX_SD20_bal_w90",base+ctx+sd20,"balanced"),("P6_CTX_raw_w90",base+ctx,"raw")]
    led=[]
    for name,fs,cm in specs:
        fs=list(dict.fromkeys(fs)); print(f"START {name}: {len(fs)} features"); led.append(predict(slot,fs,days,90,cm,args.seed,name))
    ledger=pd.concat(led,ignore_index=True); monthly,robust=summarize(ledger); out=Path(args.output_root); atomic_parquet(out/"ledger.parquet",ledger); atomic_csv(out/"monthly.csv",monthly); atomic_csv(out/"robustness.csv",robust); atomic_csv(out/"visible_context_causal_audit.csv",audit); atomic_json(out/"manifest.json",{"status":"complete","forecast_origin":"D-1 14:00","training_labels":"D-2 and earlier only","visible_context":"D-1 p1-p14 raw spread only","final_holdout_touched":False,"fresh_final_holdout_reserved":["2026-08-15","2026-08-21"],"literature_basis":["Shandong lagged spread dominance","DART 24h/48h lagged spread features under gate closure"]}); print("\nROBUSTNESS\n",robust.to_string(index=False)); top=robust.iloc[0].variant; print("\nTOP MONTHLY\n",monthly[monthly.variant.eq(top)].to_string(index=False))

if __name__=="__main__": main()
