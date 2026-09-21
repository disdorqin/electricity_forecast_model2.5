"""Iteration 5: covariate-conditioned daily/period regime prior for strict-D2 spread direction.

Motivation: DART/spread literature models regime frequency as covariate-dependent.
Here a lightweight daily regime model predicts the positive-spread fraction for each
8-hour period from the target day's cutoff-safe P6 feature profile. The prior is then
used only as a post-processing constraint on already strict OOS P6+SD20 probabilities.

For target D, the regime model is trained on completed days D-2 and earlier only.
The final holdout 2026-08-15..2026-08-21 is never loaded.
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

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import p6_features

PERIODS = ["1_8", "9_16", "17_24"]


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def strict_days(all_days: list[str], target: str, n: int) -> list[str]:
    idx = all_days.index(target)
    end = idx - 1  # exclusive => last included D-2
    start = max(0, end - n)
    out = all_days[start:end]
    if len(out) < min(120, n):
        raise ValueError(f"{target}: only {len(out)} strict days")
    return out


def build_daily(slot: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows=[]
    agg_names=[]
    for f in features:
        for stat in ("mean","std","min","max"):
            agg_names.append(f"{f}__{stat}")
    for day,g in slot.groupby("target_day", sort=True):
        g=g.sort_values("hour_business")
        if len(g)!=24 or g.target_spread.isna().any():
            continue
        rec={"target_day":str(day)}
        X=g[features].apply(pd.to_numeric,errors="coerce")
        for f in features:
            s=X[f].to_numpy(float)
            rec[f"{f}__mean"]=float(np.nanmean(s))
            rec[f"{f}__std"]=float(np.nanstd(s))
            rec[f"{f}__min"]=float(np.nanmin(s)) if np.isfinite(s).any() else math.nan
            rec[f"{f}__max"]=float(np.nanmax(s)) if np.isfinite(s).any() else math.nan
        for p in PERIODS:
            gg=g[g.period.eq(p)]
            rec[f"target_posrate_{p}"]=float((gg.target_spread.to_numpy(float)>0).mean())
        rows.append(rec)
    return pd.DataFrame(rows), agg_names


def load_strict_probs(base: str) -> pd.DataFrame:
    roots=[
        "cross_month_champion_audit_strictD2_q1",
        "cross_month_champion_audit_q2_strictD2",
        "cross_month_champion_audit_q3_strictD2",
    ]
    frames=[]
    for r in roots:
        frames.append(pd.read_parquet(f"{base}/{r}/ledger.parquet"))
    x=pd.concat(frames,ignore_index=True).drop_duplicates(["target_day","hour_business","variant"],keep="last")
    x=x[x.variant.eq("P6_SD20_w90")].copy()
    return x.sort_values(["target_day","hour_business"]).reset_index(drop=True)


def regime_predictions(daily: pd.DataFrame, agg_cols: list[str], target_days: list[str], window: int, seed: int) -> pd.DataFrame:
    all_days=sorted(daily.target_day.unique())
    rows=[]
    for i,day in enumerate(target_days,1):
        tr_days=strict_days(all_days,day,window)
        tr=daily[daily.target_day.isin(tr_days)]
        te=daily[daily.target_day.eq(day)]
        # Stack period targets into one lightweight model with period one-hot features.
        Xs=[]; ys=[]
        for pidx,p in enumerate(PERIODS):
            xx=tr[agg_cols].copy()
            for j,pp in enumerate(PERIODS): xx[f"period_{pp}"]=float(j==pidx)
            Xs.append(xx); ys.append(tr[f"target_posrate_{p}"].to_numpy(float))
        Xtr=pd.concat(Xs,ignore_index=True); ytr=np.concatenate(ys)
        model=lgb.LGBMRegressor(objective="regression_l1",n_estimators=140,learning_rate=.04,num_leaves=15,max_depth=5,min_child_samples=25,subsample=.9,colsample_bytree=.7,reg_lambda=2.0,random_state=seed,n_jobs=4,verbosity=-1)
        model.fit(Xtr,ytr)
        rec={"target_day":day,"training_last_day":tr_days[-1],"window":window}
        for pidx,p in enumerate(PERIODS):
            xx=te[agg_cols].copy()
            for j,pp in enumerate(PERIODS): xx[f"period_{pp}"]=float(j==pidx)
            rec[f"prior_{p}"]=float(np.clip(model.predict(xx)[0],.02,.98))
        rows.append(rec)
        if i%50==0: print(f"regime w{window}: {i}/{len(target_days)} {day}")
    return pd.DataFrame(rows)


def logit(x: np.ndarray) -> np.ndarray:
    x=np.clip(x,1e-5,1-1e-5); return np.log(x/(1-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-np.clip(x,-30,30)))


def shift_to_mean(p: np.ndarray, target: float) -> np.ndarray:
    z=logit(p); lo,hi=-10.,10.
    for _ in range(50):
        mid=(lo+hi)/2
        if sigmoid(z+mid).mean()<target: lo=mid
        else: hi=mid
    return sigmoid(z+(lo+hi)/2)


def apply_prior(oos: pd.DataFrame, priors: pd.DataFrame, window: int) -> pd.DataFrame:
    x=oos.merge(priors,on="target_day",how="inner")
    variants=[]
    # reference
    ref=x[["target_day","hour_business","period","target_spread"]].copy(); ref["variant"]="sd20_static"; ref["prob_positive"]=x.prob_positive; ref["predicted_direction"]=np.where(ref.prob_positive>=.5,1,-1); variants.append(ref)
    for alpha in (.25,.5,.75,1.0):
        rows=[]
        for day,g in x.groupby("target_day",sort=True):
            g=g.sort_values("hour_business")
            out=g[["target_day","hour_business","period","target_spread"]].copy(); out["prob_positive"]=np.nan
            for p in PERIODS:
                mask=g.period.eq(p); raw=g.loc[mask,"prob_positive"].to_numpy(float); prior=float(g[f"prior_{p}"].iloc[0]); target=(1-alpha)*float(raw.mean())+alpha*prior; out.loc[mask,"prob_positive"]=shift_to_mean(raw,target)
            out["variant"]=f"prior_logit_w{window}_a{alpha:.2f}"; out["predicted_direction"]=np.where(out.prob_positive>=.5,1,-1); rows.append(out)
        variants.append(pd.concat(rows,ignore_index=True))
    # rank-constrained positive count per period; direct expression of regime-frequency prior.
    for alpha in (.5,.75,1.0):
        rows=[]
        for day,g in x.groupby("target_day",sort=True):
            g=g.sort_values("hour_business"); out=g[["target_day","hour_business","period","target_spread"]].copy(); out["prob_positive"]=g.prob_positive.to_numpy(float); pred=-np.ones(len(g),int)
            for p in PERIODS:
                loc=np.flatnonzero(g.period.to_numpy()==p); raw=g.iloc[loc].prob_positive.to_numpy(float); prior=float(g[f"prior_{p}"].iloc[0]); blended=(1-alpha)*float((raw>=.5).mean())+alpha*prior; k=int(np.clip(round(8*blended),0,8)); order=np.argsort(-raw); pred[loc[order[:k]]]=1
            out["variant"]=f"prior_rank_w{window}_a{alpha:.2f}"; out["predicted_direction"]=pred; rows.append(out)
        variants.append(pd.concat(rows,ignore_index=True))
    return pd.concat(variants,ignore_index=True)


def metr(g: pd.DataFrame) -> dict:
    y=np.sign(g.target_spread.to_numpy(float)); p=g.predicted_direction.to_numpy(int); ok=y==p; pos=y>0; neg=y<0; pa=float(ok[pos].mean()); na=float(ok[neg].mean())
    return {"days":g.target_day.nunique(),"n":len(g),"direction_accuracy":float(ok.mean()),"positive_accuracy":pa,"negative_accuracy":na,"balanced_direction_accuracy":.5*(pa+na),"all_negative_accuracy":float(neg.mean())}


def summarize(ledger: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    x=ledger.copy(); x["month"]=x.target_day.str[:7]; m=[]
    for (v,mo),g in x.groupby(["variant","month"]): m.append({"variant":v,"month":mo,**metr(g)})
    m=pd.DataFrame(m); r=[]
    for v,g in m.groupby("variant"):
        acc=g.direction_accuracy.astype(float); bal=g.balanced_direction_accuracy.astype(float); gain=acc-g.all_negative_accuracy.astype(float)
        r.append({"variant":v,"months":len(g),"mean_month_acc":acc.mean(),"median_month_acc":acc.median(),"min_month_acc":acc.min(),"max_month_acc":acc.max(),"std_month_acc":acc.std(ddof=0),"months_ge_065":int((acc>=.65).sum()),"months_ge_070":int((acc>=.70).sum()),"mean_month_bal":bal.mean(),"min_month_bal":bal.min(),"months_beating_all_negative":int((gain>0).sum()),"mean_gain_vs_all_negative":gain.mean()})
    return m,pd.DataFrame(r).sort_values(["mean_month_acc","mean_month_bal"],ascending=False)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--cube-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube"); ap.add_argument("--base-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822"); ap.add_argument("--output-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration5_day_regime_prior_strictD2"); ap.add_argument("--seed",type=int,default=42); args=ap.parse_args()
    cube=Path(args.cube_root); slot=pd.read_parquet(cube/"slot_table.parquet"); groups=json.loads((cube/"feature_groups.json").read_text(encoding="utf-8")); feats=p6_features(groups); daily,agg=build_daily(slot,feats); oos=load_strict_probs(args.base_root); target_days=sorted(oos.target_day.unique())
    ledgers=[]; audits=[]
    for w in (180,365):
        pr=regime_predictions(daily,agg,target_days,w,args.seed); audits.append(pr); ledgers.append(apply_prior(oos,pr,w))
    ledger=pd.concat(ledgers,ignore_index=True).drop_duplicates(["target_day","hour_business","variant"],keep="last"); monthly,robust=summarize(ledger); out=Path(args.output_root); atomic_parquet(out/"ledger.parquet",ledger); atomic_csv(out/"monthly.csv",monthly); atomic_csv(out/"robustness.csv",robust); atomic_csv(out/"prior_audit.csv",pd.concat(audits,ignore_index=True)); atomic_json(out/"manifest.json",{"status":"complete","forecast_origin":"D-1 14:00","training_labels":"D-2 and earlier only","final_holdout_touched":False,"fresh_final_holdout_reserved":["2026-08-15","2026-08-21"],"literature_idea":"covariate-dependent spread regime frequency / hierarchical daily-period prior"}); print("\nROBUSTNESS TOP\n",robust.head(15).to_string(index=False)); top=robust.iloc[0].variant; print("\nTOP MONTHLY\n",monthly[monthly.variant.eq(top)].to_string(index=False))

if __name__=="__main__": main()
