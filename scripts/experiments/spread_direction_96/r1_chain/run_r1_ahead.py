from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = ROOT / "data/96/authoritative/pmos_96_全量.csv"

FORECAST_COLS = [
    "直调负荷预测", "地方电厂出力预测", "外电预测", "风电预测", "光伏预测", "核电预测",
    "自备电厂预测", "试验机组预测", "正备用预测", "负备用预测",
]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); tmp.replace(path)

def atomic_csv(path: Path, f: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+".tmp"); f.to_csv(tmp,index=False,encoding="utf-8-sig"); tmp.replace(path)

def atomic_parquet(path: Path, f: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+".tmp"); f.to_parquet(tmp,index=False); tmp.replace(path)


def parse_slot(s: pd.Series) -> pd.Series:
    hhmm=s.astype(str).str.extract(r"^(\d{1,2}):(\d{2})$").astype(float)
    minute=(hhmm[0]*60+hhmm[1]).astype(int)
    # 00:15->1 ... 24:00->96
    return (minute//15).astype(int)


def load_data(path: Path) -> pd.DataFrame:
    raw=pd.read_csv(path)
    raw["market_date"]=raw["market_date"].astype(str)
    raw["slot"]=parse_slot(raw["时段"])
    raw=raw.sort_values(["market_date","slot"]).reset_index(drop=True)
    for c in ["日前出清价格","实时出清价格",*FORECAST_COLS]:
        if c in raw: raw[c]=pd.to_numeric(raw[c],errors="coerce")
    raw["spread"]=raw["实时出清价格"]-raw["日前出清价格"]
    return raw


def daily_context(raw: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    by={d:g.sort_values("slot") for d,g in raw.groupby("market_date",sort=True)}
    for day in sorted(by):
        prev=(pd.Timestamp(day)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if prev not in by: continue
        vals=by[prev].loc[by[prev]["slot"]<=56,"spread"].to_numpy(float)
        vals=vals[np.isfinite(vals)]
        if len(vals)<40: continue
        rows.append({
            "market_date":day,
            "ctx_mean56":float(vals.mean()),"ctx_std56":float(vals.std()),"ctx_median56":float(np.median(vals)),
            "ctx_last":float(vals[-1]),"ctx_mean8":float(vals[-8:].mean()),"ctx_min56":float(vals.min()),"ctx_max56":float(vals.max()),
            "ctx_range56":float(vals.max()-vals.min()),"ctx_pos_rate56":float((vals>0).mean()),"ctx_neg_rate56":float((vals<0).mean()),
            "ctx_slope56":float(np.polyfit(np.arange(len(vals)),vals,1)[0]),
        })
    return pd.DataFrame(rows)


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame,list[str]]:
    x=raw.copy()
    # Same-slot safe history. Full D-1 is only available for slots <=56; later target slots fall back to D-2.
    g=x.groupby("slot",sort=False)["spread"]
    lag1=g.shift(1); lag2=g.shift(2)
    x["spread_mixed_lag"]=np.where(x["slot"]<=56,lag1,lag2)
    x["spread_lag2d"]=lag2
    x["spread_lag3d"]=g.shift(3)
    x["spread_lag7d"]=g.shift(7)
    # Safe same-slot historical statistics: shift 2 days so every target slot respects D-1 14:00.
    shifted=g.shift(2)
    # groupby rolling returns multi-index; restore row order.
    for w in (7,28):
        x[f"spread_mean{w}d"]=(shifted.groupby(x["slot"]).rolling(w,min_periods=3).mean().reset_index(level=0,drop=True).sort_index())
        x[f"spread_std{w}d"]=(shifted.groupby(x["slot"]).rolling(w,min_periods=3).std(ddof=0).reset_index(level=0,drop=True).sort_index())
    ctx=daily_context(x)
    x=x.merge(ctx,on="market_date",how="left",validate="many_to_one")
    slot=x["slot"].astype(float); dates=pd.to_datetime(x["market_date"])
    x["slot_sin"]=np.sin(2*np.pi*(slot-1)/96); x["slot_cos"]=np.cos(2*np.pi*(slot-1)/96)
    x["dow_sin"]=np.sin(2*np.pi*dates.dt.dayofweek/7); x["dow_cos"]=np.cos(2*np.pi*dates.dt.dayofweek/7)
    feats=["spread_mixed_lag","spread_lag2d","spread_lag3d","spread_lag7d","spread_mean7d","spread_std7d","spread_mean28d","spread_std28d",
           "ctx_mean56","ctx_std56","ctx_median56","ctx_last","ctx_mean8","ctx_min56","ctx_max56","ctx_range56","ctx_pos_rate56","ctx_neg_rate56","ctx_slope56",
           "slot_sin","slot_cos","dow_sin","dow_cos"]
    for c in FORECAST_COLS:
        if c not in x: continue
        alias="f_"+c; x[alias]=x[c]; feats.append(alias)
        ramp="ramp_"+c; x[ramp]=x.groupby("market_date",sort=False)[c].diff().fillna(0.0); feats.append(ramp)
    # Forecast physical relationships only.
    if all(c in x for c in ["直调负荷预测","风电预测","光伏预测"]):
        x["f_residual_load_ws"]=x["直调负荷预测"]-x["风电预测"]-x["光伏预测"]; feats.append("f_residual_load_ws")
    return x,feats


def clf(seed=42):
    return lgb.LGBMClassifier(objective="binary",class_weight="balanced",n_estimators=220,learning_rate=.035,num_leaves=31,min_child_samples=50,subsample=.9,colsample_bytree=.9,reg_lambda=2.0,verbosity=-1,n_jobs=4,random_state=seed)

def qreg(tau=.5,seed=42):
    return lgb.LGBMRegressor(objective="quantile",alpha=tau,n_estimators=220,learning_rate=.035,num_leaves=31,min_child_samples=50,subsample=.9,colsample_bytree=.9,reg_lambda=2.0,verbosity=-1,n_jobs=4,random_state=seed)


def metrics(y,pdir):
    yt=np.sign(np.asarray(y,float)); yp=np.asarray(pdir,int); m=yt!=0; pos=yt>0; neg=yt<0; c=yt==yp
    pa=float(c[pos].mean()) if pos.any() else math.nan; na=float(c[neg].mean()) if neg.any() else math.nan
    return {"n_nonzero":int(m.sum()),"direction_accuracy":float(c[m].mean()),"positive_accuracy":pa,"negative_accuracy":na,"balanced_direction_accuracy":float(np.nanmean([pa,na]))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default=str(DEFAULT_DATA.relative_to(ROOT))); ap.add_argument("--train-start",required=True); ap.add_argument("--train-end",required=True); ap.add_argument("--test-start",required=True); ap.add_argument("--test-end",required=True); ap.add_argument("--output",required=True); ap.add_argument("--seed",type=int,default=42); a=ap.parse_args()
    if a.train_end >= a.test_start:
        raise ValueError(f"train/test overlap or non-forward split: train_end={a.train_end}, test_start={a.test_start}")
    frame,feats=build_features(load_data(ROOT/a.data)); train=frame[(frame.market_date>=a.train_start)&(frame.market_date<=a.train_end)].dropna(subset=["spread"]).copy(); test=frame[(frame.market_date>=a.test_start)&(frame.market_date<=a.test_end)].dropna(subset=["spread"]).copy()
    test_counts = test.groupby("market_date")["slot"].agg(["size", "nunique"])
    bad_test_days = test_counts[(test_counts["size"] != 96) | (test_counts["nunique"] != 96)]
    if not bad_test_days.empty:
        raise RuntimeError(f"incomplete 96-point test days: {bad_test_days.head().to_dict('index')}")
    y=train.spread.to_numpy(float); yc=(y>0).astype(int)
    mclf=clf(a.seed).fit(train[feats],yc); p=mclf.predict_proba(test[feats])[:,1]
    mq=qreg(.5,a.seed+11).fit(train[feats],y); q50=mq.predict(test[feats])
    out=test[["market_date","slot","spread"]].copy(); out["direction_prob"]=p; out["q50"]=q50; out["pred_direction_cls"]=np.where(p>=.5,1,-1); out["pred_direction_q50"]=np.where(q50>=0,1,-1)
    s=pd.DataFrame([{"model":"direction_classifier",**metrics(out.spread,out.pred_direction_cls)},{"model":"q50_regression",**metrics(out.spread,out.pred_direction_q50)}])
    monthly=[]
    for mo,g in out.assign(month=out.market_date.str[:7]).groupby("month"):
        monthly.append({"month":mo,"model":"direction_classifier",**metrics(g.spread,g.pred_direction_cls)}); monthly.append({"month":mo,"model":"q50_regression",**metrics(g.spread,g.pred_direction_q50)})
    missing_rates={c:float(test[c].isna().mean()) for c in feats if test[c].isna().any()}
    od=ROOT/a.output; atomic_parquet(od/"predictions.parquet",out); atomic_csv(od/"summary.csv",s); atomic_csv(od/"monthly_metrics.csv",pd.DataFrame(monthly)); atomic_json(od/"manifest.json",{"status":"complete","experiment":"r1_inspired_96_ahead","deployable_source":True,"information_boundary_audit":"counterfactual audit required/passed separately before acceptance","forecast_origin":"D-1 14:00","target":"D 96-point RT-DA spread","target_day_actual_features":False,"target_day_da_input":False,"dminus1_post14_realized_input":False,"mixed_lag_contract":"slots1-56 use D-1 same-slot; slots57-96 use D-2 same-slot","test_day_completeness":"96/96 required","missing_features_handling":"LightGBM native NaN; rows are not dropped for feature NaN","test_feature_missing_rates":missing_rates,"train":[a.train_start,a.train_end],"test":[a.test_start,a.test_end],"n_features":len(feats),"features":feats,"production_chain_touched":False})
    print(s.to_string(index=False)); print(pd.DataFrame(monthly).to_string(index=False))
if __name__=="__main__": main()
