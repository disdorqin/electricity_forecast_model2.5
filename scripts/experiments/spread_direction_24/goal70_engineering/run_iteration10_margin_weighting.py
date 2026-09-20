"""Iteration 10: strict margin-aware / class-prior weighting for P6+SD20 direction.

Motivation:
- DART/spread literature separates regular and spike regimes; signs near zero are noisier than
  economically material spreads.
- The strict P6 route is also sensitive to positive/negative class weighting. Rather than choosing
  raw vs fully balanced globally, this screen uses fixed, train-only weighting rules.

Hard contract:
- target D forecast origin D-1 14:00;
- supervised labels for fit are D-2 or earlier only;
- Similar-Day candidates are <= D-2;
- all magnitude quantiles/class ratios are computed inside each strict training window only;
- fresh 2026-08-15..08-21 holdout remains sealed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

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


@dataclass(frozen=True)
class Spec:
    name: str
    class_power: float
    trim_quantile: float = 0.0
    magnitude_mode: str = "none"


def model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=0.04, num_leaves=31,
        min_child_samples=35, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, n_jobs=4, verbosity=-1,
    )


def weights_and_mask(y_spread: np.ndarray, spec: Spec) -> tuple[np.ndarray, np.ndarray, dict]:
    y = (y_spread > 0).astype(int)
    pos = max(1, int((y == 1).sum())); neg = max(1, int((y == 0).sum()))
    ratio = (neg / pos) ** spec.class_power
    w = np.where(y == 1, ratio, 1.0).astype(float)
    mag = np.abs(y_spread.astype(float))
    mask = np.isfinite(mag)
    trim_threshold = 0.0
    if spec.trim_quantile > 0:
        trim_threshold = float(np.nanquantile(mag[mask], spec.trim_quantile))
        mask &= mag >= trim_threshold
    if spec.magnitude_mode == "soft":
        q75 = max(float(np.nanquantile(mag[np.isfinite(mag)], 0.75)), 1e-6)
        factor = 0.65 + 0.70 * np.minimum(mag / q75, 1.5) / 1.5
        w *= factor
    elif spec.magnitude_mode == "sqrt":
        q75 = max(float(np.nanquantile(mag[np.isfinite(mag)], 0.75)), 1e-6)
        factor = 0.65 + 0.70 * np.sqrt(np.minimum(mag / q75, 2.0) / 2.0)
        w *= factor
    elif spec.magnitude_mode == "tail":
        q75 = float(np.nanquantile(mag[np.isfinite(mag)], 0.75))
        w *= np.where(mag >= q75, 1.35, 0.90)
    return w, mask, {"pos_weight": ratio, "trim_threshold": trim_threshold, "kept_fraction": float(mask.mean())}


def predict(slot: pd.DataFrame, features: list[str], days: list[str], spec: Spec, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_days = sorted(slot.target_day.dropna().astype(str).unique())
    rows = []; audits = []
    for i, day in enumerate(days, 1):
        tr_days = strict_train_days(all_days, day, 90)
        tr = slot[slot.target_day.isin(tr_days)].copy()
        te = slot[slot.target_day.eq(day)].sort_values("hour_business").copy()
        y_sp = tr.target_spread.to_numpy(float)
        w, mask, info = weights_and_mask(y_sp, spec)
        if int(mask.sum()) < 1000:
            raise RuntimeError(f"{day} {spec.name}: too few train rows {mask.sum()}")
        m = model(seed)
        m.fit(tr.loc[mask, features], (y_sp[mask] > 0).astype(int), sample_weight=w[mask])
        p = m.predict_proba(te[features])[:, 1].astype(float)
        o = te[["target_day", "hour_business", "period", "target_spread"]].copy()
        o["variant"] = spec.name; o["prob_positive"] = p; o["predicted_direction"] = np.where(p >= .5, 1, -1)
        o["training_last_day"] = tr_days[-1]; o["training_days"] = len(tr_days)
        rows.append(o)
        audits.append({"target_day": day, "variant": spec.name, "training_last_day": tr_days[-1], **info})
        if i % 40 == 0:
            print(f"{spec.name}: {i}/{len(days)} {day}", flush=True)
    audit = pd.DataFrame(audits)
    required = (pd.to_datetime(audit.target_day) - pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")
    if not audit.training_last_day.le(required).all():
        raise RuntimeError(f"strict label audit failed for {spec.name}")
    return pd.concat(rows, ignore_index=True), audit


def summarize(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = ledger.copy(); z["month"] = z.target_day.str[:7]
    mm=[]
    for (v,m),g in z.groupby(["variant","month"],sort=True):
        mm.append({"variant":v,"month":m,"days":g.target_day.nunique(),**direction_metrics(g.target_spread,g.predicted_direction)})
    monthly=pd.DataFrame(mm); rr=[]
    for v,g in monthly.groupby("variant",sort=False):
        acc=g.direction_accuracy.astype(float); bal=g.balanced_direction_accuracy.astype(float); gain=acc-g.all_negative_accuracy.astype(float)
        rr.append({"variant":v,"months":len(g),"mean_month_acc":acc.mean(),"median_month_acc":acc.median(),"min_month_acc":acc.min(),"max_month_acc":acc.max(),"std_month_acc":acc.std(ddof=0),"months_ge_065":int((acc>=.65).sum()),"months_ge_070":int((acc>=.70).sum()),"mean_month_bal":bal.mean(),"min_month_bal":bal.min(),"months_beating_all_negative":int((gain>0).sum()),"mean_gain_vs_all_negative":gain.mean()})
    return monthly,pd.DataFrame(rr).sort_values(["mean_month_acc","mean_month_bal"],ascending=False)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--cube-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube"); ap.add_argument("--start",default="2026-04-01"); ap.add_argument("--end",default="2026-08-14"); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--output-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration10_margin_weighting_strictD2"); args=ap.parse_args()
    if args.end >= "2026-08-15": raise RuntimeError("fresh final holdout remains sealed")
    cube=Path(args.cube_root); slot=pd.read_parquet(cube/"slot_table.parquet"); groups=json.loads((cube/"feature_groups.json").read_text(encoding="utf-8")); base=p6_features(groups)
    slot,sd=add_similar_day_features(slot,groups,k_values=(20,),lookback_days=365); sd20=[c for c in sd["features"] if c.startswith("sd20_")]; features=dedupe(base+sd20)
    if sd["audit"].empty or not sd["audit"].causal_ok.all(): raise RuntimeError("similar-day causal audit failed")
    days=[str(d) for d in sorted(slot.target_day.dropna().astype(str).unique()) if args.start<=str(d)<=args.end]
    specs=[
        Spec("SD20_cw0.00",0.00),
        Spec("SD20_cw0.25",0.25),
        Spec("SD20_cw0.50",0.50),
        Spec("SD20_cw0.75",0.75),
        Spec("SD20_cw1.00",1.00),
        Spec("SD20_trim10_cw0.50",0.50,0.10),
        Spec("SD20_trim25_cw0.50",0.50,0.25),
        Spec("SD20_magsoft_cw0.50",0.50,0.0,"soft"),
        Spec("SD20_magsqrt_cw0.50",0.50,0.0,"sqrt"),
        Spec("SD20_tail_cw0.50",0.50,0.0,"tail"),
    ]
    led=[]; audits=[]
    for s in specs:
        print("START",s.name,flush=True); z,a=predict(slot,features,days,s,args.seed); led.append(z); audits.append(a)
    ledger=pd.concat(led,ignore_index=True); audit=pd.concat(audits,ignore_index=True); monthly,robust=summarize(ledger); out=Path(args.output_root); atomic_parquet(out/"ledger.parquet",ledger); atomic_csv(out/"monthly.csv",monthly); atomic_csv(out/"robustness.csv",robust); atomic_csv(out/"training_audit.csv",audit); atomic_csv(out/"similar_day_causal_audit.csv",sd["audit"]); atomic_json(out/"manifest.json",{"status":"complete","experiment":"iteration10_margin_weighting","forecast_origin":"D-1 14:00","training_labels":"D-2 and earlier only","all_class_and_magnitude_statistics":"train-window only","target_day_DA_as_feature":False,"target_day_actual_as_feature":False,"fresh_final_holdout_reserved":["2026-08-15","2026-08-21"],"final_holdout_touched":False,"literature_basis":["DART regular/spike regime separation","cost-sensitive classification under imbalanced direction labels"]}); print("\nROBUSTNESS\n",robust.to_string(index=False)); top=robust.iloc[0].variant; print("\nTOP MONTHLY\n",monthly[monthly.variant.eq(top)].to_string(index=False))

if __name__=="__main__": main()
