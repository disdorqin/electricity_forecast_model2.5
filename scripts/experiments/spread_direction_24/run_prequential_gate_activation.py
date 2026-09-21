"""Causally activate a strict residual gate only after recent OOS uplift.

Reads a pre-existing strict gate ledger.  For target D it compares the gate
and baseline only on completed OOS days through D-2, then either uses the
whole gate output or falls back to baseline.  It never recalibrates against
the current target label.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")

def score(y,p):
    pos=y==1;neg=~pos
    pr=float((p[pos]==1).mean());nr=float((p[neg]==-1).mean())
    return {"n":len(y),"direction_accuracy":float((p==y).mean()),"positive_recall":pr,"negative_recall":nr,"balanced_accuracy":(pr+nr)/2,"all_negative_baseline":float(neg.mean())}

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--lookback-days',type=int,default=14);a.add_argument('--min-uplift',type=float,default=.005);args=a.parse_args()
    p=pd.read_csv(args.input.resolve());p['target_day']=pd.to_datetime(p.target_day).dt.normalize()
    needed={'target_day','hour_business','y_true','baseline_pred','predicted_direction','training_last_day'}
    if needed-set(p):raise RuntimeError('missing gate fields')
    if p.target_day.max()>=FINAL_HOLDOUT_START:raise RuntimeError('input touches final holdout')
    p['training_last_day']=pd.to_datetime(p.training_last_day).dt.normalize()
    if not (p.training_last_day<=p.target_day-pd.Timedelta(days=2)).all():raise RuntimeError('input strict boundary fails')
    rows=[];audit=[];days=sorted(p.target_day.unique())
    for d in days:
        hist_days=[x for x in days if x<=d-pd.Timedelta(days=2)][-args.lookback_days:]
        h=p[p.target_day.isin(hist_days)]; uplift=float((h.predicted_direction==h.y_true).mean()-(h.baseline_pred==h.y_true).mean()) if len(h) else 0.
        active=bool(uplift>=args.min_uplift);q=p[p.target_day.eq(d)].copy();q['activated_prediction']=np.where(active,q.predicted_direction,q.baseline_pred);q['gate_active']=active;q['historical_uplift']=uplift;rows.append(q);audit.append({'target_day':d.strftime('%Y-%m-%d'),'training_last_day':max(hist_days).strftime('%Y-%m-%d') if hist_days else None,'strict_ok':not hist_days or max(hist_days)<=d-pd.Timedelta(days=2),'historical_uplift':uplift,'active':active})
    o=pd.concat(rows,ignore_index=True);au=pd.DataFrame(audit)
    summary=[]
    for name,col in [('baseline','baseline_pred'),('prequential_gate_activation','activated_prediction')]:summary.append({'variant':name,**score(o.y_true.to_numpy(),o[col].to_numpy())})
    o['month']=o.target_day.dt.strftime('%Y-%m');monthly=[]
    for m,g in o.groupby('month'):
        for name,col in [('baseline','baseline_pred'),('prequential_gate_activation','activated_prediction')]:monthly.append({'month':m,'variant':name,**score(g.y_true.to_numpy(),g[col].to_numpy())})
    dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True);o.to_csv(dest/'predictions.csv',index=False,encoding='utf-8-sig');au.to_csv(dest/'activation_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(summary).to_csv(dest/'summary.csv',index=False,encoding='utf-8-sig');pd.DataFrame(monthly).to_csv(dest/'monthly.csv',index=False,encoding='utf-8-sig')
    (dest/'manifest.json').write_text(json.dumps({'status':'STRICT/PASS','route':'A_prequential_gate_activation','forecast_origin':'D-1 14:00','training_last_day':'gate input and activation history <= D-2','target_day_actual_as_feature':False,'target_day_DA_as_feature':False,'d1_post14_spread_as_feature':False,'final_holdout_touched':False,'lookback_days':args.lookback_days,'min_uplift':args.min_uplift},ensure_ascii=False,indent=2),encoding='utf8');print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(monthly).to_string(index=False))
if __name__=='__main__':main()
