"""Strict D-2 daily-event gate layered on the retained SD20 baseline.

Unlike a slot-only override, this pilot first asks whether tomorrow is a
candidate-positive *day*.  It uses only target-day forecasts, D-1 p1-p14
context and already-produced strict OOS candidate probabilities.  Historical
daily labels, calibration, and fitting are all cut at D-2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pos = y == 1; neg = ~pos
    pr = float((p[pos] == 1).mean()) if pos.any() else float("nan")
    nr = float((p[neg] == -1).mean()) if neg.any() else float("nan")
    return {"n": int(len(y)), "direction_accuracy": float((y == p).mean()),
            "positive_recall": pr, "negative_recall": nr,
            "balanced_accuracy": float(np.mean([pr, nr])),
            "all_negative_baseline": float(neg.mean())}


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="binary", class_weight="balanced",
        n_estimators=100, learning_rate=0.035, num_leaves=7, max_depth=3,
        min_child_samples=12, reg_lambda=10.0, random_state=seed,
        n_jobs=4, verbosity=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--cube", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--baseline", default="SD20_static")
    ap.add_argument("--candidate", default="B_meta_stack")
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--history-days", type=int, default=120)
    ap.add_argument("--min-history-days", type=int, default=60)
    ap.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    ap.add_argument("--feature-groups", default="F1,F2,F3,F4,F7")
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    ledger = pd.read_parquet(args.source.resolve())
    ledger["target_day"] = pd.to_datetime(ledger["target_day"]).dt.normalize()
    if ledger.target_day.max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("source touches final holdout")
    need = {"target_day", "hour_business", "target_spread", "variant", "predicted_direction", "prob_positive"}
    if need - set(ledger.columns): raise RuntimeError("source lacks strict OOS fields")
    x = ledger[ledger.variant.isin([args.baseline, args.candidate])].copy()
    pred = x.pivot_table(index=["target_day","hour_business"], columns="variant", values="predicted_direction", aggfunc="first")
    prob = x.pivot_table(index=["target_day","hour_business"], columns="variant", values="prob_positive", aggfunc="first")
    y = x.drop_duplicates(["target_day","hour_business"])[["target_day","hour_business","target_spread"]].set_index(["target_day","hour_business"])
    z = pred.rename(columns={args.baseline:"base", args.candidate:"cand"}).join(prob.rename(columns={args.baseline:"base_p",args.candidate:"cand_p"})).join(y).reset_index()
    z = z.dropna(subset=["base","cand","target_spread"])
    z["y"] = np.where(z.target_spread > 0, 1, -1)
    z["eligible"] = (z.base == -1) & (z.cand == 1)
    z["benefit"] = np.where(z.y == 1, 1, -1) * z.eligible.astype(int)

    slot = pd.read_parquet(args.cube.resolve() / "slot_table.parquet")
    slot["target_day"] = pd.to_datetime(slot.target_day).dt.normalize()
    groups = json.loads((args.cube.resolve() / "feature_groups.json").read_text(encoding="utf-8"))
    cols = [c for g in args.feature_groups.split(",") for c in groups[g.strip()] if c in slot.columns]
    # Day aggregates preserve target forecast shape while keeping the daily
    # classifier sufficiently low dimensional for a rolling 60-120 day fit.
    agg = slot[["target_day", *cols]].groupby("target_day").agg(["mean", "std", "min", "max"])
    agg.columns = [f"{a}__{b}" for a,b in agg.columns]
    daily = z.groupby("target_day").agg(
        day_net=("benefit","sum"), eligible_count=("eligible","sum"),
        cand_pos_count=("cand", lambda s: int((s == 1).sum())),
        base_pos_count=("base", lambda s: int((s == 1).sum())),
        cand_prob_mean=("cand_p","mean"), cand_prob_std=("cand_p","std"),
        cand_prob_max=("cand_p","max"), cand_prob_min=("cand_p","min"),
    ).join(agg, how="left").reset_index()
    daily["day_good"] = (daily.day_net > 0).astype(int)
    feature_cols = [c for c in daily.columns if c not in {"target_day","day_net","day_good"}]
    daily[feature_cols] = daily[feature_cols].apply(pd.to_numeric, errors="coerce")
    thresholds = [float(v) for v in args.thresholds.split(",")]
    output=[]; audit=[]
    all_days = sorted(daily.target_day.unique())
    for day in [d for d in all_days if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]:
        cutoff = day - pd.Timedelta(days=2)
        hist = daily[daily.target_day.le(cutoff)].tail(args.history_days).copy()
        q = daily[daily.target_day.eq(day)].copy()
        if len(hist) < args.min_history_days or hist.day_good.nunique() < 2: continue
        split = max(1, int(len(hist) * .70)); fit, cal = hist.iloc[:split], hist.iloc[split:]
        med = fit[feature_cols].median().fillna(0)
        m = make_model(args.seed); m.fit(fit[feature_cols].fillna(med), fit.day_good)
        cal_p = m.predict_proba(cal[feature_cols].fillna(med))[:,1]
        best = None
        for th in thresholds:
            active = cal_p >= th
            value = int(cal.loc[active, "day_net"].sum())
            key = (value, int(active.sum()), -th)
            if best is None or key > best[0]: best = (key, th)
        th = best[1]
        med = hist[feature_cols].median().fillna(0); m=make_model(args.seed); m.fit(hist[feature_cols].fillna(med),hist.day_good)
        day_p=float(m.predict_proba(q[feature_cols].fillna(med))[:,1][0]); active=day_p >= th
        rows=z[z.target_day.eq(day)].copy(); out=np.where(active & rows.eligible, rows.cand, rows.base).astype(int)
        for (_,r), pp in zip(rows.iterrows(),out):
            output.append({"target_day":day.strftime("%Y-%m-%d"),"hour_business":int(r.hour_business),"y_true":int(r.y),"baseline_pred":int(r.base),"candidate_pred":int(r.cand),"predicted_direction":int(pp),"day_active":bool(active),"day_probability":day_p,"threshold":th,"training_last_day":hist.target_day.max().strftime("%Y-%m-%d")})
        audit.append({"target_day":day.strftime("%Y-%m-%d"),"training_last_day":hist.target_day.max().strftime("%Y-%m-%d"),"required_last_day":cutoff.strftime("%Y-%m-%d"),"strict_ok":bool(hist.target_day.max() <= cutoff),"day_probability":day_p,"threshold":th,"active":bool(active)})
    out=pd.DataFrame(output); au=pd.DataFrame(audit)
    if out.empty or not au.strict_ok.all(): raise RuntimeError("no output or strict audit failure")
    summary=[]
    for name,col in [("baseline","baseline_pred"),("baseline_plus_daily_regime_gate","predicted_direction")]: summary.append({"variant":name,**metrics(out.y_true.to_numpy(),out[col].to_numpy())})
    monthly=[]; out["month"]=out.target_day.str[:7]
    for month,g in out.groupby("month"):
        for name,col in [("baseline","baseline_pred"),("baseline_plus_daily_regime_gate","predicted_direction")]: monthly.append({"month":month,"variant":name,**metrics(g.y_true.to_numpy(),g[col].to_numpy())})
    dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest/"predictions.csv",index=False,encoding="utf-8-sig");au.to_csv(dest/"daily_audit.csv",index=False,encoding="utf-8-sig");pd.DataFrame(summary).to_csv(dest/"summary.csv",index=False,encoding="utf-8-sig");pd.DataFrame(monthly).to_csv(dest/"monthly.csv",index=False,encoding="utf-8-sig")
    (dest/"manifest.json").write_text(json.dumps({"status":"STRICT/PASS","route":"A_daily_positive_regime_gate","forecast_origin":"D-1 14:00","training_last_day":"per target <= D-2","target_day_actual_as_feature":False,"target_day_DA_as_feature":False,"d1_post14_spread_as_feature":False,"final_holdout_touched":False,"feature_groups":args.feature_groups,"feature_count":len(feature_cols)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(monthly).to_string(index=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
