from __future__ import annotations

"""Single-model refinement after R1/R2/R3 reproduction.

One LightGBM is trained for all 24 hourly slots. R1 regime probabilities are
not used globally; instead, paper-derived regime information is exposed only
through 1-8 interaction features. This keeps one model while testing the
period-conditioned value observed in the strict causal reproduction stage.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from common import DEFAULT_CUBE, atomic_csv, atomic_json, atomic_parquet, load_p6_features  # noqa: E402
from adapted_14h_backtest import build_r1_signals  # noqa: E402


def model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=120,
        learning_rate=0.05, num_leaves=31, min_child_samples=40,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        verbosity=-1, n_jobs=4, random_state=seed,
    )


def metrics(g: pd.DataFrame, pred: str) -> dict:
    y = np.sign(g["y_true_spread"].to_numpy(float)); p = g[pred].to_numpy(int)
    pos = y > 0; neg = y < 0; eligible = y != 0
    pa = float((p[pos] == 1).mean()) if pos.any() else math.nan
    na = float((p[neg] == -1).mean()) if neg.any() else math.nan
    return {"days": int(g.target_day.nunique()), "n_slots": int(len(g)),
            "direction_accuracy": float((p[eligible] == y[eligible]).mean()),
            "positive_accuracy": pa, "negative_accuracy": na,
            "balanced_direction_accuracy": float(np.nanmean([pa, na]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--output", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/refine_p6_regime_routing")
    ap.add_argument("--start", default="2026-06-16")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]; out = root / args.output; out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    slot, p6 = load_p6_features(root / args.cube_root)
    slot["target_day"] = slot["target_day"].astype(str); slot["时刻"] = pd.to_datetime(slot["时刻"])
    all_days = sorted(slot.target_day.unique()); target_days = [d for d in all_days if args.start <= d <= args.end]
    first_idx = all_days.index(target_days[0]); last_idx = all_days.index(target_days[-1])
    if first_idx < args.training_days: raise ValueError("insufficient prehistory")
    feature_days = all_days[first_idx - args.training_days:last_idx + 1]
    earliest = feature_days[0]
    hmm_fit_end = pd.Timestamp(earliest) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    r1, r1meta = build_r1_signals(slot, feature_days, hmm_fit_end)
    work = slot.merge(r1.drop(columns=["r1_source_max_ds"]), on=["target_day", "hour_business"], how="left", validate="many_to_one")
    early = work["hour_business"].between(1, 8).astype(float)
    source_cols = ["r1_p_negative", "r1_p_neutral", "r1_p_positive", "r1_state_entropy", "r1_state_expected_spread"]
    routed = []
    for c in source_cols:
        name = c + "_x_1_8"
        work[name] = pd.to_numeric(work[c], errors="coerce") * early
        routed.append(name)
    work["r1_early_active"] = early
    routed.append("r1_early_active")
    aug_features = p6 + routed

    ledgers = []
    for n, day in enumerate(target_days, 1):
        idx = all_days.index(day); train_days = all_days[idx-args.training_days:idx]
        train = work[work.target_day.isin(train_days)]; test = work[work.target_day.eq(day)].sort_values("hour_business")
        y = train.target_spread.to_numpy(float); yb = (y > 0).astype(int)
        base = model(args.seed).fit(train[p6], yb)
        routed_model = model(args.seed).fit(train[aug_features], yb)
        pb = base.predict_proba(test[p6])[:,1]; pr = routed_model.predict_proba(test[aug_features])[:,1]
        d = test[["target_day","时刻","hour_business","period","target_spread"]].copy().rename(columns={"target_spread":"y_true_spread"})
        d["base_prob"] = pb; d["routed_prob"] = pr
        d["base_direction"] = np.where(pb>=0.5,1,-1); d["routed_direction"] = np.where(pr>=0.5,1,-1)
        ledgers.append(d)
        if n % 15 == 0: print(f"{n}/{len(target_days)} {day}")
    ledger = pd.concat(ledgers, ignore_index=True); atomic_parquet(out/"ledger.parquet", ledger)

    splits = {"development30":("2026-06-16","2026-07-15"), "confirmation15":("2026-07-16","2026-07-30"),
              "holdout15":("2026-07-31","2026-08-14"), "overall":(args.start,args.end)}
    rows=[]; periods=[]
    for s,(a,b) in splits.items():
        g=ledger[(ledger.target_day>=a)&(ledger.target_day<=b)]
        for m in ["base","routed"]:
            rows.append({"split":s,"model":m,**metrics(g,m+"_direction")})
            for period,pg in g.groupby("period"):
                periods.append({"split":s,"model":m,"period":period,**metrics(pg,m+"_direction")})
    summary=pd.DataFrame(rows); atomic_csv(out/"summary.csv",summary); atomic_csv(out/"period_metrics.csv",pd.DataFrame(periods))
    manifest={"pipeline":"p6_r1_regime_routed_single_model","status":"complete","forecast_origin":"D-1 14:00",
              "training_days":args.training_days,"p6_features":len(p6),"routed_features":routed,
              "hmm":r1meta,"production_chain_touched":False,"runtime_seconds":time.perf_counter()-t0}
    atomic_json(out/"manifest.json",manifest)
    print(summary.to_string(index=False)); print(json.dumps({"runtime_seconds":manifest["runtime_seconds"]},ensure_ascii=False))

if __name__ == "__main__": main()
