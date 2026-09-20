"""Strict-D2 online adaptation for P6 vs P6+SimilarDay20.

Consumes only previously generated out-of-sample strict predictions. For target day D,
all adaptive weights/thresholds are estimated from earlier target days only. No target-day
labels are used to choose the expert or threshold.

Branches:
A) online expert selection / exponential weighting for concept drift;
B) prequential threshold calibration for class-prior drift.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


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


def metrics(g: pd.DataFrame) -> dict:
    y = np.sign(g["target_spread"].to_numpy(float))
    p = g["predicted_direction"].to_numpy(int)
    elig = y != 0
    ok = y == p
    pos = y > 0
    neg = y < 0
    pa = float(ok[pos].mean()) if pos.any() else math.nan
    na = float(ok[neg].mean()) if neg.any() else math.nan
    return {
        "days": int(g.target_day.nunique()),
        "n": int(elig.sum()),
        "direction_accuracy": float(ok[elig].mean()),
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
        "all_negative_accuracy": float(neg[elig].mean()),
    }


def load_ledgers(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        frames.append(df)
    x = pd.concat(frames, ignore_index=True)
    x = x.drop_duplicates(["target_day", "hour_business", "variant"], keep="last")
    keep = x[x.variant.isin(["P6_w90", "P6_SD20_w90"])].copy()
    wide = keep.pivot_table(
        index=["target_day", "hour_business", "period", "target_spread"],
        columns="variant", values="prob_positive", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"P6_w90": "p_base", "P6_SD20_w90": "p_sd20"})
    wide = wide.dropna(subset=["p_base", "p_sd20"]).sort_values(["target_day", "hour_business"]).reset_index(drop=True)
    return wide


def score_hist(h: pd.DataFrame, prob_col: str, threshold: float = 0.5) -> float:
    if h.empty:
        return 0.5
    y = h.target_spread.to_numpy(float) > 0
    pred = h[prob_col].to_numpy(float) >= threshold
    return float((y == pred).mean())


def logloss_hist(h: pd.DataFrame, prob_col: str) -> float:
    if h.empty:
        return math.log(2.0)
    y = (h.target_spread.to_numpy(float) > 0).astype(float)
    p = np.clip(h[prob_col].to_numpy(float), 1e-5, 1 - 1e-5)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))


def best_threshold(h: pd.DataFrame, prob_col: str, objective: str) -> float:
    if len(h) < 24 * 7:
        return 0.5
    y = np.sign(h.target_spread.to_numpy(float))
    best = (-1e9, 0.5)
    for th in np.arange(0.30, 0.701, 0.02):
        pred = np.where(h[prob_col].to_numpy(float) >= th, 1, -1)
        ok = y == pred
        pos = y > 0; neg = y < 0
        acc = float(ok.mean())
        pa = float(ok[pos].mean()) if pos.any() else 0.0
        na = float(ok[neg].mean()) if neg.any() else 0.0
        bal = 0.5 * (pa + na)
        if objective == "accuracy":
            # Prevent degenerate all-negative calibration from winning solely on prior.
            value = acc if bal >= 0.52 else acc - 0.20*(0.52-bal)
        else:
            value = bal
        candidate = (value, -abs(th-0.5), th)
        if candidate > (best[0], -abs(best[1]-0.5), best[1]):
            best = (value, th)
    return float(best[1])


def build_variant(wide: pd.DataFrame, name: str, mode: str, horizon: int, eta: float = 4.0, per_period: bool = False, threshold_obj: str = "accuracy") -> pd.DataFrame:
    days = sorted(wide.target_day.unique())
    rows = []
    for day in days:
        test = wide[wide.target_day.eq(day)].copy()
        max_hist_day = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        history_days = [d for d in days if d <= max_hist_day][-horizon:]
        hist = wide[wide.target_day.isin(history_days)]
        probs = np.empty(len(test), float)
        thresholds = np.full(len(test), 0.5, float)
        groups = ["1_8", "9_16", "17_24"] if per_period else [None]
        for period in groups:
            te = test if period is None else test[test.period.eq(period)]
            hh = hist if period is None else hist[hist.period.eq(period)]
            if te.empty:
                continue
            if mode == "best_expert":
                sb = score_hist(hh, "p_base")
                ss = score_hist(hh, "p_sd20")
                p = te["p_sd20"].to_numpy(float) if ss >= sb else te["p_base"].to_numpy(float)
                th = 0.5
            elif mode == "exp_weight":
                lb = logloss_hist(hh, "p_base")
                ls = logloss_hist(hh, "p_sd20")
                wb = math.exp(-eta*lb); ws = math.exp(-eta*ls)
                wsum = wb + ws
                wb /= wsum; ws /= wsum
                p = wb*te.p_base.to_numpy(float) + ws*te.p_sd20.to_numpy(float)
                th = 0.5
            elif mode == "equal":
                p = 0.5*(te.p_base.to_numpy(float) + te.p_sd20.to_numpy(float))
                th = 0.5
            elif mode == "sd20_threshold":
                p = te.p_sd20.to_numpy(float)
                th = best_threshold(hh, "p_sd20", threshold_obj)
            elif mode == "blend_threshold":
                hh = hh.copy(); hh["p_blend"] = 0.5*(hh.p_base + hh.p_sd20)
                p = 0.5*(te.p_base.to_numpy(float) + te.p_sd20.to_numpy(float))
                th = best_threshold(hh, "p_blend", threshold_obj)
            else:
                raise ValueError(mode)
            idx = te.index.to_numpy() - test.index.min()
            probs[idx] = p
            thresholds[idx] = th
        out = test[["target_day", "hour_business", "period", "target_spread"]].copy()
        out["variant"] = name
        out["prob_positive"] = probs
        out["threshold"] = thresholds
        out["predicted_direction"] = np.where(probs >= thresholds, 1, -1)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarize(ledger: pd.DataFrame, eval_start: str, eval_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = ledger[(ledger.target_day >= eval_start) & (ledger.target_day <= eval_end)].copy()
    x["month"] = x.target_day.str[:7]
    monthly = []
    for (v,m), g in x.groupby(["variant", "month"]):
        monthly.append({"variant":v, "month":m, **metrics(g)})
    monthly = pd.DataFrame(monthly)
    robust = []
    for v,g in monthly.groupby("variant"):
        acc = g.direction_accuracy.astype(float); bal = g.balanced_direction_accuracy.astype(float)
        gain = acc - g.all_negative_accuracy.astype(float)
        robust.append({
            "variant":v, "months":len(g), "mean_month_acc":acc.mean(), "median_month_acc":acc.median(),
            "min_month_acc":acc.min(), "max_month_acc":acc.max(), "std_month_acc":acc.std(ddof=0),
            "months_ge_065":int((acc>=.65).sum()), "months_ge_070":int((acc>=.70).sum()),
            "mean_month_bal":bal.mean(), "min_month_bal":bal.min(), "mean_gain_vs_all_negative":gain.mean(),
            "months_beating_all_negative":int((gain>0).sum()),
        })
    robust = pd.DataFrame(robust).sort_values(["mean_month_acc", "mean_month_bal"], ascending=False)
    return monthly, robust


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration4_online_adaptation_strictD2")
    ap.add_argument("--eval-start", default="2026-01-01")
    ap.add_argument("--eval-end", default="2026-08-14")
    args=ap.parse_args()
    roots=[
        "outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/cross_month_champion_audit_warmup_strictD2/ledger.parquet",
        "outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/cross_month_champion_audit_strictD2_q1/ledger.parquet",
        "outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/cross_month_champion_audit_q2_strictD2/ledger.parquet",
        "outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/cross_month_champion_audit_q3_strictD2/ledger.parquet",
    ]
    wide=load_ledgers(roots)
    ledgers=[]
    # Static references reconstructed from the same strict OOS probabilities.
    for col,name in [("p_base","P6_w90_static"),("p_sd20","P6_SD20_w90_static")]:
        z=wide[["target_day","hour_business","period","target_spread"]].copy(); z["variant"]=name; z["prob_positive"]=wide[col]; z["threshold"]=.5; z["predicted_direction"]=np.where(wide[col]>=.5,1,-1); ledgers.append(z)
    for h in (7,14,28,56):
        ledgers.append(build_variant(wide,f"best_global_h{h}","best_expert",h,per_period=False))
        ledgers.append(build_variant(wide,f"best_period_h{h}","best_expert",h,per_period=True))
        ledgers.append(build_variant(wide,f"exp_global_h{h}","exp_weight",h,eta=6.0,per_period=False))
        ledgers.append(build_variant(wide,f"exp_period_h{h}","exp_weight",h,eta=6.0,per_period=True))
    ledgers.append(build_variant(wide,"equal_blend","equal",28))
    for h in (14,28,56):
        ledgers.append(build_variant(wide,f"sd20_thr_acc_h{h}","sd20_threshold",h,threshold_obj="accuracy"))
        ledgers.append(build_variant(wide,f"sd20_thr_bal_h{h}","sd20_threshold",h,threshold_obj="balanced"))
        ledgers.append(build_variant(wide,f"blend_thr_acc_h{h}","blend_threshold",h,threshold_obj="accuracy"))
        ledgers.append(build_variant(wide,f"blend_thr_bal_h{h}","blend_threshold",h,threshold_obj="balanced"))
    ledger=pd.concat(ledgers,ignore_index=True)
    monthly,robust=summarize(ledger,args.eval_start,args.eval_end)
    out=Path(args.output_root); atomic_parquet(out/"ledger.parquet",ledger); atomic_csv(out/"monthly.csv",monthly); atomic_csv(out/"robustness.csv",robust)
    atomic_json(out/"manifest.json",{
        "status":"complete","experiment":"iteration4_online_adaptation_strictD2","sources":roots,
        "forecast_origin":"D-1 14:00","adaptive_history":"historical OOS target days <= D-2 only",
        "fresh_final_holdout_reserved":["2026-08-15","2026-08-21"],"final_holdout_touched":False,
        "literature_basis":["concept-drift online combination","adaptive rolling-window EPF","sample-domain adaptation"],
    })
    print("\nROBUSTNESS TOP\n",robust.head(15).to_string(index=False))
    top=robust.iloc[0].variant
    print("\nTOP MONTHLY\n",monthly[monthly.variant.eq(top)].to_string(index=False))

if __name__=="__main__":
    main()
