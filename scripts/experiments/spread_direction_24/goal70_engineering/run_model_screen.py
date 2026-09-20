"""Goal-70 engineering screen for strict D-1 14:00 spread direction forecasting.

Experiment-only runner. It consumes the cutoff-safe Feature Cube and never reads
production ledgers as model inputs. It evaluates model-family / training-window /
similar-day ideas on pre-final dates only. Final fresh holdout must be run separately.

Information boundary inherits the Feature Cube:
- target-day DA/RT/spread/actual grid values are labels only;
- target-day forecast grid is allowed;
- D-1 spread only p1-p14 context is visible;
- all full historical spread/error features use D-2 or earlier.

The similar-day module is also causal: for target day D, neighbors are selected
from complete days <= D-2 using only forecast profiles known at the origin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

KEY_ERR_TOKENS = ("风电总加", "光伏总加", "直调负荷", "竞价空间", "新能源总加")
CORE_SD_TOKENS = ("直调负荷", "风电总加", "光伏总加", "竞价空间", "新能源总加", "联络线受电负荷")


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))


def p6_features(groups: dict[str, list[str]]) -> list[str]:
    base = groups["F0"] + groups["F1"]
    raw = groups["F2"]
    phys = groups["F3"] + groups["F4"]
    err_core = [
        c for c in groups["F5"]
        if c.startswith("err_net_load_") or any(token in c for token in KEY_ERR_TOKENS)
    ]
    return dedupe(base + raw + phys + err_core)


def direction_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    yt = np.sign(np.asarray(y, float))
    yp = np.asarray(pred, int)
    eligible = yt != 0
    correct = yt == yp
    pos = yt > 0
    neg = yt < 0
    pa = float(correct[pos].mean()) if pos.any() else math.nan
    na = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "n_slots_nonzero": int(eligible.sum()),
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
        "all_negative_accuracy": float(neg[eligible].mean()) if eligible.any() else math.nan,
    }


def _complete_day_index(slot: pd.DataFrame) -> tuple[list[str], dict[str, pd.DataFrame]]:
    day_map: dict[str, pd.DataFrame] = {}
    for day, g in slot.groupby("target_day", sort=True):
        gg = g.sort_values("hour_business")
        if len(gg) == 24 and gg["target_spread"].notna().all():
            day_map[str(day)] = gg
    return sorted(day_map), day_map


def add_similar_day_features(slot: pd.DataFrame, groups: dict[str, list[str]], k_values=(5, 10, 20), lookback_days=365) -> tuple[pd.DataFrame, dict]:
    """Causal nearest-day features. Candidate neighbors are <= D-2 only."""
    out = slot.copy()
    all_days, day_map = _complete_day_index(out)
    profile_cols = [c for c in groups["F2"] if any(t in c for t in CORE_SD_TOKENS)]
    # Add a few physical profiles that are scale-normalized and useful for market tightness.
    profile_cols += [c for c in groups["F3"] if c in {"residual_load_renew", "renewable_share", "bidding_space_ratio", "interconnect_share"}]
    profile_cols = dedupe(profile_cols)
    profiles: dict[str, np.ndarray] = {}
    spreads: dict[str, np.ndarray] = {}
    for d in all_days:
        g = day_map[d]
        profiles[d] = g[profile_cols].to_numpy(float).reshape(-1)
        spreads[d] = g["target_spread"].to_numpy(float)

    feats_by_day: dict[str, dict[str, np.ndarray | float]] = {}
    audit_rows = []
    for i, day in enumerate(all_days):
        target_ts = pd.Timestamp(day)
        cutoff_candidate = (target_ts - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        candidates = [d for d in all_days[max(0, i-lookback_days-2):i] if d <= cutoff_candidate]
        if len(candidates) < max(k_values):
            continue
        X = np.stack([profiles[d] for d in candidates])
        q = profiles[day]
        # Scale using candidate history only; NaNs are median-imputed from candidate history.
        med = np.nanmedian(X, axis=0)
        X2 = np.where(np.isfinite(X), X, med)
        q2 = np.where(np.isfinite(q), q, med)
        scale = np.nanstd(X2, axis=0)
        scale = np.where((~np.isfinite(scale)) | (scale < 1e-6), 1.0, scale)
        dist = np.sqrt(np.mean(((X2-q2)/scale)**2, axis=1))
        order = np.argsort(dist)
        rec: dict[str, np.ndarray | float] = {}
        for k in k_values:
            idx = order[:k]
            ds = [candidates[j] for j in idx]
            S = np.stack([spreads[d] for d in ds])
            dd = dist[idx]
            w = 1.0 / np.maximum(dd, 1e-4)
            w = w / w.sum()
            rec[f"sd{k}_spread_mean"] = np.nanmean(S, axis=0)
            rec[f"sd{k}_spread_median"] = np.nanmedian(S, axis=0)
            rec[f"sd{k}_positive_rate"] = np.nanmean(S > 0, axis=0)
            rec[f"sd{k}_weighted_positive_rate"] = np.sum((S > 0) * w[:, None], axis=0)
            rec[f"sd{k}_spread_std"] = np.nanstd(S, axis=0)
            rec[f"sd{k}_mean_distance"] = float(np.mean(dd))
            # Safe day/segment regime priors derived only from selected historical labels.
            rec[f"sd{k}_day_positive_rate"] = float(np.mean(S > 0))
            for name, sl in (("1_8", slice(0,8)), ("9_16", slice(8,16)), ("17_24", slice(16,24))):
                rec[f"sd{k}_{name}_positive_rate"] = float(np.mean(S[:, sl] > 0))
        feats_by_day[day] = rec
        audit_rows.append({
            "target_day": day,
            "latest_candidate_day": max(candidates),
            "required_latest_candidate_le": cutoff_candidate,
            "causal_ok": max(candidates) <= cutoff_candidate,
            "candidate_count": len(candidates),
        })

    new_cols: set[str] = set()
    pieces = []
    for day, g in out.groupby("target_day", sort=False):
        gg = g.copy()
        rec = feats_by_day.get(str(day))
        if rec is not None:
            for name, val in rec.items():
                if isinstance(val, np.ndarray):
                    if len(gg) != 24:
                        continue
                    gg[name] = val
                else:
                    gg[name] = val
                new_cols.add(name)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    if not audit.empty and not audit["causal_ok"].all():
        raise RuntimeError("similar-day causal audit failed")
    return out, {"features": sorted(new_cols), "audit": audit, "profile_cols": profile_cols}


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    training_days: int
    use_sd: bool = False
    segment: bool = False
    class_mode: str = "balanced"
    sd_k: int = 10
    smooth: int = 1


def make_classifier(v: Variant, pos_weight: float, seed: int):
    if v.family == "lgbm":
        return lgb.LGBMClassifier(
            objective="binary", class_weight=("balanced" if v.class_mode == "balanced" else None),
            n_estimators=180, learning_rate=0.04, num_leaves=31, min_child_samples=35,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=4, verbosity=-1,
        )
    if v.family == "catboost":
        return CatBoostClassifier(
            iterations=220, depth=6, learning_rate=0.04, loss_function="Logloss",
            auto_class_weights=("Balanced" if v.class_mode == "balanced" else None),
            random_seed=seed, verbose=False, allow_writing_files=False, thread_count=4,
        )
    if v.family == "xgboost":
        return XGBClassifier(
            n_estimators=220, max_depth=5, learning_rate=0.035,
            subsample=0.9, colsample_bytree=0.9, min_child_weight=8,
            reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=(pos_weight if v.class_mode == "balanced" else 1.0),
            random_state=seed, n_jobs=4,
        )
    if v.family == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=350, max_depth=10, min_samples_leaf=8,
            max_features=0.75, class_weight=("balanced" if v.class_mode == "balanced" else None),
            random_state=seed, n_jobs=4,
        )
    raise ValueError(v.family)


def strict_train_days(all_days: list[str], target_day: str, training_days: int) -> list[str]:
    """Completed training labels available at D-1 14:00: D-2 and earlier only."""
    idx = all_days.index(target_day)
    end = idx - 1  # exclusive; last included index is idx-2 == D-2
    start = max(0, end - training_days)
    days = all_days[start:end]
    if len(days) < min(60, training_days):
        raise ValueError(f"{target_day}: only {len(days)} strict D-2 training days")
    return days


def predict_variant(slot: pd.DataFrame, features: list[str], v: Variant, target_days: list[str], seed: int) -> pd.DataFrame:
    all_days = sorted(slot["target_day"].dropna().astype(str).unique())
    rows = []
    for day in target_days:
        train_days = strict_train_days(all_days, day, v.training_days)
        train = slot[slot["target_day"].isin(train_days)].copy()
        test = slot[slot["target_day"].eq(day)].sort_values("hour_business").copy()
        if len(train_days) < min(60, v.training_days) or len(test) != 24:
            raise ValueError(f"{v.name} {day}: train_days={len(train_days)} test={len(test)}")
        ytr = (train["target_spread"].to_numpy(float) > 0).astype(int)
        neg = max(1, int((ytr == 0).sum())); pos = max(1, int((ytr == 1).sum()))
        pw = neg / pos
        prob = np.zeros(24, float)
        if v.segment:
            for period in ("1_8", "9_16", "17_24"):
                tr = train[train["period"].eq(period)]
                te = test[test["period"].eq(period)]
                yy = (tr["target_spread"].to_numpy(float) > 0).astype(int)
                n0=max(1,int((yy==0).sum())); n1=max(1,int((yy==1).sum()))
                model = make_classifier(v, n0/n1, seed)
                model.fit(tr[features], yy)
                prob[te.index.to_numpy()-test.index.min()] = model.predict_proba(te[features])[:,1]
        else:
            model = make_classifier(v, pw, seed)
            model.fit(train[features], ytr)
            prob = model.predict_proba(test[features])[:,1].astype(float)
        if v.smooth > 1:
            # Centered smoothing uses only predictions for the same target day, all generated at origin.
            prob = pd.Series(prob).rolling(v.smooth, center=True, min_periods=1).mean().to_numpy()
        pred = np.where(prob >= 0.5, 1, -1)
        out = test[["target_day","hour_business","period","target_spread"]].copy()
        out["variant"] = v.name
        out["prob_positive"] = prob
        out["predicted_direction"] = pred
        out["training_last_day"] = train_days[-1]
        out["training_days"] = len(train_days)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def catboost_regression(slot: pd.DataFrame, features: list[str], target_days: list[str], training_days: int, seed: int, smooth: int) -> pd.DataFrame:
    all_days=sorted(slot["target_day"].dropna().astype(str).unique()); rows=[]
    for day in target_days:
        train_days=strict_train_days(all_days, day, training_days)
        tr=slot[slot.target_day.isin(train_days)]; te=slot[slot.target_day.eq(day)].sort_values("hour_business")
        preds=[]
        # Bootstrap-style ensemble inspired by Joint forecasting agent; deterministic seeds and no target-day labels.
        for j in range(5):
            m=CatBoostRegressor(iterations=220,depth=6,learning_rate=.04,loss_function="MAE",random_seed=seed+j,verbose=False,allow_writing_files=False,thread_count=4,bootstrap_type="Bayesian",bagging_temperature=1.0)
            m.fit(tr[features],tr.target_spread.to_numpy(float)); preds.append(m.predict(te[features]))
        p=np.median(np.stack(preds),axis=0)
        if smooth>1: p=pd.Series(p).rolling(smooth,center=True,min_periods=1).mean().to_numpy()
        out=te[["target_day","hour_business","period","target_spread"]].copy(); out["variant"]=f"catreg_boot5_w{training_days}_smooth{smooth}"; out["predicted_spread"]=p; out["predicted_direction"]=np.where(p>=0,1,-1); out["prob_positive"]=np.nan; out["training_last_day"]=train_days[-1]; out["training_days"]=len(train_days); rows.append(out)
    return pd.concat(rows,ignore_index=True)


def summarize(ledger: pd.DataFrame, split_ranges: dict[str, tuple[str,str]]) -> pd.DataFrame:
    rows=[]
    for split,(start,end) in split_ranges.items():
        p=ledger[(ledger.target_day>=start)&(ledger.target_day<=end)]
        for name,g in p.groupby("variant",sort=False):
            rows.append({"split":split,"variant":name,"days":g.target_day.nunique(),**direction_metrics(g.target_spread.to_numpy(),g.predicted_direction.to_numpy())})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cube-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube")
    ap.add_argument("--output-root",default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration1_model_screen")
    ap.add_argument("--start",default="2026-07-16")
    ap.add_argument("--end",default="2026-08-14")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--group", choices=["all","lgbm","trees","reg"], default="all")
    args=ap.parse_args(); t0=time.perf_counter(); cube=Path(args.cube_root); out=Path(args.output_root)
    slot=pd.read_parquet(cube/"slot_table.parquet")
    groups=json.loads((cube/"feature_groups.json").read_text(encoding="utf-8")); manifest=json.loads((cube/"manifest.json").read_text(encoding="utf-8"))
    base=p6_features(groups)
    slot,sd=add_similar_day_features(slot,groups,k_values=(5,10,20),lookback_days=365)
    atomic_csv(out/"similar_day_causal_audit.csv",sd["audit"])
    if sd["audit"].empty or not sd["audit"]["causal_ok"].all(): raise RuntimeError("similar-day audit is not fully green")
    target_days=[d for d in sorted(slot.target_day.unique()) if args.start<=d<=args.end]
    variants=[]
    for w in (60,90,120,180,365):
        variants.append(Variant(f"lgbm_bal_p6_w{w}","lgbm",w,False,False,"balanced"))
        variants.append(Variant(f"lgbm_raw_p6_w{w}","lgbm",w,False,False,"raw"))
    # Literature-inspired cheap family/representation screen centered on the prior 90d winner.
    variants += [
        Variant("lgbm_bal_p6_sd5_w90","lgbm",90,True,False,"balanced",5),
        Variant("lgbm_bal_p6_sd10_w90","lgbm",90,True,False,"balanced",10),
        Variant("lgbm_bal_p6_sd20_w90","lgbm",90,True,False,"balanced",20),
        Variant("lgbm_bal_p6_sd10_seg_w90","lgbm",90,True,True,"balanced",10),
        Variant("cat_bal_p6_w90","catboost",90,False,False,"balanced"),
        Variant("cat_bal_p6_sd10_w90","catboost",90,True,False,"balanced",10),
        Variant("cat_raw_p6_sd10_w90","catboost",90,True,False,"raw",10),
        Variant("xgb_bal_p6_sd10_w90","xgboost",90,True,False,"balanced",10),
        Variant("extra_bal_p6_sd10_w90","extratrees",90,True,False,"balanced",10),
        Variant("lgbm_bal_p6_sd10_w90_smooth3","lgbm",90,True,False,"balanced",10,3),
        Variant("cat_bal_p6_sd10_w90_smooth3","catboost",90,True,False,"balanced",10,3),
    ]
    if args.group == "lgbm":
        variants = [v for v in variants if v.family == "lgbm"]
    elif args.group == "trees":
        variants = [v for v in variants if v.family in {"catboost","xgboost","extratrees"}]
    elif args.group == "reg":
        variants = []
    ledgers=[]
    for i,v in enumerate(variants,1):
        feats=list(base)
        if v.use_sd:
            prefix=f"sd{v.sd_k}_"
            feats += [c for c in sd["features"] if c.startswith(prefix)]
        feats=dedupe(feats)
        print(f"[{i}/{len(variants)}] {v.name} n_features={len(feats)}")
        ledgers.append(predict_variant(slot,feats,v,target_days,args.seed))
    # One direct-regression bootstrap path from the Shanxi Joint paper.
    if args.group in {"all","reg"}:
        ledgers.append(catboost_regression(slot,base,target_days,90,args.seed,1))
        ledgers.append(catboost_regression(slot,base,target_days,90,args.seed,3))
    if not ledgers:
        raise RuntimeError("no variants selected")
    ledger=pd.concat(ledgers,ignore_index=True)
    atomic_parquet(out/"ledger.parquet",ledger)
    splits={"selection15":("2026-07-16","2026-07-30"),"validation15":("2026-07-31","2026-08-14"),"combined30":(args.start,args.end)}
    summary=summarize(ledger,splits); atomic_csv(out/"summary.csv",summary)
    # Selection score rewards both windows and balanced accuracy; no 8/15+ data is touched.
    piv=summary.pivot(index="variant",columns="split",values=["direction_accuracy","balanced_direction_accuracy"])
    ranking=[]
    for name in piv.index:
        def val(metric,split):
            try:return float(piv.loc[name,(metric,split)])
            except:return math.nan
        sacc=val("direction_accuracy","selection15"); vacc=val("direction_accuracy","validation15"); sb=val("balanced_direction_accuracy","selection15"); vb=val("balanced_direction_accuracy","validation15")
        score=np.nanmean([sacc,vacc,sb,vb])-0.5*abs(sacc-vacc)
        ranking.append({"variant":name,"selection_acc":sacc,"validation_acc":vacc,"selection_bal":sb,"validation_bal":vb,"robust_score":score,"min_window_acc":min(sacc,vacc)})
    ranking=pd.DataFrame(ranking).sort_values(["robust_score","min_window_acc"],ascending=False); atomic_csv(out/"ranking.csv",ranking)
    atomic_json(out/"manifest.json",{
        "status":"complete","experiment":"goal70_iteration1_model_screen","source_cube":str(cube),"source_cube_sha256":sha256(cube/"slot_table.parquet"),
        "forecast_origin":"D-1 14:00","target_day_actual_as_feature":False,"target_day_DA_as_feature":False,"d1_post14_spread_as_feature":False,"training_label_cutoff":"D-2 completed days only",
        "similar_day_contract":"neighbors <= D-2; descriptor uses target forecast grid + historical forecast profiles only",
        "screen_range":[args.start,args.end],"fresh_final_holdout_reserved":["2026-08-15","2026-08-21"],"final_holdout_touched":False,
        "p6_feature_count":len(base),"similar_day_feature_count":len(sd["features"]),"group":args.group,"variants":[v.__dict__ for v in variants]+((["catreg_boot5_w90_smooth1","catreg_boot5_w90_smooth3"]) if args.group in {"all","reg"} else []),
        "literature_ideas":[
            "Shanxi Joint framework: direct RT-DA spread, CatBoost bootstrap regression, moving average, walk-forward",
            "Shandong similar-day electricity forecasting: nearest/similar historical day construction",
            "DART statistical-learning literature: tailored features often matter more than algorithm choice",
        ],"runtime_seconds":time.perf_counter()-t0,
        "cube_information_boundary":manifest.get("information_boundary"),
    })
    print("\nTOP RANKING\n",ranking.head(12).to_string(index=False))
    print("\nSUMMARY validation\n",summary[summary.split.eq("validation15")].sort_values(["direction_accuracy","balanced_direction_accuracy"],ascending=False).head(15).to_string(index=False))

if __name__=="__main__": main()
