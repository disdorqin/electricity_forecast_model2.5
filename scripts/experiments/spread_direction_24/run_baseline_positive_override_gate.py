"""Add a causal positive-rescue gate on top of the retained strict baseline.

The baseline prediction is never replaced except when it is negative and the
candidate expert is positive.  For every D, a gate is trained only on historical
strict OOS disagreements through D-2.  Its threshold is selected on the tail of
that historical block, then the gate is refit on all available history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")
EMPIRICAL_FEATURES = [
    "emp_global_rate", "emp_recent14_rate", "emp_recent30_rate",
    "emp_hour_rate", "emp_period_rate", "emp_gap_rate", "emp_joint_rate",
    "emp_hour_count", "emp_period_count", "emp_gap_count", "emp_joint_count",
]


def _period(hour: int) -> int:
    return 0 if hour <= 8 else (1 if hour <= 16 else 2)


def _gap_bin(gap: float) -> int:
    return int(np.digitize(gap, [0.05, 0.15, 0.25, 0.40]))


def empirical_state(reference: pd.DataFrame, row: pd.Series) -> dict[str, float]:
    """Historical rescue precision features; reference must already be <= D-2."""
    if reference.empty:
        return {c: 0.5 if c.endswith("rate") else 0.0 for c in EMPIRICAL_FEATURES}
    ref = reference.copy()
    yy = (ref.y.to_numpy() == 1).astype(float)
    global_rate = float((yy.sum() + 3.0) / (len(yy) + 6.0))

    def smoothed(mask: np.ndarray) -> tuple[float, float]:
        values = yy[mask]
        n = len(values)
        return float((values.sum() + 6.0 * global_rate) / (n + 6.0)), float(n)

    day = pd.Timestamp(row.target_day)
    recent14 = ref.target_day.ge(day - pd.Timedelta(days=14)).to_numpy()
    recent30 = ref.target_day.ge(day - pd.Timedelta(days=30)).to_numpy()
    hour = int(row.hour_business); period = _period(hour); gap_bin = _gap_bin(float(row.prob_gap))
    ref_hour = ref.hour_business.to_numpy(int)
    ref_period = np.array([_period(int(x)) for x in ref_hour])
    ref_gap = np.array([_gap_bin(float(x)) for x in ref.prob_gap.to_numpy(float)])
    hr, hn = smoothed(ref_hour == hour)
    pr, pn = smoothed(ref_period == period)
    gr, gn = smoothed(ref_gap == gap_bin)
    jr, jn = smoothed((ref_period == period) & (ref_gap == gap_bin))
    r14, _ = smoothed(recent14); r30, _ = smoothed(recent30)
    return {"emp_global_rate": global_rate, "emp_recent14_rate": r14,
            "emp_recent30_rate": r30, "emp_hour_rate": hr,
            "emp_period_rate": pr, "emp_gap_rate": gr, "emp_joint_rate": jr,
            "emp_hour_count": np.log1p(hn), "emp_period_count": np.log1p(pn),
            "emp_gap_count": np.log1p(gn), "emp_joint_count": np.log1p(jn)}


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pos, neg = y == 1, y == -1
    pr = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    nr = float((pred[neg] == -1).mean()) if neg.any() else float("nan")
    return {"n": int(len(y)), "direction_accuracy": float((pred == y).mean()),
            "positive_recall": pr, "negative_recall": nr,
            "balanced_accuracy": float(np.nanmean([pr, nr])),
            "all_negative_baseline": float(neg.mean())}


def model(seed: int, kind: str):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.25, class_weight="balanced", max_iter=1000, random_state=seed))
    if kind == "extra":
        return ExtraTreesClassifier(n_estimators=160, max_depth=5, min_samples_leaf=8,
            class_weight="balanced", random_state=seed, n_jobs=4)
    if kind == "lgbm_unweighted":
        return lgb.LGBMClassifier(objective="binary", n_estimators=120,
            learning_rate=0.025, num_leaves=7, max_depth=3, min_child_samples=16,
            reg_lambda=8.0, random_state=seed, n_jobs=4, verbosity=-1)
    if kind == "lgbm_deep":
        return lgb.LGBMClassifier(objective="binary", class_weight="balanced",
            n_estimators=140, learning_rate=0.025, num_leaves=15, max_depth=4,
            min_child_samples=10, reg_lambda=10.0, random_state=seed,
            n_jobs=4, verbosity=-1)
    return lgb.LGBMClassifier(objective="binary", class_weight="balanced",
        n_estimators=80, learning_rate=0.035, num_leaves=9, max_depth=3,
        min_child_samples=20, reg_lambda=5.0, random_state=seed,
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
    ap.add_argument("--history-days", type=int, default=90)
    ap.add_argument("--min-disagreements", type=int, default=120)
    ap.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    ap.add_argument("--feature-groups", default="F2,F3,F4,F7",
                    help="cutoff-safe cube groups available to the override gate")
    ap.add_argument("--gate-model", choices=["lgbm","lgbm_unweighted","lgbm_deep","logistic","extra"], default="lgbm")
    ap.add_argument("--all-expert-probs", action="store_true",
                    help="use every source expert's strict OOS probability as gate-only features")
    ap.add_argument("--confirmation-source", type=Path,
                    help="optional independent strict-OOS ledger used only as a causal confirmation feature")
    ap.add_argument("--confirmation-variant",
                    help="variant name in --confirmation-source")
    ap.add_argument("--require-confirmation-positive", action="store_true",
                    help="permit an override only when the independent confirmation expert is also positive")
    ap.add_argument("--empirical-precision", action="store_true",
                    help="add strictly prequential rescue precision by hour/period/probability gap")
    ap.add_argument("--positive-magnitude-weight", type=float, default=0.0,
                    help="extra sample weight for large historical positive rescues; zero preserves the plain gate")
    ap.add_argument("--day-state-features", action="store_true",
                    help="include forecast-only full-day candidate-shape features")
    ap.add_argument("--negative-veto", action="store_true",
                    help="also causally correct baseline-positive/candidate-negative disagreements")
    ap.add_argument("--disable-positive-override", action="store_true",
                    help="ablation: retain only the causal negative-veto branch")
    ap.add_argument("--seed", type=int, default=20260824)
    args = ap.parse_args()

    ledger = pd.read_parquet(args.source.resolve())
    ledger["target_day"] = pd.to_datetime(ledger["target_day"]).dt.normalize()
    if ledger.target_day.max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("source touches final holdout")
    required = {"target_day", "hour_business", "target_spread", "variant", "predicted_direction", "prob_positive"}
    if required - set(ledger.columns):
        raise RuntimeError(f"source missing {sorted(required-set(ledger.columns))}")
    sub = ledger[ledger.variant.isin([args.baseline, args.candidate])].copy()
    direction = sub.pivot_table(index=["target_day","hour_business"], columns="variant", values="predicted_direction", aggfunc="first")
    probability_source = ledger if args.all_expert_probs else sub
    probability = probability_source.pivot_table(index=["target_day","hour_business"], columns="variant", values="prob_positive", aggfunc="first")
    direction = direction.rename(columns={args.baseline:"base_pred", args.candidate:"cand_pred"})
    probability = probability.rename(columns={args.baseline:"base_prob", args.candidate:"cand_prob"})
    probability = probability.rename(columns={c:f"expert_prob__{c}" for c in probability.columns if c not in {"base_prob","cand_prob"}})
    y = sub.drop_duplicates(["target_day","hour_business"])[["target_day","hour_business","target_spread"]].set_index(["target_day","hour_business"])
    wide = direction.join(probability).join(y).reset_index().dropna(subset=["base_pred","cand_pred","target_spread"])
    if bool(args.confirmation_source) != bool(args.confirmation_variant):
        raise RuntimeError("confirmation source and variant must be provided together")
    if args.require_confirmation_positive and not args.confirmation_source:
        raise RuntimeError("--require-confirmation-positive needs a confirmation source")
    if args.confirmation_source:
        confirm = pd.read_parquet(args.confirmation_source.resolve())
        confirm["target_day"] = pd.to_datetime(confirm["target_day"]).dt.normalize()
        if confirm.target_day.max() >= FINAL_HOLDOUT_START:
            raise RuntimeError("confirmation source touches final holdout")
        needed = {"target_day", "hour_business", "variant", "predicted_direction", "prob_positive"}
        if needed - set(confirm.columns):
            raise RuntimeError(f"confirmation source missing {sorted(needed-set(confirm.columns))}")
        confirm = confirm[confirm.variant.eq(args.confirmation_variant)].copy()
        confirm = confirm[["target_day","hour_business","predicted_direction","prob_positive"]].drop_duplicates(["target_day","hour_business"])
        confirm = confirm.rename(columns={"predicted_direction":"confirm_pred", "prob_positive":"confirm_prob"})
        wide = wide.merge(confirm, on=["target_day","hour_business"], how="left")
    wide["y"] = np.sign(wide.target_spread.to_numpy(float)).astype(int)

    slot = pd.read_parquet(args.cube.resolve() / "slot_table.parquet")
    slot["target_day"] = pd.to_datetime(slot.target_day).dt.normalize()
    groups = json.loads((args.cube.resolve() / "feature_groups.json").read_text(encoding="utf-8"))
    requested_groups = [g.strip() for g in args.feature_groups.split(",") if g.strip()]
    regime_features = [c for g in requested_groups for c in groups.get(g,[]) if c in slot.columns]
    slot = slot[["target_day","hour_business"] + regime_features].drop_duplicates(["target_day","hour_business"])
    wide = wide.merge(slot, on=["target_day","hour_business"], how="left").sort_values(["target_day","hour_business"])
    wide["prob_gap"] = wide.cand_prob - wide.base_prob
    wide["base_margin"] = np.abs(wide.base_prob - 0.5)
    wide["cand_margin"] = np.abs(wide.cand_prob - 0.5)
    # Forecast-only day-state features.  Positive spread episodes are often
    # multi-slot events; exposing the candidate's full-day shape lets the
    # residual gate distinguish an isolated optimistic slot from a coherent
    # next-day regime signal without looking at any realized target quantity.
    by_day = wide.groupby("target_day", sort=False)
    wide["day_cand_pos_count"] = by_day.cand_pred.transform(lambda s: s.eq(1).sum())
    wide["day_base_pos_count"] = by_day.base_pred.transform(lambda s: s.eq(1).sum())
    wide["day_disagree_pos_count"] = by_day.apply(lambda g: ((g.base_pred == -1) & (g.cand_pred == 1)).sum(), include_groups=False).reindex(wide.target_day).to_numpy()
    wide["day_cand_prob_mean"] = by_day.cand_prob.transform("mean")
    wide["day_cand_prob_std"] = by_day.cand_prob.transform("std").fillna(0.0)
    wide["day_prob_gap_mean"] = by_day.prob_gap.transform("mean")
    wide["day_prob_gap_max"] = by_day.prob_gap.transform("max")
    h = wide.hour_business.to_numpy(float); dow = wide.target_day.dt.dayofweek.to_numpy(float)
    wide["hour_sin"] = np.sin(2*np.pi*(h-1)/24); wide["hour_cos"] = np.cos(2*np.pi*(h-1)/24)
    wide["dow_sin"] = np.sin(2*np.pi*dow/7); wide["dow_cos"] = np.cos(2*np.pi*dow/7)
    expert_prob_features = [c for c in wide.columns if c.startswith("expert_prob__")]
    confirmation_features = [c for c in ["confirm_pred", "confirm_prob"] if c in wide.columns]
    day_state_features = ["day_cand_pos_count", "day_base_pos_count", "day_disagree_pos_count", "day_cand_prob_mean", "day_cand_prob_std", "day_prob_gap_mean", "day_prob_gap_max"]
    feature_cols = list(dict.fromkeys(["base_prob","cand_prob","prob_gap","base_margin","cand_margin","hour_sin","hour_cos","dow_sin","dow_cos"] + (day_state_features if args.day_state_features else []) + confirmation_features + expert_prob_features + regime_features))
    if args.empirical_precision:
        feature_cols += EMPIRICAL_FEATURES
    for c in feature_cols:
        if c in wide.columns:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    all_days = sorted(wide.target_day.unique())
    target_days = [d for d in all_days if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    predictions=[]; audits=[]
    for day in target_days:
        cutoff = day - pd.Timedelta(days=2)
        hist_days = [d for d in all_days if d <= cutoff][-args.history_days:]
        history = wide[wide.target_day.isin(hist_days)].copy()
        disagree = history[(history.base_pred==-1)&(history.cand_pred==1)].copy()
        query = wide[wide.target_day.eq(day)].copy()
        if len(query)!=24 or len(disagree)<args.min_disagreements or disagree.y.nunique()<2: continue
        if args.empirical_precision:
            states=[]
            for _, r in disagree.iterrows():
                prior = disagree[disagree.target_day <= pd.Timestamp(r.target_day)-pd.Timedelta(days=2)]
                states.append(empirical_state(prior, r))
            disagree = pd.concat([disagree.reset_index(drop=True), pd.DataFrame(states)], axis=1)
            qstates=[empirical_state(disagree, r) for _, r in query.iterrows()]
            query = pd.concat([query.reset_index(drop=True), pd.DataFrame(qstates)], axis=1)
        disagree["beneficial"] = (disagree.y==1).astype(int)
        # A DART-inspired tail emphasis: scale only historical beneficial
        # examples by their within-history positive magnitude rank.  The
        # realized magnitude is never used for the query, threshold, or any
        # target-day feature.
        sample_weight = np.ones(len(disagree), dtype=float)
        if args.positive_magnitude_weight > 0:
            positive_mag = disagree.loc[disagree.beneficial.eq(1), "target_spread"].abs()
            if len(positive_mag) >= 3:
                ranks = positive_mag.rank(pct=True).to_numpy(float)
                sample_weight[disagree.beneficial.to_numpy(bool)] += args.positive_magnitude_weight * ranks
        split=max(int(len(disagree)*0.70),1)
        fit, calib = disagree.iloc[:split], disagree.iloc[split:]
        med=fit[feature_cols].median().fillna(0); Xfit=fit[feature_cols].fillna(med); Xcal=calib[feature_cols].fillna(med)
        m=model(args.seed,args.gate_model); m.fit(Xfit,fit.beneficial, sample_weight=sample_weight[:split]); pcal=m.predict_proba(Xcal)[:,1]
        best=None
        for t in thresholds:
            selected=pcal>=t; rescue=int(((calib.y.to_numpy()==1)&selected).sum()); harm=int(((calib.y.to_numpy()==-1)&selected).sum())
            key=(rescue-harm,rescue,-harm,t)
            if best is None or key>best[0]: best=(key,t)
        threshold=float(best[1])
        med=disagree[feature_cols].median().fillna(0); m=model(args.seed,args.gate_model); m.fit(disagree[feature_cols].fillna(med),disagree.beneficial, sample_weight=sample_weight)
        eligible=(query.base_pred==-1)&(query.cand_pred==1)
        if args.require_confirmation_positive:
            eligible &= query.confirm_pred.eq(1)
        p_gate=np.zeros(len(query)); p_gate[eligible]=m.predict_proba(query.loc[eligible,feature_cols].fillna(med))[:,1] if eligible.any() else []
        override=eligible & (p_gate>=threshold) & (not args.disable_positive_override)
        veto = pd.Series(False, index=query.index)
        p_veto = np.zeros(len(query)); veto_threshold = np.nan
        if args.negative_veto:
            neg_disagree = history[(history.base_pred==1)&(history.cand_pred==-1)].copy()
            if len(neg_disagree) >= args.min_disagreements and neg_disagree.y.nunique() >= 2:
                neg_disagree["beneficial"] = (neg_disagree.y == -1).astype(int)
                ns = max(int(len(neg_disagree)*.70), 1); nf, nc = neg_disagree.iloc[:ns], neg_disagree.iloc[ns:]
                nmed = nf[feature_cols].median().fillna(0)
                nm = model(args.seed + 101, args.gate_model)
                nm.fit(nf[feature_cols].fillna(nmed), nf.beneficial)
                ncal = nm.predict_proba(nc[feature_cols].fillna(nmed))[:, 1]
                best_veto = None
                for t in thresholds:
                    selected = ncal >= t
                    rescue = int(((nc.y.to_numpy() == -1) & selected).sum())
                    harm = int(((nc.y.to_numpy() == 1) & selected).sum())
                    key = (rescue-harm, rescue, -harm, t)
                    if best_veto is None or key > best_veto[0]: best_veto = (key, t)
                veto_threshold = float(best_veto[1])
                nmed = neg_disagree[feature_cols].median().fillna(0)
                nm = model(args.seed + 101, args.gate_model)
                nm.fit(neg_disagree[feature_cols].fillna(nmed), neg_disagree.beneficial)
                eligible_veto = (query.base_pred==1) & (query.cand_pred==-1)
                if eligible_veto.any(): p_veto[eligible_veto] = nm.predict_proba(query.loc[eligible_veto,feature_cols].fillna(nmed))[:,1]
                veto = eligible_veto & (p_veto >= veto_threshold)
        pred=query.base_pred.to_numpy(int).copy(); pred[override.to_numpy()]=query.loc[override,"cand_pred"].to_numpy(int)
        pred[veto.to_numpy()] = query.loc[veto,"cand_pred"].to_numpy(int)
        for i,(_,r) in enumerate(query.iterrows()):
            predictions.append({"target_day":day.date().isoformat(),"hour_business":int(r.hour_business),"y_true":int(r.y),
                "baseline_pred":int(r.base_pred),"candidate_pred":int(r.cand_pred),"predicted_direction":int(pred[i]),
                "eligible_override":bool(eligible.iloc[i]),"override":bool(override.iloc[i]),"p_beneficial":float(p_gate[i]),
                "negative_veto":bool(veto.iloc[i]),"p_negative_veto":float(p_veto[i]),"negative_veto_threshold":veto_threshold,
                "threshold":threshold,"training_last_day":max(hist_days).date().isoformat()})
        audits.append({"target_day":day.date().isoformat(),"training_last_day":max(hist_days).date().isoformat(),
            "required_last_day":cutoff.date().isoformat(),"strict_ok":bool(max(hist_days)<=cutoff),
            "history_days":len(hist_days),"disagreements":len(disagree),"threshold":threshold,
            "negative_veto":args.negative_veto,"negative_veto_threshold":veto_threshold})
    pred=pd.DataFrame(predictions); audit=pd.DataFrame(audits)
    if pred.empty or not audit.strict_ok.all(): raise RuntimeError("no output or strict audit failed")
    rows=[]
    for name,col in [("baseline","baseline_pred"),("baseline_plus_positive_gate","predicted_direction")]:
        rows.append({"variant":name,**score(pred.y_true.to_numpy(int),pred[col].to_numpy(int))})
    pred["month"]=pred.target_day.str[:7]; monthly=[]
    for mth,g in pred.groupby("month"):
        for name,col in [("baseline","baseline_pred"),("baseline_plus_positive_gate","predicted_direction")]:
            monthly.append({"month":mth,"variant":name,**score(g.y_true.to_numpy(int),g[col].to_numpy(int))})
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    pred.to_csv(out/"predictions.csv",index=False,encoding="utf-8-sig"); audit.to_csv(out/"gate_audit.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out/"summary.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(monthly).to_csv(out/"monthly.csv",index=False,encoding="utf-8-sig")
    manifest={"status":"STRICT/PASS","route":"A_baseline_plus_positive_override","baseline":args.baseline,"candidate":args.candidate,
        "forecast_origin":"D-1 14:00","training_last_day":"per target <= D-2","target_day_actual_as_feature":False,
        "target_day_DA_as_feature":False,"d1_post14_spread_as_feature":False,"final_holdout_touched":False,
        "router_features":feature_cols,"feature_groups":requested_groups,"gate_model":args.gate_model,
        "all_expert_probs":args.all_expert_probs,"empirical_precision":args.empirical_precision,
        "confirmation_variant":args.confirmation_variant,
        "require_confirmation_positive":args.require_confirmation_positive,
        "positive_magnitude_weight":args.positive_magnitude_weight,
        "day_state_features":args.day_state_features,"negative_veto":args.negative_veto,
        "disable_positive_override":args.disable_positive_override,"pilot_only":True}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(monthly).to_string(index=False))
    return 0

if __name__=="__main__": raise SystemExit(main())
