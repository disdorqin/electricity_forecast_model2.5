"""Iteration 7: strict recurring-vs-emergent drift routing for P6 direct-spread experts.

Motivation:
- Recurring drift: causal Similar-Day (SD20) can recover previously seen market states.
- Emergent / local drift: the D-1 p1-p14 visible-context expert reacts to the latest state.
- Stable generalist: base P6 provides a fallback.

This runner DOES NOT retrain the three base experts. It consumes only strict-D2 OOS
expert ledgers already produced in the experiment area, then performs prequential
routing. For target day D, all router fitting/calibration labels are restricted to
OOS days <= D-2. D-1 labels are never used.

Fresh final holdout 2026-08-15..2026-08-21 remains untouched.
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
    add_similar_day_features,
)

ROOT = Path("outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822")
BASE_DIRS = [
    "cross_month_champion_audit_strictD2_q1",
    "cross_month_champion_audit_strictD2_q2",
    "cross_month_champion_audit_q3_strictD2",
]
CTX_DIRS = [
    "iteration6_visible_d1_context_q1_strictD2",
    "iteration6_visible_d1_context_q2_strictD2",
    "iteration6_visible_d1_context_q3_strictD2",
]


def atomic_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def metrics(g: pd.DataFrame, pred_col: str = "predicted_direction") -> dict:
    y = np.sign(g["target_spread"].to_numpy(float))
    p = g[pred_col].to_numpy(int)
    ok = y == p
    pos = y > 0
    neg = y < 0
    pa = float(ok[pos].mean()) if pos.any() else math.nan
    na = float(ok[neg].mean()) if neg.any() else math.nan
    return {
        "days": int(g.target_day.nunique()),
        "n": int(len(g)),
        "direction_accuracy": float(ok.mean()),
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
        "all_negative_accuracy": float(neg.mean()),
    }


def load_experts(root: Path) -> pd.DataFrame:
    base_parts = []
    for d in BASE_DIRS:
        x = pd.read_parquet(root / d / "ledger.parquet")
        if not x["training_last_day"].le((pd.to_datetime(x["target_day"]) - pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")).all():
            raise RuntimeError(f"strict training boundary failed: {d}")
        base_parts.append(x[x.variant.isin(["P6_w90", "P6_SD20_w90"])].copy())
    ctx_parts = []
    for d in CTX_DIRS:
        x = pd.read_parquet(root / d / "ledger.parquet")
        if not x["training_last_day"].le((pd.to_datetime(x["target_day"]) - pd.Timedelta(days=2)).dt.strftime("%Y-%m-%d")).all():
            raise RuntimeError(f"strict training boundary failed: {d}")
        ctx_parts.append(x[x.variant.eq("P6_CTX_bal_w90")].copy())
    b = pd.concat(base_parts, ignore_index=True)
    c = pd.concat(ctx_parts, ignore_index=True)

    frames = []
    for name, df in [
        ("p6", b[b.variant.eq("P6_w90")]),
        ("sd", b[b.variant.eq("P6_SD20_w90")]),
        ("ctx", c),
    ]:
        z = df[["target_day", "hour_business", "period", "target_spread", "prob_positive", "predicted_direction"]].copy()
        z = z.rename(columns={"prob_positive": f"{name}_p", "predicted_direction": f"{name}_d"})
        frames.append(z)
    m = frames[0]
    for z in frames[1:]:
        m = m.merge(z, on=["target_day", "hour_business", "period", "target_spread"], how="inner", validate="one_to_one")
    return m.sort_values(["target_day", "hour_business"]).reset_index(drop=True)


def build_meta_features(root: Path, expert: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cube = root / "feature_cube"
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    slot, sd = add_similar_day_features(slot, groups, k_values=(20,), lookback_days=365)
    if sd["audit"].empty or not sd["audit"]["causal_ok"].all():
        raise RuntimeError("similar-day causal audit failed")

    wanted = [
        "target_day", "hour_business",
        "ctx_spread_mean14", "ctx_spread_std14", "ctx_spread_median14",
        "ctx_spread_last", "ctx_spread_mean3", "ctx_spread_range14",
        "ctx_spread_absmean14", "ctx_spread_positive_rate14", "ctx_spread_slope14",
        "residual_load_renew", "renewable_share", "bidding_space_ratio", "interconnect_share",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "sd20_spread_mean", "sd20_spread_median", "sd20_positive_rate",
        "sd20_weighted_positive_rate", "sd20_spread_std", "sd20_mean_distance",
        "sd20_day_positive_rate", "sd20_1_8_positive_rate", "sd20_9_16_positive_rate", "sd20_17_24_positive_rate",
    ]
    missing = [c for c in wanted if c not in slot.columns]
    if missing:
        raise RuntimeError(f"missing legal meta features: {missing}")
    x = expert.merge(slot[wanted], on=["target_day", "hour_business"], how="left", validate="one_to_one")
    x["p6_conf"] = 2 * np.abs(x.p6_p - 0.5)
    x["sd_conf"] = 2 * np.abs(x.sd_p - 0.5)
    x["ctx_conf"] = 2 * np.abs(x.ctx_p - 0.5)
    x["sd_ctx_gap"] = np.abs(x.sd_p - x.ctx_p)
    x["sd_p6_gap"] = np.abs(x.sd_p - x.p6_p)
    x["ctx_p6_gap"] = np.abs(x.ctx_p - x.p6_p)
    x["sd_ctx_disagree"] = (x.sd_d != x.ctx_d).astype(float)
    x["p6_votes_sd"] = (x.p6_d == x.sd_d).astype(float)
    x["p6_votes_ctx"] = (x.p6_d == x.ctx_d).astype(float)
    return x, sd["audit"]


def make_gate(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=120,
        learning_rate=0.04, num_leaves=15, max_depth=5, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        random_state=seed, n_jobs=4, verbosity=-1,
    )


def choose_distance_rule(hist: pd.DataFrame, q: pd.DataFrame) -> np.ndarray:
    """Prequential simple branch: low analog distance -> SD, otherwise choose historical better general/local expert."""
    # Day-level distance is repeated across 24 slots; get historical distribution only.
    dist = pd.to_numeric(hist["sd20_mean_distance"], errors="coerce")
    candidates = [0.25, 0.50, 0.75]
    best = None
    y = np.sign(hist.target_spread.to_numpy(float))
    for qq in candidates:
        th = float(dist.quantile(qq))
        low = dist.to_numpy(float) <= th
        # On high-distance historical rows choose whichever fallback expert was better historically.
        p6_acc = float(np.mean(hist.p6_d.to_numpy()[~low] == y[~low])) if (~low).any() else 0.0
        ctx_acc = float(np.mean(hist.ctx_d.to_numpy()[~low] == y[~low])) if (~low).any() else 0.0
        fallback = "ctx" if ctx_acc >= p6_acc else "p6"
        pred = np.where(low, hist.sd_d.to_numpy(), hist[f"{fallback}_d"].to_numpy())
        score = float(np.mean(pred == y))
        if best is None or score > best[0]:
            best = (score, th, fallback)
    assert best is not None
    _, th, fallback = best
    low_q = pd.to_numeric(q["sd20_mean_distance"], errors="coerce").to_numpy(float) <= th
    return np.where(low_q, q.sd_d.to_numpy(int), q[f"{fallback}_d"].to_numpy(int))


def run_router(x: pd.DataFrame, start: str, end: str, history_days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "p6_p", "sd_p", "ctx_p", "p6_conf", "sd_conf", "ctx_conf",
        "sd_ctx_gap", "sd_p6_gap", "ctx_p6_gap", "p6_votes_sd", "p6_votes_ctx",
        "ctx_spread_mean14", "ctx_spread_std14", "ctx_spread_median14", "ctx_spread_last",
        "ctx_spread_mean3", "ctx_spread_range14", "ctx_spread_absmean14",
        "ctx_spread_positive_rate14", "ctx_spread_slope14",
        "residual_load_renew", "renewable_share", "bidding_space_ratio", "interconnect_share",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "sd20_spread_mean", "sd20_spread_median", "sd20_positive_rate",
        "sd20_weighted_positive_rate", "sd20_spread_std", "sd20_mean_distance",
        "sd20_day_positive_rate", "sd20_1_8_positive_rate", "sd20_9_16_positive_rate", "sd20_17_24_positive_rate",
    ]
    all_days = sorted(x.target_day.unique())
    target_days = [d for d in all_days if start <= d <= end]
    rows = []
    audits = []
    for i, day in enumerate(target_days, 1):
        cutoff_day = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        hist_days = [d for d in all_days if d <= cutoff_day][-history_days:]
        if len(hist_days) < 60:
            continue
        hist = x[x.target_day.isin(hist_days)].copy()
        q = x[x.target_day.eq(day)].copy()
        if len(q) != 24:
            raise RuntimeError(f"{day}: expected 24 rows, got {len(q)}")

        # Branch A: explicit disagreement gate between recurring(SD) and local-context(CTX).
        dis_hist = hist[hist.sd_d != hist.ctx_d].copy()
        if len(dis_hist) < 100:
            raise RuntimeError(f"{day}: insufficient disagreement history {len(dis_hist)}")
        gate_y = (np.sign(dis_hist.target_spread.to_numpy(float)) == dis_hist.sd_d.to_numpy(int)).astype(int)
        gate = make_gate(seed)
        gate.fit(dis_hist[feature_cols], gate_y)
        gate_prob = gate.predict_proba(q[feature_cols])[:, 1]
        gate_pred = np.where(q.sd_d.to_numpy() == q.ctx_d.to_numpy(), q.sd_d.to_numpy(), np.where(gate_prob >= 0.5, q.sd_d.to_numpy(), q.ctx_d.to_numpy()))

        # Branch B: direct strict meta-stacker over all rows. It may learn to trust P6 as stable fallback.
        stack = make_gate(seed + 17)
        stack.fit(hist[feature_cols], (hist.target_spread.to_numpy(float) > 0).astype(int))
        stack_prob = stack.predict_proba(q[feature_cols])[:, 1]
        stack_pred = np.where(stack_prob >= 0.5, 1, -1)

        # Simple recurrence-distance rule as a low-capacity guardrail.
        dist_pred = choose_distance_rule(hist, q)

        # Static equal probability blend reference.
        blend_prob = (q.p6_p.to_numpy(float) + q.sd_p.to_numpy(float) + q.ctx_p.to_numpy(float)) / 3.0
        blend_pred = np.where(blend_prob >= 0.5, 1, -1)

        for name, pred, prob in [
            ("A_disagreement_gate", gate_pred, gate_prob),
            ("B_meta_stack", stack_pred, stack_prob),
            ("B_recurrence_distance_rule", dist_pred, np.full(24, np.nan)),
            ("equal_prob_blend", blend_pred, blend_prob),
            ("P6_static", q.p6_d.to_numpy(int), q.p6_p.to_numpy(float)),
            ("SD20_static", q.sd_d.to_numpy(int), q.sd_p.to_numpy(float)),
            ("CTX_static", q.ctx_d.to_numpy(int), q.ctx_p.to_numpy(float)),
        ]:
            o = q[["target_day", "hour_business", "period", "target_spread"]].copy()
            o["variant"] = name
            o["predicted_direction"] = pred
            o["prob_positive"] = prob
            o["router_training_last_day"] = hist_days[-1]
            rows.append(o)
        audits.append({
            "target_day": day,
            "router_training_last_day": hist_days[-1],
            "required_last_day_le": cutoff_day,
            "strict_ok": hist_days[-1] <= cutoff_day,
            "history_days": len(hist_days),
            "disagreement_rows": len(dis_hist),
        })
        if i % 30 == 0:
            print(f"router {i}/{len(target_days)}: {day}")
    audit = pd.DataFrame(audits)
    if audit.empty or not audit.strict_ok.all():
        raise RuntimeError("router strict-D2 audit failed")
    return pd.concat(rows, ignore_index=True), audit


def summarize(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = ledger.copy(); z["month"] = z.target_day.str[:7]
    monthly = []
    for (v, m), g in z.groupby(["variant", "month"], sort=True):
        monthly.append({"variant": v, "month": m, **metrics(g)})
    monthly = pd.DataFrame(monthly)
    rob = []
    for v, g in monthly.groupby("variant", sort=False):
        acc = g.direction_accuracy.astype(float); bal = g.balanced_direction_accuracy.astype(float); gain = acc - g.all_negative_accuracy.astype(float)
        rob.append({
            "variant": v, "months": len(g), "mean_month_acc": acc.mean(), "median_month_acc": acc.median(),
            "min_month_acc": acc.min(), "max_month_acc": acc.max(), "std_month_acc": acc.std(ddof=0),
            "months_ge_065": int((acc >= .65).sum()), "months_ge_070": int((acc >= .70).sum()),
            "mean_month_bal": bal.mean(), "min_month_bal": bal.min(),
            "months_beating_all_negative": int((gain > 0).sum()), "mean_gain_vs_all_negative": gain.mean(),
        })
    robust = pd.DataFrame(rob).sort_values(["mean_month_acc", "mean_month_bal"], ascending=False)
    return monthly, robust


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--history-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration7_recurring_emergent_router_strictD2")
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.output_root)
    if args.end >= "2026-08-15":
        raise RuntimeError("fresh final holdout remains sealed; end must be <= 2026-08-14")

    expert = load_experts(root)
    x, sd_audit = build_meta_features(root, expert)
    ledger, router_audit = run_router(x, args.start, args.end, args.history_days, args.seed)
    monthly, robust = summarize(ledger)

    atomic_parquet(out / "ledger.parquet", ledger)
    atomic_csv(out / "monthly.csv", monthly)
    atomic_csv(out / "robustness.csv", robust)
    atomic_csv(out / "router_training_audit.csv", router_audit)
    atomic_csv(out / "similar_day_causal_audit.csv", sd_audit)

    # ORACLE is diagnostic only: upper bound if any of three experts is correct.
    eval_x = x[(x.target_day >= args.start) & (x.target_day <= args.end)].copy()
    y = np.sign(eval_x.target_spread.to_numpy(float))
    corr = np.column_stack([eval_x.p6_d.to_numpy() == y, eval_x.sd_d.to_numpy() == y, eval_x.ctx_d.to_numpy() == y])
    oracle = {
        "status": "ORACLE_DIAGNOSTIC_ONLY",
        "any_expert_correct": float(corr.any(axis=1).mean()),
        "all_three_same_direction_rate": float(((eval_x.p6_d == eval_x.sd_d) & (eval_x.sd_d == eval_x.ctx_d)).mean()),
    }
    atomic_json(out / "manifest.json", {
        "status": "complete",
        "experiment": "iteration7_recurring_emergent_router",
        "forecast_origin": "D-1 14:00",
        "base_expert_ledgers": "STRICT-D2 only",
        "router_training_labels": "historical OOS days <= D-2 only",
        "target_day_actual_features": False,
        "target_day_DA_features": False,
        "d1_post14_realized_features": False,
        "fresh_final_holdout_reserved": ["2026-08-15", "2026-08-21"],
        "final_holdout_touched": False,
        "literature_basis": [
            "DynaME (WWW 2026): recurring drift vs emergent drift, specialized experts + general expert + dynamic gate",
            "Recurrent regimes / structural breaks in electricity prices: relevant calibration data need not be most recent",
        ],
        "oracle": oracle,
    })
    print("\nROBUSTNESS\n", robust.to_string(index=False))
    print("\nMONTHLY TOP\n", monthly[monthly.variant.isin(robust.head(3).variant)].to_string(index=False))
    print("\nORACLE\n", oracle)


if __name__ == "__main__":
    main()
