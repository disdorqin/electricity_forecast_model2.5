from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import atomic_csv, atomic_json, atomic_parquet, load_p6_features  # noqa: E402
from integrate_paper_modules_p6 import (  # noqa: E402
    Q_LEVELS,
    add_quantile_derivatives,
    choose_consensus,
    choose_rule,
    direction_metrics,
    execute_consensus,
    execute_rule,
    lgb_classifier,
    lgb_quantile,
    period_report,
    quantile_feature_names,
    rescue_metrics,
)


@dataclass(frozen=True)
class Block:
    name: str
    start: str
    end: str
    r1_signals: str


def regime_model(seed: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=800, multi_class="multinomial", class_weight="balanced", C=0.35, random_state=seed)),
    ])


def safe_multiclass(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[features])
    classes = model.named_steps["logit"].classes_.astype(int)
    out = np.zeros((len(frame), 3), float)
    for j, c in enumerate(classes):
        out[:, c] = raw[:, j]
    return out


def strict_train_days(all_days: list[str], target_day: str, training_days: int) -> list[str]:
    idx = all_days.index(target_day)
    # At D-1 14:00, full labels for D-1 are NOT available. Use completed D-2 and earlier only.
    end = idx - 1  # slice end exclusive; last included index is idx-2 == D-2
    start = end - training_days
    if start < 0:
        raise ValueError(f"{target_day}: insufficient strict prehistory")
    days = all_days[start:end]
    if len(days) != training_days:
        raise ValueError(f"{target_day}: expected {training_days} strict train days, got {len(days)}")
    return days


def build_strict_quantiles(slot: pd.DataFrame, p6: list[str], all_days: list[str], days: list[str], training_days: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, day in enumerate(days, 1):
        train_days = strict_train_days(all_days, day, training_days)
        train = slot[slot["target_day"].isin(train_days)]
        test = slot[slot["target_day"].eq(day)].sort_values("hour_business")
        if len(test) != 24:
            continue
        y = train["target_spread"].to_numpy(float)
        p = test[["target_day", "hour_business"]].copy()
        for tau in Q_LEVELS:
            m = lgb_quantile(tau, seed)
            m.fit(train[p6], y)
            p[f"q{int(tau*100):02d}"] = m.predict(test[p6]).astype(float)
        rows.append(p)
        if i % 30 == 0:
            print(f"strict quantile {i}/{len(days)}: {day}")
    return add_quantile_derivatives(pd.concat(rows, ignore_index=True))


def build_strict_daily_predictions(slot_q: pd.DataFrame, p6: list[str], all_days: list[str], target_days: list[str], training_days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    qfeat = quantile_feature_names()
    rows = []
    audits = []
    for i, day in enumerate(target_days, 1):
        train_days = strict_train_days(all_days, day, training_days)
        train = slot_q[slot_q["target_day"].isin(train_days)].dropna(subset=qfeat)
        test = slot_q[slot_q["target_day"].eq(day)].sort_values("hour_business").dropna(subset=qfeat)
        if len(test) != 24:
            raise ValueError(f"{day}: strict test rows={len(test)}")
        y = train["target_spread"].to_numpy(float)
        sign = (y > 0).astype(int)

        base = lgb_classifier(seed).fit(train[p6], sign)
        base_prob = base.predict_proba(test[p6])[:, 1]

        e1_features = p6 + qfeat
        e1 = lgb_classifier(seed).fit(train[e1_features], sign)
        e1_prob = e1.predict_proba(test[e1_features])[:, 1]

        lo, hi = np.quantile(y, [0.05, 0.95])
        lower = lgb_classifier(seed).fit(train[p6], (y <= lo).astype(int))
        upper = lgb_classifier(seed).fit(train[p6], (y >= hi).astype(int))
        p_low = lower.predict_proba(test[p6])[:, 1]
        p_up = upper.predict_proba(test[p6])[:, 1]

        regime = np.where(y <= lo, 0, np.where(y >= hi, 2, 1)).astype(int)
        r3 = regime_model(seed).fit(train[p6], regime)
        p_reg = safe_multiclass(r3, test, p6)

        out = test[["target_day", "hour_business", "period", "target_spread", *qfeat]].copy()
        out = out.rename(columns={"target_spread": "y_true_spread"})
        out["base_p6_prob"] = base_prob
        out["e1_prob"] = e1_prob
        out["e1_direction"] = np.where(e1_prob >= 0.5, 1, -1)
        out["r2_p_lower_tail"] = p_low
        out["r2_p_upper_tail"] = p_up
        out["r3_p_negative"] = p_reg[:, 0]
        out["r3_p_regular"] = p_reg[:, 1]
        out["r3_p_positive"] = p_reg[:, 2]
        rows.append(out)

        target_ts = pd.Timestamp(day)
        cutoff = target_ts - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        train_last = pd.Timestamp(train_days[-1])
        audits.append({
            "target_day": day,
            "cutoff": cutoff,
            "strict_train_last_day": train_days[-1],
            "strict_train_last_day_le_D_minus_2": train_last <= target_ts - pd.Timedelta(days=2),
            "n_train_days": len(train_days),
        })
        if i % 20 == 0:
            print(f"strict daily {i}/{len(target_days)}: {day}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(audits)


def evaluate_block(frame: pd.DataFrame, block: Block) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    days = sorted(frame["target_day"].unique())
    if len(days) != 60:
        raise ValueError(f"{block.name}: expected60 got{len(days)}")
    design = frame[frame["target_day"].isin(days[:45])].copy()
    hold = frame[frame["target_day"].isin(days[45:])].copy()

    e2 = choose_rule(design, "E2_quantile_rescue")
    e3 = choose_rule(design, "E3_r1_gate")
    e4 = choose_rule(design, "E4_r3_rescue")
    e5 = choose_consensus(design, e2, e3, e4)
    rules = {"E2_quantile_rescue": e2, "E3_r1_gate": e3, "E4_r3_rescue": e4, "E5_consensus": e5}

    rows = []
    ledgers = []
    for split_name, part in [("design45", design), ("holdout15", hold)]:
        y = part["y_true_spread"].to_numpy(float)
        base = np.where(part["base_p6_prob"].to_numpy(float) >= 0.5, 1, -1)
        candidates = {
            "E0_P6_strict": base,
            "E1_P6_quantile_features": part["e1_direction"].to_numpy(int),
            "E2_quantile_rescue": execute_rule(part, e2),
            "E3_R1_period_confidence": execute_rule(part, e3),
            "E4_R3_positive_rescue": execute_rule(part, e4),
            "E5_consensus_rescue": execute_consensus(part, e5),
        }
        for name, pred in candidates.items():
            rows.append({"block": block.name, "split": split_name, "model": name,
                         **direction_metrics(y, pred), **rescue_metrics(y, base, pred)})
        l = part[["target_day", "hour_business", "period", "y_true_spread", "base_p6_prob", "e1_prob",
                  "q10", "q25", "q50", "q75", "q90", "q_width_80", "q_cross_zero", "q_positive_fraction",
                  "r1_p_positive", "r1_p_negative", "r2_p_upper_tail", "r2_p_lower_tail", "r3_p_positive", "r3_p_negative"]].copy()
        l["block"] = block.name
        l["split"] = split_name
        for name, pred in candidates.items():
            l[name] = pred
        ledgers.append(l)
    return pd.DataFrame(rows), rules, pd.concat(ledgers, ignore_index=True)


def strict_period_report(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = ["E0_P6_strict", "E1_P6_quantile_features", "E2_quantile_rescue", "E3_R1_period_confidence", "E4_R3_positive_rescue", "E5_consensus_rescue"]
    for (block, split, period), g in ledger.groupby(["block", "split", "period"], sort=False):
        y = g["y_true_spread"].to_numpy(float)
        base = g["E0_P6_strict"].to_numpy(int)
        for model in model_cols:
            pred = g[model].to_numpy(int)
            rows.append({"block": block, "split": split, "period": period, "model": model,
                         **direction_metrics(y, pred), **rescue_metrics(y, base, pred)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--output", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/p6_module_integration_strict_v2")
    ap.add_argument("--quantile-cache", default="")
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[4]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    blocks = [
        Block("A_early", "2026-04-17", "2026-06-15", "outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/adapted_14h_backtest_early60_v1/r1_causal_state_signals.parquet"),
        Block("B_late", "2026-06-16", "2026-08-14", "outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/adapted_14h_backtest_v1/r1_causal_state_signals.parquet"),
    ]

    slot, p6 = load_p6_features(root / args.cube_root)
    slot["target_day"] = slot["target_day"].astype(str)
    all_days = sorted(slot["target_day"].dropna().unique())
    first_idx = all_days.index(blocks[0].start)
    first_needed_idx = first_idx - args.training_days - 1
    if first_needed_idx < 0:
        raise ValueError("insufficient strict q history")
    needed_days = [d for d in all_days if all_days[first_needed_idx] <= d <= blocks[-1].end]

    if args.quantile_cache:
        cp = root / args.quantile_cache
        parts = sorted(cp.glob("qcache_*.parquet")) if cp.is_dir() else [cp]
        q = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        q["target_day"] = q["target_day"].astype(str)
        missing = sorted(set(needed_days) - set(q["target_day"].unique()))
        if missing:
            raise ValueError(f"strict quantile cache missing {len(missing)} days: {missing[:3]}")
        q = q[q["target_day"].isin(needed_days)].copy()
    else:
        q = build_strict_quantiles(slot, p6, all_days, needed_days, args.training_days, args.seed)
    atomic_parquet(out / "causal_quantile_predictions.parquet", q)

    slot_q = slot.merge(q, on=["target_day", "hour_business"], how="left", validate="one_to_one")
    target_days = [d for d in all_days if blocks[0].start <= d <= blocks[-1].end]
    pred, audit = build_strict_daily_predictions(slot_q, p6, all_days, target_days, args.training_days, args.seed)
    if not audit["strict_train_last_day_le_D_minus_2"].all():
        raise RuntimeError("strict D-2 train-label audit failed")
    atomic_parquet(out / "strict_daily_signals.parquet", pred)
    atomic_csv(out / "information_boundary_audit.csv", audit)

    summaries = []
    ledgers = []
    rules_all = {}
    for block in blocks:
        part = pred[(pred["target_day"] >= block.start) & (pred["target_day"] <= block.end)].copy()
        r1 = pd.read_parquet(root / block.r1_signals)
        r1["target_day"] = r1["target_day"].astype(str)
        r1cols = ["target_day", "hour_business", "r1_p_negative", "r1_p_neutral", "r1_p_positive", "r1_state_entropy", "r1_state_expected_spread"]
        part = part.merge(r1[r1cols], on=["target_day", "hour_business"], how="left", validate="one_to_one")
        if part[r1cols[2:]].isna().any().any():
            raise RuntimeError(f"{block.name}: missing R1 signals")
        s, rules, ledger = evaluate_block(part, block)
        summaries.append(s)
        ledgers.append(ledger)
        rules_all[block.name] = rules
        print(f"\n{block.name}\n{s.to_string(index=False)}")

    summary = pd.concat(summaries, ignore_index=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    periods = strict_period_report(ledger)
    atomic_csv(out / "summary.csv", summary)
    atomic_parquet(out / "ledger.parquet", ledger)
    atomic_csv(out / "period_metrics.csv", periods)
    atomic_json(out / "selected_rules.json", rules_all)

    hold = summary[summary["split"].eq("holdout15")].copy()
    base = hold[hold["model"].eq("E0_P6_strict")][["block", "direction_accuracy", "balanced_direction_accuracy"]].rename(
        columns={"direction_accuracy": "base_direction", "balanced_direction_accuracy": "base_balanced"})
    accept = hold.merge(base, on="block", how="left")
    accept["delta_direction_pp"] = 100 * (accept["direction_accuracy"] - accept["base_direction"])
    accept["delta_balanced_pp"] = 100 * (accept["balanced_direction_accuracy"] - accept["base_balanced"])
    atomic_csv(out / "holdout_acceptance.csv", accept)

    stable = accept[accept["model"].ne("E0_P6_strict")].groupby("model").agg(
        blocks=("block", "nunique"),
        min_delta_direction_pp=("delta_direction_pp", "min"),
        min_delta_balanced_pp=("delta_balanced_pp", "min"),
        mean_direction=("direction_accuracy", "mean"),
        mean_balanced=("balanced_direction_accuracy", "mean"),
        mean_positive=("positive_accuracy", "mean"),
        mean_negative=("negative_accuracy", "mean"),
    ).reset_index()
    stable["improves_both_blocks"] = (stable["min_delta_direction_pp"] > 0) & (stable["min_delta_balanced_pp"] > 0)
    atomic_csv(out / "stability_summary.csv", stable)

    manifest = {
        "status": "complete",
        "forecast_origin": "D-1 14:00",
        "strict_training_label_boundary": "full training labels end at D-2; D-1 p1-p14 only enter via prebuilt current-context features",
        "training_days": args.training_days,
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
        "stable_candidates": stable.to_dict("records"),
    }
    atomic_json(out / "manifest.json", manifest)
    print("\nSTRICT HOLDOUT ACCEPTANCE")
    print(accept.sort_values(["block", "balanced_direction_accuracy"], ascending=[True, False]).to_string(index=False))
    print("\nSTRICT STABILITY")
    print(stable.sort_values(["improves_both_blocks", "mean_balanced"], ascending=[False, False]).to_string(index=False))
    print(json.dumps({"runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
