from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import atomic_csv, atomic_json, atomic_parquet, load_p6_features  # noqa: E402

Q_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
BASE_GRID = (0.25, 0.30, 0.35, 0.40, 0.45)
R1_GRID = (0.50, 0.60, 0.70, 0.80)
R3_GRID = (0.50, 0.60, 0.70, 0.80)
R2_GRID = (0.15, 0.25, 0.35, 0.45)
QFRAC_GRID = (0.60, 0.80, 1.00)
PERIODS = ("ALL", "1_8", "9_16", "17_24")


@dataclass(frozen=True)
class Block:
    name: str
    start: str
    end: str
    source_ledger: str


def lgb_classifier(seed: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=4,
        random_state=seed,
    )


def lgb_quantile(tau: float, seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=tau,
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=4,
        random_state=seed,
    )


def direction_metrics(y_true: np.ndarray, pred_dir: np.ndarray) -> dict:
    y = np.sign(np.asarray(y_true, float))
    p = np.asarray(pred_dir, int)
    eligible = y != 0
    correct = y == p
    pos = y > 0
    neg = y < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "n_slots": int(len(y)),
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
    }


def rescue_metrics(y_true: np.ndarray, base_dir: np.ndarray, cand_dir: np.ndarray) -> dict:
    y = np.sign(np.asarray(y_true, float))
    b = np.asarray(base_dir, int)
    c = np.asarray(cand_dir, int)
    changed = b != c
    rescued_pos = int(np.sum(changed & (y > 0) & (b < 0) & (c > 0)))
    harmed_neg = int(np.sum(changed & (y < 0) & (b < 0) & (c > 0)))
    efficiency = math.inf if harmed_neg == 0 and rescued_pos > 0 else (rescued_pos / harmed_neg if harmed_neg else 0.0)
    return {
        "n_flips": int(changed.sum()),
        "rescued_positive": rescued_pos,
        "harmed_negative": harmed_neg,
        "rescue_efficiency": efficiency,
    }


def score_metrics(m: dict) -> float:
    # Equal emphasis on raw and balanced direction accuracy.
    return 0.5 * (m["direction_accuracy"] + m["balanced_direction_accuracy"])


def period_mask(frame: pd.DataFrame, period: str) -> np.ndarray:
    if period == "ALL":
        return np.ones(len(frame), dtype=bool)
    return frame["period"].astype(str).eq(period).to_numpy()


def quantile_feature_names() -> list[str]:
    names = [f"q{int(t*100):02d}" for t in Q_LEVELS]
    return names + ["q_width_80", "q_cross_zero", "q_positive_fraction", "q_median_abs"]


def add_quantile_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    qcols = [f"q{int(t*100):02d}" for t in Q_LEVELS]
    out["q_width_80"] = out["q90"] - out["q10"]
    out["q_cross_zero"] = ((out["q10"] < 0) & (out["q90"] > 0)).astype(float)
    out["q_positive_fraction"] = (out[qcols].to_numpy(float) > 0).mean(axis=1)
    out["q_median_abs"] = out["q50"].abs()
    return out


def build_causal_quantiles(slot: pd.DataFrame, p6: list[str], all_days: list[str], needed_days: list[str], training_days: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for i, day in enumerate(needed_days, 1):
        idx = all_days.index(day)
        if idx < training_days:
            continue
        train_days = all_days[idx - training_days: idx]
        train = slot[slot["target_day"].isin(train_days)]
        test = slot[slot["target_day"].eq(day)].sort_values("hour_business")
        if len(test) != 24:
            continue
        y = train["target_spread"].to_numpy(float)
        pred = test[["target_day", "hour_business"]].copy()
        for tau in Q_LEVELS:
            model = lgb_quantile(tau, seed)
            model.fit(train[p6], y)
            pred[f"q{int(tau*100):02d}"] = model.predict(test[p6]).astype(float)
        rows.append(pred)
        if i % 30 == 0:
            print(f"quantile OOF {i}/{len(needed_days)}: {day}")
    q = pd.concat(rows, ignore_index=True)
    return add_quantile_derivatives(q)


def build_e1_predictions(slot_q: pd.DataFrame, p6: list[str], all_days: list[str], target_days: list[str], training_days: int, seed: int) -> pd.DataFrame:
    qfeat = quantile_feature_names()
    features = p6 + qfeat
    rows = []
    for i, day in enumerate(target_days, 1):
        idx = all_days.index(day)
        train_days = all_days[idx - training_days: idx]
        train = slot_q[slot_q["target_day"].isin(train_days)].dropna(subset=qfeat)
        test = slot_q[slot_q["target_day"].eq(day)].sort_values("hour_business").dropna(subset=qfeat)
        if len(test) != 24:
            raise ValueError(f"E1 {day}: expected 24 test slots, got {len(test)}")
        y = train["target_spread"].to_numpy(float)
        model = lgb_classifier(seed)
        model.fit(train[features], (y > 0).astype(int))
        p = model.predict_proba(test[features])[:, 1]
        out = test[["target_day", "hour_business"]].copy()
        out["e1_prob"] = p
        out["e1_direction"] = np.where(p >= 0.5, 1, -1)
        rows.append(out)
        if i % 30 == 0:
            print(f"E1 {i}/{len(target_days)}: {day}")
    return pd.concat(rows, ignore_index=True)


def apply_positive_rescue(frame: pd.DataFrame, *, base_low: float, signal: np.ndarray, signal_threshold: float, period: str = "ALL") -> np.ndarray:
    base_prob = frame["base_p6_prob"].to_numpy(float)
    base_dir = np.where(base_prob >= 0.5, 1, -1)
    gray = (base_prob >= base_low) & (base_prob < 0.5)
    gate = gray & (np.asarray(signal, float) >= signal_threshold) & period_mask(frame, period)
    return np.where(gate, 1, base_dir)


def choose_rule(frame: pd.DataFrame, kind: str) -> dict:
    y = frame["y_true_spread"].to_numpy(float)
    base_dir = np.where(frame["base_p6_prob"].to_numpy(float) >= 0.5, 1, -1)
    base_m = direction_metrics(y, base_dir)
    best = {"kind": kind, "no_op": True, "score": score_metrics(base_m), "metrics": base_m, "params": {}}

    if kind == "E2_quantile_rescue":
        for low in BASE_GRID:
            for frac in QFRAC_GRID:
                cand = apply_positive_rescue(frame, base_low=low, signal=frame["q_positive_fraction"].to_numpy(float), signal_threshold=frac)
                m = direction_metrics(y, cand)
                s = score_metrics(m)
                if s > best["score"] + 1e-12:
                    best = {"kind": kind, "no_op": False, "score": s, "metrics": m, "params": {"base_low": low, "q_positive_fraction": frac}}

    elif kind == "E3_r1_gate":
        for low in BASE_GRID:
            for thr in R1_GRID:
                cand = apply_positive_rescue(frame, base_low=low, signal=frame["r1_p_positive"].to_numpy(float), signal_threshold=thr, period="1_8")
                m = direction_metrics(y, cand)
                s = score_metrics(m)
                if s > best["score"] + 1e-12:
                    best = {"kind": kind, "no_op": False, "score": s, "metrics": m, "params": {"base_low": low, "r1_positive": thr, "period": "1_8"}}

    elif kind == "E4_r3_rescue":
        for low in BASE_GRID:
            for thr in R3_GRID:
                for period in PERIODS:
                    cand = apply_positive_rescue(frame, base_low=low, signal=frame["r3_p_positive"].to_numpy(float), signal_threshold=thr, period=period)
                    m = direction_metrics(y, cand)
                    s = score_metrics(m)
                    if s > best["score"] + 1e-12:
                        best = {"kind": kind, "no_op": False, "score": s, "metrics": m, "params": {"base_low": low, "r3_positive": thr, "period": period}}
    else:
        raise ValueError(kind)
    return best


def execute_rule(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    base_prob = frame["base_p6_prob"].to_numpy(float)
    base_dir = np.where(base_prob >= 0.5, 1, -1)
    if rule.get("no_op", False):
        return base_dir
    p = rule["params"]
    if rule["kind"] == "E2_quantile_rescue":
        return apply_positive_rescue(frame, base_low=p["base_low"], signal=frame["q_positive_fraction"].to_numpy(float), signal_threshold=p["q_positive_fraction"])
    if rule["kind"] == "E3_r1_gate":
        return apply_positive_rescue(frame, base_low=p["base_low"], signal=frame["r1_p_positive"].to_numpy(float), signal_threshold=p["r1_positive"], period=p["period"])
    if rule["kind"] == "E4_r3_rescue":
        return apply_positive_rescue(frame, base_low=p["base_low"], signal=frame["r3_p_positive"].to_numpy(float), signal_threshold=p["r3_positive"], period=p["period"])
    raise ValueError(rule["kind"])


def choose_consensus(frame: pd.DataFrame, e2: dict, e3: dict, e4: dict) -> dict:
    y = frame["y_true_spread"].to_numpy(float)
    base_prob = frame["base_p6_prob"].to_numpy(float)
    base_dir = np.where(base_prob >= 0.5, 1, -1)
    base_m = direction_metrics(y, base_dir)
    best = {"kind": "E5_consensus", "no_op": True, "score": score_metrics(base_m), "metrics": base_m, "params": {}}

    # Reuse the thresholds already selected for E2/E3/E4; only tune the consensus strength and R2 tail threshold.
    e2p = e2.get("params", {})
    e3p = e3.get("params", {})
    e4p = e4.get("params", {})
    q_thr = e2p.get("q_positive_fraction", 0.8)
    r1_thr = e3p.get("r1_positive", 0.7)
    r3_thr = e4p.get("r3_positive", 0.7)
    r1_period = e3p.get("period", "1_8")
    r3_period = e4p.get("period", "ALL")

    for low in BASE_GRID:
        gray = (base_prob >= low) & (base_prob < 0.5)
        for r2_thr in R2_GRID:
            sig_q = frame["q_positive_fraction"].to_numpy(float) >= q_thr
            sig_r1 = (frame["r1_p_positive"].to_numpy(float) >= r1_thr) & period_mask(frame, r1_period)
            sig_r3 = (frame["r3_p_positive"].to_numpy(float) >= r3_thr) & period_mask(frame, r3_period)
            sig_r2 = frame["r2_p_upper_tail"].to_numpy(float) >= r2_thr
            votes = sig_q.astype(int) + sig_r1.astype(int) + sig_r2.astype(int) + sig_r3.astype(int)
            for min_votes in (2, 3):
                cand = np.where(gray & (votes >= min_votes), 1, base_dir)
                m = direction_metrics(y, cand)
                s = score_metrics(m)
                if s > best["score"] + 1e-12:
                    best = {
                        "kind": "E5_consensus", "no_op": False, "score": s, "metrics": m,
                        "params": {"base_low": low, "r2_upper_tail": r2_thr, "min_votes": min_votes,
                                   "q_positive_fraction": q_thr, "r1_positive": r1_thr, "r1_period": r1_period,
                                   "r3_positive": r3_thr, "r3_period": r3_period},
                    }
    return best


def execute_consensus(frame: pd.DataFrame, rule: dict) -> np.ndarray:
    base_prob = frame["base_p6_prob"].to_numpy(float)
    base_dir = np.where(base_prob >= 0.5, 1, -1)
    if rule.get("no_op", False):
        return base_dir
    p = rule["params"]
    gray = (base_prob >= p["base_low"]) & (base_prob < 0.5)
    sig_q = frame["q_positive_fraction"].to_numpy(float) >= p["q_positive_fraction"]
    sig_r1 = (frame["r1_p_positive"].to_numpy(float) >= p["r1_positive"]) & period_mask(frame, p["r1_period"])
    sig_r3 = (frame["r3_p_positive"].to_numpy(float) >= p["r3_positive"]) & period_mask(frame, p["r3_period"])
    sig_r2 = frame["r2_p_upper_tail"].to_numpy(float) >= p["r2_upper_tail"]
    votes = sig_q.astype(int) + sig_r1.astype(int) + sig_r2.astype(int) + sig_r3.astype(int)
    return np.where(gray & (votes >= p["min_votes"]), 1, base_dir)


def evaluate_block(frame: pd.DataFrame, block: Block) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    days = sorted(frame["target_day"].unique())
    if len(days) != 60:
        raise ValueError(f"{block.name}: expected 60 days, got {len(days)}")
    design_days = days[:45]
    hold_days = days[45:]
    design = frame[frame["target_day"].isin(design_days)].copy()
    hold = frame[frame["target_day"].isin(hold_days)].copy()

    e2 = choose_rule(design, "E2_quantile_rescue")
    e3 = choose_rule(design, "E3_r1_gate")
    e4 = choose_rule(design, "E4_r3_rescue")
    e5 = choose_consensus(design, e2, e3, e4)
    rules = {"E2_quantile_rescue": e2, "E3_r1_gate": e3, "E4_r3_rescue": e4, "E5_consensus": e5}

    rows = []
    ledgers = []
    for split_name, part in [("design45", design), ("holdout15", hold)]:
        y = part["y_true_spread"].to_numpy(float)
        base_dir = np.where(part["base_p6_prob"].to_numpy(float) >= 0.5, 1, -1)
        e1_dir = part["e1_direction"].to_numpy(int)
        candidates = {
            "E0_P6": base_dir,
            "E1_P6_quantile_features": e1_dir,
            "E2_quantile_rescue": execute_rule(part, e2),
            "E3_R1_period_confidence": execute_rule(part, e3),
            "E4_R3_positive_rescue": execute_rule(part, e4),
            "E5_consensus_rescue": execute_consensus(part, e5),
        }
        for name, pred in candidates.items():
            m = direction_metrics(y, pred)
            r = rescue_metrics(y, base_dir, pred)
            rows.append({"block": block.name, "split": split_name, "model": name, **m, **r})
        l = part[["target_day", "hour_business", "period", "y_true_spread", "base_p6_prob", "e1_prob",
                  "q10", "q25", "q50", "q75", "q90", "q_width_80", "q_cross_zero", "q_positive_fraction",
                  "r1_p_positive", "r1_p_negative", "r2_p_upper_tail", "r2_p_lower_tail", "r3_p_positive", "r3_p_negative"]].copy()
        l["block"] = block.name
        l["split"] = split_name
        for name, pred in candidates.items():
            l[name] = pred
        ledgers.append(l)
    return pd.DataFrame(rows), rules, pd.concat(ledgers, ignore_index=True)


def period_report(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = ["E0_P6", "E1_P6_quantile_features", "E2_quantile_rescue", "E3_R1_period_confidence", "E4_R3_positive_rescue", "E5_consensus_rescue"]
    for (block, split, period), g in ledger.groupby(["block", "split", "period"], sort=False):
        y = g["y_true_spread"].to_numpy(float)
        base = g["E0_P6"].to_numpy(int)
        for model in model_cols:
            pred = g[model].to_numpy(int)
            rows.append({"block": block, "split": split, "period": period, "model": model,
                         **direction_metrics(y, pred), **rescue_metrics(y, base, pred)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Integrate transferable R1/R2/R3 modules into P6 under strict D-1 14:00 contract.")
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--output", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/p6_module_integration_v1")
    ap.add_argument("--quantile-cache", default="")
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[4]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    blocks = [
        Block("A_early", "2026-04-17", "2026-06-15", "outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/adapted_14h_backtest_early60_v1/ledger.parquet"),
        Block("B_late", "2026-06-16", "2026-08-14", "outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/adapted_14h_backtest_v1/ledger.parquet"),
    ]

    slot, p6 = load_p6_features(root / args.cube_root)
    slot["target_day"] = slot["target_day"].astype(str)
    all_days = sorted(slot["target_day"].dropna().unique())
    earliest_idx = all_days.index(blocks[0].start) - args.training_days
    if earliest_idx < 0:
        raise ValueError("insufficient quantile prehistory")
    needed_days = [d for d in all_days if all_days[earliest_idx] <= d <= blocks[-1].end]

    if args.quantile_cache:
        cache_path = root / args.quantile_cache
        if cache_path.is_dir():
            parts = sorted(cache_path.glob("qcache_*.parquet"))
            if not parts:
                raise ValueError(f"no qcache_*.parquet under {cache_path}")
            q = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        else:
            q = pd.read_parquet(cache_path)
        q["target_day"] = q["target_day"].astype(str)
        expected = set(needed_days)
        available = set(q["target_day"].unique())
        missing_days = sorted(expected - available)
        if missing_days:
            raise ValueError(f"quantile cache missing {len(missing_days)} days, first={missing_days[:3]}")
        q = q[q["target_day"].isin(needed_days)].copy()
    else:
        q = build_causal_quantiles(slot, p6, all_days, needed_days, args.training_days, args.seed)
    atomic_parquet(out / "causal_quantile_predictions.parquet", q)
    slot_q = slot.merge(q, on=["target_day", "hour_business"], how="left", validate="one_to_one")
    target_days = [d for d in all_days if blocks[0].start <= d <= blocks[-1].end]
    e1_path = out / "e1_quantile_augmented_predictions.parquet"
    if e1_path.exists():
        e1 = pd.read_parquet(e1_path)
        e1["target_day"] = e1["target_day"].astype(str)
        if set(target_days) - set(e1["target_day"].unique()):
            raise ValueError("existing E1 cache does not cover all target days")
    else:
        e1 = build_e1_predictions(slot_q, p6, all_days, target_days, args.training_days, args.seed)
        atomic_parquet(e1_path, e1)

    summaries = []
    ledgers = []
    rules_all = {}
    for block in blocks:
        source_path = root / block.source_ledger
        source = pd.read_parquet(source_path)
        source["target_day"] = source["target_day"].astype(str)
        source = source[(source["target_day"] >= block.start) & (source["target_day"] <= block.end)].copy()
        r1_path = source_path.parent / "r1_causal_state_signals.parquet"
        r1_state = pd.read_parquet(r1_path)
        r1_state["target_day"] = r1_state["target_day"].astype(str)
        r1_cols = ["target_day", "hour_business", "r1_p_negative", "r1_p_neutral", "r1_p_positive", "r1_state_entropy", "r1_state_expected_spread"]
        source = source.merge(r1_state[r1_cols], on=["target_day", "hour_business"], how="left", validate="one_to_one")
        if source["target_day"].nunique() != 60:
            raise ValueError(f"{block.name}: source ledger days={source['target_day'].nunique()}")
        # Source R1/R2/R3 signals are already generated by a strict D-1 14:00 rolling pipeline.
        q_part = q[(q["target_day"] >= block.start) & (q["target_day"] <= block.end)]
        e1_part = e1[(e1["target_day"] >= block.start) & (e1["target_day"] <= block.end)]
        work = source.merge(q_part, on=["target_day", "hour_business"], how="left", validate="one_to_one")
        work = work.merge(e1_part, on=["target_day", "hour_business"], how="left", validate="one_to_one")
        if work[quantile_feature_names() + ["e1_prob"]].isna().any().any():
            raise RuntimeError(f"{block.name}: missing causal quantile/E1 features")
        s, rules, ledger = evaluate_block(work, block)
        summaries.append(s)
        ledgers.append(ledger)
        rules_all[block.name] = rules
        print(f"\n{block.name}\n{s.to_string(index=False)}")
        print(json.dumps(rules, ensure_ascii=False, indent=2, default=str))

    summary = pd.concat(summaries, ignore_index=True)
    ledger = pd.concat(ledgers, ignore_index=True)
    periods = period_report(ledger)
    atomic_csv(out / "summary.csv", summary)
    atomic_parquet(out / "ledger.parquet", ledger)
    atomic_csv(out / "period_metrics.csv", periods)
    atomic_json(out / "selected_rules.json", rules_all)

    # Final acceptance table: holdout performance and improvement versus P6 in BOTH blocks.
    hold = summary[summary["split"].eq("holdout15")].copy()
    base = hold[hold["model"].eq("E0_P6")][["block", "direction_accuracy", "balanced_direction_accuracy"]].rename(
        columns={"direction_accuracy": "base_direction", "balanced_direction_accuracy": "base_balanced"}
    )
    accept = hold.merge(base, on="block", how="left")
    accept["delta_direction_pp"] = 100 * (accept["direction_accuracy"] - accept["base_direction"])
    accept["delta_balanced_pp"] = 100 * (accept["balanced_direction_accuracy"] - accept["base_balanced"])
    atomic_csv(out / "holdout_acceptance.csv", accept)

    stable = (
        accept[accept["model"].ne("E0_P6")]
        .groupby("model")
        .agg(
            blocks=("block", "nunique"),
            min_delta_direction_pp=("delta_direction_pp", "min"),
            min_delta_balanced_pp=("delta_balanced_pp", "min"),
            mean_direction=("direction_accuracy", "mean"),
            mean_balanced=("balanced_direction_accuracy", "mean"),
            mean_positive=("positive_accuracy", "mean"),
            mean_negative=("negative_accuracy", "mean"),
        )
        .reset_index()
    )
    stable["improves_both_blocks"] = (stable["min_delta_direction_pp"] > 0) & (stable["min_delta_balanced_pp"] > 0)
    atomic_csv(out / "stability_summary.csv", stable)

    manifest = {
        "status": "complete",
        "goal": "short-term P6 integration of transferable R1 quantile/regime, R2 tail and R3 regime modules",
        "forecast_origin": "D-1 14:00",
        "training_days": args.training_days,
        "blocks": [b.__dict__ for b in blocks],
        "experiments": {
            "E0": "P6 baseline",
            "E1": "P6 + fully causal rolling q10/q25/q50/q75/q90 distribution features inside one LightGBM classifier",
            "E2": "P6 gray-zone positive rescue by causal quantile positivity, rule selected on first45 then frozen",
            "E3": "P6 gray-zone positive rescue by R1 positive regime in 1-8 only, selected on first45 then frozen",
            "E4": "P6 gray-zone positive rescue by R3 positive regime, period/threshold selected on first45 then frozen",
            "E5": "consensus rescue using quantile + R1 + R2 upper-tail + R3 signals",
        },
        "selection_objective": "0.5 * raw_direction_accuracy + 0.5 * balanced_direction_accuracy on each block design45; no-op baseline is always a candidate",
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
        "stable_candidates": stable.to_dict("records"),
    }
    atomic_json(out / "manifest.json", manifest)
    print("\nHOLDOUT ACCEPTANCE")
    print(accept.sort_values(["block", "balanced_direction_accuracy"], ascending=[True, False]).to_string(index=False))
    print("\nSTABILITY")
    print(stable.sort_values(["improves_both_blocks", "mean_balanced"], ascending=[False, False]).to_string(index=False))
    print(json.dumps({"runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
