from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    DEFAULT_CUBE,
    GaussianHMM1D,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    load_p6_features,
    regression_metrics,
)


def lgb_binary(seed: int = 42) -> lgb.LGBMClassifier:
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


def regime_model(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=900, multi_class="multinomial", class_weight="balanced", C=0.35, random_state=seed)),
    ])


def meta_model(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=600, class_weight="balanced", C=1.0, random_state=seed)),
    ])


def metrics_direction(frame: pd.DataFrame, pred_col: str) -> dict:
    y = np.sign(frame["y_true_spread"].to_numpy(float))
    p = frame[pred_col].to_numpy(int)
    eligible = y != 0
    correct = y == p
    pos = y > 0
    neg = y < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "days": int(frame["target_day"].nunique()),
        "n_slots": int(len(frame)),
        "n_positive_actual": int(pos.sum()),
        "n_negative_actual": int(neg.sum()),
        "direction_accuracy": float(correct[eligible].mean()),
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
    }


def _entropy(p: np.ndarray) -> np.ndarray:
    a = np.clip(np.asarray(p, float), 1e-12, 1.0)
    return -(a * np.log(a)).sum(axis=1)


def build_r1_signals(slot: pd.DataFrame, feature_days: list[str], hmm_fit_end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    history = slot[["target_day", "时刻", "target_spread"]].copy()
    history["时刻"] = pd.to_datetime(history["时刻"])
    history = history.sort_values("时刻").drop_duplicates("时刻", keep="last")
    fit_start = hmm_fit_end - pd.Timedelta(days=365)
    fit = history[(history["时刻"] > fit_start) & (history["时刻"] <= hmm_fit_end)]["target_spread"].to_numpy(float)
    hmm = GaussianHMM1D(n_states=3, max_iter=70, tol=1e-5).fit(fit)
    rows = []
    for day in feature_days:
        d = pd.Timestamp(day)
        cutoff = d - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        h = history[(history["时刻"] <= cutoff) & (history["时刻"] > cutoff - pd.Timedelta(days=30))]
        posterior = hmm.filter_history(h["target_spread"].to_numpy(float))
        target = slot[slot["target_day"].eq(day)].sort_values("hour_business")
        for _, row in target.iterrows():
            target_ts = pd.Timestamp(row["时刻"])
            steps = max(1, int(round((target_ts - cutoff).total_seconds() / 3600.0)))
            pp = hmm.propagate(posterior, steps)
            rows.append({
                "target_day": day,
                "hour_business": int(row["hour_business"]),
                "r1_p_negative": float(pp[0]),
                "r1_p_neutral": float(pp[1]),
                "r1_p_positive": float(pp[2]),
                "r1_state_entropy": float(_entropy(pp[None, :])[0]),
                "r1_state_expected_spread": float(pp @ hmm.means_),
                "r1_transition_steps": steps,
                "r1_source_max_ds": cutoff,
            })
    meta = {
        "fit_start": str(fit_start), "fit_end": str(hmm_fit_end), "fit_points": int(len(fit)),
        "means": hmm.means_.tolist(), "std": np.sqrt(hmm.vars_).tolist(),
        "transition": hmm.transmat_.tolist(), "expected_duration_hours": hmm.expected_duration_.tolist(),
    }
    return pd.DataFrame(rows), meta


def fit_binary(model, train: pd.DataFrame, features: list[str], y_binary: np.ndarray):
    model.fit(train[features], y_binary)
    return model


def safe_predict_multiclass(model: Pipeline, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[features])
    classes = model.named_steps["logit"].classes_.astype(int)
    out = np.zeros((len(frame), 3), float)
    for j, c in enumerate(classes):
        out[:, c] = raw[:, j]
    return out


def build_regime_labels(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(y <= lo, 0, np.where(y >= hi, 2, 1)).astype(int)


def period_from_hour(h: int) -> str:
    return "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict D-1 14:00 adaptation of R1/R2/R3 paper mechanisms on P6 Feature Cube.")
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--output", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/adapted_14h_backtest")
    ap.add_argument("--start", default="2026-06-16")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--calibration-days", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    slot, p6 = load_p6_features(root / args.cube_root)
    slot["target_day"] = slot["target_day"].astype(str)
    slot["时刻"] = pd.to_datetime(slot["时刻"])
    all_days = sorted(slot["target_day"].dropna().unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    if len(target_days) != 60:
        raise ValueError(f"expected 60 target days, got {len(target_days)}")
    first_idx = all_days.index(target_days[0])
    earliest_train_idx = first_idx - args.training_days
    if earliest_train_idx < 0:
        raise ValueError("insufficient prehistory")
    earliest_feature_day = all_days[earliest_train_idx]
    feature_days = all_days[earliest_train_idx: all_days.index(target_days[-1]) + 1]
    hmm_fit_end = pd.Timestamp(earliest_feature_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    r1, r1_meta = build_r1_signals(slot, feature_days, hmm_fit_end)
    atomic_parquet(out / "r1_causal_state_signals.parquet", r1)
    work = slot.merge(r1.drop(columns=["r1_source_max_ds"]), on=["target_day", "hour_business"], how="left", validate="many_to_one")
    r1_features = p6 + ["r1_p_negative", "r1_p_neutral", "r1_p_positive", "r1_state_entropy", "r1_state_expected_spread"]
    missing = [c for c in r1_features if c not in work]
    if missing:
        raise ValueError(f"missing features {missing}")

    day_ledgers = []
    threshold_rows = []
    for day_no, target_day in enumerate(target_days, 1):
        idx = all_days.index(target_day)
        train_days = all_days[idx - args.training_days: idx]
        cal_days = train_days[-args.calibration_days:]
        fit_days = train_days[:-args.calibration_days]
        train75 = work[work["target_day"].isin(fit_days)]
        cal = work[work["target_day"].isin(cal_days)]
        train90 = work[work["target_day"].isin(train_days)]
        test = work[work["target_day"].eq(target_day)].sort_values("hour_business")
        if len(test) != 24:
            raise ValueError(f"{target_day}: expected24, got {len(test)}")
        y75 = train75["target_spread"].to_numpy(float)
        y90 = train90["target_spread"].to_numpy(float)
        ycal = cal["target_spread"].to_numpy(float)
        sign75 = (y75 > 0).astype(int)
        sign90 = (y90 > 0).astype(int)
        signcal = (ycal > 0).astype(int)
        lo, hi = np.quantile(y75, [0.05, 0.95])
        threshold_rows.append({"target_day": target_day, "source_last_day": fit_days[-1], "lower_q05": lo, "upper_q95": hi})

        # Base and R1 augmented, first stage on fit window for causal calibration.
        base75 = lgb_binary(args.seed).fit(train75[p6], sign75)
        r1_75 = lgb_binary(args.seed).fit(train75[r1_features], sign75)
        cal_base = base75.predict_proba(cal[p6])[:, 1]
        cal_r1 = r1_75.predict_proba(cal[r1_features])[:, 1]

        # R2: paper-style positive/negative tail event classifiers, but thresholds are train-only q05/q95
        # because the paper's USD fixed thresholds are not portable to RMB Shandong.
        lower75 = lgb_binary(args.seed).fit(train75[p6], (y75 <= lo).astype(int))
        upper75 = lgb_binary(args.seed).fit(train75[p6], (y75 >= hi).astype(int))
        cal_lower = lower75.predict_proba(cal[p6])[:, 1]
        cal_upper = upper75.predict_proba(cal[p6])[:, 1]
        r2_meta = meta_model(args.seed).fit(np.c_[cal_base, cal_lower, cal_upper], signcal)

        # R3: covariate-dependent three-regime frequency model; use its regime probabilities
        # as an auxiliary signal. Severity was weak in strict structural reproduction and is not leaked in here.
        reg75 = build_regime_labels(y75, lo, hi)
        r3_75 = regime_model(args.seed).fit(train75[p6], reg75)
        cal_reg = safe_predict_multiclass(r3_75, cal, p6)
        r3_meta = meta_model(args.seed).fit(np.c_[cal_base, cal_reg[:, 0], cal_reg[:, 2]], signcal)

        combined_meta = meta_model(args.seed).fit(
            np.c_[cal_r1, cal_lower, cal_upper, cal_reg[:, 0], cal_reg[:, 2]], signcal
        )

        # Refit first-stage learners on all 90 causal days. Tail thresholds stay frozen from fit75
        # so the calibration definition remains exactly the same.
        base90 = lgb_binary(args.seed).fit(train90[p6], sign90)
        r1_90 = lgb_binary(args.seed).fit(train90[r1_features], sign90)
        lower90 = lgb_binary(args.seed).fit(train90[p6], (y90 <= lo).astype(int))
        upper90 = lgb_binary(args.seed).fit(train90[p6], (y90 >= hi).astype(int))
        r3_90 = regime_model(args.seed).fit(train90[p6], build_regime_labels(y90, lo, hi))

        p_base = base90.predict_proba(test[p6])[:, 1]
        p_r1 = r1_90.predict_proba(test[r1_features])[:, 1]
        p_low = lower90.predict_proba(test[p6])[:, 1]
        p_up = upper90.predict_proba(test[p6])[:, 1]
        p_reg = safe_predict_multiclass(r3_90, test, p6)
        p_r2 = r2_meta.predict_proba(np.c_[p_base, p_low, p_up])[:, 1]
        p_r3 = r3_meta.predict_proba(np.c_[p_base, p_reg[:, 0], p_reg[:, 2]])[:, 1]
        p_combined = combined_meta.predict_proba(np.c_[p_r1, p_low, p_up, p_reg[:, 0], p_reg[:, 2]])[:, 1]

        # Direct R3 expectation from causal training regime means: retained as a continuous diagnostic.
        reg90 = build_regime_labels(y90, lo, hi)
        means = np.array([np.mean(y90[reg90 == k]) if np.any(reg90 == k) else 0.0 for k in range(3)])
        r3_expected = p_reg @ means
        r1_direct = (
            test[["r1_p_negative", "r1_p_neutral", "r1_p_positive"]].to_numpy(float)
            @ np.asarray(r1_meta["means"], float)
        )

        day = test[["target_day", "时刻", "hour_business", "period", "target_spread"]].copy()
        day = day.rename(columns={"target_spread": "y_true_spread"})
        day["base_p6_prob"] = p_base
        day["r1_aug_prob"] = p_r1
        day["r2_tail_prob"] = p_r2
        day["r3_regime_prob"] = p_r3
        day["combined_prob"] = p_combined
        day["r2_p_lower_tail"] = p_low
        day["r2_p_upper_tail"] = p_up
        day["r3_p_negative"] = p_reg[:, 0]
        day["r3_p_regular"] = p_reg[:, 1]
        day["r3_p_positive"] = p_reg[:, 2]
        day["r3_expected_spread"] = r3_expected
        day["r1_direct_expected_spread"] = r1_direct
        for name in ["base_p6", "r1_aug", "r2_tail", "r3_regime", "combined"]:
            day[f"{name}_direction"] = np.where(day[f"{name}_prob"] >= 0.5, 1, -1)
        day_ledgers.append(day)
        if day_no % 10 == 0:
            print(f"completed {day_no}/{len(target_days)}: {target_day}")

    ledger = pd.concat(day_ledgers, ignore_index=True)
    atomic_parquet(out / "ledger.parquet", ledger)
    atomic_csv(out / "tail_thresholds.csv", pd.DataFrame(threshold_rows))

    split_ranges = {
        "development30": ("2026-06-16", "2026-07-15"),
        "confirmation15": ("2026-07-16", "2026-07-30"),
        "holdout15": ("2026-07-31", "2026-08-14"),
        "overall60": ("2026-06-16", "2026-08-14"),
    }
    direction_models = ["base_p6", "r1_aug", "r2_tail", "r3_regime", "combined"]
    summary_rows = []
    period_rows = []
    for split, (a, b) in split_ranges.items():
        part = ledger[(ledger["target_day"] >= a) & (ledger["target_day"] <= b)]
        for model in direction_models:
            summary_rows.append({"split": split, "model": model, **metrics_direction(part, f"{model}_direction")})
            for period, g in part.groupby("period"):
                period_rows.append({"split": split, "model": model, "period": period, **metrics_direction(g, f"{model}_direction")})
        for model, col in [("r3_direct_expected", "r3_expected_spread"), ("r1_direct_expected", "r1_direct_expected_spread")]:
            m = regression_metrics(part["y_true_spread"].to_numpy(float), part[col].to_numpy(float))
            summary_rows.append({"split": split, "model": model, "days": part["target_day"].nunique(), "n_slots": len(part), "n_positive_actual": int((part["y_true_spread"] > 0).sum()), "n_negative_actual": int((part["y_true_spread"] < 0).sum()), **m})

    summary = pd.DataFrame(summary_rows)
    periods = pd.DataFrame(period_rows)
    atomic_csv(out / "summary.csv", summary)
    atomic_csv(out / "period_metrics.csv", periods)

    # Strong information-boundary audit.
    audit = []
    for day in target_days:
        cutoff = pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        source = pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        train_last = max(d for d in all_days if d < day)
        audit.append({"target_day": day, "cutoff": cutoff, "r1_source_max_ds": source, "train_last_day": train_last, "r1_ok": source <= cutoff, "train_ok": train_last < day})
    audit_df = pd.DataFrame(audit)
    atomic_csv(out / "information_boundary_audit.csv", audit_df)
    if not audit_df[["r1_ok", "train_ok"]].all().all():
        raise RuntimeError("information boundary audit failed")

    manifest = {
        "pipeline": "paper_reproductions_adapted_14h",
        "status": "complete",
        "forecast_origin": "D-1 14:00",
        "training_days": args.training_days,
        "calibration_days": args.calibration_days,
        "base_features": len(p6),
        "r1": {"mechanism": "3-state Gaussian HMM, filtered at cutoff then Markov-propagated to each target hour", **r1_meta},
        "r2": {"mechanism": "lower/upper spread-tail LightGBM classifiers + causal calibration", "threshold": "q05/q95 derived only from first 75 of rolling 90 training days"},
        "r3": {"mechanism": "covariate-dependent 3-state multinomial frequency signal + causal calibration; strict GPD severity reproduction retained separately"},
        "combined": "single causal logistic meta classifier calibrated only on the last 15 days of each prior 90-day window",
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(out / "manifest.json", manifest)
    print(summary[summary["split"].isin(["confirmation15", "holdout15", "overall60"])].sort_values(["split", "balanced_direction_accuracy"], ascending=[True, False]).to_string(index=False))
    print(json.dumps({"runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
