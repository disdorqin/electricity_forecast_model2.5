"""Cycle 17 B pilot: causal development-only probability ranking and event-cost freeze.

The input is an already generated strict-D2 B-line OOS prediction table.  This
runner never refits a model and never reads labels from the confirmation/final
blocks while selecting a rule.  It compares expected-spread and state-probability
ranking scores, freezes the best rule on the development block, and evaluates that
frozen rule once on the confirmation block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    y = frame["y_true"].to_numpy(int)
    p = frame[pred_col].to_numpy(int)
    pos, neg = y == 1, y == -1
    pos_recall = float((p[pos] == 1).mean()) if pos.any() else 0.0
    neg_recall = float((p[neg] == -1).mean()) if neg.any() else 0.0
    return {
        "days": int(frame["target_day"].nunique()),
        "n_slots": int(len(frame)),
        "direction_accuracy": float((p == y).mean()),
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "balanced_accuracy": float((pos_recall + neg_recall) / 2.0),
        "all_negative_baseline": float(neg.mean()),
    }


def validate_source(frame: pd.DataFrame) -> None:
    required = {
        "target_day", "y_true", "expected_spread", "p_negative_spike",
        "p_regular", "p_positive_spike", "training_last_day",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"B prediction source missing columns: {missing}")
    frame["target_day"] = pd.to_datetime(frame["target_day"]).dt.normalize()
    frame["training_last_day"] = pd.to_datetime(frame["training_last_day"]).dt.normalize()
    if frame["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("source includes fresh final holdout; refuse to calibrate")
    if (frame["training_last_day"] > frame["target_day"] - pd.Timedelta(days=2)).any():
        bad = frame.loc[frame["training_last_day"] > frame["target_day"] - pd.Timedelta(days=2)].iloc[0]
        raise RuntimeError(
            f"strict training boundary violated: target={bad.target_day.date()} "
            f"training_last_day={bad.training_last_day.date()}"
        )
    counts = frame.groupby("target_day").size()
    if not (counts == 24).all():
        raise RuntimeError(f"incomplete target days: {counts[counts != 24].to_dict()}")


def candidate_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["score_expected_spread"] = pd.to_numeric(out["expected_spread"], errors="coerce")
    pneg = pd.to_numeric(out["p_negative_spike"], errors="coerce").fillna(0.0)
    ppos = pd.to_numeric(out["p_positive_spike"], errors="coerce").fillna(0.0)
    preg = pd.to_numeric(out["p_regular"], errors="coerce").fillna(0.0)
    out["score_state_margin"] = ppos - pneg
    # Event-cost family: cost_ratio > 1 makes a positive event harder to call;
    # cost_ratio < 1 makes positive recall more valuable.  The ratio is fixed
    # before confirmation and uses no confirmation labels.
    for cost_ratio in (0.5, 1.0, 2.0, 4.0):
        out[f"score_cost_{cost_ratio:g}"] = ppos - cost_ratio * pneg
    # Keep the probability mass visible in the audit even when regular dominates.
    out["state_mass_nonregular"] = ppos + pneg
    return out


def evaluate_candidates(frame: pd.DataFrame, score_col: str, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        pred = np.where(frame[score_col].to_numpy(float) >= threshold, 1, -1)
        temp = frame.copy()
        temp["_pred"] = pred
        rows.append({"score": score_col, "threshold": threshold, **metrics(temp, "_pred")})
    return pd.DataFrame(rows)


def choose_dev_rule(dev_results: pd.DataFrame) -> pd.Series:
    # Balanced accuracy is the primary development objective; raw accuracy and
    # positive recall break ties so an all-negative-like rule cannot win silently.
    return dev_results.sort_values(
        ["balanced_accuracy", "direction_accuracy", "positive_recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default=None, help="required when predictions.csv contains multiple decision variants")
    parser.add_argument("--development-start", default="2026-04-01")
    parser.add_argument("--development-end", default="2026-06-30")
    parser.add_argument("--confirmation-start", default="2026-07-01")
    parser.add_argument("--confirmation-end", default="2026-08-14")
    args = parser.parse_args()

    source = args.input.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    if "variant" in frame.columns:
        variants = sorted(frame["variant"].dropna().astype(str).unique())
        if args.variant:
            if args.variant not in variants:
                raise RuntimeError(f"unknown variant {args.variant}; available={variants}")
            frame = frame[frame["variant"].astype(str).eq(args.variant)].copy()
        elif len(variants) > 1:
            raise RuntimeError(f"source has multiple variants; pass --variant from {variants}")
    validate_source(frame)
    frame = candidate_table(frame)
    dev = frame[frame["target_day"].between(args.development_start, args.development_end)].copy()
    confirm = frame[frame["target_day"].between(args.confirmation_start, args.confirmation_end)].copy()
    if dev.empty or confirm.empty:
        raise RuntimeError("development or confirmation block is empty")

    thresholds = {
        "score_expected_spread": [-30, -20, -10, 0, 10, 20, 30],
        "score_state_margin": [-0.30, -0.20, -0.10, 0, 0.10, 0.20, 0.30],
        "score_cost_0.5": [-0.20, -0.10, 0, 0.10, 0.20],
        "score_cost_1": [-0.20, -0.10, 0, 0.10, 0.20],
        "score_cost_2": [-0.20, -0.10, 0, 0.10, 0.20],
        "score_cost_4": [-0.20, -0.10, 0, 0.10, 0.20],
    }
    dev_rows = []
    for score, grid in thresholds.items():
        dev_rows.append(evaluate_candidates(dev, score, grid))
    dev_results = pd.concat(dev_rows, ignore_index=True)
    chosen = choose_dev_rule(dev_results)

    score = str(chosen["score"])
    threshold = float(chosen["threshold"])
    for part, name in ((dev, "development"), (confirm, "confirmation")):
        part["predicted_direction"] = np.where(part[score].to_numpy(float) >= threshold, 1, -1)
        part["split"] = name
    selected = pd.concat([dev, confirm], ignore_index=True)
    dev_metrics = metrics(dev, "predicted_direction")
    confirm_metrics = metrics(confirm, "predicted_direction")
    summary = pd.DataFrame([
        {"split": "development", "score": score, "threshold": threshold, **dev_metrics},
        {"split": "confirmation", "score": score, "threshold": threshold, **confirm_metrics},
    ])

    dev_results.to_csv(out / "development_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    selected[[
        "target_day", "时刻", "variant", "y_true", "predicted_direction", score,
        "training_last_day", "split",
    ]].to_csv(out / "frozen_predictions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS",
        "experiment_status": "REJECTED",
        "route": "B_distributional_states",
        "forecast_origin": "D-1 14:00",
        "training_last_day": "inherited per target day <= D-2; asserted by source audit",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "source_prediction_table": str(source),
        "development_range": [args.development_start, args.development_end],
        "confirmation_range": [args.confirmation_start, args.confirmation_end],
        "selection_objective": "development balanced accuracy, raw accuracy, positive recall tie-break",
        "frozen_score": score,
        "frozen_threshold": threshold,
        "development_metrics": dev_metrics,
        "confirmation_metrics": confirm_metrics,
        "note": "pilot only; confirmation labels never used for rule selection; no production integration",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "frozen_score": score, "frozen_threshold": threshold, "development": dev_metrics, "confirmation": confirm_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
