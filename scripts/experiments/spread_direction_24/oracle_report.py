"""Compute retrospective complementarity ceilings for spread candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _accuracy(frame: pd.DataFrame) -> float:
    true_sign = np.sign(frame["y_true_spread"].to_numpy(float))
    pred_sign = np.sign(frame["y_pred_spread"].to_numpy(float))
    eligible = true_sign != 0
    return float((true_sign[eligible] == pred_sign[eligible]).mean()) if eligible.any() else np.nan


def oracle_report(ledger_path: Path, output_path: Path, dev_days: int) -> pd.DataFrame:
    ledger = pd.read_parquet(ledger_path).copy()
    ledger["target_day"] = pd.to_datetime(ledger["target_day"]).dt.strftime("%Y-%m-%d")
    dates = sorted(ledger["target_day"].unique())
    if len(dates) <= dev_days:
        raise ValueError("dev_days must leave a test split")
    rows = []
    for split, split_dates in (("dev", dates[:dev_days]), ("test", dates[dev_days:])):
        part = ledger[ledger["target_day"].isin(split_dates)].copy()
        best_single = max(_accuracy(g) for _, g in part.groupby("model_name"))

        slot_choices = []
        for key, group in part.groupby(["target_day", "hour_business", "period"]):
            scored = [(name, _accuracy(g)) for name, g in group.groupby("model_name")]
            slot_choices.append(max(score for _, score in scored))
        period_choices = []
        for key, group in part.groupby(["target_day", "period"]):
            scored = [(name, _accuracy(g)) for name, g in group.groupby("model_name")]
            period_choices.append(max(score for _, score in scored))
        day_choices = []
        for key, group in part.groupby("target_day"):
            scored = [(name, _accuracy(g)) for name, g in group.groupby("model_name")]
            day_choices.append(max(score for _, score in scored))
        rows.extend(
            [
                {"split": split, "ceiling": "retrospective_best_single_model", "direction_accuracy": best_single, "observations": len(split_dates)},
                {"split": split, "ceiling": "retrospective_best_model_per_day_period", "direction_accuracy": float(np.mean(period_choices)), "observations": len(period_choices)},
                {"split": split, "ceiling": "retrospective_best_model_per_slot", "direction_accuracy": float(np.mean(slot_choices)), "observations": len(slot_choices)},
                {"split": split, "ceiling": "retrospective_best_model_per_day", "direction_accuracy": float(np.mean(day_choices)), "observations": len(day_choices)},
            ]
        )
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dev-days", type=int, default=15)
    args = parser.parse_args()
    print(oracle_report(args.ledger, args.output, args.dev_days).to_string(index=False))


if __name__ == "__main__":
    main()
