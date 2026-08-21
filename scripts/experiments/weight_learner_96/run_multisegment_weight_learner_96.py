"""Causal multi-segment 96-point weight learner.

Experiment-only wrapper around ``run_weight_learner_96``.  It splits each
existing 32-slot ledger period into two 16-slot periods, then learns the
weights jointly under the same strictly-prior-day rules as the baseline.
No production ledger stage is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.weight_learner_96 import run_weight_learner_96 as base


def split_periods(day_data: dict, canonical_periods: tuple[str, ...], split_count: int) -> dict:
    """Split each canonical period into ``split_count`` equal sub-periods."""
    if split_count < 1 or 32 % split_count != 0:
        raise ValueError("split_count must be a positive divisor of the 32-slot canonical period")
    width = 32 // split_count
    output = {}
    for day, periods in day_data.items():
        out_day = {}
        for period in canonical_periods:
            matrix = periods[period]
            if matrix.X.shape[0] != 32:
                raise ValueError(f"{day}/{period}: expected 32 rows, got {matrix.X.shape[0]}")
            for part in range(split_count):
                start = part * width
                end = start + width
                label = f"{period}_{part + 1}of{split_count}"
                out_day[label] = base.DayMatrix(X=matrix.X[start:end], y=matrix.y[start:end])
        output[day] = out_day
    return output


def smoothed_selected_predictions(
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    task_models: dict[str, list[str]],
    periods: tuple[str, ...],
    smoothing_lambda: float,
) -> pd.DataFrame:
    """Apply one causal, convex adjacent-period smoothing pass to selected weights.

    The input weights were already learned from strictly earlier dates.  This
    pass uses only neighboring *same-day learned weights*, not target-day truth.
    Convex averaging preserves the per-period sum-to-one constraint and the
    signed-weight bounds.
    """
    if not 0.0 <= smoothing_lambda <= 0.5:
        raise ValueError("smoothing_lambda must be in [0, 0.5]")
    rows = []
    for task, models in task_models.items():
        task_weights = weights[(weights.task == task) & (weights.method == "selected")].copy()
        task_pred = predictions[(predictions.task == task) & predictions.method.isin(models)].copy()
        for target_day in sorted(task_weights.target_day.unique()):
            day_weights = task_weights[task_weights.target_day == target_day]
            matrix = day_weights.pivot(index="period", columns="model", values="weight").reindex(index=periods, columns=models)
            if matrix.isna().any().any():
                raise ValueError(f"missing selected weights for {task}/{target_day}")
            W = matrix.to_numpy(float)
            smooth = W.copy()
            for i in range(len(periods)):
                neighbors = []
                if i > 0:
                    neighbors.append(W[i - 1])
                if i + 1 < len(periods):
                    neighbors.append(W[i + 1])
                if neighbors:
                    smooth[i] = (1.0 - smoothing_lambda) * W[i] + smoothing_lambda * np.mean(neighbors, axis=0)

            day_pred = task_pred[task_pred.target_day == target_day]
            for i, period in enumerate(periods):
                segment = day_pred[day_pred.period == period].copy()
                wide = segment.pivot(index="business_period_in_segment", columns="method", values="y_pred").reindex(columns=models)
                truth = segment.drop_duplicates("business_period_in_segment").set_index("business_period_in_segment")["y_true"]
                if wide.empty or wide.isna().any().any():
                    raise ValueError(f"missing model predictions for {task}/{target_day}/{period}")
                pred = wide.to_numpy(float) @ smooth[i]
                for slot, y_pred in zip(wide.index, pred):
                    rows.append({
                        "task": task,
                        "target_day": target_day,
                        "period": period,
                        "business_period_in_segment": int(slot),
                        "method": "selected_smoothed",
                        "y_pred": float(y_pred),
                        "y_true": float(truth.loc[slot]),
                    })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", default="2026-02-01")
    parser.add_argument("--selection-margin", type=float, default=0.005)
    parser.add_argument("--split-count", type=int, default=2)
    parser.add_argument("--policy", choices=("instant", "validation_rolling", "regime_selector", "fixed_nonnegative", "fixed_champion"), default="instant")
    parser.add_argument("--policy-history-days", type=int, default=45)
    parser.add_argument("--regime-validation-days", type=int, default=15)
    parser.add_argument("--smoothing-lambda", type=float, default=0.25)
    choices = ("instant", "validation_rolling", "regime_selector", "fixed_nonnegative", "fixed_champion")
    parser.add_argument("--dayahead-policy", choices=choices, default=None)
    parser.add_argument("--realtime-policy", choices=choices, default=None)
    args = parser.parse_args()

    ledger_root = Path(args.ledger_root)
    if not ledger_root.is_absolute():
        ledger_root = PROJECT_ROOT / ledger_root
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve the canonical three-period loader and only split its matrices
    # in this experiment wrapper.  Production resolution metadata is untouched.
    old_periods = base.PERIODS
    task_data = {}
    for task in ("dayahead", "realtime"):
        day_data, models = base.load_task(ledger_root, task)
        task_data[task] = (split_periods(day_data, old_periods, args.split_count), models)

    base.PERIODS = tuple(
        f"{period}_{part + 1}of{args.split_count}"
        for period in old_periods
        for part in range(args.split_count)
    )
    try:

        all_predictions = []
        all_weights = []
        all_selection = []
        metadata = {}
        for task, (day_data, models) in task_data.items():
            policy = getattr(args, f"{task}_policy") or args.policy
            predictions, weights, selection, meta = base.run_task(
                task,
                day_data,
                models,
                start_date=args.start_date,
                selection_margin=args.selection_margin,
                policy=policy,
                policy_history_days=args.policy_history_days,
                regime_validation_days=args.regime_validation_days,
            )
            all_predictions.append(predictions)
            all_weights.append(weights)
            all_selection.append(selection)
            metadata[task] = meta

        predictions = pd.concat(all_predictions, ignore_index=True)
        weights = pd.concat(all_weights, ignore_index=True)
        selection = pd.concat(all_selection, ignore_index=True)
        if predictions.empty or not np.isfinite(predictions[["y_pred", "y_true"]].to_numpy(float)).all():
            raise ValueError("multi-segment experiment produced no finite predictions")

        task_models = {task: models for task, (_, models) in task_data.items()}
        smooth = smoothed_selected_predictions(
            predictions,
            weights,
            task_models,
            base.PERIODS,
            args.smoothing_lambda,
        )
        predictions = pd.concat([predictions, smooth], ignore_index=True)
        predictions.to_parquet(out_dir / "walk_forward_predictions.parquet", index=False)
        weights.to_csv(out_dir / "weights_audit.csv", index=False, encoding="utf-8-sig")
        selection.to_csv(out_dir / "selection_audit.csv", index=False, encoding="utf-8-sig")
        base.build_metrics(predictions, out_dir)
        base.build_scr(predictions, out_dir)
        base.build_scr(predictions, out_dir, selected_method="selected_smoothed", output_name="selected_smoothed_scr_monthly.csv")
        significance = pd.concat([
            base.daily_significance(predictions, "dayahead", base.REFERENCE_BEST["dayahead"]),
            base.daily_significance(predictions, "realtime", base.REFERENCE_BEST["realtime"]),
        ], ignore_index=True)
        significance.to_csv(out_dir / "significance_daily_smape.csv", index=False, encoding="utf-8-sig")

        manifest = {
            "experiment": "causal_multi_segment_weight_learner_96",
            "production_link_touched": False,
            "resolution": base.RES.label,
            "canonical_periods": list(old_periods),
            "experiment_periods": list(base.PERIODS),
            "split_count": args.split_count,
            "smoothing_lambda": args.smoothing_lambda,
            "start_date": args.start_date,
            "end_date": str(predictions["target_day"].max()),
            "policy": args.policy,
            "dayahead_policy": args.dayahead_policy or args.policy,
            "realtime_policy": args.realtime_policy or args.policy,
            "policy_history_days": args.policy_history_days,
            "regime_validation_days": args.regime_validation_days,
            "decision_rule": "target day uses strictly earlier days; no same-day truth is used for selection",
            "source_ledger_root": str(ledger_root),
            "rows": {"predictions": len(predictions), "weights": len(weights), "selection": len(selection)},
            "metadata": metadata,
        }
        (out_dir / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "complete", "output": str(out_dir), "periods": list(base.PERIODS)}, ensure_ascii=False))
        return 0
    finally:
        base.PERIODS = old_periods


if __name__ == "__main__":
    raise SystemExit(main())
