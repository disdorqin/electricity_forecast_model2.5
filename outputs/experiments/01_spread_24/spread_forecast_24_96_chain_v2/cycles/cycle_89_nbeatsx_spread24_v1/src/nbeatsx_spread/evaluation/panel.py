from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .metrics import compute_metrics


def daily_metric_row(target_day: str, prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Compute one target-day row plus regime and transition diagnostics."""
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    metrics = compute_metrics(prediction, target)
    true_sign = np.sign(target[target != 0])
    pred_sign = np.sign(prediction[target != 0])
    actual_positive_rate = float((target[target != 0] > 0).mean()) if len(true_sign) else float("nan")
    predicted_positive_rate = float((prediction[target != 0] > 0).mean()) if len(true_sign) else float("nan")
    majority = max(actual_positive_rate, 1.0 - actual_positive_rate) if len(true_sign) else float("nan")
    minority = float(np.nanmin([metrics["positive_recall"], metrics["negative_recall"]]))
    actual_switches = _transition_positions(np.sign(target))
    predicted_switches = _transition_positions(np.sign(prediction))
    transition_tp = len(actual_switches.intersection(predicted_switches))
    transition_precision = transition_tp / len(predicted_switches) if predicted_switches else float("nan")
    transition_recall = transition_tp / len(actual_switches) if actual_switches else float("nan")
    transition_f1 = (
        2.0 * transition_precision * transition_recall / (transition_precision + transition_recall)
        if np.isfinite(transition_precision) and np.isfinite(transition_recall) and transition_precision + transition_recall > 0
        else (1.0 if not actual_switches and not predicted_switches else 0.0)
    )
    return {
        "target_day": target_day,
        **metrics,
        "raw": metrics["direction_accuracy"],
        "balanced": metrics["balanced_accuracy"],
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "actual_positive_rate": actual_positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
        "majority_baseline": majority,
        "raw_minus_majority": metrics["direction_accuracy"] - majority,
        "minority_recall": minority,
        "actual_sign_switch_count": len(actual_switches),
        "predicted_sign_switch_count": len(predicted_switches),
        "transition_precision": float(transition_precision),
        "transition_recall": float(transition_recall),
        "transition_f1": float(transition_f1),
        "majority_collapse": bool(len(true_sign) > 0 and len(np.unique(pred_sign)) <= 1 and abs(metrics["direction_accuracy"] - majority) <= 1e-12),
    }


def _sign_switch_count(signs: np.ndarray) -> int:
    """Count adjacent sign changes, retaining zero as its own explicit state."""
    signs = np.asarray(signs)
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0


def _transition_positions(signs: np.ndarray) -> set[int]:
    """Return exact hourly boundary positions where the sign state changes."""
    signs = np.asarray(signs)
    return set((np.flatnonzero(signs[1:] != signs[:-1]) + 1).tolist())


def aggregate_daily_metrics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize daily macro distributions without pooling away day effects."""
    rows = list(rows)
    if not rows:
        raise ValueError("daily rows cannot be empty")
    fields = {
        "raw": "raw",
        "balanced": "balanced",
        "positive_recall": "positive_recall",
        "negative_recall": "negative_recall",
        "MAE": "MAE",
    }
    out: dict[str, Any] = {"day_count": len(rows)}
    for name, key in fields.items():
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        out[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p10": float(np.percentile(values, 10)),
            "worst": float(np.min(values)),
            "best": float(np.max(values)),
        }
    return out


def paired_daily_delta(nbeats_row: dict[str, Any], baseline_row: dict[str, Any]) -> dict[str, Any]:
    """Calculate same-date deltas before any aggregate summary."""
    if nbeats_row["target_day"] != baseline_row["target_day"]:
        raise ValueError("paired delta requires the same target day")
    return {
        "target_day": nbeats_row["target_day"],
        "delta_raw": float(nbeats_row["raw"] - baseline_row["raw"]),
        "delta_balanced": float(nbeats_row["balanced"] - baseline_row["balanced"]),
        "delta_MAE": float(nbeats_row["MAE"] - baseline_row["MAE"]),
    }


def baseline_row_from_arrays(target_day: str, prediction: np.ndarray, target: np.ndarray, model: str) -> dict[str, Any]:
    """Create the common comparison schema for a frozen comparator."""
    row = daily_metric_row(target_day, prediction, target)
    row["model"] = model
    return row
