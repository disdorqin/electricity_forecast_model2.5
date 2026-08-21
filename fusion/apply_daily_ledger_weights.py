"""
Apply daily ledger weights to fuse predictions.

Reads per-model predictions and learned weights, then produces
weighted-average fused predictions.

Key rules:
  - For each (business_day, hour_business), find available models
  - Use task + period weights
  - Renormalize weights for available models only
  - NEVER use fillna(0) — missing models are excluded
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_daily_ledger_weights(
    predictions_long: pd.DataFrame,
    weights: pd.DataFrame,
    target_day: str,
    task: str,
    allow_equal_weight_fallback: bool = False,
    strict: bool = True,
    resolution=None,
    weight_prune_threshold: float = 0.05,
    min_active_models: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply learned weights to predictions for a single day.
    Strict mode: fail on missing slots, missing models, or missing weights.

    resolution: Resolution（默认 HOURLY → 24 点行为逐字节不变）。
    weight_prune_threshold: learned weights below this value are excluded
        from the task/period fusion. Use 0 to disable the gate.
    min_active_models: safety minimum per task/period.

    Parameters
    ----------
    predictions_long : pd.DataFrame
        All model predictions in long format.
    weights : pd.DataFrame
        Per-(task, period) weights.
    target_day : str
        The target business day (YYYY-MM-DD).
    task : str
        "dayahead" or "realtime".
    allow_equal_weight_fallback : bool
        If True, use equal weights when period weights are missing.
    strict : bool
        If True (default), fail on missing hours/models/weights.

    Returns
    -------
    fused_df, debug_df
    """
    from utils.resolution import HOURLY
    from fusion.model_pool import models_for_task

    res = resolution or HOURLY
    slot_col = res.slot_column
    expected_models = set(models_for_task(task))

    # Filter predictions to target_day and task
    pred = predictions_long.copy()
    pred = pred[(pred["target_day"] == target_day) & (pred["task"] == task)]
    pred = pred[pred["model_name"].isin(expected_models)]

    if pred.empty:
        raise ValueError(f"No predictions found for {task} on {target_day}")

    # Filter weights to this task
    wdf = weights[weights["task"] == task].copy()
    wdf = wdf[wdf["model_name"].isin(expected_models)]

    if wdf.empty:
        raise ValueError(f"No weights found for {task}")

    # Get all model names from weights
    all_models = sorted(wdf["model_name"].unique())

    # Fuse slot by slot
    fused_rows = []
    debug_rows = []

    for slot in range(1, res.slots_per_day + 1):
        hour_pred = pred[pred[slot_col] == slot]

        if hour_pred.empty:
            msg = f"No predictions for slot {slot} in {task}/{target_day}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue

        # 权重 period 匹配：
        #   - period 粒度：用预测自带的 period 列（"1_32"/"33_64"/"65_96"）
        #   - hour 粒度：weights period 为 "h1".."h24"，按 hour_business 映射（96 点每小时 4 点一组）
        w_periods = set(wdf["period"].unique())
        _is_hour_granularity = any(str(p).startswith("h") for p in w_periods)
        if _is_hour_granularity:
            hb = int(hour_pred["hour_business"].iloc[0])
            period = f"h{hb}"
        else:
            period = hour_pred["period"].iloc[0]
        ds_val = hour_pred["ds"].iloc[0]
        bday = hour_pred["business_day"].iloc[0]

        # Get weights for this period
        period_weights = wdf[wdf["period"] == period]

        if period_weights.empty:
            if allow_equal_weight_fallback:
                logger.warning(f"No weights for {task}/{period}, using equal weights")
                available_models_fb = sorted(set(hour_pred["model_name"].unique()) & expected_models)
                eq_weight = 1.0 / max(len(available_models_fb), 1)
                weights_dict = {m: eq_weight for m in available_models_fb}
            elif strict:
                raise ValueError(
                    f"No weights for {task}/{period}. "
                    f"Pass --allow-equal-weight-fallback to use equal weights."
                )
            else:
                logger.warning(f"No weights for {task}/{period}, using equal weights")
                available_models_fb = sorted(set(hour_pred["model_name"].unique()) & expected_models)
                eq_weight = 1.0 / max(len(available_models_fb), 1)
                weights_dict = {m: eq_weight for m in available_models_fb}
        else:
            weights_dict = dict(zip(period_weights["model_name"], period_weights["weight"]))

        from fusion.model_quality_gate import gate_weights

        gate = gate_weights(
            weights_dict,
            threshold=weight_prune_threshold,
            min_active_models=min_active_models,
        )
        weights_dict = gate.active_weights

        # Map model to prediction
        model_preds = {}
        for _, row in hour_pred.iterrows():
            m = row["model_name"]
            y = row["y_pred"]
            if not pd.isna(y):
                model_preds[m] = float(y)

        all_model_set = set(weights_dict.keys())
        # A model pruned by the quality gate must not appear in the final
        # fusion candidate set, even when its prediction file is present.
        available = sorted(set(model_preds.keys()) & all_model_set)
        missing = sorted(all_model_set - set(available))

        if not available:
            msg = f"No available models for {task} slot {slot}"
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue

        # Collect raw weights for available models
        raw_w = {m: weights_dict.get(m, 0.0) for m in available}

        # Renormalize
        total_raw = sum(raw_w.values())
        if total_raw <= 0:
            renorm_w = {m: 1.0 / len(available) for m in available}
            was_renormalized = True
        else:
            renorm_w = {m: raw_w[m] / total_raw for m in available}
            was_renormalized = abs(total_raw - 1.0) > 0.001 or len(missing) > 0

        # Weighted average
        y_fused = sum(renorm_w[m] * model_preds[m] for m in available)

        fused_rows.append({
            "task": task,
            "business_day": bday,
            "ds": ds_val,
            slot_col: slot,
            "period": period,
            "y_fused": round(y_fused, 4),
        })

        debug_rows.append({
            "task": task,
            "business_day": bday,
            slot_col: slot,
            "period": period,
            "available_models": ",".join(available),
            "missing_models": ",".join(missing) if missing else "",
            "raw_weights": ",".join(f"{m}:{raw_w.get(m, 0):.4f}" for m in available),
            "renormalized_weights": ",".join(f"{m}:{renorm_w[m]:.4f}" for m in available),
            "renormalized": was_renormalized,
            "weight_gate_threshold": gate.threshold,
            "pruned_models": ",".join(gate.pruned_models),
            "active_models": ",".join(sorted(weights_dict)),
            "gate_fallback_used": gate.fallback_used,
            "gate_fallback_model": gate.fallback_model or "",
            "y_fused": round(y_fused, 4),
        })

    fused_df = pd.DataFrame(fused_rows)
    debug_df = pd.DataFrame(debug_rows)

    # Verify N rows — STRICT
    if len(fused_df) != res.slots_per_day:
        msg = (
            f"Fused {task}: expected {res.slots_per_day} rows, "
            f"got {len(fused_df)}. Missing slots!"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    # Check no fillna(0)
    if (fused_df["y_fused"] == 0).any():
        msg = "Fused predictions contain zeros — possible fillna(0) issue"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    return fused_df, debug_df
