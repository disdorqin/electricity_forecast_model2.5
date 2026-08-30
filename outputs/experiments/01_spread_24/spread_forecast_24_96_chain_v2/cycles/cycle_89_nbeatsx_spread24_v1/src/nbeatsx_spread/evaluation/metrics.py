from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def _arrays(pred: Tensor | np.ndarray, target: Tensor | np.ndarray, mask: Tensor | np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    p, y = np.asarray(pred.detach().cpu() if isinstance(pred, Tensor) else pred), np.asarray(target.detach().cpu() if isinstance(target, Tensor) else target)
    m = np.ones_like(y, dtype=bool) if mask is None else np.asarray(mask.detach().cpu() if isinstance(mask, Tensor) else mask).astype(bool)
    if p.shape != y.shape or p.shape != m.shape: raise ValueError("metric shapes must match")
    return p[m], y[m]


def compute_metrics(pred: Tensor | np.ndarray, target: Tensor | np.ndarray, mask: Tensor | np.ndarray | None = None) -> dict[str, float]:
    """Compute Cycle88-compatible metrics with separate numeric/directional masks.

    Numeric errors include every finite target, including zero spread.  Direction
    metrics exclude only zero targets; a zero prediction is deliberately neither
    positive nor negative and therefore is wrong for either non-zero class.
    """
    p, y = _arrays(pred, target, mask)
    finite = np.isfinite(p) & np.isfinite(y)
    p_num, y_num = p[finite], y[finite]
    if len(y_num) == 0:
        raise ValueError("no finite target values")
    direction = finite & (y != 0)
    p_dir, y_dir = p[direction], y[direction]
    if len(y_dir) == 0:
        direction_accuracy = pos_recall = neg_recall = balanced = float("nan")
        all_positive = all_negative = float("nan")
    else:
        pred_sign = np.sign(p_dir)
        true_sign = np.sign(y_dir)
        pos, neg = true_sign > 0, true_sign < 0
        direction_accuracy = float((pred_sign == true_sign).mean())
        pos_recall = float(((pred_sign == 1) & pos).sum() / pos.sum()) if pos.any() else float("nan")
        neg_recall = float(((pred_sign == -1) & neg).sum() / neg.sum()) if neg.any() else float("nan")
        balanced = float(np.nanmean([pos_recall, neg_recall]))
        all_positive = float(pos.mean())
        all_negative = float(neg.mean())
    residual = p_num - y_num
    return {
        "direction_accuracy": direction_accuracy,
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "balanced_accuracy": balanced,
        "all_positive_baseline": all_positive,
        "all_negative_baseline": all_negative,
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "sample_count": float(len(y_num)),
        "direction_sample_count": float(len(y_dir)),
    }


def metric_by_forecast_offset(pred: np.ndarray, target: np.ndarray, bridge_hours: int = 10) -> list[dict[str, Any]]:
    if pred.shape != target.shape or pred.ndim != 2: raise ValueError("offset metrics expect [N,H]")
    rows = []
    for i in range(pred.shape[1]):
        m = compute_metrics(pred[:, i], target[:, i])
        valid = np.isfinite(pred[:, i]) & np.isfinite(target[:, i])
        bias = float(np.mean(pred[valid, i] - target[valid, i])) if valid.any() else float("nan")
        rows.append({"offset": i + 1, "section": "bridge" if i < bridge_hours else "D-day", "bias": bias, **m})
    return rows
