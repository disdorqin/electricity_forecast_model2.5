"""Metrics specific to the signed 24-point spread experiment.

The project-wide price SMAPE clips prices below 50 because prices are
non-negative.  A signed spread must not use that clip: it would collapse all
negative spreads onto the same artificial positive value.  This module keeps
the standard symmetric absolute-percentage definition for spread values and
returns percentages in the conventional 0--200 range.
"""

from __future__ import annotations

import numpy as np


def smape_terms_percent(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return per-observation signed-spread sMAPE contributions in percent.

    When both values are exactly zero, the contribution is defined as zero.
    If only one value is zero, the contribution is 200 percent, matching the
    limiting value of the symmetric absolute-percentage formula.
    """

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("sMAPE requires finite true and predicted values")
    denominator = np.abs(true) + np.abs(pred)
    numerator = 2.0 * np.abs(pred - true)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    ) * 100.0


def smape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean signed-spread sMAPE in percent."""

    return float(np.mean(smape_terms_percent(y_true, y_pred)))
