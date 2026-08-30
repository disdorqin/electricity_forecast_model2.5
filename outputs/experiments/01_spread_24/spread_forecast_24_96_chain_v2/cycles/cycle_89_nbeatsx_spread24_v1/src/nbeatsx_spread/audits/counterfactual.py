from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import STRICT34
from ..data.business_dataset import build_inference_sample
from ..data.canonical_source import CanonicalHourlySource
from ..data.origin_index import build_origin_window
from .common import AuditResult, result


def run_counterfactual_audit(source: CanonicalHourlySource, target_day: str, feature_profile: str = "CORE5_RAW") -> list[AuditResult]:
    """Mutate forbidden truth and allowed forecast inputs to prove boundary behavior."""
    base = build_inference_sample(source, target_day, feature_profile=feature_profile)
    w = build_origin_window(target_day, STRICT34)
    future = list(w.forecast_timestamps)
    frame = source.frame.copy()
    actual_cols = [c for c in frame.columns if "实际值" in c] + ["日前电价", "实时电价"]
    for col in actual_cols:
        frame.loc[frame.index.intersection(future), col] = frame.loc[frame.index.intersection(future), col].fillna(0) + 9999.0
    forbidden_sample = build_inference_sample(CanonicalHourlySource.from_frame(frame), target_day, feature_profile=feature_profile)
    post14_same = (
        np.array_equal(base.y_backcast, forbidden_sample.y_backcast)
        and np.array_equal(base.x_backcast, forbidden_sample.x_backcast)
        and np.array_equal(base.x_future, forbidden_sample.x_future)
    )
    allowed = source.frame.copy()
    fcol = "直调负荷预测值"
    allowed.loc[allowed.index.intersection(future), fcol] = allowed.loc[allowed.index.intersection(future), fcol].fillna(0) + 1.0
    allowed_sample = build_inference_sample(CanonicalHourlySource.from_frame(allowed), target_day, feature_profile=feature_profile)
    changed = not np.array_equal(base.x_future, allowed_sample.x_future)
    return [result("d1_post14_and_target_actual_counterfactual", post14_same, "mutating target actual prices/fundamentals leaves model tensors unchanged"), result("allowed_core5_counterfactual", changed, "mutating a legal CORE5 forecast changes future covariates")]
