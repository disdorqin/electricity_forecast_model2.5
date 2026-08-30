from __future__ import annotations

import numpy as np

from ..contracts import STRICT34
from ..data.canonical_source import CanonicalHourlySource
from ..data.covariates import build_feature_matrices, feature_names
from ..data.origin_index import build_origin_window
from .common import AuditResult, result


def audit_covariate_availability(source: CanonicalHourlySource, target_day: str, feature_profile: str = "CORE5_RAW") -> AuditResult:
    try:
        w = build_origin_window(target_day, STRICT34)
        _, x = build_feature_matrices(source.frame, list(w.backcast_timestamps), list(w.forecast_timestamps), feature_profile=feature_profile)
        if x.shape != (STRICT34.horizon, len(feature_names(feature_profile))) or not np.isfinite(x).all():
            return result("future_covariate_availability", False, f"invalid tensor shape={x.shape}")
        return result("future_covariate_availability", True, f"34 rows x {x.shape[1]} channels; profile={feature_profile}; origin-safe transformations")
    except Exception as exc:
        return result("future_covariate_availability", False, str(exc))
