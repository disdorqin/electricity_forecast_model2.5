from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..data.covariates import FEATURE_REGISTRY, CAUSAL_PRICE_STATE_FEATURES, PHYSICAL_SHAPE_FEATURES, feature_names
from ..data.origin_index import build_origin_window


def tensor_hash(value: np.ndarray) -> str:
    """Stable hash used by counterfactual and inference-isolation audits."""
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def build_input_lineage(target_day: str, feature_profile: str = "CORE5_RAW") -> dict[str, Any]:
    """Describe every model channel and its origin-safe timestamp range."""
    window = build_origin_window(target_day)
    back = [x.isoformat() for x in window.backcast_timestamps]
    future = [x.isoformat() for x in window.forecast_timestamps]
    channels = []
    base_count = len(FEATURE_REGISTRY)
    derived = PHYSICAL_SHAPE_FEATURES if feature_profile == "PHYSICAL_SHAPE" else CAUSAL_PRICE_STATE_FEATURES if feature_profile == "CAUSAL_PRICE_STATE" else ()
    for index, spec in enumerate(FEATURE_REGISTRY):
        channels.append({
            "channel": index, "feature": spec.name, "source_column": spec.source_column,
            "role": spec.role, "availability": spec.availability,
            "origin_safe": spec.allowed_at_origin,
            "backcast_timestamps": [back[0], back[-1]],
            "future_timestamps": [future[0], future[-1]],
        })
    for index, name in enumerate(derived, start=base_count):
        is_state = feature_profile == "CAUSAL_PRICE_STATE"
        channels.append({
            "channel": index, "feature": name,
            "source_column": "forecast_primitive_transform" if not is_state else "日前电价 - 实时电价",
            "role": "forecast_derived" if not is_state else "historical_target_state",
            "availability": "published_forecast_available_at_D-1_14:00" if not is_state else "D-1_h1_h14_or_matured_history",
            "origin_safe": True,
            "backcast_timestamps": [back[0], back[-1]],
            "future_timestamps": [future[0], future[-1]],
        })
    if len(channels) != len(feature_names(feature_profile)):
        raise AssertionError("lineage channel count does not match feature profile")
    return {"target_day": target_day, "feature_profile": feature_profile, "origin": window.origin_timestamp.isoformat(),
            "channels": channels, "target_backcast": {
                "source": "日前电价 - 实时电价", "role": "historical_target_lag",
                "latest_timestamp": window.origin_timestamp.isoformat(), "origin_safe": True}}
