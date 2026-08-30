from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..contracts import CORE5_FEATURES
from .canonical_source import CANONICAL_TO_SOURCE


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source_column: str
    role: str
    availability: str
    allowed_at_origin: bool = True


FEATURE_REGISTRY = tuple(
    [FeatureSpec(name, CANONICAL_TO_SOURCE[name], "forecast", "published_forecast_available_at_D-1_14:00") for name in CORE5_FEATURES[:5]]
    + [
        FeatureSpec("hour_sin", "timestamp", "calendar", "known_in_advance"),
        FeatureSpec("hour_cos", "timestamp", "calendar", "known_in_advance"),
        FeatureSpec("dow_sin", "timestamp", "calendar", "known_in_advance"),
        FeatureSpec("dow_cos", "timestamp", "calendar", "known_in_advance"),
    ]
)

PHYSICAL_SHAPE_FEATURES = (
    "renewable_total", "net_load_proxy", "bidding_stress_ratio",
    "ramp_direct_load", "ramp_interconnection_received_load", "ramp_wind",
    "ramp_solar", "ramp_bidding_space",
)
CAUSAL_PRICE_STATE_FEATURES = (
    "last_spread_at_origin", "D1_h1_h14_mean", "D1_h1_h14_std",
    "D1_h1_h14_positive_fraction", "D1_h1_h14_linear_slope",
    "D1_h1_h14_sign_switch_count", "trailing_7d_spread_mean",
    "trailing_7d_spread_std", "trailing_28d_spread_mean", "trailing_28d_spread_std",
)


def feature_names(feature_profile: str = "CORE5_RAW") -> tuple[str, ...]:
    """Return the frozen channel names for one compact feature package."""
    if feature_profile == "CORE5_RAW":
        return tuple(s.name for s in FEATURE_REGISTRY)
    if feature_profile == "PHYSICAL_SHAPE":
        return tuple(s.name for s in FEATURE_REGISTRY) + PHYSICAL_SHAPE_FEATURES
    if feature_profile == "CAUSAL_PRICE_STATE":
        return tuple(s.name for s in FEATURE_REGISTRY) + CAUSAL_PRICE_STATE_FEATURES
    raise ValueError(f"unsupported feature profile: {feature_profile}")


def calendar_features(timestamps: tuple[pd.Timestamp, ...] | list[pd.Timestamp]) -> np.ndarray:
    """Return [time, 4] known-in-advance cyclical calendar channels."""
    ts = pd.DatetimeIndex(timestamps)
    hour = ts.hour.to_numpy(dtype=np.float32)
    dow = ts.dayofweek.to_numpy(dtype=np.float32)
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
        ]
    ).astype(np.float32)


def _core5_matrix(frame: pd.DataFrame, idx: pd.DatetimeIndex) -> np.ndarray:
    """Build the origin-safe primitive forecast/calendar channels."""
    numeric = frame.loc[idx, [s.source_column for s in FEATURE_REGISTRY[:5]]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if not np.isfinite(numeric).all():
        raise ValueError("CORE5 covariates contain NaN/Inf")
    return np.concatenate([numeric, calendar_features(list(idx))], axis=1).astype(np.float32)


def _physical_shape_matrix(core: np.ndarray) -> np.ndarray:
    """Derive compact physical shape channels from forecast primitives only."""
    numeric = core[:, :5]
    renewable = numeric[:, 2] + numeric[:, 3]
    net_load = numeric[:, 0] - renewable
    stress = numeric[:, 4] / np.maximum(np.abs(numeric[:, 0]), 1.0)
    ramps = np.vstack([np.zeros((1, 5), dtype=np.float32), np.diff(numeric, axis=0)])
    return np.column_stack([renewable, net_load, stress, ramps]).astype(np.float32)


def _causal_price_state(frame: pd.DataFrame, visible_timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Compute F2 state from spread values visible at the historical origin only."""
    spread = pd.to_numeric(frame["日前电价"], errors="coerce") - pd.to_numeric(frame["实时电价"], errors="coerce")
    visible = spread.loc[spread.index.intersection(visible_timestamps)].dropna().to_numpy(np.float64)
    if len(visible) < 14:
        raise ValueError("not enough visible spread values for causal price state")
    d1 = visible[-14:]
    all_visible = spread.loc[:visible_timestamps[-1]].dropna().to_numpy(np.float64)
    trailing7 = all_visible[-min(len(all_visible), 7 * 24):]
    trailing28 = all_visible[-min(len(all_visible), 28 * 24):]
    slope = float(np.polyfit(np.arange(len(d1), dtype=float), d1, 1)[0]) if len(d1) > 1 else 0.0
    state = np.asarray([
        d1[-1], np.mean(d1), np.std(d1), np.mean(d1 > 0), slope,
        np.count_nonzero(np.sign(d1[1:]) != np.sign(d1[:-1])),
        np.mean(trailing7), np.std(trailing7), np.mean(trailing28), np.std(trailing28),
    ], dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("causal price state contains NaN/Inf")
    return state


def build_feature_matrices(
    frame: pd.DataFrame,
    backcast_timestamps: tuple[pd.Timestamp, ...] | list[pd.Timestamp],
    future_timestamps: tuple[pd.Timestamp, ...] | list[pd.Timestamp],
    *,
    feature_profile: str = "CORE5_RAW",
) -> tuple[np.ndarray, np.ndarray]:
    """Build backcast/future channels while preserving the origin-safe feature contract."""
    back_idx = pd.DatetimeIndex(backcast_timestamps)
    future_idx = pd.DatetimeIndex(future_timestamps)
    all_idx = back_idx.append(future_idx)
    missing = all_idx.difference(frame.index)
    if len(missing):
        raise ValueError(f"missing covariate timestamps, first={missing[0]}")
    core = _core5_matrix(frame, all_idx)
    pieces = [core]
    if feature_profile == "PHYSICAL_SHAPE":
        pieces.append(_physical_shape_matrix(core))
    elif feature_profile == "CAUSAL_PRICE_STATE":
        pieces.append(np.repeat(_causal_price_state(frame, back_idx)[None, :], len(all_idx), axis=0))
    elif feature_profile != "CORE5_RAW":
        raise ValueError(f"unsupported feature profile: {feature_profile}")
    combined = np.concatenate(pieces, axis=1).astype(np.float32)
    expected = len(feature_names(feature_profile))
    if combined.shape != (len(all_idx), expected):
        raise AssertionError(f"feature shape mismatch: {combined.shape}, expected {(len(all_idx), expected)}")
    return combined[: len(back_idx)], combined[len(back_idx) :]


def build_covariate_matrix(frame: pd.DataFrame, timestamps: tuple[pd.Timestamp, ...] | list[pd.Timestamp]) -> np.ndarray:
    """Build raw CORE5 + calendar features; no actual columns are read."""
    idx = pd.DatetimeIndex(timestamps)
    missing = idx.difference(frame.index)
    if len(missing):
        raise ValueError(f"missing covariate timestamps, first={missing[0]}")
    return _core5_matrix(frame, idx)
