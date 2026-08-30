from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetScale:
    scale: float
    floor: float = 10.0

    def transform(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) / self.scale

    def inverse(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32) * self.scale


@dataclass
class RobustArrayScaler:
    median: np.ndarray
    iqr: np.ndarray
    floor: float = 1e-6
    numeric_channels: tuple[int, ...] | None = None
    identity_channels: tuple[int, ...] = ()

    @classmethod
    def fit(
        cls,
        arrays: list[np.ndarray],
        *,
        numeric_channels: tuple[int, ...] | None = None,
    ) -> "RobustArrayScaler":
        if not arrays:
            raise ValueError("cannot fit scaler on empty training arrays")
        x = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        n_channels = x.shape[1]
        if numeric_channels is None:
            numeric_channels = tuple(range(n_channels))
        numeric_channels = tuple(sorted(set(numeric_channels)))
        if any(c < 0 or c >= n_channels for c in numeric_channels):
            raise ValueError("numeric channel index is outside the feature matrix")
        identity = tuple(c for c in range(n_channels) if c not in numeric_channels)
        median = np.zeros(n_channels, dtype=np.float32)
        iqr = np.ones(n_channels, dtype=np.float32)
        if numeric_channels:
            numeric = x[:, numeric_channels]
            median[list(numeric_channels)] = np.nanmedian(numeric, axis=0).astype(np.float32)
            q75, q25 = np.nanpercentile(numeric, [75, 25], axis=0)
            iqr[list(numeric_channels)] = np.maximum(q75 - q25, cls.floor).astype(np.float32)
        return cls(median, iqr, cls.floor, numeric_channels, identity)

    def transform(self, x: np.ndarray) -> np.ndarray:
        value = np.asarray(x, dtype=np.float32)
        if value.shape[-1] != self.median.shape[0]:
            raise ValueError("feature count does not match fitted scaler")
        # Calendar channels are deliberately identity-transformed.  Their
        # bounded trigonometric semantics must not depend on train-day
        # composition or on future truth.
        return ((value - self.median) / self.iqr).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
            "floor": self.floor,
            "numeric_channels": list(self.numeric_channels or ()),
            "identity_channels": list(self.identity_channels),
            "transform_policy": "robust_numeric_only_calendar_identity",
        }


def fit_target_scale(targets: list[np.ndarray], floor: float = 10.0) -> TargetScale:
    if not targets:
        raise ValueError("cannot fit target scale on empty training targets")
    values = np.concatenate([np.abs(np.asarray(x, dtype=np.float64).ravel()) for x in targets])
    scale = max(float(np.median(values)), floor)
    return TargetScale(scale=scale, floor=floor)
