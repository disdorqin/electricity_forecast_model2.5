"""Strategy-specific views over the strict daily-origin business samples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .business_dataset import BusinessSample, build_business_split
from .canonical_source import CanonicalHourlySource
from .normalization import RobustArrayScaler, TargetScale, fit_target_scale


@dataclass
class StrategyView:
    """Unscaled view of one legal H34 sample for a strategy/stage."""

    sample: BusinessSample
    future_slice: slice
    bridge_context: np.ndarray | None = None

    @property
    def y_future(self) -> np.ndarray:
        return self.sample.y_future[self.future_slice]

    @property
    def x_future(self) -> np.ndarray:
        return self.sample.x_future[self.future_slice]


class StrategyDataset(Dataset):
    """Tensor dataset with train-only target/exogenous scaling."""

    def __init__(self, views: list[StrategyView], target_scale: TargetScale, x_scaler: RobustArrayScaler, context_scale: float | None = None):
        if not views:
            raise ValueError("strategy dataset cannot be empty")
        self.views = views
        self.target_scale = target_scale
        self.x_scaler = x_scaler
        self.context_scale = context_scale
        self.horizon = views[0].y_future.shape[0]
        self.n_features = views[0].sample.x_backcast.shape[1] + (10 if views[0].bridge_context is not None else 0)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> dict[str, Any]:
        view = self.views[index]
        y_back = view.sample.y_backcast / float(self.target_scale.scale)
        x_back = self.x_scaler.transform(view.sample.x_backcast)
        x_future = self.x_scaler.transform(view.x_future)
        if view.bridge_context is not None:
            if self.context_scale is None or self.context_scale <= 0:
                raise ValueError("bridge context requires a positive context scale")
            context = np.asarray(view.bridge_context, dtype=np.float32) / float(self.context_scale)
            x_back = np.concatenate([x_back, np.repeat(context[None, :], len(x_back), axis=0)], axis=1)
            x_future = np.concatenate([x_future, np.repeat(context[None, :], len(x_future), axis=0)], axis=1)
        if x_back.shape[1] != self.n_features or x_future.shape[1] != self.n_features:
            raise AssertionError("strategy feature width mismatch")
        horizon = self.horizon
        return {
            "y_backcast": torch.from_numpy(y_back).float(),
            "x_backcast": torch.from_numpy(x_back).float(),
            "x_future": torch.from_numpy(x_future).float(),
            "y_future": torch.from_numpy(view.y_future / float(self.target_scale.scale)).float(),
            "bridge_mask": torch.zeros(horizon).float(),
            "score_mask": torch.ones(horizon).float(),
            "target_day": view.sample.target_day,
            "origin_timestamp": view.sample.origin_timestamp,
            "training_last_day": view.sample.training_last_day,
        }


def _views(samples: list[BusinessSample], future_slice: slice, *, bridge_context: bool = False, context_scale: float | None = None) -> list[StrategyView]:
    """Create a deterministic strategy view without changing sample dates."""
    return [StrategyView(s, future_slice, s.y_future[:10].copy() if bridge_context else None) for s in samples]


def build_strategy_split(
    source: CanonicalHourlySource,
    target_day: str,
    *,
    strategy: str,
    stage: str = "direct",
    validation_days: int = 28,
    training_months: int = 36,
) -> tuple[StrategyDataset, StrategyDataset, dict[str, Any]]:
    """Build C1 or C2A data from the strict H34 split.

    C1 trains only on D-day h1-h24. C2A stage 1 trains on the bridge and
    stage 2 receives true historical bridge values in training/validation.
    No target-day labels are read by this builder.
    """
    if strategy == "C1_GAP_DIRECT_D24":
        future_slice, bridge = slice(10, 34), False
    elif strategy == "C2A_BRIDGE_TF" and stage == "stage1":
        future_slice, bridge = slice(0, 10), False
    elif strategy == "C2A_BRIDGE_TF" and stage == "stage2":
        future_slice, bridge = slice(10, 34), True
    else:
        raise ValueError(f"unsupported strategy/stage: {strategy}/{stage}")
    base_train, base_val, base_manifest = build_business_split(source, target_day, validation_days=validation_days, training_months=training_months, feature_profile="CORE5_RAW")
    train_views = _views(base_train.samples, future_slice, bridge_context=bridge)
    val_views = _views(base_val.samples, future_slice, bridge_context=bridge)
    target_scale = fit_target_scale([view.y_future for view in train_views])
    x_scaler = RobustArrayScaler.fit(
        [view.sample.x_backcast for view in train_views] + [view.x_future for view in train_views],
        numeric_channels=tuple(range(5)),
    )
    context_scale = None
    if bridge:
        # Context is a target-derived representation, normalized by the
        # stage-1 train bridge scale and never fitted on validation/target D.
        context_scale = float(max(np.median(np.concatenate([np.abs(view.bridge_context) for view in train_views])), 10.0))
    train = StrategyDataset(train_views, target_scale, x_scaler, context_scale)
    val = StrategyDataset(val_views, target_scale, x_scaler, context_scale)
    manifest = dict(base_manifest)
    manifest.update({
        "strategy": strategy, "stage": stage, "target_slice": {"start": future_slice.start, "stop": future_slice.stop},
        "target_scale": {"scale": target_scale.scale, "floor": target_scale.floor},
        "x_scaler": x_scaler.to_dict(), "context_scale": context_scale,
        "input_feature_count": train.n_features, "horizon": train.horizon,
        "label_source": "historical_samples_only; target-day labels excluded from builder",
    })
    return train, val, manifest
