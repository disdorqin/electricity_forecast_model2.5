from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..contracts import BusinessContract, STRICT34, assert_contract, latest_complete_label_day
from .canonical_source import CanonicalHourlySource
from .covariates import FEATURE_REGISTRY, build_feature_matrices, feature_names
from .normalization import RobustArrayScaler, TargetScale, fit_target_scale
from .origin_index import OriginWindow, build_origin_window, strict_train_days


@dataclass
class BusinessSample:
    target_day: str
    origin_timestamp: str
    training_last_day: str
    y_backcast: np.ndarray
    x_backcast: np.ndarray
    x_future: np.ndarray
    y_future: np.ndarray
    bridge_mask: np.ndarray
    score_mask: np.ndarray
    feature_availability_mask: np.ndarray

    def tensors(self) -> dict[str, Any]:
        return {
            "y_backcast": torch.from_numpy(self.y_backcast).float(),
            "x_backcast": torch.from_numpy(self.x_backcast).float(),
            "x_future": torch.from_numpy(self.x_future).float(),
            "y_future": torch.from_numpy(self.y_future).float(),
            "bridge_mask": torch.from_numpy(self.bridge_mask).float(),
            "score_mask": torch.from_numpy(self.score_mask).float(),
            "feature_availability_mask": torch.from_numpy(self.feature_availability_mask).float(),
            "target_day": self.target_day,
            "origin_timestamp": self.origin_timestamp,
            "training_last_day": self.training_last_day,
        }


@dataclass
class InferenceSample:
    """Origin-safe model inputs; intentionally contains no target-day labels."""
    target_day: str
    origin_timestamp: str
    training_last_day: str
    y_backcast: np.ndarray
    x_backcast: np.ndarray
    x_future: np.ndarray
    feature_availability_mask: np.ndarray

    def tensors(self) -> dict[str, Any]:
        return {
            "y_backcast": torch.from_numpy(self.y_backcast).float(),
            "x_backcast": torch.from_numpy(self.x_backcast).float(),
            "x_future": torch.from_numpy(self.x_future).float(),
            "feature_availability_mask": torch.from_numpy(self.feature_availability_mask).float(),
            "target_day": self.target_day,
            "origin_timestamp": self.origin_timestamp,
            "training_last_day": self.training_last_day,
        }


def spread_series(frame: pd.DataFrame) -> pd.Series:
    """Return the frozen Cycle 89 target DA - RT."""
    value = pd.to_numeric(frame["日前电价"], errors="coerce") - pd.to_numeric(frame["实时电价"], errors="coerce")
    return value.astype(np.float32)


def business_day_for_timestamp(ts: pd.Timestamp) -> str:
    return (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if ts.hour == 0 else ts.strftime("%Y-%m-%d")


def complete_business_days(source: CanonicalHourlySource, contract: BusinessContract = STRICT34) -> list[str]:
    """Find days with all required observed labels and CORE5 rows."""
    assert_contract(contract)
    frame = source.frame
    labels = spread_series(frame)
    days: list[str] = []
    for day in source.business_days():
        window = build_origin_window(day, contract)
        ts = list(window.forecast_timestamps)
        all_ts = list(window.backcast_timestamps) + ts
        if any(t not in frame.index for t in all_ts):
            continue
        if labels.loc[frame.index.intersection(ts)].isna().any():
            continue
        required = source.source_columns(("fcast_直调负荷", "fcast_联络线受电负荷", "fcast_风电总加", "fcast_光伏总加", "fcast_竞价空间"))
        if frame.loc[all_ts, required].apply(pd.to_numeric, errors="coerce").isna().any().any():
            continue
        days.append(day)
    return days


def _make_sample(source: CanonicalHourlySource, target_day: str, require_target: bool = True, contract: BusinessContract = STRICT34, feature_profile: str = "CORE5_RAW") -> BusinessSample:
    assert_contract(contract)
    window = build_origin_window(target_day, contract)
    frame = source.frame
    y = spread_series(frame)
    back_ts = list(window.backcast_timestamps)
    future_ts = list(window.forecast_timestamps)
    if any(t not in frame.index for t in back_ts + future_ts):
        raise ValueError(f"incomplete timestamp coverage for target day {target_day}")
    x_back, x_future = build_feature_matrices(frame, back_ts, future_ts, feature_profile=feature_profile)
    y_back = y.loc[back_ts].to_numpy(np.float32)
    y_future = y.loc[future_ts].to_numpy(np.float32)
    if not np.isfinite(y_back).all() or (require_target and not np.isfinite(y_future).all()):
        raise ValueError(f"target contains missing values for {target_day}")
    if not require_target:
        y_future = np.nan_to_num(y_future, nan=0.0).astype(np.float32)
    bridge_mask = np.r_[np.ones(contract.bridge_hours), np.zeros(contract.scored_hours)].astype(np.float32)
    score_mask = 1.0 - bridge_mask
    availability = np.ones((contract.horizon, x_future.shape[1]), dtype=np.float32)
    return BusinessSample(
        target_day=target_day,
        origin_timestamp=window.origin_timestamp.isoformat(),
        training_last_day=latest_complete_label_day(target_day, contract),
        y_backcast=y_back,
        x_backcast=x_back,
        x_future=x_future,
        y_future=y_future,
        bridge_mask=bridge_mask,
        score_mask=score_mask,
        feature_availability_mask=availability,
    )


def build_inference_sample(source: CanonicalHourlySource, target_day: str, contract: BusinessContract = STRICT34, feature_profile: str = "CORE5_RAW") -> InferenceSample:
    """Build only the legal input tensor for target day D.

    The target-day DA/RT labels are not read here.  Evaluation must call
    :func:`load_evaluation_labels` only after model forward has completed.
    """
    assert_contract(contract)
    window = build_origin_window(target_day, contract)
    frame = source.frame
    back_ts = list(window.backcast_timestamps)
    future_ts = list(window.forecast_timestamps)
    if any(t not in frame.index for t in back_ts + future_ts):
        raise ValueError(f"incomplete inference timestamp coverage for {target_day}")
    y = spread_series(frame)
    y_back = y.loc[back_ts].to_numpy(np.float32)
    if not np.isfinite(y_back).all():
        raise ValueError("inference backcast contains missing labels")
    x_back, x_future = build_feature_matrices(frame, back_ts, future_ts, feature_profile=feature_profile)
    if x_future.shape != (contract.horizon, len(feature_names(feature_profile))):
        raise AssertionError("inference future covariate shape mismatch")
    return InferenceSample(target_day, window.origin_timestamp.isoformat(), latest_complete_label_day(target_day, contract), y_back, x_back, x_future, np.ones_like(x_future, dtype=np.float32))


def load_evaluation_labels(source: CanonicalHourlySource, target_day: str, contract: BusinessContract = STRICT34) -> np.ndarray:
    """Load labels after inference; never used by ``build_inference_sample``."""
    window = build_origin_window(target_day, contract)
    labels = spread_series(source.frame).loc[list(window.forecast_timestamps)].to_numpy(np.float32)
    if not np.isfinite(labels).all():
        raise ValueError(f"missing evaluation labels for {target_day}")
    return labels


class BusinessDataset(Dataset):
    """One legal daily-origin sample per target day."""

    def __init__(self, samples: list[BusinessSample], target_scale: TargetScale | None = None, x_scaler: RobustArrayScaler | None = None):
        self.samples = samples
        self.target_scale = target_scale
        self.x_scaler = x_scaler

    @classmethod
    def from_days(cls, source: CanonicalHourlySource, days: list[str], require_target: bool = True, target_scale: TargetScale | None = None, x_scaler: RobustArrayScaler | None = None, feature_profile: str = "CORE5_RAW") -> "BusinessDataset":
        samples = [_make_sample(source, d, require_target=require_target, feature_profile=feature_profile) for d in days]
        return cls(samples, target_scale=target_scale, x_scaler=x_scaler)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index].tensors()
        if self.target_scale is not None:
            item["y_backcast"] = item["y_backcast"] / float(self.target_scale.scale)
            item["y_future"] = item["y_future"] / float(self.target_scale.scale)
        if self.x_scaler is not None:
            item["x_backcast"] = torch.from_numpy(self.x_scaler.transform(item["x_backcast"].numpy())).float()
            item["x_future"] = torch.from_numpy(self.x_scaler.transform(item["x_future"].numpy())).float()
        return item

    def split_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "target_day": s.target_day,
                "origin_timestamp": s.origin_timestamp,
                "training_last_day": s.training_last_day,
                "backcast_first": str(s.y_backcast.shape[0]),
                "backcast_length": int(s.y_backcast.shape[0]),
                "horizon": int(s.y_future.shape[0]),
            }
            for s in self.samples
        ]


def build_business_split(source: CanonicalHourlySource, target_day: str, contract: BusinessContract = STRICT34, validation_days: int = 28, training_months: int = 9, feature_profile: str = "CORE5_RAW") -> tuple[BusinessDataset, BusinessDataset, dict[str, Any]]:
    """Build the strict rolling calibration/train/validation split for D."""
    assert_contract(contract)
    latest = pd.Timestamp(latest_complete_label_day(target_day, contract))
    start = latest - pd.DateOffset(months=training_months) + pd.Timedelta(days=1)
    available = complete_business_days(source, contract)
    calibration = strict_train_days(target_day, [d for d in available if start <= pd.Timestamp(d)], contract)
    if len(calibration) <= validation_days:
        raise ValueError(f"not enough calibration days: {len(calibration)} <= {validation_days}")
    val_days = calibration[-validation_days:]
    train_days = calibration[:-validation_days]
    if pd.Timestamp(train_days[-1]) > latest or pd.Timestamp(val_days[-1]) > latest:
        raise AssertionError("split admitted labels newer than D-2")
    train = BusinessDataset.from_days(source, train_days, feature_profile=feature_profile)
    val = BusinessDataset.from_days(source, val_days, feature_profile=feature_profile)
    target_scale = fit_target_scale([s.y_future for s in train.samples])
    # CORE5 are the only train-fitted numeric channels.  Calendar encodings
    # are deterministic functions of timestamps and remain identity-scaled.
    x_scaler = RobustArrayScaler.fit(
        [s.x_backcast for s in train.samples] + [s.x_future for s in train.samples],
        numeric_channels=tuple(range(len(feature_names(feature_profile)) - 4)),
    )
    train.target_scale = val.target_scale = target_scale
    train.x_scaler = val.x_scaler = x_scaler
    manifest = {
        "target_day": target_day,
        "training_history_months": int(training_months),
        "validation_history_days": int(validation_days),
        "feature_profile": feature_profile,
        "forecast_origin": "D-1 14:00",
        "training_last_day": latest.strftime("%Y-%m-%d"),
        "calibration_start": start.strftime("%Y-%m-%d"),
        "calibration_end": latest.strftime("%Y-%m-%d"),
        "train_days": train_days,
        "validation_days": val_days,
        "train_count": len(train),
        "validation_count": len(val),
        "target_scale": asdict(target_scale),
        "x_scaler": x_scaler.to_dict(),
        "x_scaler_fit_scope": {
            "samples": "train_days only",
            "channels": {
                "numeric_robust": list(range(len(feature_names(feature_profile)) - 4)),
                "calendar_identity": list(range(len(feature_names(feature_profile)) - 4, len(feature_names(feature_profile)))),
            },
        },
        "final_holdout_touched": False,
    }
    return train, val, manifest
