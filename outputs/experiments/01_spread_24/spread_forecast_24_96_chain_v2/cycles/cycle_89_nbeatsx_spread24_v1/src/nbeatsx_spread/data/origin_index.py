from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utils.resolution import HOURLY, Resolution

from ..contracts import BusinessContract, STRICT34, assert_contract, latest_complete_label_day


@dataclass(frozen=True)
class OriginWindow:
    target_day: str
    origin_timestamp: pd.Timestamp
    backcast_timestamps: tuple[pd.Timestamp, ...]
    bridge_timestamps: tuple[pd.Timestamp, ...]
    scored_timestamps: tuple[pd.Timestamp, ...]

    @property
    def forecast_timestamps(self) -> tuple[pd.Timestamp, ...]:
        return self.bridge_timestamps + self.scored_timestamps

    @property
    def training_last_day(self) -> str:
        return latest_complete_label_day(self.target_day)


def build_origin_window(
    target_day: str,
    contract: BusinessContract = STRICT34,
    resolution: Resolution = HOURLY,
) -> OriginWindow:
    """Construct the exact v1 D-1 14:00 -> D h24 timeline.

    Uses the repository's business-time conversion so h24 retains the project
    convention of D+1 00:00 while belonging to business day D.
    """
    assert_contract(contract)
    if resolution.label != "hourly":
        raise ValueError("Cycle 89 v1 is hourly-only")

    d = pd.Timestamp(target_day)
    previous_day = (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    origin = resolution.timestamp_from_business(previous_day, contract.forecast_origin_hour)

    backcast = tuple(
        pd.date_range(
            end=origin,
            periods=contract.input_hours,
            freq=resolution.freq,
        )
    )
    bridge = tuple(
        resolution.timestamp_from_business(previous_day, slot)
        for slot in range(contract.forecast_origin_hour + 1, resolution.slots_per_day + 1)
    )
    scored = tuple(
        resolution.timestamp_from_business(target_day, slot)
        for slot in range(1, resolution.slots_per_day + 1)
    )

    window = OriginWindow(
        target_day=target_day,
        origin_timestamp=pd.Timestamp(origin),
        backcast_timestamps=backcast,
        bridge_timestamps=bridge,
        scored_timestamps=scored,
    )
    assert_origin_window(window, contract=contract, resolution=resolution)
    return window


def assert_origin_window(
    window: OriginWindow,
    contract: BusinessContract = STRICT34,
    resolution: Resolution = HOURLY,
) -> None:
    if len(window.backcast_timestamps) != contract.input_hours:
        raise AssertionError("backcast length mismatch")
    if len(window.bridge_timestamps) != contract.bridge_hours:
        raise AssertionError("bridge length mismatch")
    if len(window.scored_timestamps) != contract.scored_hours:
        raise AssertionError("scored length mismatch")
    if len(window.forecast_timestamps) != contract.horizon:
        raise AssertionError("forecast horizon mismatch")
    if window.backcast_timestamps[-1] != window.origin_timestamp:
        raise AssertionError("backcast must end exactly at forecast origin")
    expected_first = window.origin_timestamp + pd.Timedelta(minutes=resolution.minutes_per_slot)
    if window.forecast_timestamps[0] != expected_first:
        raise AssertionError("forecast must start one slot after the origin")
    if window.bridge_timestamps[-1] != pd.Timestamp(window.target_day):
        raise AssertionError("D-1 h24 must be target-day midnight")
    if window.scored_timestamps[0] != pd.Timestamp(window.target_day) + pd.Timedelta(hours=1):
        raise AssertionError("D h1 must be target-day 01:00")
    if resolution.business_day_from_timestamp(window.scored_timestamps[-1]) != window.target_day:
        raise AssertionError("final scored timestamp must belong to target business day")


def strict_train_days(
    target_day: str,
    candidate_days: list[str] | tuple[str, ...],
    contract: BusinessContract = STRICT34,
) -> list[str]:
    """Filter complete daily labels using the business-time D-2 rule."""
    assert_contract(contract)
    latest = pd.Timestamp(latest_complete_label_day(target_day, contract))
    days = sorted(pd.Timestamp(d).strftime("%Y-%m-%d") for d in candidate_days)
    result = [d for d in days if pd.Timestamp(d) <= latest]
    if result and pd.Timestamp(result[-1]) > latest:
        raise AssertionError("strict training helper admitted a label after D-2")
    return result
