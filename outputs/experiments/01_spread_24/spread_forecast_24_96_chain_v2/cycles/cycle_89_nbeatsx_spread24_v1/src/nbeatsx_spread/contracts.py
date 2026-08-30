from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class BusinessContract:
    target_name: str = "spread_DA_minus_RT"
    forecast_origin_hour: int = 14
    input_hours: int = 168
    bridge_hours: int = 10
    scored_hours: int = 24
    training_label_lag_days: int = 2
    target_day_da_as_feature: bool = False
    target_day_actual_as_feature: bool = False
    d1_post14_realized_as_feature: bool = False

    @property
    def horizon(self) -> int:
        return self.bridge_hours + self.scored_hours


STRICT34 = BusinessContract()
CORE5_FEATURES: Final[tuple[str, ...]] = (
    "fcast_直调负荷",
    "fcast_联络线受电负荷",
    "fcast_风电总加",
    "fcast_光伏总加",
    "fcast_竞价空间",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)


def assert_contract(contract: BusinessContract = STRICT34) -> None:
    if contract.forecast_origin_hour != 14:
        raise AssertionError("Cycle 89 v1 forecast origin must remain D-1 14:00")
    if contract.horizon != 34:
        raise AssertionError(f"Cycle 89 v1 horizon must be 34, got {contract.horizon}")
    if contract.target_day_da_as_feature:
        raise AssertionError("target-day DA is forbidden in Cycle 89 strict direct-spread")
    if contract.target_day_actual_as_feature:
        raise AssertionError("target-day actual values are forbidden features")
    if contract.d1_post14_realized_as_feature:
        raise AssertionError("D-1 post-14 realized values are forbidden features")


def latest_complete_label_day(target_day: str, contract: BusinessContract = STRICT34) -> str:
    """Return the latest complete supervised target day visible at origin(D)."""
    assert_contract(contract)
    import pandas as pd

    return (pd.Timestamp(target_day) - pd.Timedelta(days=contract.training_label_lag_days)).strftime("%Y-%m-%d")
