"""
Temporal resolution abstraction for the forecast pipeline.

单一事实来源：24 点（hourly）与 96 点（15min）两种分辨率的所有参数。
任何地方都不应再硬编码 24 / 96 / 8 / 32 —— 一律从 Resolution 取。

关键语义：
  * business_period (1..slots_per_day)：区间末标注。p=slots_per_day 的 ds = D+1 00:00。
  * hourly 模式下 business_period ≡ hour_business（外部契约零变化）。
  * period 分段沿用三段：24点 (1_8, 9_16, 17_24) / 96点 (1_32, 33_64, 65_96)。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Resolution:
    label: str                    # "hourly" | "15min"
    slots_per_day: int            # 24 | 96
    slots_per_period: int         # 8  | 32
    period_names: tuple[str, ...] # ("1_8","9_16","17_24") | ("1_32","33_64","65_96")
    slot_column: str              # "hour_business" | "business_period"
    freq: str                     # "h" | "15min"
    minutes_per_slot: int         # 60 | 15
    # period 边界（段起点索引 0..n，供 if/elif 复用）
    period_bounds: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        # 从 period_names 推导边界，如 ("1_8","9_16","17_24") → (0,8,16,24)
        bounds = [0]
        for name in self.period_names:
            _, end = name.split("_")
            bounds.append(int(end))
        object.__setattr__(self, "period_bounds", tuple(bounds))

    @property
    def period_count(self) -> int:
        return len(self.period_names)

    def infer_period(self, slot: int) -> str:
        """slot(1..slots_per_day) → period 标签，如 5 → '1_8' / '1_32'。"""
        if slot < 1 or slot > self.slots_per_day:
            raise ValueError(
                f"{self.label}: slot must be 1..{self.slots_per_day}, got {slot}"
            )
        bounds = self.period_bounds
        for i in range(self.period_count):
            if bounds[i] < slot <= bounds[i + 1]:
                return self.period_names[i]
        raise ValueError(f"cannot infer period for slot {slot}")

    # --- 时间戳 ↔ 业务槽 换算 ---
    def business_period_from_timestamp(self, ts) -> int:
        """wall-clock → business_period (1..slots_per_day)。

        slots_per_day 的周期末 = 次日 00:00（如 h24 / p96）。其余按
        区间末标注：period p 覆盖 (p-1)×Δ .. p×Δ，标注时刻 p×Δ。
        即 business_period = ceil(minute_of_day / minutes_per_slot)。
        """
        import pandas as pd

        ts = pd.Timestamp(ts)
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            return self.slots_per_day
        minute_of_day = ts.hour * 60 + ts.minute
        # 区间末：ceil(minute/Δ)。01:00 → ceil(60/15)=4；00:15 → ceil(15/15)=1
        slot_idx = -(-minute_of_day // self.minutes_per_slot)  # 整数向上取整
        if slot_idx < 1:
            slot_idx = 1
        return int(slot_idx)

    def business_day_from_timestamp(self, ts) -> str:
        """wall-clock → business_day（00:00:00 归前一业务日，其余取日期部分）。"""
        import pandas as pd

        ts = pd.Timestamp(ts)
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            return (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return ts.strftime("%Y-%m-%d")

    def timestamp_from_business(self, business_day: str, slot: int):
        """(business_day, business_period) → wall-clock timestamp。

        slot == slots_per_day → D+1 00:00:00，否则 D + slot×minutes_per_slot。
        """
        import pandas as pd

        day = pd.Timestamp(business_day)
        if slot == self.slots_per_day:
            return day + pd.Timedelta(days=1)
        return day + pd.Timedelta(minutes=slot * self.minutes_per_slot)


HOURLY = Resolution(
    label="hourly",
    slots_per_day=24,
    slots_per_period=8,
    period_names=("1_8", "9_16", "17_24"),
    slot_column="hour_business",
    freq="h",
    minutes_per_slot=60,
)

QUARTER = Resolution(
    label="15min",
    slots_per_day=96,
    slots_per_period=32,
    period_names=("1_32", "33_64", "65_96"),
    slot_column="business_period",
    freq="15min",
    minutes_per_slot=15,
)

_RESOLUTIONS = {"hourly": HOURLY, "15min": QUARTER}


def resolve_resolution(label: str) -> Resolution:
    """CLI 标签 → Resolution。hourly/15min（默认 hourly）。"""
    return _RESOLUTIONS.get(label, HOURLY)
