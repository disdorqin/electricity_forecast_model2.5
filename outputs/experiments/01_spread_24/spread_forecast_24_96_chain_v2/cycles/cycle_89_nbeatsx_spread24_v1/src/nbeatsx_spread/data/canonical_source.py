from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CANONICAL_TO_SOURCE = {
    "fcast_直调负荷": "直调负荷预测值",
    "fcast_联络线受电负荷": "联络线受电负荷预测值",
    "fcast_风电总加": "风电总加预测值",
    "fcast_光伏总加": "光伏总加预测值",
    "fcast_竞价空间": "竞价空间预测值",
}


@dataclass
class CanonicalHourlySource:
    """Read the 24-point canonical table without changing its values."""

    frame: pd.DataFrame
    path: Path | None = None

    @classmethod
    def from_csv(cls, path: str | Path) -> "CanonicalHourlySource":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "gbk", "utf-8"):
            try:
                frame = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
        else:
            raise last_error or ValueError(f"cannot decode {path}")
        return cls.from_frame(frame, path=path)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, path: str | Path | None = None) -> "CanonicalHourlySource":
        out = frame.copy()
        if "时刻" not in out.columns:
            if isinstance(out.index, pd.DatetimeIndex):
                out = out.reset_index(names="时刻")
            else:
                raise ValueError("canonical source must contain 时刻")
        out["时刻"] = pd.to_datetime(out["时刻"], errors="coerce")
        out = out.dropna(subset=["时刻"]).sort_values("时刻")
        if out["时刻"].duplicated().any():
            raise ValueError("canonical source contains duplicate timestamps")
        missing = [v for v in ("日前电价", "实时电价", *CANONICAL_TO_SOURCE.values()) if v not in out]
        if missing:
            raise ValueError(f"canonical source missing columns: {missing}")
        out = out.set_index("时刻")
        return cls(out, Path(path) if path else None)

    @property
    def frame_by_timestamp(self) -> pd.DataFrame:
        return self.frame

    def business_days(self) -> list[str]:
        ts = self.frame.index.to_series()
        days = ts.map(lambda x: (x - pd.Timedelta(days=1)).strftime("%Y-%m-%d") if x.hour == 0 else x.strftime("%Y-%m-%d"))
        return sorted(days.unique().tolist())

    def source_columns(self, feature_names: tuple[str, ...]) -> list[str]:
        return [CANONICAL_TO_SOURCE.get(name, name) for name in feature_names]
