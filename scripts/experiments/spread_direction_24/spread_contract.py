"""Causal information contract for the hourly spread experiment.

The experiment predicts the next business day at ``D-1 14:00``.  This module
keeps the information boundary independent from any particular model runner:

* target-day DA/RT/spread labels are never available to a predictor;
* D-1 realtime/spread is visible only through p14;
* target-day forecast grid features are available;
* training rows before D use actual grid values mapped into the forecast
  feature names, while the target-day inference row uses forecast values;
* the safe mixed-lag variant replaces only unavailable D-1 spread slots with
  D-2 values and records the source age explicitly.

The returned frame deliberately keeps the project's original Chinese column
names so existing model adapters can consume it without changing production
code.  This is an experiment-only adapter; it never mutates the source file.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DA_ALIASES = ("日前电价", "日前出清电价", "日前出清价格")
RT_ALIASES = ("实时电价", "实时出清电价", "实时出清价格")


@dataclass(frozen=True)
class ContractConfig:
    target_day: str
    cutoff_hour: int = 14
    input_scheme: str = "masked_direct"

    @property
    def cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.target_day) - pd.Timedelta(days=1) + pd.Timedelta(
            hours=self.cutoff_hour
        )


def first_existing(frame: pd.DataFrame, aliases: Iterable[str], label: str) -> str:
    for name in aliases:
        if name in frame.columns:
            return name
    raise ValueError(f"missing {label} column; tried={list(aliases)}")


def business_columns(frame: pd.DataFrame, business_day_fn, business_period_fn) -> pd.DataFrame:
    out = frame.copy()
    if "时刻" not in out.columns:
        raise ValueError("spread source must contain 时刻")
    out["时刻"] = pd.to_datetime(out["时刻"], errors="coerce")
    out = out.dropna(subset=["时刻"]).sort_values("时刻").reset_index(drop=True)
    # FeatureStore spread bases already contain these keys.  Reusing them is
    # both faster and safer than re-deriving business-time semantics for every
    # model/day view.
    if "_business_day" not in out.columns or "_business_period" not in out.columns:
        out["_business_day"] = out["时刻"].map(business_day_fn)
        out["_business_period"] = out["时刻"].map(business_period_fn).astype(int)
    else:
        out["_business_period"] = pd.to_numeric(out["_business_period"], errors="coerce").astype(int)
    if out["时刻"].duplicated().any():
        raise ValueError("spread source contains duplicate timestamps")
    return out


def _paired_grid_columns(frame: pd.DataFrame) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for actual_col in frame.columns:
        if not actual_col.endswith("实际值"):
            continue
        forecast_col = actual_col[:-3] + "预测值"
        if forecast_col in frame.columns:
            pairs.append((actual_col, forecast_col))
    if not pairs:
        raise ValueError("no actual/forecast grid feature pairs found")
    return pairs


def _target_day_mask(raw: pd.DataFrame, target_day: str) -> pd.Series:
    return raw["_business_day"].astype(str).eq(str(target_day))


def _safe_mixed_spread(
    raw: pd.DataFrame,
    spread: pd.Series,
    target_day: str,
    cutoff: pd.Timestamp,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return safe spread, source lag and visibility mask.

    For the only unavailable block used by the direct model (D-1 p15-p24),
    use the same business periods from D-2.  No value after ``cutoff`` is
    copied from D-1.  Source lag is 1 for observed D-1 values, 2 for the
    replacement values, and NaN elsewhere.
    """

    # Start from the observable history only.  In particular, do not carry
    # any raw target/future spread values through to the model view.
    out = spread.mask(~raw["时刻"].le(cutoff))
    source_lag = pd.Series(np.nan, index=raw.index, dtype=float)
    visible = pd.Series(0, index=raw.index, dtype=int)
    available = raw["时刻"].le(cutoff)
    out.loc[available & spread.notna()] = spread.loc[available & spread.notna()]
    source_lag.loc[available & spread.notna()] = 1.0
    visible.loc[available & spread.notna()] = 1

    d1 = raw["_business_day"].astype(str).eq(str(pd.Timestamp(target_day) - pd.Timedelta(days=1)).split(" ")[0])
    # The string conversion above is intentionally replaced by a normalized
    # date expression below; keeping the actual comparison date-only avoids
    # depending on the source's business-day return type.
    d1 = raw["_business_day"].astype(str).eq(
        (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )
    unavailable = d1 & raw["时刻"].gt(cutoff)
    history = raw.set_index("时刻")
    for idx in raw.index[unavailable]:
        source_ts = pd.Timestamp(raw.at[idx, "时刻"]) - pd.Timedelta(days=1)
        value = history[SPREAD_COL].get(source_ts, np.nan)
        if pd.notna(value):
            out.at[idx] = float(value)
            source_lag.at[idx] = 2.0
    return out, source_lag, visible


SPREAD_COL = "价差"


def materialize_asof_input(
    raw: pd.DataFrame,
    target_day: str,
    da_col: str,
    rt_col: str,
    output_path: Path,
    *,
    input_scheme: str = "masked_direct",
    cutoff_hour: int = 14,
    business_day_fn=None,
    business_period_fn=None,
) -> dict:
    """Materialize an experiment-only, causal input view and audit metadata."""

    if input_scheme not in {"masked_direct", "safe_mixed_lag"}:
        raise ValueError(f"unsupported input_scheme={input_scheme}")
    if business_day_fn is None or business_period_fn is None:
        raise ValueError("business-day and business-period functions are required")

    work = business_columns(raw, business_day_fn, business_period_fn)
    if SPREAD_COL not in work:
        work[SPREAD_COL] = pd.to_numeric(work[rt_col], errors="coerce") - pd.to_numeric(
            work[da_col], errors="coerce"
        )
    target_mask = _target_day_mask(work, target_day)
    cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour)
    after_cutoff = work["时刻"].gt(cutoff)
    future = work["_business_day"].astype(str).gt(str(target_day))

    out = work.drop(columns=["_business_day", "_business_period"], errors="ignore").copy()

    # Actual-to-forecast training contract: historical actual grid values are
    # mapped into the forecast feature names.  Target-day inference remains
    # forecast-sourced.  Actual-side columns are retained only for fully
    # completed history (D-2 and earlier); D-1 partial actual grid values are
    # deliberately not used by this first causal design.
    pairs = _paired_grid_columns(work)
    for actual_col, forecast_col in pairs:
        historical = work["_business_day"].astype(str).lt(
            (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        out.loc[historical, forecast_col] = work.loc[historical, actual_col]
        out.loc[~historical, actual_col] = np.nan

    spread = pd.to_numeric(work[SPREAD_COL], errors="coerce")
    visible_mask = (~after_cutoff & spread.notna()).astype(int)
    source_lag = pd.Series(np.nan, index=work.index, dtype=float)
    if input_scheme == "safe_mixed_lag":
        spread, source_lag, visible_mask = _safe_mixed_spread(
            work.assign(**{SPREAD_COL: spread}), spread, target_day, cutoff
        )

    # Price labels and target-day DA are never model inputs.  Historical DA
    # remains available because D-1 DA is known in the business setting and
    # older values are historical context.
    out[SPREAD_COL] = spread.mask(target_mask | (after_cutoff & (input_scheme == "masked_direct")))
    out[rt_col] = pd.to_numeric(work[rt_col], errors="coerce")
    out.loc[after_cutoff, rt_col] = np.nan
    out.loc[target_mask, rt_col] = np.nan
    out[da_col] = pd.to_numeric(work[da_col], errors="coerce")
    out.loc[target_mask | future, da_col] = np.nan

    # D-day forecast features are allowed; forecast features after D are not.
    forecast_cols = [c for c in out.columns if c.endswith("预测值")]
    if forecast_cols:
        out.loc[future, forecast_cols] = np.nan

    # Persist explicit availability channels for adapters that consume them.
    out["价差可见标记"] = visible_mask.to_numpy(dtype=int)
    out["价差来源滞后日"] = source_lag.to_numpy(dtype=float)
    out["信息截止时间"] = str(cutoff)

    target_rows = target_mask
    audit = {
        "target_day": target_day,
        "input_scheme": input_scheme,
        "forecast_origin": "D-1 14:00",
        "cutoff": str(cutoff),
        "target_rows": int(target_rows.sum()),
        "target_dayahead_non_null": int(out.loc[target_rows, da_col].notna().sum()),
        "target_realtime_non_null": int(out.loc[target_rows, rt_col].notna().sum()),
        "target_spread_non_null": int(out.loc[target_rows, SPREAD_COL].notna().sum()),
        "target_actual_grid_non_null": int(out.loc[target_rows, [a for a, _ in pairs]].notna().sum().sum()),
        "d1_actual_grid_non_null": int(
            out.loc[
                work["_business_day"].astype(str).eq(
                    (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                ),
                [a for a, _ in pairs],
            ].notna().sum().sum()
        ),
        "inference_forecast_grid_columns": [f for _, f in pairs],
        "actual_grid_columns_hidden": [a for a, _ in pairs],
        "post_cutoff_realtime_non_null": int(out.loc[after_cutoff, rt_col].notna().sum()),
        "post_cutoff_direct_spread_non_null": int(
            out.loc[after_cutoff, SPREAD_COL].notna().sum()
        )
        if input_scheme == "masked_direct"
        else int(out.loc[after_cutoff & ~target_rows, SPREAD_COL].notna().sum()),
        "d1_visible_spread_slots": int(
            ((work["_business_day"].astype(str) == (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
             & work["时刻"].le(cutoff) & spread.notna()).sum()
        ),
        "d1_mixed_lag2_slots": int((source_lag == 2).sum()),
        "future_forecast_non_null": int(out.loc[future, forecast_cols].notna().sum().sum())
        if forecast_cols
        else 0,
        "rows": int(len(out)),
    }
    expected = {
        "target_rows": 24,
        "target_dayahead_non_null": 0,
        "target_realtime_non_null": 0,
        "target_spread_non_null": 0,
        "target_actual_grid_non_null": 0,
        "d1_actual_grid_non_null": 0,
        "post_cutoff_realtime_non_null": 0,
        "future_forecast_non_null": 0,
    }
    bad = {k: (audit[k], v) for k, v in expected.items() if audit[k] != v}
    if bad:
        raise RuntimeError(f"{target_day}: masked spread contract failed: {bad}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, output_path)
    return audit
