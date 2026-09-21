"""Regression tests for the causal hourly spread information contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.spread_direction_24.spread_contract import (  # noqa: E402
    DA_ALIASES,
    RT_ALIASES,
    SPREAD_COL,
    business_columns,
    first_existing,
    materialize_asof_input,
)
from utils.resolution import HOURLY  # noqa: E402


def _read(path: Path) -> pd.DataFrame:
    for encoding in ("gb18030", "gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def run(data_path: Path, output_root: Path, target_day: str) -> dict:
    raw = business_columns(_read(data_path), HOURLY.business_day_from_timestamp, HOURLY.business_period_from_timestamp)
    da_col = first_existing(raw, DA_ALIASES, "dayahead")
    rt_col = first_existing(raw, RT_ALIASES, "realtime")
    source = raw.copy()
    source[SPREAD_COL] = pd.to_numeric(source[rt_col], errors="coerce") - pd.to_numeric(
        source[da_col], errors="coerce"
    )
    results = {}
    for scheme in ("masked_direct", "safe_mixed_lag"):
        path = output_root / f"{scheme}.parquet"
        audit = materialize_asof_input(
            source,
            target_day,
            da_col,
            rt_col,
            path,
            input_scheme=scheme,
            business_day_fn=HOURLY.business_day_from_timestamp,
            business_period_fn=HOURLY.business_period_from_timestamp,
        )
        view = pd.read_parquet(path)
        checked = business_columns(
            view,
            HOURLY.business_day_from_timestamp,
            HOURLY.business_period_from_timestamp,
        )
        target = checked[checked["_business_day"].astype(str).eq(target_day)]
        if target[da_col].notna().any() or target[rt_col].notna().any() or target[SPREAD_COL].notna().any():
            raise AssertionError(f"{scheme}: target-day labels leaked")
        actual_cols = [c for c in view.columns if c.endswith("实际值")]
        if actual_cols and target[actual_cols].notna().any().any():
            raise AssertionError(f"{scheme}: target-day actual grid leaked")
        d1_actual = checked[
            checked["_business_day"].astype(str).eq(
                (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            )
        ]
        if actual_cols and d1_actual[actual_cols].notna().any().any():
            raise AssertionError(f"{scheme}: partial D-1 actual grid leaked")
        cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        post = view[pd.to_datetime(view["时刻"]) > cutoff]
        if post[rt_col].notna().any():
            raise AssertionError(f"{scheme}: post-cutoff realtime values leaked")
        if scheme == "masked_direct" and post[SPREAD_COL].notna().any():
            raise AssertionError(f"{scheme}: post-cutoff spread values leaked")
        d1 = checked[
            checked["_business_day"].astype(str).eq(
                (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            )
        ]
        if int((d1["价差可见标记"] == 1).sum()) != 14:
            raise AssertionError(f"{scheme}: expected 14 visible D-1 slots")
        if scheme == "safe_mixed_lag" and int((view["价差来源滞后日"] == 2).sum()) != 10:
            raise AssertionError(f"{scheme}: expected 10 safe lag-2 slots")
        if scheme == "safe_mixed_lag" and int(post[SPREAD_COL].notna().sum()) != 10:
            raise AssertionError(f"{scheme}: only the 10 D-1 post-cutoff slots may be safe-filled")
        results[scheme] = audit

    # Information-boundary mutation: changing target truth and D-1 post-cutoff
    # truth must not change either materialized predictor view.
    mutated = source.copy()
    target_mask = mutated["_business_day"].astype(str).eq(target_day)
    cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
    d1_post = (
        mutated["_business_day"].astype(str).eq(
            (pd.Timestamp(target_day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        & pd.to_datetime(mutated["时刻"]).gt(cutoff)
    )
    mutated.loc[target_mask, da_col] = 999999.0
    mutated.loc[target_mask | d1_post, rt_col] = -999999.0
    mutated[SPREAD_COL] = mutated[rt_col] - mutated[da_col]
    for scheme in ("masked_direct", "safe_mixed_lag"):
        original_path = output_root / f"{scheme}.parquet"
        mutated_path = output_root / f"{scheme}_mutated.parquet"
        materialize_asof_input(
            mutated,
            target_day,
            da_col,
            rt_col,
            mutated_path,
            input_scheme=scheme,
            business_day_fn=HOURLY.business_day_from_timestamp,
            business_period_fn=HOURLY.business_period_from_timestamp,
        )
        left = pd.read_parquet(original_path)
        right = pd.read_parquet(mutated_path)
        compare_cols = [c for c in left.columns if c not in {"信息截止时间"}]
        pd.testing.assert_frame_equal(
            left[compare_cols].reset_index(drop=True),
            right[compare_cols].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=0,
        )

    return {"status": "PASS", "target_day": target_day, "schemes": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--output-root", default="outputs/experiments/01_spread_24/invalid_leakage/legacy_cutoff_leakage_spread_direction_24_cutoff14_v3/contract")
    parser.add_argument("--target-day", default="2026-07-01")
    args = parser.parse_args()
    result = run(Path(args.data_path), Path(args.output_root), args.target_day)
    print(pd.Series(result).to_json(force_ascii=False, indent=2, default_handler=str))


if __name__ == "__main__":
    main()
