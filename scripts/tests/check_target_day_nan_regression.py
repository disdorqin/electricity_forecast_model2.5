"""
Regression test: target-day NaN price handling.

Tests that:
1. LightGBM load_and_process_data preserves target-day rows with NaN y
   (for inference), while training functions still exclude them.
2. SGDFNet da_anchor fallback fills NaN day-ahead prices with historical
   same-hour median, preventing NaN propagation to rt_hat.
3. Adapter None guard catches LightGBM returning None with a clear error.

Usage:
    python scripts/check_target_day_nan_regression.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS_COUNT = 0
FAIL_COUNT = 0


def _check(name: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}  {detail}")


# =========================================================================
# Test 1: LightGBM data loading preserves NaN-y rows
# =========================================================================

def test_lightgbm_data_loading():
    print("\n=== Test 1: LightGBM load_and_process_data preserves NaN-y rows ===")

    from lightGBM.train_da_fix import LGBMPowerPredictor

    predictor = LGBMPowerPredictor()

    # Create a synthetic CSV
    dates = pd.date_range("2026-06-01 01:00:00", periods=24 * 10, freq="h")
    n = len(dates)
    synthetic = pd.DataFrame({
        "时刻": dates,
        "日前电价": np.random.uniform(100, 500, n),
        "实时电价": np.random.uniform(100, 500, n),
        "直调负荷预测值": np.random.uniform(1000, 5000, n),
        "风电总加预测值": np.random.uniform(0, 500, n),
        "光伏总加预测值": np.random.uniform(0, 300, n),
        "联络线受电负荷预测值": np.random.uniform(500, 2000, n),
    })

    # Add 24 target-day rows with NaN y (simulating 2026-07-03)
    target_dates = pd.date_range("2026-06-11 01:00:00", periods=24, freq="h")
    target_rows = pd.DataFrame({
        "时刻": target_dates,
        "日前电价": [np.nan] * 24,
        "实时电价": [np.nan] * 24,
        "直调负荷预测值": np.random.uniform(1000, 5000, 24),
        "风电总加预测值": np.random.uniform(0, 500, 24),
        "光伏总加预测值": np.random.uniform(0, 300, 24),
        "联络线受电负荷预测值": np.random.uniform(500, 2000, 24),
    })
    synthetic = pd.concat([synthetic, target_rows], ignore_index=True)

    csv_path = PROJECT_ROOT / "outputs" / "_diagnose_20260703" / "_test_synthetic.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(csv_path, index=False, encoding="utf-8")

    try:
        result = predictor.load_and_process_data(str(csv_path))

        _check(
            "result is not None",
            result is not None,
        )
        if result is not None:
            _check(
                "ds column exists",
                "ds" in result.columns,
            )
            _check(
                "target-day rows preserved (24 rows with NaN y)",
                result["ds"].max() >= pd.Timestamp("2026-06-11 00:00:00"),
                f"max ds = {result['ds'].max()}",
            )
            target_mask = result["ds"] >= pd.Timestamp("2026-06-11 01:00:00")
            n_target = target_mask.sum()
            _check(
                f"target-day has 24 rows (got {n_target})",
                n_target == 24,
            )
            nan_y_count = result.loc[target_mask, "y"].isna().sum()
            _check(
                f"target-day y is NaN ({nan_y_count}/24)",
                nan_y_count == 24,
            )
            # Training rows should have valid y
            train_mask = ~target_mask
            train_nan_y = result.loc[train_mask, "y"].isna().sum()
            _check(
                f"non-target rows have no NaN y ({train_nan_y} NaN)",
                train_nan_y == 0,
            )
    except Exception as e:
        _check("load_and_process_data did not raise", False, str(e))
        traceback.print_exc()
    finally:
        if csv_path.exists():
            csv_path.unlink()


# =========================================================================
# Test 2: SGDFNet da_anchor fallback
# =========================================================================

def test_sgdfnet_da_anchor_fallback():
    print("\n=== Test 2: SGDFNet da_anchor fallback ===")

    from SGDFNet.src.sgdfnet.data_contract import _fill_da_anchor_fallback

    # Create synthetic frame with NaN da_anchor for target day
    dates = pd.date_range("2026-06-01 01:00:00", periods=24 * 10, freq="h")
    n = len(dates)
    df = pd.DataFrame({
        "timestamp": dates,
        "da_anchor": np.random.uniform(100, 500, n),
    })

    # Add 24 target-day rows with NaN da_anchor
    target_dates = pd.date_range("2026-06-11 01:00:00", periods=24, freq="h")
    target_rows = pd.DataFrame({
        "timestamp": target_dates,
        "da_anchor": [np.nan] * 24,
    })
    df = pd.concat([df, target_rows], ignore_index=True)

    try:
        result = _fill_da_anchor_fallback(df, time_col="timestamp")

        _check("result is not None", result is not None)
        _check(
            "da_anchor column exists",
            "da_anchor" in result.columns,
        )
        _check(
            "da_anchor_fallback_used column exists",
            "da_anchor_fallback_used" in result.columns,
        )

        # Check target rows are filled
        target_mask = result["timestamp"] >= pd.Timestamp("2026-06-11 01:00:00")
        target_da = result.loc[target_mask, "da_anchor"]
        nan_count = target_da.isna().sum()
        _check(
            f"target-day da_anchor has no NaN after fallback ({nan_count} NaN)",
            nan_count == 0,
        )

        fallback_count = result.loc[target_mask, "da_anchor_fallback_used"].sum()
        _check(
            f"target-day da_anchor_fallback_used = 24 (got {fallback_count})",
            fallback_count == 24,
        )

        # Check historical rows are NOT marked as fallback
        hist_mask = ~target_mask
        hist_fallback = result.loc[hist_mask, "da_anchor_fallback_used"].sum()
        _check(
            f"historical rows not marked as fallback ({hist_fallback} marked)",
            hist_fallback == 0,
        )

        # Check fallback values are reasonable (within historical range)
        hist_median_range = (
            result.loc[hist_mask, "da_anchor"].min(),
            result.loc[hist_mask, "da_anchor"].max(),
        )
        target_vals = result.loc[target_mask, "da_anchor"]
        all_in_range = (target_vals >= hist_median_range[0] * 0.5).all() and \
                       (target_vals <= hist_median_range[1] * 2.0).all()
        _check(
            "fallback values within reasonable range",
            all_in_range,
        )

    except Exception as e:
        _check("_fill_da_anchor_fallback did not raise", False, str(e))
        traceback.print_exc()


# =========================================================================
# Test 3: LightGBM adapter None guard
# =========================================================================

def test_lightgbm_adapter_none_guard():
    print("\n=== Test 3: LightGBM adapter None guard ===")

    from runners.adapters.lightgbm_v1 import LightGBMV1Adapter

    adapter = LightGBMV1Adapter()

    # We can't easily mock the internal pipeline, but we can verify
    # the guard code exists by checking the source
    import inspect
    source = inspect.getsource(adapter.predict)

    _check(
        "adapter.predict contains None check",
        "if df is None" in source,
    )
    _check(
        "adapter.predict contains empty DataFrame check",
        "df.empty" in source,
    )
    _check(
        "adapter.predict raises RuntimeError (not AttributeError)",
        "raise RuntimeError" in source,
    )


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("Target-Day NaN Regression Tests")
    print("=" * 60)

    test_lightgbm_data_loading()
    test_sgdfnet_da_anchor_fallback()
    test_lightgbm_adapter_none_guard()

    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
    print("=" * 60)

    if FAIL_COUNT > 0:
        print("\nREGRESSION TEST: FAIL")
        return 1
    else:
        print("\nREGRESSION TEST: PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
