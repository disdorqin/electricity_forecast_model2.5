"""
Regression tests for adaptive training day selection.

Tests select_complete_training_days() with synthetic ledger data covering:
  Test 1: Formal dress-rehearsal scenario (D-1 partial actual, 30 complete before)
  Test 2: Middle day missing (skip one day, still collect 30)
  Test 3: Model prediction missing (sgdfnet missing 24h)
  Test 4: Cannot collect 30 days within lookback
  Test 5: Dayahead adaptive too (D-1 incomplete, select 30 from D-2 onwards)
  Test 6: Prediction with wrong hour set (missing h24, has h25)
  Test 7: Actual with wrong hour set (missing h1, has h0)
  Test 8: age_days is position-based when window_days_list is explicit
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipelines.ledger_weight import (
    select_complete_training_days,
    DAYAHEAD_MODELS,
    REALTIME_MODELS,
)

REALTIME_MODELS_SET = REALTIME_MODELS  # ("timesfm", "sgdfnet", "timemixer", "rt916")
DAYAHEAD_MODELS_SET = DAYAHEAD_MODELS  # ["lightgbm", "timesfm", "timemixer"]

passed = 0
failed = 0


def _assert(cond: bool, msg: str):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS: {msg}")
    else:
        failed += 1
        print(f"  FAIL: {msg}")


# ===========================================================================
# Helpers: build synthetic ledgers
# ===========================================================================


def _build_prediction_rows(
    task: str,
    days: list[str],
    models: list[str],
    nan_models: list[str] | None = None,
    missing_models: list[str] | None = None,
    partial_hours_models: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Build synthetic prediction ledger rows for given days × models × 24h."""
    rows = []
    for day in days:
        for model in models:
            if missing_models and model in missing_models:
                continue  # skip this model entirely
            n_hours = 24
            if partial_hours_models and model in partial_hours_models:
                n_hours = partial_hours_models[model]
            for h in range(1, n_hours + 1):
                y_pred = 200.0 + h * 5.0
                if nan_models and model in nan_models:
                    y_pred = np.nan
                rows.append({
                    "task": task,
                    "model_name": model,
                    "forecast_date": day,
                    "target_day": day,
                    "business_day": day,
                    "ds": f"{day} {h:02d}:00:00",
                    "hour_business": h,
                    "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                    "y_pred": y_pred,
                    "run_id": "test",
                })
    return pd.DataFrame(rows)


def _build_actual_rows(
    task: str,
    days: list[str],
    n_hours: int = 24,
    nan_hours: int = 0,
) -> pd.DataFrame:
    """Build synthetic actual ledger rows."""
    rows = []
    for day in days:
        for h in range(1, n_hours + 1):
            y_true = 210.0 + h * 3.0
            if nan_hours > 0 and h <= nan_hours:
                y_true = np.nan
            rows.append({
                "task": task,
                "target_day": day,
                "business_day": day,
                "ds": f"{day} {h:02d}:00:00",
                "hour_business": h,
                "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                "y_true": y_true,
            })
    return pd.DataFrame(rows)


def _write_ledger(tmpdir: str, task: str, pred_df: pd.DataFrame, act_df: pd.DataFrame):
    """Write prediction and actual ledger parquet files."""
    pred_dir = Path(tmpdir) / task / "prediction"
    act_dir = Path(tmpdir) / task / "actual"
    pred_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(pred_dir / "prediction_ledger.parquet", index=False)
    act_df.to_parquet(act_dir / "actual_ledger.parquet", index=False)


def _date_range_strings(end_exclusive: str, n: int) -> list[str]:
    """Return n date strings ending at end_exclusive (exclusive), going backwards."""
    D = pd.Timestamp(end_exclusive)
    return [(D - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, n + 1)]


# ===========================================================================
# Test 1: Formal dress-rehearsal scenario
# ===========================================================================


def test1_formal_dress_rehearsal():
    """
    Target: 2026-07-03
    D-1 (2026-07-02): realtime actual only 14/24 hours → skip
    D-2..D-31 (2026-07-01..2026-06-02): all complete → select 30
    """
    print("\n=== Test 1: Formal dress-rehearsal scenario ===")
    tmpdir = tempfile.mkdtemp(prefix="test1_")
    try:
        target = "2026-07-03"
        # D-1 partial actual day
        partial_day = "2026-07-02"
        # 30 complete days: 2026-07-01 back to 2026-06-02
        complete_days = _date_range_strings("2026-07-02", 30)  # D-2..D-31

        # Prediction: all days complete (D-1 + 30 complete)
        all_days = [partial_day] + complete_days
        pred_df = _build_prediction_rows("realtime", all_days, REALTIME_MODELS_SET)
        # Actual: D-1 has 14h, rest have 24h
        act_partial = _build_actual_rows("realtime", [partial_day], n_hours=14)
        act_complete = _build_actual_rows("realtime", complete_days, n_hours=24)
        act_df = pd.concat([act_partial, act_complete], ignore_index=True)

        _write_ledger(tmpdir, "realtime", pred_df, act_df)

        result = select_complete_training_days(
            task="realtime",
            target_date=target,
            ledger_root=Path(tmpdir),
            expected_models=REALTIME_MODELS_SET,
            required_days=30,
            max_lookback_days=90,
        )

        _assert(result["status"] == "PASS", f"status=PASS (got {result['status']})")
        _assert(result["selected_count"] == 30, f"selected_count=30 (got {result['selected_count']})")
        _assert(result["selected_days"][0] == "2026-07-01", f"latest selected=2026-07-01 (got {result['selected_days'][0]})")

        # Check 2026-07-02 is in skipped
        skipped_days = [s["day"] for s in result["skipped_days"]]
        _assert(partial_day in skipped_days, f"2026-07-02 in skipped_days")

        # Check skip reason
        skip_entry = [s for s in result["skipped_days"] if s["day"] == partial_day]
        if skip_entry:
            _assert("actual incomplete" in skip_entry[0]["reason"], f"reason contains 'actual incomplete' (got {skip_entry[0]['reason']})")
            _assert("14/24" in skip_entry[0]["detail"], f"detail contains '14/24' (got {skip_entry[0]['detail']})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 2: Middle day missing
# ===========================================================================


def test2_middle_day_missing():
    """
    Target: 2026-07-03
    D-1..D-31 except 2026-06-20 (actual missing) → skip 2026-06-20, still get 30
    """
    print("\n=== Test 2: Middle day missing ===")
    tmpdir = tempfile.mkdtemp(prefix="test2_")
    try:
        target = "2026-07-03"
        # Generate 32 complete days (D-1..D-32)
        all_days = _date_range_strings(target, 32)
        missing_day = "2026-06-20"

        # Remove missing_day from the list
        complete_days = [d for d in all_days if d != missing_day]

        pred_df = _build_prediction_rows("realtime", all_days, REALTIME_MODELS_SET)
        # Actual: missing_day has 0 rows, others have 24h
        act_df = _build_actual_rows("realtime", complete_days, n_hours=24)

        _write_ledger(tmpdir, "realtime", pred_df, act_df)

        result = select_complete_training_days(
            task="realtime",
            target_date=target,
            ledger_root=Path(tmpdir),
            expected_models=REALTIME_MODELS_SET,
            required_days=30,
            max_lookback_days=90,
        )

        _assert(result["status"] == "PASS", f"status=PASS (got {result['status']})")
        _assert(result["selected_count"] == 30, f"selected_count=30 (got {result['selected_count']})")

        skipped_days = [s["day"] for s in result["skipped_days"]]
        _assert(missing_day in skipped_days, f"2026-06-20 in skipped_days")

        # The 31st and 32nd complete days should NOT be selected (only 30 needed)
        _assert(len(result["selected_days"]) == 30, f"exactly 30 selected")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 3: Model prediction missing
# ===========================================================================


def test3_model_prediction_missing():
    """
    Target: 2026-07-03
    One day (2026-06-25) has sgdfnet prediction missing 24/24 → skip that day
    """
    print("\n=== Test 3: Model prediction missing ===")
    tmpdir = tempfile.mkdtemp(prefix="test3_")
    try:
        target = "2026-07-03"
        all_days = _date_range_strings(target, 32)
        bad_day = "2026-06-25"

        # Build prediction with sgdfnet missing on bad_day
        pred_rows = []
        for day in all_days:
            if day == bad_day:
                # sgdfnet missing entirely for this day
                day_pred = _build_prediction_rows(
                    "realtime", [day], REALTIME_MODELS_SET,
                    missing_models=["sgdfnet"],
                )
            else:
                day_pred = _build_prediction_rows("realtime", [day], REALTIME_MODELS_SET)
            pred_rows.append(day_pred)
        pred_df = pd.concat(pred_rows, ignore_index=True)

        act_df = _build_actual_rows("realtime", all_days, n_hours=24)
        _write_ledger(tmpdir, "realtime", pred_df, act_df)

        result = select_complete_training_days(
            task="realtime",
            target_date=target,
            ledger_root=Path(tmpdir),
            expected_models=REALTIME_MODELS_SET,
            required_days=30,
            max_lookback_days=90,
        )

        _assert(result["status"] == "PASS", f"status=PASS (got {result['status']})")
        _assert(result["selected_count"] == 30, f"selected_count=30 (got {result['selected_count']})")

        skipped_days = [s["day"] for s in result["skipped_days"]]
        _assert(bad_day in skipped_days, f"2026-06-25 in skipped_days")

        # Check reason mentions sgdfnet
        skip_entry = [s for s in result["skipped_days"] if s["day"] == bad_day]
        if skip_entry:
            _assert("sgdfnet" in skip_entry[0]["detail"].lower(), f"reason mentions sgdfnet (got {skip_entry[0]['detail']})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 4: Cannot collect 30 days
# ===========================================================================


def test4_cannot_collect_30():
    """
    Max lookback 40 days, but only 28 complete → FAIL
    """
    print("\n=== Test 4: Cannot collect 30 days ===")
    tmpdir = tempfile.mkdtemp(prefix="test4_")
    try:
        target = "2026-07-03"
        # Only 28 complete days + 12 days with missing actual
        complete_days = _date_range_strings(target, 28)  # D-1..D-28
        incomplete_days = _date_range_strings(
            (pd.Timestamp(target) - pd.Timedelta(days=28)).strftime("%Y-%m-%d"), 12
        )

        pred_df = _build_prediction_rows("realtime", complete_days + incomplete_days, REALTIME_MODELS_SET)
        # Actual: only complete_days have 24h, incomplete_days have 0h
        act_df = _build_actual_rows("realtime", complete_days, n_hours=24)

        _write_ledger(tmpdir, "realtime", pred_df, act_df)

        result = select_complete_training_days(
            task="realtime",
            target_date=target,
            ledger_root=Path(tmpdir),
            expected_models=REALTIME_MODELS_SET,
            required_days=30,
            max_lookback_days=40,
        )

        _assert(result["status"] == "FAIL", f"status=FAIL (got {result['status']})")
        _assert(result["selected_count"] == 28, f"selected_count=28 (got {result['selected_count']})")
        _assert(len(result["errors"]) > 0, f"errors non-empty")
        _assert("collected=28" in result["errors"][0], f"error mentions collected=28 (got {result['errors'][0]})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 5: Dayahead adaptive too
# ===========================================================================


def test5_dayahead_adaptive():
    """
    Confirm dayahead also uses adaptive complete-day selection.
    Target: 2026-07-03
    D-1 (2026-07-02): prediction missing → skip
    D-2..D-31 (2026-07-01..2026-06-02): all complete → select 30
    """
    print("\n=== Test 5: Dayahead adaptive too ===")
    tmpdir = tempfile.mkdtemp(prefix="test5_")
    try:
        target = "2026-07-03"
        # D-1 is incomplete (prediction missing)
        incomplete_day = "2026-07-02"
        # 30 complete days from D-2 onwards
        complete_days = _date_range_strings("2026-07-02", 30)  # D-2..D-31

        # Prediction: only complete_days have data, incomplete_day has nothing
        pred_df = _build_prediction_rows("dayahead", complete_days, DAYAHEAD_MODELS_SET)
        # Actual: all days including incomplete_day have 24h
        all_days = [incomplete_day] + complete_days
        act_df = _build_actual_rows("dayahead", all_days, n_hours=24)
        _write_ledger(tmpdir, "dayahead", pred_df, act_df)

        # select_complete_training_days should work for dayahead
        result = select_complete_training_days(
            task="dayahead",
            target_date=target,
            ledger_root=Path(tmpdir),
            expected_models=DAYAHEAD_MODELS_SET,
            required_days=30,
            max_lookback_days=90,
        )

        _assert(result["status"] == "PASS", f"dayahead selection PASS (got {result['status']})")
        _assert(result["selected_count"] == 30, f"dayahead selected_count=30 (got {result['selected_count']})")
        _assert(result["selected_days"][0] == "2026-07-01", f"latest selected=2026-07-01 (got {result['selected_days'][0]})")

        # Check 2026-07-02 is in skipped
        skipped_days = [s["day"] for s in result["skipped_days"]]
        _assert(incomplete_day in skipped_days, f"2026-07-02 in skipped_days")

        # Verify run_ledger_weight source code uses select_complete_training_days for dayahead
        from pipelines.ledger_weight import run_ledger_weight
        import inspect
        src = inspect.getsource(run_ledger_weight)
        _assert(
            "select_complete_training_days" in src,
            "run_ledger_weight calls select_complete_training_days"
        )
        # Check that there is no "fixed_contiguous" method reference
        _assert(
            "fixed_contiguous" not in src,
            "run_ledger_weight does NOT reference fixed_contiguous"
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 6: Prediction with wrong hour set (24 rows but missing h24, has h25)
# ===========================================================================


def test6_prediction_wrong_hour_set():
    """
    A day has 24 prediction rows for a model, but hour_business = {1..23, 25}
    instead of {1..24}. Must be skipped with detail mentioning missing_hours
    and extra_hours.
    """
    print("\n=== Test 6: Prediction wrong hour set ===")
    tmpdir = tempfile.mkdtemp(prefix="test6_")
    try:
        target = "2026-07-03"
        all_days = _date_range_strings(target, 32)
        bad_day = all_days[0]  # D-1

        # Build normal predictions for all days
        pred_frames = []
        for day in all_days:
            if day == bad_day:
                # For sgdfnet on bad_day: hours {1..23, 25} — 24 rows but wrong set
                bad_rows = []
                for h in list(range(1, 24)) + [25]:
                    bad_rows.append({
                        "task": "realtime",
                        "model_name": "sgdfnet",
                        "forecast_date": day,
                        "target_day": day,
                        "business_day": day,
                        "ds": f"{day} {h:02d}:00:00",
                        "hour_business": h,
                        "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                        "y_pred": 200.0,
                        "run_id": "test",
                    })
                pred_frames.append(pd.DataFrame(bad_rows))
                # Other models normal on bad_day
                for model in ["timesfm", "timemixer", "rt916"]:
                    for h in range(1, 25):
                        pred_frames.append(pd.DataFrame([{
                            "task": "realtime",
                            "model_name": model,
                            "forecast_date": day,
                            "target_day": day,
                            "business_day": day,
                            "ds": f"{day} {h:02d}:00:00",
                            "hour_business": h,
                            "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                            "y_pred": 200.0,
                            "run_id": "test",
                        }]))
            else:
                day_pred = _build_prediction_rows("realtime", [day], REALTIME_MODELS_SET)
                pred_frames.append(day_pred)
        pred_df = pd.concat(pred_frames, ignore_index=True)
        act_df = _build_actual_rows("realtime", all_days, n_hours=24)
        _write_ledger(tmpdir, "realtime", pred_df, act_df)

        result = select_complete_training_days(
            task="realtime",
            target_date=target,
            ledger_root=Path(tmpdir),
            expected_models=REALTIME_MODELS_SET,
            required_days=30,
            max_lookback_days=90,
        )

        _assert(result["status"] == "PASS", f"status=PASS (got {result['status']})")
        _assert(result["selected_count"] == 30, f"selected_count=30 (got {result['selected_count']})")

        skipped_days = [s["day"] for s in result["skipped_days"]]
        _assert(bad_day in skipped_days, f"D-1 ({bad_day}) in skipped_days")

        skip_entry = [s for s in result["skipped_days"] if s["day"] == bad_day]
        if skip_entry:
            detail = skip_entry[0]["detail"]
            _assert("missing_hours" in detail, f"detail mentions missing_hours (got {detail})")
            _assert("extra_hours" in detail, f"detail mentions extra_hours (got {detail})")
            _assert("24" in detail, f"detail mentions missing hour 24 (got {detail})")
            _assert("25" in detail, f"detail mentions extra hour 25 (got {detail})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 7: Actual with wrong hour set (24 rows but abnormal hours)
# ===========================================================================


def test7_actual_wrong_hour_set():
    """
    A day has 24 actual rows but hour_business = {0, 2, 3, ..., 24} (missing 1,
    has 0). Must be skipped.
    """
    print("\n=== Test 7: Actual wrong hour set ===")
    tmpdir = tempfile.mkdtemp(prefix="test7_")
    try:
        target = "2026-07-03"
        all_days = _date_range_strings(target, 32)
        bad_day = all_days[0]  # D-1

        pred_df = _build_prediction_rows("realtime", all_days, REALTIME_MODELS_SET)

        # Build actual: bad_day has hours {0, 2..24} (24 rows but wrong set)
        act_rows = []
        for day in all_days:
            if day == bad_day:
                hours = [0] + list(range(2, 25))  # 0,2,3,...,24 = 24 values
            else:
                hours = list(range(1, 25))
            for h in hours:
                act_rows.append({
                    "task": "realtime",
                    "target_day": day,
                    "business_day": day,
                    "ds": f"{day} {h:02d}:00:00",
                    "hour_business": h,
                    "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
                    "y_true": 210.0,
                })
        act_df = pd.DataFrame(act_rows)
        _write_ledger(tmpdir, "realtime", pred_df, act_df)

        result = select_complete_training_days(
            task="realtime",
            target_date=target,
            ledger_root=Path(tmpdir),
            expected_models=REALTIME_MODELS_SET,
            required_days=30,
            max_lookback_days=90,
        )

        _assert(result["status"] == "PASS", f"status=PASS (got {result['status']})")
        _assert(result["selected_count"] == 30, f"selected_count=30 (got {result['selected_count']})")

        skipped_days = [s["day"] for s in result["skipped_days"]]
        _assert(bad_day in skipped_days, f"D-1 ({bad_day}) in skipped_days")

        skip_entry = [s for s in result["skipped_days"] if s["day"] == bad_day]
        if skip_entry:
            detail = skip_entry[0]["detail"]
            _assert("missing_hours" in detail, f"detail mentions missing_hours (got {detail})")
            _assert("1" in detail, f"detail mentions missing hour 1 (got {detail})")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Test 8: age_days is position-based when window_days_list is explicit
# ===========================================================================


def test8_age_days_position_based():
    """
    When D-1 is skipped and the latest selected day is D-2, the training table
    should have age_days=1 for D-2 (position-based, not calendar-based which
    would give age_days=2).
    """
    print("\n=== Test 8: age_days position-based for explicit window_days_list ===")
    from pipelines.prediction_ledger import build_ledger_training_table

    target = "2026-07-03"
    # Simulate: D-1 skipped, selected = [D-2, D-3, ..., D-31]
    selected_days = _date_range_strings(
        (pd.Timestamp(target) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), 30
    )
    # selected_days[0] = 2026-07-01 (D-2), selected_days[29] = 2026-06-02 (D-31)

    # Build minimal ledgers
    pred_df = _build_prediction_rows("realtime", selected_days, REALTIME_MODELS_SET)
    act_df = _build_actual_rows("realtime", selected_days, n_hours=24)

    # Build training table with explicit window_days_list
    training = build_ledger_training_table(
        prediction_ledger=pred_df,
        actual_ledger=act_df,
        target_day=target,
        window_days=30,
        window_days_list=selected_days,
    )

    # The latest selected day (selected_days[0] = 2026-07-01) should have age_days=1
    latest_day = selected_days[0]
    latest_rows = training[training["target_day"] == latest_day]
    if not latest_rows.empty:
        age = latest_rows["age_days"].iloc[0]
        _assert(age == 1, f"latest selected day ({latest_day}) age_days=1 (got {age})")
    else:
        _assert(False, f"latest selected day ({latest_day}) not found in training table")

    # The oldest selected day (selected_days[29]) should have age_days=30
    oldest_day = selected_days[29]
    oldest_rows = training[training["target_day"] == oldest_day]
    if not oldest_rows.empty:
        age = oldest_rows["age_days"].iloc[0]
        _assert(age == 30, f"oldest selected day ({oldest_day}) age_days=30 (got {age})")
    else:
        _assert(False, f"oldest selected day ({oldest_day}) not found in training table")

    # day_gate for latest should be the maximum (most recent = highest weight)
    if not latest_rows.empty and not oldest_rows.empty:
        gate_latest = latest_rows["day_gate"].iloc[0]
        gate_oldest = oldest_rows["day_gate"].iloc[0]
        _assert(gate_latest > gate_oldest, f"latest day_gate ({gate_latest:.3f}) > oldest day_gate ({gate_oldest:.3f})")

    # Verify old behavior (no explicit list) uses calendar-based age_days
    training_old = build_ledger_training_table(
        prediction_ledger=pred_df,
        actual_ledger=act_df,
        target_day=target,
        window_days=30,
        # No window_days_list → old behavior
    )
    if not training_old.empty:
        # With old behavior, D-2 should have age_days=2 (calendar difference)
        d2_rows = training_old[training_old["target_day"] == latest_day]
        if not d2_rows.empty:
            age_old = d2_rows["age_days"].iloc[0]
            _assert(age_old == 2, f"old behavior: D-2 age_days=2 (got {age_old})")


# ===========================================================================
# Runner
# ===========================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("CHECK_ADAPTIVE_REALTIME_WEIGHT_DAYS")
    print("=" * 60)

    test1_formal_dress_rehearsal()
    test2_middle_day_missing()
    test3_model_prediction_missing()
    test4_cannot_collect_30()
    test5_dayahead_adaptive()
    test6_prediction_wrong_hour_set()
    test7_actual_wrong_hour_set()
    test8_age_days_position_based()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"RESULT: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
