#!/usr/bin/env python
"""TimeMixer 时间对齐检查工具。

验证 TimeMixer 输出是否符合营业日对齐要求：
  - 24 行 (hour_business 1..24)
  - hour 24 对应的是 D+1 00:00，不是 D 00:00
  - 不存在 hour 0
  - 不存在 D 00:00 的时间戳

可选：与其他模型对比 business_day / hour_business 集合是否一致。

用法:
    python scripts/check_timemixer_alignment.py outputs/runs/2026-02-24
    python scripts/check_timemixer_alignment.py --date 2026-02-24 --runs-root outputs/smoke/runs --compare
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] Cannot read {path.name}: {e}")
        return None


def _check_single_file(df: pd.DataFrame, label: str) -> list[str]:
    """Check alignment for a single DataFrame. Returns list of issues."""
    issues: list[str] = []

    if df is None:
        issues.append(f"[{label}] Cannot read file")
        return issues

    # 1) Row count
    if len(df) != 24:
        issues.append(f"[{label}] expected 24 rows, got {len(df)}")

    # 2) hour_business check
    if "hour_business" in df.columns:
        hours = sorted(df["hour_business"].dropna().astype(int).unique())
        expected = list(range(1, 25))
        if hours != expected:
            missing = set(expected) - set(hours)
            extra = set(hours) - set(expected)
            if missing:
                issues.append(f"[{label}] missing hours: {sorted(missing)}")
            if extra:
                issues.append(f"[{label}] extra hours: {sorted(extra)}")

        # Check no hour 0
        if 0 in df["hour_business"].values:
            issues.append(f"[{label}] hour_business=0 found!")

    # 3) Strict timestamp alignment: business_day + hour_business must match ds
    if "ds" in df.columns and "business_day" in df.columns and "hour_business" in df.columns:
        for idx, row in df.iterrows():
            ds = pd.Timestamp(row["ds"])
            bd = pd.Timestamp(row["business_day"]).normalize()
            hb = int(row["hour_business"])
            if hb == 24:
                expected_ds = bd + pd.Timedelta(days=1)
            else:
                expected_ds = bd + pd.Timedelta(hours=hb)
            if ds != expected_ds:
                issues.append(
                    f"[{label}] timestamp mismatch row={idx}: "
                    f"hour_business={hb}, ds={ds}, expected={expected_ds}"
                )

    # 4) hour 24 must be D+1 00:00 (strict check)
    if "hour_business" in df.columns and "ds" in df.columns:
        h24 = df[df["hour_business"] == 24]
        for idx, row in h24.iterrows():
            ds = pd.Timestamp(row["ds"])
            business_day = pd.Timestamp(row["business_day"]).normalize() if "business_day" in row and pd.notna(row["business_day"]) else None
            if business_day is not None:
                expected = business_day + pd.Timedelta(days=1)
                if ds != expected:
                    issues.append(
                        f"[{label}] hour 24 wrong at row {idx}: ds={ds}, expected={expected} "
                        f"for business_day={business_day.date()}"
                    )
            # Also check it's not D 00:00
            # If hour_business=24 maps to midnight, ds must have hour=0
            if ds.hour != 0:
                issues.append(f"[{label}] hour 24 ds={ds}: expected hour=00 (D+1 00:00)")

        # Detect D 00:00 entries (hour_business=24 should be D+1 00:00, which is correct)
        ts_all = pd.to_datetime(df["ds"], errors="coerce")
        midnight_entries = ts_all[ts_all.apply(lambda x: x.hour == 0)]
        if not midnight_entries.empty and "hour_business" in df.columns:
            invalid_midnight = []
            for idx in midnight_entries.index:
                if df.loc[idx, "hour_business"] != 24:
                    invalid_midnight.append(str(midnight_entries[idx]))
            if invalid_midnight:
                issues.append(f"[{label}] non-hour-24 midnight entries: {invalid_midnight}")

    # 5) business_day consistency
    if "business_day" in df.columns and "ds" in df.columns:
        ts_all = pd.to_datetime(df["ds"], errors="coerce")
        biz_days = pd.to_datetime(df["business_day"], errors="coerce")
        # hour 1-23 should have same date as business_day
        non_midnight = df[df["hour_business"] != 24]
        mismatches = 0
        for _, row in non_midnight.iterrows():
            ts = pd.to_datetime(row["ds"])
            bd = pd.to_datetime(row["business_day"])
            if ts.date() != bd.date():
                mismatches += 1
        if mismatches > 0:
            issues.append(f"[{label}] {mismatches} rows have ds date != business_day (non-midnight)")

    return issues


def check_timemixer_alignment(pred_dir: str, compare: bool = False) -> bool:
    """Check TimeMixer alignment for a given prediction directory."""
    path = Path(pred_dir)
    if not path.is_dir():
        print(f"ERROR: Directory not found: {pred_dir}")
        return False

    print(f"=== TimeMixer Alignment Check: {pred_dir} ===\n")

    all_ok = True

    # Locate TimeMixer predictions
    tm_files = list(path.rglob("*timemixer*_predictions.csv")) + list(
        path.rglob("timemixer*/predictions.csv")
    )
    # Also check direct filenames
    for candidate in ["timemixer_predictions.csv"]:
        fp = path / candidate
        if fp.exists():
            tm_files.append(fp)

    # Also check all_model_predictions_long.csv for timemixer rows
    for long_candidate in ["all_model_predictions_long.csv"]:
        for task_dir in [path / "dayahead" / "prediction", path / "realtime" / "prediction"]:
            lp = task_dir / long_candidate
            if lp.exists():
                df_long = _read_csv_safe(lp)
                if df_long is not None and "model_name" in df_long.columns:
                    tm_long = df_long[df_long["model_name"].str.lower() == "timemixer"]
                    if not tm_long.empty:
                        tm_files.append(lp)  # will filter unique later

    if not tm_files:
        print("  [FAIL] No TimeMixer predictions found")
        return False

    for tm_path in sorted(set(tm_files)):
        print(f"--- {tm_path.relative_to(path.parent)} ---")
        df = _read_csv_safe(tm_path)
        if df is None:
            print("  [FAIL] Cannot read file")
            all_ok = False
            continue

        # If this is the long table, filter to timemixer rows
        if "model_name" in df.columns:
            df_tm = df[df["model_name"].str.lower() == "timemixer"].copy()
            label = f"{tm_path.parent.name}/timemixer"
        else:
            df_tm = df
            label = tm_path.stem

        issues = _check_single_file(df_tm, label)

        for issue in issues:
            print(f"  {issue}")
            all_ok = False

        if not issues:
            print(f"  {label}: PASS")
        print()

    # Optional: compare with other models
    if compare:
        print("--- Cross-model comparison ---")
        # Get all model prediction CSVs
        for task in ["dayahead", "realtime"]:
            task_pred_dir = path / task / "prediction"
            if not task_pred_dir.exists():
                continue
            long_csv = task_pred_dir / "all_model_predictions_long.csv"
            df_long = _read_csv_safe(long_csv)
            if df_long is None or "model_name" not in df_long.columns:
                continue

            models = df_long["model_name"].unique()
            tm_biz = set()
            tm_hours = set()
            for _, row in df_long[df_long["model_name"].str.lower() == "timemixer"].iterrows():
                if "business_day" in df_long.columns and pd.notna(row.get("business_day")):
                    tm_biz.add(str(row["business_day"]))
                if "hour_business" in df_long.columns and pd.notna(row.get("hour_business")):
                    tm_hours.add(int(row["hour_business"]))

            for model in models:
                if model.lower() == "timemixer":
                    continue
                model_df = df_long[df_long["model_name"] == model]
                if "business_day" in model_df.columns:
                    other_biz = set(str(v) for v in model_df["business_day"].dropna().unique())
                    match = "MATCH" if other_biz == tm_biz else "MISMATCH"
                    if match == "MISMATCH":
                        print(f"  [{task}] business_day {model}: {sorted(other_biz)} vs TM: {sorted(tm_biz)} ({match})")
                        all_ok = False
                if "hour_business" in model_df.columns:
                    other_hours = sorted(model_df["hour_business"].dropna().astype(int).unique())
                    match = "MATCH" if other_hours == list(range(1, 25)) else "MISMATCH"
                    if match == "MISMATCH":
                        print(f"  [{task}] hour_business {model}: {other_hours} vs expected 1..24 ({match})")
                        all_ok = False

            print(f"  [{task}] cross-model comparison done")
        print()

    print(f"=== Result: {'ALL OK' if all_ok else 'ISSUES FOUND'} ===")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Check TimeMixer hour alignment")
    parser.add_argument("pred_dir", nargs="?", default=None, help="Path to prediction directory (e.g. outputs/runs/2026-02-24)")
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD")
    parser.add_argument("--runs-root", default="outputs/smoke/runs", help="Root directory for runs")
    parser.add_argument("--compare", action="store_true", help="Compare with other models")
    args = parser.parse_args()

    if args.pred_dir:
        pred_dir = args.pred_dir
    elif args.date:
        pred_dir = str(Path(args.runs_root) / args.date)
    else:
        parser.error("provide pred_dir or --date")

    ok = check_timemixer_alignment(pred_dir, compare=args.compare)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
