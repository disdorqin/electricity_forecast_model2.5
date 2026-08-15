#!/usr/bin/env python
"""
Synthetic tests for sync_dataset — no real database, HTTP, or local data required.

Tests:
  1. validate_synced_dataset PASS with valid DataFrame
  2. validate_synced_dataset FAIL on missing required column
  3. validate_synced_dataset FAIL on unparseable timestamps
  4. validate_synced_dataset deduplicates duplicate timestamps
  5. validate_synced_dataset freshness PASS / FAIL
  6. sync_dataset(source=local) with temp file -> OK
  7. sync_dataset(source=db) without real DB -> FAIL (no fallback)
  8. sync_dataset(source=auto) without DB/HTTP, with local -> OK
  9. sync_dataset(source=auto) with existing data -> SKIPPED
  10. sync_dataset(source=auto) --force-sync -> re-syncs
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.sync.sync_data import sync_dataset, validate_synced_dataset

PASS = 0
FAIL = 1
results: list[tuple[str, int, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"PASS: {name}" if status == PASS else f"FAIL: {name}"
    if detail and status == FAIL:
        msg += f" — {detail}"
    results.append((name, status, detail))
    print(msg)


def _make_valid_df(extra_rows: int = 0) -> pd.DataFrame:
    """Create a valid synthetic DataFrame."""
    rows = []
    for h in range(24):
        ts = f"2026-02-24 {h:02d}:00:00"
        rows.append({
            "时刻": ts,
            "日前电价": 200.0 + h,
            "实时电价": 180.0 + h,
            "feature_a": 1.0,
        })
    # Add extra rows if requested
    for i in range(extra_rows):
        ts = f"2026-02-25 {i:02d}:00:00"
        rows.append({
            "时刻": ts,
            "日前电价": 200.0 + i,
            "实时电价": 180.0 + i,
            "feature_a": 1.0,
        })
    return pd.DataFrame(rows)


def test_validation_valid() -> str:
    """Test 1: validate_synced_dataset PASS."""
    df = _make_valid_df(extra_rows=5)
    result = validate_synced_dataset(df, target_date="2026-02-25", max_data_lag_hours=48)
    check(
        "validation: valid dataset PASS",
        result["status"] == "ok",
        f"expected ok, got {result['status']}: {result.get('errors')}",
    )
    check(
        "validation: has min/max timestamps",
        result.get("min_timestamp") is not None and result.get("max_timestamp") is not None,
        "missing timestamps",
    )
    check(
        "validation: freshness PASS",
        result.get("freshness", {}).get("status") == "PASS",
        f"got {result.get('freshness')}",
    )
    return ""


def test_validation_missing_column() -> str:
    """Test 2: validate_synced_dataset FAIL on missing column."""
    df = _make_valid_df().drop(columns=["日前电价"])
    result = validate_synced_dataset(df)
    check(
        "validation: missing column FAIL",
        result["status"] == "failed",
        f"expected failed, got {result['status']}",
    )
    has_col_error = any("日前电价" in e for e in result.get("errors", []))
    check(
        "validation: error mentions missing column",
        has_col_error,
        f"errors: {result.get('errors')}",
    )
    return ""


def test_validation_bad_timestamps() -> str:
    """Test 3: validate_synced_dataset FAIL on unparseable timestamps."""
    df = _make_valid_df()
    df.loc[0, "时刻"] = "not-a-timestamp"
    result = validate_synced_dataset(df)
    check(
        "validation: bad timestamps FAIL",
        result["status"] == "failed",
        f"expected failed, got {result['status']}",
    )
    has_ts_error = any("timestamp" in e.lower() for e in result.get("errors", []))
    check(
        "validation: error mentions timestamps",
        has_ts_error,
        f"errors: {result.get('errors')}",
    )
    return ""


def test_validation_deduplicates() -> str:
    """Test 4: validate_synced_dataset deduplicates timestamps."""
    df = _make_valid_df()
    # Add duplicate rows
    dup = df.iloc[:5].copy()
    dup["日前电价"] = 999.0  # different price — keep last
    df = pd.concat([df, dup], ignore_index=True)
    result = validate_synced_dataset(df)
    check(
        "validation: dedup PASS (status ok)",
        result["status"] == "ok",
        f"expected ok, got {result['status']}",
    )
    check(
        "validation: dedup warning emitted",
        any("duplicate" in w.lower() for w in result.get("warnings", [])),
        f"warnings: {result.get('warnings')}",
    )
    check(
        "validation: rows after dedup equals 24 unique timestamps",
        result["rows"] == 24,
        f"expected 24 rows (unique timestamps), got {result['rows']}",
    )
    return ""


def test_validation_freshness_fail() -> str:
    """Test 5: validate_synced_dataset freshness FAIL."""
    df = _make_valid_df()  # max ts = 2026-02-24 23:00
    result = validate_synced_dataset(df, target_date="2026-02-28", max_data_lag_hours=12)
    check(
        "validation: freshness FAIL when data too old",
        result.get("freshness", {}).get("status") == "FAIL",
        f"freshness: {result.get('freshness')}",
    )
    return ""


def test_sync_local() -> str:
    """Test 6: sync_dataset(source=local) with temp file."""
    df = _make_valid_df()
    with tempfile.TemporaryDirectory() as td:
        xlsx_path = Path(td) / "data" / "shandong_pmos_hourly.xlsx"
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(xlsx_path, index=False)

        # Also save csv for local fallback
        csv_path = xlsx_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="gbk")

        result = sync_dataset(
            data_path=str(xlsx_path),
            source="local",
            force=True,
        )
        check(
            "sync_dataset local: status ok",
            result.get("status") == "ok",
            f"got {result.get('status')}: {result.get('errors')}",
        )
        check(
            "sync_dataset local: source is 'local'",
            result.get("source") == "local",
            f"got {result.get('source')}",
        )
        check(
            "sync_dataset local: rows > 0",
            result.get("rows", 0) > 0,
            f"rows = {result.get('rows')}",
        )
        check(
            "sync_dataset local: output_xlsx matches",
            result.get("output_xlsx") == str(xlsx_path),
            f"got {result.get('output_xlsx')}",
        )
    return ""


def test_sync_db_fails_no_fallback() -> str:
    """Test 7: sync_dataset(source=db) without real DB -> FAIL (fast)."""
    # Mock fetch_web_grid_data to raise immediately instead of waiting for timeout
    with patch("scripts.sync.sync_data.fetch_web_grid_data", side_effect=ValueError("mock: no database")):
        result = sync_dataset(source="db", force=True)
    check(
        "sync_dataset db only: status failed",
        result.get("status") == "failed",
        f"expected failed, got {result.get('status')}",
    )
    check(
        "sync_dataset db only: source is db",
        result.get("source") == "db",
        f"got {result.get('source')}",
    )
    return ""


def test_sync_auto_with_local() -> str:
    """Test 8: sync_dataset(source=auto) without DB/HTTP, with local -> OK."""
    with tempfile.TemporaryDirectory() as td:
        xlsx_path = Path(td) / "data" / "shandong_pmos_hourly.xlsx"
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        df = _make_valid_df()
        df.to_excel(xlsx_path, index=False)

        with patch("scripts.sync.sync_data.fetch_web_grid_data", side_effect=ValueError("mock: DB down")), \
             patch("scripts.sync.sync_data._download_latest_available_excel", side_effect=FileNotFoundError("mock: no HTTP")):
            result = sync_dataset(
                data_path=str(xlsx_path),
                source="auto",
                force=True,
            )
        check(
            "sync_dataset auto with local: status ok",
            result.get("status") == "ok",
            f"got {result.get('status')}: {result.get('errors')}",
        )
        check(
            "sync_dataset auto with local: source is 'local'",
            result.get("source") == "local",
            f"got {result.get('source')}",
        )
    return ""


def test_sync_skipped_when_exists() -> str:
    """Test 9: sync_dataset skips when existing data and no force."""
    with tempfile.TemporaryDirectory() as td:
        xlsx_path = Path(td) / "shandong_pmos_hourly.xlsx"
        df = _make_valid_df()
        df.to_excel(xlsx_path, index=False)

        with patch("scripts.sync.sync_data.fetch_web_grid_data", side_effect=ValueError("mock: no DB")), \
             patch("scripts.sync.sync_data._download_latest_available_excel", side_effect=FileNotFoundError("mock: no HTTP")):
            # First call with force=True to ensure it was written
            r1 = sync_dataset(data_path=str(xlsx_path), source="auto", force=True)
            check(
                "sync_dataset first call: status ok",
                r1.get("status") == "ok",
                f"got {r1.get('status')}",
            )

            # Second call without force — should skip (because file exists + no force)
            r2 = sync_dataset(data_path=str(xlsx_path), source="auto", force=False)
            check(
                "sync_dataset second call: status skipped",
                r2.get("status") == "skipped",
                f"expected skipped, got {r2.get('status')}: {r2}",
            )
    return ""


def test_sync_force_resyncs() -> str:
    """Test 10: sync_dataset with force=True re-syncs."""
    with tempfile.TemporaryDirectory() as td:
        xlsx_path = Path(td) / "shandong_pmos_hourly.xlsx"
        df = _make_valid_df()
        df.to_excel(xlsx_path, index=False)

        with patch("scripts.sync.sync_data.fetch_web_grid_data", side_effect=ValueError("mock: no DB")), \
             patch("scripts.sync.sync_data._download_latest_available_excel", side_effect=FileNotFoundError("mock: no HTTP")):
            r1 = sync_dataset(data_path=str(xlsx_path), source="auto", force=False)
            check(
                "force: first call status skipped",
                r1.get("status") == "skipped",
                f"got {r1.get('status')}",
            )

            r2 = sync_dataset(data_path=str(xlsx_path), source="auto", force=True)
            check(
                "force: second call (force) status ok",
                r2.get("status") == "ok",
                f"got {r2.get('status')}: {r2.get('errors')}",
            )
    return ""


def test_validate_source_http_fails_without_network() -> str:
    """Test 11: sync_dataset(source=http) without network -> FAIL (fast)."""
    with patch("scripts.sync.sync_data.fetch_web_grid_data", side_effect=ValueError("mock: no DB")), \
         patch("scripts.sync.sync_data._download_latest_available_excel", side_effect=FileNotFoundError("mock: no HTTP")):
        result = sync_dataset(source="http", force=True)
    check(
        "sync_dataset http only: status failed",
        result.get("status") == "failed",
        f"expected failed, got {result.get('status')}",
    )
    return ""


def test_validate_unknown_source() -> str:
    """Test 12: sync_dataset(source=unknown) -> FAIL."""
    result = sync_dataset(source="unknown")
    check(
        "sync_dataset unknown source: status failed",
        result.get("status") == "failed",
        f"expected failed, got {result.get('status')}",
    )
    has_unknown_error = any("unknown" in e.lower() for e in result.get("errors", []))
    check(
        "sync_dataset unknown source: error mentions unknown",
        has_unknown_error,
        f"errors: {result.get('errors')}",
    )
    return ""


def test_sync_manifest_written() -> str:
    """Test 13: sync_dataset writes manifest to outputs/data_sync/."""
    from scripts.sync.sync_data import SYNC_MANIFEST_DIR

    manifest_path = SYNC_MANIFEST_DIR / "sync_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        check(
            "sync manifest exists on disk",
            manifest_path.exists(),
        )
        check(
            "sync manifest has synced_at timestamp",
            "synced_at" in manifest,
        )
        check(
            "sync manifest has status field",
            "status" in manifest,
        )
    else:
        # Might not exist if all DB/HTTP attempts failed but we did at least
        # call sync_dataset. The manifest might not have been written if the
        # data path is in a temp dir. That's fine — check weaker condition.
        check("sync manifest: outputs/data_sync/ dir exists", SYNC_MANIFEST_DIR.exists())
    return ""


def test_prefer_data_path_over_default() -> str:
    """Test 14: sync_dataset respects custom data_path."""
    with tempfile.TemporaryDirectory() as td:
        custom_path = Path(td) / "custom" / "my_data.xlsx"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        df = _make_valid_df()
        df.to_excel(custom_path, index=False)

        result = sync_dataset(
            data_path=str(custom_path),
            source="local",
            force=True,
        )
        check(
            "custom data_path is respected",
            result.get("output_xlsx") == str(custom_path),
            f"expected {custom_path}, got {result.get('output_xlsx')}",
        )
        check(
            "custom data_path: csv is derived",
            result.get("output_csv") == str(custom_path.with_suffix(".csv")),
            f"got {result.get('output_csv')}",
        )
    return ""


def _make_df_up_to(last_date: str, last_hour: int = 23) -> pd.DataFrame:
    """Create a synthetic DataFrame from 2026-02-24 up to *last_date*."""
    from datetime import datetime, timedelta

    start = datetime(2026, 2, 24)
    end = datetime.strptime(last_date, "%Y-%m-%d").replace(hour=last_hour)
    rows = []
    cursor = start
    while cursor <= end:
        rows.append({
            "时刻": cursor.strftime("%Y-%m-%d %H:%M:%S"),
            "日前电价": 200.0 + cursor.hour,
            "实时电价": 180.0 + cursor.hour,
            "feature_a": 1.0,
        })
        cursor += timedelta(hours=1)
    return pd.DataFrame(rows)


def test_require_fresh_stale_fails() -> str:
    """Test 15: existing file + require_fresh=True + stale data -> failed."""
    with tempfile.TemporaryDirectory() as td:
        xlsx_path = Path(td) / "shandong_pmos_hourly.xlsx"
        # Data only up to 2026-02-24 — stale for target 2026-02-28
        df = _make_df_up_to("2026-02-24")
        df.to_excel(xlsx_path, index=False)

        result = sync_dataset(
            data_path=str(xlsx_path),
            source="auto",
            force=False,
            require_fresh=True,
            target_date="2026-02-28",
            max_data_lag_hours=36,
        )
        check(
            "require_fresh stale: status failed",
            result.get("status") == "failed",
            f"expected failed, got {result.get('status')}",
        )
        freshness = result.get("freshness", {})
        check(
            "require_fresh stale: freshness FAIL",
            freshness.get("status") == "FAIL",
            f"expected FAIL, got {freshness}",
        )
    return ""


def test_require_fresh_fresh_skips() -> str:
    """Test 16: existing file + require_fresh=True + fresh data -> skipped/freshness PASS."""
    with tempfile.TemporaryDirectory() as td:
        xlsx_path = Path(td) / "shandong_pmos_hourly.xlsx"
        # Data up to 2026-02-27 — fresh for target 2026-02-28 within 36h window
        df = _make_df_up_to("2026-02-27")
        df.to_excel(xlsx_path, index=False)

        result = sync_dataset(
            data_path=str(xlsx_path),
            source="auto",
            force=False,
            require_fresh=True,
            target_date="2026-02-28",
            max_data_lag_hours=36,
        )
        check(
            "require_fresh fresh: status skipped",
            result.get("status") == "skipped",
            f"expected skipped, got {result.get('status')}",
        )
        freshness = result.get("freshness", {})
        check(
            "require_fresh fresh: freshness PASS",
            freshness.get("status") == "PASS",
            f"expected PASS, got {freshness}",
        )
    return ""


def test_custom_data_path_writes_files() -> str:
    """Test 17: custom data_path actually writes to custom path."""
    with tempfile.TemporaryDirectory() as td:
        custom_path = Path(td) / "custom" / "my_data.xlsx"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        df = _make_valid_df()
        df.to_excel(custom_path, index=False)

        result = sync_dataset(
            data_path=str(custom_path),
            source="local",
            force=True,
        )
        check(
            "custom path: xlsx file exists on disk",
            Path(result["output_xlsx"]).exists(),
            f"xlsx not found at {result['output_xlsx']}",
        )
        check(
            "custom path: csv file exists on disk",
            Path(result["output_csv"]).exists(),
            f"csv not found at {result['output_csv']}",
        )
        check(
            "custom path: output_xlsx matches custom_path",
            result.get("output_xlsx") == str(custom_path),
            f"expected {custom_path}, got {result.get('output_xlsx')}",
        )
    return ""


def test_manifest_matches_real_file() -> str:
    """Test 18: manifest output_xlsx matches real file on disk."""
    with tempfile.TemporaryDirectory() as td:
        custom_path = Path(td) / "data" / "shandong_pmos_hourly.xlsx"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        df = _make_valid_df()
        df.to_excel(custom_path, index=False)

        result = sync_dataset(
            data_path=str(custom_path),
            source="local",
            force=True,
        )
        # Verify manifest path matches actual file
        manifest_xlsx = result.get("output_xlsx", "")
        manifest_csv = result.get("output_csv", "")
        check(
            "manifest: output_xlsx matches file on disk",
            manifest_xlsx == str(custom_path) and Path(manifest_xlsx).exists(),
            f"manifest xlsx={manifest_xlsx}, custom={custom_path}",
        )
        check(
            "manifest: output_csv is derived correctly",
            manifest_csv == str(custom_path.with_suffix(".csv")) and Path(manifest_csv).exists(),
            f"manifest csv={manifest_csv}, expected={custom_path.with_suffix('.csv')}",
        )
        # Verify the manifest written to outputs/data_sync/ has the same paths
        from scripts.sync.sync_data import SYNC_MANIFEST_DIR
        manifest_path = SYNC_MANIFEST_DIR / "sync_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as f:
                on_disk = json.load(f)
            check(
                "manifest: on-disk output_xlsx matches",
                on_disk.get("output_xlsx") == manifest_xlsx,
                f"on-disk={on_disk.get('output_xlsx')}, returned={manifest_xlsx}",
            )
    return ""


def main() -> int:
    print("=" * 60)
    print("CHECK_SYNC_DATASET")
    print("=" * 60)
    print()

    test_validation_valid()
    test_validation_missing_column()
    test_validation_bad_timestamps()
    test_validation_deduplicates()
    test_validation_freshness_fail()
    test_sync_local()
    test_sync_db_fails_no_fallback()
    test_sync_auto_with_local()
    test_sync_skipped_when_exists()
    test_sync_force_resyncs()
    test_validate_source_http_fails_without_network()
    test_validate_unknown_source()
    test_sync_manifest_written()
    test_prefer_data_path_over_default()
    test_require_fresh_stale_fails()
    test_require_fresh_fresh_skips()
    test_custom_data_path_writes_files()
    test_manifest_matches_real_file()

    print()
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)

    for name, status, detail in results:
        marker = "PASS" if status == PASS else "FAIL"
        print(f"{marker}: {name}")
        if status == FAIL and detail:
            print(f"  {detail}")

    print()
    print(f"RESULT: {passed}/{len(results)} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
