#!/usr/bin/env python
"""
Synthetic tests for the native 96-point (15min) database synchronization.

No real database, HTTP, or production data required. The remote DB fetch
functions and server-version probe are mocked; only the local-mirror logic,
atomic writes, dedup, completeness/validation, and manifest are exercised.

Tests:
  CLI
    1. default resolution is hourly
    2. explicit hourly parses
    3. 15min parses correctly
    4. invalid resolution fails clearly
    5. full vs incremental sync-mode parses
  Configuration
    6. missing DB config fails safely
    7. secrets are not written to the manifest
  Database synchronization (mocks)
    8. full sync downloads all rows
    9. incremental sync uses overlap window
   10. dedup works on true keys
   11. multiple units preserved (no filtering to one unit)
   12. partial table failure does not destroy a previous valid file
   13. atomic-write rollback preserves prior file on write error
   14. read-only SQL only (no INSERT/UPDATE/DELETE/DROP/... issued)
  Data integrity
   15. 96 rows per complete day
   16. period_no == 1..96
   17. p1/p96 interval-end mapping
   18. no duplicate keys
   19. manifest row counts match local files
   20. remote/local selected-range row counts match
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date, datetime, timedelta, time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.sync import sync_data_96_core as core
import utils.database_operate as db
from cli.parser import build_parser

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_market_df(days: int = 2, start: date = date(2024, 1, 1)) -> pd.DataFrame:
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        for p in range(1, 97):
            dt = datetime.combine(d, time()) + timedelta(minutes=15 * p)
            rows.append({
                "data_time": dt, "market_date": d, "period_no": p,
                "actual_direct_load": 100.0 + p, "fcast_direct_load": 99.0 + p,
            })
    return pd.DataFrame(rows)


def _make_unit_df(days: int = 1, units=("U1", "U2"), start: date = date(2024, 1, 1)) -> pd.DataFrame:
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        for p in range(1, 97):
            dt = datetime.combine(d, time()) + timedelta(minutes=15 * p)
            for u in units:
                rows.append({
                    "data_time": dt, "market_date": d, "period_no": p, "unit_id": u,
                    "da_cq_price": 300.0 + p, "rt_cq_price": 280.0 + p,
                    "da_power": 50.0, "rt_power": 48.0,
                })
    return pd.DataFrame(rows)


def _mock_market_full(start_date=None, end_date=None):
    df = _make_market_df(2)
    if start_date:
        df = df[pd.to_datetime(df["market_date"]) >= pd.Timestamp(start_date)]
    return df.reset_index(drop=True)


def _mock_unit_full(start_date=None, end_date=None, units=("U1", "U2")):
    df = _make_unit_df(1, units=units)
    if start_date:
        df = df[pd.to_datetime(df["market_date"]) >= pd.Timestamp(start_date)]
    return df.reset_index(drop=True)


def _mock_summary(table):
    if table == "epf_market_data_96":
        return {"d_min": "2024-01-01", "d_max": "2024-01-02", "rows_total": 192}
    return {"d_min": "2024-01-01", "d_max": "2024-01-01", "rows_total": 192}


# ---------------------------------------------------------------------------
# Temp-mirror harness
# ---------------------------------------------------------------------------


class MirrorHarness:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "remote_96"
        self.root = root
        core.REMOTE_96_ROOT = root
        core.RAW_DIR = root / "raw"
        core.PARQUET_DIR = root / "parquet"
        core.METADATA_DIR = root / "metadata"
        core.MANIFEST_DIR = root / "outputs" / "data_sync_96"
        core.SYNC_MANIFEST_PATH = core.MANIFEST_DIR / "sync_manifest.json"

    def close(self):
        self.tmp.cleanup()


def _args(**kw):
    class A:
        pass
    a = A()
    a.sync_source = kw.get("sync_source", "db")
    a.sync_mode = kw.get("sync_mode", "full")
    a.include_extended = kw.get("include_extended", False)
    a.sync_overlap_days = kw.get("overlap_days", 7)
    a.force_sync = kw.get("force_sync", False)
    # Unit tests use synthetic 2024-era data, so the 2022->2026 audit-baseline
    # floor must be relaxed (real runs keep it True via the CLI default).
    a.enforce_audit_baseline = kw.get("enforce_audit_baseline", False)
    return a


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_default_hourly():
    p = build_parser()
    ns = p.parse_args(["--pipeline", "sync_dataset"])
    check("cli: default resolution is hourly", ns.resolution == "hourly",
          f"got {ns.resolution}")


def test_cli_explicit_hourly():
    p = build_parser()
    ns = p.parse_args(["--pipeline", "sync_dataset", "--resolution", "hourly"])
    check("cli: explicit hourly parses", ns.resolution == "hourly")


def test_cli_15min():
    p = build_parser()
    ns = p.parse_args(["--pipeline", "sync_dataset", "--resolution", "15min",
                        "--sync-source", "db", "--sync-mode", "full"])
    check("cli: 15min parses", ns.resolution == "15min")
    check("cli: 15min sync-mode full", ns.sync_mode == "full")


def test_cli_invalid_resolution():
    p = build_parser()
    raised = False
    try:
        p.parse_args(["--pipeline", "sync_dataset", "--resolution", "foo"])
    except SystemExit:
        raised = True
    check("cli: invalid resolution fails clearly", raised)


def test_cli_sync_mode_parse():
    p = build_parser()
    ns = p.parse_args(["--pipeline", "sync_dataset", "--resolution", "15min",
                        "--sync-mode", "incremental", "--sync-overlap-days", "10"])
    check("cli: incremental mode parses", ns.sync_mode == "incremental")
    check("cli: overlap days parses", ns.sync_overlap_days == 10)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


def test_missing_config_fails():
    h = MirrorHarness()
    try:
        with patch.object(db, "get_db_connection", side_effect=ValueError("Database env vars are incomplete")):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        check("config: missing config fails safely",
              manifest["status"] in ("failed", "partial"),
              f"status={manifest['status']}")
        check("config: failed tables recorded",
              len(manifest.get("tables_failed", [])) > 0,
              f"failed={manifest.get('tables_failed')}")
    finally:
        h.close()


def test_secrets_not_in_manifest():
    h = MirrorHarness()
    try:
        pwd = ""
        env_path = Path("D:/作业/大创_挑战杯_互联网/大学生创新创业计划/大创实现/其他资料/electricity_forecast_model2.5/.env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DB_PWD="):
                    pwd = line.split("=", 1)[1].strip().strip("'\"")
        with patch.object(db, "fetch_market_data_96_full", _mock_market_full), \
             patch.object(db, "fetch_unit_data_96_full", _mock_unit_full), \
             patch.object(db, "fetch_96_table_summary", _mock_summary), \
             patch.object(db, "get_db_server_version", return_value="5.7.40-log"):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        blob = json.dumps(manifest, ensure_ascii=False, default=str)
        check("secrets: DB password absent from manifest",
              (pwd == "" or pwd not in blob),
              "password leaked into manifest" if pwd and pwd in blob else "")
    finally:
        h.close()


# ---------------------------------------------------------------------------
# DB synchronization tests (mocks)
# ---------------------------------------------------------------------------


def test_full_sync_all_rows():
    h = MirrorHarness()
    try:
        with patch.object(db, "fetch_market_data_96_full", _mock_market_full), \
             patch.object(db, "fetch_unit_data_96_full", _mock_unit_full), \
             patch.object(db, "fetch_96_table_summary", _mock_summary), \
             patch.object(db, "get_db_server_version", return_value="5.7.40-log"):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        check("full: status ok", manifest["status"] == "ok", str(manifest.get("errors")))
        check("full: market rows == 192",
              manifest["local_row_count"]["epf_market_data_96"] == 192)
        check("full: unit rows == 192",
              manifest["local_row_count"]["epf_unit_data_96"] == 192)
    finally:
        h.close()


def test_incremental_uses_overlap():
    h = MirrorHarness()
    try:
        # Pre-seed a local mirror with day 1 only.
        df_day1 = _make_market_df(1)
        core.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        df_day1.to_parquet(core._table_parquet_path("epf_market_data_96"), index=False)
        # Incremental: fetch returns 2 days; merge should yield 2 days.
        with patch.object(db, "fetch_market_data_96_full", _mock_market_full), \
             patch.object(db, "fetch_unit_data_96_full", _mock_unit_full), \
             patch.object(db, "fetch_96_table_summary", _mock_summary), \
             patch.object(db, "get_db_server_version", return_value="5.7.40-log"):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="incremental",
                                          overlap_days=7))
        check("incremental: market complete 96-days == 2",
              manifest["complete_96_days_per_table"]["epf_market_data_96"] == 2,
              str(manifest["complete_96_days_per_table"]))
        check("incremental: market local rows == 192",
              manifest["local_row_count"]["epf_market_data_96"] == 192)
    finally:
        h.close()


def test_dedup_true_keys():
    h = MirrorHarness()
    try:
        df = _make_market_df(1)
        df = pd.concat([df, df.iloc[[-1]].copy()], ignore_index=True)  # dup key
        with patch.object(db, "fetch_market_data_96_full", lambda *a, **k: df), \
             patch.object(db, "fetch_unit_data_96_full", _mock_unit_full), \
             patch.object(db, "fetch_96_table_summary", _mock_summary), \
             patch.object(db, "get_db_server_version", return_value="5.7.40-log"):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        check("dedup: market dup keys == 0 after sync",
              manifest["duplicate_key_count"]["epf_market_data_96"] == 0,
              str(manifest["duplicate_key_count"]))
    finally:
        h.close()


def test_multiple_units_preserved():
    h = MirrorHarness()
    try:
        with patch.object(db, "fetch_market_data_96_full", _mock_market_full), \
             patch.object(db, "fetch_unit_data_96_full",
                          lambda *a, **k: _mock_unit_full(units=("U1", "U2"))), \
             patch.object(db, "fetch_96_table_summary", _mock_summary), \
             patch.object(db, "get_db_server_version", return_value="5.7.40-log"):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        # 1 day, 2 units -> 192 rows, 2 distinct units.
        local = pd.read_parquet(core._table_parquet_path("epf_unit_data_96"))
        check("multi-unit: rows == 192 (no unit filtering)", len(local) == 192)
        check("multi-unit: 2 distinct units preserved",
              local["unit_id"].nunique() == 2,
              f"units={local['unit_id'].nunique()}")
    finally:
        h.close()


def test_partial_failure_preserves_valid_file():
    h = MirrorHarness()
    try:
        # Pre-seed valid unit mirror (1 day, single unit -> 96 rows).
        core.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        _make_unit_df(1, units=("U1",)).to_parquet(
            core._table_parquet_path("epf_unit_data_96"), index=False)
        # Make unit fetch fail; market fetch ok.
        def _unit_fail(*a, **k):
            raise RuntimeError("simulated DB error for unit table")
        with patch.object(db, "fetch_market_data_96_full", _mock_market_full), \
             patch.object(db, "fetch_unit_data_96_full", _unit_fail), \
             patch.object(db, "fetch_96_table_summary", _mock_summary), \
             patch.object(db, "get_db_server_version", return_value="5.7.40-log"):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        check("partial: market succeeded",
              "epf_market_data_96" in manifest["tables_succeeded"])
        check("partial: unit failed",
              "epf_unit_data_96" in manifest["tables_failed"])
        # The previously-valid unit file must still exist with 96 rows.
        local = pd.read_parquet(core._table_parquet_path("epf_unit_data_96"))
        check("partial: prior valid unit file preserved (96 rows)",
              len(local) == 96, f"rows={len(local)}")
    finally:
        h.close()


def test_atomic_rollback_on_write_error():
    h = MirrorHarness()
    try:
        # Pre-seed valid market mirror (1 day).
        core.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        _make_market_df(1).to_parquet(core._table_parquet_path("epf_market_data_96"), index=False)
        before = pd.read_parquet(core._table_parquet_path("epf_market_data_96"))
        # Force the atomic parquet write to fail.
        with patch.object(db, "fetch_market_data_96_full", _mock_market_full), \
             patch.object(db, "fetch_unit_data_96_full", _mock_unit_full), \
             patch.object(db, "fetch_96_table_summary", _mock_summary), \
             patch.object(db, "get_db_server_version", return_value="5.7.40-log"), \
             patch.object(core, "_atomic_write_parquet", side_effect=IOError("disk full")):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        check("rollback: market sync failed on write error",
              "epf_market_data_96" in manifest["tables_failed"])
        after = pd.read_parquet(core._table_parquet_path("epf_market_data_96"))
        check("rollback: prior valid file unchanged (96 rows)",
              len(after) == len(before) == 96,
              f"before={len(before)} after={len(after)}")
        # No .partial left behind
        partials = list(core.PARQUET_DIR.glob("*.partial"))
        check("rollback: no leftover .partial files", len(partials) == 0,
              f"partials={partials}")
    finally:
        h.close()


def test_read_only_sql_only():
    h = MirrorHarness()
    try:
        recorded = []

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                recorded.append(str(sql))

            def fetchall(self):
                if recorded and "COUNT" in recorded[-1]:
                    return [{"d_min": "2024-01-01", "d_max": "2024-01-02", "rows_total": 192}]
                return [dict(data_time=datetime(2024, 1, 1, 0, 15), market_date=date(2024, 1, 1),
                             period_no=1, actual_direct_load=1.0, fcast_direct_load=1.0)]

        class FakeConn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def cursor(self):
                return FakeCursor()

            def get_server_info(self):
                return "5.7.40-log"

            def close(self):
                pass

        with patch.object(db, "get_db_connection", return_value=FakeConn()):
            manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
        forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
                     "TRUNCATE", "REPLACE")
        bad = [s for s in recorded if any(w in s.upper() for w in forbidden)]
        check("readonly: no mutating SQL issued", len(bad) == 0,
              f"forbidden SQL: {bad}")
        check("readonly: only SELECT issued", all(s.upper().strip().startswith("SELECT")
                                                  for s in recorded),
              f"recorded={recorded}")
    finally:
        h.close()


# ---------------------------------------------------------------------------
# Data integrity tests
# ---------------------------------------------------------------------------


def _run_full_mock_sync():
    h = MirrorHarness()
    with patch.object(db, "fetch_market_data_96_full", _mock_market_full), \
         patch.object(db, "fetch_unit_data_96_full", _mock_unit_full), \
         patch.object(db, "fetch_96_table_summary", _mock_summary), \
         patch.object(db, "get_db_server_version", return_value="5.7.40-log"):
        manifest = core.sync_96(_args(sync_source="db", sync_mode="full"))
    return h, manifest


def test_integrity_96_rows_per_day():
    h, manifest = _run_full_mock_sync()
    try:
        local = pd.read_parquet(core._table_parquet_path("epf_market_data_96"))
        day_counts = local.groupby("market_date")["period_no"].nunique()
        check("integrity: every day has exactly 96 periods",
              bool((day_counts == 96).all()), f"counts={day_counts.to_dict()}")
    finally:
        h.close()


def test_integrity_period_range():
    h, manifest = _run_full_mock_sync()
    try:
        local = pd.read_parquet(core._table_parquet_path("epf_market_data_96"))
        pn = pd.to_numeric(local["period_no"], errors="coerce")
        check("integrity: period_no within 1..96",
              int(pn.min()) == 1 and int(pn.max()) == 96,
              f"min={pn.min()} max={pn.max()}")
    finally:
        h.close()


def test_integrity_p1_p96():
    h, manifest = _run_full_mock_sync()
    try:
        local = pd.read_parquet(core._table_parquet_path("epf_market_data_96"))
        ok, detail = core._check_p1_p96(local)
        check("integrity: p1/p96 interval-end semantics", ok, detail)
    finally:
        h.close()


def test_integrity_no_dup_keys():
    h, manifest = _run_full_mock_sync()
    try:
        local = pd.read_parquet(core._table_parquet_path("epf_market_data_96"))
        dups = local.duplicated(subset=["market_date", "period_no"]).sum()
        check("integrity: no duplicate (market_date, period_no) keys",
              int(dups) == 0, f"dups={dups}")
    finally:
        h.close()


def test_integrity_manifest_matches_local():
    h, manifest = _run_full_mock_sync()
    try:
        local = pd.read_parquet(core._table_parquet_path("epf_market_data_96"))
        check("integrity: manifest rows == local file rows",
              manifest["local_row_count"]["epf_market_data_96"] == len(local),
              f"manifest={manifest['local_row_count']['epf_market_data_96']} local={len(local)}")
    finally:
        h.close()


def test_integrity_remote_local_match():
    h, manifest = _run_full_mock_sync()
    try:
        check("integrity: remote/local row_count_match True",
              manifest["row_count_match"]["epf_market_data_96"] is True,
              str(manifest["row_count_match"]))
        check("integrity: remote total == local (full mode)",
              manifest["remote_row_count"]["epf_market_data_96"]
              == manifest["local_row_count"]["epf_market_data_96"],
              f"remote={manifest['remote_row_count']['epf_market_data_96']} "
              f"local={manifest['local_row_count']['epf_market_data_96']}")
    finally:
        h.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 60)
    print("CHECK_SYNC_DATASET_96")
    print("=" * 60)
    print()

    test_cli_default_hourly()
    test_cli_explicit_hourly()
    test_cli_15min()
    test_cli_invalid_resolution()
    test_cli_sync_mode_parse()
    test_missing_config_fails()
    test_secrets_not_in_manifest()
    test_full_sync_all_rows()
    test_incremental_uses_overlap()
    test_dedup_true_keys()
    test_multiple_units_preserved()
    test_partial_failure_preserves_valid_file()
    test_atomic_rollback_on_write_error()
    test_read_only_sql_only()
    test_integrity_96_rows_per_day()
    test_integrity_period_range()
    test_integrity_p1_p96()
    test_integrity_no_dup_keys()
    test_integrity_manifest_matches_local()
    test_integrity_remote_local_match()

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
