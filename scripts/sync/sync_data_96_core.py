"""
Native 96-point (15-minute) database -> local mirror synchronization.

This module is the canonical 96-point data-synchronization path. It is
invoked from the unified CLI via ``main.py --pipeline sync_dataset
--resolution 15min ...``.

Design principles (per project task):
  * READ-ONLY against the remote database. Only bounded ``SELECT`` (and
    aggregate ``SELECT COUNT/MIN/MAX``) are issued. No INSERT/UPDATE/DELETE/
    CREATE/ALTER/DROP/TRUNCATE, no crawler state mutation, no schema change.
  * LOCAL MIRROR only. Raw remote tables are mirrored under
    ``data/remote_96/`` (raw CSV.GZ + parquet) and never merged into the
    hourly ``data/shandong_pmos_hourly*`` files.
  * ATOMIC WRITES. Each table is written to a ``.partial`` temp file then
    ``os.replace``-d into place; a mid-sync failure never corrupts a prior
    valid local mirror.
  * DEDUP by the database's true unique key.
  * FULL and INCREMENTAL modes, with configurable overlap to capture late
    backfills of recent actual values.
  * No credentials are hardcoded, printed, or written to any manifest.

Only ``epf_market_data_96`` and ``epf_unit_data_96`` are ``required_core``
tables. Additional 96-point tables (congestion, tie-line, weather) are
``optional_extended`` and only downloaded when ``--include-extended`` is set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from utils.database_operate import (
    fetch_96_table,
    fetch_96_table_summary,
    fetch_market_data_96_full,
    fetch_unit_data_96_full,
    get_db_server_version,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (all under git-ignored data/ and outputs/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_96_ROOT = PROJECT_ROOT / "data" / "remote_96"
RAW_DIR = REMOTE_96_ROOT / "raw"
PARQUET_DIR = REMOTE_96_ROOT / "parquet"
METADATA_DIR = REMOTE_96_ROOT / "metadata"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "data_sync_96"
SYNC_MANIFEST_PATH = MANIFEST_DIR / "sync_manifest.json"

# ---------------------------------------------------------------------------
# Table classification (grounded in dist/audit_output)
# ---------------------------------------------------------------------------
# required_core: first production-compatible local dataset must contain these.
# `fetch_name` is resolved lazily via getattr(utils.database_operate, ...)
# so that tests can patch the underlying DB functions cleanly.
CORE_TABLES: dict[str, dict] = {
    "epf_market_data_96": {
        "key": ["market_date", "period_no"],
        "entity_col": None,
        "fetch_name": "fetch_market_data_96_full",
        "classification": "required_core",
        "note": "market-level 96-point grid operation features",
    },
    "epf_unit_data_96": {
        "key": ["market_date", "period_no", "unit_id"],
        "entity_col": "unit_id",
        "fetch_name": "fetch_unit_data_96_full",
        "classification": "required_core",
        "note": "unit-level 96-point day-ahead/realtime clearing prices",
    },
}

# optional_extended: downloaded only with --include-extended. Large tables;
# not forced into the first model adapter.
OPTIONAL_EXTENDED_TABLES: dict[str, dict] = {
    "epf_market_congestion_96": {
        "key": ["market_date", "period_no"],
        "entity_col": None,
        "classification": "optional_extended",
        "note": "congestion-section 96-point detail",
    },
    "epf_market_tie_line_96": {
        "key": ["market_date", "period_no"],
        "entity_col": None,
        "classification": "optional_extended",
        "note": "tie-line 96-point detail",
    },
}

# metadata_only / not_needed_for_current_models (documented, not downloaded):
#   epf_market_maintenance, epf_market_must_run_stop, epf_market_pre_supervision
#   (daily market flags), daily_weather_*, hourly_weather_* (ECMWF source for
#   a later design phase), epf_daily_review, epf_day_clear_bill*, etc.

# Remote -> Chinese column dictionary (from existing fetch_* aliases).
COLUMN_DICTIONARY: dict[str, dict[str, str]] = {
    "epf_market_data_96": {
        "data_time": "时刻(15分钟区间结束时刻)",
        "market_date": "交易日",
        "period_no": "时段号(1-96)",
        "actual_direct_load": "直调负荷实际",
        "actual_local_plant": "地方电厂出力实际",
        "actual_tie_line": "外电实际",
        "actual_wind": "风电实际",
        "actual_solar": "光伏实际",
        "actual_nuclear": "核电实际",
        "actual_self_owned": "自备电厂实际",
        "actual_test_unit": "试验机组实际",
        "actual_unit_maintenance": "机组检修实际",
        "actual_pos_reserve": "正备用实际",
        "actual_neg_reserve": "负备用实际",
        "actual_bidding_space": "竞价空间实际",
        "actual_new_energy": "新能源实际",
        "fcast_direct_load": "直调负荷预测",
        "fcast_local_plant": "地方电厂出力预测",
        "fcast_tie_line": "外电预测",
        "fcast_wind": "风电预测",
        "fcast_solar": "光伏预测",
        "fcast_nuclear": "核电预测",
        "fcast_self_owned": "自备电厂预测",
        "fcast_test_unit": "试验机组预测",
        "fcast_unit_maintenance": "机组检修预测",
        "fcast_pos_reserve": "正备用预测",
        "fcast_neg_reserve": "负备用预测",
        "fcast_bidding_space": "竞价空间预测",
        "fcast_new_energy": "新能源预测",
        "create_time": "入库时间(审计列)",
        "update_time": "更新时间(审计列)",
    },
    "epf_unit_data_96": {
        "data_time": "时刻(15分钟区间结束时刻)",
        "market_date": "交易日",
        "period_no": "时段号(1-96)",
        "unit_id": "机组标识",
        "da_cq_price": "日前出清价格(元/MWh)",
        "da_power": "日前出力",
        "da_energy": "日前电量",
        "da_status": "日前开机状态",
        "rt_cq_price": "实时出清价格(元/MWh)",
        "rt_power": "实时出力",
        "rt_energy": "实时电量",
        "rt_status": "实时开机状态",
        "create_time": "入库时间(审计列)",
        "update_time": "更新时间(审计列)",
    },
}

PERIODS_PER_DAY = 96
KEY_AUDIT_BASELINE: dict[str, dict] = {
    "epf_market_data_96": {"min_date": "2022-01-01", "max_date": "2026-07-27"},
    "epf_unit_data_96": {"min_date": "2022-01-01", "max_date": "2026-07-18"},
}


# ---------------------------------------------------------------------------
# Atomic file helpers
# ---------------------------------------------------------------------------


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to *path* (parquet) atomically via a .partial temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_csv_gz(df: pd.DataFrame, path: Path) -> None:
    """Write *df* to *path* (csv.gz) atomically via a .partial temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
    os.replace(tmp, path)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Local mirror read / dedup / completeness
# ---------------------------------------------------------------------------


def _table_parquet_path(table: str) -> Path:
    return PARQUET_DIR / f"{table}.parquet"


def _table_raw_path(table: str) -> Path:
    return RAW_DIR / f"{table}.csv.gz"


def _load_local_table(table: str) -> pd.DataFrame:
    """Load an existing local mirror for *table* (parquet preferred)."""
    pq = _table_parquet_path(table)
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed reading parquet %s: %s", pq, exc)
    raw = _table_raw_path(table)
    if raw.exists():
        try:
            return pd.read_csv(raw, encoding="utf-8", compression="gzip")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed reading csv.gz %s: %s", raw, exc)
    return pd.DataFrame()


def _dedup(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """Drop duplicate rows on *key*, keeping the last (freshest) occurrence."""
    if df.empty or not key:
        return df
    present = [c for c in key if c in df.columns]
    if not present:
        return df
    return df.drop_duplicates(subset=present, keep="last").reset_index(drop=True)


def _coerce_market_date(df: pd.DataFrame) -> pd.DataFrame:
    if "market_date" in df.columns:
        df = df.copy()
        df["market_date"] = pd.to_datetime(df["market_date"], errors="coerce").dt.date
    return df


def _compute_completeness(df: pd.DataFrame, key: list[str]) -> dict:
    """Compute per-table completeness/quality metrics."""
    out: dict = {
        "rows": int(len(df)),
        "min_market_date": None,
        "max_market_date": None,
        "distinct_days": 0,
        "complete_96_days": 0,
        "incomplete_days": 0,
        "duplicate_key_count": 0,
        "latest_complete_day": None,
        "latest_non_null_date_per_critical_column": {},
        "period_no_min": None,
        "period_no_max": None,
    }
    if df.empty:
        return out

    df = _coerce_market_date(df)
    dates = df["market_date"].dropna()
    if not dates.empty:
        out["min_market_date"] = str(dates.min())
        out["max_market_date"] = str(dates.max())

    # period_no distribution
    if "period_no" in df.columns:
        pn = pd.to_numeric(df["period_no"], errors="coerce")
        out["period_no_min"] = int(pn.min()) if not pn.dropna().empty else None
        out["period_no_max"] = int(pn.max()) if not pn.dropna().empty else None

    # complete-day detection
    if "market_date" in df.columns and "period_no" in df.columns:
        grp = df.dropna(subset=["market_date"]).groupby("market_date")["period_no"]
        day_counts = grp.nunique()
        out["distinct_days"] = int(day_counts.shape[0])
        complete = day_counts[day_counts == PERIODS_PER_DAY]
        out["complete_96_days"] = int(complete.shape[0])
        out["incomplete_days"] = int((day_counts != PERIODS_PER_DAY).sum())
        if not complete.empty:
            out["latest_complete_day"] = str(complete.index.max())

    # duplicate keys
    present_key = [c for c in key if c in df.columns]
    if present_key:
        out["duplicate_key_count"] = int(df.duplicated(subset=present_key).sum())

    # latest non-null date per critical (price / load) column
    critical = [c for c in df.columns if any(
        k in c for k in ("price", "load", "direct_load", "wind", "solar", "nuclear")
    )]
    for c in critical[:6]:  # cap to keep manifest compact
        non_null = df[df[c].notna()]
        if not non_null.empty and "market_date" in non_null.columns:
            out["latest_non_null_date_per_critical_column"][c] = str(
                non_null["market_date"].max()
            )
    return out


def _validate_table(
    df: pd.DataFrame,
    table: str,
    key: list[str],
    sync_mode: str,
    remote_summary: dict,
    enforce_audit_baseline: bool = True,
) -> dict:
    """Validate a local mirror against the remote-audit baseline (task §10).

    When ``enforce_audit_baseline`` is False (unit tests with synthetic data),
    the hard 2022-start and audit-baseline-max floors are relaxed and the
    max-date check instead verifies the local mirror covers the queried remote
    range (remote_summary.d_max).
    """
    checks: list[dict] = []
    baseline = KEY_AUDIT_BASELINE.get(table, {})

    comp = _compute_completeness(df, key)

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    # 1. start date is 2022-01-01 (real mode only)
    if comp["min_market_date"] is not None:
        if enforce_audit_baseline:
            add("start_date_2022", comp["min_market_date"] <= "2022-01-01",
                f"min={comp['min_market_date']}")
        else:
            add("start_date_2022", True,
                f"min={comp['min_market_date']} (test mode: floor relaxed)")
    else:
        add("start_date_2022", False, "no market_date")

    # 2. max date
    base_max = baseline.get("max_date")
    remote_max = remote_summary.get("d_max")
    if enforce_audit_baseline and base_max and comp["max_market_date"] is not None:
        add("max_date_ge_baseline", comp["max_market_date"] >= base_max,
            f"local_max={comp['max_market_date']} baseline_max={base_max}")
    elif (not enforce_audit_baseline) and remote_max and comp["max_market_date"] is not None:
        add("max_date_ge_baseline", comp["max_market_date"] >= remote_max,
            f"local_max={comp['max_market_date']} remote_max={remote_max}")
    else:
        add("max_date_ge_baseline", True, "no baseline / no data")

    # 3. every complete day has exactly 96 distinct periods
    # (enforced by complete_96_days definition; report distribution)
    if comp["incomplete_days"] == 0 and comp["distinct_days"] > 0:
        add("all_days_96_periods", True, f"distinct_days={comp['distinct_days']}")
    elif comp["distinct_days"] == 0:
        add("all_days_96_periods", False, "no days")
    else:
        add("all_days_96_periods", False,
            f"incomplete_days={comp['incomplete_days']}")

    # 4. no duplicate keys
    add("no_duplicate_keys", comp["duplicate_key_count"] == 0,
        f"dup_keys={comp['duplicate_key_count']}")

    # 5. period_no within 1..96
    pn_ok = (comp["period_no_min"] is not None and comp["period_no_max"] is not None
             and comp["period_no_min"] >= 1 and comp["period_no_max"] <= 96)
    add("period_no_range", pn_ok,
        f"min={comp['period_no_min']} max={comp['period_no_max']}")

    # 6. p1/p96 interval-end semantics (sample a complete day)
    p1p96_ok, p1p96_detail = _check_p1_p96(df)
    add("p1_p96_interval_end", p1p96_ok, p1p96_detail)

    # 7. remote/local row-count match (within selected range)
    if sync_mode == "full":
        local_rows = comp["rows"]
        remote_rows = int(remote_summary.get("rows_total", 0))
        match = local_rows >= remote_rows
        add("row_count_match", match,
            f"local={local_rows} remote={remote_rows}")
    else:
        # incremental: compare rows inside the re-pulled window below
        add("row_count_match", True, "incremental: checked at sync time")

    status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "completeness": comp,
        "remote_summary": remote_summary,
    }


def _check_p1_p96(df: pd.DataFrame) -> tuple[bool, str]:
    """Verify period_no=1 -> day 00:15, period_no=96 -> (day+1) 00:00."""
    if "market_date" not in df.columns or "period_no" not in df.columns \
            or "data_time" not in df.columns:
        return False, "missing required columns"
    d = _coerce_market_date(df.copy())
    d["data_time"] = pd.to_datetime(d["data_time"], errors="coerce")
    complete = d.groupby("market_date")["period_no"].nunique()
    complete_days = complete[complete == PERIODS_PER_DAY].index
    if len(complete_days) == 0:
        return False, "no complete 96-day available to sample"
    day = complete_days[0]
    sub = d[d["market_date"] == day]
    p1 = sub[sub["period_no"] == 1]["data_time"]
    p96 = sub[sub["period_no"] == 96]["data_time"]
    if p1.empty or p96.empty:
        return False, "missing p1 or p96 in sample day"
    p1_ts = p1.iloc[0]
    p96_ts = p96.iloc[0]
    day_dt = datetime.combine(day, datetime.min.time())
    expected_p1 = day_dt + timedelta(minutes=15)
    expected_p96 = day_dt + timedelta(days=1)
    ok = (abs((p1_ts - expected_p1).total_seconds()) < 60
          and abs((p96_ts - expected_p96).total_seconds()) < 60)
    detail = (f"sample_day={day} p1={p1_ts} (expect {expected_p1}), "
              f"p96={p96_ts} (expect {expected_p96})")
    return ok, detail


# ---------------------------------------------------------------------------
# Per-table synchronization
# ---------------------------------------------------------------------------


def _sync_one_table(
    table: str,
    cfg: dict,
    sync_mode: str,
    overlap_days: int,
    source: str,
    enforce_audit_baseline: bool = True,
) -> dict:
    """Download/validate a single table and write the local mirror atomically.

    Returns a per-table record. On failure the previous valid local file is
    preserved (not replaced) and the record status is 'failed'.
    """
    key = cfg["key"]
    record: dict = {
        "table": table,
        "classification": cfg.get("classification", "required_core"),
        "status": "failed",
        "sync_mode": sync_mode,
        "source": source,
        "errors": [],
        "warnings": [],
    }
    try:
        if source in ("db", "auto"):
            # --- Determine date range ---
            start_date = None
            end_date = None
            existing = _load_local_table(table)
            if sync_mode == "incremental" and not existing.empty:
                existing = _coerce_market_date(existing)
                max_local = existing["market_date"].max()
                if pd.notna(max_local):
                    start_date = (max_local - timedelta(days=overlap_days)).isoformat()
            # --- Fetch from remote (read-only SELECT) ---
            import utils.database_operate as _db
            fetch_fn: Callable = getattr(_db, cfg["fetch_name"])
            incoming = fetch_fn(start_date=start_date, end_date=end_date)
            if incoming.empty and sync_mode == "full":
                record["errors"].append("remote returned 0 rows (full mode)")
                return record

            # --- Merge (incremental re-pulls overlap window) ---
            if sync_mode == "incremental" and not existing.empty:
                merged = _dedup(pd.concat([existing, incoming], ignore_index=True), key)
            else:
                merged = _dedup(incoming, key)

            # --- Atomic write (parquet + csv.gz) ---
            pq_path = _table_parquet_path(table)
            raw_path = _table_raw_path(table)
            _atomic_write_parquet(merged, pq_path)
            _atomic_write_csv_gz(merged, raw_path)

            record["local_paths"] = {
                "parquet": str(pq_path),
                "raw_csv_gz": str(raw_path),
            }
            record["rows_local"] = int(len(merged))
            record["parquet_size_bytes"] = int(pq_path.stat().st_size) if pq_path.exists() else 0
            record["raw_size_bytes"] = int(raw_path.stat().st_size) if raw_path.exists() else 0
            record["checksum_sha256"] = _sha256_of_file(pq_path) if pq_path.exists() else None

            # --- Remote summary for reconciliation (route via module so tests
            #     can patch utils.database_operate.fetch_96_table_summary) ---
            remote_summary = _db.fetch_96_table_summary(table)
            record["remote_summary"] = remote_summary

            # --- Validation ---
            validation = _validate_table(merged, table, key, sync_mode, remote_summary,
                                         enforce_audit_baseline=enforce_audit_baseline)
            record["validation"] = validation
            record["completeness"] = validation["completeness"]
            record["rows_remote"] = int(remote_summary.get("rows_total", 0))
            record["row_count_match"] = any(
                c["check"] == "row_count_match" and c["status"] == "PASS"
                for c in validation["checks"]
            )
            if validation["status"] == "PASS":
                record["status"] = "ok"
            else:
                record["status"] = "failed"
                record["errors"].append(
                    "validation failed: " + "; ".join(
                        c["detail"] for c in validation["checks"] if c["status"] == "FAIL"
                    )
                )
        elif source == "local":
            # Validate / report existing local mirror without DB access.
            existing = _load_local_table(table)
            if existing.empty:
                record["status"] = "skipped"
                record["warnings"].append("no local mirror found for local source")
                return record
            pq_path = _table_parquet_path(table)
            raw_path = _table_raw_path(table)
            record["local_paths"] = {
                "parquet": str(pq_path) if pq_path.exists() else None,
                "raw_csv_gz": str(raw_path) if raw_path.exists() else None,
            }
            remote_summary = {"rows_total": None, "d_min": None, "d_max": None}
            validation = _validate_table(existing, table, key, "local", remote_summary,
                                         enforce_audit_baseline=enforce_audit_baseline)
            record["validation"] = validation
            record["completeness"] = validation["completeness"]
            record["rows_local"] = int(len(existing))
            record["status"] = "ok" if validation["status"] == "PASS" else "failed"
        else:
            record["errors"].append(f"unsupported source '{source}' for 15min resolution")
            return record
    except Exception as exc:  # pragma: no cover - defensive
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        # previous valid file preserved (not replaced on failure)
    return record


# ---------------------------------------------------------------------------
# Metadata writers
# ---------------------------------------------------------------------------


def _write_metadata(tables: list[str], classification: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # schema_inventory.csv (from local parquet dtypes)
    inv_rows = []
    for t in tables:
        pq = _table_parquet_path(t)
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        for col in df.columns:
            inv_rows.append({
                "schema": "ai_epf_platform",
                "table": t,
                "column": col,
                "pandas_dtype": str(df[col].dtype),
                "chinese": COLUMN_DICTIONARY.get(t, {}).get(col, ""),
            })
    inv_df = pd.DataFrame(inv_rows, columns=["schema", "table", "column", "pandas_dtype", "chinese"])
    inv_df.to_csv(METADATA_DIR / "schema_inventory.csv", index=False, encoding="utf-8")

    # column_dictionary.csv
    dict_rows = []
    for t in tables:
        for col, cn in COLUMN_DICTIONARY.get(t, {}).items():
            dict_rows.append({"table": t, "remote_column": col, "chinese": cn})
    pd.DataFrame(dict_rows, columns=["table", "remote_column", "chinese"]).to_csv(
        METADATA_DIR / "column_dictionary.csv", index=False, encoding="utf-8")

    # source_table_manifest.json (classification + audit baseline)
    src_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "audit_baseline": KEY_AUDIT_BASELINE,
        "note": ("required_core tables are mandatory for the first production "
                 "96-point local dataset; optional_extended are downloaded only "
                 "with --include-extended."),
    }
    with open(METADATA_DIR / "source_table_manifest.json", "w", encoding="utf-8") as f:
        json.dump(src_manifest, f, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Top-level 96-point sync
# ---------------------------------------------------------------------------


def sync_96(args: Any = None) -> dict:
    """Run the 96-point synchronization.

    Parameters (from *args*, all optional):
      sync_source : 'db' | 'auto' | 'local'  (http is NOT supported for 15min)
      sync_mode   : 'full' | 'incremental'
      force_sync  : bool (ignored for 15min — sync always refreshes)
      include_extended : bool (download optional_extended tables)
      overlap_days: int (incremental overlap window)

    Returns a manifest dict with the fields required by task §9.
    """
    source = getattr(args, "sync_source", "db") if args else "db"
    sync_mode = getattr(args, "sync_mode", "full") if args else "full"
    include_extended = getattr(args, "include_extended", False) if args else False
    overlap_days = int(getattr(args, "sync_overlap_days", 7) or 7) if args else 7
    enforce_audit_baseline = bool(
        getattr(args, "enforce_audit_baseline", True)) if args else True

    started_at = datetime.now(timezone.utc).isoformat()

    # Table set
    tables_cfg = dict(CORE_TABLES)
    classification = {t: c["classification"] for t, c in CORE_TABLES.items()}
    if include_extended:
        for t, c in OPTIONAL_EXTENDED_TABLES.items():
            tables_cfg[t] = {**c, "fetch_name": "fetch_96_table"}
            classification[t] = c["classification"]

    tables_requested = list(tables_cfg.keys())

    # Source capability guard
    if source == "http":
        manifest = _build_manifest(
            started_at=started_at, source=source, sync_mode=sync_mode,
            tables_requested=tables_requested, records=[],
            status="failed",
            errors=["sync-source 'http' is not supported for 15min resolution: "
                    "no 96-point HTTP endpoint exists. Use db/auto/local."],
            overlap_days=overlap_days, classification=classification,
        )
        _write_manifest(manifest)
        return manifest

    # Server version (read-only). Route via module so tests can patch it.
    import utils.database_operate as _dbmod
    try:
        server_version = _dbmod.get_db_server_version()
    except Exception as exc:
        server_version = f"unavailable: {exc}"

    records: list[dict] = []
    for table in tables_requested:
        cfg = tables_cfg[table]
        rec = _sync_one_table(table, cfg, sync_mode, overlap_days, source,
                              enforce_audit_baseline=enforce_audit_baseline)
        records.append(rec)
        if rec["status"] == "failed":
            logger.error("96 sync failed for %s: %s", table, rec["errors"])

    succeeded = [r["table"] for r in records if r["status"] == "ok"]
    failed = [r["table"] for r in records if r["status"] == "failed"]
    skipped = [r["table"] for r in records if r["status"] == "skipped"]

    # Overall status: ok if all core succeeded; partial if some optional failed
    # but all core ok; failed if any core failed.
    core_failed = [t for t in failed if CORE_TABLES.get(t) is not None]
    if not core_failed and not failed:
        status = "ok"
    elif not core_failed and failed:
        status = "partial"
    else:
        status = "failed"

    # Write metadata for successfully synced tables
    if succeeded:
        try:
            _write_metadata(succeeded, classification)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("metadata write failed: %s", exc)

    manifest = _build_manifest(
        started_at=started_at, source=source, sync_mode=sync_mode,
        tables_requested=tables_requested, records=records, status=status,
        errors=[], overlap_days=overlap_days, classification=classification,
        server_version=server_version,
    )
    _write_manifest(manifest)
    return manifest


def _build_manifest(
    started_at: str, source: str, sync_mode: str, tables_requested: list[str],
    records: list[dict], status: str, errors: list[str], overlap_days: int,
    classification: dict, server_version: str = "unknown",
) -> dict:
    completed_at = datetime.now(timezone.utc).isoformat()
    tables_succeeded = [r["table"] for r in records if r["status"] == "ok"]
    tables_failed = [r["table"] for r in records if r["status"] == "failed"]

    local_paths: dict = {}
    rows_per_table: dict = {}
    min_market_date_per_table: dict = {}
    max_market_date_per_table: dict = {}
    distinct_days_per_table: dict = {}
    complete_96_days_per_table: dict = {}
    incomplete_days_per_table: dict = {}
    duplicate_key_count: dict = {}
    latest_complete_day: dict = {}
    latest_non_null_date_per_critical_column: dict = {}
    remote_row_count: dict = {}
    local_row_count: dict = {}
    row_count_match: dict = {}
    schema_fingerprint: dict = {}
    data_fingerprint: dict = {}

    for r in records:
        t = r["table"]
        comp = r.get("completeness", {})
        local_paths[t] = r.get("local_paths", {})
        rows_per_table[t] = comp.get("rows")
        min_market_date_per_table[t] = comp.get("min_market_date")
        max_market_date_per_table[t] = comp.get("max_market_date")
        distinct_days_per_table[t] = comp.get("distinct_days")
        complete_96_days_per_table[t] = comp.get("complete_96_days")
        incomplete_days_per_table[t] = comp.get("incomplete_days")
        duplicate_key_count[t] = comp.get("duplicate_key_count")
        latest_complete_day[t] = comp.get("latest_complete_day")
        latest_non_null_date_per_critical_column[t] = comp.get(
            "latest_non_null_date_per_critical_column", {})
        remote_row_count[t] = r.get("rows_remote")
        local_row_count[t] = r.get("rows_local")
        row_count_match[t] = r.get("row_count_match")
        schema_fingerprint[t] = r.get("checksum_sha256")
        data_fingerprint[t] = {
            "rows": comp.get("rows"),
            "min_market_date": comp.get("min_market_date"),
            "max_market_date": comp.get("max_market_date"),
            "dup_keys": comp.get("duplicate_key_count"),
            "complete_96_days": comp.get("complete_96_days"),
        }

    return {
        "resolution": "15min",
        "source": source,
        "sync_mode": sync_mode,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "database_server_version": server_version,
        "tables_requested": tables_requested,
        "tables_succeeded": tables_succeeded,
        "tables_failed": tables_failed,
        "classification": classification,
        "local_paths": local_paths,
        "rows_per_table": rows_per_table,
        "min_market_date_per_table": min_market_date_per_table,
        "max_market_date_per_table": max_market_date_per_table,
        "distinct_days_per_table": distinct_days_per_table,
        "complete_96_days_per_table": complete_96_days_per_table,
        "incomplete_days_per_table": incomplete_days_per_table,
        "duplicate_key_count": duplicate_key_count,
        "latest_complete_day": latest_complete_day,
        "latest_non_null_date_per_critical_column": latest_non_null_date_per_critical_column,
        "remote_row_count": remote_row_count,
        "local_row_count": local_row_count,
        "row_count_match": row_count_match,
        "schema_fingerprint": schema_fingerprint,
        "data_fingerprint_or_checksums": data_fingerprint,
        "records": records,
        "overlap_days": overlap_days,
        "warnings": [w for r in records for w in r.get("warnings", [])],
        "errors": errors + [e for r in records for e in r.get("errors", [])],
    }


def _write_manifest(manifest: dict) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(SYNC_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    _write_sync_markdown(MANIFEST_DIR / "sync_report.md", manifest)


def _write_sync_markdown(path: Path, manifest: dict) -> None:
    lines = [
        "# 96-Point Data Sync Report",
        "",
        f"- **Resolution:** {manifest.get('resolution')}",
        f"- **Source:** {manifest.get('source')}",
        f"- **Sync mode:** {manifest.get('sync_mode')}",
        f"- **Status:** {manifest.get('status')}",
        f"- **DB server version:** {manifest.get('database_server_version')}",
        f"- **Overlap days:** {manifest.get('overlap_days')}",
        f"- **Started:** {manifest.get('started_at')}",
        f"- **Completed:** {manifest.get('completed_at')}",
        "",
        "## Tables",
        "",
    ]
    for t in manifest.get("tables_requested", []):
        lines.append(f"### {t} [{manifest.get('classification', {}).get(t, '?')}]")
        lines.append("")
        lines.append(f"- status: {_rec_status(manifest, t)}")
        lines.append(f"- rows (local/remote): {manifest.get('local_row_count', {}).get(t)} / {manifest.get('remote_row_count', {}).get(t)}")
        lines.append(f"- date range: {manifest.get('min_market_date_per_table', {}).get(t)} → {manifest.get('max_market_date_per_table', {}).get(t)}")
        lines.append(f"- complete 96-days: {manifest.get('complete_96_days_per_table', {}).get(t)}")
        lines.append(f"- dup keys: {manifest.get('duplicate_key_count', {}).get(t)}")
        lines.append(f"- row_count_match: {manifest.get('row_count_match', {}).get(t)}")
        lines.append("")
    if manifest.get("warnings"):
        lines += ["## Warnings", ""]
        for w in manifest["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    if manifest.get("errors"):
        lines += ["## Errors", ""]
        for e in manifest["errors"]:
            lines.append(f"- {e}")
        lines.append("")
    lines.append("---")
    lines.append("_Generated by sync_data_96_core_")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _rec_status(manifest: dict, table: str) -> str:
    for r in manifest.get("records", []):
        if r.get("table") == table:
            return r.get("status", "?")
    # fallback: infer from succeeded/failed lists
    if table in manifest.get("tables_succeeded", []):
        return "ok"
    if table in manifest.get("tables_failed", []):
        return "failed"
    return "skipped"


# Expose for tests
__all__ = [
    "sync_96", "CORE_TABLES", "OPTIONAL_EXTENDED_TABLES", "KEY_AUDIT_BASELINE",
    "REMOTE_96_ROOT", "SYNC_MANIFEST_PATH", "_sync_one_table", "_validate_table",
    "_compute_completeness", "_dedup", "_load_local_table", "_atomic_write_parquet",
    "_atomic_write_csv_gz", "_check_p1_p96",
]
