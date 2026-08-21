"""
Data synchronization: DB, HTTP cloud, and local file fallback.

Main entry point: sync_dataset()
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import pandas as pd
from dotenv import load_dotenv

from utils.database_operate import fetch_web_grid_data
from utils.data_layout import DATA

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 24-point domain.  Downloads are staged under reference/; only the two
# canonical files are consumed by production defaults.
DATA_DIR = DATA.hourly_root / "reference"
SYNC_MANIFEST_DIR = DATA.sync_24_root
CANONICAL_XLSX = DATA.hourly_root / "canonical" / "shandong_pmos_hourly.xlsx"
CANONICAL_CSV = DATA.hourly_root / "canonical" / "shandong_pmos_hourly.csv"
BASE_URL = "http://qiniu.dirx.com.cn/workspace/eprice_forecast"
TIMESTAMP_COL = "时刻"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_timestamp_from_excel(path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_excel(path, usecols=[TIMESTAMP_COL])
    except Exception:
        return None
    series = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    if series.empty:
        return None
    return series.max()


def _max_timestamp_from_csv(path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(path, usecols=[TIMESTAMP_COL], encoding="gbk")
    except Exception:
        try:
            df = pd.read_csv(path, usecols=[TIMESTAMP_COL], encoding="utf-8-sig")
        except Exception:
            return None
    series = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    if series.empty:
        return None
    return series.max()


def _download_latest_available_excel(max_lookback_days: int = 60) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    latest_attempted_url = None
    for delta in range(max_lookback_days):
        current_day = date.today() - timedelta(days=delta)
        filename = f"shandong_pmos_hourly_20220101_{current_day:%Y%m%d}.xlsx"
        file_url = f"{BASE_URL}/{filename}"
        latest_attempted_url = file_url
        local_file = DATA_DIR / filename
        if local_file.exists():
            return local_file
        try:
            urlretrieve(file_url, str(local_file))
            return local_file
        except HTTPError as exc:
            if getattr(exc, "code", None) == 404:
                continue
            raise
        except URLError:
            raise
    raise FileNotFoundError(
        f"No downloadable dataset found from cloud source. "
        f"Last attempted: {latest_attempted_url}"
    )


def _candidate_local_excel_files() -> list[Path]:
    candidates = list(DATA_DIR.glob("shandong_pmos_hourly_20220101_*.xlsx"))
    if CANONICAL_XLSX.exists():
        candidates.append(CANONICAL_XLSX)
    return sorted(set(candidates), reverse=True)


def _latest_local_dataset_source() -> tuple[Path, str]:
    csv_ts = _max_timestamp_from_csv(CANONICAL_CSV) if CANONICAL_CSV.exists() else None
    xlsx_candidates = _candidate_local_excel_files()

    best_xlsx: Path | None = None
    best_xlsx_ts: pd.Timestamp | None = None
    for candidate in xlsx_candidates:
        ts = _max_timestamp_from_excel(candidate)
        if ts is None:
            continue
        if best_xlsx_ts is None or ts > best_xlsx_ts:
            best_xlsx = candidate
            best_xlsx_ts = ts

    if csv_ts is not None and (best_xlsx_ts is None or csv_ts >= best_xlsx_ts):
        return CANONICAL_CSV, "csv"
    if best_xlsx is not None:
        return best_xlsx, "xlsx"
    raise FileNotFoundError(
        f"No readable local dataset candidates found under: {DATA_DIR}"
    )


def _save_frame(df: pd.DataFrame, xlsx_path: Path, csv_path: Path) -> str:
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(xlsx_path, index=False)
    # Always try gbk first for Chinese character compatibility
    try:
        df.to_csv(csv_path, index=False, encoding="gbk")
    except Exception:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return str(xlsx_path)


# ---------------------------------------------------------------------------
# Per-source sync implementations
# ---------------------------------------------------------------------------


def _sync_from_database() -> pd.DataFrame:
    """Fetch data from the MySQL database. Raises on failure."""
    data = fetch_web_grid_data()
    if data is None or data.empty:
        raise ValueError("Database returned empty dataset")
    return data


def _sync_from_http() -> pd.DataFrame:
    """Download the latest available .xlsx from the cloud. Raises on failure."""
    source_file = _download_latest_available_excel()
    df = pd.read_excel(source_file)
    if df.empty:
        raise ValueError("HTTP download produced empty dataset")
    return df


def _sync_from_local(canonical_xlsx: Path = CANONICAL_XLSX) -> pd.DataFrame:
    """Use the latest local dataset. Raises on failure.

    If *canonical_xlsx* already exists (e.g. from a previous write or a
    custom ``data_path``), read it directly.  Otherwise fall back to the
    ``DATA_DIR`` file-discovery logic.
    """
    if canonical_xlsx.exists():
        df = pd.read_excel(canonical_xlsx)
    else:
        local_source, source_kind = _latest_local_dataset_source()
        if source_kind == "csv":
            df = pd.read_csv(local_source, encoding="gbk")
        else:
            df = pd.read_excel(local_source)
    if df.empty:
        raise ValueError("Local dataset is empty")
    return df


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

REQUIRED_SYNC_COLUMNS = ["时刻", "日前电价", "实时电价"]


def validate_synced_dataset(
    df: pd.DataFrame,
    target_date: str | None = None,
    max_data_lag_hours: int = 36,
) -> dict:
    """Validate a synced dataset for structural integrity and freshness.

    Checks
    ------
    1. Required columns exist.
    2. ``时刻`` is parseable as datetime.
    3. Timestamps are sorted ascending.
    4. Duplicate timestamps are deduplicated (keep last).
    5. Row count > 0.
    6. ``日前电价`` and ``实时电价`` are numeric.
    7. If *target_date* is given: the latest timestamp must be no older
       than *max_data_lag_hours* before the target date's decision time.

    Returns
    -------
    dict with keys: status, warnings, errors, min_timestamp, max_timestamp,
    rows, columns.
    """
    result: dict = {
        "status": "ok",
        "warnings": [],
        "errors": [],
        "rows": 0,
        "columns": [],
        "min_timestamp": None,
        "max_timestamp": None,
    }

    # 1. Required columns
    missing = [c for c in REQUIRED_SYNC_COLUMNS if c not in df.columns]
    if missing:
        result["status"] = "failed"
        result["errors"].append(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )
        return result

    result["columns"] = list(df.columns)

    # 2. Parse timestamps
    ts_series = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    bad_ts = ts_series.isna().sum()
    if bad_ts > 0:
        result["errors"].append(f"{bad_ts} unparseable timestamps in column '{TIMESTAMP_COL}'")
        result["status"] = "failed"
        return result

    result["min_timestamp"] = str(ts_series.min())
    result["max_timestamp"] = str(ts_series.max())
    result["rows"] = len(df)

    # 3. Sort and deduplicate (handled here so callers get clean data)
    df_sorted = df.copy()
    df_sorted[TIMESTAMP_COL] = ts_series
    df_sorted = df_sorted.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    # 4. Duplicate timestamps
    before_dedup = len(df_sorted)
    df_sorted = df_sorted.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    after_dedup = len(df_sorted)
    if after_dedup < before_dedup:
        result["warnings"].append(
            f"Removed {before_dedup - after_dedup} duplicate timestamp(s) "
            f"(kept last)"
        )
    result["rows"] = after_dedup

    if after_dedup == 0:
        result["status"] = "failed"
        result["errors"].append("Dataset is empty after timestamp deduplication")
        return result

    # 5. Numeric price columns
    for col in ["日前电价", "实时电价"]:
        non_numeric = pd.to_numeric(df_sorted[col], errors="coerce").isna().sum()
        if non_numeric > 0:
            result["warnings"].append(
                f"Column '{col}' contains {non_numeric} non-numeric value(s)"
            )

    # 6. Freshness check
    if target_date is not None:
        freshness_result = _check_freshness(
            ts_series.max(), target_date, max_data_lag_hours
        )
        result["freshness"] = freshness_result
        if freshness_result.get("status") == "FAIL":
            result["errors"].append(freshness_result.get("detail", "Freshness check failed"))
            result["status"] = "failed"

    return result


def _check_freshness(
    latest_ts: pd.Timestamp,
    target_date: str,
    max_data_lag_hours: int,
) -> dict:
    """Check that the latest data timestamp is recent enough for *target_date*.

    The target's decision moment is ``target_date 00:00:00`` (start of D).
    If *latest_ts* is more than *max_data_lag_hours* before that moment,
    the data is considered stale.
    """
    target_dt = pd.Timestamp(target_date)
    cutoff = target_dt - pd.Timedelta(hours=max_data_lag_hours)
    is_fresh = latest_ts >= cutoff
    lag_hours = round((target_dt - latest_ts).total_seconds() / 3600, 1)

    result = {
        "target_date": target_date,
        "latest_data_timestamp": str(latest_ts),
        "target_decision_time": str(target_dt),
        "max_data_lag_hours": max_data_lag_hours,
        "actual_lag_hours": lag_hours,
        "status": "PASS" if is_fresh else "FAIL",
    }

    if not is_fresh:
        result["detail"] = (
            f"Latest data timestamp {latest_ts} is {lag_hours}h "
            f"before target date {target_date}, exceeding "
            f"max lag of {max_data_lag_hours}h"
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sync_dataset(
    data_path: str | Path | None = None,
    source: str = "auto",
    force: bool = False,
    require_fresh: bool = False,
    target_date: str | None = None,
    max_data_lag_hours: int = 36,
) -> dict:
    """Synchronise the canonical dataset from one of several sources.

    Parameters
    ----------
    data_path : str | Path, optional
        If given, use this path as the canonical xlsx location instead of
        ``data/shandong_pmos_hourly.xlsx``.
    source : str
        ``"auto"`` (default) — try db, then http, then local.
        ``"db"`` — database only; no fallback.
        ``"http"`` — HTTP cloud download only.
        ``"local"`` — local files only.
    force : bool
        If False (default), skip sync when the canonical xlsx already exists
        and is non-empty (and no source is specified).  If True, always
        re-sync.
    require_fresh : bool
        If True and *target_date* is given, fail if the data is not fresh
        enough.  Only meaningful together with a non-None *target_date*.
    target_date : str, optional
        Target forecast date ``YYYY-MM-DD`` used for freshness checks.
    max_data_lag_hours : int
        Maximum allowed lag between the target date's decision moment and
        the latest available data timestamp (default 36).

    Returns
    -------
    dict — a manifest with keys: status, source, output_xlsx, output_csv,
    rows, columns, min_timestamp, max_timestamp, freshness (optional),
    warnings, errors.
    """
    # Resolve the canonical xlsx path
    if data_path is not None:
        canonical_xlsx = Path(data_path)
        canonical_csv = canonical_xlsx.with_suffix(".csv")
        data_dir = canonical_xlsx.parent
    else:
        canonical_xlsx = CANONICAL_XLSX
        canonical_csv = CANONICAL_CSV
        data_dir = DATA_DIR

    # When source is "auto" and no explicit source is needed, and force is
    # False, and the file already exists, skip the sync entirely — unless
    # require_fresh asks us to validate freshness first.
    if (
        source == "auto"
        and not force
        and canonical_xlsx.exists()
        and canonical_xlsx.stat().st_size > 0
    ):
        df = pd.read_excel(canonical_xlsx)
        validation = validate_synced_dataset(
            df,
            target_date=target_date if require_fresh else None,
            max_data_lag_hours=max_data_lag_hours,
        )
        freshness = validation.get("freshness")

        if require_fresh and target_date is not None and freshness and freshness.get("status") == "FAIL":
            manifest = {
                "status": "failed",
                "source": "local",
                "output_xlsx": str(canonical_xlsx),
                "output_csv": str(canonical_csv),
                "rows": validation.get("rows", 0),
                "columns": validation.get("columns", []),
                "min_timestamp": validation.get("min_timestamp"),
                "max_timestamp": validation.get("max_timestamp"),
                "freshness": freshness,
                "warnings": validation.get("warnings", []),
                "errors": validation.get("errors", []),
            }
            _write_sync_manifest(manifest)
            return manifest

        manifest = {
            "status": "skipped",
            "source": "local",
            "output_xlsx": str(canonical_xlsx),
            "output_csv": str(canonical_csv),
            "rows": len(df),
            "columns": list(df.columns),
            "min_timestamp": validation.get("min_timestamp"),
            "max_timestamp": validation.get("max_timestamp"),
            "freshness": freshness,
            "warnings": [],
            "errors": [],
            "message": "Canonical dataset already exists; use --force-sync to refresh",
        }
        _write_sync_manifest(manifest)
        return manifest

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # Source selection
    source_chain: list[tuple[str, str, callable]] = []
    if source == "db":
        source_chain = [("db", "database", _sync_from_database)]
    elif source == "http":
        source_chain = [("http", "http", _sync_from_http)]
    elif source == "local":
        source_chain = [("local", "local", lambda: _sync_from_local(canonical_xlsx))]
    elif source == "auto":
        source_chain = [
            ("db", "database", _sync_from_database),
            ("http", "http", _sync_from_http),
            ("local", "local", lambda: _sync_from_local(canonical_xlsx)),
        ]
    else:
        return {
            "status": "failed",
            "source": source,
            "errors": [f"Unknown sync source: '{source}'. Use 'auto', 'db', 'http', or 'local'."],
            "warnings": [],
        }

    last_error = None
    final_df: pd.DataFrame | None = None
    used_source: str = "unknown"

    for src_key, src_label, sync_fn in source_chain:
        try:
            final_df = sync_fn()
            used_source = src_key
            break
        except Exception as exc:
            last_error = exc
            # In non-auto mode, don't fall through
            if source != "auto":
                break
            continue

    if final_df is None:
        err_msg = f"sync_dataset({source}): all sources exhausted."
        if last_error:
            err_msg += f" Last error: {last_error}"
        manifest = {
            "status": "failed",
            "source": source,
            "output_xlsx": str(canonical_xlsx),
            "output_csv": str(canonical_csv),
            "rows": 0,
            "columns": [],
            "min_timestamp": None,
            "max_timestamp": None,
            "freshness": None,
            "warnings": [],
            "errors": [err_msg],
        }
        _write_sync_manifest(manifest)
        return manifest

    # Validate
    validation = validate_synced_dataset(
        final_df,
        target_date=target_date if require_fresh or target_date else None,
        max_data_lag_hours=max_data_lag_hours,
    )

    if validation["status"] == "failed":
        # Structural failure — still save the file but report failure
        _save_frame(final_df, canonical_xlsx, canonical_csv)
        manifest = {
            "status": "failed",
            "source": used_source,
            "output_xlsx": str(canonical_xlsx),
            "output_csv": str(canonical_csv),
            "rows": validation["rows"],
            "columns": validation["columns"],
            "min_timestamp": validation.get("min_timestamp"),
            "max_timestamp": validation.get("max_timestamp"),
            "freshness": validation.get("freshness"),
            "warnings": validation.get("warnings", []),
            "errors": validation.get("errors", []),
        }
        _write_sync_manifest(manifest)
        return manifest

    # Save
    _save_frame(final_df, canonical_xlsx, canonical_csv)

    manifest = {
        "status": "ok",
        "source": used_source,
        "output_xlsx": str(canonical_xlsx),
        "output_csv": str(canonical_csv),
        "rows": validation["rows"],
        "columns": validation["columns"],
        "min_timestamp": validation.get("min_timestamp"),
        "max_timestamp": validation.get("max_timestamp"),
        "freshness": validation.get("freshness"),
        "warnings": validation.get("warnings", []),
        "errors": [],
    }
    _write_sync_manifest(manifest)
    return manifest


def _write_sync_manifest(manifest: dict) -> None:
    """Write sync manifest and markdown report to outputs/data_sync/."""
    SYNC_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # Add metadata
    manifest["synced_at"] = datetime.now(timezone.utc).isoformat()

    json_path = SYNC_MANIFEST_DIR / "sync_manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    md_path = SYNC_MANIFEST_DIR / "sync_report.md"
    _write_sync_markdown(md_path, manifest)


def _write_sync_markdown(path: Path, manifest: dict) -> None:
    """Write a human-readable sync report."""
    lines = [
        "# Data Sync Report",
        "",
        f"- **Status:** {manifest.get('status', 'unknown')}",
        f"- **Source:** {manifest.get('source', 'unknown')}",
        f"- **Synced at:** {manifest.get('synced_at', 'unknown')}",
        f"- **Output XLSX:** `{manifest.get('output_xlsx', 'N/A')}`",
        f"- **Output CSV:** `{manifest.get('output_csv', 'N/A')}`",
        f"- **Rows:** {manifest.get('rows', 0)}",
        f"- **Min timestamp:** {manifest.get('min_timestamp', 'N/A')}",
        f"- **Max timestamp:** {manifest.get('max_timestamp', 'N/A')}",
        "",
    ]

    freshness = manifest.get("freshness")
    if freshness:
        lines.extend([
            "## Freshness",
            "",
            f"- **Target date:** {freshness.get('target_date', 'N/A')}",
            f"- **Latest data:** {freshness.get('latest_data_timestamp', 'N/A')}",
            f"- **Lag:** {freshness.get('actual_lag_hours', 'N/A')} hours",
            f"- **Max allowed lag:** {freshness.get('max_data_lag_hours', 'N/A')} hours",
            f"- **Status:** {freshness.get('status', 'N/A')}",
            "",
        ])

    if manifest.get("warnings"):
        lines.extend(["## Warnings", ""])
        for w in manifest["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    if manifest.get("errors"):
        lines.extend(["## Errors", ""])
        for e in manifest["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    lines.append("---")
    lines.append("_Generated by sync_dataset pipeline_")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync dataset from various sources")
    parser.add_argument("--source", default="auto", choices=["auto", "db", "http", "local"])
    parser.add_argument("--force-sync", action="store_true", default=False)
    parser.add_argument("--require-fresh-data", action="store_true", default=False)
    parser.add_argument("--target-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--max-data-lag-hours", type=int, default=36)
    parser.add_argument("--data-path", default=None)

    args = parser.parse_args()
    result = sync_dataset(
        data_path=args.data_path,
        source=args.source,
        force=args.force_sync,
        require_fresh=args.require_fresh_data,
        target_date=args.target_date,
        max_data_lag_hours=args.max_data_lag_hours,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") != "ok" and result.get("status") != "skipped":
        raise SystemExit(1)
