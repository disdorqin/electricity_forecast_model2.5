"""
96-point (15-min) data synchronization: DB → local Excel/CSV files.

Synced files
------------
data/shandong_pmos_96.xlsx(.csv)  — market-level 96-point feature data
data/unit_data_96.xlsx(.csv)      — unit-level 96-point price/power data

Usage
-----
  # Full historical backfill
  python sync_data_96.py --start-date 2022-01-01 --end-date 2026-07-18

  # Incremental (append-only, skip duplicates by data_time)
  python sync_data_96.py --start-date 2026-07-01

  # Unit-specific
  python sync_data_96.py --unit-id 123456
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from utils.database_operate import fetch_market_data_96, fetch_unit_data_96

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SYNC_MANIFEST_DIR = PROJECT_ROOT / "outputs" / "data_sync_96"
MARKET_XLSX = DATA_DIR / "shandong_pmos_96.xlsx"
MARKET_CSV = DATA_DIR / "shandong_pmos_96.csv"
UNIT_XLSX = DATA_DIR / "unit_data_96.xlsx"
UNIT_CSV = DATA_DIR / "unit_data_96.csv"
TIMESTAMP_COL = "时刻"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_frame(df: pd.DataFrame, xlsx_path: Path, csv_path: Path) -> str:
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(xlsx_path, index=False)
    try:
        df.to_csv(csv_path, index=False, encoding="gbk")
    except Exception:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return str(xlsx_path)


def _merge_and_dedup(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge existing + incoming data on 时刻, keeping last on duplicates."""
    if existing.empty:
        return incoming
    if incoming.empty:
        return existing
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined[TIMESTAMP_COL] = pd.to_datetime(combined[TIMESTAMP_COL], errors="coerce")
    combined = combined.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    combined = combined.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return combined


def _merge_unit_and_dedup(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge existing + incoming unit data on (时刻, unit_id), keeping last on duplicates."""
    if existing.empty:
        return incoming
    if incoming.empty:
        return existing
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined[TIMESTAMP_COL] = pd.to_datetime(combined[TIMESTAMP_COL], errors="coerce")
    combined = combined.drop_duplicates(subset=[TIMESTAMP_COL, "unit_id"], keep="last")
    combined = combined.sort_values([TIMESTAMP_COL, "unit_id"]).reset_index(drop=True)
    return combined


def _load_existing(xlsx_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_path)
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sync implementations
# ---------------------------------------------------------------------------


def sync_market_96(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Sync market-level 96-point data from DB to local files.

    By default, merges with existing local data (dedup on 时刻).
    Use *force* = True to overwrite entirely.

    Returns a manifest dict.
    """
    out_dir = Path(output_dir) if output_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "shandong_pmos_96.xlsx"
    csv_path = out_dir / "shandong_pmos_96.csv"

    rows_count = 0
    min_ts: str | None = None
    max_ts: str | None = None
    errors: list[str] = []
    warnings: list[str] = []

    try:
        incoming = fetch_market_data_96(start_date=start_date, end_date=end_date)
    except Exception as e:
        errors.append(f"DB query failed: {e}")
        return {
            "status": "failed",
            "type": "market_96",
            "errors": errors,
            "rows": 0,
        }

    if incoming.empty:
        warnings.append("No data returned from database for the given date range")
        # Still save (may be empty)
        _save_frame(incoming, xlsx_path, csv_path)
    else:
        if not force:
            existing = _load_existing(xlsx_path)
            merged = _merge_and_dedup(existing, incoming)
        else:
            merged = incoming

        rows_count = len(merged)
        if not merged.empty:
            min_ts = str(merged[TIMESTAMP_COL].min())
            max_ts = str(merged[TIMESTAMP_COL].max())

        _save_frame(merged, xlsx_path, csv_path)

        dedup_count = len(incoming) - (len(merged) - (len(_load_existing(xlsx_path)) if not force else 0))
        # simplified dedup count: not critical, just for info
        if not force and not _load_existing(xlsx_path).empty:
            pass  # merge handled dedup silently

    manifest = {
        "status": "ok" if not errors else "failed",
        "type": "market_96",
        "output_xlsx": str(xlsx_path),
        "output_csv": str(csv_path),
        "rows": rows_count,
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "start_date": start_date,
        "end_date": end_date,
        "warnings": warnings,
        "errors": errors,
    }
    return manifest


def sync_unit_96(
    unit_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Sync unit-level 96-point data from DB to local files.

    By default, merges with existing local data (dedup on 时刻+unit_id).
    Use *force* = True to overwrite entirely.

    Returns a manifest dict.
    """
    out_dir = Path(output_dir) if output_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "unit_data_96.xlsx"
    csv_path = out_dir / "unit_data_96.csv"

    rows_count = 0
    min_ts: str | None = None
    max_ts: str | None = None
    errors: list[str] = []
    warnings: list[str] = []

    try:
        incoming = fetch_unit_data_96(
            unit_id=unit_id,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        errors.append(f"DB query failed: {e}")
        return {
            "status": "failed",
            "type": "unit_96",
            "errors": errors,
            "rows": 0,
        }

    if incoming.empty:
        warnings.append("No unit data returned from database for the given date range")
        _save_frame(incoming, xlsx_path, csv_path)
    else:
        if not force:
            existing = _load_existing(xlsx_path)
            merged = _merge_unit_and_dedup(existing, incoming)
        else:
            merged = incoming

        rows_count = len(merged)
        if not merged.empty:
            min_ts = str(merged[TIMESTAMP_COL].min())
            max_ts = str(merged[TIMESTAMP_COL].max())

        _save_frame(merged, xlsx_path, csv_path)

    manifest = {
        "status": "ok" if not errors else "failed",
        "type": "unit_96",
        "output_xlsx": str(xlsx_path),
        "output_csv": str(csv_path),
        "rows": rows_count,
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "unit_id": unit_id,
        "start_date": start_date,
        "end_date": end_date,
        "warnings": warnings,
        "errors": errors,
    }
    return manifest


def sync_all_96(
    unit_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Sync both market and unit 96-point data from DB to local files.

    Returns an aggregate manifest with nested ``market`` and ``unit`` keys.
    """
    market_manifest = sync_market_96(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        force=force,
    )
    unit_manifest = sync_unit_96(
        unit_id=unit_id,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        force=force,
    )

    status = "ok"
    errors: list[str] = []
    if market_manifest.get("status") != "ok":
        status = "failed"
        errors.extend(market_manifest.get("errors", []))
    if unit_manifest.get("status") != "ok":
        status = "failed"
        errors.extend(unit_manifest.get("errors", []))

    total_rows = (market_manifest.get("rows", 0) or 0) + (unit_manifest.get("rows", 0) or 0)
    manifest: dict = {
        "status": status,
        "type": "all_96",
        "total_rows": total_rows,
        "market": market_manifest,
        "unit": unit_manifest,
        "errors": errors,
    }

    _write_sync_manifest(manifest, start_date, end_date)
    return manifest


# ---------------------------------------------------------------------------
# Manifest / report
# ---------------------------------------------------------------------------


def _write_sync_manifest(
    manifest: dict,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    SYNC_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest["synced_at"] = datetime.now(timezone.utc).isoformat()

    # Build a label for the filename
    label = "full"
    if start_date or end_date:
        label = f"{start_date or 'begin'}_{end_date or date.today().isoformat()}"
    json_path = SYNC_MANIFEST_DIR / f"sync_96_manifest_{label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    md_path = SYNC_MANIFEST_DIR / f"sync_96_report_{label}.md"
    _write_sync_markdown(md_path, manifest)


def _write_sync_markdown(path: Path, manifest: dict) -> None:
    lines = [
        "# 96-Point Data Sync Report",
        "",
        f"- **Status:** {manifest.get('status', 'unknown')}",
        f"- **Type:** {manifest.get('type', 'unknown')}",
        f"- **Synced at:** {manifest.get('synced_at', 'unknown')}",
        f"- **Total rows:** {manifest.get('total_rows', 'N/A')}",
        "",
    ]

    for sub in ("market", "unit"):
        sub_m = manifest.get(sub, {})
        lines.extend([
            f"## {sub.title()} 96-point",
            "",
            f"- **Status:** {sub_m.get('status', 'N/A')}",
            f"- **Output XLSX:** `{sub_m.get('output_xlsx', 'N/A')}`",
            f"- **Output CSV:** `{sub_m.get('output_csv', 'N/A')}`",
            f"- **Rows:** {sub_m.get('rows', 0)}",
            f"- **Min timestamp:** {sub_m.get('min_timestamp', 'N/A')}",
            f"- **Max timestamp:** {sub_m.get('max_timestamp', 'N/A')}",
            "",
        ])
        if sub_m.get("warnings"):
            lines.extend(["### Warnings", ""])
            for w in sub_m["warnings"]:
                lines.append(f"- {w}")
            lines.append("")
        if sub_m.get("errors"):
            lines.extend(["### Errors", ""])
            for e in sub_m["errors"]:
                lines.append(f"- {e}")
            lines.append("")

    lines.append("---")
    lines.append("_Generated by sync_data_96 pipeline_")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync 96-point (15-min) data from DB to local files"
    )
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (inclusive)")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (inclusive)")
    parser.add_argument("--unit-id", default=None, help="Filter by unit ID (unit data only)")
    parser.add_argument("--type", default="all", choices=["all", "market", "unit"],
                        help="Which data to sync (default: all)")
    parser.add_argument("--output-dir", default=None, help="Custom output directory")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Overwrite local files instead of merging")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.type == "market":
        result = sync_market_96(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            force=args.force,
        )
    elif args.type == "unit":
        result = sync_unit_96(
            unit_id=args.unit_id,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            force=args.force,
        )
    else:
        result = sync_all_96(
            unit_id=args.unit_id,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            force=args.force,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") != "ok":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
