"""AUX-V1 isolated database writer.

Only ``epf_pmos_aux_*`` tables are reachable from this module.  The 96-point
``run_crawler`` writer is intentionally not imported.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger("disclosure_aux_db")
BUILD_VERSION = "2026-09-22-disclosure-aux-v1"
TABLES = {"records": "epf_pmos_aux_records"}
ALLOWED_TABLES = frozenset(TABLES.values())


def _sql_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "sql" / "003_create_epf_pmos_aux.sql",
        Path(__file__).resolve().parent.parent.parent.parent / "dist" / "crawler" / "辅助信息披露爬虫" / "辅助信息披露数据库表设计.sql",
    ]
    # Frozen AUX EXE is distributed with the design DDL beside it.
    import sys
    if getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable).resolve().parent / "辅助信息披露数据库表设计.sql")
    return next((path for path in candidates if path.exists()), candidates[0])


def ddl_text() -> str:
    return _sql_path().read_text(encoding="utf-8")


def validate_ddl(text: str | None = None) -> tuple[bool, list[str]]:
    text = ddl_text() if text is None else text
    errors: list[str] = []
    executable = re.sub(r"--[^\n]*", "", text)
    executable = re.sub(r"/\*.*?\*/", "", executable, flags=re.S)
    if "epf_pmos_96_full" in executable:
        errors.append("DDL must not reference epf_pmos_96_full")
    names = set(re.findall(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+`?([A-Za-z0-9_]+)`?", executable, flags=re.I))
    if names - ALLOWED_TABLES:
        errors.append(f"unexpected tables: {sorted(names - ALLOWED_TABLES)}")
    if names != ALLOWED_TABLES:
        errors.append(f"expected only {sorted(ALLOWED_TABLES)}, got {sorted(names)}")
    return not errors, errors


def get_db(cfg: Mapping[str, Any]):
    import pymysql

    # [AUX-V1-r10-diag1] Log DB connection phase and duration, never credentials or host/user values.
    started = time.monotonic()
    timeout = int(cfg.get("connect_timeout", 10))
    logger.info("AUX DB connect begin timeout_sec=%s tls=driver_default", timeout)
    try:
        conn = pymysql.connect(
            host=cfg["host"], port=int(cfg.get("port", 3306)), user=cfg["user"], password=cfg["password"],
            database=cfg["database"], charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=timeout, autocommit=False,
        )
    except Exception as exc:
        logger.exception("AUX DB connect failed error_type=%s elapsed_sec=%.1f", type(exc).__name__, time.monotonic() - started)
        raise
    logger.info("AUX DB connect complete elapsed_sec=%.1f", time.monotonic() - started)
    return conn


def _statements(text: str) -> list[str]:
    # [AUX-V1-r6] Strip SQL comments before splitting; semicolons in comments
    # must not silently cause CREATE statements to be skipped.
    executable = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    executable = re.sub(r"(?m)^\s*--[^\n]*(?:\n|$)", "", executable)
    return [part.strip() for part in executable.split(";") if part.strip()]


def init_aux_tables(db_cfg: Mapping[str, Any]) -> bool:
    # [AUX-V1] isolated migration; never reaches the 96-point table.
    # [AUX-V1-r10-diag1] Add schema initialization timing and failure context only.
    started = time.monotonic()
    logger.info("AUX DB schema init begin ddl_path=%s", _sql_path())
    ok, errors = validate_ddl()
    if not ok:
        logger.error("AUX DB schema validation failed errors=%s", errors)
        raise ValueError("; ".join(errors))
    conn = get_db(db_cfg)
    try:
        with conn.cursor() as cur:
            for stmt in _statements(ddl_text()):
                normalized = stmt.strip()
                if normalized.upper().startswith("CREATE TABLE"):
                    cur.execute(normalized)
        conn.commit()
        logger.info("AUX DB schema init complete tables=%s elapsed_sec=%.1f", sorted(ALLOWED_TABLES), time.monotonic() - started)
        return True
    except Exception as exc:
        conn.rollback()
        logger.exception("AUX DB schema init failed error_type=%s elapsed_sec=%.1f", type(exc).__name__, time.monotonic() - started)
        raise
    finally:
        conn.close()


def _insert_upsert(cursor: Any, table: str, row: Mapping[str, Any], *, unique: str) -> None:
    # [AUX-V1] application-keyed idempotent upsert for AUX tables only.
    if table not in ALLOWED_TABLES:
        raise ValueError(f"AUX writer rejected table: {table}")
    values = dict(row)
    columns = [str(k) for k in values]
    if not columns:
        return
    quoted = ", ".join(f"`{c}`" for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    update_columns = [c for c in columns if c != unique]
    if update_columns:
        update = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in update_columns)
    else:
        update = f"`{unique}`=`{unique}`"
    sql = f"INSERT INTO `{table}` ({quoted}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update}"
    cursor.execute(sql, [values[c] for c in columns])


def _mysql_datetime(value: Any) -> Any:
    """Normalize ISO timestamps for MySQL DATETIME, which stores no timezone."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return value
    else:
        return value
    # [AUX-V1-r9] The DATETIME column stores local wall time without an offset;
    # the original offset-bearing value remains in local raw provenance.
    if parsed.utcoffset() is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _upsert_many(db: Any, table: str, rows: Iterable[Mapping[str, Any]], *, unique: str) -> int:
    rows = [dict(row) for row in rows]
    if not rows:
        return 0
    conn = db
    own = False
    if isinstance(db, Mapping):
        conn = get_db(db)
        own = True
    try:
        with conn.cursor() as cursor:
            for row in rows:
                if "captured_at" in row:
                    row["captured_at"] = _mysql_datetime(row["captured_at"])
                _insert_upsert(cursor, table, row, unique=unique)
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        if own:
            conn.close()


def upsert_raw(db: Any, row: Mapping[str, Any]) -> int:
    # [AUX-V1-r6] raw responses and source metadata share the unified record table.
    value = dict(row)
    request = str(value.get("request_key") or "")
    digest = str(value.get("raw_hash") or "")
    source_name = str(value.get("source_name") or "")
    identity = __import__("hashlib").sha256(f"raw|{source_name}|{request}|{digest}".encode("utf-8")).hexdigest()
    request_params = value.get("request_params", value.get("request_params_json", {}))
    raw_payload = value.get("raw_json", {})
    value = {
        "record_key": identity, "run_id": value.get("run_id"), "record_kind": "raw",
        "source_group": value.get("source_group"), "source_name": value.get("source_name"),
        "record_type": value.get("source_name"), "business_date": value.get("business_date"),
        "resolution": value.get("resolution"), "source_api": value.get("source_api"),
        "request_key": request, "request_params_json": json.dumps(request_params, ensure_ascii=False, sort_keys=True, default=str),
        "http_status": value.get("http_status"), "business_code": str(value.get("business_code") or ""),
        "source_status": value.get("source_status"), "source_row_count": value.get("source_row_count"),
        "raw_hash": digest, "raw_json": json.dumps(raw_payload, ensure_ascii=False, default=str),
        "record_json": None, "captured_at": value.get("captured_at"),
        "schema_version": value.get("schema_version", "AUX-V1"),
    }
    value.pop("raw_path", None)
    return _upsert_many(db, TABLES["records"], [value], unique="record_key")


def enrich_structured_rows(rows: Iterable[Mapping[str, Any]], *, source_result: Any) -> list[dict[str, Any]]:
    """Add provenance to parser rows without changing their business meaning."""
    captured = source_result.raw.get("captured_at") if getattr(source_result, "raw", None) else None
    raw_digest = source_result.raw.get("raw_hash") if getattr(source_result, "raw", None) else None
    out = []
    for row in rows:
        value = dict(row)
        value.setdefault("raw_hash", raw_digest)
        value.setdefault("captured_at", captured or datetime.now().isoformat(sep=" "))
        value.setdefault("schema_version", "AUX-V1")
        out.append(value)
    return out


def upsert_source_result(db: Any, source_result: Any) -> int:
    # [AUX-V1-r6] route both raw and structured payloads to one AUX table.
    """Persist raw-first provenance and lossless structured source fields."""
    raw = dict(source_result.raw)
    raw.setdefault("source_name", source_result.name)
    raw.pop("raw_path", None)  # local-only diagnostic, not a DB column
    raw.setdefault("request_key", raw.get("request_key") or "")
    raw.setdefault("raw_hash", raw.get("raw_hash") or "")
    # [AUX-V1-r10-diag1] Report raw/structured persistence counts and elapsed time.
    # request_key is derived by the collector; retaining an empty value is
    # preferable to inventing a business identity in this writer.
    started = time.monotonic()
    logger.info("AUX DB persist begin source=%s status=%s raw_present=%s structured_rows=%s",
                source_result.name, source_result.status, bool(raw), len(source_result.rows))
    count = upsert_raw(db, raw)
    if source_result.status not in {"COMPLETE", "PARTIAL"} or not source_result.rows:
        logger.info("AUX DB persist complete source=%s raw_rows=%s structured_rows=0 elapsed_sec=%.1f",
                    source_result.name, count, time.monotonic() - started)
        return count
    structured = []
    for row in enrich_structured_rows(source_result.rows, source_result=source_result):
        business_key = str(row.get("record_key") or __import__("hashlib").sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest())
        structured.append({
            "record_key": business_key, "run_id": raw.get("run_id"), "record_kind": "structured",
            "source_group": source_result.group, "source_name": source_result.name,
            "record_type": row.get("record_type") or row.get("event_type") or row.get("curve_type") or row.get("stat_type") or source_result.name,
            "business_date": row.get("business_date") or row.get("snapshot_date") or raw.get("business_date"),
            "resolution": raw.get("resolution"), "source_api": row.get("source_api") or raw.get("source_api"),
            "request_key": raw.get("request_key"), "request_params_json": json.dumps(raw.get("request_params", {}), ensure_ascii=False, sort_keys=True, default=str),
            "http_status": raw.get("http_status"), "business_code": str(raw.get("business_code") or ""),
            "source_status": source_result.status, "source_row_count": len(source_result.rows),
            "raw_hash": raw.get("raw_hash"), "raw_json": None,
            "record_json": json.dumps(row, ensure_ascii=False, sort_keys=True, default=str),
            "captured_at": row.get("captured_at") or raw.get("captured_at"),
            "schema_version": row.get("schema_version", "AUX-V1"),
        })
    structured_count = _upsert_many(db, TABLES["records"], structured, unique="record_key")
    logger.info("AUX DB persist complete source=%s raw_rows=%s structured_rows=%s elapsed_sec=%.1f",
                source_result.name, count, structured_count, time.monotonic() - started)
    return count + structured_count
