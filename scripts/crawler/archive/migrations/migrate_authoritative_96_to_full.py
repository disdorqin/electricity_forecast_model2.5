"""将权威 96 点 CSV 同步到 epf_pmos_96_full。

该工具只读取 data/96/authoritative/pmos_96_全量.csv，先严格验证表头、日期和每日日96点，
再把业务字段原样写入数据库同名列。它不从 epf_market_data_96/epf_unit_data_96 补值，
也不把旧英文宽表中的数据迁移成新数据；旧表由 ensure_full_table_schema 保留为 legacy。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# 允许从仓库根目录直接执行本文件。
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.crawler.crawl import parse_number, period_no_from_time
from scripts.crawler.run_crawler import (
    FULL_DATASET_COLUMNS,
    FULL_NUMERIC_COLUMNS,
    ensure_full_table_schema,
)

DEFAULT_CSV = ROOT / "data" / "96" / "authoritative" / "pmos_96_全量.csv"
DEFAULT_DB = ROOT / "dist" / "crawler" / "db_config.json"


def _clean(value: Any, column: str) -> Any:
    if value is None or str(value).strip() == "":
        return None
    if column in FULL_NUMERIC_COLUMNS:
        parsed = parse_number(value)
        if parsed is None:
            raise RuntimeError(f"数值字段无法解析: {column}={value!r}")
        return parsed
    return str(value).strip()


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = tuple(reader.fieldnames or ())
        if headers != FULL_DATASET_COLUMNS:
            raise RuntimeError(
                "CSV表头必须与权威数据集完全一致。\n"
                f"期望: {list(FULL_DATASET_COLUMNS)}\n实际: {list(headers)}"
            )
        rows: list[dict[str, Any]] = []
        day_counts: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()
        for line_no, raw in enumerate(reader, start=2):
            market_date = str(raw["market_date"] or "").strip()
            period = str(raw["时段"] or "").strip()
            try:
                date.fromisoformat(market_date)
                period_no_from_time(period)
            except Exception as exc:
                raise RuntimeError(f"第{line_no}行日期/时段非法: {market_date} {period}") from exc
            key = (market_date, period)
            if key in seen:
                raise RuntimeError(f"CSV存在重复键: {key}")
            seen.add(key)
            item = {column: _clean(raw.get(column), column) for column in FULL_DATASET_COLUMNS}
            rows.append(item)
            day_counts[market_date] = day_counts.get(market_date, 0) + 1

    bad = {d: n for d, n in day_counts.items() if n != 96}
    if bad:
        raise RuntimeError(f"CSV存在非96点日期: {bad}")
    if len(rows) != len(day_counts) * 96:
        raise RuntimeError("CSV总行数与日期/96点乘积不一致")
    return rows, day_counts


def _load_db_config(path: Path) -> tuple[dict[str, Any], str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    required = ("host", "port", "user", "password", "database")
    missing = [key for key in required if not str(cfg.get(key) or "").strip()]
    if missing:
        raise RuntimeError(f"数据库配置缺少: {missing}")
    unit_id = str(cfg.get("unit_id") or "").strip()
    if not unit_id:
        config_path = path.with_name("config.json")
        if config_path.exists():
            unit_id = str(json.loads(config_path.read_text(encoding="utf-8")).get("unit_id") or "").strip()
    if not unit_id:
        raise RuntimeError("未找到 unit_id，请在 db_config.json 或 config.json 中填写")
    return cfg, unit_id


def migrate(csv_path: Path, db_path: Path, batch_size: int = 500, dry_run: bool = False) -> dict[str, Any]:
    import pymysql

    rows, day_counts = _read_rows(csv_path)
    result = {
        "csv": str(csv_path), "rows": len(rows), "dates": len(day_counts),
        "min_date": min(day_counts), "max_date": max(day_counts),
        "dry_run": dry_run,
    }
    if dry_run:
        return result

    db, unit_id = _load_db_config(db_path)
    captured = datetime.fromtimestamp(os.path.getmtime(csv_path))
    columns = [*FULL_DATASET_COLUMNS, "unit_id", "source_captured_at"]
    quoted = lambda c: f"`{c}`"
    sql = (
        f"INSERT INTO epf_pmos_96_full ({', '.join(map(quoted, columns))}) "
        f"VALUES ({','.join(['%s'] * len(columns))}) ON DUPLICATE KEY UPDATE "
        + ",".join(f"{quoted(c)}=VALUES({quoted(c)})" for c in columns if c not in {"market_date", "时段", "unit_id"})
    )

    conn = pymysql.connect(
        host=str(db["host"]), port=int(db["port"]), user=str(db["user"]),
        password=str(db["password"]), database=str(db["database"]), charset="utf8mb4",
        autocommit=False, connect_timeout=int(db.get("connect_timeout") or 30),
        read_timeout=int(db.get("read_timeout") or 60), write_timeout=int(db.get("write_timeout") or 60),
    )
    try:
        ensure_full_table_schema(conn)
        with conn.cursor() as cur:
            for start in range(0, len(rows), max(1, batch_size)):
                batch = rows[start:start + max(1, batch_size)]
                values = [[row.get(column) for column in FULL_DATASET_COLUMNS] + [unit_id, captured] for row in batch]
                cur.executemany(sql, values)
                print(f"uploaded {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    result["unit_id"] = unit_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="同步权威本地96点CSV到 epf_pmos_96_full")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db-config", type=Path, default=DEFAULT_DB)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="只验证CSV，不连接/写入数据库")
    args = parser.parse_args()
    result = migrate(args.csv, args.db_config, args.batch_size, args.dry_run)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
