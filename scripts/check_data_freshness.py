#!/usr/bin/env python
"""
数据新鲜度检查 — 用于 GitHub Actions 每日监控

检查数据库是否有当天的数据。
返回码:
  0  — 数据正常 (有今天的数据)
  1  — 数据缺失 (触发 Issue 告警)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta

import pymysql
from dotenv import load_dotenv

# 北京时间
BJT = timezone(timedelta(hours=8))
TODAY = datetime.now(BJT).strftime("%Y-%m-%d")


def _env(key: str) -> str:
    v = os.getenv(key, "").strip().strip("\"'")
    if not v:
        print(f"[FATAL] 环境变量 {key} 未设置")
        sys.exit(1)
    return v


def check_table(conn, table: str, date_col: str, label: str) -> bool:
    """检查某张表是否有今天的数据"""
    with conn.cursor() as cur:
        sql = f"SELECT COUNT(*) FROM {table} WHERE {date_col} = %s"
        cur.execute(sql, (TODAY,))
        cnt = cur.fetchone()[0]
    if cnt > 0:
        print(f"[OK] {label} ({table}): {cnt} 条记录 [{TODAY}]")
        return True
    else:
        print(f"[WARN] {label} ({table}): 无 {TODAY} 数据")
        return False


def main() -> None:
    print(f"=== 数据新鲜度检查 [{TODAY}] ===\n")

    host = _env("DB_HOST")
    port = int(os.getenv("DB_PORT", "3306"))
    user = _env("DB_USER")
    pwd = _env("DB_PWD")
    db = _env("DB_NAME")

    conn = pymysql.connect(
        host=host, port=port, user=user, password=pwd,
        database=db, charset="utf8mb4",
        connect_timeout=10,
    )

    try:
        c1 = check_table(conn, "epf_unit_data_96", "market_date", "机组级96点数据")
        c2 = check_table(conn, "epf_market_data_96", "market_date", "全省级96点数据")

        print()

        if c1 or c2:
            print("[PASS] 数据正常")
            sys.exit(0)
        else:
            print("[FAIL] 所有表均无今日数据")
            print("可能原因: 远程电脑未开机 / Cookie过期 / 网络异常")
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
