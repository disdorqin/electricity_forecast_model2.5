"""
数据监控检查脚本 — 用于 GitHub Actions 定时任务

检查项：
  1. 昨日市场96点数据是否已爬取
  2. 昨日机组96点数据是否已爬取
  3. 近7天是否有数据缺失
  4. 数据时间戳是否在合理范围内

退出码: 0=正常, 1=异常
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta

import pymysql


def get_db() -> pymysql.Connection | None:
    """从环境变量读取数据库配置并连接"""
    host = os.getenv("DB_HOST", "").strip()
    port = int(os.getenv("DB_PORT", "3306"))
    database = os.getenv("DB_NAME") or os.getenv("DB", "")
    user = os.getenv("DB_USER", "").strip()
    password = os.getenv("DB_PWD", "").strip()

    if not all([host, database, user, password]):
        print("❌ 数据库环境变量不完整")
        return None

    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=15,
        )
        return conn
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        return None


def check_yesterday_data(conn) -> list[str]:
    """检查昨日数据是否完整"""
    errors: list[str] = []
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    with conn.cursor() as cur:
        # 市场96点
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM epf_market_data_96 WHERE market_date = %s",
            (yesterday,),
        )
        market_cnt = cur.fetchone()["cnt"]
        if market_cnt == 0:
            errors.append(f"❌ {yesterday} 市场96点数据缺失（epf_market_data_96）")
        elif market_cnt < 96:
            errors.append(f"⚠ {yesterday} 市场96点数据不完整（{market_cnt}/96）")
        else:
            print(f"✅ {yesterday} 市场96点数据: {market_cnt}/96")

        # 机组96点
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM epf_unit_data_96 WHERE market_date = %s",
            (yesterday,),
        )
        unit_cnt = cur.fetchone()["cnt"]
        if unit_cnt == 0:
            errors.append(f"❌ {yesterday} 机组96点数据缺失（epf_unit_data_96）")
        elif unit_cnt % 96 != 0:
            errors.append(f"⚠ {yesterday} 机组96点数据可能不完整（{unit_cnt} 行，应为 96 的倍数）")
        else:
            print(f"✅ {yesterday} 机组96点数据: {unit_cnt} 行")

    return errors


def check_recent_gaps(conn) -> list[str]:
    """检查最近7天是否有数据缺失"""
    errors: list[str] = []
    today = date.today()

    with conn.cursor() as cur:
        # 市场96点 最近7天
        cur.execute(
            "SELECT market_date, COUNT(*) AS cnt FROM epf_market_data_96 "
            "WHERE market_date >= %s AND market_date < %s "
            "GROUP BY market_date ORDER BY market_date",
            ((today - timedelta(days=7)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")),
        )
        market_dates = {row["market_date"].strftime("%Y-%m-%d"): row["cnt"] for row in cur.fetchall()}

        # 机组96点 最近7天
        cur.execute(
            "SELECT market_date, COUNT(*) AS cnt FROM epf_unit_data_96 "
            "WHERE market_date >= %s AND market_date < %s "
            "GROUP BY market_date ORDER BY market_date",
            ((today - timedelta(days=7)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")),
        )
        unit_dates = {row["market_date"].strftime("%Y-%m-%d"): row["cnt"] for row in cur.fetchall()}

    # 检查每天数据
    for i in range(1, 8):  # D-1 ~ D-7
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        m_cnt = market_dates.get(d, 0)
        u_cnt = unit_dates.get(d, 0)

        if m_cnt == 0:
            errors.append(f"❌ {d} 市场96点数据缺失")
        elif m_cnt < 96:
            errors.append(f"⚠ {d} 市场96点数据不完整（{m_cnt}/96）")

        if u_cnt == 0:
            # 机组数据可能是空的（还没历史），不发告警
            pass
        elif u_cnt % 96 != 0:
            errors.append(f"⚠ {d} 机组96点数据不完整（{u_cnt} 行）")

    print(f"📅 最近7天检查完成（{len(errors)} 个问题）")
    return errors


def check_data_recency(conn) -> list[str]:
    """检查最新数据是否为昨天"""
    errors: list[str] = []
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    with conn.cursor() as cur:
        # 市场数据最新日期
        cur.execute("SELECT MAX(market_date) AS md FROM epf_market_data_96")
        max_market = cur.fetchone()["md"]
        if max_market:
            max_market_str = max_market.strftime("%Y-%m-%d") if hasattr(max_market, "strftime") else str(max_market)
            if max_market_str < yesterday:
                errors.append(f"❌ 市场数据最新为 {max_market_str}，未覆盖昨日 {yesterday}")
            else:
                print(f"✅ 市场数据最新日期: {max_market_str}")
        else:
            errors.append("❌ 市场数据表为空")

        # 机组数据最新日期
        cur.execute("SELECT MAX(market_date) AS md FROM epf_unit_data_96")
        max_unit = cur.fetchone()["md"]
        if max_unit:
            max_unit_str = max_unit.strftime("%Y-%m-%d") if hasattr(max_unit, "strftime") else str(max_unit)
            if max_unit_str < yesterday:
                errors.append(f"⚠ 机组数据最新为 {max_unit_str}，未覆盖昨日 {yesterday}")
            else:
                print(f"✅ 机组数据最新日期: {max_unit_str}")
        else:
            print("ℹ 机组数据表暂无数据（回填未完成）")

    return errors


def main() -> int:
    print("=" * 50)
    print("  电力数据监控检查")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    conn = get_db()
    if conn is None:
        print("❌ 数据库连接失败，视为异常")
        return 1

    all_errors: list[str] = []

    try:
        print("\n── 数据新鲜度 ──")
        all_errors.extend(check_data_recency(conn))

        print("\n── 昨日数据完整性 ──")
        all_errors.extend(check_yesterday_data(conn))

        print("\n── 近期数据连续性 ──")
        all_errors.extend(check_recent_gaps(conn))
    finally:
        conn.close()

    print()
    print("=" * 50)
    if all_errors:
        print(f"❌ 发现 {len(all_errors)} 个问题:")
        for e in all_errors:
            print(f"  {e}")
        return 1
    else:
        print("✅ 所有检查通过，数据正常")
        return 0


if __name__ == "__main__":
    sys.exit(main())
