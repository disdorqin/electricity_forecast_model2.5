#!/usr/bin/env python
"""
智能补缺爬虫 — 每天检查最近 N 天缺失数据并自动补爬

用法:
  python scripts/crawler/auto_fill_96.py                  # 检查最近14天，缺失的自动补爬
  python scripts/crawler/auto_fill_96.py --lookback 7     # 只检查最近7天
  python scripts/crawler/auto_fill_96.py --dry-run        # 仅显示缺失情况，不实际爬取
  auto_fill_96.exe                                        # .exe 模式（PyInstaller 打包后）

原理:
  1. 查询云数据库 epf_market_data_96 / epf_unit_data_96 获取最近 N 天已有数据
  2. 对比完整日期范围，找出缺失日期
  3. 逐日调用 PmosCrawler 爬取缺失数据
  4. 写入云数据库

这样即使某天爬虫失败（Cookie过期、网络波动等），第二天也会自动补上。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv

# ── PyInstaller / 路径 ──────────────────────────────────────────────
_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
else:
    BASE_DIR = Path(__file__).resolve().parents[2]
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

# PyInstaller 静态分析
from crawl import PmosCrawler, parse_number, period_no_from_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("auto_fill_96")

# 同时输出到日志文件
CRAWLER_DIR = BASE_DIR if _FROZEN else BASE_DIR / "scripts" / "crawler"
OUTPUT_DIR = BASE_DIR / "output"
if _FROZEN:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            str(OUTPUT_DIR / "auto_fill_96.log"), encoding="utf-8", mode="a"
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(fh)
    except Exception:
        pass

# ── 市场特征字段映射 ──────────────────────────────────────────────
MARKET_FIELD_MAP: dict[str, str] = {
    "systemload": "actual_direct_load",
    "dfdcload": "actual_local_plant",
    "excload": "actual_tie_line",
    "fdload": "actual_wind",
    "gfload": "actual_solar",
    "sytsjz": "actual_nuclear",
    "selfunit": "actual_self_owned",
    "syjzzj": "actual_test_unit",
}


# ── 配置加载 ──────────────────────────────────────────────────────
def load_config() -> dict:
    config_path = CRAWLER_DIR / "config.json"
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict = json.load(f)
    if not cfg.get("cookie", "").strip():
        logger.error("config.json 中 cookie 为空")
        sys.exit(1)
    uid = cfg.get("unit_id") or cfg.get("unitid") or ""
    uid = str(uid).strip()
    if not uid:
        logger.error("config.json 中 unit_id 为空")
        sys.exit(1)
    cfg["unit_id"] = uid
    return cfg


def load_db_config() -> dict:
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
    def _env(key: str) -> str:
        return os.getenv(key, "").strip().strip("\"'")
    return {
        "host": _env("DB_HOST"),
        "port": int(_env("DB_PORT") or "3306"),
        "user": _env("DB_USER"),
        "password": _env("DB_PWD"),
        "database": _env("DB") or _env("DB_NAME"),
    }


# ── 数据库操作 ──────────────────────────────────────────────────────
def get_db(cfg: dict):
    return pymysql.connect(
        host=cfg["host"], port=cfg["port"], user=cfg["user"],
        password=cfg["password"], database=cfg["database"],
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=15,
    )


def get_existing_dates(db_cfg: dict, table: str, lookback: int,
                       unit_id: str = "") -> set[str]:
    """查询指定表中最近 N 天已有数据的日期"""
    start = (date.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")
    if table == "epf_unit_data_96":
        sql = ("SELECT DISTINCT market_date FROM epf_unit_data_96 "
               "WHERE market_date >= %s AND unit_id = %s")
        params = (start, unit_id)
    else:
        sql = ("SELECT DISTINCT market_date FROM epf_market_data_96 "
               "WHERE market_date >= %s")
        params = (start,)
    try:
        conn = get_db(db_cfg)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return {r["market_date"].strftime("%Y-%m-%d") for r in cur.fetchall()}
        finally:
            conn.close()
    except Exception as e:
        logger.warning("查询 %s 已有日期失败: %s", table, e)
        return set()


def find_missing_dates(db_cfg: dict, unit_id: str,
                       lookback: int) -> dict[str, set[str]]:
    """找出最近 N 天缺失的市场/机组数据日期"""
    today = date.today()
    all_dates = set()
    d = today - timedelta(days=lookback)
    while d < today:
        all_dates.add(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    existing_market = get_existing_dates(db_cfg, "epf_market_data_96", lookback)
    existing_unit = get_existing_dates(db_cfg, "epf_unit_data_96", lookback, unit_id)

    missing_market = all_dates - existing_market
    missing_unit = all_dates - existing_unit

    return {"market": missing_market, "unit": missing_unit}


# ── 爬取与写入 ──────────────────────────────────────────────────────
def crawl_and_save(date_str: str, cookie: str, unit_id: str,
                   base_url: str, db_cfg: dict,
                   no_db: bool = False) -> dict[str, int]:
    """爬取单日数据并写入数据库"""
    result = {"market": 0, "da": 0, "rt": 0}

    spider = PmosCrawler(base_url=base_url, cookie=cookie, unit_id=unit_id)
    spider.fetch_csrf_token()
    if not spider.change_date(date_str):
        logger.warning("  ⚠ 日期切换失败，跳过 %s", date_str)
        return result

    time.sleep(1.5)

    # 市场特征
    market_rows: list[dict] = []
    try:
        market_rows = spider.crawl_market_overview()
        result["market"] = len(market_rows)
        logger.info("  市场特征 → %d 行", len(market_rows))
    except Exception as e:
        logger.warning("  市场特征爬取失败: %s", e)

    # 日前电价
    da_rows: list[dict] = []
    try:
        da_rows = spider.crawl_day_ahead()
        result["da"] = len(da_rows)
        logger.info("  日前电价 → %d 行", len(da_rows))
    except Exception as e:
        logger.warning("  日前电价爬取失败: %s", e)

    # 实时电价
    rt_rows: list[dict] = []
    try:
        rt_rows = spider.crawl_realtime()
        result["rt"] = len(rt_rows)
        logger.info("  实时电价 → %d 行", len(rt_rows))
    except Exception as e:
        logger.warning("  实时电价爬取失败: %s", e)

    # 写入数据库
    if not no_db:
        try:
            conn = get_db(db_cfg)
            try:
                if market_rows:
                    _upsert_market(conn, date_str, market_rows)
                if da_rows or rt_rows:
                    _upsert_unit(conn, date_str, unit_id, da_rows, rt_rows)
                logger.info("  [DB] 写入完成")
            finally:
                conn.close()
        except Exception as e:
            logger.error("  [DB] 写入失败: %s", e)

    return result


def _upsert_market(conn, market_date: str, rows: list[dict]):
    field_map = MARKET_FIELD_MAP
    db_cols = ["market_date", "period_no", "data_time"] + list(field_map.values())
    placeholders = ", ".join(["%s"] * len(db_cols))
    update_parts = ", ".join([f"{c}=VALUES({c})" for c in field_map.values()])
    sql = (f"INSERT INTO epf_market_data_96 ({', '.join(db_cols)}) "
           f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_parts}")
    dt_base = datetime.strptime(market_date, "%Y-%m-%d")
    with conn.cursor() as cur:
        for row in rows:
            pno = period_no_from_time(row.get("Periodid", ""))
            data_time = dt_base + timedelta(minutes=pno * 15)
            vals = [market_date, pno, data_time]
            for col in field_map:
                vals.append(parse_number(row.get(col)))
            cur.execute(sql, vals)
    conn.commit()


def _upsert_unit(conn, market_date: str, unit_id: str,
                 da_rows: list[dict], rt_rows: list[dict]):
    da_idx = {r.get("periodid", ""): r for r in da_rows}
    rt_idx = {r.get("periodid", ""): r for r in rt_rows}
    all_labels = sorted(set(da_idx.keys()) | set(rt_idx.keys()))
    if not all_labels:
        return

    db_cols = [
        "market_date", "period_no", "data_time", "unit_id",
        "da_cq_price", "da_power", "da_energy", "da_status",
        "rt_cq_price", "rt_power", "rt_energy", "rt_status",
    ]
    placeholders = ", ".join(["%s"] * len(db_cols))
    update_cols = [c for c in db_cols if c not in
                   ("market_date", "period_no", "data_time", "unit_id")]
    update_parts = ", ".join([f"{c}=VALUES({c})" for c in update_cols])
    sql = (f"INSERT INTO epf_unit_data_96 ({', '.join(db_cols)}) "
           f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_parts}")

    dt_base = datetime.strptime(market_date, "%Y-%m-%d")
    with conn.cursor() as cur:
        for label in all_labels:
            pno = period_no_from_time(label)
            data_time = dt_base + timedelta(minutes=pno * 15)
            da = da_idx.get(label, {})
            rt = rt_idx.get(label, {})
            vals = [
                market_date, pno, data_time, unit_id,
                parse_number(da.get("cqPrice")),
                parse_number(da.get("power")),
                parse_number(da.get("energy")),
                da.get("kt") or None,
                parse_number(rt.get("cqPrice")),
                parse_number(rt.get("power")),
                parse_number(rt.get("energy")),
                rt.get("kt") or None,
            ]
            cur.execute(sql, vals)
    conn.commit()


# ── 主入口 ──────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="智能补缺爬虫 — 自动检查并补爬最近缺失的96点数据"
    )
    parser.add_argument("--lookback", type=int, default=14,
                        help="检查最近多少天 (默认 14)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅显示缺失情况，不实际爬取")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="每次请求间隔秒数 (默认 2.0)")
    parser.add_argument("--no-db", action="store_true",
                        help="不写入数据库（仅测试）")
    args = parser.parse_args()

    print("=" * 55)
    print("  智能补缺爬虫 — 自动检查并补爬缺失数据")
    print(f"  检查范围: 最近 {args.lookback} 天")
    if args.dry_run:
        print("  模式: 🔍 DRY RUN（仅检查，不爬取）")
    print("=" * 55)

    # 加载配置
    config = load_config()
    cookie = config["cookie"]
    unit_id = config["unit_id"]
    base_url = config.get("base_url",
                          "https://pmos.sd.sgcc.com.cn:18080/trade")
    db_cfg = load_db_config()
    db_ok = all([db_cfg.get("host"), db_cfg.get("database"),
                 db_cfg.get("user"), db_cfg.get("password")])

    if not db_ok:
        logger.error("数据库配置不完整，无法检查缺失日期")
        return 1

    # 查找缺失日期
    logger.info("正在查询数据库已有数据...")
    missing = find_missing_dates(db_cfg, unit_id, args.lookback)
    missing_market = sorted(missing["market"])
    missing_unit = sorted(missing["unit"])

    all_missing = sorted(set(missing_market + missing_unit))
    if not all_missing:
        print("\n✅ 最近 %d 天数据完整，无需补爬！" % args.lookback)
        return 0

    print(f"\n📋 检查结果:")
    print(f"    市场数据缺失: {len(missing_market)} 天")
    print(f"    机组数据缺失: {len(missing_unit)} 天")
    if all_missing:
        missing_str = ", ".join(all_missing[:10])
        if len(all_missing) > 10:
            missing_str += f" ... 共 {len(all_missing)} 天"
        print(f"    缺失日期: {missing_str}")

    if args.dry_run:
        print("\n🔍 DRY RUN 模式，未执行实际爬取。")
        print("   去掉 --dry-run 即可补爬。")
        return 0

    # 逐日补爬
    print()
    for idx, date_str in enumerate(all_missing):
        print(f"\n── [{idx+1}/{len(all_missing)}] {date_str} ──")

        # 检查该日是否需要爬（如果只有 market 缺失，只爬 market 太复杂，统一爬全部）
        for attempt in range(2):
            try:
                res = crawl_and_save(date_str, cookie, unit_id, base_url,
                                     db_cfg, no_db=args.no_db)
                break
            except Exception as e:
                logger.warning("  第 %d 次重试失败: %s", attempt + 1, e)
                time.sleep(3)
        else:
            logger.error("  ⛔ 重试耗尽，跳过 %s", date_str)

        if idx < len(all_missing) - 1:
            time.sleep(args.delay)

    # 最终汇总
    print(f"\n{'=' * 55}")
    logger.info("补爬完成！")
    print(f"{'=' * 55}")

    # 二次检查还剩多少缺失
    leftover = find_missing_dates(db_cfg, unit_id, args.lookback)
    remaining = sorted(set(leftover["market"]) | set(leftover["unit"]))
    if remaining:
        logger.warning("仍有 %d 天缺失: %s", len(remaining),
                       ", ".join(remaining[:5]))
        return 1
    else:
        logger.info("✅ 最近 %d 天数据已全部补全！", args.lookback)
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(1)
    except Exception as e:
        logger.exception("程序异常: %s", e)
        sys.exit(1)
