"""
epf_unit_data_96 历史数据回填脚本

从 2022-01-01 ~ 2026-07-17 逐日爬取机组级96点数据，
跳过数据库中已有的日期（断点续爬）。

用法：
  # 查看待爬取日期范围（不实际爬取）
  python scripts/backfill_unit_data_96.py --dry-run

  # 正式回填全部缺失日期
  python scripts/backfill_unit_data_96.py

  # 指定区间
  python scripts/backfill_unit_data_96.py --start 2024-01-01 --end 2024-06-30

  # 强制重新爬取（即使数据库中已有）
  python scripts/backfill_unit_data_96.py --force

注意：必须在国网内网办公电脑上运行（PMOS 网站仅内网可达）。
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

# ── 屏蔽 SSL 警告（必须在导入 requests 之前生效） ─────────────────────
import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 设置环境变量确保全局生效
os.environ["PYTHONWARNINGS"] = "ignore::urllib3.exceptions.InsecureRequestWarning"
# 捕获 logging 级别的警告
logging.captureWarnings(True)
warnings_logger = logging.getLogger("py.warnings")
warnings_logger.setLevel(logging.ERROR)

# ── PyInstaller / 路径 ──────────────────────────────────────────────
_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
else:
    BASE_DIR = Path(__file__).resolve().parents[4]
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
import pymysql

# PyInstaller 静态分析能看到这个顶层 import，自动打包 scripts.crawler.crawl
from scripts.crawler.crawl import PmosCrawler, parse_number, period_no_from_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_unit")

# ── 配置加载 ────────────────────────────────────────────────────────

CRAWLER_DIR = BASE_DIR if _FROZEN else BASE_DIR / "scripts" / "crawler"
PROGRESS_FILE = BASE_DIR / "outputs" / "backfill_progress.json"

# 致命错误类型：这类错误重试也没用，直接退出
_FATAL_ERROR_KEYWORDS = [
    "SSLError", "UNEXPECTED_EOF", "ConnectionError",
    "RemoteDisconnected", "Connection aborted",
    "Connection refused", "NameResolutionError",
    "Timeout", "ConnectTimeout",
]


def load_config() -> dict:
    config_path = CRAWLER_DIR / "config.json"
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        logger.error("请先在办公电脑上配置 scripts/crawler/config.json")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict = json.load(f)
    if not cfg.get("cookie", "").strip():
        logger.error("config.json 中 cookie 为空")
        sys.exit(1)
    uid = cfg.get("unit_id") or cfg.get("unitid") or ""
    uid = str(uid).strip()
    if not uid:
        logger.error("config.json 中 unit_id/unitid 为空")
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


def _get_conn(db_cfg: dict):
    return pymysql.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        database=db_cfg["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
    )


def fetch_crawled_dates(db_cfg: dict, unit_id: str) -> set[str]:
    """查询 epf_unit_data_96 表中已经爬取过的市场日期（去重）"""
    sql = "SELECT DISTINCT market_date FROM epf_unit_data_96 WHERE unit_id = %s"
    try:
        conn = _get_conn(db_cfg)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (unit_id,))
                return {row["market_date"].strftime("%Y-%m-%d") for row in cur.fetchall()}
        finally:
            conn.close()
    except Exception as e:
        logger.warning("查询已有日期失败: %s", e)
        return set()


def upsert_unit_data(
    conn,
    market_date: str,
    unit_id: str,
    da_rows: list[dict[str, Any]],
    rt_rows: list[dict[str, Any]],
) -> int:
    """将机组日前/实时数据 upsert 到 epf_unit_data_96"""
    da_index: dict[str, dict] = {r.get("periodid", ""): r for r in da_rows}
    rt_index: dict[str, dict] = {r.get("periodid", ""): r for r in rt_rows}
    all_labels = sorted(set(da_index.keys()) | set(rt_index.keys()))
    if not all_labels:
        return 0

    db_cols = [
        "market_date", "period_no", "data_time", "unit_id",
        "da_cq_price", "da_power", "da_energy", "da_status",
        "rt_cq_price", "rt_power", "rt_energy", "rt_status",
    ]
    placeholders = ", ".join(["%s"] * len(db_cols))
    update_cols = [c for c in db_cols if c not in ("market_date", "period_no", "data_time", "unit_id")]
    update_parts = ", ".join([f"{c}=VALUES({c})" for c in update_cols])
    sql = (
        f"INSERT INTO epf_unit_data_96 ({', '.join(db_cols)}) "
        f"VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_parts}"
    )

    dt_base = datetime.strptime(market_date, "%Y-%m-%d")
    count = 0
    with conn.cursor() as cur:
        for label in all_labels:
            pno = period_no_from_time(label)
            data_time = dt_base + timedelta(minutes=pno * 15)
            da = da_index.get(label, {})
            rt = rt_index.get(label, {})
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
            try:
                cur.execute(sql, vals)
                count += 1
            except pymysql.err.IntegrityError as e:
                logger.warning("  ⚠ 跳过 %s period=%d: %s", label, pno, e)
    conn.commit()
    return count


# ── 进度管理 ────────────────────────────────────────────────────────


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_success_date": None, "total_crawled": 0, "skipped_dates": []}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── 错误分类 ────────────────────────────────────────────────────────


def classify_error(err_msg: str) -> str:
    """对错误消息分类，返回中文描述"""
    err_upper = err_msg.upper()
    if "SSL" in err_upper or "UNEXPECTED_EOF" in err_upper:
        return "SSL/网络错误（PMOS 服务器不可达，请检查网络连接）"
    if "REMOTEDISCONNECTED" in err_upper or "CONNECTION ABORTED" in err_upper:
        return "连接被断开（PMOS 服务器主动断开，请检查网络或 Cookie 是否过期）"
    if "CONNECTION REFUSED" in err_upper:
        return "连接被拒绝（PMOS 端口不通）"
    if "TIMEOUT" in err_upper or "CONNECTTIMEOUT" in err_upper:
        return "连接超时（PMOS 响应过慢）"
    if "401" in err_msg or "403" in err_msg:
        return "认证失败（HTTP 401/403，Cookie 已过期，请更新 config.json）"
    if "500" in err_msg:
        return "PMOS 服务器内部错误 (HTTP 500)"
    if "EXPECTING VALUE" in err_upper and "CHAR 0" in err_upper:
        return "PMOS 返回空响应（Cookie 可能过期或该日期无数据）"
    if "CSRF" in err_upper:
        return "CSRF token 获取失败"
    return f"未知错误: {err_msg[:120]}"


def is_fatal_error(err_msg: str) -> bool:
    """判断是否为致命错误（重试也没用）"""
    err_upper = err_msg.upper()
    for kw in _FATAL_ERROR_KEYWORDS:
        if kw.upper() in err_upper:
            return True
    return False


# ── PMOS 连通性预检 ────────────────────────────────────────────────


def check_pmos_connectivity(spider: PmosCrawler) -> str:
    """快速检测 PMOS 是否可达，返回诊断结果"""
    logger.info("正在检测 PMOS 服务器连通性...")
    try:
        resp = spider.session.get(
            f"{spider.base_url}/main/index.do",
            timeout=15,
            verify=False,
        )
        logger.info(f"  HTTP 状态码: {resp.status_code}")
        logger.info(f"  响应长度: {len(resp.text)} 字符")
        if resp.status_code == 200 and len(resp.text) > 100:
            logger.info("  ✅ PMOS 服务器连接正常")
            return "ok"
        elif resp.status_code in (302, 401, 403):
            logger.error("  ❌ PMOS 返回 HTTP %d — Cookie 可能已过期", resp.status_code)
            return "auth_failed"
        else:
            logger.warning(f"  ⚠ PMOS 返回 HTTP {resp.status_code}，响应过短")
            return "unexpected"
    except Exception as e:
        err_str = str(e)
        category = classify_error(err_str)
        logger.error(f"  ❌ {category}")
        logger.error(f"     详细: {err_str[:200]}")
        if is_fatal_error(err_str):
            return "fatal"
        return err_str[:50]


# ── 爬取单日 ────────────────────────────────────────────────────────


def crawl_single_day(
    spider: PmosCrawler,
    date_str: str,
    unit_id: str,
    db_cfg: dict,
    no_db: bool = False,
) -> dict:
    """爬取单日数据并写入 DB。

    返回记录数统计: {da, rt, market, errors, fatal}
    """
    result: dict[str, Any] = {"da": 0, "rt": 0, "market": 0, "errors": [], "fatal": False, "status": "ok"}

    # 切换日期
    if not spider.change_date(date_str):
        result["status"] = "skip"
        result["errors"].append("change_date 失败")
        return result

    time.sleep(0.5)

    # 爬取日前
    da_rows: list[dict] = []
    try:
        da_rows = spider.crawl_day_ahead()
        result["da"] = len(da_rows)
    except Exception as e:
        err_str = str(e)
        result["errors"].append(f"日前电价: {classify_error(err_str)}")
        if is_fatal_error(err_str):
            result["fatal"] = True

    # 爬取实时
    rt_rows: list[dict] = []
    try:
        rt_rows = spider.crawl_realtime()
        result["rt"] = len(rt_rows)
    except Exception as e:
        err_str = str(e)
        result["errors"].append(f"实时电价: {classify_error(err_str)}")
        if is_fatal_error(err_str):
            result["fatal"] = True

    # 爬取市场特征-预测(日前)
    forecast_rows: list[dict] = []
    try:
        forecast_rows = spider.crawl_market_overview()
        result["market"] = len(forecast_rows)
    except Exception as e:
        err_str = str(e)
        result["errors"].append(f"市场特征(预测): {classify_error(err_str)}")
        if is_fatal_error(err_str):
            result["fatal"] = True

    # 爬取市场特征-实际(实时)
    actual_rows: list[dict] = []
    try:
        actual_rows = spider.crawl_market_overview_actual()
        result["market"] = max(result["market"], len(actual_rows))
    except Exception as e:
        err_str = str(e)
        result["errors"].append(f"市场特征(实际): {classify_error(err_str)}")
        if is_fatal_error(err_str):
            result["fatal"] = True

    # 写入 DB
    if not no_db:
        try:
            conn = _get_conn(db_cfg)
            try:
                if da_rows or rt_rows:
                    cnt = upsert_unit_data(conn, date_str, unit_id, da_rows, rt_rows)
                    logger.debug("  unit upsert: %d", cnt)
                if forecast_rows:
                    _upsert_market_forecast(conn, date_str, forecast_rows)
                if actual_rows:
                    _upsert_market_overview(conn, date_str, actual_rows)
            finally:
                conn.close()
        except Exception as e:
            result["errors"].append(f"数据库写入: {e}")
            result["status"] = "db_error"

    if result["errors"]:
        result["status"] = "partial" if result["da"] + result["rt"] > 0 else "failed"

    return result


# 预测(DaJyxxPlDa)→fcast_* 列；实际(DaJyxxPlYx)→actual_* 列
FORECAST_FIELD_MAP: dict[str, str] = {
    "systemload": "fcast_direct_load", "dfdcload": "fcast_local_plant",
    "excload": "fcast_tie_line", "fdload": "fcast_wind",
    "gfload": "fcast_solar", "sytsjz": "fcast_nuclear",
    "selfunit": "fcast_self_owned", "syjzzj": "fcast_test_unit",
}

ACTUAL_FIELD_MAP: dict[str, str] = {
    "systemload": "actual_direct_load", "dfdcload": "actual_local_plant",
    "excload": "actual_tie_line", "fdload": "actual_wind",
    "gfload": "actual_solar",  # cxload(抽蓄) 无对应列跳过
    "hdload": "actual_nuclear", "zbload": "actual_self_owned",
    "syjzload": "actual_test_unit",
}


def _upsert_market_by_map(conn, market_date: str, rows: list[dict[str, Any]],
                          field_map: dict[str, str]):
    """按给定字段映射 upsert 到 epf_market_data_96（只写有值的列）。"""
    db_cols = ["market_date", "period_no", "data_time"] + list(field_map.values())
    placeholders = ", ".join(["%s"] * len(db_cols))
    update_parts = ", ".join([f"{c}=VALUES({c})" for c in field_map.values()])
    sql = (
        f"INSERT INTO epf_market_data_96 ({', '.join(db_cols)}) "
        f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_parts}"
    )
    dt_base = datetime.strptime(market_date, "%Y-%m-%d")
    count = 0
    with conn.cursor() as cur:
        for row in rows:
            period_label = row.get("Periodid", "")
            if not period_label:
                continue
            pno = period_no_from_time(period_label)
            data_time = dt_base + timedelta(minutes=pno * 15)
            present = {
                c: field_map[c] for c in field_map
                if row.get(c) not in (None, "")
            }
            if not present:
                continue
            cols = ["market_date", "period_no", "data_time"] + list(present.values())
            ph = ", ".join(["%s"] * len(cols))
            up = ", ".join([f"{c}=VALUES({c})" for c in present.values()])
            s = (
                f"INSERT INTO epf_market_data_96 ({', '.join(cols)}) "
                f"VALUES ({ph}) ON DUPLICATE KEY UPDATE {up}"
            )
            vals = [market_date, pno, data_time]
            vals += [parse_number(row.get(c)) for c in present]
            try:
                cur.execute(s, vals)
                count += 1
            except pymysql.err.IntegrityError:
                pass
    conn.commit()
    return count


def _upsert_market_overview(conn, market_date: str, rows: list[dict[str, Any]]):
    """将实际接口(DaJyxxPlYx)数据 upsert 到 actual_* 列"""
    return _upsert_market_by_map(conn, market_date, rows, ACTUAL_FIELD_MAP)


def _upsert_market_forecast(conn, market_date: str, rows: list[dict[str, Any]]):
    """将预测接口(DaJyxxPlDa)数据 upsert 到 fcast_* 列"""
    return _upsert_market_by_map(conn, market_date, rows, FORECAST_FIELD_MAP)


# ── 主流程 ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="机组级96点数据历史回填 (2022-01-01 ~ 爬取前日)"
    )
    parser.add_argument("--start", default="2022-01-01", help="开始日期 (默认 2022-01-01)")
    parser.add_argument("--end", default=None, help="结束日期 (默认爬取前日)")
    parser.add_argument("--force", action="store_true", help="强制重新爬取已有日期")
    parser.add_argument("--dry-run", action="store_true", help="仅显示待爬取日期，不实际爬取")
    parser.add_argument("--no-db", action="store_true", help="不写入数据库")
    parser.add_argument("--delay", type=float, default=1.0, help="每次请求间延迟秒数 (默认 1.0)")
    parser.add_argument("--max-retry", type=int, default=2, help="单日最大重试次数 (默认 2)")
    args = parser.parse_args()

    # ── 日期范围 ──
    end_date = args.end or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_dates: list[str] = []
    cur = start_dt
    while cur <= end_dt:
        all_dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    total_days = len(all_dates)
    logger.info("日期范围: %s ~ %s (%d 天)", args.start, end_date, total_days)

    # ── 加载配置 ──
    config = load_config()
    cookie = config["cookie"]
    unit_id = config["unit_id"]
    base_url = config.get("base_url", "https://pmos.sd.sgcc.com.cn:18080/trade")
    db_cfg = load_db_config()
    db_ok = all([db_cfg.get("host"), db_cfg.get("database"), db_cfg.get("user"), db_cfg.get("password")])

    # ── 查询已有日期 ──
    if not args.force and db_ok:
        crawled = fetch_crawled_dates(db_cfg, unit_id)
        logger.info("数据库中已有 %d 天数据", len(crawled))
    else:
        crawled = set()

    # ── 筛选待爬取日期 ──
    to_crawl = [d for d in all_dates if d not in crawled]
    if args.force:
        to_crawl = all_dates

    if not to_crawl:
        logger.info("所有日期均已爬取，无需回填")
        return

    logger.info("待爬取: %d 天 (已跳过 %d 天已有数据)", len(to_crawl), total_days - len(to_crawl))

    if args.dry_run:
        logger.info("DRY RUN 模式，以下日期将爬取:")
        for d in to_crawl[:10]:
            logger.info("  %s", d)
        if len(to_crawl) > 10:
            logger.info("  ... 共 %d 天", len(to_crawl))
        return

    # ── PMOS 连通性预检（连不上直接退，不刷屏） ──
    test_spider = PmosCrawler(base_url=base_url, cookie=cookie, unit_id=unit_id)
    test_spider.fetch_csrf_token()
    pmos_status = check_pmos_connectivity(test_spider)
    if pmos_status in ("fatal", "auth_failed"):
        logger.error("=" * 55)
        logger.error("PMOS 服务器连接失败，无法开始回填。")
        if pmos_status == "auth_failed":
            logger.error("原因：Cookie 已过期，请在办公电脑浏览器 F12 获取新 Cookie")
            logger.error("      更新 config.json 中的 'cookie' 字段后重试")
        else:
            logger.error("原因：网络不可达，请确认在国网内网环境运行")
            logger.error("提示：本程序必须在国网内网办公电脑上运行")
        logger.error("=" * 55)
        sys.exit(1)

    # ── 逐日爬取 ──
    progress = load_progress()
    stats = {"success": 0, "skip": 0, "failed": 0, "total_upserted": 0}
    errors: list[dict] = []
    fatal_encountered = False

    for idx, date_str in enumerate(to_crawl):
        pct = (idx + 1) / len(to_crawl) * 100
        logger.info("[%d/%d (%.1f%%)] %s", idx + 1, len(to_crawl), pct, date_str)

        spider = PmosCrawler(base_url=base_url, cookie=cookie, unit_id=unit_id)
        spider.fetch_csrf_token()

        retries = 0
        result = None
        while retries < args.max_retry:
            result = crawl_single_day(spider, date_str, unit_id, db_cfg, no_db=args.no_db)
            if result["status"] in ("ok", "partial", "skip"):
                break
            # 致命错误 → 不重试
            if result.get("fatal"):
                logger.error(f"  ⛔ 致命错误，停止重试")
                fatal_encountered = True
                break
            retries += 1
            if retries < args.max_retry:
                wait = args.delay * 2 ** retries
                logger.warning("  失败，%.0fs 后重试 (%d/%d)...", wait, retries, args.max_retry)
                time.sleep(wait)

        # ── 致命错误 → 立即退出 ──
        if fatal_encountered:
            logger.error("  ⛔ 检测到致命错误（网络/SSL），终止回填")
            if result:
                for e in result.get("errors", []):
                    logger.error(f"    → {e}")
            break

        # ── 统计 ──
        if result is None:
            logger.error("  ❌ 所有重试均失败")
            stats["failed"] += 1
            errors.append({"date": date_str, "error": "max retries exceeded"})
        elif result["status"] == "skip":
            logger.info("  ⏭ 跳过 (change_date 失败)")
            stats["skip"] += 1
        else:
            da_cnt = result.get("da", 0)
            rt_cnt = result.get("rt", 0)
            market_cnt = result.get("market", 0)
            status_icon = "✅" if result["status"] == "ok" else "⚠"
            logger.info(
                f"  {status_icon} DA={da_cnt} RT={rt_cnt} 市场={market_cnt}"
                + (f" ({result['errors'][0]})" if result["errors"] else "")
            )
            if da_cnt + rt_cnt + market_cnt > 0:
                stats["success"] += 1
                stats["total_upserted"] += da_cnt + rt_cnt
            else:
                stats["skip"] += 1

        # ── 每 10 天保存一次进度 ──
        if (idx + 1) % 10 == 0:
            progress["last_success_date"] = date_str
            progress["total_crawled"] = stats["success"]
            save_progress(progress)

        # ── 请求间隔 ──
        time.sleep(args.delay)

    # ── 最终统计 ──
    save_progress({
        "last_success_date": to_crawl[-1] if stats["success"] else None,
        "total_crawled": stats["success"],
        "total_days": len(to_crawl),
        "skipped_dates": [],
    })

    print("\n" + "=" * 55)
    print("  回填完成" if not fatal_encountered else "  回填已终止（网络问题）")
    print("=" * 55)
    logger.info("成功: %d 天 | 跳过: %d | 失败: %d", stats["success"], stats["skip"], stats["failed"])
    logger.info("共写入 %d 条记录", stats["total_upserted"])
    if errors:
        logger.warning("失败详情:")
        for e in errors[:10]:
            logger.warning("  %s: %s", e["date"], e["error"])
        if len(errors) > 10:
            logger.warning("  ... 共 %d 个失败", len(errors))

    if stats["failed"] > 0 or fatal_encountered:
        sys.exit(1)


if __name__ == "__main__":
    main()
