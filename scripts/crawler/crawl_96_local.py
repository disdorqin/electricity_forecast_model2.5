#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
96 点（15 分钟）市场数据本地爬虫 —— 甲方电脑专用（无 Python / 无云端 DB）

从山东省电力交易网站（PMOS）爬取全省市场特征 96 点数据，**预测(DaJyxxPlDa)
与实际(DaJyxxPlYx)合并成一张总表**，所有特征列都在同一张表里，
持续增量追加，不依赖云端数据库。

用法（脚本模式 / exe 模式通用）：
  crawl_96_local.exe --start 2022-01-01            # 从 2022-01-01 一直爬到今天（增量续爬）
  crawl_96_local.exe                               # 只补爬最近 14 天
  crawl_96_local.exe --start 2022-01-01 --end 2026-08-01  # 指定区间
  crawl_96_local.exe --date 2026-08-10             # 指定爬某一天
  crawl_96_local.exe --dry-run                     # 只显示待爬日期，不实际爬
  crawl_96_local.exe --ssl-check                   # 排查 SSL/网络连通性

依赖文件（与 exe 同目录）：
  config.json         # PMOS 登录 Cookie（从浏览器 F12 复制，见 README）
  config.example.json # 配置模板

输出（exe 同目录）：
  output_96/
    pmos_96_全量.csv     # 唯一总表：每天 96 行，预测+实际全部特征列合并
    （可选）预测_YYYY-MM-DD.csv / 实际_YYYY-MM-DD.csv 每日分表（--split 开启）
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# ── 屏蔽 SSL 警告（必须在任何网络导入之前生效） ─────────────────────
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONWARNINGS"] = "ignore::urllib3.exceptions.InsecureRequestWarning"
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
# ─────────────────────────────────────────────────────────────────

# ── PyInstaller / 路径 ──────────────────────────────────────────────
_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    BASE_DIR = Path(__file__).resolve().parents[2]

for _p in (str(BASE_DIR), str(BASE_DIR / "scripts" / "crawler")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# PyInstaller 静态分析（打包后 scripts.crawler.crawl 才是模块全名）
try:
    from scripts.crawler.crawl import PmosCrawler, parse_number  # noqa: E402
except ImportError:
    from crawl import PmosCrawler, parse_number  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crawl_96_local")

CRAWLER_DIR = BASE_DIR if _FROZEN else BASE_DIR / "scripts" / "crawler"
CONFIG_PATH = CRAWLER_DIR / "config.json"
OUT_DIR = BASE_DIR / "output_96"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_FILE = OUT_DIR / "pmos_96_全量.csv"

# 预测接口字段（DaJyxxPlDa）→ 中文列名
FORECAST_ZH = {
    "systemload": "直调负荷预测",
    "dfdcload": "地方电厂出力预测",
    "excload": "外电预测",
    "fdload": "风电预测",
    "gfload": "光伏预测",
    "sytsjz": "核电预测",
    "selfunit": "自备电厂预测",
    "syjzzj": "试验机组预测",
}

# 实际接口字段（DaJyxxPlYx）→ 中文列名
ACTUAL_ZH = {
    "systemload": "直调负荷实际",
    "dfdcload": "地方电厂出力实际",
    "excload": "外电实际",
    "fdload": "风电实际",
    "gfload": "光伏实际",
    "hdload": "核电实际",
    "zbload": "自备电厂实际",
    "syjzload": "试验机组实际",
    "cxload": "抽蓄实际",
}

# 总表列顺序
TABLE_COLUMNS = ["market_date", "时段"] + list(FORECAST_ZH.values()) + list(ACTUAL_ZH.values())


def _sanitize_config_text(raw: str) -> str:
    """清理 JSON 文本中的非法控制字符（浏览器复制 Cookie 时常混入换行/制表符）。"""
    import re as _re

    def _fix_str(m: _re.Match) -> str:
        return m.group(0).replace("\n", "").replace("\r", "").replace("\t", "")

    pattern = _re.compile(r'"((?:[^"\\]|\\.)*)"')
    return pattern.sub(_fix_str, raw)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error("配置文件不存在: %s", CONFIG_PATH)
        logger.error("请复制 config.example.json 为 config.json，并填入 Cookie")
        sys.exit(1)
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        cfg: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("config.json 含非法控制字符，已自动清理后重试")
        cfg = json.loads(_sanitize_config_text(raw))
    if not cfg.get("cookie", "").strip():
        logger.error("config.json 中 cookie 为空")
        logger.error("请在浏览器登录 PMOS 后 F12 → Network → 复制 Cookie 填入 config.json")
        sys.exit(1)
    return cfg


# ── 总表读写 ────────────────────────────────────────────────────────
def table_existing_dates() -> set[str]:
    """读取总表里已有哪些 market_date（增量续爬去重用）。"""
    if not TABLE_FILE.exists():
        return set()
    have = set()
    try:
        with open(TABLE_FILE, "r", encoding="utf-8-sig") as f:
            rd = csv.reader(f)
            next(rd, None)  # 跳过表头
            for row in rd:
                if row and row[0]:
                    have.add(row[0])
    except Exception as e:
        logger.warning("读取总表失败: %s", e)
    return have


def _to_row(date_str: str, period: str, f: dict, a: dict) -> list:
    """把一天的预测行+实际行合成总表的一行（按 时段 对齐）。"""
    row = [date_str, period]
    for col in TABLE_COLUMNS[2:]:
        # 从预测映射取，再从实际映射取
        v = None
        for src, zh in FORECAST_ZH.items():
            if zh == col:
                v = f.get(src)
                break
        if v is None:
            for src, zh in ACTUAL_ZH.items():
                if zh == col:
                    v = a.get(src)
                    break
        row.append("" if v is None else v)
    return row


def append_day_to_table(date_str: str, f_rows: list[dict], a_rows: list[dict]) -> None:
    """把一天的预测+实际数据合并追加到总表（先删旧日期行防重复，再追加）。"""
    f_idx = {r.get("Periodid", ""): r for r in (f_rows or [])}
    a_idx = {r.get("Periodid", ""): r for r in (a_rows or [])}
    periods = sorted(set(f_idx) | set(a_idx))
    if not periods:
        logger.warning("%s 预测/实际均无数据，跳过写表", date_str)
        return

    # 读取现有总表（剔除该日期旧行）
    lines = []
    if TABLE_FILE.exists():
        with open(TABLE_FILE, "r", encoding="utf-8-sig") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            if header:
                lines.append(header)
            for row in rd:
                if row and row[0] != date_str:
                    lines.append(row)

    if not lines:
        lines.append(TABLE_COLUMNS)

    # 追加该日期 96 行（按 period 排序）
    for p in periods:
        lines.append(_to_row(date_str, p, f_idx.get(p, {}), a_idx.get(p, {})))

    with open(TABLE_FILE, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerows(lines)
    logger.info("总表已更新 %s -> %s (%d 行)", date_str, TABLE_FILE.name, len(lines))


def split_save(date_str: str, f_rows: list[dict], a_rows: list[dict]) -> None:
    """可选：每日分表（预测/实际各一个文件）。"""
    if f_rows:
        _save_single(date_str, f_rows, FORECAST_ZH, "预测")
    if a_rows:
        _save_single(date_str, a_rows, ACTUAL_ZH, "实际")


def _save_single(date_str: str, rows: list[dict], zh_map: dict, prefix: str) -> None:
    cols = ["market_date", "时段"] + list(zh_map.values())
    path = OUT_DIR / f"{prefix}_{date_str}.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([date_str, r.get("Periodid", "")] + [r.get(k, "") for k in zh_map])
    logger.info("已保存 %s -> %s", prefix, path.name)


# ── 爬取单日 ─────────────────────────────────────────────────────────
def crawl_one_day(date_str: str, cfg: dict, split: bool = False) -> dict:
    spider = PmosCrawler(
        base_url=cfg.get("base_url", "https://pmos.sd.sgcc.com.cn:18080/trade"),
        cookie=cfg.get("cookie", ""),
        unit_id=cfg.get("unit_id", ""),
    )
    if not spider.fetch_csrf_token():
        raise RuntimeError(
            "CSRF token 获取失败。若提示 SSL/Connection/RemoteDisconnected 等网络错误，"
            "请检查网络连通性（国网内网才能访问 PMOS）并确认电脑未走失效代理；"
            "若提示'CSRF token not found'且 index.do 响应很短，才是 Cookie 过期，"
            "请重新从浏览器复制 Cookie"
        )
    if not spider.change_date(date_str):
        raise RuntimeError(f"change_date({date_str}) 失败")

    time.sleep(1.5)

    result = {"date": date_str, "forecast": 0, "actual": 0}

    f_rows: list[dict] = []
    try:
        f_rows = spider.crawl_market_overview()
        result["forecast"] = len(f_rows)
        logger.info("  %s 预测 → %d 行", date_str, len(f_rows))
    except Exception as e:
        logger.warning("%s 预测爬取失败: %s", date_str, e)

    a_rows: list[dict] = []
    try:
        a_rows = spider.crawl_market_overview_actual()
        result["actual"] = len(a_rows)
        logger.info("  %s 实际 → %d 行", date_str, len(a_rows))
    except Exception as e:
        logger.warning("%s 实际爬取失败: %s", date_str, e)

    # 合并写入总表（预测/实际任一成功即写，另一侧留空）
    if f_rows or a_rows:
        append_day_to_table(date_str, f_rows, a_rows)
        if split:
            split_save(date_str, f_rows, a_rows)
    else:
        logger.warning("%s 预测/实际均爬取失败，未写表", date_str)

    return result


# ── 主流程 ───────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="96点市场数据本地爬虫（预测+实际合并总表，增量追加）")
    parser.add_argument("--start", help="开始日期 (YYYY-MM-DD)，默认总表最新日+1，无表则最近14天")
    parser.add_argument("--end", help="结束日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--date", help="指定爬某一天 (YYYY-MM-DD)")
    parser.add_argument("--lookback", type=int, default=14, help="无 start 时补爬最近 N 天")
    parser.add_argument("--dry-run", action="store_true", help="仅显示待爬日期")
    parser.add_argument("--split", action="store_true", help="额外保存每日分表（预测/实际分开）")
    parser.add_argument("--delay", type=float, default=2.0, help="请求间隔秒数")
    parser.add_argument("--ssl-check", action="store_true", help="仅检测 SSL/网络连通性（排查用）")
    args = parser.parse_args()

    print("=" * 55)
    print("  96点市场数据本地爬虫（预测+实际合并总表）")
    print(f"  总表: {TABLE_FILE}")
    print("=" * 55)

    cfg = load_config()

    if args.ssl_check:
        return _ssl_check(cfg)

    # 确定日期范围
    if args.date:
        dates = [args.date]
    else:
        end_dt = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else date.today()
        if args.start:
            start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
        else:
            have = table_existing_dates()
            if have:
                start_dt = datetime.strptime(max(have), "%Y-%m-%d").date() + timedelta(days=1)
            else:
                start_dt = end_dt - timedelta(days=args.lookback)
        dates = []
        d = start_dt
        while d <= end_dt:
            dates.append(d.isoformat())
            d += timedelta(days=1)

    # 增量去重：跳过总表里已有的日期
    have = table_existing_dates()
    todo = [d for d in dates if d not in have]
    skipped = len(dates) - len(todo)

    print(f"\n待爬日期: {len(dates)} 天（其中 {skipped} 天已在总表，跳过）")
    if not todo:
        print("✅ 日期范围内数据已全部爬取，无需补爬")
        return 0

    if args.dry_run:
        print("DRY RUN 待爬日期:", ", ".join(todo[:10]) + (f" ... 共 {len(todo)} 天" if len(todo) > 10 else ""))
        return 0

    results = []
    for i, d in enumerate(todo):
        print(f"\n── [{i+1}/{len(todo)}] {d} ──")
        for attempt in range(2):
            try:
                r = crawl_one_day(d, cfg, split=args.split)
                results.append(r)
                break
            except Exception as e:
                logger.warning("第 %d 次失败: %s", attempt + 1, e)
                time.sleep(3)
        else:
            logger.error("⛔ 重试耗尽，跳过 %s", d)
        if i < len(todo) - 1:
            time.sleep(args.delay)

    ok = sum(1 for r in results if r["forecast"] > 0 or r["actual"] > 0)
    print(f"\n{'='*55}")
    print(f"完成：成功 {ok}/{len(todo)} 天")
    print(f"总表: {TABLE_FILE}")
    if TABLE_FILE.exists():
        import itertools
        with open(TABLE_FILE, "r", encoding="utf-8-sig") as f:
            nrows = sum(1 for _ in f) - 1
        print(f"总表当前行数（不含表头）: {nrows}")
    return 0 if ok == len(todo) else 1


def _ssl_check(cfg: dict) -> int:
    """SSL/网络自检：打印 OpenSSL 版本 + 实际请求 PMOS，定位连接问题。"""
    import ssl

    print(f"\n[SSL 自检] OpenSSL: {ssl.OPENSSL_VERSION}")
    print(f"[SSL 自检] Python: {__import__('sys').version.split()[0]}")
    print(f"[SSL 自检] base_url: {cfg.get('base_url')}")
    spider = PmosCrawler(
        base_url=cfg.get("base_url", "https://pmos.sd.sgcc.com.cn:18080/trade"),
        cookie=cfg.get("cookie", ""),
        unit_id=cfg.get("unit_id", ""),
    )
    try:
        ok = spider.fetch_csrf_token()
        print(f"\n[结果] CSRF token 获取: {'成功 ✓' if ok else '失败'}")
        return 0 if ok else 1
    except Exception as e:
        print(f"\n[结果] 连接失败: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("用户中断")
        sys.exit(130)
    except Exception as e:
        logger.exception("程序异常: %s", e)
        sys.exit(1)
