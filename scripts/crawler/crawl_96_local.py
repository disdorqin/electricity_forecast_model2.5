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
  crawl_96_local.exe --auth-only                   # 只登录/刷新 Cookie，不爬数据

依赖文件（与 exe 同目录）：
  config.json         # PMOS 登录 Cookie（从浏览器 F12 复制，见 README）
  config.example.json # 配置模板

输出（exe 同目录）：
  output_96/
    crawler.log              # 唯一追加式运行日志，所有后续运行继续写入此文件
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

BUILD_VERSION = "2026-08-23-auth1"

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
    from scripts.crawler.crawl import PmosCrawler, parse_number, period_no_from_time  # noqa: E402
except ImportError:
    from crawl import PmosCrawler, parse_number, period_no_from_time  # noqa: E402

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
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
TABLE_FILE = OUT_DIR / "pmos_96_全量.csv"


def _configure_run_log() -> None:
    """把所有模块日志追加到一个文件；重复启动只追加，不创建新日志文件。"""
    log_path = OUT_DIR / "crawler.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    resolved = str(log_path.resolve()).lower()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and str(Path(handler.baseFilename).resolve()).lower() == resolved:
            return
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)


_configure_run_log()

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

# 机组级日前/实时数据。两套接口严格分开，不能互相回填。
DA_UNIT_ZH = {
    "cqPrice": "日前出清价格",
    "power": "日前出力",
    "energy": "日前电量",
    "kt": "日前开机状态",
    "bq": "日前电源类型",
}
RT_UNIT_ZH = {
    "cqPrice": "实时出清价格",
    "power": "实时出力",
    "energy": "实时电量",
    "kt": "实时开机状态",
    "bq": "实时电源类型",
}

# HAR5 中的备用数据是 96 点 × 2 类型，保留到宽表；检修/抽蓄等日级或列表级
# 信息写入 raw/{date}.json，不把日级值错误广播到每个 15 分钟点。
RESERVE_ZH = {
    "positive": "正备用预测",
    "negative": "负备用预测",
}

# 总表列顺序：核心预测、核心实际、机组级价格/出力/状态、备用。
# 所有列都来自独立接口；没有任何 forecast -> actual 的 fallback。
TABLE_COLUMNS = (
    ["market_date", "时段"]
    + list(FORECAST_ZH.values())
    + list(ACTUAL_ZH.values())
    + list(DA_UNIT_ZH.values())
    + list(RT_UNIT_ZH.values())
    + list(RESERVE_ZH.values())
)

MONITORED_CORE_FIELDS = ("systemload", "dfdcload", "excload", "fdload", "gfload")


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
        logger.error("请复制 config.example.json 为 config.json，并填入账号密码或 Cookie")
        sys.exit(1)
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    try:
        cfg: dict = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("config.json 含非法控制字符，已自动清理后重试")
        cfg = json.loads(_sanitize_config_text(raw))
    # Cookie 允许为空：认证运行时会按 auth_mode 使用账号密码自动登录，
    # 失败后可切换到真实浏览器读取登录态。
    return cfg


# ── 总表读写 ────────────────────────────────────────────────────────
def _as_float(value: Any) -> float | None:
    """将接口值转成数字；空值保持 None，绝不使用另一来源补值。"""
    if value is None or str(value).strip() in ("", "-", "--", "null", "None"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _period_key(value: Any) -> int:
    try:
        return period_no_from_time(str(value).strip())
    except Exception:
        return -1


def _rows_by_period(rows: list[dict] | None) -> dict[str, dict]:
    return {
        str(row.get("Periodid", row.get("periodid", ""))).strip(): row
        for row in (rows or [])
        if str(row.get("Periodid", row.get("periodid", ""))).strip()
    }


def _complete_96(rows: list[dict] | None, required_fields: tuple[str, ...]) -> bool:
    """严格检查一套接口是否返回完整且可用的 96 点。"""
    idx = _rows_by_period(rows)
    if len(rows or []) != 96 or len(idx) != 96:
        return False
    if set(_period_key(p) for p in idx) != set(range(1, 97)):
        return False
    for pid in idx:
        for field in required_fields:
            if _as_float(idx[pid].get(field)) is None:
                return False
    return True


def _same_ratio(f_rows: list[dict], a_rows: list[dict], field: str) -> tuple[float, int]:
    """返回预测/实际同值比例；常量零列不作为污染证据。"""
    f_idx, a_idx = _rows_by_period(f_rows), _rows_by_period(a_rows)
    pairs = []
    for pid in set(f_idx) & set(a_idx):
        f, a = _as_float(f_idx[pid].get(field)), _as_float(a_idx[pid].get(field))
        if f is not None and a is not None:
            pairs.append((f, a))
    if len(pairs) < 48:
        return 0.0, len(pairs)
    # 只有两套数据都不是近似常量时才将同值比例作为污染证据。
    values = [x for pair in pairs for x in pair]
    if max(values) - min(values) < 1e-9:
        return 0.0, len(pairs)
    same = sum(abs(f - a) <= 1e-9 for f, a in pairs)
    return same / len(pairs), len(pairs)


def validate_forecast_actual_separation(
    f_rows: list[dict], a_rows: list[dict]
) -> tuple[bool, dict[str, Any]]:
    """防止 actual/fcast 再次互相拷贝。

    单个稳定字段偶尔相同是允许的；但直调负荷或两项以上可变核心字段
    96 点几乎完全相同，说明接口错用或发生拷贝，整天拒绝入总表。
    """
    ratios = {}
    suspect = []
    for field in MONITORED_CORE_FIELDS:
        ratio, n = _same_ratio(f_rows, a_rows, field)
        ratios[field] = {"same_ratio": round(ratio, 6), "pairs": n}
        if ratio >= 0.99:
            suspect.append(field)
    copied = "systemload" in suspect or len(suspect) >= 2
    return (not copied), {"same_ratio": ratios, "suspect_fields": suspect, "copied": copied}


def _read_table_rows() -> tuple[list[str], list[dict[str, str]]]:
    if not TABLE_FILE.exists():
        return [], []
    with open(TABLE_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def table_complete_dates() -> set[str]:
    """只把 96 行完整且通过 actual/fcast 审计的日期视为已完成。

    这样旧版污染总表不会阻止新程序重爬覆盖。
    """
    if not TABLE_FILE.exists():
        return set()
    _, rows = _read_table_rows()
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("market_date", "")), []).append(row)
    complete: set[str] = set()
    f_cols = list(FORECAST_ZH.values())
    a_cols = list(ACTUAL_ZH.values())
    for day, day_rows in grouped.items():
        if len(day_rows) != 96:
            continue
        periods = {_period_key(row.get("时段", "")) for row in day_rows}
        if periods != set(range(1, 97)):
            continue
        # 旧表列名可能不同；只对当前 rich schema 做完成判定。
        if not all(str(row.get(c, "")).strip() for row in day_rows for c in f_cols[:5] + a_cols[:5]):
            continue
        f_rows = [{k: row.get(v, "") for k, v in FORECAST_ZH.items()} for row in day_rows]
        a_rows = [{k: row.get(v, "") for k, v in ACTUAL_ZH.items()} for row in day_rows]
        ok, _ = validate_forecast_actual_separation(f_rows, a_rows)
        if ok:
            complete.add(day)
    return complete


def table_existing_dates() -> set[str]:
    """兼容旧调用：只返回真正完整并通过真实性审计的日期。"""
    try:
        return table_complete_dates()
    except Exception as e:
        logger.warning("读取总表完成日期失败: %s", e)
        return set()


def _mapped_row(pid: str, f: dict, a: dict, da: dict, rt: dict, reserve: dict) -> list:
    """按严格来源映射一行；没有值就留空，不跨来源复制。"""
    row = [pid]
    for src in FORECAST_ZH:
        row.append(f.get(src, ""))
    for src in ACTUAL_ZH:
        row.append(a.get(src, ""))
    for src in DA_UNIT_ZH:
        row.append(da.get(src, ""))
    for src in RT_UNIT_ZH:
        row.append(rt.get(src, ""))
    row.extend([reserve.get("positive", ""), reserve.get("negative", "")])
    return row


def _reserve_by_period(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows or []:
        pid = str(r.get("PERIODID", r.get("Periodid", ""))).strip()
        typ = str(r.get("TYPE", "")).strip()
        key = "positive" if typ == "正备用" else "negative" if typ == "负备用" else ""
        if pid and key:
            out.setdefault(pid, {})[key] = r.get("ZBY", "")
    return out


def append_day_to_table(
    date_str: str,
    f_rows: list[dict],
    a_rows: list[dict],
    da_rows: list[dict],
    rt_rows: list[dict],
    reserve_rows: list[dict],
) -> None:
    """原子覆盖一天数据；只接受完整且通过来源隔离审计的日期。"""
    if not _complete_96(f_rows, MONITORED_CORE_FIELDS) or not _complete_96(a_rows, MONITORED_CORE_FIELDS):
        raise ValueError(f"{date_str} 核心预测/实际不是完整96点，拒绝写入")
    separated, audit = validate_forecast_actual_separation(f_rows, a_rows)
    if not separated:
        raise ValueError(f"{date_str} 疑似 actual/fcast 拷贝，拒绝写入: {audit}")

    f_idx, a_idx = _rows_by_period(f_rows), _rows_by_period(a_rows)
    da_idx, rt_idx = _rows_by_period(da_rows), _rows_by_period(rt_rows)
    reserve_idx = _reserve_by_period(reserve_rows)
    new_rows = []
    for pid in sorted(f_idx, key=_period_key):
        new_rows.append([date_str] + _mapped_row(
            pid, f_idx[pid], a_idx[pid], da_idx.get(pid, {}), rt_idx.get(pid, {}), reserve_idx.get(pid, {})
        ))

    # 读取旧数据时按列名迁移，允许旧版 19 列总表被新 rich schema 覆盖升级。
    old_header, old_rows = _read_table_rows()
    kept = []
    for old in old_rows:
        if str(old.get("market_date", "")) == date_str:
            continue
        kept.append([old.get(col, "") for col in TABLE_COLUMNS])

    tmp = TABLE_FILE.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(TABLE_COLUMNS)
        wr.writerows(kept)
        wr.writerows(new_rows)
    os.replace(tmp, TABLE_FILE)
    logger.info("总表已原子更新 %s -> %s (新增96行)", date_str, TABLE_FILE)


def save_raw_bundle(date_str: str, bundle: dict[str, Any]) -> None:
    """保存该日所有接口原始 JSON，便于审计和后续重新映射。"""
    path = RAW_DIR / f"{date_str}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


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
def _new_spider(cfg: dict) -> PmosCrawler:
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
    return spider


def crawl_one_day(
    date_str: str,
    cfg: dict,
    split: bool = False,
    spider: PmosCrawler | None = None,
) -> dict:
    """抓取一天并先落原始包，再按严格审计结果决定是否写入总表。

    ``spider`` 可由主循环复用，减少每天重新认证的时间；网络异常时主循环会
    丢弃该实例并重新建立会话。核心预测/实际不完整或疑似互相拷贝时，原始包
    仍会保存，但该日不会进入总表，也不会被标记为成功。
    """
    spider = spider or _new_spider(cfg)
    if not spider.change_date(date_str):
        raise RuntimeError(f"change_date({date_str}) 失败")

    time.sleep(1.5)

    result = {
        "date": date_str,
        "forecast": 0,
        "actual": 0,
        "day_ahead_unit": 0,
        "realtime_unit": 0,
        "complete": False,
        "audit": {},
    }
    # 仅供 main() 复用认证会话；不会写入 raw JSON。
    result["_spider"] = spider

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

    da_rows: list[dict] = []
    try:
        da_rows = spider.crawl_day_ahead()
        result["day_ahead_unit"] = len(da_rows)
        logger.info("  %s 日前机组明细 → %d 行", date_str, len(da_rows))
    except Exception as e:
        logger.warning("%s 日前机组明细爬取失败（核心表保留空值，不跨源补值）: %s", date_str, e)

    rt_rows: list[dict] = []
    try:
        rt_rows = spider.crawl_realtime()
        result["realtime_unit"] = len(rt_rows)
        logger.info("  %s 实时机组明细 → %d 行", date_str, len(rt_rows))
    except Exception as e:
        logger.warning("%s 实时机组明细爬取失败（核心表保留空值，不跨源补值）: %s", date_str, e)

    optional = spider.crawl_optional_market_data()
    separated, audit = validate_forecast_actual_separation(f_rows, a_rows)
    core_complete = (
        _complete_96(f_rows, MONITORED_CORE_FIELDS)
        and _complete_96(a_rows, MONITORED_CORE_FIELDS)
    )
    result["audit"] = {
        "core_complete": core_complete,
        "forecast_complete": _complete_96(f_rows, MONITORED_CORE_FIELDS),
        "actual_complete": _complete_96(a_rows, MONITORED_CORE_FIELDS),
        "separated": separated,
        **audit,
    }

    bundle = {
        "schema_version": "pmos-96-raw-v2",
        "market_date": date_str,
        "captured_at": datetime.now().astimezone().isoformat(),
        "base_url": spider.base_url,
        "status": "complete" if core_complete and separated else "rejected",
        "source": {
            "forecast": "DaJyxxPlDa.do?method=getNewDetailGridList",
            "actual": "DaJyxxPlYx.do?method=getNewDetailGridList",
            "day_ahead_unit": "DaJyjgfbPlantQuery24.do?method=getDetail",
            "realtime_unit": "YxJyjgfbPlantQuery24.do?method=getDetail",
            "optional": "HAR5-discovered PMOS DataTables endpoints",
        },
        "audit": result["audit"],
        "forecast": f_rows,
        "actual": a_rows,
        "day_ahead_unit": da_rows,
        "realtime_unit": rt_rows,
        "optional": optional,
    }
    save_raw_bundle(date_str, bundle)
    result["raw_path"] = str(RAW_DIR / f"{date_str}.json")

    if core_complete and separated:
        append_day_to_table(
            date_str,
            f_rows,
            a_rows,
            da_rows,
            rt_rows,
            optional.get("reserve_da", []),
        )
        if split:
            split_save(date_str, f_rows, a_rows)
        result["complete"] = True
    else:
        logger.error(
            "⛔ %s 核心96点未通过完整性/来源隔离审计，已保存 raw，但拒绝写入总表: %s",
            date_str,
            result["audit"],
        )

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
    parser.add_argument("--force", action="store_true", help="强制重爬范围内日期（用于替换旧/污染数据）")
    parser.add_argument("--delay", type=float, default=2.0, help="请求间隔秒数")
    parser.add_argument("--ssl-check", action="store_true", help="仅检测 SSL/网络连通性（排查用）")
    parser.add_argument("--auth-only", action="store_true", help="仅完成认证并刷新 config.json，不爬数据")
    parser.add_argument("--auth-mode", choices=("auto", "account", "browser", "static"), help="覆盖 config.json 的认证模式")
    parser.add_argument("--auth-timeout-sec", type=int, help="认证等待秒数，覆盖配置")
    parser.add_argument("--auth-retries", type=int, help="账号密码自动登录重试次数，覆盖配置")
    parser.add_argument("--skip-auth", action="store_true", help="跳过认证（仅用于已有 Cookie 的离线/兼容测试）")
    args = parser.parse_args()

    print("=" * 55)
    print("  96点市场数据本地爬虫（预测+实际合并总表）")
    print(f"  总表: {TABLE_FILE}")
    print(f"  日志: {OUT_DIR / 'crawler.log'}")
    print("=" * 55)

    cfg = load_config()
    logger.info(
        "RUN start version=%s frozen=%s pid=%s auth_only=%s skip_auth=%s args=%s",
        BUILD_VERSION, _FROZEN, os.getpid(), args.auth_only, args.skip_auth,
        {k: v for k, v in vars(args).items() if k not in {"auth_mode", "auth_timeout_sec", "auth_retries"} or v is not None},
    )

    if not args.skip_auth:
        try:
            from scripts.crawler.auth_runtime import ensure_authenticated_config

            ensure_authenticated_config(
                cfg,
                CONFIG_PATH,
                base_dir=BASE_DIR,
                mode=args.auth_mode,
                timeout_sec=args.auth_timeout_sec,
                max_retries=args.auth_retries,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("RUN auth FAIL: %s", exc)
            print(f"\n认证失败：{exc}")
            print(f"请查看追加日志：{OUT_DIR / 'crawler.log'}")
            return 2
    else:
        logger.warning("RUN auth SKIP：仅用于兼容/离线测试")

    if args.auth_only:
        logger.info("RUN auth-only PASS")
        print("\n✅ 认证完成，Cookie 已写回 config.json")
        return 0

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
    todo = list(dates) if args.force else [d for d in dates if d not in have]
    skipped = len(dates) - len(todo)

    print(f"\n待爬日期: {len(dates)} 天（其中 {skipped} 天已在总表，跳过）")
    if not todo:
        print("✅ 日期范围内数据已全部爬取，无需补爬")
        return 0

    if args.dry_run:
        print("DRY RUN 待爬日期:", ", ".join(todo[:10]) + (f" ... 共 {len(todo)} 天" if len(todo) > 10 else ""))
        return 0

    results = []
    spider: PmosCrawler | None = None
    for i, d in enumerate(todo):
        print(f"\n── [{i+1}/{len(todo)}] {d} ──")
        for attempt in range(2):
            try:
                r = crawl_one_day(d, cfg, split=args.split, spider=spider)
                # 认证会话可复用；只有发生异常才在下一次尝试重建。
                spider = r.pop("_spider", spider)
                results.append(r)
                break
            except Exception as e:
                logger.warning("第 %d 次失败: %s", attempt + 1, e)
                spider = None
                time.sleep(3)
        else:
            logger.error("⛔ 重试耗尽，跳过 %s", d)
        if i < len(todo) - 1:
            time.sleep(args.delay)

    ok = sum(1 for r in results if r.get("complete", False))
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
