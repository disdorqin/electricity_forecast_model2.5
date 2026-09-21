#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
96 点（15 分钟）市场数据爬虫 —— 当前 v6 生产入口源码

从山东省电力交易网站（PMOS）爬取全省市场特征 96 点数据，**预测与实际**
合并成一张总表；日前电价使用二次出清最终版，实时电价使用正式版，
所有特征列都在同一张表里，
    持续增量追加，保留部分数据并把本地数据同步到云端 `epf_pmos_96_full`。

用法（脚本模式 / exe 模式通用）：
  crawl_96_auto_v8.exe --start 2022-01-01          # 从 2022-01-01 一直爬到今天（增量续爬）
  crawl_96_auto_v8.exe                              # 只补爬最近 14 天
  crawl_96_auto_v8.exe --start 2022-01-01 --end 2026-08-01  # 指定区间
  crawl_96_auto_v8.exe --date 2026-08-10            # 指定爬某一天
  crawl_96_auto_v8.exe --dry-run                    # 只显示待爬日期，不实际爬
  crawl_96_auto_v8.exe --ssl-check                  # 排查 SSL/网络连通性
  crawl_96_auto_v8.exe --auth-only                  # 只登录/刷新 Cookie，不爬数据

依赖文件（与 exe 同目录）：
  config.json         # PMOS 登录 Cookie（从浏览器 F12 复制，见 README）
  config.example.json # 配置模板

输出（exe 同目录）：
  output_96/
    crawler.log              # 详细追加式运行日志
    report.json              # 累计结构化运行报告
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

BUILD_VERSION = "2026-09-21-browser-path-detection-fix-v9"

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
    BASE_DIR = Path(__file__).resolve().parents[3]

for _p in (str(BASE_DIR), str(BASE_DIR / "scripts" / "crawler")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 运行时依赖分层：collect.crawl 负责 PMOS 接口，auth.auto_crawler 负责浏览器
# 认证，run_crawler.py 负责数据库 schema/上传。这里显式导入接口核心，便于
# PyInstaller 收集模块；不要在此恢复已归档的旧爬虫入口。
# PyInstaller 静态分析（打包后使用稳定的 scripts.crawler.collect 模块名）
try:
    from scripts.crawler.collect.crawl import PmosCrawler, parse_number, period_no_from_time  # noqa: E402
    from scripts.crawler.observability import RunReport, cookie_summary  # noqa: E402
except ImportError:
    from crawl import PmosCrawler, parse_number, period_no_from_time  # noqa: E402
    from observability import RunReport, cookie_summary  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crawl_96_local")

# 供最外层异常/人工中断处理器更新累计 report.json；正常路径仍由 main()
# 在各阶段结束时写入最终状态。
_ACTIVE_REPORTER = None

CRAWLER_DIR = BASE_DIR if _FROZEN else BASE_DIR / "scripts" / "crawler"
CONFIG_PATH = CRAWLER_DIR / "config.json"


def _resolve_runtime_output_dir(base_dir: Path, *, frozen: bool) -> Path:
    """Return the crawler-owned runtime directory for one execution mode.

    Packaged deployments intentionally keep the long-standing ``output_96``
    sibling next to the EXE. Source-mode development must not create runtime
    state in the repository root; it is isolated under ``outputs/crawl``.
    """
    if frozen:
        return base_dir / "output_96"
    return base_dir / "outputs" / "crawl" / "runtime_96"


OUT_DIR = _resolve_runtime_output_dir(BASE_DIR, frozen=_FROZEN)
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
NEXT_FORECAST_RAW_DIR = RAW_DIR / "next_forecast"
NEXT_FORECAST_RAW_DIR.mkdir(parents=True, exist_ok=True)
TABLE_FILE = OUT_DIR / "pmos_96_全量.csv"
REPORT_FILE = OUT_DIR / "report.json"
DB_CONFIG_PATH = BASE_DIR / "db_config.json"
UPLOAD_QUEUE_DIR = OUT_DIR / "upload_queue"
UPLOAD_QUEUE_DIR.mkdir(parents=True, exist_ok=True)


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
    "qwfh": "全网负荷预测",
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
    "qwfh": "全网负荷实际",
}

# RealityTmpData 与正式 RealityData 必须分列。它们可以在 snapshot 层组合，
# 但数据库层永远保留各自 provenance。
TEMP_ACTUAL_ZH = {
    "systemload": "直调负荷临时实际",
    "dfdcload": "地方电厂出力临时实际",
    "excload": "外电临时实际",
    "fdload": "风电临时实际",
    "gfload": "光伏临时实际",
    "hdload": "核电临时实际",
    "zbload": "自备电厂临时实际",
    "syjzload": "试验机组临时实际",
    "cxload": "抽蓄临时实际",
    "qwfh": "全网负荷临时实际",
}

# ForecastBoundaryData 是独立市场边界口径，绝不能覆盖 ForecastData。
BOUNDARY_ZH = {
    "qwfh": "边界全网负荷预测",
    "systemload": "边界直调负荷预测",
    "excload": "边界外电预测",
    "fdload": "边界风电预测",
    "gfload": "边界光伏预测",
    "sytsjz": "边界核电预测",
}

# 日前一次出清仅新增价格列；现有 DA_UNIT_ZH 继续表示二次/最终出清，保持兼容。
DA_FIRST_ZH = {
    "cqPrice": "日前一次出清价格",
}

# 机组级日前二次/实时数据。各套接口严格分开，不能互相回填。
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
    + list(TEMP_ACTUAL_ZH.values())
    + list(BOUNDARY_ZH.values())
    + list(DA_FIRST_ZH.values())
    + list(DA_UNIT_ZH.values())
    + list(RT_UNIT_ZH.values())
    + list(RESERVE_ZH.values())
)

MONITORED_CORE_FIELDS = ("systemload", "dfdcload", "excload", "fdload", "gfload")
BOUNDARY_CORE_FIELDS = ("qwfh", "systemload", "excload", "fdload", "gfload")
PRICE_FIELDS = ("cqPrice",)


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


def _write_cookie_config(config_path: Path, cfg: dict, cookie: str) -> None:
    """把浏览器 CDP 获取的 Cookie 原子写回配置，保留其它字段。"""
    updated = dict(cfg)
    updated["cookie"] = cookie
    tmp = config_path.with_name(f".{config_path.name}.tmp")
    tmp.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, config_path)


def _ensure_browser_state_machine(cfg: dict, reporter=None, *, force_new: bool = False) -> str:
    """调用当前浏览器认证状态机获取 Cookie，不改变后续数据接口。"""
    from dataclasses import replace
    from scripts.crawler.auth.auto_crawler.config import AuthConfig
    from scripts.crawler.auth.auto_crawler.state_machine import AuthenticationStateMachine

    auth_cfg = AuthConfig.from_file(CONFIG_PATH)
    if force_new:
        # 已有浏览器可能能完成旧门户认证，但无法继续完成 QCTC SSO；此时
        # 不再复用该实例，直接启用独立 profile 和新的 DevTools 端口。
        auth_cfg = replace(auth_cfg, browser_reuse=False)
        logger.warning("浏览器恢复：放弃当前 CDP，准备启动新的认证浏览器")
        if reporter is not None:
            reporter.stage("browser_cdp", "RETRY", mode="force_new",
                           reason="existing_browser_unusable")
    result = AuthenticationStateMachine(auth_cfg, reporter=reporter).run()
    cookie = str(result.cookie or "").strip()
    if not cookie:
        raise RuntimeError("浏览器认证完成但未读取到 Cookie")
    # 认证可能复用了扫描到的非 9222 端口，或因 9222 被占用而选择了备用端口。
    # 必须把最终端口传给 collect，否则 QCTC 会拿旧端口请求，表现为价格全空。
    if int(getattr(result, "debug_port", 0) or 0) > 0:
        cfg["debug_port"] = int(result.debug_port)
    _write_cookie_config(CONFIG_PATH, cfg, cookie)
    cfg["cookie"] = cookie
    logger.info("浏览器认证完成，Cookie 已写回配置 path=%s summary=%s（不记录值）",
                CONFIG_PATH, cookie_summary(cookie))
    if reporter is not None:
        reporter.stage("auth_cookie", "PASS", config_path=str(CONFIG_PATH), **cookie_summary(cookie))
    return cookie


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


def _rows_by_period_no(rows: list[dict] | None) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for row in rows or []:
        label = str(row.get("Periodid", row.get("periodid", ""))).strip()
        try:
            pno = _period_key(label)
        except Exception:
            continue
        if 1 <= pno <= 96:
            out[pno] = row
    return out


def _period_label(pno: int) -> str:
    return "24:00" if pno == 96 else f"{pno // 4:02d}:{(pno % 4) * 15:02d}"


def _coverage(rows: list[dict] | None, fields: tuple[str, ...] = ()) -> dict[str, Any]:
    """生成部分数据也可用的覆盖统计，不把缺失数据伪装成完整。"""
    rows = rows or []
    by_no = _rows_by_period_no(rows)
    missing = [p for p in range(1, 97) if p not in by_no]
    nonnull = {
        field: sum(_as_float(row.get(field)) is not None for row in rows)
        for field in fields
    }
    return {
        "rows": len(rows),
        "unique_periods": len(by_no),
        "valid_periods": sorted(by_no),
        "missing_periods": missing,
        "nonnull": nonnull,
        "status": "COMPLETE" if len(by_no) == 96 and all(nonnull.get(f, 0) == 96 for f in fields) else (
            "PARTIAL" if rows else "EMPTY"
        ),
    }


def _rows_have_payload(rows: list[dict] | None) -> bool:
    """判断响应是否真的带业务值，而不是只有 96 个空时段占位。"""
    for row in rows or []:
        if any(
            key not in {"Periodid", "periodid"} and str(value or "").strip() not in {"", "-", "--", "null", "None"}
            for key, value in row.items()
        ):
            return True
    return False


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
        # 旧表列名可能不同；完整日期还必须具备日前/实时价格，避免价格空的
        # 半成品阻止后续增量重爬。
        required_cols = f_cols[:5] + a_cols[:5] + ["日前出清价格", "实时出清价格"]
        if not all(str(row.get(c, "")).strip() for row in day_rows for c in required_cols):
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


def _mapped_row(
    pid: str,
    f: dict,
    a: dict,
    a_tmp: dict,
    boundary: dict,
    da_first: dict,
    da: dict,
    rt: dict,
    reserve: dict,
) -> list:
    """按严格来源映射一行；没有值就留空，不跨来源复制。"""
    row = [pid]
    for src in FORECAST_ZH:
        row.append(f.get(src, ""))
    for src in ACTUAL_ZH:
        row.append(a.get(src, ""))
    for src in TEMP_ACTUAL_ZH:
        row.append(a_tmp.get(src, ""))
    for src in BOUNDARY_ZH:
        row.append(boundary.get(src, ""))
    for src in DA_FIRST_ZH:
        row.append(da_first.get(src, ""))
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
            pno = _period_key(pid)
            if 1 <= pno <= 96:
                out.setdefault(str(pno), {})[key] = r.get("ZBY", "")
    return out


def append_day_to_table(
    date_str: str,
    f_rows: list[dict],
    a_rows: list[dict],
    da_first_rows: list[dict],
    da_rows: list[dict],
    rt_rows: list[dict],
    reserve_rows: list[dict],
    actual_temporary_rows: list[dict] | None = None,
    boundary_rows: list[dict] | None = None,
) -> None:
    """原子覆盖一天数据；部分数据也保存，缺失值保持空，不跨源补值。"""
    actual_temporary_rows = actual_temporary_rows or []
    boundary_rows = boundary_rows or []
    if not any((f_rows, a_rows, actual_temporary_rows, boundary_rows, da_first_rows, da_rows, rt_rows, reserve_rows)):
        raise ValueError(f"{date_str} 没有任何可写入的有效数据")
    separated, audit = validate_forecast_actual_separation(f_rows, a_rows)
    separation_failed = not separated
    if not separated:
        logger.error("%s 疑似 actual/fcast 拷贝，保留raw但不把可疑值写入CSV: %s", date_str, audit)
        f_rows = []
        a_rows = []

    f_idx, a_idx = _rows_by_period_no(f_rows), _rows_by_period_no(a_rows)
    a_tmp_idx = _rows_by_period_no(actual_temporary_rows)
    boundary_idx = _rows_by_period_no(boundary_rows)
    da_first_idx = _rows_by_period_no(da_first_rows)
    da_idx, rt_idx = _rows_by_period_no(da_rows), _rows_by_period_no(rt_rows)
    reserve_idx = _reserve_by_period(reserve_rows)
    new_rows = []
    for pno in range(1, 97):
        pid = _period_label(pno)
        new_rows.append([date_str] + _mapped_row(
            pid,
            f_idx.get(pno, {}).copy() | {"Periodid": pid},
            a_idx.get(pno, {}).copy() | {"Periodid": pid},
            a_tmp_idx.get(pno, {}).copy() | {"Periodid": pid},
            boundary_idx.get(pno, {}).copy() | {"Periodid": pid},
            da_first_idx.get(pno, {}).copy() | {"periodid": pid},
            da_idx.get(pno, {}).copy() | {"periodid": pid},
            rt_idx.get(pno, {}).copy() | {"periodid": pid},
            reserve_idx.get(str(pno), {}),
        ))

    # 读取旧数据时按列名迁移，并保留旧日期的非空值，避免部分重爬用空值覆盖历史有效值。
    old_header, old_rows = _read_table_rows()
    kept = []
    old_same_day: dict[int, dict[str, str]] = {}
    for old in old_rows:
        if str(old.get("market_date", "")) == date_str:
            old_same_day[_period_key(old.get("时段", ""))] = old
            continue
        kept.append([old.get(col, "") for col in TABLE_COLUMNS])

    merged_rows = []
    for row in new_rows:
        row_dict = dict(zip(TABLE_COLUMNS, row))
        old = old_same_day.get(_period_key(row_dict.get("时段", "")), {})
        for col in TABLE_COLUMNS:
            # 本次真实性审计失败时，不能把旧表中可能同样污染的 fcast/actual
            # 值重新合并回来；宁可留空，等待后续拿到可信原始响应再补写。
            blocked_old = separation_failed and (col in FORECAST_ZH.values() or col in ACTUAL_ZH.values())
            if col not in {"market_date", "时段"} and not blocked_old and str(row_dict.get(col, "")).strip() == "":
                row_dict[col] = old.get(col, "")
        merged_rows.append([row_dict.get(col, "") for col in TABLE_COLUMNS])

    tmp = TABLE_FILE.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(TABLE_COLUMNS)
        wr.writerows(kept)
        wr.writerows(merged_rows)
    os.replace(tmp, TABLE_FILE)
    logger.info("总表已原子更新 %s -> %s (保存96个结构时段，缺失值保持空)", date_str, TABLE_FILE)


def save_raw_bundle(date_str: str, bundle: dict[str, Any]) -> None:
    """保存该日所有接口原始 JSON，便于审计和后续重新映射。"""
    path = RAW_DIR / f"{date_str}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def prefetch_next_day_forecast(
    cfg: dict,
    spider: PmosCrawler | None = None,
    reporter=None,
    target_date: str | None = None,
) -> tuple[dict[str, Any], PmosCrawler, list[dict]]:
    """预取 D+1 ForecastData；9个原始预测字段全部96/96时写正式总表。"""
    target = target_date or (date.today() + timedelta(days=1)).isoformat()
    spider = spider or _new_spider(cfg, reporter=reporter)
    captured_at = datetime.now().astimezone().isoformat()
    rows: list[dict] = []
    boundary_rows: list[dict] = []
    error = ""
    boundary_error = ""
    try:
        rows = spider.crawl_market_forecast_for_date(target)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("次日预测 %s 抓取失败: %s", target, exc)
    try:
        boundary_rows = spider.crawl_market_boundary_for_date(target)
    except Exception as exc:  # noqa: BLE001
        boundary_error = f"{type(exc).__name__}: {exc}"
        logger.warning("次日边界预测 %s 抓取失败: %s", target, exc)

    required_fields = tuple(FORECAST_ZH.keys())
    complete = _complete_96(rows, required_fields)
    coverage = _coverage(rows, required_fields)
    boundary_complete = _complete_96(boundary_rows, BOUNDARY_CORE_FIELDS)
    boundary_coverage = _coverage(boundary_rows, BOUNDARY_CORE_FIELDS)
    raw_path = NEXT_FORECAST_RAW_DIR / f"{target}.json"
    payload = {
        "schema_version": "pmos-96-next-forecast-v2",
        "target_date": target,
        "captured_at": captured_at,
        "source": {
            "forecast": "QCTC informationDisclosure/ForecastData/getLoadData",
            "forecast_boundary": "QCTC informationDisclosure/ForecastBoundaryData/getLoadData",
        },
        "complete": complete,
        "coverage": coverage,
        "error": error,
        "forecast": rows,
        "forecast_boundary_complete": boundary_complete,
        "forecast_boundary_coverage": boundary_coverage,
        "forecast_boundary_error": boundary_error,
        "forecast_boundary": boundary_rows,
    }
    tmp = raw_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, raw_path)
    logger.info(
        "次日边界预测观测 %s: rows=%d complete=%s coverage=%s",
        target, len(boundary_rows), boundary_complete, boundary_coverage,
    )

    if complete:
        append_day_to_table(
            target, rows, [], [], [], [], [],
            boundary_rows=boundary_rows,
        )
        logger.info("次日预测预取成功 %s: 96行，9个ForecastData字段全部完整", target)
        if reporter is not None:
            reporter.event(
                "INFO", "NEXT_FORECAST_READY", "次日96点预测已提前获取",
                date=target, rows=len(rows), raw_path=str(raw_path), coverage=coverage,
            )
    else:
        logger.warning("次日预测 %s 未达到96点全字段完整，仅保存raw，不写正式表: %s", target, coverage)
        if reporter is not None:
            reporter.event(
                "WARN", "NEXT_FORECAST_PARTIAL", "次日预测未达到96点全字段完整，仅保存raw",
                date=target, rows=len(rows), raw_path=str(raw_path), coverage=coverage, error=error,
            )

    return {
        "date": target,
        "rows": len(rows),
        "complete": complete,
        "raw_path": str(raw_path),
        "coverage": coverage,
        "error": error,
        "forecast_boundary_rows": len(boundary_rows),
        "forecast_boundary_complete": boundary_complete,
        "forecast_boundary_coverage": boundary_coverage,
        "forecast_boundary_error": boundary_error,
    }, spider, rows


def _load_remote_db_config(cfg: dict) -> dict[str, Any]:
    """读取 EXE 同目录数据库配置；不依赖 Python/.env。"""
    raw: dict[str, Any] = {}
    configured_path = str(cfg.get("db_config_file") or "").strip()
    db_path = Path(configured_path).expanduser() if configured_path else DB_CONFIG_PATH
    if not db_path.is_absolute():
        db_path = BASE_DIR / db_path
    if db_path.exists():
        raw = json.loads(db_path.read_text(encoding="utf-8"))
    # 也支持直接放在主配置中，便于临时部署；外部 db_config.json 优先。
    aliases = {
        "host": ("host", "DB_HOST", "db_host"),
        "port": ("port", "DB_PORT", "db_port"),
        "user": ("user", "DB_USER", "db_user"),
        "password": ("password", "DB_PWD", "db_password"),
        "database": ("database", "DB", "DB_NAME", "db_name"),
    }
    merged = dict(cfg)
    merged.update(raw)
    out: dict[str, Any] = {}
    for target, keys in aliases.items():
        for key in keys:
            value = merged.get(key)
            if value not in (None, ""):
                out[target] = value
                break
    out["port"] = int(out.get("port") or 3306)
    out["connect_timeout"] = int(merged.get("connect_timeout") or merged.get("DB_CONNECT_TIMEOUT") or 10)
    missing = [k for k in ("host", "user", "password", "database") if not str(out.get(k) or "").strip()]
    if missing:
        raise RuntimeError(f"数据库配置不完整，缺少: {', '.join(missing)}；请填写 {DB_CONFIG_PATH.name}")
    return out


def _upload_bundle_to_mysql(date_str: str, raw_path: Path, db_cfg: dict[str, Any], unit_id: str) -> dict[str, int]:
    """把本地 raw 包幂等写入唯一生产目标 ``epf_pmos_96_full``。"""
    import pymysql
    from scripts.crawler.sync_db.run_crawler import upsert_full_dataset_table

    bundle = json.loads(raw_path.read_text(encoding="utf-8"))
    audit = bundle.get("audit") or {}
    if str(bundle.get("status") or "").lower() not in {"complete", "partial"}:
        raise RuntimeError(f"{date_str} 没有可同步的有效数据")
    if audit.get("copied"):
        logger.warning("%s 检测到预测/实际疑似拷贝；仅同步非可疑来源字段", date_str)
    conn = pymysql.connect(
        host=str(db_cfg["host"]), port=int(db_cfg["port"]), user=str(db_cfg["user"]),
        password=str(db_cfg["password"]), database=str(db_cfg["database"]),
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(db_cfg["connect_timeout"]), autocommit=False,
    )
    try:
        counts = upsert_full_dataset_table(
            conn,
            date_str,
            unit_id,
            bundle.get("forecast") or [],
            bundle.get("actual_final") or bundle.get("actual") or [],
            bundle.get("day_ahead_unit") or [],
            bundle.get("realtime_unit") or [],
            (bundle.get("optional") or {}).get("reserve_da") or [],
            bundle.get("captured_at"),
            da_first_rows=bundle.get("day_ahead_first_unit") or [],
            actual_temporary_rows=bundle.get("actual_temporary") or [],
            boundary_rows=bundle.get("forecast_boundary") or [],
        )
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _upload_forecast_prefetch_to_mysql(
    target_date: str,
    forecast_rows: list[dict],
    db_cfg: dict[str, Any],
    unit_id: str,
    reporter=None,
    boundary_rows: list[dict] | None = None,
) -> bool:
    """更新 D+1 ForecastData，并同时保存已发布的 ForecastBoundaryData。"""
    if not _complete_96(forecast_rows, tuple(FORECAST_ZH.keys())):
        return False
    import pymysql
    from scripts.crawler.sync_db.run_crawler import init_database_tables, upsert_full_dataset_table

    try:
        if not init_database_tables(db_cfg):
            raise RuntimeError("epf_pmos_96_full 初始化失败")
        conn = pymysql.connect(
            host=str(db_cfg["host"]), port=int(db_cfg["port"]), user=str(db_cfg["user"]),
            password=str(db_cfg["password"]), database=str(db_cfg["database"]),
            charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=int(db_cfg["connect_timeout"]), autocommit=False,
        )
        try:
            counts = upsert_full_dataset_table(
                conn,
                target_date,
                unit_id,
                forecast_rows,
                [], [], [], [],
                datetime.now().astimezone().isoformat(),
                boundary_rows=boundary_rows or [],
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        logger.info("[DB] 次日预测 %s 提前同步成功 forecast=%s", target_date, counts.get("forecast_periods", 0))
        if reporter is not None:
            reporter.event(
                "INFO", "NEXT_FORECAST_DB_UPLOAD_OK", "次日预测已提前同步到 epf_pmos_96_full",
                date=target_date, target_table="epf_pmos_96_full", **counts,
            )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.exception("[DB] 次日预测 %s 提前同步失败，本地预取数据已保留", target_date)
        if reporter is not None:
            reporter.event(
                "ERROR", "NEXT_FORECAST_DB_UPLOAD_FAIL", str(exc),
                date=target_date, target_table="epf_pmos_96_full",
            )
        return False


def _run_next_forecast_step(
    cfg: dict,
    spider: PmosCrawler | None,
    db_upload_enabled: bool,
    db_cfg: dict[str, Any] | None,
    reporter=None,
) -> tuple[PmosCrawler | None, dict[str, Any], int, dict[str, Any] | None]:
    """每次正式采集运行后执行一次 D+1 预测预取。"""
    try:
        info, spider, rows = prefetch_next_day_forecast(cfg, spider=spider, reporter=reporter)
    except Exception as exc:  # noqa: BLE001
        logger.exception("次日预测预取步骤失败")
        if reporter is not None:
            reporter.event("ERROR", "NEXT_FORECAST_FAIL", str(exc))
        return spider, {"complete": False, "error": str(exc)}, 0, db_cfg

    db_failures = 0
    if info.get("complete") and db_upload_enabled:
        try:
            db_cfg = db_cfg or _load_remote_db_config(cfg)
            boundary_rows: list[dict] = []
            try:
                next_raw = json.loads(Path(str(info.get("raw_path") or "")).read_text(encoding="utf-8"))
                boundary_rows = next_raw.get("forecast_boundary") or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("次日边界预测raw读取失败，仅同步ForecastData: %s", exc)
            if not _upload_forecast_prefetch_to_mysql(
                str(info["date"]),
                rows,
                db_cfg,
                str(cfg.get("unit_id") or ""),
                reporter=reporter,
                boundary_rows=boundary_rows,
            ):
                db_failures = 1
        except Exception as exc:  # noqa: BLE001
            db_failures = 1
            logger.exception("[DB] 次日预测配置/同步失败")
            if reporter is not None:
                reporter.event("ERROR", "NEXT_FORECAST_DB_UPLOAD_FAIL", str(exc), date=info.get("date"))
    return spider, info, db_failures, db_cfg


def _queue_upload(date_str: str, error: str) -> None:
    path = UPLOAD_QUEUE_DIR / f"{date_str}.json"
    path.write_text(json.dumps({"date": date_str, "raw_path": str(RAW_DIR / f"{date_str}.json"), "error": error}, ensure_ascii=False, indent=2), encoding="utf-8")


def _upload_one_date(date_str: str, db_cfg: dict[str, Any], unit_id: str, reporter=None) -> bool:
    raw_path = RAW_DIR / f"{date_str}.json"
    try:
        counts = _upload_bundle_to_mysql(date_str, raw_path, db_cfg, unit_id)
        (UPLOAD_QUEUE_DIR / f"{date_str}.json").unlink(missing_ok=True)
        logger.info(
            "[DB] %s epf_pmos_96_full 上传成功 rows=%s merged_complete=%s",
            date_str, counts.get("rows", 0), counts.get("merged_complete", 0),
        )
        if reporter is not None:
            reporter.event("INFO", "DB_UPLOAD_OK", "已同步到 epf_pmos_96_full", date=date_str,
                           target_table="epf_pmos_96_full", **counts)
        return True
    except Exception as exc:  # noqa: BLE001
        _queue_upload(date_str, f"{type(exc).__name__}: {exc}")
        logger.exception("[DB] %s 上传失败，已加入重试队列", date_str)
        if reporter is not None:
            reporter.event("ERROR", "DB_UPLOAD_FAIL", str(exc), date=date_str,
                           target_table="epf_pmos_96_full")
        return False


def _flush_upload_queue(db_cfg: dict[str, Any], unit_id: str, reporter=None) -> int:
    ok = 0
    for marker in sorted(UPLOAD_QUEUE_DIR.glob("*.json")):
        date_str = marker.stem
        if _upload_one_date(date_str, db_cfg, unit_id, reporter=reporter):
            ok += 1
    return ok


def _verify_remote_date(date_str: str, db_cfg: dict[str, Any], unit_id: str) -> int:
    """只读核验某个交易日是否已写入远程库，供企业机无 Python 环境使用。"""
    import pymysql

    conn = pymysql.connect(
        host=str(db_cfg["host"]), port=int(db_cfg["port"]), user=str(db_cfg["user"]),
        password=str(db_cfg["password"]), database=str(db_cfg["database"]),
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=int(db_cfg["connect_timeout"]), read_timeout=20,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS row_count, COUNT(DISTINCT `时段`) AS periods, "
                "SUM(`日前出清价格` IS NOT NULL) AS da_price_rows, "
                "SUM(`实时出清价格` IS NOT NULL) AS rt_price_rows, "
                "SUM(`直调负荷预测` IS NOT NULL) AS forecast_nonnull, "
                "SUM(`直调负荷实际` IS NOT NULL) AS actual_nonnull, "
                "MAX(update_time) AS latest_update "
                "FROM epf_pmos_96_full WHERE market_date=%s AND unit_id=%s",
                (date_str, unit_id),
            )
            full = cur.fetchone() or {}
    finally:
        conn.close()

    print(f"远程数据库核验日期: {date_str}")
    print(
        "epf_pmos_96_full: rows={row_count} periods={periods} "
        "da_price={da_price_rows} rt_price={rt_price_rows} "
        "forecast_nonnull={forecast_nonnull} actual_nonnull={actual_nonnull} "
        "latest_update={latest_update} "
        "unit_id={unit_id}".format(unit_id=unit_id, **full)
    )
    rows = int(full.get("row_count") or 0)
    periods = int(full.get("periods") or 0)
    da = int(full.get("da_price_rows") or 0)
    rt = int(full.get("rt_price_rows") or 0)
    if rows == 96 and periods == 96 and da == 96 and rt == 96:
        print("核验结果: PASS（epf_pmos_96_full 完整96点且日前/实时价格齐全）")
        return 0
    if rows > 0:
        print("核验结果: PARTIAL（已有数据，但字段或时段尚未完整）")
        return 0
    print("核验结果: FAIL（epf_pmos_96_full 没有该日期数据）")
    return 1


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
def _new_spider(cfg: dict, reporter=None) -> PmosCrawler:
    api_mode = str(cfg.get("data_api_mode") or "qctc").strip().lower()
    unit_id = str(cfg.get("unit_id") or cfg.get("unitid") or "").strip()
    if api_mode == "qctc" and not unit_id:
        raise RuntimeError("配置缺少 unit_id：日前/实时 getDetail96 必须按机组ID请求，不能在空ID下静默采集")
    spider = PmosCrawler(
        base_url=cfg.get("base_url", "https://pmos.sd.sgcc.com.cn:18080/trade"),
        cookie=cfg.get("cookie", ""),
        unit_id=unit_id,
        browser_debug_port=int(cfg.get("debug_port") or cfg.get("browser_debug_port") or 0) or None,
        reporter=reporter,
        # 新门户 HAR6 的 QCTC 信息披露接口；旧 trade DataTables 接口在企业网返回 502。
        data_api_mode=api_mode,
        qctc_auth_wait_sec=float(cfg.get("qctc_auth_wait_sec") or 90.0),
    )
    if not spider.fetch_csrf_token():
        if api_mode == "qctc":
            raise RuntimeError(
                "QCTC 浏览器上下文不可用：未能在 :18080 上取得可 fetch 的同源页面。"
                "请确认调试浏览器（端口 9222、profile=browser_profile_dir）已登录门户，"
                "且能打开 https://pmos.sd.sgcc.com.cn:18080/zcq/main/index.do 或 "
                "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/informationDisclosure/forecast10424 "
                "而不被弹回 /dashboard。注意：没有 sessionStorage Bearer token 本身不算失败，"
                "真正的判定在数据接口的返回码。"
            )
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
    reporter=None,
) -> dict:
    """抓取一天并先落原始包；完整性不足时保留部分数据并记录 PARTIAL。

    ``spider`` 可由主循环复用，减少每天重新认证的时间；网络异常时主循环会
    丢弃该实例并重新建立会话。疑似 actual/fcast 拷贝的可疑市场值只保留在 raw，
    不写入正式字段。
    """
    spider = spider or _new_spider(cfg, reporter=reporter)
    if not spider.change_date(date_str):
        raise RuntimeError(f"change_date({date_str}) 失败")

    time.sleep(1.5)

    result = {
        "date": date_str,
        "forecast": 0,
        "actual": 0,
        "actual_final": 0,
        "actual_temporary": 0,
        "forecast_boundary": 0,
        "day_ahead_first_unit": 0,
        "day_ahead_unit": 0,
        "realtime_unit": 0,
        "complete": False,
        "has_data": False,
        "status": "FAILED",
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
        if reporter is not None:
            reporter.event("ERROR", "COLLECT_SOURCE_FAIL", str(e), date=date_str, source="forecast")

    # 正式实际、临时实际、边界预测是三套独立语义，必须每次并行采集，绝不 fallback 混写。
    a_rows: list[dict] = []
    try:
        a_rows = spider.crawl_market_overview_actual_final()
        result["actual"] = len(a_rows)  # 兼容旧报告：actual 始终表示正式 RealityData
        result["actual_final"] = len(a_rows)
        logger.info("  %s 正式实际 RealityData → %d 行", date_str, len(a_rows))
    except Exception as e:
        logger.warning("%s 正式实际 RealityData 爬取失败: %s", date_str, e)
        if reporter is not None:
            reporter.event("ERROR", "COLLECT_SOURCE_FAIL", str(e), date=date_str, source="actual_final")

    a_tmp_rows: list[dict] = []
    try:
        a_tmp_rows = spider.crawl_market_overview_actual_temporary()
        result["actual_temporary"] = len(a_tmp_rows)
        logger.info("  %s 临时实际 RealityTmpData → %d 行", date_str, len(a_tmp_rows))
    except Exception as e:
        logger.warning("%s 临时实际 RealityTmpData 爬取失败: %s", date_str, e)
        if reporter is not None:
            reporter.event("WARN", "COLLECT_SOURCE_FAIL", str(e), date=date_str, source="actual_temporary")

    boundary_rows: list[dict] = []
    try:
        boundary_rows = spider.crawl_market_boundary()
        result["forecast_boundary"] = len(boundary_rows)
        logger.info("  %s 市场披露边界 ForecastBoundaryData → %d 行", date_str, len(boundary_rows))
    except Exception as e:
        logger.warning("%s 市场披露边界 ForecastBoundaryData 爬取失败: %s", date_str, e)
        if reporter is not None:
            reporter.event("WARN", "COLLECT_SOURCE_FAIL", str(e), date=date_str, source="forecast_boundary")

    da_first_rows: list[dict] = []
    try:
        da_first_rows = spider.crawl_day_ahead_first()
        result["day_ahead_first_unit"] = len(da_first_rows)
        logger.info("  %s 日前一次出清 → %d 行", date_str, len(da_first_rows))
    except Exception as e:
        # 一次出清是新增补充列，不影响现有完整性判定与二次出清主链路。
        logger.warning("%s 日前一次出清爬取失败（新增列保留空值）: %s", date_str, e)
        if reporter is not None:
            reporter.event("WARN", "COLLECT_SOURCE_FAIL", str(e), date=date_str, source="day_ahead_first_unit")

    da_rows: list[dict] = []
    try:
        da_rows = spider.crawl_day_ahead()
        result["day_ahead_unit"] = len(da_rows)
        logger.info("  %s 日前机组明细 → %d 行", date_str, len(da_rows))
    except Exception as e:
        logger.warning("%s 日前机组明细爬取失败（核心表保留空值，不跨源补值）: %s", date_str, e)
        if reporter is not None:
            reporter.event("ERROR", "COLLECT_SOURCE_FAIL", str(e), date=date_str, source="day_ahead_unit")

    rt_rows: list[dict] = []
    try:
        rt_rows = spider.crawl_realtime()
        result["realtime_unit"] = len(rt_rows)
        logger.info("  %s 实时机组明细 → %d 行", date_str, len(rt_rows))
    except Exception as e:
        logger.warning("%s 实时机组明细爬取失败（核心表保留空值，不跨源补值）: %s", date_str, e)
        if reporter is not None:
            reporter.event("ERROR", "COLLECT_SOURCE_FAIL", str(e), date=date_str, source="realtime_unit")

    try:
        optional = spider.crawl_optional_market_data()
    except Exception as e:
        # 备用/图表接口不影响核心四套数据；失败要报告，但不能丢弃已拿到的核心数据。
        optional = {}
        logger.warning("%s 可选接口爬取失败（不影响核心数据）: %s", date_str, e)
        if reporter is not None:
            reporter.event("WARN", "COLLECT_OPTIONAL_FAIL", str(e), date=date_str, source="optional")
    separated, audit = validate_forecast_actual_separation(f_rows, a_rows)
    if not separated:
        # 疑似预测/实际拷贝时仍保留 raw，但不把可疑市场特征写入 CSV/数据库。
        logger.error("%s 疑似 actual/fcast 拷贝；市场特征列不落正式数据，只保留 raw: %s",
                     date_str, audit)
        f_rows = []
        a_rows = []
    core_complete = (
        _complete_96(f_rows, MONITORED_CORE_FIELDS)
        and _complete_96(a_rows, MONITORED_CORE_FIELDS)
    )
    result["audit"] = {
        "core_complete": core_complete,
        "forecast_complete": _complete_96(f_rows, MONITORED_CORE_FIELDS),
        "actual_complete": _complete_96(a_rows, MONITORED_CORE_FIELDS),
        "actual_temporary_complete": _complete_96(a_tmp_rows, MONITORED_CORE_FIELDS),
        "forecast_boundary_complete": _complete_96(boundary_rows, BOUNDARY_CORE_FIELDS),
        "day_ahead_price_complete": _complete_96(da_rows, PRICE_FIELDS),
        "realtime_price_complete": _complete_96(rt_rows, PRICE_FIELDS),
        "separated": separated,
        **audit,
    }
    price_complete = (
        result["audit"]["day_ahead_price_complete"]
        and result["audit"]["realtime_price_complete"]
    )
    optional_has_data = any(bool(value) for value in optional.values()) if isinstance(optional, dict) else bool(optional)
    result["has_data"] = any((
        _rows_have_payload(f_rows), _rows_have_payload(a_rows),
        _rows_have_payload(a_tmp_rows), _rows_have_payload(boundary_rows),
        _rows_have_payload(da_first_rows), _rows_have_payload(da_rows),
        _rows_have_payload(rt_rows), optional_has_data,
    ))
    result["complete"] = bool(core_complete and separated and price_complete)
    result["status"] = "COMPLETE" if result["complete"] else ("PARTIAL" if result["has_data"] else "FAILED")
    result["audit"]["forecast_coverage"] = _coverage(f_rows, MONITORED_CORE_FIELDS)
    result["audit"]["actual_coverage"] = _coverage(a_rows, MONITORED_CORE_FIELDS)
    result["audit"]["actual_temporary_coverage"] = _coverage(a_tmp_rows, MONITORED_CORE_FIELDS)
    result["audit"]["forecast_boundary_coverage"] = _coverage(boundary_rows, BOUNDARY_CORE_FIELDS)
    result["audit"]["day_ahead_coverage"] = _coverage(da_rows, PRICE_FIELDS)
    result["audit"]["realtime_coverage"] = _coverage(rt_rows, PRICE_FIELDS)

    bundle = {
        "schema_version": "pmos-96-raw-v2",
        "market_date": date_str,
        "captured_at": datetime.now().astimezone().isoformat(),
        "base_url": spider.base_url,
        "status": result["status"].lower(),
        "source": {
            "forecast": "QCTC informationDisclosure/ForecastData/getLoadData",
            "actual": "QCTC informationDisclosure/RealityData/getLoadData (正式实际；兼容键)",
            "actual_final": "QCTC informationDisclosure/RealityData/getLoadData (正式实际)",
            "actual_temporary": "QCTC informationDisclosure/RealityTmpData/getLoadData (临时实际)",
            "forecast_boundary": "QCTC informationDisclosure/ForecastBoundaryData/getLoadData (市场披露边界)",
            "day_ahead_first_unit": "QCTC trade/DaJyjgfbPlantFirQuery/getDetail96 (一次出清)",
            "day_ahead_unit": "QCTC trade/DaJyjgfbPlantQuery/getDetail96 (二次出清/最终版)",
            "realtime_unit": "QCTC YxJyjgfbPlantQuery/getDetail96 (fallback rtTmpFdcQuery10456)",
            "optional": "HAR5-discovered PMOS DataTables endpoints",
        },
        "audit": result["audit"],
        "forecast": f_rows,
        "actual": a_rows,
        "actual_final": a_rows,
        "actual_temporary": a_tmp_rows,
        "forecast_boundary": boundary_rows,
        "day_ahead_first_unit": da_first_rows,
        "day_ahead_unit": da_rows,
        "realtime_unit": rt_rows,
        "optional": optional,
    }
    save_raw_bundle(date_str, bundle)
    result["raw_path"] = str(RAW_DIR / f"{date_str}.json")
    if reporter is not None:
        reporter.event("INFO", "LOCAL_RAW_SAVED", "原始响应已保存", date=date_str,
                       path=result["raw_path"], status=result["status"], audit=result["audit"])

    if result["has_data"]:
        append_day_to_table(
            date_str,
            f_rows,
            a_rows,
            da_first_rows,
            da_rows,
            rt_rows,
            optional.get("reserve_da", []),
            actual_temporary_rows=a_tmp_rows,
            boundary_rows=boundary_rows,
        )
        if split:
            split_save(date_str, f_rows, a_rows)
        logger.info("%s 本地保存状态=%s；完整性不足仅记录为PARTIAL，不丢弃已获取数据",
                    date_str, result["status"])
        if reporter is not None:
            reporter.event("INFO", "LOCAL_TABLE_UPDATED", "本地总表已更新", date=date_str,
                           path=str(TABLE_FILE), status=result["status"])
    else:
        logger.error("%s 没有任何有效数据，仅保存 raw，状态=FAILED", date_str)

    return result


# ── 主流程 ───────────────────────────────────────────────────────────
def main() -> int:
    global _ACTIVE_REPORTER
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
    parser.add_argument("--no-db-upload", action="store_true", help="本次运行跳过远程MySQL上传")
    parser.add_argument("--db-check", action="store_true", help="只测试远程MySQL地址/端口/账号，不登录PMOS")
    parser.add_argument("--db-verify", metavar="YYYY-MM-DD", help="只读核验指定日期是否已同步到远程数据库，不登录PMOS")
    args = parser.parse_args()

    print("=" * 55)
    print("  96点市场数据本地爬虫（预测+实际合并总表）")
    print(f"  总表: {TABLE_FILE}")
    print(f"  日志: {OUT_DIR / 'crawler.log'}")
    print("=" * 55)

    cfg = load_config()
    reporter = RunReport(
        REPORT_FILE,
        build_version=BUILD_VERSION,
        args={
            k: v
            for k, v in vars(args).items()
            if k not in {"auth_mode", "auth_timeout_sec", "auth_retries"} or v is not None
        },
    )
    _ACTIVE_REPORTER = reporter
    reporter.stage(
        "startup",
        "PASS",
        config_path=str(CONFIG_PATH),
        output_dir=str(OUT_DIR),
        frozen=_FROZEN,
        pid=os.getpid(),
    )
    logger.info(
        "RUN start version=%s frozen=%s pid=%s auth_only=%s skip_auth=%s args=%s",
        BUILD_VERSION,
        _FROZEN,
        os.getpid(),
        args.auth_only,
        args.skip_auth,
        {
            k: v
            for k, v in vars(args).items()
            if k not in {"auth_mode", "auth_timeout_sec", "auth_retries"} or v is not None
        },
    )

    if args.db_check:
        try:
            db_cfg = _load_remote_db_config(cfg)
            import pymysql

            conn = pymysql.connect(
                host=str(db_cfg["host"]),
                port=int(db_cfg["port"]),
                user=str(db_cfg["user"]),
                password=str(db_cfg["password"]),
                database=str(db_cfg["database"]),
                connect_timeout=int(db_cfg["connect_timeout"]),
            )
            conn.close()
            print(f"远程MySQL连接成功: {db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}")
            reporter.finish("PASS", db_check=True, target_table="epf_pmos_96_full")
            return 0
        except Exception as exc:  # noqa: BLE001
            logger.exception("远程MySQL连接失败")
            reporter.exception("database_sync", exc, target_table="epf_pmos_96_full")
            reporter.finish("FAIL", db_check=False)
            print(f"远程MySQL连接失败: {type(exc).__name__}: {exc}")
            return 3

    if args.db_verify:
        try:
            db_cfg = _load_remote_db_config(cfg)
            code = _verify_remote_date(
                args.db_verify,
                db_cfg,
                str(cfg.get("unit_id") or ""),
            )
            reporter.finish(
                "PASS" if code == 0 else "FAIL",
                db_verify=(code == 0),
                target_table="epf_pmos_96_full",
                date=args.db_verify,
            )
            return code
        except Exception as exc:  # noqa: BLE001
            logger.exception("远程数据库核验失败")
            reporter.exception("database_sync", exc, target_table="epf_pmos_96_full")
            reporter.finish("FAIL", db_verify=False)
            print(f"远程数据库核验失败: {type(exc).__name__}: {exc}")
            return 3

    if not args.skip_auth:
        try:
            auth_mode = str(args.auth_mode or cfg.get("auth_mode") or "browser").strip().lower()
            if auth_mode in {"browser", "auto"}:
                _ensure_browser_state_machine(cfg, reporter=reporter)
            else:
                from scripts.crawler.auth.auth_runtime import ensure_authenticated_config

                ensure_authenticated_config(
                    cfg,
                    CONFIG_PATH,
                    base_dir=BASE_DIR,
                    mode=auth_mode,
                    timeout_sec=args.auth_timeout_sec,
                    max_retries=args.auth_retries,
                )
            reporter.stage(
                "auth_cookie",
                "PASS",
                mode=auth_mode,
                **cookie_summary(str(cfg.get("cookie") or "")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("RUN auth FAIL: %s", exc)
            reporter.exception("auth_cookie", exc)
            reporter.finish("FAIL", auth=False)
            print(f"\n认证失败：{exc}")
            print(f"请查看追加日志：{OUT_DIR / 'crawler.log'}")
            return 2
    else:
        logger.warning("RUN auth SKIP：仅用于兼容/离线测试")

    db_cfg: dict[str, Any] | None = None
    db_ready = False
    db_upload_enabled = bool(cfg.get("db_upload", False)) and not args.no_db_upload
    if db_upload_enabled:
        try:
            db_cfg = _load_remote_db_config(cfg)
            logger.info("[DB] 已读取配置；数据库初始化延迟到本地数据保存之后")
            reporter.stage(
                "database_sync",
                "START",
                target_table="epf_pmos_96_full",
                local_first=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[DB] 配置读取失败，本次仍继续本地采集: %s", exc)
            reporter.exception(
                "database_sync",
                exc,
                target_table="epf_pmos_96_full",
                local_first=True,
            )

    if args.auth_only:
        logger.info("RUN auth-only PASS")
        reporter.finish("PASS", auth_only=True)
        print("\n✅ 认证完成，Cookie 已写回 config.json")
        return 0

    if args.ssl_check:
        code = _ssl_check(cfg)
        reporter.finish("PASS" if code == 0 else "FAIL", ssl_check=(code == 0))
        return code

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

    have = table_existing_dates()
    todo = list(dates) if args.force else [d for d in dates if d not in have]
    skipped = len(dates) - len(todo)

    print(f"\n待爬日期: {len(dates)} 天（其中 {skipped} 天已在总表，跳过）")
    if args.dry_run:
        print(
            "DRY RUN 待爬日期:",
            ", ".join(todo[:10]) + (f" ... 共 {len(todo)} 天" if len(todo) > 10 else ""),
        )
        reporter.finish("PASS", dry_run=True, dates=len(todo), skipped=skipped)
        return 0

    if not todo:
        print("✅ 日期范围内数据已全部爬取；继续预取明日96点预测")
        spider, next_info, next_db_failures, db_cfg = _run_next_forecast_step(
            cfg,
            None,
            db_upload_enabled,
            db_cfg,
            reporter=reporter,
        )
        next_ok = bool(next_info.get("complete")) and next_db_failures == 0
        status = "PASS" if next_ok else "PARTIAL"
        reporter.finish(
            status,
            dates=0,
            skipped=skipped,
            next_forecast=next_info,
            next_forecast_db_failures=next_db_failures,
        )
        print(
            f"明日预测: {next_info.get('date', '-')} rows={next_info.get('rows', 0)} "
            f"complete={bool(next_info.get('complete'))}"
        )
        return 0 if next_ok else 1

    results: list[dict[str, Any]] = []
    upload_failures = 0
    spider: PmosCrawler | None = None
    reporter.stage(
        "collect",
        "START",
        dates=len(todo),
        local_table=str(TABLE_FILE),
        raw_dir=str(RAW_DIR),
    )

    for i, d in enumerate(todo):
        print(f"\n── [{i + 1}/{len(todo)}] {d} ──")
        for attempt in range(2):
            try:
                r = crawl_one_day(
                    d,
                    cfg,
                    split=args.split,
                    spider=spider,
                    reporter=reporter,
                )
                spider = r.pop("_spider", spider)
                results.append(r)
                reporter.date(
                    d,
                    r.get("status", "FAILED"),
                    **{
                        k: v
                        for k, v in r.items()
                        if k not in {"_spider", "status", "audit"}
                    },
                    audit=r.get("audit", {}),
                )

                if db_upload_enabled and r.get("has_data"):
                    if db_cfg is None:
                        try:
                            db_cfg = _load_remote_db_config(cfg)
                        except Exception as db_exc:  # noqa: BLE001
                            logger.error("[DB] %s 配置不可用，本地数据已保存: %s", d, db_exc)
                            _queue_upload(d, f"{type(db_exc).__name__}: {db_exc}")
                            reporter.event(
                                "ERROR",
                                "DB_CONFIG_FAIL",
                                str(db_exc),
                                date=d,
                            )
                            upload_failures += 1
                            db_cfg = None

                    if db_cfg is not None and not db_ready:
                        try:
                            from scripts.crawler.sync_db.run_crawler import init_database_tables

                            if not init_database_tables(db_cfg):
                                raise RuntimeError("epf_pmos_96_full 初始化失败")
                            db_ready = True
                            reporter.stage(
                                "database_sync",
                                "PASS",
                                target_table="epf_pmos_96_full",
                            )
                            queued = _flush_upload_queue(
                                db_cfg,
                                str(cfg.get("unit_id") or ""),
                                reporter=reporter,
                            )
                            if queued:
                                logger.info("[DB] 启动后重试队列上传成功 %d 天", queued)
                        except Exception as db_exc:  # noqa: BLE001
                            logger.error("[DB] %s 初始化失败，本地数据已保存: %s", d, db_exc)
                            _queue_upload(d, f"{type(db_exc).__name__}: {db_exc}")
                            reporter.exception(
                                "database_sync",
                                db_exc,
                                date=d,
                                target_table="epf_pmos_96_full",
                                local_first=True,
                            )
                            db_ready = False

                    if db_ready and db_cfg is not None:
                        if not _upload_one_date(
                            d,
                            db_cfg,
                            str(cfg.get("unit_id") or ""),
                            reporter=reporter,
                        ):
                            upload_failures += 1
                    elif not db_ready:
                        upload_failures += 1
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("第 %d 次失败: %s", attempt + 1, e)
                reporter.event(
                    "ERROR",
                    "DATE_ATTEMPT_FAILED",
                    str(e),
                    date=d,
                    attempt=attempt + 1,
                )
                spider = None
                recoverable_browser_error = (
                    "QCTC认证上下文未建立" in str(e)
                    or "QCTC CDP连接已中断" in str(e)
                    or ("127.0.0.1" in str(e) and "/json" in str(e))
                )
                if attempt == 0 and recoverable_browser_error:
                    reporter.event(
                        "WARN",
                        "BROWSER_RECOVERY_START",
                        "当前浏览器无法完成QCTC任务，切换到新浏览器重试",
                        reason=str(e)[:300],
                    )
                    try:
                        _ensure_browser_state_machine(cfg, reporter=reporter, force_new=True)
                        reporter.event(
                            "INFO",
                            "BROWSER_RECOVERY_READY",
                            "新浏览器认证完成，将使用新CDP端口重试",
                            debug_port=cfg.get("debug_port"),
                        )
                    except Exception as recovery_exc:  # noqa: BLE001
                        logger.exception("浏览器恢复失败: %s", recovery_exc)
                        reporter.event(
                            "ERROR",
                            "BROWSER_RECOVERY_FAIL",
                            str(recovery_exc),
                        )
                time.sleep(3)
        else:
            logger.error("⛔ 重试耗尽，跳过 %s", d)
            reporter.date(d, "FAILED", error="重试耗尽")

        if i < len(todo) - 1:
            time.sleep(args.delay)

    spider, next_info, next_db_failures, db_cfg = _run_next_forecast_step(
        cfg,
        spider,
        db_upload_enabled,
        db_cfg,
        reporter=reporter,
    )
    upload_failures += next_db_failures
    next_forecast_ok = bool(next_info.get("complete")) and next_db_failures == 0

    ok = sum(1 for r in results if r.get("complete", False))
    print(f"\n{'=' * 55}")
    print(f"完成：成功 {ok}/{len(todo)} 天")
    print(f"总表: {TABLE_FILE}")
    if TABLE_FILE.exists():
        import itertools

        with open(TABLE_FILE, "r", encoding="utf-8-sig") as f:
            nrows = sum(1 for _ in f) - 1
        print(f"总表当前行数（不含表头）: {nrows}")

    print(
        f"明日预测: {next_info.get('date', '-')} rows={next_info.get('rows', 0)} "
        f"complete={bool(next_info.get('complete'))}"
    )
    if upload_failures:
        print(
            f"远程数据库上传失败：{upload_failures} 次；"
            "日常数据失败会进入 upload_queue，明日预测 raw 已单独保留"
        )

    no_data_failures = (
        sum(1 for r in results if not r.get("has_data"))
        + len(todo)
        - len(results)
    )
    partial_count = sum(1 for r in results if r.get("status") == "PARTIAL")
    run_status = (
        "PASS"
        if (
            no_data_failures == 0
            and upload_failures == 0
            and partial_count == 0
            and next_forecast_ok
        )
        else ("PARTIAL" if results or next_info else "FAIL")
    )

    reporter.stage(
        "collect",
        run_status,
        dates=len(todo),
        complete=ok,
        partial=partial_count,
        failed=len(todo) - len(results),
    )
    db_status = (
        "SKIP"
        if not db_upload_enabled
        else (
            "PASS"
            if db_ready and upload_failures == 0
            else ("PARTIAL" if upload_failures else "SKIP")
        )
    )
    reporter.stage(
        "database_sync",
        db_status,
        target_table="epf_pmos_96_full",
        failures=upload_failures,
        attempted=db_ready,
    )
    reporter.finish(
        run_status,
        dates=len(todo),
        complete=ok,
        upload_failures=upload_failures,
        next_forecast=next_info,
    )
    return 0 if no_data_failures == 0 and upload_failures == 0 and next_forecast_ok else 1


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
        if _ACTIVE_REPORTER is not None:
            _ACTIVE_REPORTER.event("WARN", "RUN_INTERRUPTED", "用户手动中断程序")
            _ACTIVE_REPORTER.finish("INTERRUPTED")
        sys.exit(130)
    except Exception as e:
        logger.exception("程序异常: %s", e)
        if _ACTIVE_REPORTER is not None:
            _ACTIVE_REPORTER.exception("runtime", e)
            _ACTIVE_REPORTER.finish("FAIL")
        sys.exit(1)
