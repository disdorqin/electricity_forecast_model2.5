#!/usr/bin/env python
"""
修复爬虫：从 2022 重爬 96 点市场特征「真实实际值」，覆盖云端 epf_market_data_96.actual_* 列。

背景
----
epf_market_data_96 的 actual_* 列历史段（2022-01-01 ~ 2026-07-18，1658 天）是
「预测值拷贝」（actual == fcast），这是 2026-04 批量回填把预测值冒充实际值造成的。
本程序通过平台「导出实际」接口 (DaJyxxPlDa.do?method=exportsj) 重爬真实 15 分钟
实际值，交叉验证后用 ON DUPLICATE KEY UPDATE 覆盖云端 actual 列。

用法
----
  # 单日探测（真机跑一次，确认 exportsj 返回格式，不写库）
  python scripts/backfill_actual_96.py --probe 2024-06-15

  # 从 2022 到昨天，全量重爬并覆盖云端（先备份整表）
  python scripts/backfill_actual_96.py --start 2022-01-01 --end 2026-07-31

  # 只爬不写库（验证爬取正确性）
  python scripts/backfill_actual_96.py --start 2024-06-01 --end 2024-06-15 --no-db

  # 强制重爬已处理过的日期
  python scripts/backfill_actual_96.py --start 2024-01-01 --force

支持断点续爬（outputs/backfill_actual_progress.json），可打包成 exe 在公司电脑跑。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# ── 屏蔽 SSL 警告（必须在任何网络导入之前生效） ─────────────────────
import urllib3

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["PYTHONWARNINGS"] = "ignore::urllib3.exceptions.InsecureRequestWarning"
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
# ─────────────────────────────────────────────────────────────────────

import pandas as pd
import pymysql
from dotenv import load_dotenv

# ── PyInstaller / 路径 ──────────────────────────────────────────────
_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    BASE_DIR = Path(sys.executable).parent.resolve()
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
else:
    BASE_DIR = Path(__file__).resolve().parents[1]
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

# PyInstaller 静态分析
from scripts.crawler.crawl import (  # noqa: E402
    PmosCrawler,
    parse_number,
    period_no_from_time,
)
from utils.database_operate import (  # noqa: E402
    get_db_connection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_actual_96")

OUTPUT_DIR = BASE_DIR / "output"
BACKUP_DIR = OUTPUT_DIR / "backfill_actual_backup"
PROGRESS_PATH = OUTPUT_DIR / "backfill_actual_progress.json"

# ── 市场特征字段映射（爬虫列名 → DB actual 列名） ─────────────────────
# 与 run_crawler.py / backfill_unit_data_96.py / auto_fill_96.py 一致。
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

# 需要更新的所有 actual 列（含 96 独有的检修/备用，来自 sync_data_96_core）
ALL_ACTUAL_COLS = [
    "actual_direct_load", "actual_local_plant", "actual_tie_line",
    "actual_wind", "actual_solar", "actual_nuclear", "actual_self_owned",
    "actual_test_unit", "actual_unit_maintenance", "actual_pos_reserve",
    "actual_neg_reserve", "actual_bidding_space", "actual_new_energy",
]

# 24 点表（本地，用于交叉验证）路径
H24_XLSX = BASE_DIR / "data" / "shandong_pmos_hourly.xlsx"


# ── 配置加载（与现有爬虫一致） ──────────────────────────────────────
def load_config() -> dict:
    config_path = BASE_DIR / "scripts" / "crawler" / "config.json"
    if _FROZEN:
        config_path = BASE_DIR / "config.json"
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: dict = json.load(f)
    if not cfg.get("cookie", "").strip():
        logger.error("config.json 中 cookie 为空，请先自动登录或填入 Cookie")
        sys.exit(1)
    uid = cfg.get("unit_id") or cfg.get("unitid") or ""
    uid = str(uid).strip()
    if not uid:
        logger.error("config.json 中 unit_id 为空")
        sys.exit(1)
    cfg["unit_id"] = uid
    # export_type: 平台「导出实际」的 type 参数。前端 dcbd 值，tab 0「负荷信息」= 1，
    # 对应 systemload/dfdcload 等字段。默认 1；config 里显式填其他值则覆盖。
    raw_export_type = str(cfg.get("export_type") or "").strip()
    cfg["export_type"] = raw_export_type or "1"
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


# ── 24 点表交叉验证 ─────────────────────────────────────────────────
_H24_CACHE: Optional[Any] = None


def load_h24_actual() -> Optional[pd.DataFrame]:
    """加载 24 点表 actual 列（惰性，用于交叉验证）。

    返回 DataFrame: 时刻(datetime), <中文列名>=实际值 ...
    24 点表 actual 从 2022 起是真实值（已验证），可作基准。
    """
    global _H24_CACHE
    if _H24_CACHE is not None:
        return _H24_CACHE
    if not H24_XLSX.exists():
        logger.warning("24 点表不存在: %s，跳过交叉验证", H24_XLSX)
        return None

    df = pd.read_excel(H24_XLSX)
    df["时刻"] = pd.to_datetime(df["时刻"], errors="coerce")
    df = df.dropna(subset=["时刻"])
    _H24_CACHE = df
    return df


def cross_validate_day(actual_rows: list[dict[str, Any]], date_str: str) -> dict:
    """用 24 点表实际值交叉验证爬回来的 96 点实际值。

    把 96 点实际值按 hour=ceil(period/4) 聚合为小时均值，与 24 点表当日
    实际值对比。返回 dict: {status: ok|mismatch|no_baseline, details: [...]}.

    如果 24 点表无该日数据，或差异超阈值，返回 mismatch/no_baseline，
    调用方据此决定是否告警（不自动阻止写库，除非 --strict）。
    """
    h24 = load_h24_actual()
    if h24 is None:
        return {"status": "no_baseline", "details": ["24点表不可用"]}

    # 24 点表该日实际值（小时级别）
    day_mask = (h24["时刻"].dt.date == datetime.strptime(date_str, "%Y-%m-%d").date())
    day24 = h24[day_mask]
    # 24 点表小时：01:00..24:00 → hour_business 1..24；00:00 行属前一日 h24
    if day24.empty:
        return {"status": "no_baseline", "details": [f"24点表无 {date_str} 数据"]}

    # 24 点表 build 小时字典：hour_business → 实际值（取第一列非空的特征做基准）
    baseline: dict[int, float] = {}
    for _, r in day24.iterrows():
        ts = r["时刻"]
        # 24 点表「时刻」即区间末：h24 = 次日 00:00
        hb = ts.hour if ts.hour != 0 else 24
        for col in ["直调负荷实际值", "直调负荷实际", "系统负荷实际值"]:
            if col in day24.columns and pd.notna(r.get(col)):
                v = parse_number(r[col])
                if v is not None:
                    baseline[hb] = v
                    break

    if not baseline:
        return {"status": "no_baseline", "details": [f"24点表 {date_str} 实际列为空"]}

    # 96 点实际值按小时聚合（直调负荷）
    agg: dict[int, list[float]] = {}
    for row in actual_rows:
        pno = period_no_from_time(row.get("Periodid", ""))
        if pno < 1 or pno > 96:
            continue
        hb = (pno + 3) // 4  # ceil(pno/4) → 1..24
        v = parse_number(row.get("systemload"))
        if v is not None:
            agg.setdefault(hb, []).append(v)

    mismatches: list[str] = []
    matched_hours = 0
    total_mad = 0.0
    for hb, hval in sorted(baseline.items()):
        qvals = agg.get(hb)
        if not qvals:
            continue
        qmean = sum(qvals) / len(qvals)
        matched_hours += 1
        mad = abs(qmean - hval)
        total_mad += mad
        if mad > 500:  # 阈值：聚合后差异超 500 MW 视为可疑
            mismatches.append(f"hour={hb}: 96点均值={qmean:.1f} vs 24点实际={hval:.1f} (Δ={mad:.1f})")

    if matched_hours == 0:
        return {"status": "no_baseline", "details": ["96点实际值无有效小时"]}

    avg_mad = total_mad / matched_hours
    status = "ok" if not mismatches else "mismatch"
    return {
        "status": status,
        "matched_hours": matched_hours,
        "avg_mad": avg_mad,
        "mismatches": mismatches[:10],
        "details": [f"matched_hours={matched_hours} avg_mad={avg_mad:.1f}"] + mismatches[:10],
    }


# ── 数据库：备份 / 覆盖 ─────────────────────────────────────────────
def backup_table(db_cfg: dict) -> str:
    """备份 epf_market_data_96 整表到本地 CSV，返回备份目录。

    只写 CSV（避免引入 pyarrow 依赖，适合 exe 轻量打包）。160,800 行
    ≈ 10MB，足够回滚用。
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BACKUP_DIR / f"epf_market_data_96_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM epf_market_data_96 ORDER BY market_date, period_no")
        rows = cur.fetchall()
        df = pd.DataFrame(rows)
        csv_path = out_dir / "epf_market_data_96.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("备份完成: %d 行 -> %s", len(df), out_dir)
        return str(out_dir)
    finally:
        conn.close()


def upsert_actual_cols(
    conn, market_date: str, actual_rows: list[dict[str, Any]]
) -> int:
    """用爬回来的实际值 upsert 某一天的 actual_* 列。

    只更新 MARKET_FIELD_MAP 对应的 8 个 actual 列（爬虫能拿到的）。
    96 独有的 5 列（检修/正负备用/竞价空间/新能源）若在爬取结果里有则也更新。
    使用 (market_date, period_no) 作唯一键，ON DUPLICATE KEY UPDATE。
    """
    import pymysql

    field_map = dict(MARKET_FIELD_MAP)
    # 逐行只更新该行有值的列——避免把 NULL 覆盖进已有的真实值
    # （例如实时接口不返回核电/自备，若整列都 NULL 就不 UPDATE，保留表中原值）
    dt_base = datetime.strptime(market_date, "%Y-%m-%d")
    count = 0
    with conn.cursor() as cur:
        for row in actual_rows:
            period_label = row.get("Periodid", "")
            if not period_label:
                continue
            pno = period_no_from_time(period_label)
            if pno < 1 or pno > 96:
                continue
            # 该行有值的 actual 列：列名(爬虫字段) → DB列名
            present = {
                crawler_col: field_map[crawler_col]
                for crawler_col in field_map
                if parse_number(row.get(crawler_col)) is not None
            }
            if not present:
                continue
            # present: {systemload: actual_direct_load, ...}（有值的那几列）
            db_cols = ["market_date", "period_no", "data_time"] + list(present.values())
            placeholders = ", ".join(["%s"] * len(db_cols))
            update_parts = ", ".join([f"{c}=VALUES({c})" for c in present.values()])
            sql = (
                f"INSERT INTO epf_market_data_96 ({', '.join(db_cols)}) "
                f"VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {update_parts}"
            )
            data_time = dt_base + timedelta(minutes=pno * 15)
            # 值 = 该行中 present 对应列的 parse_number 结果
            row_vals = [parse_number(row.get(crawler_col)) for crawler_col in present]
            vals = [market_date, pno, data_time] + row_vals
            try:
                cur.execute(sql, vals)
                count += 1
            except pymysql.err.IntegrityError as e:
                logger.warning("跳过 period=%d: %s", pno, e)
    conn.commit()
    return count


# ── 进度管理 ────────────────────────────────────────────────────────
def _load_progress() -> dict:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ── 保存实际值 CSV（供人工核对） ────────────────────────────────────
ACTUAL_COLUMNS_ZH = {
    "Periodid": "时刻",
    "systemload": "直调负荷实际",
    "dfdcload": "地方电厂出力实际",
    "excload": "外电实际",
    "fdload": "风电实际",
    "gfload": "光伏实际",
    "sytsjz": "核电实际",
    "selfunit": "自备电厂实际",
    "syjzzj": "试验机组实际",
}


def _save_actual_csv(date_str: str, actual_rows: list[dict], save_dir: Path) -> Path:
    """把某天爬回的 96 点实际值保存为 CSV（中文列名，供人工核对）。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in actual_rows:
        row = {"market_date": date_str}
        for key, zh in ACTUAL_COLUMNS_ZH.items():
            row[zh] = r.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)
    # 排序：时刻 00:15 → 24:00
    df = df.sort_values("时刻").reset_index(drop=True)
    path = save_dir / f"actual_{date_str}.csv"
    try:
        df.to_csv(path, index=False, encoding="gbk")
    except Exception:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("已保存实际值 %d 行 -> %s", len(df), path)
    return path


# ── 24 点表下采样补核电/自备（实时接口不提供这 2 列） ────────────────
# 24 点表列名 → 96 点爬虫字段名
H24_DOWNSAMPLE_MAP: dict[str, str] = {
    "核电总加实际值": "sytsjz",      # → actual_nuclear
    "自备机组总加实际值": "selfunit",  # → actual_self_owned
}

_H24_DOWNSAMPLE_CACHE: Optional[pd.DataFrame] = None


def _fill_from_h24(actual_rows: list[dict], date_str: str) -> list[dict]:
    """用 24 点表真实实际值下采样，补核电/自备到 actual_rows。

    实时接口 DaJyxxPlYx.do 不返回 sytsjz(核电)/selfunit(自备)，但 24 点表
    （epf_market_data）有这 2 列的真实实际值（已验证：核电 5.9% 拷贝、自备 0% 拷贝）。
    按 hour_business=ceil(period_no/4) 把小时值复制到 4 个 15 分钟段。

    若 24 点表不可用或该日无数据，静默跳过（保持原值）。
    """
    global _H24_DOWNSAMPLE_CACHE
    if _H24_DOWNSAMPLE_CACHE is None:
        if not H24_XLSX.exists():
            logger.warning("24 点表不存在 %s，跳过核电/自备下采样", H24_XLSX)
            return actual_rows
        _H24_DOWNSAMPLE_CACHE = pd.read_excel(H24_XLSX)
        _H24_DOWNSAMPLE_CACHE["时刻"] = pd.to_datetime(
            _H24_DOWNSAMPLE_CACHE["时刻"], errors="coerce"
        )

    h24 = _H24_DOWNSAMPLE_CACHE
    day = h24[h24["时刻"].dt.date == datetime.strptime(date_str, "%Y-%m-%d").date()]
    if day.empty:
        logger.warning("%s: 24 点表无该日数据，跳过核电/自备下采样", date_str)
        return actual_rows

    # 按 hour_business 建小时实际值字典
    day = day.copy()
    day["hb"] = day["时刻"].dt.hour.replace(0, 24)
    filled = 0
    for hcol, crawler_key in H24_DOWNSAMPLE_MAP.items():
        if hcol not in day.columns:
            continue
        hb_values = {}
        for _, r in day.iterrows():
            v = r.get(hcol)
            if pd.notna(v):
                hb_values[int(r["hb"])] = float(v)
        if not hb_values:
            continue
        # 补到 96 点
        for row in actual_rows:
            pno = period_no_from_time(row.get("Periodid", ""))
            if pno < 1 or pno > 96:
                continue
            hb = (pno + 3) // 4
            if hb in hb_values:
                row[crawler_key] = hb_values[hb]
                filled += 1
    if filled:
        logger.info("%s: 24点下采样补核电/自备 %d 值", date_str, filled)
    return actual_rows


# ── 单日爬取 ────────────────────────────────────────────────────────
def crawl_one_day(
    date_str: str,
    config: dict,
    *,
    probe_only: bool = False,
    strict: bool = False,
    no_db: bool = False,
    db_cfg: Optional[dict] = None,
    save_dir: Optional[Path] = None,
) -> dict:
    """爬取单日 96 点实际值，交叉验证，可选写入数据库。

    probe_only: 只爬取 + 验证 + 打印，绝不写库（用于确认 exportsj 格式）。
    save_dir: 若给定，把当日实际值保存为 CSV（不写库），供人工核对。
    返回 dict: {status, market_rows, validation, written}
    """
    spider = PmosCrawler(
        base_url=config["base_url"],
        cookie=config["cookie"],
        unit_id=config["unit_id"],
    )
    if not spider.fetch_csrf_token():
        # CSRF 获取失败 = cookie 过期/未登录。中止，不继续爬（避免误导性的 exportsj 错误）。
        logger.error(
            "CSRF 获取失败 (%s)：config.json 的 cookie 可能已过期，"
            "请重新从浏览器 F12 复制最新 Cookie。跳过本日。",
            date_str,
        )
        return {"status": "auth_failed", "market_rows": [], "validation": {}, "written": 0}
    if not spider.change_date(date_str):
        logger.warning("日期切换失败，跳过 %s", date_str)
        return {"status": "change_date_failed", "market_rows": [], "validation": {}, "written": 0}

    time.sleep(1.5)

    # 爬取实际值（exportsj）
    try:
        actual_rows = spider.crawl_market_overview_actual(
            export_type=config.get("export_type")
        )
    except Exception as e:
        logger.error("实际值爬取失败 (%s): %s", date_str, e)
        return {"status": "crawl_failed", "market_rows": [], "validation": {}, "written": 0, "error": str(e)}

    if len(actual_rows) < 90:
        logger.warning("%s: 实际值仅 %d 行（不足 96），可疑", date_str, len(actual_rows))

    # 用 24 点表下采样补核电/自备（实时接口不返回这 2 列）
    actual_rows = _fill_from_h24(actual_rows, date_str)

    # 交叉验证
    validation = cross_validate_day(actual_rows, date_str)
    logger.info("%s: 验证 status=%s (%s)", date_str, validation.get("status"),
                "; ".join(validation.get("details", [])[:2]))

    if probe_only:
        print(f"\n=== PROBE {date_str} ===")
        print(f"实际值行数: {len(actual_rows)}")
        print(f"交叉验证: {json.dumps(validation, ensure_ascii=False, indent=2)}")
        print("前 5 行:")
        for row in actual_rows[:5]:
            print("  ", row)
        print("后 3 行:")
        for row in actual_rows[-3:]:
            print("  ", row)
        return {"status": "probe", "market_rows": actual_rows, "validation": validation, "written": 0}

    if strict and validation.get("status") == "mismatch":
        logger.warning("%s: 交叉验证失败，strict 模式跳过写库", date_str)
        return {"status": "validation_failed", "market_rows": actual_rows,
                "validation": validation, "written": 0}

    # 保存 CSV 到本地（供人工核对，不写库）
    if save_dir:
        _save_actual_csv(date_str, actual_rows, save_dir)
        return {"status": "saved", "market_rows": actual_rows,
                "validation": validation, "written": 0}

    if no_db or not db_cfg:
        return {"status": "crawled", "market_rows": actual_rows,
                "validation": validation, "written": 0}

    # 写库
    conn = _get_conn(db_cfg)
    try:
        written = upsert_actual_cols(conn, date_str, actual_rows)
    finally:
        conn.close()
    return {"status": "written", "market_rows": actual_rows,
            "validation": validation, "written": written}


# ── 主流程 ──────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="96点市场特征实际值修复爬虫（重爬真实actual并覆盖云端）"
    )
    parser.add_argument("--start", default="2022-01-01", help="开始日期 (默认 2022-01-01)")
    parser.add_argument("--end", default=None, help="结束日期 (默认昨天)")
    parser.add_argument("--probe", metavar="DATE", default=None,
                        help="单日探测模式: 只爬1天+验证+打印，不写库。用于确认exportsj格式")
    parser.add_argument("--probe-da", metavar="DATE", default=None,
                        help="诊断模式: 对比日前接口(DaJyxxPlDa)与实时接口(DaJyxxPlYx)对同一历史日期的systemload，"
                             "确认日前接口历史返回预测还是实际值")
    parser.add_argument("--force", action="store_true", help="强制重爬已处理过的日期")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待爬日期，不爬取不写库")
    parser.add_argument("--no-db", action="store_true", help="只爬取不写库")
    parser.add_argument("--save-days", type=int, default=0, metavar="N",
                        help="爬最近 N 天实际值并保存为 CSV（output/actual_saved/），不写库。"
                             "供人工核对后再全量写库")
    parser.add_argument("--strict", action="store_true", help="交叉验证失败则跳过该日写库")
    parser.add_argument("--backup", action="store_true", default=True,
                        help="写库前备份整表 (默认 True，覆盖云端前必做)")
    parser.add_argument("--delay", type=float, default=1.5, help="每次请求间隔秒数 (默认 1.5)")
    parser.add_argument("--max-retry", type=int, default=2, help="单日最大重试次数 (默认 2)")
    args = parser.parse_args()

    # 探测模式：单日，不写库
    if args.probe:
        config = load_config()
        result = crawl_one_day(
            args.probe, config, probe_only=True,
        )
        print(f"\nPROBE 完成: {args.probe} status={result['status']}")
        return 0 if result["status"] == "probe" else 1

    # 诊断模式：对比日前接口(DaJyxxPlDa) vs 实时接口(DaJyxxPlYx) 对历史日期的 systemload
    if args.probe_da:
        config = load_config()
        date_str = args.probe_da
        spider = PmosCrawler(
            base_url=config["base_url"],
            cookie=config["cookie"],
            unit_id=config["unit_id"],
        )
        if not spider.fetch_csrf_token():
            print("CSRF 失败，cookie 可能过期")
            return 1
        if not spider.change_date(date_str):
            print(f"change_date({date_str}) 失败")
            return 1
        time.sleep(1.5)

        print(f"\n=== PROBE-DA {date_str}: 日前 vs 实时 接口对比 ===")
        try:
            da_rows = spider.crawl_market_overview_da()
            print(f"日前接口(DaJyxxPlDa): {len(da_rows)} 行")
            if da_rows:
                print("  字段:", list(da_rows[0].keys()))
                print(f"  前2行 systemload: {[r.get('systemload') for r in da_rows[:2]]}")
        except Exception as e:
            print(f"日前接口失败: {e}")
        try:
            yx_rows = spider._crawl_market_actual_via_json()
            print(f"实时接口(DaJyxxPlYx): {len(yx_rows)} 行")
            if yx_rows:
                print("  字段:", list(yx_rows[0].keys()))
                print(f"  前2行 systemload: {[r.get('systemload') for r in yx_rows[:2]]}")
        except Exception as e:
            print(f"实时接口失败: {e}")

        # 对比结论
        if da_rows and yx_rows:
            da_sl = [float(r.get('systemload') or 0) for r in da_rows[:5]]
            yx_sl = [float(r.get('systemload') or 0) for r in yx_rows[:5]]
            match = all(abs(a-b) < 0.01 for a,b in zip(da_sl, yx_sl))
            print(f"\n前5个 systemload 对比:")
            print(f"  日前: {da_sl}")
            print(f"  实时: {yx_sl}")
            print(f"  {'✅ 一致 = 日前接口历史返回实际值(含核电等9列)' if match else '❌ 不一致 = 日前接口历史返回预测值'}")
        return 0

    # 保存模式：最近 N 天实际值 → CSV（不写库）
    save_dir: Optional[Path] = None
    if args.save_days > 0:
        end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=args.save_days - 1)
        args.start = start_dt.strftime("%Y-%m-%d")
        args.end = end_date
        save_dir = OUTPUT_DIR / "actual_saved"
        print(f"保存模式: 爬最近 {args.save_days} 天实际值 → {save_dir} (不写库)")

    # 日期范围
    end_date = args.end or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if start_dt > end_dt:
        logger.error("start 必须 <= end")
        return 1

    config = load_config()
    db_cfg = load_db_config()
    db_ok = all([db_cfg.get("host"), db_cfg.get("database"),
                 db_cfg.get("user"), db_cfg.get("password")])
    if not db_ok:
        logger.error("数据库配置不完整")
        return 1

    # 待爬日期
    progress = _load_progress()
    done = set(progress.get("done", []))
    if args.force:
        done = set()

    all_dates: list[str] = []
    cur = start_dt
    while cur <= end_dt:
        all_dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    to_crawl = [d for d in all_dates if d not in done]
    logger.info("日期范围: %s ~ %s (%d 天)，待爬: %d", args.start, end_date,
                len(all_dates), len(to_crawl))

    if not to_crawl:
        logger.info("所有日期已处理，无需爬取")
        return 0

    if args.dry_run:
        print("DRY RUN 待爬日期:")
        for d in to_crawl[:20]:
            print("  ", d)
        if len(to_crawl) > 20:
            print(f"  ... 共 {len(to_crawl)} 天")
        return 0

    # 写库前备份整表（一次）—— save_dir 模式不写库，跳过备份
    backup_dir = None
    if not save_dir and not args.no_db and db_ok and args.backup:
        backup_dir = backup_table(db_cfg)
        print(f"备份: {backup_dir}")

    # 逐日爬取
    results = {"written": 0, "crawled": 0, "saved": 0, "failed": 0, "skipped": 0}
    for idx, date_str in enumerate(to_crawl):
        print(f"\n── [{idx+1}/{len(to_crawl)}] {date_str} ──")
        ok = False
        for attempt in range(1, args.max_retry + 1):
            try:
                res = crawl_one_day(
                    date_str, config, no_db=args.no_db, db_cfg=db_cfg,
                    strict=args.strict, save_dir=save_dir,
                )
            except Exception as e:
                logger.error("  第 %d 次异常: %s", attempt, e)
                res = {"status": "error"}
            if res.get("status") in ("written", "crawled", "saved"):
                ok = True
                break
            if res.get("status") in ("change_date_failed", "validation_failed"):
                break  # 不重试这类
            time.sleep(3)
        if not ok:
            logger.error("  ⛔ 重试耗尽，跳过 %s (status=%s)", date_str, res.get("status"))
            results["failed"] += 1
            continue

        if res.get("status") == "written":
            results["written"] += 1
        elif res.get("status") == "saved":
            results["saved"] += 1
        else:
            results["crawled"] += 1
        progress.setdefault("done", []).append(date_str)
        _save_progress(progress)

        if idx < len(to_crawl) - 1:
            time.sleep(args.delay)

    # 汇总
    print(f"\n{'='*55}")
    print(f"完成！ 写入={results['written']} 保存={results['saved']} "
          f"只爬={results['crawled']} 失败={results['failed']}")
    if backup_dir:
        print(f"备份目录: {backup_dir}")
    if save_dir:
        print(f"保存目录: {save_dir}")
    print(f"进度文件: {PROGRESS_PATH}")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n被用户中断，进度已保存")
        sys.exit(130)
