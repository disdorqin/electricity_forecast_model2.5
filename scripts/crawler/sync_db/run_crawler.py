#!/usr/bin/env python
"""
国网 PMOS 96 点爬虫运行时支持模块

当前生产入口是 `dist/crawler/crawl_96_auto_v7.exe`，本模块只提供
`epf_pmos_96_full` 建表、schema 校验和上传支持。旧源表同步代码已归档，
直接执行本文件会退出并提示使用 v3，不再保留第二个生产入口。
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
#  运行时检测 & 路径
# ---------------------------------------------------------------------------

_FROZEN = getattr(sys, "frozen", False)  # PyInstaller .exe 模式

if _FROZEN:
    # .exe 模式：路径相对于可执行文件所在目录
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    # Python 脚本模式：确保项目根在 sys.path 中
    _BASE_DIR = Path(__file__).resolve().parents[3]
    if str(_BASE_DIR) not in sys.path:
        sys.path.insert(0, str(_BASE_DIR))
    BASE_DIR = _BASE_DIR

FULL_MIGRATION_SQL = (
    BASE_DIR / "scripts" / "crawler" / "sync_db" / "sql"
    / "002_create_epf_pmos_96_full.sql"
)

# 同级目录导入（避免 PyInstaller 找不到 scripts 包）
# 用完整包路径，非冻结/冻结模式均可靠
from scripts.crawler.collect.crawl import parse_number, period_no_from_time  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_crawler")

# ===================================================================
#  数据库操作
# ===================================================================


def get_db(cfg: dict):
    """创建数据库连接"""
    import pymysql

    return pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
    )


# 96 点 canonical 宽表；仅此表属于当前生产同步目标。
CREATE_FULL_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `epf_pmos_96_full` (
    `id` BIGINT NOT NULL AUTO_INCREMENT COMMENT '数据库技术主键',
    `market_date` DATE NOT NULL COMMENT '数据集字段: market_date',
    `时段` VARCHAR(5) NOT NULL COMMENT '数据集字段: 时段，00:15..24:00',
    `直调负荷预测` DECIMAL(14,4) DEFAULT NULL,
    `地方电厂出力预测` DECIMAL(14,4) DEFAULT NULL,
    `外电预测` DECIMAL(14,4) DEFAULT NULL,
    `风电预测` DECIMAL(14,4) DEFAULT NULL,
    `光伏预测` DECIMAL(14,4) DEFAULT NULL,
    `核电预测` DECIMAL(14,4) DEFAULT NULL,
    `自备电厂预测` DECIMAL(14,4) DEFAULT NULL,
    `试验机组预测` DECIMAL(14,4) DEFAULT NULL,
    `全网负荷预测` DECIMAL(14,4) DEFAULT NULL COMMENT 'ForecastData.qwfh',
    `直调负荷实际` DECIMAL(14,4) DEFAULT NULL,
    `地方电厂出力实际` DECIMAL(14,4) DEFAULT NULL,
    `外电实际` DECIMAL(14,4) DEFAULT NULL,
    `风电实际` DECIMAL(14,4) DEFAULT NULL,
    `光伏实际` DECIMAL(14,4) DEFAULT NULL,
    `核电实际` DECIMAL(14,4) DEFAULT NULL,
    `自备电厂实际` DECIMAL(14,4) DEFAULT NULL,
    `试验机组实际` DECIMAL(14,4) DEFAULT NULL,
    `抽蓄实际` DECIMAL(14,4) DEFAULT NULL,
    `全网负荷实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityData.qwfh',
    `直调负荷临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `地方电厂出力临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `外电临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `风电临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `光伏临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `核电临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `自备电厂临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `试验机组临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `抽蓄临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData',
    `全网负荷临时实际` DECIMAL(14,4) DEFAULT NULL COMMENT 'RealityTmpData.qwfh',
    `边界全网负荷预测` DECIMAL(14,4) DEFAULT NULL COMMENT 'ForecastBoundaryData.qwfh',
    `边界直调负荷预测` DECIMAL(14,4) DEFAULT NULL COMMENT 'ForecastBoundaryData.zdfh',
    `边界外电预测` DECIMAL(14,4) DEFAULT NULL COMMENT 'ForecastBoundaryData.llxfh',
    `边界风电预测` DECIMAL(14,4) DEFAULT NULL COMMENT 'ForecastBoundaryData.fd',
    `边界光伏预测` DECIMAL(14,4) DEFAULT NULL COMMENT 'ForecastBoundaryData.gf',
    `边界核电预测` DECIMAL(14,4) DEFAULT NULL COMMENT 'ForecastBoundaryData.hd',
    `日前一次出清价格` DECIMAL(14,4) DEFAULT NULL COMMENT '补充字段：日前首次发布/一次出清96点价格',
    `日前出清价格` DECIMAL(14,4) DEFAULT NULL COMMENT '现有主字段：日前二次出清/最终版价格',
    `日前出力` DECIMAL(14,4) DEFAULT NULL,
    `日前电量` DECIMAL(14,4) DEFAULT NULL,
    `日前开机状态` VARCHAR(20) DEFAULT NULL,
    `日前电源类型` VARCHAR(64) DEFAULT NULL,
    `实时出清价格` DECIMAL(14,4) DEFAULT NULL,
    `实时出力` DECIMAL(14,4) DEFAULT NULL,
    `实时电量` DECIMAL(14,4) DEFAULT NULL,
    `实时开机状态` VARCHAR(20) DEFAULT NULL,
    `实时电源类型` VARCHAR(64) DEFAULT NULL,
    `正备用预测` DECIMAL(14,4) DEFAULT NULL,
    `负备用预测` DECIMAL(14,4) DEFAULT NULL,
    `unit_id` VARCHAR(64) NOT NULL COMMENT '数据库元数据：价格对应的机组ID',
    `source_captured_at` DATETIME DEFAULT NULL COMMENT '数据库元数据：原始包采集时间',
    `create_time` DATETIME DEFAULT CURRENT_TIMESTAMP,
    `update_time` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_full_date_period_unit` (`market_date`, `时段`, `unit_id`),
    KEY `idx_full_market_date` (`market_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci COMMENT='PMOS 96点宽表：业务字段与权威CSV一致';
"""


def init_database_tables(db_cfg: dict) -> bool:
    """只初始化当前生产同步目标 ``epf_pmos_96_full``。"""
    # 当前链路只初始化这个目标表；旧源表不在生产同步范围。
    sql_parts = []
    if FULL_MIGRATION_SQL.exists():
        sql_parts.append(FULL_MIGRATION_SQL.read_text(encoding="utf-8"))
    else:
        sql_parts.append(CREATE_FULL_TABLE_SQL)
    sql = "\n".join(sql_parts)

    statements = [s.strip() for s in sql.split(";") if s.strip()]

    conn = get_db(db_cfg)
    try:
        with conn.cursor() as cur:
            for stmt in statements:
                # migration 文件允许以 -- 注释开头；判断前去掉前置注释，
                # 否则冻结 EXE 会静默跳过新的合并表建表语句。
                normalized = re.sub(r"^(?:\s*--[^\n]*(?:\n|$)|\s*/\*.*?\*/\s*)+", "", stmt, flags=re.S)
                if normalized.upper().startswith("CREATE") or normalized.upper().startswith("ALTER"):
                    logger.info("执行: %s ...", stmt[:60])
                    cur.execute(normalized)
        conn.commit()
        ensure_full_table_schema(conn)
        logger.info("[OK] epf_pmos_96_full 初始化完成（仅此表）")
        return True
    except Exception as e:
        logger.error("建表失败: %s", e)
        conn.rollback()
        return False
    finally:
        conn.close()


def ensure_full_table_schema(conn) -> None:
    """确保合并表是权威 CSV 的同构业务表。

    旧版表使用英文派生字段，且包含数据集没有的 maintenance/bidding_space/
    new_energy/complete 标记。不能在原表上继续混写；首次发现旧 schema 时先
    改名保留为 *_legacy，再创建新表。历史数据由本地权威 CSV 迁移工具显式回灌。
    """
    legacy_name = f"epf_pmos_96_full_legacy_{datetime.now():%Y%m%d%H%M%S}"
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'epf_pmos_96_full'")
        exists = cur.fetchone() is not None
        if exists:
            cur.execute("SHOW COLUMNS FROM `epf_pmos_96_full`")
            columns = set()
            for row in cur.fetchall():
                if isinstance(row, dict):
                    columns.add(str(row.get("Field") or row.get("field") or ""))
                else:
                    columns.add(str(row[0]))
            # 中文“时段”+原 canonical 字段是成熟生产表的稳定标志。2026-09-19
            # 新增的 disclosure 字段只允许增量 ADD COLUMN，绝不能因为扩展列缺失就
            # 把现有生产表重命名/重建。
            if "时段" not in columns or not set(BASE_DATASET_COLUMNS).issubset(columns):
                cur.execute(
                    f"RENAME TABLE `epf_pmos_96_full` TO `{legacy_name}`"
                )
                logger.warning(
                    "检测到旧版 epf_pmos_96_full，已保留为 %s；不会自动把旧数据当作权威数据",
                    legacy_name,
                )
            else:
                for column in DISCLOSURE_EXTENSION_COLUMNS:
                    if column in columns:
                        continue
                    cur.execute(
                        "ALTER TABLE `epf_pmos_96_full` "
                        f"ADD COLUMN `{column}` DECIMAL(14,4) DEFAULT NULL "
                        "COMMENT 'QCTC信息披露扩展字段'"
                    )
                    columns.add(column)
                    logger.info("[DB] 已增量新增信息披露字段: %s", column)
                if "日前一次出清价格" not in columns:
                    cur.execute(
                        "ALTER TABLE `epf_pmos_96_full` "
                        "ADD COLUMN `日前一次出清价格` DECIMAL(14,4) DEFAULT NULL "
                        "COMMENT '补充字段：日前首次发布/一次出清96点价格'"
                    )
                    logger.info("[DB] 已增量新增补充字段: 日前一次出清价格")
        cur.execute(CREATE_FULL_TABLE_SQL)
    conn.commit()
    logger.info("[DB] epf_pmos_96_full schema 已与权威 CSV 业务字段对齐")


# epf_pmos_96_full 的业务字段必须与
# data/96/authoritative/pmos_96_全量.csv 一一对应。
# 这里保留中文字段名，避免爬虫、数据库、模型输入之间再做一层隐式映射。
# market_date/时段是数据集原有键；unit_id/source_captured_at 是数据库审计元数据。
BASE_DATASET_COLUMNS: tuple[str, ...] = (
    "market_date", "时段",
    "直调负荷预测", "地方电厂出力预测", "外电预测", "风电预测", "光伏预测",
    "核电预测", "自备电厂预测", "试验机组预测",
    "直调负荷实际", "地方电厂出力实际", "外电实际", "风电实际", "光伏实际",
    "核电实际", "自备电厂实际", "试验机组实际", "抽蓄实际",
    "日前出清价格", "日前出力", "日前电量", "日前开机状态", "日前电源类型",
    "实时出清价格", "实时出力", "实时电量", "实时开机状态", "实时电源类型",
    "正备用预测", "负备用预测",
)

DISCLOSURE_EXTENSION_COLUMNS: tuple[str, ...] = (
    "全网负荷预测",
    "全网负荷实际",
    "直调负荷临时实际", "地方电厂出力临时实际", "外电临时实际",
    "风电临时实际", "光伏临时实际", "核电临时实际",
    "自备电厂临时实际", "试验机组临时实际", "抽蓄临时实际",
    "全网负荷临时实际",
    "边界全网负荷预测", "边界直调负荷预测", "边界外电预测",
    "边界风电预测", "边界光伏预测", "边界核电预测",
)

FULL_DATASET_COLUMNS: tuple[str, ...] = (
    *BASE_DATASET_COLUMNS,
    *DISCLOSURE_EXTENSION_COLUMNS,
)

FULL_NUMERIC_COLUMNS: frozenset[str] = frozenset(
    c for c in FULL_DATASET_COLUMNS
    if c not in {"market_date", "时段", "日前开机状态", "日前电源类型", "实时开机状态", "实时电源类型"}
)


def _rows_are_96(rows: list[dict[str, Any]] | None, require_price: bool = False) -> bool:
    """检查一套接口是否覆盖 1..96；价格接口额外检查 cqPrice 非空。"""
    rows = rows or []
    idx = {str(r.get("periodid", r.get("Periodid", ""))).strip(): r for r in rows}
    if len(rows) != 96 or len(idx) != 96:
        return False
    try:
        periods = {period_no_from_time(k) for k in idx}
    except Exception:
        return False
    if periods != set(range(1, 97)):
        return False
    if require_price and any(parse_number(r.get("cqPrice")) is None for r in idx.values()):
        return False
    return True


def _period_index(rows: list[dict[str, Any]] | None) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows or []:
        label = str(row.get("periodid", row.get("Periodid", ""))).strip()
        if not label:
            continue
        try:
            out[period_no_from_time(label)] = row
        except Exception:
            continue
    return out



def upsert_full_dataset_table(
    conn,
    market_date: str,
    unit_id: str,
    forecast_rows: list[dict[str, Any]],
    actual_rows: list[dict[str, Any]],
    da_rows: list[dict[str, Any]],
    rt_rows: list[dict[str, Any]],
    reserve_rows: list[dict[str, Any]] | None = None,
    captured_at: Any = None,
    da_first_rows: list[dict[str, Any]] | None = None,
    actual_temporary_rows: list[dict[str, Any]] | None = None,
    boundary_rows: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """将一日数据写入与权威 CSV 同字段的 ``epf_pmos_96_full``。

    该函数是 v3 EXE 的唯一合并表写入入口。数据库业务列直接使用 CSV 的中文列名，
    不再写入旧版英文派生列；各接口缺少的字段保持 NULL，不从其它接口补值。
    """
    actual_temporary_rows = actual_temporary_rows or []
    boundary_rows = boundary_rows or []
    if not any((
        forecast_rows, actual_rows, actual_temporary_rows, boundary_rows,
        da_first_rows, da_rows, rt_rows, reserve_rows,
    )):
        return {"rows": 0, "market_complete": 0, "merged_complete": 0}
    da_complete = int(_rows_are_96(da_rows, require_price=True))
    rt_complete = int(_rows_are_96(rt_rows, require_price=True))

    f_idx, a_idx = _period_index(forecast_rows), _period_index(actual_rows)
    a_tmp_idx = _period_index(actual_temporary_rows)
    boundary_idx = _period_index(boundary_rows)
    da_first_idx = _period_index(da_first_rows)
    da_idx, rt_idx = _period_index(da_rows), _period_index(rt_rows)
    reserve_idx: dict[int, dict[str, Any]] = {}
    for row in reserve_rows or []:
        try:
            pno = period_no_from_time(str(row.get("PERIODID", row.get("Periodid", ""))).strip())
        except Exception:
            continue
        typ = str(row.get("TYPE", "")).strip()
        if typ == "正备用":
            reserve_idx.setdefault(pno, {})["正备用预测"] = row.get("ZBY")
        elif typ == "负备用":
            reserve_idx.setdefault(pno, {})["负备用预测"] = row.get("ZBY")

    forecast_map = {
        "systemload": "直调负荷预测", "dfdcload": "地方电厂出力预测",
        "excload": "外电预测", "fdload": "风电预测", "gfload": "光伏预测",
        "sytsjz": "核电预测", "selfunit": "自备电厂预测", "syjzzj": "试验机组预测",
        "qwfh": "全网负荷预测",
    }
    actual_map = {
        "systemload": "直调负荷实际", "dfdcload": "地方电厂出力实际",
        "excload": "外电实际", "fdload": "风电实际", "gfload": "光伏实际",
        "hdload": "核电实际", "zbload": "自备电厂实际", "syjzload": "试验机组实际",
        "cxload": "抽蓄实际", "qwfh": "全网负荷实际",
    }
    actual_temporary_map = {
        "systemload": "直调负荷临时实际", "dfdcload": "地方电厂出力临时实际",
        "excload": "外电临时实际", "fdload": "风电临时实际", "gfload": "光伏临时实际",
        "hdload": "核电临时实际", "zbload": "自备电厂临时实际",
        "syjzload": "试验机组临时实际", "cxload": "抽蓄临时实际",
        "qwfh": "全网负荷临时实际",
    }
    boundary_map = {
        "qwfh": "边界全网负荷预测", "systemload": "边界直调负荷预测",
        "excload": "边界外电预测", "fdload": "边界风电预测",
        "gfload": "边界光伏预测", "sytsjz": "边界核电预测",
    }
    interface_map: dict[str, tuple[str, str]] = {
        zh: ("forecast", src) for src, zh in forecast_map.items()
    }
    interface_map.update({zh: ("actual", src) for src, zh in actual_map.items()})
    interface_map.update({zh: ("actual_tmp", src) for src, zh in actual_temporary_map.items()})
    interface_map.update({zh: ("boundary", src) for src, zh in boundary_map.items()})
    interface_map.update({
        "日前出清价格": ("da", "cqPrice"), "日前出力": ("da", "power"),
        "日前电量": ("da", "energy"), "日前开机状态": ("da", "kt"),
        "日前电源类型": ("da", "bq"),
        "实时出清价格": ("rt", "cqPrice"), "实时出力": ("rt", "power"),
        "实时电量": ("rt", "energy"), "实时开机状态": ("rt", "kt"),
        "实时电源类型": ("rt", "bq"),
    })

    captured = None
    if captured_at:
        try:
            captured = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            captured = None

    def period_label(pno: int) -> str:
        return "24:00" if pno == 96 else f"{pno // 4:02d}:{(pno % 4) * 15:02d}"

    rows: list[tuple[list[str], list[Any], list[str]]] = []
    for pno in range(1, 97):
        sources = {
            "forecast": f_idx.get(pno, {}),
            "actual": a_idx.get(pno, {}),
            "actual_tmp": a_tmp_idx.get(pno, {}),
            "boundary": boundary_idx.get(pno, {}),
            "da": da_idx.get(pno, {}),
            "rt": rt_idx.get(pno, {}),
        }
        fields: dict[str, Any] = {}
        first_price = parse_number(da_first_idx.get(pno, {}).get("cqPrice"))
        if first_price is not None:
            fields["日前一次出清价格"] = first_price
        for column in FULL_DATASET_COLUMNS[2:]:
            if column in {"正备用预测", "负备用预测"}:
                value = reserve_idx.get(pno, {}).get(column)
            else:
                source_name, source_col = interface_map[column]
                value = sources[source_name].get(source_col)
            parsed = parse_number(value) if column in FULL_NUMERIC_COLUMNS else (value or None)
            if parsed is not None:
                fields[column] = parsed
        if captured is not None:
            fields["source_captured_at"] = captured
        # 每个业务日保留96个结构时段，但只更新本次实际拿到的非空字段。
        db_cols = ["market_date", "时段", "unit_id", *fields]
        values = [market_date, period_label(pno), unit_id, *fields.values()]
        update_cols = list(fields)
        rows.append((db_cols, values, update_cols))

    with conn.cursor() as cur:
        for db_cols, values, update_cols in rows:
            quoted = lambda c: f"`{c}`"
            placeholders = ", ".join(["%s"] * len(db_cols))
            if update_cols:
                updates = ", ".join(
                    f"{quoted(c)}=COALESCE(VALUES({quoted(c)}), {quoted(c)})" for c in update_cols
                )
            else:
                updates = "`时段`=`时段`"
            sql = (
                f"INSERT INTO epf_pmos_96_full ({', '.join(map(quoted, db_cols))}) "
                f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updates}"
            )
            cur.execute(sql, values)
    logger.info(
        "full_table canonical upsert: %d rows date=%s forecast=%d actual=%d actual_tmp=%d boundary=%d da_first=%d da=%d rt=%d",
        len(rows), market_date, len(f_idx), len(a_idx), len(a_tmp_idx), len(boundary_idx),
        len(da_first_idx), len(da_idx), len(rt_idx),
    )
    return {
        "rows": len(rows), "market_complete": int(_rows_are_96(forecast_rows) and _rows_are_96(actual_rows)),
        "unit_da_complete": da_complete, "unit_rt_complete": rt_complete,
        "merged_complete": int(da_complete and rt_complete),
        "forecast_periods": len(f_idx), "actual_periods": len(a_idx),
        "actual_temporary_periods": len(a_tmp_idx),
        "forecast_boundary_periods": len(boundary_idx),
        "day_ahead_first_periods": len(da_first_idx),
        "day_ahead_periods": len(da_idx), "realtime_periods": len(rt_idx),
    }


# 兼容旧调用方；模块加载完成后统一指向 canonical 实现，避免任何旧入口继续
# 向已经废弃的英文宽表字段写入。
upsert_full_table = upsert_full_dataset_table


# ===================================================================
#  主流程
# ===================================================================


def main() -> None:
    """Prevent the support module from becoming a second production entry."""
    raise SystemExit(
        "run_crawler.py is a runtime support module. "
        "Use dist/crawler/crawl_96_auto_v7.exe instead."
    )


if __name__ == "__main__":
    main()
