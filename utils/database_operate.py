from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import pymysql
from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPF_ROOT = PROJECT_ROOT.parent / "epf"
LOCAL_ENV = PROJECT_ROOT / ".env"
EPF_ENV = EPF_ROOT / ".env"


def _load_env_sources() -> dict[str, str]:
    load_dotenv(dotenv_path=LOCAL_ENV, override=False)
    merged: dict[str, str] = {}

    if EPF_ENV.exists():
        merged.update({k: str(v) for k, v in dotenv_values(EPF_ENV).items() if v is not None})
    if LOCAL_ENV.exists():
        merged.update({k: str(v) for k, v in dotenv_values(LOCAL_ENV).items() if v is not None})

    for key in ("DB_HOST", "DB", "DB_USER", "DB_PWD", "DB_PORT", "DB_CONNECT_TIMEOUT"):
        env_value = os.getenv(key)
        if env_value:
            merged[key] = env_value
    return merged


def get_db_connection():
    cfg = _load_env_sources()
    host = cfg.get("DB_HOST", "").strip().strip("'\"")
    database = cfg.get("DB", "").strip().strip("'\"")
    user = cfg.get("DB_USER", "").strip().strip("'\"")
    password = cfg.get("DB_PWD", "").strip().strip("'\"")
    port_str = cfg.get("DB_PORT", "3306").strip().strip("'\"")
    timeout_str = cfg.get("DB_CONNECT_TIMEOUT", "10").strip().strip("'\"")

    if not all([host, database, user, password]):
        raise ValueError(
            "Database env vars are incomplete. Required: DB_HOST, DB, DB_USER, DB_PWD. "
            f"Looked in: {LOCAL_ENV} and fallback {EPF_ENV}"
        )

    try:
        port = int(port_str)
    except ValueError:
        port = 3306

    try:
        connect_timeout = int(timeout_str)
    except ValueError:
        connect_timeout = 10

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=connect_timeout,
    )


def fetch_web_grid_data(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch market data from the database, optionally filtered by time range.

    Parameters
    ----------
    start_time : str, optional
        If provided, only rows with ``data_time >= start_time`` are returned.
        Accepts any pandas-compatible datetime string.
    end_time : str, optional
        If provided, only rows with ``data_time <= end_time`` are returned.

    Returns
    -------
    pd.DataFrame with columns matching the 1.0 field mapping.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query = (
                "SELECT "
                "data_time as 时刻, "
                "price_dayahead as 日前电价, "
                "price_realtime as 实时电价, "
                "fcast_local_plant as 地方电厂总加预测值, "
                "fcast_tie_line as 联络线受电负荷预测值, "
                "fcast_wind as 风电总加预测值, "
                "fcast_solar as 光伏总加预测值, "
                "fcast_nuclear as 核电总加预测值, "
                "fcast_self_owned as 自备机组总加预测值, "
                "fcast_test_unit as 试验机组总加预测值, "
                "fcast_direct_load as 直调负荷预测值, "
                "fcast_bidding_space as 竞价空间预测值, "
                "fcast_new_energy as 新能源总加预测值, "
                "actual_local_plant as 地方电厂总加实际值, "
                "actual_tie_line as 联络线受电负荷实际值, "
                "actual_wind as 风电总加实际值, "
                "actual_solar as 光伏总加实际值, "
                "actual_nuclear as 核电总加实际值, "
                "actual_self_owned as 自备机组总加实际值, "
                "actual_test_unit as 试验机组总加实际值, "
                "actual_direct_load as 直调负荷实际值, "
                "actual_bidding_space as 竞价空间实际值, "
                "actual_new_energy as 新能源总加实际值 "
                "FROM epf_market_data"
            )

            params: list[str] = []
            where_clauses: list[str] = []
            if start_time is not None:
                where_clauses.append("data_time >= %s")
                params.append(start_time)
            if end_time is not None:
                where_clauses.append("data_time <= %s")
                params.append(end_time)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += " ORDER BY data_time ASC;"

            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        conn.close()

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["时刻"] = pd.to_datetime(frame["时刻"], errors="coerce")
    frame = frame.sort_values("时刻").reset_index(drop=True)
    return frame


# ═══════════════════════════════════════════════════════════════
#  96-point (15-min) data queries
# ═══════════════════════════════════════════════════════════════


def fetch_market_data_96(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch 96-point market feature data from ``epf_market_data_96``.

    Parameters
    ----------
    start_date, end_date : str, optional
        ``YYYY-MM-DD`` range filter on *market_date*.

    Returns
    -------
    pd.DataFrame with columns:
        时刻, 直调负荷, 地方电厂出力, 外电, 风电, 光伏,
        核电, 自备电厂, 试验机组, 直调负荷预测, ...
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query = (
                "SELECT "
                "data_time AS 时刻, "
                "market_date, period_no, "
                "actual_direct_load AS 直调负荷, "
                "actual_local_plant AS 地方电厂出力, "
                "actual_tie_line AS 外电, "
                "actual_wind AS 风电, "
                "actual_solar AS 光伏, "
                "actual_nuclear AS 核电, "
                "actual_self_owned AS 自备电厂, "
                "actual_test_unit AS 试验机组, "
                "actual_unit_maintenance AS 机组检修, "
                "actual_pos_reserve AS 正备用, "
                "actual_neg_reserve AS 负备用, "
                "actual_bidding_space AS 竞价空间, "
                "actual_new_energy AS 新能源, "
                "fcast_direct_load AS 直调负荷预测, "
                "fcast_local_plant AS 地方电厂出力预测, "
                "fcast_tie_line AS 外电预测, "
                "fcast_wind AS 风电预测, "
                "fcast_solar AS 光伏预测, "
                "fcast_nuclear AS 核电预测, "
                "fcast_self_owned AS 自备电厂预测, "
                "fcast_test_unit AS 试验机组预测, "
                "fcast_unit_maintenance AS 机组检修预测, "
                "fcast_pos_reserve AS 正备用预测, "
                "fcast_neg_reserve AS 负备用预测, "
                "fcast_bidding_space AS 竞价空间预测, "
                "fcast_new_energy AS 新能源预测 "
                "FROM epf_market_data_96"
            )

            params: list[str] = []
            where: list[str] = []
            if start_date is not None:
                where.append("market_date >= %s")
                params.append(start_date)
            if end_date is not None:
                where.append("market_date <= %s")
                params.append(end_date)
            if where:
                query += " WHERE " + " AND ".join(where)
            query += " ORDER BY data_time ASC;"

            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        conn.close()

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["时刻"] = pd.to_datetime(frame["时刻"], errors="coerce")
        frame = frame.sort_values("时刻").reset_index(drop=True)
    return frame


def fetch_unit_data_96(
    unit_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch unit-level 96-point price/power data from ``epf_unit_data_96``.

    Parameters
    ----------
    unit_id : str, optional
        Filter by unit.  ``None`` returns all units.
    start_date, end_date : str, optional
        ``YYYY-MM-DD`` range filter on *market_date*.

    Returns
    -------
    pd.DataFrame with columns:
        时刻, 日前电价, 实时电价, 日前出力, 实时出力, ...
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query = (
                "SELECT "
                "data_time AS 时刻, "
                "market_date, period_no, unit_id, "
                "da_cq_price AS 日前电价, "
                "da_power AS 日前出力, "
                "da_energy AS 日前电量, "
                "da_status AS 日前开机状态, "
                "rt_cq_price AS 实时电价, "
                "rt_power AS 实时出力, "
                "rt_energy AS 实时电量, "
                "rt_status AS 实时开机状态 "
                "FROM epf_unit_data_96"
            )

            params: list[str] = []
            where: list[str] = []
            if unit_id is not None:
                where.append("unit_id = %s")
                params.append(unit_id)
            if start_date is not None:
                where.append("market_date >= %s")
                params.append(start_date)
            if end_date is not None:
                where.append("market_date <= %s")
                params.append(end_date)
            if where:
                query += " WHERE " + " AND ".join(where)
            query += " ORDER BY data_time, unit_id ASC;"

            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        conn.close()

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["时刻"] = pd.to_datetime(frame["时刻"], errors="coerce")
        frame = frame.sort_values("时刻").reset_index(drop=True)
    return frame


# ═══════════════════════════════════════════════════════════════
#  Native 96-point (15-min) read-only mirror queries
#  These are LOCAL SYNCHRONIZATION paths only — they never write,
#  update, delete, or alter the remote database. They use `SELECT`
#  (optionally aggregate `SELECT COUNT/MIN/MAX`) with bounded params.
#  Original remote column names are preserved (no Chinese aliases).
# ═══════════════════════════════════════════════════════════════


def fetch_96_table(
    table: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    extra_where: Optional[list[str]] = None,
    extra_params: Optional[list] = None,
    columns: Optional[list[str]] = None,
    order_by: str = "data_time ASC",
) -> pd.DataFrame:
    """Generic read-only fetch of a 96-point table.

    Preserves original remote column names. Returns a DataFrame sorted by
    *order_by*. Only ever issues a bounded ``SELECT`` — never mutates the DB.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            col_clause = ", ".join(columns) if columns else "*"
            query = f"SELECT {col_clause} FROM {table}"
            where: list[str] = []
            params: list = []
            if start_date is not None:
                where.append("market_date >= %s")
                params.append(start_date)
            if end_date is not None:
                where.append("market_date <= %s")
                params.append(end_date)
            for w in (extra_where or []):
                where.append(w)
            for p in (extra_params or []):
                params.append(p)
            if where:
                query += " WHERE " + " AND ".join(where)
            query += f" ORDER BY {order_by};"
            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows)


def fetch_market_data_96_full(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch the full ``epf_market_data_96`` table (market-level 96-point).

    Returns original remote columns (data_time, market_date, period_no,
    actual_*, fcast_*, create_time, update_time...). Read-only.
    """
    return fetch_96_table("epf_market_data_96", start_date=start_date, end_date=end_date)


def fetch_unit_data_96_full(
    unit_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch the full ``epf_unit_data_96`` table (unit-level 96-point prices).

    Returns original remote columns (data_time, market_date, period_no,
    unit_id, da_cq_price, rt_cq_price, da_power, rt_power, ...). Read-only.
    """
    extra_where: list[str] = []
    extra_params: list = []
    if unit_id is not None:
        extra_where.append("unit_id = %s")
        extra_params.append(unit_id)
    return fetch_96_table(
        "epf_unit_data_96",
        start_date=start_date,
        end_date=end_date,
        extra_where=extra_where,
        extra_params=extra_params,
        order_by="data_time ASC, unit_id ASC",
    )


def fetch_96_table_summary(table: str) -> dict:
    """Read-only aggregate summary (MIN/MAX market_date, row count).

    Used for remote/local row-count reconciliation. Never mutates the DB.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            query = (
                f"SELECT "
                f"MIN(market_date) AS d_min, "
                f"MAX(market_date) AS d_max, "
                f"COUNT(*) AS rows_total "
                f"FROM {table}"
            )
            cursor.execute(query)
            rows = cursor.fetchall()
    finally:
        conn.close()
    if not rows:
        return {"d_min": None, "d_max": None, "rows_total": 0}
    row = rows[0]
    return {
        "d_min": str(row.get("d_min")) if row.get("d_min") is not None else None,
        "d_max": str(row.get("d_max")) if row.get("d_max") is not None else None,
        "rows_total": int(row.get("rows_total") or 0),
    }


def fetch_96_table_consistent(
    table: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    extra_where: Optional[list[str]] = None,
    extra_params: Optional[list] = None,
    columns: Optional[list[str]] = None,
    order_by: str = "data_time ASC",
) -> tuple[pd.DataFrame, dict]:
    """Read a 96-point table and its summary from one InnoDB snapshot.

    The formal 96 production sync reads a live table while the crawler may be
    appending rows.  Keeping the detail query and COUNT/MIN/MAX query on the
    same ``WITH CONSISTENT SNAPSHOT`` transaction prevents a harmless concurrent
    append from being reported as a false row-count mismatch.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT")

            col_clause = ", ".join(columns) if columns else "*"
            query = f"SELECT {col_clause} FROM {table}"
            where: list[str] = []
            params: list = []
            if start_date is not None:
                where.append("market_date >= %s")
                params.append(start_date)
            if end_date is not None:
                where.append("market_date <= %s")
                params.append(end_date)
            for condition in (extra_where or []):
                where.append(condition)
            params.extend(extra_params or [])
            if where:
                query += " WHERE " + " AND ".join(where)
            query += f" ORDER BY {order_by};"
            cursor.execute(query, params)
            rows = cursor.fetchall()

            cursor.execute(
                f"SELECT MIN(market_date) AS d_min, "
                f"MAX(market_date) AS d_max, "
                f"MAX(update_time) AS latest_update_time, "
                f"COUNT(*) AS rows_total "
                f"FROM {table}"
            )
            summary_row = cursor.fetchone() or {}
        conn.rollback()
    finally:
        conn.close()

    summary = {
        "d_min": str(summary_row.get("d_min")) if summary_row.get("d_min") is not None else None,
        "d_max": str(summary_row.get("d_max")) if summary_row.get("d_max") is not None else None,
        "latest_update_time": (
            str(summary_row.get("latest_update_time"))
            if summary_row.get("latest_update_time") is not None
            else None
        ),
        "rows_total": int(summary_row.get("rows_total") or 0),
    }
    return pd.DataFrame(rows), summary


def get_db_server_version() -> str:
    """Read-only: return the MySQL server version string.

    Never mutates the DB. Used for the synchronization manifest.
    """
    conn = get_db_connection()
    try:
        return conn.get_server_info()
    finally:
        conn.close()
