"""PMOS AUX-V1 辅助信息披露采集与解析。

本模块只负责辅助信息来源；96 点主爬虫的业务接口、字段映射和写入器均不在
这里导入。认证/浏览器 transport 通过 ``PmosCrawler`` 旁路复用。
"""

from __future__ import annotations

import calendar
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit

import requests

try:
    from scripts.crawler.collect.crawl import (
        PmosCrawler, parse_number, _CdpClient, _get_pmos_page,
        _navigate_for_fetch, _same_origin,
    )
except ImportError:  # pragma: no cover - frozen/直接脚本导入
    from crawl import PmosCrawler, parse_number, _CdpClient, _get_pmos_page, _navigate_for_fetch, _same_origin

logger = logging.getLogger("disclosure_aux")


def _safe_diag_text(value: Any, limit: int = 400) -> str:
    """[AUX-V1-r10-diag] Keep error context while masking common credentials."""
    text = str(value or "")
    text = re.sub(r"(?i)(authorization|cookie|set-cookie|token|password|csrf|ticket|bearer|tk)(\s*[=:]\s*)[^\s,;]+",
                  r"\1\2<redacted>", text)
    return text.replace("\r", " ").replace("\n", " ")[:limit]

BUILD_VERSION = "2026-10-08-disclosure-aux-v1-r14"  # [AUX-V1-r14] record_key粒度修复/CSRF导航诊断/zcq_contract_curve自动unitid
AUX_SCHEMA_VERSION = "AUX-V1"
STATUS_COMPLETE = "COMPLETE"
STATUS_EMPTY_VALID = "EMPTY_VALID"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED_SOURCE = "FAILED_SOURCE"
STATUS_SKIPPED_NOT_READY = "SKIPPED_NOT_READY"


class AuxAuthRejected(RuntimeError):
    """A 401/403 must stop the whole AUX run."""


class AuxBrowserLost(RuntimeError):
    """[AUX-V1-r11f] 承载采集的浏览器进程已死亡，必须重新走三层防护认证。

    判据：连续多个源的异常都指向本地 CDP 调试端口（127.0.0.1:922x）。
    此前的实现只记一条 warning 就 continue，导致浏览器死亡后整轮长跑静默白跑
    直到人工停止；现在改为向上抛出，由 AUX 入口触发完整的三层防护重认证。
    """


def _is_browser_lost(exc: BaseException) -> bool:
    """[AUX-V1-r11f] 判据 N3：异常是否指向本地 CDP 调试端口。

    与「PMOS 域名不可达」区分对待：目标是 127.0.0.1:922x 说明是浏览器进程没了
    （需要重新登录），目标是业务域名说明是网络抖动（按可重试处理）。
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if "127.0.0.1" not in text and "localhost" not in text:
        return False
    return "port=" in text or ":922" in text


@dataclass(frozen=True)
class SourceSpec:
    name: str
    group: str
    method: str
    path: str
    page_url: str
    resolution: str
    parser: str
    target_table: str | None
    enabled_by_default: bool = True
    raw_only: bool = False
    evidence_level: str = "UNVERIFIED"
    frontend_path: str = ""
    param_contract: tuple[str, ...] = ()
    pagination_mode: str = "none"
    max_pages: int = 200
    max_rows: int = 100000
    # [AUX-V1-r3] Some confirmed legacy endpoints are POST requests whose
    # frontend contract keeps all arguments in the query string.
    params_in_query: bool = False

    @property
    def request_path(self) -> str:
        """[AUX-V1-r1] Frontend path is distinct from the resolved gateway URL."""
        return self.frontend_path or self.path


@dataclass
class SourceResult:
    name: str
    group: str
    status: str
    http_status: int | None
    business_code: str | None
    rows: list[dict[str, Any]]
    raw: dict[str, Any]
    error: str = ""
    transport_failure: bool = False


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def canonical_text(parts: Iterable[Any]) -> str:
    """Canonical identity text. Empty components are retained in order."""
    return "|".join("" if value is None else str(value).strip() for value in parts)


def record_key(*parts: Any) -> str:
    return hashlib.sha256(canonical_text(parts).encode("utf-8")).hexdigest()


def raw_hash(payload: Any) -> str:
    text = json.dumps(_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_key(source_api: str, params: Mapping[str, Any], business_date: str | None) -> str:
    safe = {str(k): params[k] for k in sorted(params) if str(k).lower() not in {"cookie", "token", "authorization", "password"}}
    return record_key(source_api, json.dumps(_clean(safe), ensure_ascii=False, sort_keys=True), business_date or "")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _records_total(payload: Any) -> int | None:
    """[AUX-V1-r2] Read pagination total from top-level or nested PMOS wrapper."""
    if not isinstance(payload, Mapping):
        return None
    for key in ("recordsTotal", "totalCount", "records", "total"):
        value = payload.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    for key in ("data", "result", "tableData"):
        nested = _records_total(payload.get(key))
        if nested is not None:
            return nested
    return None


def _payload_rows(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Extract common PMOS wrappers without guessing a business schema."""
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, Mapping)], None
    if not isinstance(payload, Mapping):
        return [], None
    code = payload.get("code", payload.get("status", payload.get("resultCode")))
    for key in ("data", "rows", "list", "tableData", "TableData", "dataList", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(x) for x in value if isinstance(x, Mapping)], None if code is None else str(code)
        if isinstance(value, Mapping):
            rows, nested_code = _payload_rows(value)
            if rows or value == {}:
                return rows, nested_code if nested_code is not None else (None if code is None else str(code))
    return [], None if code is None else str(code)


def _payload_container_state(payload: Any) -> str:
    """[AUX-V1-r1] Distinguish explicit empty from an unknown successful dict."""
    if isinstance(payload, list):
        return "empty" if not payload else "rows"
    if not isinstance(payload, Mapping):
        return "unknown"
    if payload.get("recordsTotal") == 0 or payload.get("totalCount") == 0:
        return "empty"
    for key in ("data", "rows", "list", "tableData", "TableData", "dataList", "result"):
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, list):
            return "empty" if not value else "rows"
        if isinstance(value, Mapping):
            nested = _payload_container_state(value)
            if nested != "unknown":
                return nested
            # Known parallel-array/chart wrappers are structured data even
            # though they are not row lists.  [AUX-V1-r11] adds the wrappers
            # observed in HAR20 for the informationDisclosure gateway.
            if any(k in value for k in (
                "pointList", "valList", "pdateList", "valueList", "tableData", "TableData",
                "blockTreeData", "chartsMap", "tieLineTableCols", "sparexx", "updateTime",
            )):
                return "rows"
            return "unknown"
    return "unknown"


def _status_for_payload(http_status: int, payload: Any, rows: list[dict[str, Any]]) -> tuple[str, str | None]:
    if http_status >= 400 or http_status == 0:
        return STATUS_FAILED_SOURCE, None
    code = None
    if isinstance(payload, Mapping):
        raw_code = payload.get("code", payload.get("status", payload.get("resultCode")))
        code = None if raw_code is None else str(raw_code)
        if code not in (None, "0", "200", "true", "True", "success", "SUCCESS"):
            return STATUS_FAILED_SOURCE, code
        if payload.get("partial") is True or str(payload.get("sourceStatus", "")).upper() == STATUS_PARTIAL:
            return STATUS_PARTIAL, code
    container_state = _payload_container_state(payload)
    if rows or container_state == "rows":
        return STATUS_COMPLETE, code
    if container_state == "empty":
        # HTTP 200 plus explicit success/empty data is a valid empty source.
        return STATUS_EMPTY_VALID, code
    # HTTP 200 with an unrecognised object is not an empty business result.
    return STATUS_PARTIAL, code


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def parse_unit(payload: Any, *, snapshot_date: str | None = None, business_date: str | None = None, source_api: str = "") -> list[dict[str, Any]]:
    # [AUX-V1] unit master parser; platform fuel/type fields remain authoritative.
    snapshot_date = snapshot_date or business_date
    rows, _ = _payload_rows(payload)
    out = []
    for row in rows:
        unit_id = _row_value(row, "unitid", "unitId", "id", "ddunitid")
        plant_id = _row_value(row, "plantid", "plantId", "ddplantid")
        identity = record_key(source_api, snapshot_date or "", unit_id or "", plant_id or "", _row_value(row, "unitname", "unitName") or "")
        out.append({
            "record_key": identity, "snapshot_date": snapshot_date,
            "plant_id": plant_id, "plant_name": _row_value(row, "plantname", "plantName"),
            "unit_id": unit_id, "unit_name": _row_value(row, "unitname", "unitName"),
            "gengroup_name": _row_value(row, "gengroupname", "gengroupName"),
            "dispatch_plant_id": _row_value(row, "ddplantid"), "dispatch_plant_name": _row_value(row, "ddplantname"),
            "dispatch_unit_id": _row_value(row, "ddunitid"), "dispatch_unit_name": _row_value(row, "ddunitname"),
            "fuel_type": _row_value(row, "fuel_type", "fuelType"), "unit_type": _row_value(row, "type", "jzlx"),
            "base_type": _row_value(row, "basetype", "baseType"), "app_type": _row_value(row, "apptype", "appType"),
            "state": _row_value(row, "state", "status"),
            "max_capacity_mw": parse_number(_row_value(row, "maxcapacity", "maxCapacity", "maxCap")),
            "min_capacity_mw": parse_number(_row_value(row, "mincapacity", "minCapacity", "min_cap")),
            "real_min_capacity_mw": parse_number(_row_value(row, "realMincapacity", "realMinCapacity")),
            "normal_capacity_mw": parse_number(_row_value(row, "normal_cap", "normalCap")),
            "max_capacity_winter_mw": parse_number(_row_value(row, "maxCapWinter")),
            "max_capacity_summer_mw": parse_number(_row_value(row, "maxCapSummer")),
            "emergency_min_capacity_mw": parse_number(_row_value(row, "emergencyMinCap")),
            "begin_time": _row_value(row, "begintime", "beginTime"), "end_time": _row_value(row, "endtime", "endTime"),
            "up_rate": parse_number(_row_value(row, "upRate")), "down_rate": parse_number(_row_value(row, "dnRate", "downRate")),
            "min_on_time": parse_number(_row_value(row, "minOnTime")), "min_off_time": parse_number(_row_value(row, "minOffTime")),
            "max_onoff_times": parse_number(_row_value(row, "maxOnoffTimes")),
            "fac_rate": parse_number(_row_value(row, "facRate")), "cold_note_time": parse_number(_row_value(row, "coldNoteTime")),
            "warm_note_time": parse_number(_row_value(row, "warmNoteTime")), "hot_note_time": parse_number(_row_value(row, "hotNoteTime")),
            "mhcs": parse_number(_row_value(row, "mhcs")), "kzcb": parse_number(_row_value(row, "kzcb")),
            "bdcb": parse_number(_row_value(row, "bdcb")), "apex_cap": parse_number(_row_value(row, "apexCap")),
            "is_combined_cycle": _row_value(row, "isCombinedCycle"), "is_heat_supply": _row_value(row, "isHeatSupply"),
            "heat_type": _row_value(row, "heatType"), "cyscfs": _row_value(row, "cyscfs"),
            "source_api": source_api, "extra_json": json.dumps(_clean(row), ensure_ascii=False),
        })
    return out


def parse_constraint(payload: Any, *, business_date: str | None = None, source_api: str = "") -> list[dict[str, Any]]:
    # [AUX-V1] constraint parser; pointMap96 is retained as source JSON.
    rows, _ = _payload_rows(payload)
    out = []
    for row in rows:
        unit_id = _row_value(row, "unitid", "unitId")
        plan_id = _row_value(row, "planid", "planId")
        name = _row_value(row, "name", "constraintName")
        value = _row_value(row, "value", "constraintValue")
        out.append({
            "record_key": record_key(source_api, business_date or "", unit_id or "", plan_id or "", name or "", _row_value(row, "point_no", "pointNo") or ""),
            "business_date": business_date, "unit_id": unit_id, "unit_name": _row_value(row, "unitname", "unitName"),
            "plant_id": _row_value(row, "plantid", "plantId"), "plant_name": _row_value(row, "plantname", "plantName"),
            "plan_id": plan_id, "constraint_type": _row_value(row, "constraintType", "type"),
            "constraint_name": name, "constraint_value": parse_number(value),
            "constraint_value_text": None if parse_number(value) is not None else (None if value is None else str(value)),
            "sb_type": _row_value(row, "sbType"), "point_no": _row_value(row, "point_no", "pointNo"),
            "point_map96_json": json.dumps(_clean(_row_value(row, "pointMap96")), ensure_ascii=False) if _row_value(row, "pointMap96") is not None else None,
            "jzlx_cn": _row_value(row, "jzlxCn"), "source_api": source_api,
            "extra_json": json.dumps(_clean(row), ensure_ascii=False),
        })
    return out


def _event_rows(payload: Any, event_type: str) -> list[dict[str, Any]]:
    rows, _ = _payload_rows(payload)
    out = []
    for row in rows:
        unit_id = _row_value(row, "unitID", "unitId", "Unitid", "unitid")
        start = _row_value(row, "startTime", "maintenanceStartTime", "kssj", "start_time")
        end = _row_value(row, "endTime", "maintenanceEndTime", "jssj", "end_time")
        source_event_id = _row_value(row, "approveApplyId", "spid", "id", "eventId")
        out.append({
            "record_key": record_key(event_type, source_event_id or "", unit_id or "", start or "", end or ""),
            "event_type": event_type, "unit_id": unit_id, "unit_name": _row_value(row, "unitName", "unitname"),
            "plant_id": _row_value(row, "plantID", "plantId", "plantid"), "plant_name": _row_value(row, "plantName", "plantname"),
            "trade_unit_id": _row_value(row, "tradeUnitId"), "source_event_id": source_event_id,
            "start_time": start, "end_time": end, "work_nature": _row_value(row, "workNature", "gzxz"),
            "tag_type": _row_value(row, "Type", "type", "level"), "description": _row_value(row, "gznr", "sbName", "value"),
            "apply_reason": _row_value(row, "sqyy"), "capacity_mw": parse_number(_row_value(row, "Capacity", "capacity")),
            "kj_count": _row_value(row, "kjCount"), "tj_count": _row_value(row, "tjCount"), "csbm": _row_value(row, "csbm"),
            "event_level": _row_value(row, "level"), "event_value": _row_value(row, "value"),
            "approve_apply_id": _row_value(row, "approveApplyId"), "approve_state": _row_value(row, "state"),
            "approver": _row_value(row, "spr"), "approve_time": _row_value(row, "spsj"), "approve_opinion": _row_value(row, "spyj"),
            "zxcs": parse_number(_row_value(row, "zxcs")), "fxcs": parse_number(_row_value(row, "fxcs")),
            "max_zby": parse_number(_row_value(row, "maxZby")), "min_zby": parse_number(_row_value(row, "minZby")),
            "max_fby": parse_number(_row_value(row, "maxFby")), "min_fby": parse_number(_row_value(row, "minFby")),
            "source_api": "", "extra_json": json.dumps(_clean(row), ensure_ascii=False),
        })
    return out


def parse_event(payload: Any, *, event_type: str, business_date: str | None = None, source_api: str = "") -> list[dict[str, Any]]:
    # [AUX-V1-r1] explicit Tsjz/Sbdjx/Dwby wrappers; one event remains one row.
    data = payload.get("data", payload) if isinstance(payload, Mapping) else payload
    if isinstance(data, Mapping):
        if isinstance(data.get("tableData"), list):
            rows = [dict(item) for item in data["tableData"] if isinstance(item, Mapping)]
            for row in rows:
                for key in ("kjCount", "tjCount"):
                    if key not in row and key in data:
                        row[key] = data[key]
            payload = rows
        elif isinstance(data.get("data"), list):
            rows = [dict(item) for item in data["data"] if isinstance(item, Mapping)]
            for row in rows:
                for key in ("totalCount", "maxZby", "minZby", "maxFby", "minFby"):
                    if key not in row and key in data:
                        row[key] = data[key]
            payload = rows
    rows = _event_rows(payload, event_type)
    for row in rows:
        row["business_date"] = business_date
        row["source_api"] = source_api
        row["record_key"] = record_key(source_api, row["event_type"], row.get("source_event_id") or "", row.get("unit_id") or "", row.get("start_time") or "", row.get("end_time") or "")
    return rows


def parse_curve(payload: Any, *, curve_type: str, business_date: str | None = None, source_api: str = "") -> list[dict[str, Any]]:
    # [AUX-V1-r1] confirmed curve parser; primary ForecastData-equivalent series remain raw-only.
    if source_api.endswith("getFhChar") or "getZdLlx" in source_api:
        return []  # 主96点同义/语义未确认：raw-only
    rows, _ = _payload_rows(payload)
    data = payload.get("data", payload) if isinstance(payload, Mapping) else payload
    if isinstance(data, Mapping):
        points = data.get("pointList", data.get("pdateList", [])) or []
        values = data.get("valList", data.get("valueList", [])) or []
        types = data.get("typeList", []) or []
        if values and isinstance(values[0], (list, tuple)):
            rows = []
            for series_no, series_values in enumerate(values):
                series_type = types[series_no] if series_no < len(types) else series_no
                for point_no, point in enumerate(points):
                    rows.append({"point": point, "value": series_values[point_no] if point_no < len(series_values) else None, "type": series_type, "series_no": series_no})
        elif points and values:
            rows = [{"point": p, "value": values[i] if i < len(values) else None, "type": types[i] if i < len(types) else None} for i, p in enumerate(points)]
    out = []
    for i, row in enumerate(rows):
        point = _row_value(row, "point", "pointTime", "pdate", "time")
        value = _row_value(row, "value", "val", "valueMw")
        out.append({
            "record_key": record_key(source_api, curve_type, business_date or "", _row_value(row, "unitid", "unitId") or "", "", i, point or ""),
            "curve_type": curve_type, "business_date": business_date,
            "unit_id": _row_value(row, "unitid", "unitId"), "unit_name": _row_value(row, "unitname", "unitName"),
            "series_name": _row_value(row, "seriesName", "name"), "series_type": _row_value(row, "type", "seriesType"),
            "point_seq": i + 1, "period_id": _row_value(row, "periodid", "periodId"), "period_name": point,
            "point_time": point, "value_mw": parse_number(value), "min_value_mw": parse_number(_row_value(row, "minValue", "min")),
            "max_value_mw": parse_number(_row_value(row, "maxValue", "max")), "source_api": source_api,
            "extra_json": json.dumps(_clean(row), ensure_ascii=False),
        })
    return out


def parse_stat(payload: Any, *, stat_type: str, business_date: str | None = None, source_api: str = "") -> list[dict[str, Any]]:
    # [AUX-V1] thermal-unit count/stat parser using platform query dimensions.
    rows, _ = _payload_rows(payload)
    out = []
    for row in rows:
        period = _row_value(row, "periodid", "periodId", "point")
        name = _row_value(row, "name", "cnName", "type")
        out.append({
            "record_key": record_key(source_api, stat_type, business_date or "", period or "", name or ""),
            "stat_type": stat_type, "business_date": business_date, "period_id": period,
            "period_name": _row_value(row, "periodname", "periodName"), "name": name,
            "cn": _row_value(row, "cn"), "capacity": parse_number(_row_value(row, "capacity")),
            "dakj": _row_value(row, "dakj"), "dakj_capacity": parse_number(_row_value(row, "dakjCapacity")),
            "daqt": _row_value(row, "daqt"), "daqt_capacity": parse_number(_row_value(row, "daqtCapacity")),
            "zcqdl": parse_number(_row_value(row, "zcqdl")), "zcqdj": parse_number(_row_value(row, "zcqdj")),
            "dadl": parse_number(_row_value(row, "dadl")), "dadj": parse_number(_row_value(row, "dadj")),
            "dapcdl": parse_number(_row_value(row, "dapcdl")), "rtdl": parse_number(_row_value(row, "rtdl")),
            "rtdj": parse_number(_row_value(row, "rtdj")), "rtpcdl": parse_number(_row_value(row, "rtpcdl")),
            "zcdadj": parse_number(_row_value(row, "zcdadj")), "source_api": source_api,
            "extra_json": json.dumps(_clean(row), ensure_ascii=False),
        })
    return out


def parse_contract(payload: Any, *, record_type: str, business_date: str | None = None, source_api: str = "", unit_id: str | None = None, use_row_date: bool = False) -> list[dict[str, Any]]:
    # [AUX-V1] contract parser; jzdlzb is preserved without inventing a ratio definition.
    # [AUX-V1-r13] zcq_contract_curve24/96 行内自带 pdate 且无 unitid 字段：
    # use_row_date=True 时 record_key 取行内 pdate（否则同月同点行互相碰撞），
    # unit_id 回落到采集时的 dyid（调用方注入）。
    rows, _ = _payload_rows(payload)
    out = []
    for row in rows:
        row_unit_id = _row_value(row, "unitid", "unitId") or unit_id
        row_date = str(_row_value(row, "pdate") or "") if use_row_date else ""
        key_date = row_date or (business_date or "")
        period = _row_value(row, "periodid", "periodId", "point")
        customer = _row_value(row, "customername", "customname", "qyname")
        item = {
            "record_key": record_key(source_api, record_type, key_date, _row_value(row, "dmonth", "contractMonth") or "", row_unit_id or "", customer or "", period or "", _row_value(row, "planid", "planId") or ""),
            "record_type": record_type, "business_date": row_date or business_date, "contract_month": _row_value(row, "dmonth", "contractMonth"),
            "period_id": period, "period_name": _row_value(row, "periodname", "periodName"),
            "plant_id": _row_value(row, "plantid", "plantId"), "plant_name": _row_value(row, "plantname", "plantName"),
            "unit_id": row_unit_id, "unit_name": _row_value(row, "unitname", "unitName"), "customer_name": customer,
            "qyname": _row_value(row, "qyname"), "plan_id": _row_value(row, "planid", "planId"),
            "contract_type": _row_value(row, "type", "contractType"), "quantity": parse_number(_row_value(row, "quantity", "kmrdl", "jhydl", "cjdl")),
            "price": parse_number(_row_value(row, "price", "jhycjdl", "cjjj")), "limit_value": parse_number(_row_value(row, "limit", "kmcdl")),
            "jzdlzb": parse_number(_row_value(row, "jzdlzb")), "ratio_definition": "source_field:jzdlzb, business definition pending" if "jzdlzb" in row else None,
            "ratio_status": "UNDEFINED" if "jzdlzb" in row else None,
            "source_api": source_api, "extra_json": json.dumps(_clean(row), ensure_ascii=False),
        }
        # Preserve known source names without interpreting the contract semantics.
        for name in ("sbxsdl", "sbxsdj", "hyzrdl", "hyzrdj", "sbgpdl", "sbgpdj", "dxhydl", "dxhydj", "ynlxchdl", "ynlxchdj", "ydlxchdl", "ydlxchdj", "ljjhydl", "ljjhydj", "ljdl", "kmrdl", "jhydlsx", "jhycjdl", "syjhydl", "ljhydlsx", "ljcjdl", "syljdl", "kmcdl", "kmchj", "lsydl", "ymrdl", "jhydl", "ljjydl", "sbr", "cost", "dh", "spower", "epower", "costjm", "sbsj", "wtupcost", "ltupcost", "rtupcost", "kzcost", "wtupcostjm", "ltupcostjm", "rtupcostjm", "kztype", "cydl", "cjdl", "cjjj"):
            item[name] = parse_number(row.get(name)) if name not in {"sbr", "dh", "sbsj", "kztype"} else row.get(name)
        out.append(item)
    return out


def parse_disclosure(payload: Any, *, source_api: str = "", business_date: str | None = None) -> list[dict[str, Any]]:
    """[AUX-V1-r11] QCTC informationDisclosure parser.

    HAR20 shows this gateway prefix returning several shapes: ``data`` as a row
    list, ``data.tableData``, ``data.TableData`` and ``data.blockTreeData``.
    Normalise those shapes only; unknown wrappers stay raw-only and never gain
    invented columns.
    """
    rows, _ = _payload_rows(payload)
    if not rows:
        container = _as_dict(_as_dict(payload).get("data"))
        tree = container.get("blockTreeData")
        if isinstance(tree, list):
            rows = [dict(x) for x in tree if isinstance(x, Mapping)]
    out: list[dict[str, Any]] = []
    for row in rows:
        period = _row_value(row, "periodname", "periodName", "periodid", "Periodid", "point", "piontid")
        item = {
            "record_key": record_key(
                source_api, business_date or "", str(period or ""),
                str(_row_value(row, "pdate", "type", "mold", "dataType") or ""),
                str(_row_value(row, "id", "label", "NUM", "num") or ""),
                str(_row_value(row, "subjectId") or ""),
                str(_row_value(row, "subjectName") or ""),
                str(_row_value(row, "valuetime1") or ""),
                str(_row_value(row, "valuetime2") or ""),
            ),
            "business_date": _row_value(row, "pdate") or business_date,
            "period_name": period,
            "mold": _row_value(row, "mold"),
            "data_type": _row_value(row, "dataType"),
            "power": parse_number(_row_value(row, "power", "value")),
            "label": _row_value(row, "label"),
            "tree_id": _row_value(row, "id"),
            "source_api": source_api,
            "extra_json": json.dumps(_clean(row), ensure_ascii=False),
        }
        out.append({k: v for k, v in item.items() if v is not None})
    return out


PARSER_REGISTRY: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "unit": parse_unit, "constraint": parse_constraint, "event": parse_event,
    "curve": parse_curve, "stat": parse_stat, "contract": parse_contract,
    "disclosure": parse_disclosure,
}


# [AUX-V1-r1] SourceSpec is a request contract, not merely a parser registry.
QCTC_CONTEXT_PAGE = "https://pmos.sd.sgcc.com.cn:18080/home"
ZCQ_BASE = "https://pmos.sd.sgcc.com.cn:18080"
ZCQ_ROUTE_PAGES = {
    # [AUX-V1-r10-route1] These are separate legacy document contexts observed
    # in HAR16/17/18.  The browser target must visit the matching appkey page
    # before issuing its same-origin POST; reusing /home produces 503/empty
    # responses even though the endpoint itself is valid.
    "net_contract_day": ZCQ_BASE + "/zcq/JyjgZcqXxpl.do?appkey=18",
    "generation_contract_limit": ZCQ_BASE + "/zcq/jysbys/fdczxsbedcx.do?appkey=93",
    "generation_hourly_net": ZCQ_BASE + "/zcq/fdaxsjhyxc.do?appkey=94",
    "unit_month_limit": ZCQ_BASE + "/zcq/jysbys/ydfdcsxyhcx.do?appkey=81",
    # [AUX-V1-r13] 电量信息详情查询（本主体合约成交曲线）：页面 HTML 实测携带
    # _csrf meta（r12 explore bodies/0152、0149）。
    "zcq_contract_curve24": ZCQ_BASE + "/zcq/dlxxxqcx/dlxxxqYhCx.do?appkey=21",
    "zcq_contract_curve96": ZCQ_BASE + "/zcq/dlxxxqcx96/dlxxxqYhCx.do?appkey=15",
}

# [AUX-V1-r11] HAR20-confirmed QCTC information-disclosure contract.
# Real gateway path is
#   /qctc/qctc_pm_trade_outside/informationDisclosure/<Module>/<Method>?pdate=...
# and each request carries an ``X-Web-Path`` page header.  The previously
# inferred ``qctc_pm_trade_inside/trade/daRqxxpl`` paths are not routed at all
# and are kept only as disabled historical entries.
QCTC_FORECAST_PAGE = ZCQ_BASE + "/qctc-trade/informationDisclosure/forecast10424"
QCTC_ACTUAL_TMP_PAGE = ZCQ_BASE + "/qctc-trade/informationDisclosure/actualTemporary10425"
QCTC_ACTUAL_PAGE = ZCQ_BASE + "/qctc-trade/informationDisclosure/actual10426"
QCTC_BOUNDARY_PAGE = ZCQ_BASE + "/qctc-trade/informationDisclosure/forecastBoundary10427"
QCTC_DISCLOSURE_BASE = "/qctc/qctc_pm_trade_outside/informationDisclosure"


def _pdate_params(business_date: str | None) -> dict[str, Any]:
    return {"pdate": business_date or ""}


def _pdate_versions_params(business_date: str | None) -> dict[str, Any]:
    """[AUX-V1-r11] ForecastData methods require an explicit empty versions."""
    return {"pdate": business_date or "", "versions": ""}


def _unit_data_params(business_date: str | None) -> dict[str, Any]:
    # [AUX-V1-r2] mirrors unitTotalPage's initial pageSize=100/start=0.
    return {"pdate": business_date or "", "unitname": "", "qyid": "", "type": "", "gengroupid": "", "start": 0, "length": 100}


def _contract_day_params(business_date: str | None) -> dict[str, Any]:
    # [AUX-V1-r10-route1] HAR16 contract is POST/query, not the inferred
    # qctc JSON contract.  Keep the observed type/sort names verbatim.
    return {"startTime": business_date or "", "endTime": business_date or "", "type": 0, "sort": "bdsj"}


def _month_params(business_date: str | None) -> dict[str, Any]:
    """[AUX-V1-r3] Match the confirmed ZCQ monthly query contract."""
    month = str(business_date or "")[:7]
    return {"dmonth": month, "smonth": month}


def _legacy_contract_schema_params(business_date: str | None) -> dict[str, Any]:
    """[AUX-V1-r3] Fetch only the confirmed column schema; no fake faids."""
    return {"dmonth": str(business_date or "")[:7]}


def _constraint_params(business_date: str | None) -> dict[str, Any]:
    # [AUX-V1-r2] unitid is a required dependency; collect() refuses to
    # construct/send a blank-id request.  The builder remains explicit so the
    # request contract cannot silently grow a fan-out fallback.
    return {"pdate": business_date or ""}


def _unknown_params(business_date: str | None) -> dict[str, Any]:
    return {"pdate": business_date or ""}


# [AUX-V1-r13] r12 explore 实测（bodies/0154、0156）：
#   POST /zcq/dlxxxqcx[/96]/dlxxxqYhCx.do  query: method=get24|96CjTableData&
#   dyid=<unitid>&userProp=1&sDate=<月首>&eDate=<月末>&jylx=ALL
#   DataTables 侧 draw=1&start=0&length=50（Spring @RequestParam 同时绑定
#   query 与表单，AUX 沿用 zcq route1 的 params_in_query 全 query 惯例）。
# 响应：{"recordsFiltered":744,"recordsTotal":744,"draw":1,"data":[{pdate:"20261001",point,cjdl,cjjj,...}]}
_CONTRACT_CURVE_METHODS = {"zcq_contract_curve24": "get24CjTableData", "zcq_contract_curve96": "get96CjTableData"}
_ZCQ_CSRF_SOURCES = {"unit_month_limit", "net_contract_day", "zcq_contract_curve24", "zcq_contract_curve96"}
_ROW_DATE_CONTRACT_SOURCES = {"zcq_contract_curve24", "zcq_contract_curve96"}


def _contract_curve_params(business_date: str | None, unitid: str) -> dict[str, Any]:
    """[AUX-V1-r13] sDate/eDate 覆盖 business_date 所在整月（DataTables 分页由
    collect() 的 offset 模式推进 start）。"""
    day = str(business_date or "")[:10]
    month = day[:7]
    s_date = e_date = day
    if len(month) == 7 and month[4] == "-":
        try:
            last = calendar.monthrange(int(month[:4]), int(month[5:7]))[1]
            s_date, e_date = f"{month}-01", f"{month}-{last:02d}"
        except ValueError:
            pass
    return {"dyid": str(unitid or "").strip(), "userProp": "1", "sDate": s_date, "eDate": e_date, "jylx": "ALL", "draw": 1, "start": 0, "length": 50}


SOURCE_REGISTRY: dict[str, SourceSpec] = {
    # [AUX-V1-r11b] The qctc_pm_trade_inside entries below were reverse
    # engineered from QCTC frontend JS strings.  18810 HAR urls contain **zero**
    # real requests to those paths and every live attempt answered HTTP 503
    # Whitelabel, so they are downgraded to UNVERIFIED: all-designed reports
    # them as SKIPPED_NOT_READY instead of sending doomed requests.  The real
    # module lives at qctc_pm_trade_outside/informationDisclosure (see dcst_*).
    # [AUX-V1-r2] HAR_FRONTEND_JS: GET + query params.  The frontend starts
    # currentPage=0/pageSize=100; start/length are the wire contract.
    "unit_master": SourceSpec("unit_master", "unit", "GET", "/qctc/qctc_pm_trade_inside/DaUnitParamQuery/getDataList", QCTC_CONTEXT_PAGE, "snapshot", "unit", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="UNVERIFIED", frontend_path="/qctc/qctc_pm_trade_inside/DaUnitParamQuery/getDataList", param_contract=("pdate", "unitname", "qyid", "type", "gengroupid", "start", "length"), pagination_mode="offset", max_pages=200, max_rows=100000),
    "unit_type": SourceSpec("unit_type", "unit", "GET", "/qctc/qctc_pm_trade_inside/DaUnitParamQuery/getTypeList", QCTC_CONTEXT_PAGE, "snapshot", "unit", None, enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc/qctc_pm_trade_inside/DaUnitParamQuery/getTypeList", param_contract=("pdate",)),
    "unit_gengroup": SourceSpec("unit_gengroup", "unit", "GET", "/qctc/qctc_pm_trade_inside/DaUnitParamQuery/getGengroupList", QCTC_CONTEXT_PAGE, "snapshot", "unit", None, enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc/qctc_pm_trade_inside/DaUnitParamQuery/getGengroupList", param_contract=("pdate",)),
    # [AUX-V1-r3] HAR_NETWORK route that returned visible unit entities.
    "unit_info": SourceSpec("unit_info", "unit", "GET", "/qctc/qctc_pm_trade_outside/trade/DaJyjgfbPlantQuery/getUnitInfo", QCTC_CONTEXT_PAGE, "snapshot", "unit", "epf_pmos_aux_records", evidence_level="HAR_NETWORK", frontend_path="/qctc/qctc_pm_trade_outside/trade/DaJyjgfbPlantQuery/getUnitInfo", param_contract=("pdate",)),
    # [AUX-V1-r1] HAR_NETWORK: outside contract is a GET through the qctc gateway.
    "unit_constraint": SourceSpec("unit_constraint", "constraint", "GET", "/qctc_pm_trade_outside/DaJysbPlant/getUnitConstraint", QCTC_CONTEXT_PAGE, "event", "constraint", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="HAR_NETWORK", frontend_path="/qctc_pm_trade_outside/DaJysbPlant/getUnitConstraint", param_contract=("pdate", "unitid")),
    "unit_component": SourceSpec("unit_component", "constraint", "GET", "/qctc_pm_trade_outside/JJCQDaJysbPlantCommon/initComponent", QCTC_CONTEXT_PAGE, "event", "constraint", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="HAR_NETWORK", frontend_path="/qctc_pm_trade_outside/JJCQDaJysbPlantCommon/initComponent", param_contract=("pdate", "unitid")),
    "unit_constraint_jjcq": SourceSpec("unit_constraint_jjcq", "constraint", "GET", "/qctc_pm_trade_outside/JJCQDaJysbPlantCommon/getUnitConstraint", QCTC_CONTEXT_PAGE, "event", "constraint", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="HAR_NETWORK", frontend_path="/qctc_pm_trade_outside/JJCQDaJysbPlantCommon/getUnitConstraint", param_contract=("pdate", "unitid")),
    "special_unit_tag": SourceSpec("special_unit_tag", "event", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getTsjzTableAndText", QCTC_CONTEXT_PAGE, "daily", "event", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getTsjzTableAndText", param_contract=("pdate",)),
    "transmission_maintenance": SourceSpec("transmission_maintenance", "event", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getSbdjxTableAndText", QCTC_CONTEXT_PAGE, "daily", "event", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getSbdjxTableAndText", param_contract=("pdate",)),
    "reserve_security": SourceSpec("reserve_security", "event", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getDwbyTableAndText", QCTC_CONTEXT_PAGE, "daily", "event", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getDwbyTableAndText", param_contract=("pdate",)),
    # [AUX-V1-r11m] 检修计划：路径取自前端 JS 模块导出（pmos...12.har 模块 "32ef"），
    # 并按实测网关规律补 `/qctc` 前缀（旧值缺前缀 → 打根路径 → 503）。方法与 JS 一致。
    # ⚠️ 仍需真机验证：HAR 里对这些路径的真实请求为 0，目前只有 JS 字面量证据。
    "maintenance_plan": SourceSpec("maintenance_plan", "event", "POST", "/qctc/qctc-pm-trade-zcq-out-sxed/unitMaintenancePlanOutQuery/getUnitMaintenanceDetail", QCTC_CONTEXT_PAGE, "event", "event", "epf_pmos_aux_records", enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc/qctc-pm-trade-zcq-out-sxed/unitMaintenancePlanOutQuery/getUnitMaintenanceDetail", param_contract=("dmonth", "smonth", "currentPage", "pageSize", "type", "spid", "level", "draw", "start")),
    "maintenance_init": SourceSpec("maintenance_init", "event", "GET", "/qctc/qctc-pm-trade-zcq-out-sxed/unitMaintenancePlanOutQuery/init", QCTC_CONTEXT_PAGE, "event", "event", "epf_pmos_aux_records", enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc/qctc-pm-trade-zcq-out-sxed/unitMaintenancePlanOutQuery/init", param_contract=("dmonth", "smonth", "currentPage", "pageSize", "type", "spid", "level", "draw", "start")),
    "maintenance_tree": SourceSpec("maintenance_tree", "event", "GET", "/qctc/qctc-pm-trade-zcq-out-sxed/unitMaintenancePlanUpdate/getTree", QCTC_CONTEXT_PAGE, "event", "event", "epf_pmos_aux_records", enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc/qctc-pm-trade-zcq-out-sxed/unitMaintenancePlanUpdate/getTree", param_contract=("dmonth",)),
    # getFireTree＝火电机组树（用户要的"火电机组"），来自 in-sxed 变体。
    "maintenance_fire_tree": SourceSpec("maintenance_fire_tree", "unit", "GET", "/qctc/qctc-pm-trade-zcq-in-sxed/unitMaintenancePlanUpdate/getFireTree", QCTC_CONTEXT_PAGE, "snapshot", "unit", "epf_pmos_aux_records", enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc/qctc-pm-trade-zcq-in-sxed/unitMaintenancePlanUpdate/getFireTree", param_contract=("dmonth",)),
    "run_line": SourceSpec("run_line", "curve", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getRunLine", QCTC_CONTEXT_PAGE, "15min", "curve", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getRunLine", param_contract=("pdate",)),
    "debug_line": SourceSpec("debug_line", "curve", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getDegLine", QCTC_CONTEXT_PAGE, "15min", "curve", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getDegLine", param_contract=("pdate",)),
    "fh_char_raw": SourceSpec("fh_char_raw", "curve", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getFhChar", QCTC_CONTEXT_PAGE, "15min", "curve", None, enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getFhChar", param_contract=("pdate", "type")),
    "zd_llx_raw": SourceSpec("zd_llx_raw", "curve", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getZdLlx", QCTC_CONTEXT_PAGE, "daily", "curve", None, enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getZdLlx", param_contract=("pdate",)),
    "max_min_raw": SourceSpec("max_min_raw", "curve", "GET", "/qctc_pm_trade_inside/trade/daRqxxpl/getMaxMinZdSc", QCTC_CONTEXT_PAGE, "daily", "curve", None, enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/daRqxxpl/getMaxMinZdSc", param_contract=("pdate",)),
    "unit_count_stat": SourceSpec("unit_count_stat", "stat", "GET", "/qctc_pm_trade_inside/trade/marketDetailsQuery/zcqEnergyPriceQuery/getTableData", QCTC_CONTEXT_PAGE, "daily", "stat", "epf_pmos_aux_records", enabled_by_default=False, raw_only=True, evidence_level="UNVERIFIED", frontend_path="/qctc_pm_trade_inside/trade/marketDetailsQuery/zcqEnergyPriceQuery/getTableData", param_contract=("ztType", "tjInfo", "tjType", "rqType", "pdate")),
    # [AUX-V1-r10-route1] HAR16 appkey=18 legacy route.
    # [AUX-V1-r13] 启用默认采集（中长期合约成交行，火电合约占比的全省面）。
    "net_contract_day": SourceSpec("net_contract_day", "contract", "POST", "/zcq/JyjgZcqXxpl.do?method=getTableDate", ZCQ_ROUTE_PAGES["net_contract_day"], "daily", "contract", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="HAR_NETWORK", frontend_path="/zcq/JyjgZcqXxpl.do?method=getTableDate", param_contract=("startTime", "endTime", "type", "sort"), pagination_mode="page", params_in_query=True),
    # [AUX-V1-r3] Exact legacy ZCQ route/method from pmos...16.har. This
    # endpoint is schema/raw-only; detail rows require a separately observed
    # faids dependency and are not guessed here.
    # [AUX-V1-r10-route1] HAR16/17 appkey=93 and appkey=94 legacy routes.
    "generation_contract_limit": SourceSpec("generation_contract_limit", "contract", "POST", "/zcq/jysbys/fdczxsbedcx.do?method=getTableRows", ZCQ_ROUTE_PAGES["generation_contract_limit"], "monthly", "contract", "epf_pmos_aux_records", enabled_by_default=False, raw_only=True, evidence_level="HAR_NETWORK", frontend_path="/zcq/jysbys/fdczxsbedcx.do?method=getTableRows", param_contract=("dmonth",), params_in_query=True),
    "generation_hourly_net": SourceSpec("generation_hourly_net", "contract", "POST", "/zcq/fdaxsjhyxc.do?method=getTableDate", ZCQ_ROUTE_PAGES["generation_hourly_net"], "15min", "contract", "epf_pmos_aux_records", enabled_by_default=False, raw_only=True, evidence_level="HAR_NETWORK", frontend_path="/zcq/fdaxsjhyxc.do?method=getTableDate", param_contract=("unitid", "time", "isYd"), params_in_query=True),
    # [AUX-V1-r3] Confirmed response contains jzdlzb (页面列名：机制电量比例).
    # [AUX-V1-r10-route1] HAR18 appkey=81 document + CSRF route.
    "unit_month_limit": SourceSpec("unit_month_limit", "contract", "POST", "/zcq/jysbys/ydfdcsxyhcx.do?method=getarcdetailNxdcFd", ZCQ_ROUTE_PAGES["unit_month_limit"], "monthly", "contract", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="HAR_NETWORK", frontend_path="/zcq/jysbys/ydfdcsxyhcx.do?method=getarcdetailNxdcFd", param_contract=("dmonth", "smonth"), params_in_query=True),
    # ---- [AUX-V1-r11] QCTC informationDisclosure (HAR20 HTTP 200 evidence) ----
    # Every entry below was observed live in pmos...20.har with a real status
    # code.  Method names are frontend string concatenations and are copied
    # verbatim on purpose.  ForecastData methods take versions="", RealityTmpData
    # methods do not.
    "dcst_forecast_load": SourceSpec("dcst_forecast_load", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getLoadData", QCTC_FORECAST_PAGE, "15min", "disclosure", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="HAR_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getLoadData", param_contract=("pdate", "versions")),
    "dcst_forecast_tieline": SourceSpec("dcst_forecast_tieline", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getTieLineData", QCTC_FORECAST_PAGE, "15min", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="HAR_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getTieLineData", param_contract=("pdate", "versions")),
    "dcst_forecast_unit_overhaul": SourceSpec("dcst_forecast_unit_overhaul", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getUnitOverhaulData", QCTC_FORECAST_PAGE, "daily", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="HAR_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getUnitOverhaulData", param_contract=("pdate", "versions")),
    "dcst_tmp_table_cols": SourceSpec("dcst_tmp_table_cols", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/RealityTmpData/getTableCols", QCTC_ACTUAL_TMP_PAGE, "snapshot", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, raw_only=True, evidence_level="HAR_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/RealityTmpData/getTableCols", param_contract=("pdate",)),
    "dcst_tmp_load": SourceSpec("dcst_tmp_load", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/RealityTmpData/getLoadData", QCTC_ACTUAL_TMP_PAGE, "15min", "disclosure", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="HAR_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/RealityTmpData/getLoadData", param_contract=("pdate",)),
    "dcst_tmp_update_time": SourceSpec("dcst_tmp_update_time", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/RealityTmpData/getUpdateTime", QCTC_ACTUAL_TMP_PAGE, "snapshot", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, raw_only=True, evidence_level="HAR_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/RealityTmpData/getUpdateTime", param_contract=("pdate",)),
    # [AUX-V1-r13] 原 5 条虚构 RealityTmpData/* 路径（真机证实 RealityTmpData 仅有
    # getLoadData/getTableCols/getUpdateTime）改指 r12 explore 实测 200 的
    # ForecastData/* 端点；page 与参数契约同步切换（pdate&versions）。
    "dcst_tmp_block": SourceSpec("dcst_tmp_block", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getBlockData", QCTC_FORECAST_PAGE, "daily", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="EXPLORE_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getBlockData", param_contract=("pdate", "versions")),
    "dcst_tmp_spare": SourceSpec("dcst_tmp_spare", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getSpareData", QCTC_FORECAST_PAGE, "15min", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="EXPLORE_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getSpareData", param_contract=("pdate", "versions")),
    # [AUX-V1-r13] 与 dcst_forecast_unit_overhaul 同一端点，保留为禁用别名。
    "dcst_tmp_unit_overhaul": SourceSpec("dcst_tmp_unit_overhaul", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getUnitOverhaulData", QCTC_FORECAST_PAGE, "daily", "disclosure", "epf_pmos_aux_records", enabled_by_default=False, evidence_level="EXPLORE_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getUnitOverhaulData", param_contract=("pdate", "versions")),
    "dcst_tmp_open_stop": SourceSpec("dcst_tmp_open_stop", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getOpenAndStopUnitData", QCTC_FORECAST_PAGE, "daily", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="EXPLORE_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getOpenAndStopUnitData", param_contract=("pdate", "versions")),
    "dcst_tmp_trans_overhaul": SourceSpec("dcst_tmp_trans_overhaul", "disclosure", "GET", QCTC_DISCLOSURE_BASE + "/ForecastData/getPowerTransmissionAndTransformationOverhaulData", QCTC_FORECAST_PAGE, "daily", "disclosure", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="EXPLORE_NETWORK", frontend_path=QCTC_DISCLOSURE_BASE + "/ForecastData/getPowerTransmissionAndTransformationOverhaulData", param_contract=("pdate", "versions")),
    # ---- [AUX-V1-r13] 火电合约占比：本主体合约成交曲线（r12 explore 真机 200 证据）----
    # 页面 HTML 携带 _csrf meta；行结构 {id,pdate:"20261001",point,cjdl,cjjj,time,detail}，
    # 行内 pdate 进 record_key（use_row_date），unitid 由 --unitid/config 注入（不 fan-out）。
    "zcq_contract_curve24": SourceSpec("zcq_contract_curve24", "contract", "POST", "/zcq/dlxxxqcx/dlxxxqYhCx.do?method=get24CjTableData", ZCQ_ROUTE_PAGES["zcq_contract_curve24"], "hourly", "contract", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="EXPLORE_NETWORK", frontend_path="/zcq/dlxxxqcx/dlxxxqYhCx.do?method=get24CjTableData", param_contract=("dyid", "userProp", "sDate", "eDate", "jylx", "draw", "start", "length"), pagination_mode="offset", params_in_query=True),
    "zcq_contract_curve96": SourceSpec("zcq_contract_curve96", "contract", "POST", "/zcq/dlxxxqcx96/dlxxxqYhCx.do?method=get96CjTableData", ZCQ_ROUTE_PAGES["zcq_contract_curve96"], "15min", "contract", "epf_pmos_aux_records", enabled_by_default=True, evidence_level="EXPLORE_NETWORK", frontend_path="/zcq/dlxxxqcx96/dlxxxqYhCx.do?method=get96CjTableData", param_contract=("dyid", "userProp", "sDate", "eDate", "jylx", "draw", "start", "length"), pagination_mode="offset", params_in_query=True),
}

PARAM_BUILDERS: dict[str, Callable[[str | None], dict[str, Any]]] = {
    "unit_master": _unit_data_params,
    "unit_info": _pdate_params,
    "unit_type": _pdate_params,
    "unit_gengroup": _pdate_params,
    "unit_constraint": _constraint_params,
    "unit_component": _constraint_params,
    "unit_constraint_jjcq": _constraint_params,
    "special_unit_tag": _pdate_params,
    "transmission_maintenance": _pdate_params,
    "reserve_security": _pdate_params,
    "run_line": _pdate_params,
    "debug_line": _pdate_params,
    "fh_char_raw": _pdate_params,
    "zd_llx_raw": _pdate_params,
    "max_min_raw": _pdate_params,
    "unit_count_stat": lambda d: {"ztType": 1, "tjInfo": 1, "tjType": "t", "rqType": 1, "pdate": d or ""},
    "net_contract_day": _contract_day_params,
    "generation_contract_limit": _legacy_contract_schema_params,
    "unit_month_limit": _month_params,
    "maintenance_plan": _unknown_params,
    "maintenance_init": _unknown_params,
    "maintenance_tree": _unknown_params,
    "maintenance_fire_tree": _unknown_params,
    # [AUX-V1-r11] informationDisclosure parameter contracts (HAR20 verbatim).
    "dcst_forecast_load": _pdate_versions_params,
    "dcst_forecast_tieline": _pdate_versions_params,
    "dcst_forecast_unit_overhaul": _pdate_versions_params,
    "dcst_tmp_table_cols": _pdate_params,
    "dcst_tmp_load": _pdate_params,
    "dcst_tmp_update_time": _pdate_params,
    "dcst_tmp_block": _pdate_versions_params,
    "dcst_tmp_spare": _pdate_versions_params,
    "dcst_tmp_unit_overhaul": _pdate_versions_params,
    "dcst_tmp_open_stop": _pdate_versions_params,
    "dcst_tmp_trans_overhaul": _pdate_versions_params,
}


class PmosDisclosureAuxCrawler(PmosCrawler):
    """[AUX-V1] PmosCrawler transport adapter; no main business writer reuse."""

    def __init__(self, *args: Any, output_dir: str | Path | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.output_dir = Path(output_dir or "output_aux")
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._transport_failure_count = 0
        # [AUX-V1-r13] r11f 判据 N3 计数器必须在构造时存在，否则首个 transport
        # 失败在 except 分支里 self._browser_lost_count += 1 直接 AttributeError。
        self._browser_lost_count = 0
        self._zcq_csrf_cache: dict[str, dict[str, str]] = {}

    def _zcq_csrf_headers(self, page_url: str) -> dict[str, str]:
        """[AUX-V1-r8/r13] Read the target appkey page via real document
        navigation (never fetch) and cache per page; token stays in memory."""
        cached = self._zcq_csrf_cache.get(page_url)
        if cached is not None:
            return dict(cached)
        page = self._legacy_zcq_document_get(page_url)
        if int(getattr(page, "status_code", 0) or 0) != 200:
            raise RuntimeError(
                "LEGACY_ZCQ_PAGE_UNAVAILABLE "
                f"url={page_url} "
                f"http_status={getattr(page, 'status_code', None)}"
            )
        source = str(getattr(page, "text", "") or "")
        meta: dict[str, dict[str, str]] = {}
        for tag in re.findall(r"<meta\b[^>]*>", source, flags=re.IGNORECASE):
            attrs = {
                key.lower(): value
                for key, _quote, value in re.findall(
                    r"([\w-]+)\s*=\s*(['\"])(.*?)\2", tag, flags=re.IGNORECASE
                )
            }
            if attrs.get("name") in {"_csrf", "_csrf_header"}:
                meta[attrs["name"]] = attrs
        token = str(meta.get("_csrf", {}).get("content") or "").strip()
        header = str(meta.get("_csrf_header", {}).get("content") or "X-CSRF-TOKEN").strip()
        if not token or not re.fullmatch(r"[A-Za-z0-9_-]+", header):
            raise RuntimeError(
                "LEGACY_ZCQ_CSRF_MISSING "
                f"page={page_url}"
            )
        # The token is held only in memory; never include its value in logs/raw metadata.
        self._zcq_csrf_cache[page_url] = {header: token, "Referer": page_url}
        return dict(self._zcq_csrf_cache[page_url])

    def _navigate_for_document(self, cdp: _CdpClient, page_url: str, timeout: float = 45.0) -> dict[str, Any]:
        """[AUX-V1-r11b] Navigate until the *document* really is ``page_url``.

        ``crawl._navigate_for_fetch`` only proves same origin, which is enough
        for a same-origin fetch but not for reading a page-specific DOM such as
        the appkey=81 CSRF meta.  It returns as soon as any same-origin document
        reports ``complete``, so the caller can parse whatever page was loaded
        before (observed: 560 KB ``/zcq/main/index.do`` DOM instead of the 31 KB
        appkey=81 page).  Poll ``location.href`` until path and query match.
        """
        target = urlsplit(page_url)
        cdp.call("Page.navigate", {"url": page_url}, timeout=10)
        deadline = time.monotonic() + timeout
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                evaluated = cdp.call(
                    "Runtime.evaluate",
                    {
                        "expression": "({url: location.href, ready: document.readyState})",
                        "returnByValue": True,
                    },
                    timeout=10,
                )
                value = ((evaluated.get("result") or {}).get("value") or {})
                href = str(value.get("url") or "")
                last = {"url": href, "ready": str(value.get("ready") or "")}
                now = urlsplit(href)
                if (
                    now.netloc == target.netloc
                    and now.path == target.path
                    and str(value.get("ready") or "") == "complete"
                ):
                    return last
            except Exception:
                pass
            time.sleep(0.5)
        logger.warning(
            "AUX document navigation did not confirm target target=%s last_url=%s",
            target.path[:120], str(last.get("url") or "")[:120],
        )
        return last

    def _legacy_zcq_document_get(self, page_url: str) -> requests.Response:
        """[AUX-V1-r8] Navigate the authenticated tab so PMOS serves document HTML."""
        if not self.browser_debug_port:
            raise RuntimeError("LEGACY_ZCQ_PAGE_UNAVAILABLE browser_debug_port_missing")
        page_target = _get_pmos_page(self.browser_debug_port, prefer_qctc=True)
        cdp = _CdpClient(page_target["webSocketDebuggerUrl"])
        try:
            cdp.call("Page.enable")
            # [AUX-V1-r11b] Same-origin is not enough: this page's DOM carries
            # the CSRF meta, so wait until location.href actually matches.
            state = self._navigate_for_document(cdp, page_url)
            final_url = str(state.get("url") or "")
            if not _same_origin(final_url, page_url):
                raise RuntimeError(
                    "LEGACY_ZCQ_PAGE_REDIRECTED "
                    f"expected_origin=pmos:18080 final_url={final_url[:180]}"
                )
            evaluated = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": """(() => {
                      const nav = performance.getEntriesByType('navigation')[0];
                      return {
                        url: location.href,
                        ready: document.readyState,
                        content_type: document.contentType || '',
                        status: Number(nav && nav.responseStatus || 0),
                        html: document.documentElement ? document.documentElement.outerHTML : ''
                      };
                    })()""",
                    "returnByValue": True,
                },
                timeout=15,
            )
            value = ((evaluated.get("result") or {}).get("value") or {})
            if not isinstance(value, dict):
                raise RuntimeError("LEGACY_ZCQ_PAGE_READ_FAILED invalid_document_result")
            response = requests.Response()
            response.status_code = int(value.get("status") or 0)
            response.url = str(value.get("url") or final_url)
            response._content = str(value.get("html") or "").encode("utf-8", errors="replace")
            response.encoding = "utf-8"
            response.headers["content-type"] = str(value.get("content_type") or "")
            # [AUX-V1-r13b] Verify CSRF meta is present in the navigated DOM;
            # if missing, the tab likely landed on a portal/dashboard instead of
            # the intended appkey page — log detailed diagnostics before returning.
            html_text = str(value.get("html") or "")
            csrf_present = '_csrf' in html_text and 'meta' in html_text[:2048].lower()
            logger.info(
                "AUX appkey=81 document navigation status=%s content_type=%s body_len=%s csrf_meta=%s",
                response.status_code, response.headers["content-type"], len(response.content), csrf_present,
            )
            if not csrf_present:
                logger.warning(
                    "LEGACY_ZCQ_CSRF_META_MISSING url=%s status=%s body_len=%s ready=%s content_type=%s",
                    response.url, response.status_code, len(response.content),
                    str(value.get("ready") or ""), str(value.get("content_type") or ""),
                )
            return response
        finally:
            cdp.close()

    def _navigate_legacy_route_page(self, spec: SourceSpec) -> None:
        """[AUX-V1-r10-route1] Enter the HAR-observed legacy page context.

        ``PmosCrawler._browser_fetch`` intentionally reuses any same-origin
        document.  That is correct for modern QCTC APIs, but legacy ZCQ pages
        expose different appkey-specific server contexts.  Navigate the
        authenticated target explicitly before the AJAX POST so one AUX run
        can switch routes without changing the shared browser implementation.
        """
        page_url = str(spec.page_url or "")
        if not page_url or not page_url.lower().startswith(ZCQ_BASE.lower() + "/zcq/"):
            return
        if not self.browser_debug_port:
            raise RuntimeError(f"LEGACY_ZCQ_ROUTE_UNAVAILABLE source={spec.name} browser_debug_port_missing")
        page_target = _get_pmos_page(self.browser_debug_port, prefer_qctc=True)
        cdp = _CdpClient(page_target["webSocketDebuggerUrl"])
        try:
            cdp.call("Page.enable")
            state = _navigate_for_fetch(cdp, page_url)
            final_url = str(state.get("url") or "")
            if not _same_origin(final_url, page_url):
                raise RuntimeError(
                    "LEGACY_ZCQ_ROUTE_REDIRECTED "
                    f"source={spec.name} final_url={final_url[:180]}"
                )
            logger.info("AUX legacy route ready source=%s page=%s", spec.name, page_url)
        finally:
            cdp.close()

    def _browser_route_request(self, spec: SourceSpec, method: str, url: str, **kwargs: Any) -> requests.Response:
        """[AUX-V1-r10-route1] Browser request with source-specific route prep."""
        # appkey=81 navigation is already performed by the CSRF document
        # contract; navigating a second time would risk losing its token.
        if spec.name != "unit_month_limit":
            self._navigate_legacy_route_page(spec)
        return self._browser_req(method, url, page_url=spec.page_url, **kwargs)

    def _source_url(self, spec: SourceSpec) -> str:
        # [AUX-V1-r1] Resolve frontend service path to the single HAR-supported
        # gateway URL; never probe both prefixed and unprefixed variants.
        path = spec.request_path
        if path.startswith("/qctc/") or path.startswith("/jysbys") or path.startswith("/fdax") or path.startswith("/ydfd"):
            return "https://pmos.sd.sgcc.com.cn:18080" + path
        if path.startswith("/qctc_"):
            return "https://pmos.sd.sgcc.com.cn:18080/qctc" + path
        return "https://pmos.sd.sgcc.com.cn:18080" + path

    def _aux_request(self, spec: SourceSpec, params: Mapping[str, Any]) -> requests.Response:
        """[AUX-V1-r5, AUX-V1-r10-diag2] Use the 96 crawler transport for QCTC APIs.

        The 96 production path sends QCTC requests from the authenticated browser
        same-origin page.  A Python ``requests`` session carries portal cookies but
        cannot carry the QCTC ``sessionStorage`` context, so it predictably receives
        HTTP 401.  AUX remains isolated, but adopts the same browser-primary
        transport for modern ``/qctc/`` APIs; legacy ``/zcq`` keeps its existing
        CSRF/Python-first contract.
        """
        method = spec.method.upper()
        url = self._source_url(spec)
        started = time.monotonic()
        headers: dict[str, str] = {}
        if spec.name in _ZCQ_CSRF_SOURCES:
            headers = self._zcq_csrf_headers(spec.page_url)
        elif "/informationdisclosure/" in urlsplit(url).path.lower():
            # [AUX-V1-r11] HAR20 shows every informationDisclosure request
            # carrying the browsing page path; without it the gateway cannot
            # resolve the module's page context and answers 503/504.
            headers = {
                "Accept": "application/json, text/plain, */*",
                "X-Web-Path": urlsplit(spec.page_url).path,
            }
        browser_kwargs = {"params": dict(params)} if method == "GET" or spec.params_in_query else {"data": dict(params)}
        if headers:
            browser_kwargs["headers"] = headers
        # [AUX-V1-r10-diag2] Match 96 _qctc_get: browser same-origin is primary,
        # not merely a fallback after Python has already been rejected with 401.
        qctc_browser_primary = "/qctc/" in urlsplit(url).path.lower()
        transport = "browser_primary" if qctc_browser_primary else "python"
        logger.info("AUX request begin source=%s method=%s api=%s params_keys=%s transport=%s",
                    spec.name, method, spec.request_path, sorted(str(k) for k in params), transport)
        if qctc_browser_primary:
            try:
                response = self._browser_route_request(spec, method, url, **browser_kwargs)
                # [AUX-V1-r11b] 504 is an nginx gateway timeout, not an auth or
                # permission rejection, and HAR20 shows it is intermittent.
                # Retry twice with backoff before reporting a real failure.
                attempts = 0
                while getattr(response, "status_code", None) == 504 and attempts < 2:
                    attempts += 1
                    time.sleep(2 * attempts)
                    logger.warning("AUX request retry source=%s status=504 attempt=%s", spec.name, attempts)
                    response = self._browser_route_request(spec, method, url, **browser_kwargs)
            except Exception as exc:
                logger.error("AUX request browser_primary_error source=%s error_type=%s elapsed_sec=%.1f error=%s",
                             spec.name, type(exc).__name__, time.monotonic() - started, _safe_diag_text(exc))
                raise
            logger.info("AUX request browser_primary_result source=%s status=%s content_type=%s body_bytes=%s elapsed_sec=%.1f",
                        spec.name, getattr(response, "status_code", None),
                        getattr(response, "headers", {}).get("content-type", ""),
                        len(getattr(response, "content", b"") or b""), time.monotonic() - started)
            return response

        kwargs: dict[str, Any] = {"timeout": 15, "verify": self._verify}
        if headers:
            kwargs["headers"] = headers
        if method == "GET" or spec.params_in_query:
            kwargs["params"] = dict(params)
        else:
            kwargs["data"] = dict(params)
        try:
            response = self.session.request(method, url, **kwargs)
        except Exception as exc:
            logger.warning("AUX request transport_error source=%s error_type=%s elapsed_sec=%.1f error=%s; fallback=browser",
                           spec.name, type(exc).__name__, time.monotonic() - started, _safe_diag_text(exc))
            browser_kwargs = {"params": dict(params)} if method == "GET" or spec.params_in_query else {"data": dict(params)}
            if headers:
                browser_kwargs["headers"] = headers
            fallback_started = time.monotonic()
            try:
                response = self._browser_route_request(spec, method, url, **browser_kwargs)
            except Exception as fallback_exc:
                logger.error("AUX request browser_fallback_error source=%s error_type=%s elapsed_sec=%.1f error=%s",
                             spec.name, type(fallback_exc).__name__, time.monotonic() - fallback_started,
                             _safe_diag_text(fallback_exc))
                raise
            logger.info("AUX request browser_fallback_result source=%s status=%s content_type=%s body_bytes=%s elapsed_sec=%.1f",
                        spec.name, getattr(response, "status_code", None),
                        getattr(response, "headers", {}).get("content-type", ""),
                        len(getattr(response, "content", b"") or b""), time.monotonic() - fallback_started)
            return response
        response_headers = getattr(response, "headers", {}) or {}
        response_content = getattr(response, "content", b"") or b""
        logger.info("AUX request result source=%s status=%s content_type=%s body_bytes=%s elapsed_sec=%.1f",
                    spec.name, getattr(response, "status_code", None), response_headers.get("content-type", ""),
                    len(response_content), time.monotonic() - started)
        if response.status_code in (401, 403, 404, 405):
            return response
        if response.status_code == 502 or response.status_code == 0 or self._looks_bad_json_response(response, {}):
            logger.warning("AUX browser fallback source=%s status=%s", spec.name, response.status_code)
            browser_kwargs = {"params": dict(params)} if method == "GET" or spec.params_in_query else {"data": dict(params)}
            if headers:
                browser_kwargs["headers"] = headers
            return self._browser_route_request(spec, method, url, **browser_kwargs)
        return response

    def _save_raw(self, raw: dict[str, Any], business_date: str | None, name: str) -> Path:
        day = business_date or "unknown"
        out = self.raw_dir / day
        out.mkdir(parents=True, exist_ok=True)
        digest = raw_hash(raw)
        path = out / f"{name}_{int(time.time() * 1000)}_{digest[:8]}.json"
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def capture_source(self, spec: SourceSpec, *, business_date: str | None = None, params: Mapping[str, Any] | None = None) -> SourceResult:
        """[AUX-V1-r1] raw-first capture. Parser failures never remove local raw."""
        params = dict(params or {})
        url = self._source_url(spec)
        started = time.monotonic()
        # [AUX-V1-r10-diag1] Instrument source/request/parse/raw lifecycle without logging parameter values.
        logger.info("AUX source begin source=%s date=%s resolution=%s evidence=%s raw_only=%s api=%s params_keys=%s",
                    spec.name, business_date, spec.resolution, spec.evidence_level, spec.raw_only,
                    spec.request_path, sorted(str(k) for k in params))
        captured = datetime.now().astimezone().isoformat()
        response = None
        payload: Any = None
        error = ""
        transport_failure = False
        try:
            response = self._aux_request(spec, params)
            if int(getattr(response, "status_code", 0) or 0) == 0:
                transport_failure = True
                self._transport_failure_count += 1
            else:
                self._transport_failure_count = 0
            non_json_error = ""
            try:
                payload = response.json()
            except Exception:
                # [AUX-V1-r2] Preserve gateway HTML/text responses as raw evidence
                # instead of surfacing a misleading JSONDecodeError.
                response_text = str(getattr(response, "text", "") or "")
                response_status = int(getattr(response, "status_code", 0) or 0)
                text_limit = 65536
                payload = {
                    "_aux_non_json_response": response_text[:text_limit],
                    "_aux_response_truncated": len(response_text) > text_limit,
                }
                non_json_error = (
                    f"HTTP {response_status} returned non-JSON response: "
                    f"{response_text[:500].replace(chr(10), ' ').replace(chr(13), ' ')}"
                )
            rows, code = _payload_rows(payload)
            status, code = _status_for_payload(int(response.status_code), payload, rows)
            if int(response.status_code or 0) >= 400:
                error_preview = str(getattr(response, "text", "") or "")[:300]
                logger.warning("AUX HTTP error response source=%s api=%s status=%s content_type=%s body_preview=%s",
                               spec.name, spec.request_path, response.status_code,
                               getattr(response, "headers", {}).get("content-type", ""),
                               _safe_diag_text(error_preview, 300))
            if non_json_error and int(response.status_code) < 400:
                status = STATUS_PARTIAL
            elif non_json_error:
                status = STATUS_FAILED_SOURCE
            error = non_json_error
            raw = {
                "run_id": getattr(self.reporter, "run_id", ""), "source_group": spec.group, "source_page": spec.page_url,
                "source_api": spec.request_path, "resolved_url": url, "method": spec.method, "evidence_level": spec.evidence_level,
                "request_params": {k: v for k, v in params.items() if str(k).lower() not in {"cookie", "token", "authorization", "password"}},
                "business_date": business_date, "resolution": spec.resolution, "http_status": response.status_code,
                "business_code": code, "source_status": status, "source_row_count": len(rows), "source_version": BUILD_VERSION,
                "captured_at": captured, "response_content_type": str(getattr(response, "headers", {}).get("content-type", "")),
                "raw_hash": raw_hash(payload), "raw_json": payload, "schema_version": AUX_SCHEMA_VERSION,
            }
            raw["request_key"] = request_key(spec.request_path, params, business_date)
            raw_path = self._save_raw(raw, business_date, spec.name)
            logger.info("AUX raw saved source=%s date=%s status=%s http_status=%s rows=%s body_bytes=%s raw_hash=%s path=%s",
                        spec.name, business_date, status, response.status_code, len(rows),
                        len(getattr(response, "content", b"") or b""), raw.get("raw_hash"), raw_path)
            if int(response.status_code) in (401, 403):
                logger.error("AUX source auth_rejected source=%s date=%s http_status=%s api=%s raw_path=%s elapsed_sec=%.1f",
                             spec.name, business_date, response.status_code, spec.request_path,
                             raw_path, time.monotonic() - started)
                raise AuxAuthRejected(f"HTTP {response.status_code}")
            parser_rows: list[dict[str, Any]] = []
            if not spec.raw_only and status in {STATUS_COMPLETE, STATUS_EMPTY_VALID, STATUS_PARTIAL}:
                kwargs: dict[str, Any] = {"source_api": spec.request_path, "business_date": business_date}
                if spec.group == "event": kwargs["event_type"] = spec.name
                if spec.group == "curve": kwargs["curve_type"] = spec.name
                if spec.group == "stat": kwargs["stat_type"] = spec.name
                if spec.group == "contract":
                    kwargs["record_type"] = spec.name
                    # [AUX-V1-r13] curve 行无 unitid 字段，注入采集 dyid；
                    # 其余 contract 源行自带 unitid，注入 None 不改变行为。
                    kwargs["unit_id"] = str(params.get("dyid") or "").strip() or None
                    kwargs["use_row_date"] = spec.name in _ROW_DATE_CONTRACT_SOURCES
                try:
                    parser_rows = PARSER_REGISTRY[spec.parser](payload, **kwargs)
                except Exception as exc:  # retain raw and expose a non-fatal partial source
                    status = STATUS_PARTIAL
                    error = f"parser {type(exc).__name__}: {exc}"
                    logger.exception("AUX parser failed source=%s date=%s parser=%s", spec.name, business_date, spec.parser)
            raw["source_status"] = status
            if error:
                raw["response_error"] = error
            logger.info("AUX source complete source=%s date=%s status=%s http_status=%s business_code=%s payload_rows=%s parsed_rows=%s elapsed_sec=%.1f error=%s",
                        spec.name, business_date, status, response.status_code, code, len(rows),
                        len(parser_rows), time.monotonic() - started, _safe_diag_text(error))
            return SourceResult(spec.name, spec.group, status, int(response.status_code), code, parser_rows, {**raw, "raw_path": str(raw_path)}, error)
        except AuxAuthRejected:
            raise
        except Exception as exc:  # raw failure record still matters
            transport_failure = response is None
            if transport_failure:
                self._transport_failure_count += 1
                # [AUX-V1-r11f] 判据 N3：区分「浏览器死亡」与「网络抖动」。
                if _is_browser_lost(exc):
                    self._browser_lost_count += 1
                else:
                    self._browser_lost_count = 0
            else:
                self._browser_lost_count = 0
            error = f"{type(exc).__name__}: {exc}"
            failed_response = getattr(exc, "response", None)
            failed_status = getattr(failed_response, "status_code", None)
            raw = {
                "run_id": getattr(self.reporter, "run_id", ""), "source_group": spec.group, "source_page": spec.page_url,
                "source_api": spec.request_path, "resolved_url": url, "method": spec.method, "evidence_level": spec.evidence_level,
                "request_params": {k: v for k, v in params.items() if str(k).lower() not in {"cookie", "token", "authorization", "password"}},
                "business_date": business_date, "resolution": spec.resolution, "http_status": failed_status or getattr(response, "status_code", None),
                "business_code": None, "source_status": STATUS_FAILED_SOURCE, "source_row_count": 0, "source_version": BUILD_VERSION,
                "captured_at": captured, "raw_hash": raw_hash({"error": error, "url": url}), "raw_json": {"error": error}, "schema_version": AUX_SCHEMA_VERSION,
            }
            raw["request_key"] = request_key(spec.request_path, params, business_date)
            self._save_raw(raw, business_date, spec.name)
            logger.error("AUX source failed source=%s date=%s status=%s error_type=%s elapsed_sec=%.1f error=%s",
                         spec.name, business_date, failed_status, type(exc).__name__,
                         time.monotonic() - started, _safe_diag_text(error))
            if failed_status in (401, 403):
                raise AuxAuthRejected(f"HTTP {failed_status}") from exc
            return SourceResult(spec.name, spec.group, STATUS_FAILED_SOURCE, getattr(response, "status_code", None), None, [], raw, error, transport_failure)

    def _dependency_missing_result(self, spec: SourceSpec, *, business_date: str | None, dependency: str) -> SourceResult:
        """[AUX-V1-r2] Fail/skip dependency sources before any HTTP call."""
        message = f"AUX_DEPENDENCY_MISSING {dependency}"
        logger.error("%s source=%s", message, spec.name)
        raw = {
            "source_api": spec.request_path,
            "business_date": business_date,
            "source_status": STATUS_FAILED_SOURCE,
            "error": message,
            "source_version": BUILD_VERSION,
        }
        return SourceResult(spec.name, spec.group, STATUS_FAILED_SOURCE, None, None, [], raw, message, False)

    def collect(
        self, *, source: str = "all", business_date: str | None = None,
        unitid: str | None = None, skip_sources: set[str] | None = None,
    ) -> list[SourceResult]:
        # [AUX-V1-r11b] unitid is already known from the shared crawler config
        # (it equals config.json unit_id).  Dependency sources must not be
        # skipped just because the CLI omitted --unitid.
        if not str(unitid or "").strip():
            unitid = getattr(self, "unit_id", None) or unitid
        # [AUX-V1-r2] group selection never implicitly enables disabled/raw-only
        # sources; only an exact source name can explicitly select one.
        # [AUX-V1-r10] all-designed is a separate opt-in batch mode; preserve
        # the established meaning of `all` and never relax request contracts.
        if source == "all":
            selected = [s for s in SOURCE_REGISTRY.values() if s.enabled_by_default]
        elif source == "all-designed":
            selected = list(SOURCE_REGISTRY.values())
        elif source in {s.group for s in SOURCE_REGISTRY.values()}:
            selected = [s for s in SOURCE_REGISTRY.values() if s.group == source and s.enabled_by_default]
        else:
            selected = [SOURCE_REGISTRY[source]] if source in SOURCE_REGISTRY else []
        logger.info("AUX collect selection source=%s date=%s selected=%s skipped=%s unitid_present=%s",
                    source, business_date, [spec.name for spec in selected],
                    sorted(skip_sources or set()), bool(str(unitid or "").strip()))
        results = []
        for spec in selected:
            if spec.name in (skip_sources or set()):
                logger.info("AUX source skipped source=%s reason=monthly_already_scheduled date=%s", spec.name, business_date)
                continue
            if source == "all-designed" and spec.evidence_level == "UNVERIFIED":
                results.append(self._not_ready_result(
                    spec, business_date=business_date,
                    reason="AUX_SOURCE_UNVERIFIED; no live request sent",
                ))
                continue
            # [AUX-V1-r1] keep source-specific request parameters explicit; no 96-point broadcast.
            if spec.name in {"unit_constraint", "unit_component", "unit_constraint_jjcq", "generation_hourly_net"} and not str(unitid or "").strip():
                if source == "all-designed":
                    results.append(self._not_ready_result(
                        spec, business_date=business_date,
                        reason="AUX_DEPENDENCY_MISSING unitid; no live request sent",
                    ))
                else:
                    results.append(self._dependency_missing_result(spec, business_date=business_date, dependency="unitid"))
                continue
            if spec.name in {"unit_constraint", "unit_component", "unit_constraint_jjcq"}:
                params = {"pdate": business_date or "", "unitid": str(unitid).strip()}
            elif spec.name == "generation_hourly_net":
                params = {"unitid": str(unitid).strip(), "time": business_date or "", "isYd": "fd"}
            elif spec.name in _CONTRACT_CURVE_METHODS:
                params = _contract_curve_params(business_date, str(unitid).strip())
            else:
                builder = PARAM_BUILDERS.get(spec.name, _pdate_params)
                params = builder(business_date)
            if source == "all-designed":
                missing_params = sorted(set(spec.param_contract) - set(params))
                if missing_params:
                    results.append(self._not_ready_result(
                        spec, business_date=business_date,
                        reason=("AUX_SOURCE_PARAMS_INCOMPLETE missing=" + ",".join(missing_params)
                                + "; no live request sent"),
                    ))
                    continue
            if spec.pagination_mode in {"page", "offset"}:
                seen_pages: set[str] = set()
                page_results: list[SourceResult] = []
                combined_rows: list[dict[str, Any]] = []
                total: int | None = None
                stopped_reason = ""
                for page in range(1, spec.max_pages + 1):
                    page_params = dict(params)
                    if spec.pagination_mode == "page":
                        page_params["page"] = page
                    else:
                        page_params["start"] = (page - 1) * int(page_params.get("length", 100))
                        page_params["length"] = min(int(page_params.get("length", 100)), 100)
                    result = self.capture_source(spec, business_date=business_date, params=page_params)
                    page_results.append(result)
                    combined_rows.extend(result.rows)
                    page_signature = raw_hash(result.rows or result.raw.get("raw_json"))
                    if page_signature in seen_pages:
                        result.status = STATUS_PARTIAL
                        result.error = "repeated page identity; bounded pagination stopped"
                        stopped_reason = result.error
                        break
                    seen_pages.add(page_signature)
                    if result.transport_failure and self._transport_failure_count >= 3:
                        # [AUX-V1-r10-route2] A full registry sweep must keep
                        # source-level failure isolation.  One bad page/route
                        # cannot hide later confirmed sources; narrow source
                        # modes retain the circuit-breaker safety stop.
                        message = "AUX transport failure on 3 consecutive sources/pages"
                        # [AUX-V1-r11f] 浏览器已死亡：交回上层走三层防护重认证，
                        # 而不是继续扫完剩余源（否则整轮长跑静默白跑到人工停止）。
                        if self._browser_lost_count >= 3:
                            raise AuxBrowserLost(
                                f"浏览器进程已死亡：连续 {self._browser_lost_count} 次 CDP 端口不可达；{message}"
                            )
                        if source not in {"all", "all-designed"}:
                            raise RuntimeError(message)
                        logger.warning("%s; continue all-designed sweep after source=%s", message, spec.name)
                        self._transport_failure_count = 0
                    raw_payload = result.raw.get("raw_json") if isinstance(result.raw, Mapping) else None
                    candidate_total = _records_total(raw_payload)
                    if candidate_total is not None:
                        total = candidate_total
                    page_size = int(page_params.get("pageSize", page_params.get("length", 100)))
                    if result.status in {STATUS_FAILED_SOURCE, STATUS_EMPTY_VALID} or len(result.rows) < page_size or (total is not None and len(combined_rows) >= total):
                        break
                if total is not None and len(combined_rows) < total:
                    stopped_reason = stopped_reason or "pagination_pending"
                if page_results:
                    aggregate = page_results[-1]
                    aggregate.rows = combined_rows[:spec.max_rows]
                    if len(combined_rows) > spec.max_rows:
                        stopped_reason = stopped_reason or "pagination_pending:max_rows"
                    if stopped_reason:
                        aggregate.status = STATUS_PARTIAL
                        aggregate.error = stopped_reason
                    aggregate.raw["source_row_count"] = len(aggregate.rows)
                    aggregate.raw["pagination"] = {"pages": len(page_results), "records_total": total, "captured_rows": len(combined_rows), "status": aggregate.status, "error": stopped_reason}
                    results.append(aggregate)
            else:
                result = self.capture_source(spec, business_date=business_date, params=params)
                results.append(result)
                if result.transport_failure and self._transport_failure_count >= 3:
                    # [AUX-V1-r10-route2] See bounded source isolation above.
                    message = "AUX transport failure on 3 consecutive sources"
                    # [AUX-V1-r11f] 同上：浏览器死亡必须触发重认证而非继续扫描。
                    if self._browser_lost_count >= 3:
                        raise AuxBrowserLost(
                            f"浏览器进程已死亡：连续 {self._browser_lost_count} 次 CDP 端口不可达；{message}"
                        )
                    if source not in {"all", "all-designed"}:
                        raise RuntimeError(message)
                    logger.warning("%s; continue all-designed sweep after source=%s", message, spec.name)
                    self._transport_failure_count = 0
        return results

    def _not_ready_result(self, spec: SourceSpec, *, business_date: str | None, reason: str) -> SourceResult:
        """[AUX-V1-r10] Report planned but unsafe/unverified sources without HTTP."""
        return SourceResult(
            spec.name, spec.group, STATUS_SKIPPED_NOT_READY, None, None, [],
            {
                "source_api": spec.request_path,
                "business_date": business_date,
                "resolution": spec.resolution,
                "source_status": STATUS_SKIPPED_NOT_READY,
                "evidence_level": spec.evidence_level,
                "error": reason,
            }, reason, False,
        )
