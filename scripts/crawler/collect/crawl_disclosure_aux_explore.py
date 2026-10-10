#!/usr/bin/env python
"""[AUX-V1-r12] AUX 探索模式 —— 被动录制 PMOS 请求，替代「F12 手工导 HAR 再搬运」。

Version lineage: [AUX-V1-r12] 新增探索模式（本模块）；r11x 采集主链零改动。

为什么需要这一层：
    公司电脑只有浏览器 + UKey，没有 Python。此前每确认一个接口，都要人工按 F12、
    导 HAR、拷回本机、再解析；一次往返半天，且大 HAR 经常截断损坏。本模式复用 AUX
    已有的三层认证（``_auth``），登录成功后**用户什么都不用特殊操作**——正常点击要
    调查的页面即可；程序通过 CDP ``Network`` 域被动录制全部业务请求与响应，回车结束后
    产出人读得懂的《发现清单》（接口 -> 参数 -> 响应字段 -> 一条样例 -> 是否已登记），
    并把整包压成一个 zip 带回来。

边界（红线，必须守住）：
    - 只读录制：不解析入库、不写数据库、不调用任何采集接口、不改 AUX/96 业务代码；
    - 共享认证与采集实现只 import 复用，不修改；
    - Cookie / Authorization / CSRF / 口令一律脱敏后才落盘；zip 内不含日志与配置文件。
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import logging
import re
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit

import requests

# Source-mode direct execution does not put the repository root on sys.path;
# frozen builds keep their own bootstrap. [AUX-V1]
_FROZEN = getattr(sys, "frozen", False)
REPO_ROOT = Path(sys.executable).resolve().parent if _FROZEN else Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # pragma: no cover - 打包/直接脚本两种导入形态
    from scripts.crawler.collect.crawl_disclosure_aux import (
        AUX_DIR,
        BUILD_VERSION,
        _auth,
        _resolve,
        configure_aux_logging,
        load_aux_config,
    )
    from scripts.crawler.collect.disclosure_aux import SOURCE_REGISTRY, _safe_diag_text
    from scripts.crawler.observability import RunReport
    from scripts.crawler.runtime_lock import RuntimeLock
except ImportError:  # pragma: no cover
    from crawl_disclosure_aux import (
        AUX_DIR,
        BUILD_VERSION,
        _auth,
        _resolve,
        configure_aux_logging,
        load_aux_config,
    )
    from disclosure_aux import SOURCE_REGISTRY, _safe_diag_text
    from observability import RunReport
    from runtime_lock import RuntimeLock

try:  # pragma: no cover - 由打包环境保证
    import websocket
except ImportError:  # pragma: no cover
    websocket = None  # type: ignore[assignment]

logger = logging.getLogger("crawl_disclosure_aux_explore")

#: [AUX-V1-r12] 探索模式自身的能力标记；业务实现 BUILD_VERSION 保持 r11x 不变，
#: 产物里两者都记录，便于区分「采集链没动」与「录制器是哪一版」。
EXPLORE_BUILD = "2026-10-08-disclosure-aux-v1-r12-explore"

_MAX_BODY_BYTES = 4 * 1024 * 1024
_MAX_FIELD_PATHS = 220
_MAX_SAMPLE_CHARS = 2400
_PORT_SCAN_RANGE = range(9222, 9231)

_STATIC_EXT_RE = re.compile(
    r"\.(?:js|mjs|css|woff2?|ttf|eot|otf|png|jpe?g|gif|svg|ico|bmp|webp|mp4|mp3|wav|map)(?:\?|#|$)",
    re.IGNORECASE,
)
_STATIC_MIME_PREFIXES = (
    "text/javascript", "application/javascript", "application/x-javascript",
    "text/css", "image/", "font/", "application/font-", "video/", "audio/",
    "application/wasm",
)
_SECRET_HEADER_RE = re.compile(r"(?i)(cookie|authoriz|csrf|ticket|token|secret|api[-_]?key)")
#: [AUX-V1-r12] 短口令类键名必须整词匹配，否则 mapping/spanning 这类正常字段会被误脱敏。
_SECRET_EXACT_RE = re.compile(r"(?i)^(password|passwd|pwd|pin|pass|userpwd|userpass)$")
_SECRET_FORM_RE = re.compile(
    r"(?i)([?&;]?(?:password|passwd|pwd|pin|token|ticket|csrf|secret|signature|accesskey)=)[^&;]*"
)
_INTEREST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "供需关系": ("供需", "平衡", "balance", "supply", "demand"),
    "系统预测": ("预测", "forecast", "负荷", "boundary", "边界"),
    "检修计划": ("检修", "维护", "overhaul", "maintenance", "repair"),
    "火电合约占比": ("合约", "合同", "中长期", "contract", "占比", "比例", "ratio", "火电"),
    "煤价": ("煤", "coal"),
    "钢铁网": ("钢铁", "steel"),
    "价格类": ("price", "价格", "电价", "节点"),
}

#: [AUX-V1-r12] 本机 headless Chrome 实测：`Target.setAutoAttach` 还会挂上
#: `background_page`（Google Hangouts 之类扩展）和 `service_worker`，它们的 favicon
#: 等请求会混进清单。业务接口只可能来自页面/内嵌框架，因此按类型白名单收。
_ALLOWED_TARGET_TYPES = ("page", "iframe", "webview", "other")
_NON_BUSINESS_SCHEMES = ("chrome://", "chrome-extension://", "devtools://", "edge://",
                         "about:", "data:", "blob:", "filesystem:", "file://")


def _cprint(text: str) -> None:
    """[AUX-V1-r12] 公司机控制台常是 GBK：先按控制台编码可替换地编一遍再打印。

    探索模式正等着用户回车，绝不能因为一句中文/箭头触发 UnicodeEncodeError 崩掉。
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        safe = text.encode(encoding, "replace").decode(encoding, "replace")
    except Exception:  # noqa: BLE001
        safe = text
    print(safe, flush=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _looks_static(url: str, mime: str = "") -> bool:
    """[AUX-V1-r12] 静态资源过滤——只留业务请求，zip 才不会几百 MB。"""
    if _STATIC_EXT_RE.search(url or ""):
        return True
    lowered = (mime or "").split(";")[0].strip().lower()
    return any(lowered.startswith(prefix) for prefix in _STATIC_MIME_PREFIXES)


def _is_business_url(url: str) -> bool:
    """[AUX-V1-r12] 只有 http(s) 的站点请求算业务请求；浏览器/扩展内部协议一律不收。"""
    text = (url or "").strip()
    if not text.startswith(("http://", "https://")):
        return False
    return not any(marker in text for marker in ("chrome-extension://", "devtools://", "web+"))


def _is_secret_key(key: Any) -> bool:
    """[AUX-V1-r12] 凭证键判定：认证类键按子串，口令类键按整词（避免误伤 mapping 等字段）。"""
    text = str(key)
    return bool(_SECRET_HEADER_RE.search(text) or _SECRET_EXACT_RE.match(text))


def _flat(text: Any, limit: int = 400) -> str:
    r"""[AUX-V1-r12] 只做换行压平与截断，不额外遮罩——参数样例要靠它看清契约。

    ``_safe_diag_text`` 的凭证正则以 ``[^\s,;]+`` 收尾，会把 ``a=1&b=2`` 整串吃掉；
    对错误日志是对的，对「接口参数值样例」是致命的，所以脱敏在本模块自己做。
    """
    return str(text or "").replace("\r", " ").replace("\n", " ")[:limit]


def _redact_headers(headers: Any) -> dict[str, str]:
    """[AUX-V1-r12] 会话凭证永不落盘：认证类头保留键名、丢弃取值。"""
    out: dict[str, str] = {}
    if not isinstance(headers, dict):
        return out
    for key, value in headers.items():
        if _is_secret_key(key):
            out[str(key)] = "<redacted>"
        else:
            out[str(key)] = _safe_diag_text(value, 200)
    return out


def _redact_params_text(text: str) -> str:
    """[AUX-V1-r12] 参数取值里的口令/票据脱敏（键名留着，因为要看得出契约）。"""
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except Exception:  # noqa: BLE001
            payload = None
        if payload is not None:
            return _flat(json.dumps(_redact_tree(payload), ensure_ascii=False), 2000)
    return _flat(_SECRET_FORM_RE.sub(r"\1<redacted>", text), 2000)


def _redact_tree(node: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "<deep>"
    if isinstance(node, dict):
        return {
            key: ("<redacted>" if _is_secret_key(key) and not isinstance(value, (dict, list))
                  else _redact_tree(value, depth + 1))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_redact_tree(item, depth + 1) for item in node[:50]]
    return node


def _param_names(url: str, post_data: str) -> tuple[str, ...]:
    """[AUX-V1-r12] 接口身份 = 方法 + 路径 + 参数名集合（不含取值）。

    按参数名去重，所以同一接口按不同日期点 30 次只留 1 份响应样例，日期取值记成参数值
    样例——探索模式要的是「一个接口的契约」，不是 30 份重复数据。
    """
    names: set[str] = set()
    try:
        names.update(str(key) for key, _ in parse_qsl(urlsplit(url or "").query))
    except Exception:  # noqa: BLE001
        pass
    if post_data:
        try:
            payload = json.loads(post_data)
            if isinstance(payload, dict):
                names.update(str(key) for key in payload)
            elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                names.update(str(key) for key in payload[0])
        except Exception:  # noqa: BLE001
            try:
                names.update(str(key) for key, _ in parse_qsl(post_data))
            except Exception:  # noqa: BLE001
                pass
    return tuple(sorted(names))


def _endpoint_key(method: str, url: str, post_data: str) -> tuple[str, str, tuple[str, ...]]:
    split = urlsplit(url or "")
    return (str(method or "GET").upper(), f"{split.netloc}{split.path}", _param_names(url, post_data))


def _find_row_list(node: Any, depth: int = 0) -> tuple[int, list[dict[str, Any]]] | None:
    """[AUX-V1-r12] 取载荷里最大的对象数组当数据行，兼容 data/tableData/TableData/TreeData。"""
    if depth > 6:
        return None
    best: tuple[int, list[dict[str, Any]]] | None = None
    if isinstance(node, list):
        rows = [item for item in node if isinstance(item, dict)]
        if rows:
            best = (len(rows), rows)
    if isinstance(node, dict):
        for value in node.values():
            candidate = _find_row_list(value, depth + 1)
            if candidate and (best is None or candidate[0] > best[0]):
                best = candidate
    return best


def _field_paths(node: Any, prefix: str = "", depth: int = 0, out: list[str] | None = None) -> list[str]:
    """[AUX-V1-r12] 抽取响应字段名（点分路径），让人一眼判断接口有没有价值。"""
    collected = out if out is not None else []
    if len(collected) >= _MAX_FIELD_PATHS or depth > 5:
        return collected
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                _field_paths(value, path, depth + 1, collected)
            elif path not in collected:
                collected.append(path)
    elif isinstance(node, list):
        for item in node[:20]:
            _field_paths(item, f"{prefix}[]" if prefix else "[]", depth + 1, collected)
    return collected


def _registered_label(path: str) -> str | None:
    """[AUX-V1-r12] 对照 AUX 登记表：已登记=采集链已有；未登记=候选新来源。"""
    lowered = (path or "").lower()
    hit: set[str] = set()
    for name, spec in SOURCE_REGISTRY.items():
        for candidate in (spec.path, spec.frontend_path):
            needle = (candidate or "").lower().strip()
            if len(needle) > 4 and needle in lowered:
                hit.add(name)
                break
    return ", ".join(sorted(hit)[:4]) if hit else None


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


class ExploreSession:
    """[AUX-V1-r12] 纯逻辑录制器：输入 CDP 事件，输出请求索引、响应体与《发现清单》。

    与 transport 解耦是为了能在本机用合成 CDP 事件做回归——公司电脑没有 Python，
    任何只能在那里验证的逻辑都等于没验证。
    """

    def __init__(self, out_dir: Path, *, keep_bodies: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.bodies_dir = self.out_dir / "bodies"
        self.keep_bodies = bool(keep_bodies)
        self.started_at = _now_iso()
        self.records: dict[tuple[str, str], dict[str, Any]] = {}
        #: 每个接口只保留第一份响应体的解析结果（字段名 + 样例行 + 关键词）
        self.harvested: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
        self.body_requests: collections.deque[tuple[str, str, Any]] = collections.deque()
        self.skipped_static = 0
        self.body_errors = 0
        self._body_seq = 0
        self._stop_reason = ""

    # ── 事件摄取 ─────────────────────────────────────────────────
    def handle_network_event(self, method: str, session_id: str, params: dict[str, Any]) -> None:
        request_id = str(params.get("requestId") or "")
        if not request_id:
            return
        key = (session_id, request_id)
        if method == "Network.requestWillBeSent":
            request = params.get("request") or {}
            url = str(request.get("url") or "")
            if url.startswith(_NON_BUSINESS_SCHEMES) or not _is_business_url(url):
                self.skipped_static += 1
                return
            if _looks_static(url):
                self.skipped_static += 1
                return
            self.records[key] = {
                "session_id": session_id,
                "request_id": request_id,
                "url": url,
                "method": str(request.get("method") or "GET"),
                "post_data": _redact_params_text(str(request.get("postData") or "")),
                "request_headers": _redact_headers(request.get("headers")),
                "resource_type": str(params.get("type") or ""),
                "started_at": _now_iso(),
                "status": None,
                "mime": "",
            }
            return
        record = self.records.get(key)
        if record is None:
            return
        if method == "Network.responseReceived":
            response = params.get("response") or {}
            mime = str(response.get("mimeType") or "")
            url = str(response.get("url") or record["url"])
            if _looks_static(url, mime):
                self.skipped_static += 1
                self.records.pop(key, None)
                return
            record["status"] = response.get("status")
            record["mime"] = mime
            record["url"] = url
            record["response_headers"] = _redact_headers(response.get("headers"))
        elif method == "Network.loadingFinished":
            record["encoded_data_length"] = params.get("encodedDataLength")
            if self.keep_bodies:
                self.body_requests.append((session_id, request_id, key))
        elif method == "Network.loadingFailed":
            record["failed"] = _safe_diag_text(params.get("errorText") or "", 160)

    def store_body(self, key: tuple[str, str], payload: dict[str, Any] | None,
                   error: str | None = None) -> None:
        """[AUX-V1-r12] 保存一条响应体；同一接口（参数名相同）只保存第一份。"""
        record = self.records.get(key)
        if record is None:
            return
        if error:
            record["body_error"] = _safe_diag_text(error, 160)
            self.body_errors += 1
            return
        if not payload:
            return
        raw = payload.get("body") or ""
        base64_encoded = bool(payload.get("base64Encoded"))
        data = base64.b64decode(raw) if base64_encoded else str(raw).encode("utf-8", "replace")
        truncated = len(data) > _MAX_BODY_BYTES
        data = data[:_MAX_BODY_BYTES]
        record["body_bytes"] = len(data)
        group_key = _endpoint_key(record["method"], record["url"], record["post_data"])
        if group_key in self.harvested:
            return
        self._body_seq += 1
        self.bodies_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self._body_seq:04d}_{'bin' if base64_encoded else 'dat'}"
        (self.bodies_dir / name).write_bytes(data)
        self.harvested[group_key] = self._harvest(data, base64_encoded, {
            "body_file": str((self.bodies_dir / name).relative_to(self.out_dir)),
            "body_size": len(data),
            "body_truncated": truncated,
        })

    def _harvest(self, data: bytes, base64_encoded: bool, into: dict[str, Any]) -> dict[str, Any]:
        text = data.decode("utf-8", "replace")
        parsed: Any = None
        if not base64_encoded:
            try:
                parsed = json.loads(text)
            except Exception:  # noqa: BLE001 - 非 JSON 只留原文片段
                parsed = None
        into["field_paths"] = _field_paths(parsed if parsed is not None else "")
        into["sample_row"] = None
        into["row_count"] = None
        rows = _find_row_list(parsed) if parsed is not None else None
        if rows:
            into["row_count"] = rows[0]
            into["sample_row"] = _redact_tree(rows[1][0])
        into["sample_text"] = "" if parsed is not None else text[:_MAX_SAMPLE_CHARS]
        into["keywords"] = self._keywords(into)
        return into

    def _keywords(self, harvested: dict[str, Any]) -> list[str]:
        parts = [
            " ".join(harvested.get("field_paths") or []),
            json.dumps(harvested.get("sample_row"), ensure_ascii=False, default=str)[:_MAX_SAMPLE_CHARS]
            if harvested.get("sample_row") is not None else "",
            (harvested.get("sample_text") or "")[:_MAX_SAMPLE_CHARS],
        ]
        haystack = " ".join(parts).lower()
        return sorted(
            label for label, needles in _INTEREST_KEYWORDS.items()
            if any(needle.lower() in haystack for needle in needles)
        )

    # ── 产物 ────────────────────────────────────────────────────
    def endpoint_groups(self) -> list[dict[str, Any]]:
        """[AUX-V1-r12] 由请求索引聚合接口视图；计数只在这里发生，避免重复累加。"""
        groups: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
        for record in sorted(self.records.values(), key=lambda item: str(item.get("started_at") or "")):
            key = _endpoint_key(record["method"], record["url"], record["post_data"])
            group = groups.get(key)
            if group is None:
                group = {
                    "method": key[0], "endpoint": key[1], "param_names": list(key[2]),
                    "count": 0, "statuses": set(), "resource_types": set(), "mimes": set(),
                    "param_samples": [], "first_seen": record.get("started_at"), "last_seen": None,
                    "url_samples": [],
                }
                groups[key] = group
            group["count"] += 1
            group["last_seen"] = record.get("started_at")
            if record.get("status") is not None:
                group["statuses"].add(int(record["status"]))
            if record.get("resource_type"):
                group["resource_types"].add(record["resource_type"])
            if record.get("mime"):
                group["mimes"].add(record["mime"])
            sample = _param_sample(record)
            if sample and sample not in group["param_samples"] and len(group["param_samples"]) < 4:
                group["param_samples"].append(sample)
            group.update(self.harvested.get(key, {}))
            group["registered"] = _registered_label(key[1])
            if group.get("sample_row") is not None:
                group["keywords"] = self._keywords(group)
        return sorted(groups.values(), key=lambda item: (-len(item.get("keywords") or ()), item["endpoint"]))

    def finalize(self, *, debug_port: int, stop_reason: str, elapsed_sec: float) -> dict[str, Any]:
        self._stop_reason = stop_reason
        self.out_dir.mkdir(parents=True, exist_ok=True)
        finished = _now_iso()
        groups = self.endpoint_groups()
        records = sorted(self.records.values(), key=lambda item: str(item.get("started_at") or ""))
        with (self.out_dir / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(_plain(record), ensure_ascii=False) + "\n")
        (self.out_dir / "发现清单.md").write_text(
            self._manifest(groups, debug_port=debug_port, finished=finished,
                           elapsed_sec=elapsed_sec, record_count=len(records)),
            encoding="utf-8")
        meta = {
            "explore_build": EXPLORE_BUILD,
            "impl_build_version": BUILD_VERSION,
            "started_at": self.started_at,
            "finished_at": finished,
            "elapsed_sec": round(elapsed_sec, 1),
            "debug_port": debug_port,
            "stop_reason": stop_reason,
            "business_requests": len(records),
            "distinct_endpoints": len(groups),
            "skipped_static": self.skipped_static,
            "body_errors": self.body_errors,
            "endpoints": _plain(groups),
            "redaction": "cookie/authorization/csrf/token/password 已脱敏；zip 不含日志与配置",
        }
        (self.out_dir / "explore_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return meta

    def _manifest(self, groups: list[dict[str, Any]], *, debug_port: int, finished: str,
                  elapsed_sec: float, record_count: int) -> str:
        lines: list[str] = [
            "# PMOS 探索模式发现清单",
            "",
            f"- 探索版本：`{EXPLORE_BUILD}`（采集链业务版本 `{BUILD_VERSION}`，本次未改动采集逻辑）",
            f"- 起止：{self.started_at} ~ {finished}（{elapsed_sec:.0f} 秒）｜结束原因：{self._stop_reason}",
            f"- 浏览器调试端口：{debug_port}",
            f"- 业务请求 {record_count} 条 -> 去重后接口 {len(groups)} 个｜跳过静态资源 {self.skipped_static} 条",
            "",
            "> 被动录制，未做任何解析/入库。字段名与样例只用于判断接口有没有业务价值。",
            "",
            "## 目标字段命中概览",
            "",
            "| 需求字段 | 命中接口数 | 命中接口 |",
            "|---|---|---|",
        ]
        by_keyword: dict[str, list[str]] = {}
        untagged: list[str] = []
        for group in groups:
            keywords = group.get("keywords") or []
            if not keywords:
                untagged.append(group["endpoint"])
            for label in keywords:
                by_keyword.setdefault(label, []).append(group["endpoint"])
        for label in _INTEREST_KEYWORDS:
            hits = by_keyword.get(label, [])
            preview = "、".join(dict.fromkeys(hits[:3])) + ("…" if len(hits) > 3 else "")
            lines.append(f"| {label} | {len(dict.fromkeys(hits))} | {preview or '—'} |")
        lines.append(f"| 未命中关键词 | {len(set(untagged))} | （多为字典表/静态数据，见明细） |")
        lines += ["", "## 接口明细", ""]
        for index, group in enumerate(groups, start=1):
            lines.append(f"### {index}. `{group['method']} {group['endpoint']}`")
            lines.append("")
            statuses = sorted(group["statuses"]) if group["statuses"] else "未收到响应"
            lines.append(f"- 调用次数：{group['count']}｜状态码：{statuses}"
                         f"｜请求类型：{sorted(group['resource_types']) or '—'}")
            lines.append(f"- 参数名：{', '.join(group['param_names']) or '（无）'}")
            if group["param_samples"]:
                lines.append(f"- 参数值样例：{'` ／ `'.join(group['param_samples'])}`")
            lines.append(f"- AUX 登记表：{group['registered'] or '**未登记**（候选新来源）'}")
            if group.get("keywords"):
                lines.append(f"- 价值线索：{'、'.join(group['keywords'])}")
            if group.get("row_count") is not None:
                lines.append(f"- 响应行数：约 {group['row_count']} 行")
            field_paths = group.get("field_paths") or []
            if field_paths:
                lines.append(f"- 响应字段（共 {len(field_paths)} 个，展示前 60）：")
                lines.append(f"  `{'`, `'.join(field_paths[:60])}`")
            else:
                lines.append("- 响应字段：（没解析出 JSON 字段，见原文样例）")
            if group.get("sample_row") is not None:
                lines += ["", "```json",
                          json.dumps(group["sample_row"], ensure_ascii=False, indent=2,
                                     default=str)[:_MAX_SAMPLE_CHARS],
                          "```"]
            elif group.get("sample_text"):
                lines += ["", "```text", group["sample_text"], "```"]
            if group.get("body_file"):
                note = "，已截断" if group.get("body_truncated") else ""
                lines.append(f"- 原始响应：`{group['body_file']}`（{group['body_size']} 字节{note}）")
            else:
                lines.append("- 原始响应：未取到响应体（可能已出浏览器内存，或该请求本身失败）")
            lines.append("")
        return "\n".join(lines) + "\n"


def _param_sample(record: dict[str, Any]) -> str:
    query = urlsplit(record.get("url") or "").query
    post = str(record.get("post_data") or "")
    return _flat(f"{query}{' POST ' + post if post else ''}", 160)


class CdpNetworkRecorder:
    """[AUX-V1-r12] CDP flat-session 网络录制循环（本模块唯一使用 socket 的地方）。

    用浏览器级 target + ``Target.setAutoAttach(flatten=True)``，而不是只挂某一个页面：
    用户点出来的新标签页、内嵌 iframe 都会自动 attach，探索模式不会漏请求。
    事件处理只入队、命令只在顶层循环发送，避免在同一 WebSocket 上重入取响应。
    """

    def __init__(self, debug_port: int, session: ExploreSession) -> None:
        self.debug_port = int(debug_port)
        self.session = session
        self.ws: Any = None
        self._id = 0
        self._waiting: dict[int, Any] = {}
        self._deferred: list[dict[str, Any]] = []
        self._sessions: set[str] = set()
        self.broken_reason = ""

    def _endpoint(self, suffix: str) -> str:
        return f"http://127.0.0.1:{self.debug_port}/{suffix}"

    def _open_socket(self, ws_url: str) -> Any:
        """[AUX-V1-r12] 不带 Origin 头连接 CDP——本机 Chrome 146 实测验证过的形态。

        认证状态机自己启动浏览器时会加 ``--remote-allow-origins=*``，但探索模式经常
        走 L1 复用**已经开着的**浏览器（96 主爬虫或上一次 AUX 留下的）。那种浏览器不
        一定带这个 flag，带 Origin 的握手会被 Chrome 直接 403 拒绝（本机实测复现）。
        探索模式不重启浏览器（会踢掉用户会话），所以只能在客户端侧兼容。
        """
        try:
            return websocket.create_connection(ws_url, timeout=2, suppress_origin=True)
        except TypeError:  # pragma: no cover - 极旧 websocket-client 无该参数
            return websocket.create_connection(ws_url, timeout=2)

    def connect(self) -> None:
        if websocket is None:
            raise RuntimeError("缺少 websocket-client 依赖，探索模式无法录制")
        reply = requests.get(self._endpoint("json/version"), timeout=5)
        reply.raise_for_status()
        ws_url = str(reply.json().get("webSocketDebuggerUrl") or "")
        if not ws_url:
            raise RuntimeError("浏览器未暴露 DevTools WebSocket 地址")
        self.ws = self._open_socket(ws_url)
        self._send("Target.setDiscoverTargets", {"discover": True})
        self._send("Target.setAutoAttach", {
            "autoAttach": True, "waitForDebuggerOnStart": False, "flatten": True,
        })
        existing = 0
        try:
            targets = requests.get(self._endpoint("json"), timeout=5).json()
        except Exception:  # noqa: BLE001
            targets = []
        for item in targets if isinstance(targets, list) else []:
            if item.get("type") not in _ALLOWED_TARGET_TYPES:
                continue
            target_id = str(item.get("id") or "")
            if not target_id:
                continue
            existing += 1
            self._deferred.append({"method": "Target.attachToTarget",
                                   "params": {"targetId": target_id, "flatten": True}})
        logger.info("AUX explore recorder attached port=%s existing_targets=%s",
                    self.debug_port, existing)

    def _send(self, method: str, params: dict[str, Any] | None = None,
              session_id: str | None = None, wait: bool = False) -> int | None:
        if self.ws is None:
            return None
        self._id += 1
        message: dict[str, Any] = {"id": self._id, "method": method, "params": params or {}}
        if session_id:
            message["sessionId"] = session_id
        if wait:
            self._waiting[self._id] = None
        try:
            self.ws.send(json.dumps(message))
        except Exception as exc:  # noqa: BLE001
            self.broken_reason = _safe_diag_text(exc, 160)
            self._waiting.pop(self._id, None)
            logger.warning("AUX explore recorder send failed method=%s error=%s",
                           method, self.broken_reason)
        return self._id if wait else None

    def _flush_deferred(self) -> None:
        while self._deferred:
            command = self._deferred.pop(0)
            self._send(command["method"], command.get("params"), command.get("session_id"))

    def _pump(self, timeout: float = 0.4) -> None:
        if self.ws is None:
            return
        try:
            self.ws.settimeout(timeout)
            raw = self.ws.recv()
        except Exception as exc:  # noqa: BLE001
            if "Timeout" in exc.__class__.__name__:
                return
            self.broken_reason = _safe_diag_text(exc, 160)
            logger.warning("AUX explore recorder connection lost: %s", self.broken_reason)
            return
        try:
            message = json.loads(raw)
        except Exception:  # noqa: BLE001
            return
        self._dispatch(message)

    def _dispatch(self, message: dict[str, Any]) -> None:
        if "id" in message:
            if message["id"] in self._waiting:
                self._waiting[message["id"]] = message
            return
        method = str(message.get("method") or "")
        params = message.get("params") or {}
        if method == "Target.attachedToTarget":
            info = params.get("targetInfo") or {}
            if str(info.get("type") or "") not in _ALLOWED_TARGET_TYPES:
                return  # 扩展 background_page / service_worker 不是业务来源，挂了只会灌进噪音
            new_session = str(params.get("sessionId") or "")
            if new_session and new_session not in self._sessions:
                self._sessions.add(new_session)
                self._deferred.append({"method": "Network.enable", "params": {},
                                       "session_id": new_session})
            return
        if method.startswith("Network."):
            self.session.handle_network_event(method, str(message.get("sessionId") or ""), params)

    def _call(self, session_id: str, method: str, params: dict[str, Any] | None = None,
              timeout: float = 5.0) -> tuple[dict[str, Any] | None, str | None]:
        message_id = self._send(method, params, session_id, wait=True)
        if message_id is None:
            return None, "send failed"
        deadline = time.monotonic() + timeout
        while self._waiting.get(message_id) is None and time.monotonic() < deadline:
            self._pump(0.3)
            self._flush_deferred()
        reply = self._waiting.pop(message_id, None)
        if reply is None:
            return None, "timeout"
        if "error" in reply:
            return None, _safe_diag_text(reply.get("error"), 160)
        return reply.get("result") or {}, None

    def _serve_bodies(self, limit: int = 4) -> None:
        for _ in range(limit):
            if not self.session.body_requests:
                return
            session_id, request_id, key = self.session.body_requests.popleft()
            result, error = self._call(session_id, "Network.getResponseBody", {"requestId": request_id})
            self.session.store_body(key, result, error)

    def run(self, stop_event: threading.Event, deadline: float) -> str:
        """录制主循环，返回结束原因（中文，直接进清单与日志）。"""
        last_liveness = time.monotonic()
        while not stop_event.is_set() and time.monotonic() < deadline and not self.broken_reason:
            self._pump(0.3)
            self._flush_deferred()
            self._serve_bodies()
            if time.monotonic() - last_liveness >= 5.0:
                last_liveness = time.monotonic()
                if not self._browser_alive():
                    return "浏览器已关闭或调试端口失联"
        if stop_event.is_set():
            return "用户在控制台按回车结束"
        if self.broken_reason:
            return f"录制连接中断：{self.broken_reason}"
        return "达到最长自动结束时间"

    def _browser_alive(self) -> bool:
        try:
            return requests.get(self._endpoint("json/version"), timeout=2).ok
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:  # noqa: BLE001
            pass
        self.ws = None


def _detect_debug_port(preferred: int, cfg: dict[str, Any]) -> int:
    """[AUX-V1-r12] 认证结果没带端口时，按配置基准端口和常用区间找回调试端口。"""
    candidates: list[int] = []
    if preferred:
        candidates.append(int(preferred))
    base = int(cfg.get("debug_port") or 0)
    if base:
        candidates.extend([base, base + 1, base + 2])
    candidates.extend(_PORT_SCAN_RANGE)
    for port in dict.fromkeys(candidates):
        try:
            if requests.get(f"http://127.0.0.1:{port}/json/version", timeout=1).ok:
                return port
        except Exception:  # noqa: BLE001
            continue
    return 0


def build_explore_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PMOS AUX 探索模式（被动录制，免 F12 免导 HAR）")
    parser.add_argument("--explore", action="store_true",
                        help="进入探索模式：登录后录制你在浏览器里点出来的全部接口")
    parser.add_argument("--config", default=None,
                        help="AUX 配置 JSON，默认同目录 config_disclosure_aux.json")
    parser.add_argument("--explore-max-sec", type=int, default=1800,
                        help="最长录制秒数（默认 1800），到点自动结束并出清单")
    parser.add_argument("--explore-no-bodies", action="store_true",
                        help="只记录接口与字段名，不保存响应体（产物最小）")
    parser.add_argument("--explore-out", default=None, help="探索产物目录，默认 output_aux/explore/<run_id>")
    return parser


def _usage_banner(seconds: int) -> None:
    _cprint("")
    _cprint("=" * 66)
    _cprint("AUX 探索模式（正在录制）—— 你只需要正常点网页")
    _cprint("=" * 66)
    _cprint("1. 浏览器已经登录好了，把它切到前台，像平时查数据一样逐个打开你要调查的页面；")
    _cprint("2. 每个页面等数据加载出来再点下一个（数据出来了才算录到）；")
    _cprint("3. 不用按 F12、不用导出任何东西，程序在后台自动记录全部网站请求和响应；")
    _cprint("4. 可以随便开新标签页、来回切换，全都在录制范围内；")
    _cprint("5. 点完之后回到本窗口按【回车】结束；结束前请不要关闭浏览器。")
    _cprint("")
    _cprint(f"最长自动结束：{seconds} 秒（到点也会生成清单，不会白跑）。")
    _cprint("Cookie / 口令一类凭证已自动脱敏，产物可以安全拷回。")
    _cprint("=" * 66)
    _cprint("")


def make_zip(session_dir: Path) -> Path:
    """[AUX-V1-r12] 整包压成**一个**文件——搬运成本才是决定这套流程成败的东西。

    只打包清单/请求索引/响应体/元数据；日志与配置故意不进 zip，避免把会话凭证或
    数据库密码一起拷回来。
    """
    archive = session_dir.parent / f"explore_{session_dir.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(session_dir.rglob("*")):
            if path.is_file() and path != archive:
                handle.write(path, arcname=str(path.relative_to(session_dir)))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
    logger.info("AUX explore archive path=%s bytes=%s sha256=%s",
                archive, archive.stat().st_size, digest)
    return archive


def main(argv: list[str] | None = None) -> int:
    args, unknown = build_explore_parser().parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if unknown:
        logger.info("AUX explore 忽略不适用参数 unknown=%s", ",".join(unknown[:8]))
    config_path = _resolve(AUX_DIR, args.config, "config_disclosure_aux.json")
    cfg = load_aux_config(config_path)
    output_dir = _resolve(AUX_DIR, str(cfg.get("output_dir") or "output_aux"), "output_aux")
    configure_aux_logging(output_dir)
    report = RunReport(output_dir / "aux_report.json", build_version=EXPLORE_BUILD, args=vars(args))
    session_dir = (Path(args.explore_out) if args.explore_out and Path(args.explore_out).is_absolute()
                   else output_dir / "explore" / report.run_id)
    logger.info("AUX explore start build=%s impl=%s run_id=%s config=%s out=%s",
                EXPLORE_BUILD, BUILD_VERSION, report.run_id, config_path, session_dir)
    lock_path = _resolve(AUX_DIR, str(cfg.get("shared_lock_path") or "../output_96/.crawler.lock"),
                         "../output_96/.crawler.lock")
    try:
        with RuntimeLock(lock_path):
            auth_path = _resolve(AUX_DIR, str(cfg.get("auth_config_path") or "../config.json"),
                                 "../config.json")
            result = _auth(auth_path, report)
            report.stage("auth", "PASS", debug_port=int(getattr(result, "debug_port", 0) or 0),
                         cookie_present=bool(result.cookie))
            port = _detect_debug_port(int(getattr(result, "debug_port", 0) or 0), cfg)
            if not port:
                report.event("ERROR", "EXPLORE_NO_CDP", "认证后找不到浏览器调试端口，无法录制")
                report.finish("FAIL", reason="EXPLORE_NO_CDP")
                _cprint("找不到浏览器调试端口，本次没能录制。请把本目录 aux_crawler.log 拷回。")
                return 3
            session = ExploreSession(session_dir, keep_bodies=not args.explore_no_bodies)
            recorder = CdpNetworkRecorder(port, session)
            recorder.connect()
            _usage_banner(args.explore_max_sec)
            stop_event = _start_console_trigger(report)
            started = time.monotonic()
            deadline = started + max(60, int(args.explore_max_sec))
            reason = recorder.run(stop_event, deadline)
            recorder.close()
            elapsed = time.monotonic() - started
            meta = session.finalize(debug_port=port, stop_reason=reason, elapsed_sec=elapsed)
            archive = make_zip(session_dir)
            report.stage("explore_record", "PASS", requests=meta["business_requests"],
                         endpoints=meta["distinct_endpoints"], skipped_static=meta["skipped_static"],
                         stop_reason=reason)
            report.finish("PASS", mode="explore", requests=meta["business_requests"],
                          endpoints=meta["distinct_endpoints"], zip=str(archive))
            _cprint("")
            _cprint("=" * 66)
            _cprint(f"探索结束（{reason}）｜业务请求 {meta['business_requests']} 条 "
                    f"-> 接口 {meta['distinct_endpoints']} 个")
            _cprint(f"清单：{session_dir / '发现清单.md'}")
            _cprint(f"请把这一个文件拷回来给我：{archive}")
            _cprint("（现在可以关闭浏览器了）")
            _cprint("=" * 66)
            return 0
    except KeyboardInterrupt:
        logger.warning("AUX explore interrupted run_id=%s", report.run_id)
        report.finish("INTERRUPTED", reason="KeyboardInterrupt")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("AUX explore failed")
        report.exception("explore", exc)
        report.finish("FAIL", reason=_safe_diag_text(exc, 200))
        _cprint(f"探索模式失败：{_safe_diag_text(exc, 200)}")
        _cprint("请把本目录 aux_crawler.log 拷回（日志里的凭证类字段已脱敏）。")
        return 1


def _start_console_trigger(report: RunReport) -> threading.Event:
    """[AUX-V1-r12] 回车结束：stdin 读放在守护线程，浏览器失联时主循环自己收尾。

    只有**真的读到一行**才算用户按了回车。EOF（stdin 被重定向到空、或双击启动时没有
    可用输入流）绝不能当成「立即结束」，否则探索模式会在用户还没点网页时就收工——
    这种情况按「无控制台」处理，交给最长秒数 / 浏览器失联两个兜底。
    """
    stop_event = threading.Event()

    def _wait() -> None:
        try:
            line = sys.stdin.readline() if sys.stdin is not None else ""
        except Exception as exc:  # noqa: BLE001
            line = ""
            reason = _safe_diag_text(exc, 120)
        else:
            reason = "stdin EOF" if not line else ""
        if line:
            stop_event.set()
            return
        logger.warning("AUX explore 无可用控制台输入（%s），改为等待到最长自动结束", reason or "-")
        report.event("WARN", "EXPLORE_NO_CONSOLE", "无控制台输入，按最长秒数自动结束")

    threading.Thread(target=_wait, name="aux-explore-console", daemon=True).start()
    return stop_event


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
