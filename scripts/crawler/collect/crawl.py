"""
国网PMOS爬虫核心 — 认证、爬取
"""

from __future__ import annotations

import logging
import json
import re
import time
import urllib.parse
from typing import Any, Optional

import requests
import websocket

logger = logging.getLogger(__name__)

# 96 个 15 分钟时段标签
TIME_LABELS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15)][1:] + ["24:00"]

# QCTC 前端当前把 OAuth 凭据放在 sessionStorage 的 token 键中；这里保留
# 常见兼容键名，读取时只取值用于当前浏览器 fetch，绝不写入日志或 raw。
QCTC_TOKEN_STORAGE_KEYS = ("token", "access_token", "accessToken", "id_token")
QCTC_ORIGIN = "https://pmos.sd.sgcc.com.cn:18080"
QCTC_SSO_SERVICE_URL = "https://pmos.sd.sgcc.com.cn:18080/qctc/admin/sdsso/SSOLogin"


class QCTCAuthRejected(SystemExit):
    """QCTC 认证失败时终止整轮运行，避免对几十个日期重复请求 401。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(1)


def _is_qctc_page_url(url: str) -> bool:
    """识别 QCTC SSO/业务页面，兼容 /qctc/ 与 /qctc-trade/。"""
    parts = urllib.parse.urlsplit(str(url or ""))
    return parts.netloc.lower() == "pmos.sd.sgcc.com.cn:18080" and (
        parts.path.startswith("/qctc/") or parts.path.startswith("/qctc-trade/")
    )


def _is_qctc_origin_page_url(url: str) -> bool:
    """识别 QCTC origin 根页；SSO 完成后根页也可能保留 Bearer。"""
    return urllib.parse.urlsplit(str(url or "")).netloc.lower() == QCTC_ORIGIN.removeprefix("https://")


def _is_qctc_context_page_url(url: str) -> bool:
    """识别 QCTC origin 上的任意页面；具体 SPA route 只用于诊断。"""
    return _is_qctc_origin_page_url(url)


def _safe_url_for_log(url: str) -> str:
    """去掉 query/fragment，尤其不能把 SSO ticket 写入日志或报告。"""
    parts = urllib.parse.urlsplit(str(url or ""))
    if parts.scheme or parts.netloc:
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))[:240]
    return parts.path[:240]


class _CdpClient:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=8)
        self._id = 0

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass

    def call(self, method: str, params: Optional[dict[str, Any]] = None, timeout: int = 30) -> dict[str, Any]:
        self._id += 1
        msg_id = self._id
        self.ws.settimeout(timeout)
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
        while True:
            raw = self.ws.recv()
            data = json.loads(raw)
            if data.get("id") == msg_id:
                if "error" in data:
                    raise RuntimeError(f"CDP {method} 失败: {data['error']}")
                return data.get("result") or {}


def _page_targets(debug_port: int) -> list[dict[str, Any]]:
    response = requests.get(f"http://127.0.0.1:{debug_port}/json", timeout=5)
    response.raise_for_status()
    return [
        p for p in response.json()
        if p.get("type") == "page" and p.get("webSocketDebuggerUrl")
    ]


def _get_pmos_page(debug_port: int, *, prefer_qctc: bool = False) -> dict[str, Any]:
    pages = _page_targets(debug_port)
    if not pages:
        raise RuntimeError(f"DevTools port={debug_port} 没有可用页面")
    if prefer_qctc:
        for p in pages:
            if _is_qctc_page_url(str(p.get("url") or "")):
                return p
        for p in pages:
            if _is_qctc_origin_page_url(str(p.get("url") or "")):
                return p
    for p in pages:
        url = str(p.get("url") or "")
        if "pmos.sd.sgcc.com.cn:18080/trade" in url:
            return p
    for p in pages:
        url = str(p.get("url") or "")
        if "pmos.sd.sgcc.com.cn" in url:
            return p
    return pages[0]


def _browser_page_state(cdp: _CdpClient) -> dict[str, Any]:
    """读取浏览器上下文诊断信息；只返回存储键名，不返回凭据值。"""
    result = cdp.call(
        "Runtime.evaluate",
        {
            "expression": """(() => {
              const names = ["token", "access_token", "accessToken", "id_token"];
              const keys = storage => { try { return Object.keys(storage || {}); } catch (_) { return []; } };
              const sessionKeys = keys(sessionStorage);
              const localKeys = keys(localStorage);
              const present = names.some(k => {
                try { return !!(sessionStorage.getItem(k) || localStorage.getItem(k)); }
                catch (_) { return false; }
              });
              return {
                url: location.href,
                origin: location.origin,
                path: location.pathname,
                ready: document.readyState,
                qctc_route: location.pathname.indexOf("/qctc") >= 0,
                bearer_present: present,
                session_storage_keys: sessionKeys,
                local_storage_keys: localKeys,
              };
            })()""",
            "returnByValue": True,
        },
        timeout=5,
    )
    return ((result.get("result") or {}).get("value") or {})


def _try_portal_qctc_entry(cdp: _CdpClient) -> dict[str, Any]:
    """在已登录门户中尝试点击新版 QCTC 菜单。

    旧门户 Cookie 不会自动生成 QCTC token；门户菜单通常会附带一次 SSO
    跳转参数。这里优先点击明确指向 QCTC/信息披露/现货交易的可见菜单项，
    不读取或记录任何凭据。找不到菜单时仍由上层保留窗口等待人工操作。
    """
    result = cdp.call(
        "Runtime.evaluate",
        {
            "expression": """(() => {
              const visible = el => {
                if (!el || !(el.offsetWidth || el.offsetHeight || el.getClientRects().length)) return false;
                const s = getComputedStyle(el), r = el.getBoundingClientRect();
                return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity || 1) > 0.01
                  && r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0;
              };
              const clean = value => String(value || '').replace(/\\s+/g, '').toLowerCase();
              const nodes = [...document.querySelectorAll(
                'a,button,[role="menuitem"],[role="button"],li,.el-menu-item,.el-submenu__title'
              )];
              const candidates = nodes.map((el, index) => {
                const text = String(el.innerText || el.textContent || '').trim().slice(0, 100);
                const href = String(el.href || el.getAttribute('href') || '').slice(0, 300);
                const key = clean(text + ' ' + href);
                let score = 0;
                if (key.includes('qctc-trade')) score += 1000;
                if (key.includes('qctc')) score += 900;
                if (key.includes('信息披露')) score += 700;
                if (key.includes('电力现货')) score += 450;
                if (key.includes('现货交易')) score += 350;
                if (key.includes('电力交易')) score += 250;
                if (key.includes('实时') || key.includes('日前')) score += 80;
                if (key.includes('退出') || key.includes('注销') || key.includes('logout')) score -= 1200;
                return {el, index, text, href, score};
              }).filter(x => visible(x.el) && x.score >= 250)
                .sort((a, b) => b.score - a.score);
              const best = candidates[0];
              if (!best) return {clicked:false, candidates:[]};
              best.el.focus();
              best.el.click();
              return {
                clicked:true,
                text:best.text,
                href:best.href,
                candidates:candidates.slice(0, 8).map(x => ({text:x.text, href:x.href, score:x.score}))
              };
            })()""",
            "returnByValue": True,
        },
        timeout=8,
    )
    return ((result.get("result") or {}).get("value") or {})


def _open_qctc_via_portal_ticket(cdp: _CdpClient) -> dict[str, Any]:
    """在门户同源上下文获取短期 ticket，并打开真正的 QCTC SSO 地址。

    ticket 只在浏览器 JS 内流转，CDP 返回值只包含状态摘要，避免进入
    Python 日志、report 或 raw。先同步创建 about:blank 也能降低 popup blocker
    拦截异步 window.open 的概率。
    """
    result = cdp.call(
        "Runtime.evaluate",
        {
            "expression": f"""(async () => {{
              const blank = window.open('about:blank', '_blank');
              try {{
                const response = await fetch('/px-common-authcenter/sso/token', {{
                  method: 'POST',
                  credentials: 'include',
                  cache: 'no-store',
                  headers: {{
                    'Accept': 'application/json, text/plain, */*',
                    'Content-Type': 'application/json;charset=UTF-8',
                    'ClientTag': 'OUTNET_BROWSE'
                  }},
                  body: '{{}}'
                }});
                let payload = {{}};
                try {{ payload = await response.json(); }} catch (_) {{ payload = {{}}; }}
                const businessStatus = payload && payload.status;
                if (!response.ok || businessStatus !== 0 || !payload.data) {{
                  if (blank) blank.close();
                  return {{
                    ok: false,
                    http_status: response.status,
                    business_status: businessStatus,
                    message: 'ticket_request_failed'
                  }};
                }}
                const target = {json.dumps(QCTC_SSO_SERVICE_URL)}
                  + '?ticket=' + encodeURIComponent(payload.data);
                if (blank) blank.location.href = target;
                else window.location.href = target;
                return {{
                  ok: true,
                  opened_new_window: !!blank,
                  http_status: response.status,
                  business_status: businessStatus
                }};
              }} catch (_) {{
                if (blank) blank.close();
                return {{
                  ok: false,
                  http_status: 0,
                  business_status: null,
                  message: 'ticket_request_exception'
                }};
              }}
            }})()""",
            "awaitPromise": True,
            "returnByValue": True,
            "userGesture": True,
        },
        timeout=20,
    )
    value = ((result.get("result") or {}).get("value") or {})
    safe = {
        "ok": bool(value.get("ok")),
        "opened_new_window": bool(value.get("opened_new_window")),
        "http_status": value.get("http_status"),
        "business_status": value.get("business_status"),
        "message": str(value.get("message") or "")[:100],
    }
    return safe


# 同源兜底页：fetch 只要求「同源文档」，不要求停在 QCTC 的 SPA 路由上。
# 现状（2026-09-16 审计）：/qctc-trade/* 的 SPA 路由可能在加载完成后被弹回
# 门户 pmos.sd.sgcc.com.cn/#/dashboard（换成另一个 origin），使后续 fetch 直接
# 以 TypeError: Failed to fetch (status=0) 失败；而 2026-09-13 确实在 :18080 的
# 浏览器上下文里成功取到过 status=200 的 QCTC 数据。
# 因此这里准备一组确定的同源页面，**仅在 fetch 已经失败后**用来重试——
# 属于待验证的兜底假设，不作为成功路径。
_ORIGIN_FALLBACK_PATHS: tuple[str, ...] = (
    "/zcq/main/index.do",
    "/favicon.ico",
)


def _same_origin(left: str, right: str) -> bool:
    a = urllib.parse.urlsplit(left or "")
    b = urllib.parse.urlsplit(right or "")
    return bool(a.netloc) and a.scheme == b.scheme and a.netloc == b.netloc


def _origin_fallback_pages(page_url: str) -> tuple[str, ...]:
    """由目标接口 URL 推导同源兜底页面（仅用于 fetch 失败后的重试）。"""
    parts = urllib.parse.urlsplit(page_url)
    if not parts.netloc:
        return ()
    origin = f"{parts.scheme}://{parts.netloc}"
    return tuple(origin + path for path in _ORIGIN_FALLBACK_PATHS)


def _fetch_retryable(result: dict[str, Any]) -> bool:
    """是否需要换同源兜底页重试。

    ``status=0`` 是 Chromium 取消了跨源/无效请求（页面被弹回门户时必然
    出现）；``502`` 是网关在旧 ``/trade`` 端点上返回的 Bad Gateway。
    """
    try:
        status = int(result.get("status") or 0)
    except (TypeError, ValueError):
        status = 0
    return status in (0, 502)


def _wait_page_ready(cdp: _CdpClient, page_url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """等待标签页回到 page_url 的同源且 document 加载完成。

    只校验同源，不要求命中具体 SPA 路由：路由可能被站点弹回门户，而
    fetch 只需要同源文档 + 浏览器自带的 Cookie。
    """
    target = urllib.parse.urlsplit(page_url)
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last = _browser_page_state(cdp)
            now = urllib.parse.urlsplit(str(last.get("url") or ""))
            if (
                now.scheme == target.scheme
                and now.netloc == target.netloc
                and last.get("ready") == "complete"
            ):
                return last
        except Exception:
            pass
        time.sleep(0.5)
    return last


def _navigate_for_fetch(cdp: _CdpClient, page_url: str) -> dict[str, Any]:
    cdp.call("Page.navigate", {"url": page_url}, timeout=10)
    # 新启动的 Edge 没有缓存前端资源时，3 秒不足以完成导航；此时立刻
    # fetch 会被 Chromium 以 TypeError: Failed to fetch 取消。
    state = _wait_page_ready(cdp, page_url)
    logger.info(
        "浏览器页面就绪检查 url=%s ready=%s token_present=%s",
        str(state.get("url") or "")[:180],
        state.get("ready"),
        state.get("bearer_present"),
    )
    return state


def _browser_fetch(
    debug_port: int,
    *,
    page_url: str,
    url: str,
    method: str,
    headers: dict[str, str],
    body: Optional[str] = None,
    origin_fallback_pages: Optional[tuple[str, ...]] = None,
) -> dict[str, Any]:
    """在真实浏览器页面里执行 fetch。"""
    page = _get_pmos_page(
        debug_port,
        prefer_qctc=_is_qctc_page_url(page_url),
    )
    cdp = _CdpClient(page["webSocketDebuggerUrl"])
    try:
        cdp.call("Page.enable")
        cur_url = str(page.get("url") or "")
        # fetch 必须从同源页面发起。旧版是 /trade，新版 QCTC 是 /qctc-trade。
        if _same_origin(cur_url, page_url):
            # 已经站在同源文档上就不再导航：导航既浪费时间，又可能把
            # 当前可用的同源文档换成会被弹回门户的 SPA 路由。
            logger.info("浏览器 fetch 复用同源页面: %s", cur_url[:180])
        else:
            logger.info("浏览器 fetch 前导航到交易页: %s", page_url)
            _navigate_for_fetch(cdp, page_url)

        # JS 不能设置 Cookie/User-Agent/Connection/Referer 等 forbidden headers。
        forbidden = {
            "cookie", "user-agent", "connection", "host", "origin", "referer",
            "content-length", "accept-encoding",
        }
        safe_headers = {
            str(k): str(v)
            for k, v in headers.items()
            if str(k).lower() not in forbidden and v is not None
        }

        expr = f"""
(async () => {{
  const url = {json.dumps(url, ensure_ascii=False)};
  const opts = {{
    method: {json.dumps(method)},
    credentials: 'include',
    cache: 'no-store',
    headers: {json.dumps(safe_headers, ensure_ascii=False)}
  }};
  // QCTC 前端把 OAuth token 保存在 sessionStorage，并由 axios 拦截器
  // 写入 Authorization。CDP 直接 fetch 需要复现这一行为；token 不落盘、不写日志。
  const token = ['token', 'access_token', 'accessToken', 'id_token']
    .map(k => {{ try {{ return sessionStorage.getItem(k) || localStorage.getItem(k); }} catch (_) {{ return ''; }} }})
    .find(Boolean) || '';
  if (token && !opts.headers.Authorization && !opts.headers.authorization) {{
    opts.headers.Authorization = token.startsWith('Bearer ') ? token : `Bearer ${{token}}`;
  }}
  const body = {json.dumps(body, ensure_ascii=False)};
  if (body !== null && body !== undefined) opts.body = body;
  try {{
    const res = await fetch(url, opts);
    const text = await res.text();
    return {{
      ok: res.ok,
      status: res.status,
      statusText: res.statusText,
      url: res.url,
      contentType: res.headers.get('content-type') || '',
      text: text
    }};
  }} catch (e) {{
    return {{
      ok: false,
      status: 0,
      statusText: String(e && (e.stack || e.message || e)),
      url: url,
      contentType: '',
      text: ''
    }};
  }}
}})()
"""
        def _do_fetch() -> dict[str, Any]:
            out = cdp.call(
                "Runtime.evaluate",
                {
                    "expression": expr,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
                timeout=90,
            )
            value = ((out.get("result") or {}).get("value") or {})
            if not isinstance(value, dict):
                raise RuntimeError(f"浏览器 fetch 返回异常: {out}")
            return value

        result = _do_fetch()

        # 只有在请求已经被同源策略/网关打断时才换页重试，成功路径不受影响。
        if origin_fallback_pages is None:
            fallbacks = _origin_fallback_pages(page_url) if _fetch_retryable(result) else ()
        else:
            fallbacks = origin_fallback_pages
        for fallback in fallbacks:
            if not _fetch_retryable(result):
                break
            if not _same_origin(fallback, page_url):
                logger.warning("跳过跨源兜底页（必须与接口同源）: %s", fallback)
                continue
            logger.warning(
                "浏览器 fetch 失败(status=%s)，改用同源兜底页重试: %s",
                result.get("status"), fallback,
            )
            try:
                _navigate_for_fetch(cdp, fallback)
                result = _do_fetch()
            except Exception as exc:  # noqa: BLE001
                logger.warning("同源兜底页重试异常: %s: %s", type(exc).__name__, exc)
        return result
    finally:
        cdp.close()


def period_no_from_time(time_str: str) -> int:
    """将时间标签转为 96 点序号 (00:15→1, 24:00→96)"""
    parts = time_str.split(":")
    return (int(parts[0]) * 60 + int(parts[1])) // 15


# 导出实际文件 → (Periodid, systemload, ...) 列映射
# 导出列名可能带单位/别名，这里列出所有已知别名，解析时做归一化。
ACTUAL_EXPORT_COLUMN_ALIASES: dict[str, str] = {
    "periodid": "Periodid",
    "period": "Periodid",
    "时刻": "Periodid",
    "时间": "Periodid",
    "systemload": "systemload",
    "系统负荷": "systemload",
    "直调负荷": "systemload",
    "统调负荷": "systemload",
    "dfdcload": "dfdcload",
    "地方电厂": "dfdcload",
    "excload": "excload",
    "联络线": "excload",
    "fdload": "fdload",
    "风电": "fdload",
    "gfload": "gfload",
    "光伏": "gfload",
    "sytsjz": "sytsjz",
    "核电": "sytsjz",
    "selfunit": "selfunit",
    "自备": "selfunit",
    "syjzzj": "syjzzj",
    "试验机组": "syjzzj",
}

ACTUAL_EXPORT_TARGET_COLS = [
    "Periodid", "systemload", "dfdcload", "excload",
    "fdload", "gfload", "sytsjz", "selfunit", "syjzzj",
]


def _maybe_transpose_export(df) -> pd.DataFrame:
    """处理平台 exportsj 的横向表布局。

    检测条件：列名中第 0 个含指标名（如「统调负荷（MW）」，匹配
    ACTUAL_EXPORT_COLUMN_ALIASES），其余列全是 Unnamed: N（pandas 对无表头
    列自动命名）。此时结构是「指标名 | 96 个时段值」，转置成
    「period | 指标值」的标准长表。

    转置后：
      * 第一列变成 `Periodid`（00:15, 00:30, ..., 24:00）
      * 后续列是指标名
    """
    import pandas as _pd

    cols = [str(c).strip() for c in df.columns]
    # 是否「首个列名是指标名 + 其余全是 Unnamed」
    first_is_metric = any(
        alias in cols[0].lower()
        for alias in ("systemload", "统调负荷", "直调负荷", "dfdcload", "地方电厂",
                      "excload", "联络线", "fdload", "风电", "gfload", "光伏",
                      "sytsjz", "核电", "selfunit", "自备", "syjzzj", "试验机组")
    ) if cols else False
    rest_unnamed = all(c.startswith("Unnamed") for c in cols[1:]) if len(cols) > 1 else False

    if not (first_is_metric and rest_unnamed):
        return df

    # 横向表：df 是 1 行（只有这一行指标有值）或多行（多个指标），
    # 每行 = [指标名, 时段1值, 时段2值, ..., 时段96值]
    import io as _io

    records = []
    for _, row in df.iterrows():
        metric = str(row.iloc[0]).strip() if _pd.notna(row.iloc[0]) else ""
        if not metric:
            continue
        # 96 个时段值
        vals = row.iloc[1:].tolist()
        for i, v in enumerate(vals, start=1):
            if i > 96:
                break
            mm = (i * 15) % 60
            hh = (i * 15) // 60
            pid = "24:00" if i == 96 else f"{hh:02d}:{mm:02d}"
            records.append({"Periodid": pid, metric: parse_number(v)})

    if not records:
        return df

    tdf = _pd.DataFrame(records)
    # 转置后列名 = 指标名（如「统调负荷（MW）」），归一化为标准列名
    # （systemload/dfdcload/...），供下游 _parse_exported_actual_file 识别。
    rename = {}
    for col in tdf.columns:
        if col == "Periodid":
            continue
        key = str(col).lower().replace(" ", "").replace("(", "(").replace(")", ")")
        for alias, std in ACTUAL_EXPORT_COLUMN_ALIASES.items():
            if alias in key:
                rename[col] = std
                break
    tdf = tdf.rename(columns=rename)
    return tdf


def _parse_exported_actual_file(raw: bytes) -> list[dict[str, Any]]:
    """解析「导出实际」返回的文件字节为 96 点实际值行。

    兼容三种格式（按魔数识别）：
      * ``PK``       -> xlsx（新 Excel，zip 容器）
      * ``D0CF11E0`` -> xls（旧 Excel，OLE2 二进制；平台导出实际常返回这种）
      * 其他可解码   -> csv（utf-8-sig / gbk）

    返回 ``[{'Periodid': ..., 'systemload': ..., ...}, ...]``。

    若解析失败或列不全，抛 ``ValueError`` —— 由调用方决定重试/告警。
    """
    import io

    import pandas as pd

    if not raw:
        raise ValueError("导出实际响应为空字节")

    # ---- 识别文件类型 ----
    df: pd.DataFrame
    magic = raw[:8]
    if raw[:2] == b"PK":
        # xlsx (zip 魔数)
        df = pd.read_excel(io.BytesIO(raw))
    elif magic[:4] == b"\xd0\xcf\x11\xe0":
        # OLE2 旧版 .xls —— 平台「导出实际」常见格式
        df = pd.read_excel(io.BytesIO(raw))  # pandas 底层走 xlrd/calamine
    else:
        # csv —— 先尝试 utf-8-sig，再 gbk
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gbk")
            except UnicodeDecodeError:
                raise ValueError(
                    f"导出实际响应既不是 xlsx/xls 也不是可解码 csv "
                    f"(魔数={magic.hex()})"
                )
        df = pd.read_csv(io.StringIO(text))

    if df is None or df.empty:
        raise ValueError("导出实际文件解析为空表")

    # ---- 转置布局检测：平台 exportsj 导出「负荷信息」是横向表 ----
    # 结构：第0列=指标名（如「统调负荷（MW）」），第1..96列=96个时段数值。
    # pandas 默认读成 header=第0行（指标名 + Unnamed: 1..96）+ 数据行。
    # 检测到「第一个列名含指标名 + 其余列全是 Unnamed」时，转置为标准 96 点布局。
    df = _maybe_transpose_export(df)

    # ---- 列归一化（中文列名/别名 → 标准列名）----
    df.columns = [str(c).strip() for c in df.columns]
    rename: dict[str, str] = {}
    for col in df.columns:
        key = col.lower().replace(" ", "").replace("(", "(").replace(")", ")")
        for alias, std in ACTUAL_EXPORT_COLUMN_ALIASES.items():
            if alias in key:
                rename[col] = std
                break
    df = df.rename(columns=rename)

    # 诊断：打印转置后的完整列结构，便于确认平台导出多少指标
    logger.info("exportsj 解析后列: %s (行数=%d)", list(df.columns), len(df))

    missing = [c for c in ACTUAL_EXPORT_TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"导出实际文件缺少列: {missing}; 实际列: {list(df.columns)}"
        )

    # ---- 转行结构（跳过可能的表头/合计行）----
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        pid = str(r["Periodid"]).strip()
        if not pid:
            continue
        # 仅保留形如 00:15 / 24:00 / 00:15:00 的时段标签
        if not re.match(r"^\d{1,2}(:\d{2}){1,2}$", pid):
            continue
        row: dict[str, Any] = {"Periodid": pid}
        for col in ACTUAL_EXPORT_TARGET_COLS[1:]:
            row[col] = parse_number(r.get(col))
        rows.append(row)

    if not rows:
        raise ValueError("导出实际文件未解析出任何有效时段行")

    # 排序（00:15 第一个，24:00 最后）
    rows.sort(key=lambda x: period_no_from_time(x["Periodid"]))
    return rows


def parse_number(val: Any) -> Optional[float]:
    """解析数值，去除千分位逗号"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", "").replace(" ", "").strip()
    if not s or s in ("-", "--", "null", "None", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


class PmosCrawler:
    """国网 PMOS 爬虫"""

    def __init__(
        self,
        base_url: str = "https://pmos.sd.sgcc.com.cn:18080/trade",
        cookie: str = "",
        unit_id: str = "",
        browser_debug_port: Optional[int] = None,
        data_api_mode: str = "legacy",
        qctc_auth_wait_sec: float = 90.0,
        reporter=None,
    ):
        self.base_url = base_url.rstrip("/")
        self.unit_id = unit_id
        self.browser_debug_port = browser_debug_port
        self.data_api_mode = str(data_api_mode or "legacy").strip().lower()
        self.qctc_auth_wait_sec = max(0.0, float(qctc_auth_wait_sec or 0.0))
        self.reporter = reporter
        self.market_date: Optional[str] = None
        self.qctc_api_base = "https://pmos.sd.sgcc.com.cn:18080/qctc/qctc_pm_trade_outside"
        self.qctc_forecast_page = "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/informationDisclosure/forecast10424"
        self.qctc_actual_tmp_page = "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/informationDisclosure/actualTemporary10425"
        self.qctc_actual_page = "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/informationDisclosure/actual10426"
        self.qctc_boundary_page = "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/informationDisclosure/forecastBoundary10427"

        self.session = requests.Session()
        # 忽略系统代理，强制直连（PMOS 为国网内网站点；本机 Clash 等
        # 代理会导致 ProxyError / [ASN1: NOT_ENOUGH_DATA] 握手失败）
        self.session.trust_env = False
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "X-Requested-With": "XMLHttpRequest",
            "Connection": "keep-alive",
            # PMOS 认证头（与 auto_crawler_v2 一致；服务器据此类头识别登录态，
            # 缺失会导致 index.do 返回 135 字节登录跳转页 → CSRF 获取失败）
            "X-Ticket": "undefined",
            "X-Token": "null",
            "ClientTag": "OUTNET_BROWSE",
            "CurrentRoute": "/outNet",
            "Origin": "https://pmos.sd.sgcc.com.cn",
            "Referer": "https://pmos.sd.sgcc.com.cn/",
        })

        # 解析 Cookie 字符串为 requests CookieJar
        for item in cookie.split(";"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                self.session.cookies.set(k.strip(), v.strip())

        self.csrf_token: Optional[str] = None
        self._verify = False  # 禁用 SSL 验证（国网内网证书问题）

    def _report_event(self, level: str, code: str, message: str, **details: Any) -> None:
        if self.reporter is not None:
            self.reporter.event(level, code, message, **details)

    def _abort_qctc_auth(self, reason: str, **details: Any) -> None:
        """记录认证失败并终止本轮，禁止后续日期继续重复请求。"""
        logger.error("QCTC 认证失败，终止本轮运行 reason=%s", reason)
        self._report_event("ERROR", "QCTC_AUTH_REJECTED", "QCTC认证被拒绝，终止本轮运行",
                           reason=reason, **details)
        if self.reporter is not None:
            stage_details = {key: value for key, value in details.items() if key != "status"}
            self.reporter.stage("qctc_context", "FAIL", reason=reason, **stage_details)
            self.reporter.finish("FAIL", qctc_auth=False, reason=reason)
        raise QCTCAuthRejected(reason)

    # ------------------------------------------------------------------
    #  底层请求
    # ------------------------------------------------------------------

    def _req(self, method: str, url: str, **kwargs) -> requests.Response:
        # 内网响应快；短超时让非内网/失效代理场景快速失败而非干等
        kwargs.setdefault("timeout", 15)
        kwargs.setdefault("verify", self._verify)
        headers = kwargs.pop("headers", {})
        if method.upper() == "POST" and self.csrf_token:
            headers.setdefault("x-csrf-token", self.csrf_token)
        kwargs["headers"] = headers

        try:
            resp = self.session.request(method, url, **kwargs)
        except Exception as e:
            self._report_event("ERROR", "HTTP_REQUEST_FAIL", str(e), method=method.upper(), url=url[:240])
            if self.browser_debug_port:
                logger.warning(
                    "requests 请求失败，改用浏览器 fetch 兜底: %s %s -> %s",
                    method.upper(),
                    url,
                    e,
                )
                return self._browser_req(method, url, **kwargs)
            raise

        if resp.status_code == 502 and self.browser_debug_port:
            self._report_event("WARN", "HTTP_502", "requests返回502，切换浏览器上下文",
                               method=method.upper(), url=url[:240])
            logger.warning(
                "接口返回502，暂不判定为Cookie过期，改用浏览器上下文请求: %s %s",
                method.upper(), url,
            )
            return self._browser_req(method, url, **kwargs)
        if resp.status_code in (302, 401, 403):
            raise requests.HTTPError(
                f"认证失败 (HTTP {resp.status_code})，Cookie 可能已过期",
                response=resp,
            )
        resp.raise_for_status()
        self._report_event("INFO", "HTTP_RESPONSE", "接口请求完成", method=method.upper(),
                           url=url[:240], status=resp.status_code, length=len(resp.text or ""))
        if self.browser_debug_port and self._looks_bad_json_response(resp, headers):
            logger.warning(
                "requests 返回疑似非数据响应，改用浏览器 fetch 兜底: %s %s status=%s len=%s",
                method.upper(),
                resp.url,
                resp.status_code,
                len(resp.text or ""),
            )
            return self._browser_req(method, url, **kwargs)
        return resp

    def _page_get(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        referer: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> requests.Response:
        """访问 HTML 页面入口。

        注意：页面入口不能带 X-Requested-With: XMLHttpRequest，也不能用
        application/json 的 Accept。否则该站可能返回自定义 911 空响应。
        """
        old_xrw = self.session.headers.pop("X-Requested-With", None)
        old_accept = self.session.headers.get("Accept")
        try:
            headers = {
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/webp,image/apng,*/*;q=0.8"
                ),
                "Upgrade-Insecure-Requests": "1",
            }
            if referer:
                headers["Referer"] = referer
            if extra_headers:
                headers.update(extra_headers)
            resp = self.session.get(
                url,
                params=params,
                timeout=60,
                verify=self._verify,
                allow_redirects=True,
                headers=headers,
            )
            logger.info(
                "页面访问: status=%s url=%s len=%s content_type=%s server=%s browser_fallback=%s",
                resp.status_code, resp.url, len(resp.text or ""),
                resp.headers.get("Content-Type", ""), resp.headers.get("Server", ""),
                bool(self.browser_debug_port),
            )
            if resp.status_code == 502 and self.browser_debug_port:
                logger.warning("交易页面返回502，尝试使用已登录浏览器上下文重新访问: %s", url)
                return self._browser_req("GET", url, params=params, headers=headers)
            return resp
        finally:
            if old_xrw is not None:
                self.session.headers["X-Requested-With"] = old_xrw
            if old_accept is not None:
                self.session.headers["Accept"] = old_accept

    def _looks_bad_json_response(self, resp: requests.Response, headers: dict) -> bool:
        """判断接口是否返回了空串/HTML 登录页，而不是 JSON 数据。"""
        accept = (headers.get("Accept") or self.session.headers.get("Accept") or "").lower()
        if "json" not in accept and "javascript" not in accept:
            return False
        text = (resp.text or "").strip()
        if not text:
            return True
        low = text[:300].lower()
        if low.startswith("<!doctype") or low.startswith("<html"):
            return True
        if "请输入账号" in text or "滑块" in text or "login" in low:
            return True
        return False

    def _browser_req(self, method: str, url: str, **kwargs) -> requests.Response:
        """通过已登录浏览器上下文执行 fetch，并包装成 requests.Response。

        用途：部分 PMOS 接口会对 Python requests 直接断开连接，但浏览器
        自己发出的 XHR/fetch 可以正常返回。这里作为兜底通道。
        """
        if not self.browser_debug_port:
            raise RuntimeError("browser_debug_port 未配置，无法使用浏览器 fetch")

        method = method.upper()
        params = kwargs.get("params")
        data = kwargs.get("data")
        headers = dict(kwargs.get("headers") or {})
        page_url = str(kwargs.get("page_url") or (self.base_url + "/DaJyjgfbPlantQuery.do?appkey=187"))

        if params:
            sep = "&" if "?" in url else "?"
            url = url + sep + urllib.parse.urlencode(params, doseq=True)

        body = None
        if data is not None:
            if isinstance(data, str):
                body = data
            else:
                body = urllib.parse.urlencode(data, doseq=True)
            headers.setdefault("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")

        headers.setdefault("Accept", "application/json, text/javascript, */*; q=0.01")
        headers.setdefault("X-Requested-With", "XMLHttpRequest")

        result = _browser_fetch(
            self.browser_debug_port,
            page_url=page_url,
            url=url,
            method=method,
            headers=headers,
            body=body,
        )

        resp = requests.Response()
        resp.status_code = int(result.get("status") or 0)
        resp.url = str(result.get("url") or url)
        text = str(result.get("text") or "")
        resp._content = text.encode("utf-8", errors="replace")
        resp.encoding = "utf-8"
        resp.headers["content-type"] = str(result.get("contentType") or "")
        resp.reason = str(result.get("statusText") or "")[:300]
        if resp.status_code >= 400 or resp.status_code == 0:
            error_text = "authentication_rejected" if resp.status_code == 401 else text[:300]
            logger.error(
                "浏览器 fetch 失败: status=%s statusText=%s text=%s",
                resp.status_code,
                str(result.get("statusText") or "")[:300],
                error_text,
            )
        else:
            logger.info("浏览器 fetch 成功: %s %s status=%s len=%s", method, url, resp.status_code, len(text))
        return resp

    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    # ------------------------------------------------------------------
    #  认证
    # ------------------------------------------------------------------

    def fetch_csrf_token(self) -> bool:
        """从主页提取 CSRF token"""
        if self.data_api_mode == "qctc":
            if not self.browser_debug_port:
                logger.error("QCTC 数据接口必须使用已登录浏览器的 debug_port")
                self._abort_qctc_auth("debug_port_missing")
            # 新 QCTC API 使用 Authorization Bearer token，不使用旧 /trade CSRF。
            # 旧门户 Cookie 登录成功后，QCTC 仍可能没有建立自己的 18080
            # origin/sessionStorage 上下文；先显式等待并审计这一状态，避免
            # 后面四套接口全部以 status=0 静默失败。
            logger.info("QCTC 模式：跳过旧 /trade CSRF，先建立并检查 QCTC 浏览器认证上下文")
            ready = self.ensure_qctc_context()
            if not ready:
                # 没有 Bearer 时不要让主循环进入几十个日期；SystemExit
                # 绕过 crawl_one_day 的单日期 Exception fallback。
                self._abort_qctc_auth("context_missing")
            return True
        try:
            # 先访问门户页面，再访问主页。HAR 里浏览器是先进入 appkey=187 页面，
            # 随后再加载 main/index.do?ticket=...
            entry = self._page_get(
                f"{self.base_url}/DaJyjgfbPlantQuery.do",
                params={"appkey": "187"},
            )
            resp = self._page_get(
                f"{self.base_url}/main/index.do",
                referer=entry.url,
            )
            html = resp.text

            patterns = [
                r'name="_csrf".*?content="([^"]+)"',
                r"name=\"_csrf\".*?content='([^']+)'",
                r"csrfToken\s*[=:]\s*['\"]([^'\"]+)['\"]",
            ]
            for pat in patterns:
                m = re.search(pat, html, re.IGNORECASE)
                if m:
                    self.csrf_token = m.group(1)
                    logger.info("CSRF token acquired from HTML")
                    return True

            for c in self.session.cookies:
                if "csrf" in c.name.lower():
                    self.csrf_token = c.value
                    logger.info("CSRF token acquired from cookie: %s", c.name)
                    return True

            logger.warning("CSRF token not found in any location")
            if entry.status_code == 502 or resp.status_code == 502:
                if self.browser_debug_port:
                    logger.warning(
                        "交易入口为502且已连接浏览器调试端口；跳过CSRF硬失败，"
                        "后续接口启用浏览器fetch兜底"
                    )
                    return True
                logger.warning(
                    "交易入口返回HTTP 502，属于网关/网络问题，不等同于Cookie过期；"
                    "请检查PMOS入口和公司网络"
                )
            # 诊断：index.do 返回长度很短（如 135）通常是登录跳转页，cookie 已过期
            logger.warning(
                "CSRF 获取失败 — index.do 响应长度=%s（正常登录后应 >1000；"
                "<=200 通常为登录跳转页，说明 config.json 的 cookie 已过期，"
                "请重新从浏览器复制最新 Cookie）",
                len(html or ""),
            )
            # 详细诊断：打印响应内容前 400 字符，定位 135 字节到底是跳转页/错误页
            if html:
                logger.warning("index.do 响应内容前 400 字符: %s", html[:400].replace("\n", " "))
            logger.warning("当前会话 cookie 名: %s", [c.name for c in self.session.cookies])
            return False
        except Exception as e:
            logger.warning("fetch_csrf_token failed: %s", e)
            return False

    def ensure_qctc_context(self) -> bool:
        """先完成门户到新版 QCTC 的 ticket SSO，再确认浏览器上下文可用。

        门户 Cookie 与 QCTC 的 sessionStorage token 是两层认证。这里在门户
        target 内 POST 获取 ticket，再打开 SSOLogin?ticket=...；随后可能在
        同一标签页或新标签页落到 QCTC，因此每轮都重新扫描 DevTools targets。
        """
        if not self.browser_debug_port:
            raise RuntimeError("QCTC 浏览器认证上下文检查需要 debug_port")

        pages = _page_targets(self.browser_debug_port)
        if not pages:
            raise RuntimeError(f"DevTools port={self.browser_debug_port} 没有可用页面")

        qctc_pages = [p for p in pages if _is_qctc_context_page_url(str(p.get("url") or ""))]
        if qctc_pages:
            page = qctc_pages[0]
        else:
            # 优先复用已登录门户；没有顶层门户页时才退回当前 PMOS 页面。
            page = next(
                (
                    p for p in pages
                    if urllib.parse.urlsplit(str(p.get("url") or "")).netloc.lower()
                    == "pmos.sd.sgcc.com.cn"
                ),
                None,
            )
            if page is None:
                page = _get_pmos_page(self.browser_debug_port)
        cdp = _CdpClient(page["webSocketDebuggerUrl"])
        target = urllib.parse.urlsplit(self.qctc_forecast_page)
        target_origin = QCTC_ORIGIN
        last_signature = None
        prompted = False
        fallback_used = False
        cdp_failures = 0
        last_reported_target = None
        ignored_qctc_targets: set[str] = set()
        started = time.monotonic()

        def bind_page(new_page: dict[str, Any]) -> bool:
            """切换到新 target；返回是否真的更换了 websocket。"""
            nonlocal cdp, page
            old_id = page.get("id") or page.get("webSocketDebuggerUrl")
            new_id = new_page.get("id") or new_page.get("webSocketDebuggerUrl")
            if old_id == new_id:
                return False
            cdp.close()
            page = new_page
            cdp = _CdpClient(page["webSocketDebuggerUrl"])
            cdp.call("Page.enable")
            return True

        def report_target_found(target_page: dict[str, Any], *, same_tab: bool) -> None:
            url = _safe_url_for_log(str(target_page.get("url") or ""))
            logger.info("QCTC target 已发现 same_tab=%s url=%s", same_tab, url)
            self._report_event(
                "INFO", "QCTC_TARGET_FOUND", "已发现 QCTC 浏览器 target",
                target_url=url, same_tab=same_tab,
            )

        try:
            cdp.call("Page.enable")
            initial = _browser_page_state(cdp)
            initial_url = str(initial.get("url") or "")
            initial_parts = urllib.parse.urlsplit(initial_url)
            already_ready = bool(
                initial.get("bearer_present")
                and initial_parts.netloc == target.netloc
                and initial.get("ready") in {"interactive", "complete"}
            )
            initial_is_qctc = _is_qctc_context_page_url(initial_url)
            if initial_is_qctc:
                report_target_found(page, same_tab=True)
                last_reported_target = (
                    page.get("id") or page.get("webSocketDebuggerUrl"),
                    initial_url[:240],
                )

                # 9222 上可能残留一个已失效的 QCTC 标签页；只有上下文可用
                # 时才直接复用，否则切回已登录门户再发起真正的 SSO，避免
                # 在旧 QCTC target 上重复导航门户入口。
                if not already_ready and not initial.get("bearer_present"):
                    stale_id = page.get("id") or page.get("webSocketDebuggerUrl")
                    portal_page = next(
                        (
                            p for p in pages
                            if urllib.parse.urlsplit(str(p.get("url") or "")).netloc.lower()
                            == "pmos.sd.sgcc.com.cn"
                        ),
                        None,
                    )
                    if portal_page is not None:
                        ignored_qctc_targets.add(str(stale_id))
                        bind_page(portal_page)
                        initial_url = str(portal_page.get("url") or "")
                        initial_parts = urllib.parse.urlsplit(initial_url)
                        initial = {
                            **initial,
                            "url": initial_url,
                            "origin": f"{initial_parts.scheme}://{initial_parts.netloc}",
                            "path": initial_parts.path,
                            "qctc_route": False,
                            "bearer_present": False,
                        }

            if already_ready:
                elapsed = round(time.monotonic() - started, 3)
                safe_url = _safe_url_for_log(initial_url)
                storage_keys = (
                    list(initial.get("session_storage_keys") or [])[:40]
                    + list(initial.get("local_storage_keys") or [])[:40]
                )
                self._report_event(
                    "INFO", "QCTC_STORAGE_STATE", "复用已有 QCTC storage 上下文",
                    url=safe_url, path=str(initial.get("path") or "")[:180],
                    bearer_present=True,
                    session_storage_keys=list(initial.get("session_storage_keys") or [])[:40],
                    local_storage_keys=list(initial.get("local_storage_keys") or [])[:40],
                )
                self._report_event(
                    "INFO", "QCTC_EXISTING_CONTEXT_READY", "已有 QCTC Bearer 上下文可直接复用",
                    origin=target_origin, path=str(initial.get("path") or "")[:180],
                    elapsed_sec=elapsed, storage_keys=storage_keys,
                )
                self._report_event(
                    "INFO", "QCTC_CONTEXT_READY", "QCTC认证上下文已就绪",
                    origin=target_origin, path=str(initial.get("path") or "")[:180],
                    elapsed_sec=elapsed, storage_keys=storage_keys,
                )
                if self.reporter is not None:
                    self.reporter.stage(
                        "qctc_context", "PASS", origin=target_origin,
                        path=str(initial.get("path") or "")[:180], elapsed_sec=elapsed,
                    )
                return True

            if not already_ready:
                # /psso?service=... 只是门户菜单描述地址，不能直接导航。
                # 这里复现门户点击的真实顺序：门户同源 POST 取 ticket，
                # 再由浏览器打开 SSOLogin?ticket=...。
                portal_page = page if initial_parts.netloc == "pmos.sd.sgcc.com.cn" else next(
                    (
                        p for p in pages
                        if urllib.parse.urlsplit(str(p.get("url") or "")).netloc.lower()
                        == "pmos.sd.sgcc.com.cn"
                    ),
                    None,
                )
                if portal_page is None:
                    message = "没有可用的 PMOS 门户 target，无法获取 QCTC SSO ticket"
                    logger.error(message)
                    self._report_event("ERROR", "QCTC_TICKET_REQUEST_FAIL", message,
                                       http_status=0, business_status=None)
                    return False
                if (page.get("id") or page.get("webSocketDebuggerUrl")) != (
                    portal_page.get("id") or portal_page.get("webSocketDebuggerUrl")
                ):
                    stale_id = page.get("id") or page.get("webSocketDebuggerUrl")
                    ignored_qctc_targets.add(str(stale_id))
                    bind_page(portal_page)
                    initial_url = str(portal_page.get("url") or "")
                    initial_parts = urllib.parse.urlsplit(initial_url)

                logger.info("QCTC ticket 请求开始 portal_path=%s", initial_parts.path)
                self._report_event(
                    "INFO", "QCTC_TICKET_REQUEST_START", "在门户上下文请求 QCTC SSO ticket",
                    portal_origin=initial_parts.netloc,
                    portal_path=initial_parts.path,
                )
                ticket_result = _open_qctc_via_portal_ticket(cdp)
                if ticket_result.get("ok"):
                    logger.info(
                        "QCTC ticket 请求成功 http_status=%s business_status=%s",
                        ticket_result.get("http_status"), ticket_result.get("business_status"),
                    )
                    self._report_event(
                        "INFO", "QCTC_TICKET_REQUEST_OK", "QCTC SSO ticket 请求成功",
                        http_status=ticket_result.get("http_status"),
                        business_status=ticket_result.get("business_status"),
                    )
                    self._report_event(
                        "INFO", "QCTC_SSO_WINDOW_OPENED", "已打开 QCTC SSOLogin target",
                        opened_new_window=ticket_result.get("opened_new_window", False),
                    )
                else:
                    logger.warning(
                        "QCTC ticket 请求失败 http_status=%s business_status=%s message=%s",
                        ticket_result.get("http_status"), ticket_result.get("business_status"),
                        ticket_result.get("message"),
                    )
                    self._report_event(
                        "WARN", "QCTC_TICKET_REQUEST_FAIL", "QCTC SSO ticket 请求失败",
                        http_status=ticket_result.get("http_status"),
                        business_status=ticket_result.get("business_status"),
                        ticket_message=ticket_result.get("message"),
                    )

            deadline = started + self.qctc_auth_wait_sec
            while True:
                try:
                    live_pages = _page_targets(self.browser_debug_port)
                    live_qctc = next(
                        (
                            p for p in live_pages
                            if _is_qctc_context_page_url(str(p.get("url") or ""))
                            and str(p.get("id") or p.get("webSocketDebuggerUrl"))
                            not in ignored_qctc_targets
                        ),
                        None,
                    )
                    if live_qctc is not None:
                        same_tab = (
                            (page.get("id") or page.get("webSocketDebuggerUrl"))
                            == (live_qctc.get("id") or live_qctc.get("webSocketDebuggerUrl"))
                        )
                        if not same_tab:
                            bind_page(live_qctc)
                        target_identity = (
                            live_qctc.get("id") or live_qctc.get("webSocketDebuggerUrl"),
                            str(live_qctc.get("url") or "")[:240],
                        )
                        if target_identity != last_reported_target:
                            report_target_found(live_qctc, same_tab=same_tab)
                            last_reported_target = target_identity
                    state = _browser_page_state(cdp)
                    cdp_failures = 0
                except Exception as exc:
                    cdp_failures += 1
                    if cdp_failures >= 3:
                        message = "QCTC CDP连接已中断：浏览器窗口或调试端口已不可用"
                        logger.error("%s last_error=%s", message, exc)
                        self._report_event(
                            "ERROR", "QCTC_CDP_DISCONNECTED", message,
                            failures=cdp_failures, error_type=type(exc).__name__,
                        )
                        if self.reporter is not None:
                            self.reporter.stage(
                                "qctc_context", "FAIL", error=message,
                                error_type=type(exc).__name__, failures=cdp_failures,
                            )
                        raise RuntimeError(message) from exc
                    state = {
                        "url": "", "origin": "", "path": "", "ready": "unknown",
                        "qctc_route": False, "bearer_present": False,
                        "session_storage_keys": [], "local_storage_keys": [],
                    }
                    logger.warning("QCTC 上下文读取失败，将继续等待: %s", exc)

                signature = (
                    str(state.get("url") or "")[:240],
                    str(state.get("ready") or ""),
                    bool(state.get("qctc_route")),
                    bool(state.get("bearer_present")),
                    tuple(state.get("session_storage_keys") or []),
                    tuple(state.get("local_storage_keys") or []),
                )
                if signature != last_signature:
                    last_signature = signature
                    safe_state = {
                        "url": _safe_url_for_log(signature[0]),
                        "origin": str(state.get("origin") or "")[:120],
                        "path": str(state.get("path") or "")[:180],
                        "ready": signature[1],
                        "qctc_route": signature[2],
                        "bearer_present": signature[3],
                        "session_storage_keys": list(signature[4])[:40],
                        "local_storage_keys": list(signature[5])[:40],
                    }
                    logger.info("QCTC 上下文状态: %s", safe_state)
                    self._report_event("INFO", "QCTC_CONTEXT_STATE", "QCTC上下文状态变化", **safe_state)
                    self._report_event("INFO", "QCTC_STORAGE_STATE", "QCTC storage 状态变化",
                                       url=safe_state["url"], path=safe_state["path"],
                                       bearer_present=safe_state["bearer_present"],
                                       session_storage_keys=safe_state["session_storage_keys"],
                                       local_storage_keys=safe_state["local_storage_keys"])

                current_origin = str(state.get("origin") or "")
                if (
                    current_origin == target_origin
                    and state.get("ready") in {"interactive", "complete"}
                    and state.get("bearer_present")
                ):
                    elapsed = round(time.monotonic() - started, 3)
                    logger.info("QCTC 浏览器认证上下文就绪 elapsed=%.3fs", elapsed)
                    self._report_event(
                        "INFO", "QCTC_CONTEXT_READY", "QCTC认证上下文已就绪",
                        origin=current_origin,
                        path=str(state.get("path") or "")[:180],
                        elapsed_sec=elapsed,
                        storage_keys=(list(state.get("session_storage_keys") or [])[:40]
                                      + list(state.get("local_storage_keys") or [])[:40]),
                    )
                    if self.reporter is not None:
                        self.reporter.stage(
                            "qctc_context", "PASS",
                            origin=current_origin,
                            path=str(state.get("path") or "")[:180], elapsed_sec=elapsed,
                        )
                    return True

                if not prompted and not state.get("bearer_present"):
                    prompted = True
                    logger.warning(
                        "QCTC 尚未建立 Bearer 上下文。请在当前浏览器窗口从已登录门户进入新版QCTC交易/信息披露页面；"
                        "不要关闭该窗口，程序将继续等待 %.0f 秒。",
                        self.qctc_auth_wait_sec,
                    )
                    self._report_event(
                        "WARN", "QCTC_CONTEXT_ACTION_REQUIRED",
                        "需要从门户菜单进入新版QCTC以完成SSO",
                        wait_sec=self.qctc_auth_wait_sec,
                    )

                if time.monotonic() >= deadline:
                    if not fallback_used:
                        fallback_used = True
                        portal_page = next(
                            (
                                p for p in _page_targets(self.browser_debug_port)
                                if urllib.parse.urlsplit(str(p.get("url") or "")).netloc.lower()
                                == "pmos.sd.sgcc.com.cn"
                            ),
                            None,
                        )
                        if portal_page is not None:
                            same_tab = (
                                (page.get("id") or page.get("webSocketDebuggerUrl"))
                                == (portal_page.get("id") or portal_page.get("webSocketDebuggerUrl"))
                            )
                            if not same_tab:
                                bind_page(portal_page)
                            try:
                                menu_result = _try_portal_qctc_entry(cdp)
                            except Exception as exc:
                                menu_result = {"clicked": False, "error": f"{type(exc).__name__}: {exc}"}
                            logger.info("QCTC SSO DOM fallback result=%s", {
                                "clicked": bool(menu_result.get("clicked")),
                                "text": str(menu_result.get("text") or "")[:100],
                                "href": _safe_url_for_log(str(menu_result.get("href") or "")),
                            })
                            self._report_event(
                                "INFO" if menu_result.get("clicked") else "WARN",
                                "QCTC_SSO_FALLBACK_DOM",
                                "SSO直达未完成，尝试门户菜单兜底",
                                clicked=bool(menu_result.get("clicked")),
                                text=str(menu_result.get("text") or "")[:100],
                                href=_safe_url_for_log(str(menu_result.get("href") or "")),
                                error=str(menu_result.get("error") or "")[:200],
                            )
                            deadline = time.monotonic() + max(30.0, min(60.0, self.qctc_auth_wait_sec or 30.0))
                            prompted = False
                            continue
                    elapsed = round(time.monotonic() - started, 3)
                    message = (
                        "QCTC sessionStorage 未建立 Bearer token(qctc_route=%s)；"
                        "认证上下文不可用，本轮将在进入日期循环前终止。"
                    ) % bool(state.get("qctc_route"))
                    logger.warning("%s elapsed=%.3fs", message, elapsed)
                    self._report_event(
                        "ERROR", "QCTC_CONTEXT_MISSING", message,
                        elapsed_sec=elapsed,
                        current_url=_safe_url_for_log(str(state.get("url") or "")),
                        current_origin=current_origin,
                        current_path=str(state.get("path") or "")[:180],
                        qctc_route=bool(state.get("qctc_route")),
                        bearer_present=bool(state.get("bearer_present")),
                        session_storage_keys=list(state.get("session_storage_keys") or [])[:40],
                        local_storage_keys=list(state.get("local_storage_keys") or [])[:40],
                    )
                    if self.reporter is not None:
                        self.reporter.stage(
                            "qctc_context", "FAIL",
                            note="no bearer token; stop before date loop",
                            current_origin=current_origin,
                            current_path=str(state.get("path") or "")[:180],
                            qctc_route=bool(state.get("qctc_route")),
                            bearer_present=bool(state.get("bearer_present")),
                            elapsed_sec=elapsed,
                        )
                    return False
                time.sleep(0.75)
        finally:
            cdp.close()

    # ------------------------------------------------------------------
    #  日期切换
    # ------------------------------------------------------------------

    def change_date(self, date_str: str) -> bool:
        """切换市场日期"""
        if self.data_api_mode == "qctc":
            self.market_date = date_str
            return True
        try:
            self._req(
                "GET",
                f"{self.base_url}/main/home/changeDate.do",
                params={"pdate": date_str, "_": self._ts()},
            )
            return True
        except Exception as e:
            logger.warning("changeDate(%s) failed: %s", date_str, e)
            return False

    # ------------------------------------------------------------------
    #  数据爬取
    # ------------------------------------------------------------------

    def crawl_market_overview(self) -> list[dict[str, Any]]:
        """爬取全省市场特征 96 点预测数据（日前接口 DaJyxxPlDa，9 列含核电/自备/试验）。"""
        if self.data_api_mode == "qctc":
            return self._crawl_qctc_market("ForecastData", self.qctc_forecast_page, actual=False)
        return self._crawl_market_overview_host("DaJyxxPlDa")

    def crawl_market_forecast_for_date(self, date_str: str) -> list[dict[str, Any]]:
        """Fetch forecast-only market data for one explicit business date."""
        if self.data_api_mode == "qctc":
            return self._crawl_qctc_market(
                "ForecastData", self.qctc_forecast_page, actual=False, market_date=date_str
            )
        original_date = self.market_date
        if not self.change_date(date_str):
            raise RuntimeError(f"change_date({date_str}) failed")
        try:
            return self._crawl_market_overview_host("DaJyxxPlDa")
        finally:
            if original_date and original_date != date_str:
                self.change_date(original_date)

    def crawl_market_boundary_for_date(self, date_str: str) -> list[dict[str, Any]]:
        """独立抓取市场披露边界信息；不与常规 ForecastData 混写。"""
        if self.data_api_mode != "qctc":
            return []
        return self._crawl_qctc_market(
            "ForecastBoundaryData",
            self.qctc_boundary_page,
            actual=False,
            market_date=date_str,
            field_map=self.QCTC_BOUNDARY_MAP,
        )

    def crawl_market_boundary(self) -> list[dict[str, Any]]:
        if not self.market_date:
            raise RuntimeError("边界信息请求前必须调用 change_date(date_str)")
        return self.crawl_market_boundary_for_date(self.market_date)

    # QCTC 信息披露接口：HAR6 已验证为 HTTP 200 + 96 行 TableData。
    # 严格映射到旧标准原始字段，后续 DB 映射继续分别写 fcast_* / actual_*。
    QCTC_FORECAST_MAP = {
        "qwfh": "qwfh",
        "zdfh": "systemload", "dfdczj": "dfdcload", "llxfh": "excload",
        "fd": "fdload", "gf": "gfload", "hd": "sytsjz",
        "zbjz": "selfunit", "syzj": "syjzzj",
    }
    QCTC_ACTUAL_MAP = {
        "qwfh": "qwfh",
        "zdfh": "systemload", "dfdczj": "dfdcload", "llxfh": "excload",
        "fd": "fdload", "gf": "gfload", "cx": "cxload", "hd": "hdload",
        "zbjz": "zbload", "syzj": "syjzload",
    }
    QCTC_BOUNDARY_MAP = {
        "qwfh": "qwfh", "zdfh": "systemload", "llxfh": "excload",
        "fd": "fdload", "gf": "gfload", "hd": "sytsjz",
    }

    def _qctc_get(self, path: str, *, params: dict[str, Any], page_url: str, web_path: str) -> dict[str, Any]:
        if not self.browser_debug_port:
            raise RuntimeError("QCTC 请求需要 browser_debug_port")
        response = self._browser_req(
            "GET", f"{self.qctc_api_base}/{path.lstrip('/')}", params=params,
            page_url=page_url,
            headers={
                "Accept": "application/json, text/plain, */*",
                "X-Web-Path": web_path,
            },
        )
        if response.status_code != 200:
            auth_rejected = response.status_code == 401
            detail = (
                "authentication_rejected"
                if auth_rejected
                else (response.text or str(getattr(response, "reason", "")) or "").strip()[:300]
            )
            self._report_event(
                "ERROR", "QCTC_HTTP_FAIL", detail,
                path=path, status=response.status_code,
                status_text=str(getattr(response, "reason", ""))[:300],
            )
            if auth_rejected:
                self._abort_qctc_auth("http_401", path=path, status=response.status_code)
            raise RuntimeError(f"QCTC {path} HTTP {response.status_code}: {detail}")
        result = response.json()
        result_dict = result if isinstance(result, dict) else {}
        self._report_event("INFO" if result_dict.get("code") == 0 else "ERROR",
                           "QCTC_RESPONSE", "QCTC接口返回", path=path,
                           status=response.status_code, business_code=result_dict.get("code"),
                           business_message=str(result_dict.get("msg") or "")[:200],
                           length=len(response.text or ""))
        if (
            isinstance(result_dict, dict)
            and result_dict.get("code") == 2
            and (
                "登录" in str(result_dict.get("msg") or "")
                or "authentication" in str(result_dict.get("data") or "").lower()
            )
        ):
            self._abort_qctc_auth("business_auth_rejected", path=path, status=response.status_code)
        if not isinstance(result, dict) or result.get("code") != 0:
            raise RuntimeError(f"QCTC {path} 业务失败: {str(result)[:500]}")
        return result

    def _crawl_qctc_market(
        self,
        source: str,
        page_url: str,
        *,
        actual: bool,
        market_date: str | None = None,
        field_map: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        requested_date = market_date or self.market_date
        if not requested_date:
            raise RuntimeError("QCTC 请求前必须指定交易日或调用 change_date(date_str)")
        web_path = urllib.parse.urlsplit(page_url).path
        result = self._qctc_get(
            f"informationDisclosure/{source}/getLoadData",
            params={"pdate": requested_date, "versions": ""},
            page_url=page_url,
            web_path=web_path,
        )
        data = result.get("data") or {}
        table = data.get("TableData") if isinstance(data, dict) else None
        if not isinstance(table, list):
            raise ValueError(f"QCTC {source} 未返回 TableData")
        field_map = field_map or (self.QCTC_ACTUAL_MAP if actual else self.QCTC_FORECAST_MAP)
        rows: list[dict[str, Any]] = []
        for item in table:
            if not isinstance(item, dict):
                continue
            period = item.get("periodname") or item.get("periodid")
            if period in (None, ""):
                continue
            row = {"Periodid": str(period)}
            row.update({target: item.get(source_key, "") for source_key, target in field_map.items()})
            rows.append(row)
        logger.info("QCTC %s market overview: %d rows", source, len(rows))
        return rows

    # 预测接口字段（DaJyxxPlDa）：systemload/dfdcload/excload/fdload/gfload + 核电/自备/试验
    FORECAST_COLUMNS = [
        "Periodid", "systemload", "dfdcload", "excload",
        "fdload", "gfload", "sytsjz", "selfunit", "syjzzj",
    ]

    # 实际接口字段（DaJyxxPlYx）：systemload/dfdcload/excload/fdload/gfload + 抽蓄/核电/自备/试验
    ACTUAL_COLUMNS = [
        "Periodid", "systemload", "dfdcload", "excload",
        "fdload", "gfload", "cxload", "hdload", "zbload", "syjzload",
    ]

    def _crawl_market_overview_host(self, host: str, columns: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """通用：请求某 host 的 getNewDetailGridList。

        host: 'DaJyxxPlDa'(日前,预测9列) / 'DaJyxxPlYx'(实时,实际10列) / 'DaJyxxPlYxTmp'(实时临时)
        columns: 自定义字段集；默认 FORECAST_COLUMNS（预测接口字段）。
        """
        if columns is None:
            columns = self.FORECAST_COLUMNS
        data = {
            "method": "getNewDetailGridList",
            "draw": "1",
            "start": "0",
            "length": "-1",  # -1 → 返回全部
            "search[value]": "",
            "search[regex]": "false",
            "_": self._ts(),
        }
        for i, col in enumerate(columns):
            data[f"columns[{i}][data]"] = col
            data[f"columns[{i}][name]"] = ""
            data[f"columns[{i}][searchable]"] = "true"
            data[f"columns[{i}][orderable]"] = "false"
            data[f"columns[{i}][search][value]"] = ""
            data[f"columns[{i}][search][regex]"] = "false"
        data["order[0][column]"] = "0"
        data["order[0][dir]"] = "asc"

        resp = self._req(
            "POST",
            f"{self.base_url}/{host}.do?method=getNewDetailGridList",
            data=data,
            headers={
                "Referer": f"{self.base_url}/{host}.do",
                "Origin": self.base_url.rsplit("/", 1)[0],
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
        )
        result = resp.json()
        rows = result.get("data", []) if isinstance(result, dict) else result
        logger.info("market_overview: %d rows", len(rows))
        return rows

    def _crawl_datatable(
        self,
        endpoint: str,
        method_name: str,
        columns: list[str],
        *,
        query: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """请求 PMOS DataTables JSON 接口并返回原始行。

        HAR5 证明 PMOS 的备用、检修和断面页面也是标准 DataTables 接口。
        统一封装请求，避免为每个页面复制一套参数拼接代码；该函数只返回
        网站原始字段，不做预测/实际语义转换。
        """
        data: dict[str, Any] = {
            "draw": "1",
            "start": "0",
            "length": "-1",
            "search[value]": "",
            "search[regex]": "false",
            "_": self._ts(),
        }
        for i, col in enumerate(columns):
            data[f"columns[{i}][data]"] = col
            data[f"columns[{i}][name]"] = ""
            data[f"columns[{i}][searchable]"] = "true"
            data[f"columns[{i}][orderable]"] = "false"
            data[f"columns[{i}][search][value]"] = ""
            data[f"columns[{i}][search][regex]"] = "false"

        resp = self._req(
            "POST",
            f"{self.base_url}/{endpoint}",
            params={"method": method_name, **(query or {})},
            data=data,
            headers={
                "Referer": f"{self.base_url}/{endpoint}",
                "Origin": self.base_url.rsplit("/", 1)[0],
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
        )
        result = resp.json()
        rows = result.get("data", []) if isinstance(result, dict) else result
        if not isinstance(rows, list):
            raise ValueError(f"{endpoint}?method={method_name} 返回格式异常")
        logger.info("%s?method=%s: %d rows", endpoint, method_name, len(rows))
        return rows

    def _crawl_raw_json(
        self,
        endpoint: str,
        method_name: str,
        *,
        query: Optional[dict[str, Any]] = None,
    ) -> Any:
        """保存 HAR 中非 DataTables 接口的原始 JSON 结构。"""
        resp = self._req(
            "POST",
            f"{self.base_url}/{endpoint}",
            params={"method": method_name, **(query or {})},
            headers={
                "Referer": f"{self.base_url}/{endpoint}",
                "Origin": self.base_url.rsplit("/", 1)[0],
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
        )
        result = resp.json()
        logger.info("%s?method=%s: raw JSON captured", endpoint, method_name)
        return result

    # 日前二次出清（最终版）保持现有主口径；另外独立抓取一次出清，仅新增一列，
    # 不回填、不替换现有「日前出清价格」。
    QCTC_DA_FIRST_UNIT_PAGE = (
        "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/dayAheadTransaction/"
        "onceCqrqjyjg/result/fdcResultQuery10452"
    )
    QCTC_DA_UNIT_PAGE = (
        "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/dayAheadTransaction/"
        "declare/towFdcResultQuery10454"
    )
    QCTC_RT_UNIT_PAGE = (
        "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/realTimeTransaction/result/formalResultFdc10458"
    )

    def _crawl_qctc_unit_detail(
        self,
        source: str,
        page_url: str,
        *,
        pdate_iso: bool = False,
    ) -> list[dict[str, Any]]:
        """从新版 QCTC 获取机组96点明细。

        抓包只用于确认 getDetail96 的响应字段形状（periodid/cqPrice/
        power/energy/kt/bq），不代表每个日期已经发布完整数据。返回空数组
        或价格不全代表该业务日尚未发布/接口未返回，绝不复制另一套来源。
        """
        if not self.market_date:
            raise RuntimeError("QCTC 机组请求前必须调用 change_date(date_str)")
        web_path = urllib.parse.urlsplit(page_url).path
        if source == "day_ahead_first":
            endpoint = "trade/DaJyjgfbPlantFirQuery/getDetail96"
        elif source == "day_ahead":
            # 二次出清（最终版）：保持现有主口径
            endpoint = "trade/DaJyjgfbPlantQuery/getDetail96"
        elif source == "realtime_tmp":
            endpoint = "YxJyjgfbPlantQueryTmp/getDetail96"
        else:
            endpoint = "YxJyjgfbPlantQuery/getDetail96"
        pdate = self.market_date + ("T00:00:00.000Z" if pdate_iso else "")
        result = self._qctc_get(
            endpoint,
            params={"pdate": pdate, "unitid": self.unit_id},
            page_url=page_url,
            web_path=web_path,
        )
        payload = result.get("data") or {}
        raw_rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(raw_rows, list):
            raw_rows = []
        rows: list[dict[str, Any]] = []
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            label = item.get("periodid") or item.get("Periodid")
            if label in (None, ""):
                continue
            rows.append({
                "periodid": str(label),
                "cqPrice": item.get("cqPrice", ""),
                "power": item.get("power", ""),
                "energy": item.get("energy", ""),
                "kt": item.get("kt", ""),
                "bq": item.get("bq", ""),
            })
        logger.info("QCTC %s unit detail: %d rows", source, len(rows))
        return rows

    def _crawl_qctc_realtime_unit_detail(self) -> list[dict[str, Any]]:
        rows = self._crawl_qctc_unit_detail(
            "realtime", self.QCTC_RT_UNIT_PAGE, pdate_iso=False
        )
        if rows:
            return rows
        # 正式实时结果未发布时，页面会再请求临时结果；仍保持实时来源，
        # 不会回退到日前接口。
        tmp_page = (
            "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/realTimeTransaction/"
            "result/rtTmpFdcQuery10456"
        )
        try:
            return self._crawl_qctc_unit_detail(
                "realtime_tmp", tmp_page, pdate_iso=False
            )
        except Exception as exc:
            logger.info("QCTC realtime temporary unit detail unavailable: %s", exc)
            return []

    def _crawl_unit_detail(self, endpoint: str) -> list[dict[str, Any]]:
        """爬取机组级 96 点明细数据（日前/实时共用）"""
        # 保留 HAR5 中出现的全部机组级字段；不存在的字段由服务器返回空值，
        # 绝不拿日前值填充实时值，也不拿实时值填充日前值。
        cols = [
            "periodid", "power", "energy", "cqPrice", "bq", "kt",
            "jsprice", "price", "powerstate", "tsjzbq", "custprice",
        ]
        parts = [f"method=getDetail", f"unitid={self.unit_id}", "draw=1"]
        for i, c in enumerate(cols):
            parts.append(
                f"columns[{i}][data]={c}"
                f"&columns[{i}][searchable]=true"
                f"&columns[{i}][orderable]=false"
                f"&columns[{i}][search][value]="
                f"&columns[{i}][search][regex]=false"
            )
        parts.append("start=0&length=-1")
        parts.append("search[value]=&search[regex]=false")
        parts.append(f"_={self._ts()}")

        url = f"{self.base_url}/{endpoint}?{'&'.join(parts)}"
        resp = self._req("GET", url)
        result = resp.json()
        rows = result.get("data", []) if isinstance(result, dict) else result
        logger.info("%s: %d rows", endpoint.split(".")[0], len(rows))
        return rows

    def crawl_day_ahead_first(self) -> list[dict[str, Any]]:
        """日前一次出清 96 点；仅作为独立补充列，不替换二次出清。"""
        if self.data_api_mode == "qctc":
            return self._crawl_qctc_unit_detail(
                "day_ahead_first", self.QCTC_DA_FIRST_UNIT_PAGE, pdate_iso=False
            )
        return self._crawl_unit_detail("DaJyjgfbPlantFirQuery.do")

    def crawl_day_ahead(self) -> list[dict[str, Any]]:
        """日前二次出清（最终版）价格/出力 96 点"""
        if self.data_api_mode == "qctc":
            return self._crawl_qctc_unit_detail(
                "day_ahead", self.QCTC_DA_UNIT_PAGE, pdate_iso=False
            )
        return self._crawl_unit_detail("DaJyjgfbPlantQuery24.do")

    def crawl_realtime(self) -> list[dict[str, Any]]:
        """实时出清价格/出力 96 点"""
        if self.data_api_mode == "qctc":
            return self._crawl_qctc_realtime_unit_detail()
        return self._crawl_unit_detail("YxJyjgfbPlantQuery24.do")

    def crawl_optional_market_data(self) -> dict[str, Any]:
        """爬取 HAR5 中发现的附加市场信息。

        这些数据不直接写入 actual/fcast 核心列，先按原始字段保存，避免
        把日级容量、备用或检修记录错误广播成 96 点序列。
        """
        out: dict[str, Any] = {}
        if self.data_api_mode == "qctc":
            # QCTC 主链路已经覆盖当前 canonical 表所需的四类核心数据；
            # 下面的旧 /trade DataTables 接口在公司网络会统一返回 nginx 502，
            # 继续调用只会制造噪声和延迟，不能作为 QCTC 的回退来源。
            logger.info("QCTC 模式：跳过旧 /trade 附加接口（避免502噪声），仅保存核心QCTC数据")
            self._report_event(
                "INFO", "OPTIONAL_LEGACY_SKIPPED",
                "QCTC模式跳过旧版附加接口",
                reason="legacy_trade_gateway_502",
            )
            return out
        queries = {
            "reserve_da": (
                "DaJyxxPlDa.do",
                "getByList",
                ["NUM", "DATE", "PERIODID", "TYPE", "ZBY"],
                {"type": "正备用"},
            ),
            "maintenance_da": (
                "DaJyxxPlDa.do",
                "getjzjxList",
                ["NUM", "DATE", "DA_CAPACITY"],
                {},
            ),
            "maintenance_rt": (
                "DaJyxxPlYxTmp.do",
                "getjzjxList",
                ["NUM", "DATE", "YX_CAPACITY"],
                {"type": "全部"},
            ),
            "substation_outage": (
                "DaJyxxPlDa.do",
                "getSbdYearList",
                ["num", "date", "eq", "devicetype", "begintime", "endtime"],
                {},
            ),
            "pumped_storage_da": (
                "DaJyxxPlDa.do",
                "getTsjzList",
                ["NUM", "DATE", "TYPE", "OPENCAPACITY", "CLOSECAPACITY"],
                {},
            ),
        }
        for name, (endpoint, method_name, columns, query) in queries.items():
            try:
                out[name] = self._crawl_datatable(endpoint, method_name, columns, query=query)
            except Exception as exc:
                logger.warning("附加接口 %s 获取失败（不伪造数据）: %s", name, exc)
                out[name] = []
        # 这些页面返回的是图表对象/价格曲线，不强行压扁成 96 点列；原样保存，
        # 以后如果确认字段语义和时段映射，再由特征工程显式消费。
        raw_queries = {
            "market_chart_da": ("DaJyxxPlDa.do", "getChart", {}),
            "market_chart_rt": ("DaJyxxPlYx.do", "getChart", {}),
            "tie_line_chart_da": ("DaJyxxPlDa.do", "getLlxChart", {}),
            "tie_line_chart_rt": ("DaJyxxPlYxTmp.do", "getLlxChart", {}),
            "system_comparison_chart": ("XxplBjQuery.do", "getChart", {}),
            "system_comparison_tie_line_chart": ("XxplBjQuery.do", "getLlxChart", {}),
            "day_ahead_price_curve": (
                "DaJyjgfbPlantQuery24.do", "getPowerPrice", {"unitid": self.unit_id}
            ),
            "realtime_price_curve": (
                "YxJyjgfbPlantQuery24.do", "getPowerPrice", {"unitid": self.unit_id}
            ),
            "price_24_detail": ("JyjgPriceQuery.do", "getDetail", {}),
        }
        for name, (endpoint, method_name, query) in raw_queries.items():
            try:
                out[name] = self._crawl_raw_json(endpoint, method_name, query=query)
            except Exception as exc:
                logger.warning("附加图表接口 %s 获取失败（不伪造数据）: %s", name, exc)
                out[name] = []

        table_queries = {
            "tie_line_detail_da": (
                "DaJyxxPlDa.do", "getNewLlxDetailGridList",
                ["Periodid", "value0", "value1", "value2", "value3", "value4"], {},
            ),
            "tie_line_detail_rt": (
                "DaJyxxPlYxTmp.do", "getNewLlxDetailGridList",
                ["Periodid", "value0", "value1", "value2", "value3", "value4"], {},
            ),
            "section_constraints_da": (
                "DaJyxxPlDa.do", "getDmxxList",
                ["speriodid", "SYSTEMLDAT", "SYSTEMDART", "SYSTEMSJFZ"],
                {"type": "2490"},
            ),
            "system_comparison_96": (
                "XxplBjQuery.do", "getTableDate",
                ["Periodid", "qwload", "zdload", "llxload", "fdload", "gfload"], {},
            ),
        }
        for name, (endpoint, method_name, columns, query) in table_queries.items():
            try:
                out[name] = self._crawl_datatable(endpoint, method_name, columns, query=query)
            except Exception as exc:
                logger.warning("附加表格接口 %s 获取失败（不伪造数据）: %s", name, exc)
                out[name] = []
        return out

    def crawl_market_overview_actual_final(self) -> list[dict[str, Any]]:
        """正式实际 RealityData；不回退临时实际。"""
        if self.data_api_mode == "qctc":
            return self._crawl_qctc_market("RealityData", self.qctc_actual_page, actual=True)
        return self._crawl_market_actual_via_json()

    def crawl_market_overview_actual_temporary(self) -> list[dict[str, Any]]:
        """临时实际 RealityTmpData；与正式实际并行采集、独立保存。"""
        if self.data_api_mode != "qctc":
            return []
        return self._crawl_qctc_market(
            "RealityTmpData", self.qctc_actual_tmp_page, actual=True
        )

    def crawl_market_overview_actual(
        self,
        *,
        export_type: Optional[str] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> list[dict[str, Any]]:
        """爬取全省市场特征 96 点「实际值」。

        主路径：调 **`DaJyxxPlYx.do?method=getNewDetailGridList`**（实时接口）。
        经 HAR 实证：该接口返回的全部 8 个特征（systemload/dfdcload/excload/
        fdload/gfload/sytsjz/selfunit/syjzzj）是**真实实际值**——其 systemload
        与平台「导出实际」（exportsj）导出的统调负荷实际值**逐点一致**。
        而 `DaJyxxPlDa.do`（日前，现有爬虫用的）返回的是**预测值**——这正是
        历史 actual=fcast 拷贝的根源。

        回退：若 JSON 接口失败，回退到 exportsj（.xls 导出）解析。

        调用前必须先 ``change_date(date_str)`` 切到目标日期。

        Returns
        -------
        list[dict] —— 与 ``crawl_market_overview()`` 同结构的 96 行
        （键：Periodid + systemload + dfdcload + excload + fdload + gfload
         + sytsjz + selfunit + syjzzj）。
        """
        if raw_bytes is not None:
            # 显式传入原始响应（诊断/测试用）→ 走 .xls 解析
            return _parse_exported_actual_file(raw_bytes)

        if self.data_api_mode == "qctc":
            # 兼容旧调用方：正式实际有效值不足时才返回临时实际。
            # 新生产主流程会显式同时抓 final/temporary，分别保存，绝不混写。
            try:
                rows = self.crawl_market_overview_actual_final()
                core = ("systemload", "dfdcload", "excload", "fdload", "gfload")
                usable = sum(
                    1 for row in rows
                    if any(
                        str(row.get(k, "")).strip() not in {"", "None", "null", "-", "--"}
                        for k in core
                    )
                )
                if usable >= 90:
                    return rows
                logger.warning("QCTC RealityData 有效业务时段仅 %d，尝试临时实际接口", usable)
            except Exception as exc:
                logger.warning("QCTC RealityData 获取失败，尝试临时实际接口: %s", exc)
            return self.crawl_market_overview_actual_temporary()

        # ---- 主路径：DaJyxxPlYx.do（实时）JSON 接口 ----
        try:
            return self._crawl_market_actual_via_json()
        except Exception as e:
            logger.warning("DaJyxxPlYx JSON 接口失败 (%s)，回退 exportsj 导出", e)
            resp = self._get_export_actual_response(export_type=export_type)
            return _parse_exported_actual_file(resp.content)

    def _crawl_market_actual_via_json(self) -> list[dict[str, Any]]:
        """用实时接口 DaJyxxPlYx.do?method=getNewDetailGridList 爬实际值（JSON）。

        返回 96 行，字段含 systemload/dfdcload/excload/fdload/gfload
        + cxload(抽蓄)/hdload(核电)/zbload(自备)/syjzload(试验) 实际值。
        """
        rows = self._crawl_market_overview_host("DaJyxxPlYx", columns=self.ACTUAL_COLUMNS)
        logger.info("DaJyxxPlYx market actual: %d rows", len(rows))
        if len(rows) < 90:
            raise ValueError(f"DaJyxxPlYx 返回 {len(rows)} 行（<90），疑似异常")
        return rows

    def crawl_market_overview_da(self) -> list[dict[str, Any]]:
        """用日前接口 DaJyxxPlDa.do 爬市场特征（9 列，含核电/自备/试验机组）。

        用于诊断：确认日前接口对历史日期返回预测还是实际值。
        """
        rows = self._crawl_market_overview_host("DaJyxxPlDa")
        logger.info("DaJyxxPlDa market overview: %d rows", len(rows))
        return rows

    def _get_export_actual_response(
        self,
        *,
        export_type: Optional[str] = None,
    ) -> requests.Response:
        """请求「导出实际」端点并返回响应对象。

        与现有成功爬取 `getNewDetailGridList` 保持一致：
          * 带 Referer = DaJyxxPlDa.do?appkey=112（市场概况页面）；
          * 带 Origin = 站点根；
          * 用 HTML 模式（_page_get，不带 X-Requested-With）——因为 exportsj
            是浏览器 location.href 导航下载，不是 XHR。
        """
        url = f"{self.base_url}/DaJyxxPlDa.do"
        params = {"method": "exportsj"}
        # 前端 dcbd 值：tab 0「负荷信息」= 1，对应 systemload 等字段。默认 1。
        params["type"] = export_type or "1"
        headers = {
            "Referer": f"{self.base_url}/DaJyxxPlDa.do?appkey=112",
            "Origin": self.base_url.rsplit("/", 1)[0],
        }
        try:
            return self._page_get(
                url,
                params=params,
                referer=headers["Referer"],
                extra_headers={"Origin": headers["Origin"]},
            )
        except Exception as e:
            # exportsj 下载端点常被服务器断连（RemoteDisconnected / HTTP 911）。
            # 有浏览器 CDP 通道时，走真实浏览器 fetch 兜底（最可能绕过反爬）。
            if self.browser_debug_port:
                logger.warning(
                    "exportsj requests 请求失败 (%s)，改用浏览器 fetch 兜底", e
                )
                return self._browser_req(
                    "GET",
                    url,
                    params=params,
                    headers=headers,
                )
            raise

    def _raw_export_actual(
        self,
        *,
        export_type: Optional[str] = None,
    ) -> bytes:
        """返回「导出实际」端点的原始响应字节（供 probe 诊断/浏览器兜底）。"""
        resp = self._get_export_actual_response(export_type=export_type)
        return resp.content
