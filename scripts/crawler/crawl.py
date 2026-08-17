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


def _get_pmos_page(debug_port: int) -> dict[str, Any]:
    r = requests.get(f"http://127.0.0.1:{debug_port}/json", timeout=5)
    r.raise_for_status()
    pages = [
        p for p in r.json()
        if p.get("type") == "page" and p.get("webSocketDebuggerUrl")
    ]
    if not pages:
        raise RuntimeError(f"DevTools port={debug_port} 没有可用页面")
    for p in pages:
        url = str(p.get("url") or "")
        if "pmos.sd.sgcc.com.cn:18080/trade" in url:
            return p
    for p in pages:
        url = str(p.get("url") or "")
        if "pmos.sd.sgcc.com.cn" in url:
            return p
    return pages[0]


def _browser_fetch(
    debug_port: int,
    *,
    page_url: str,
    url: str,
    method: str,
    headers: dict[str, str],
    body: Optional[str] = None,
) -> dict[str, Any]:
    """在真实浏览器页面里执行 fetch。"""
    page = _get_pmos_page(debug_port)
    cdp = _CdpClient(page["webSocketDebuggerUrl"])
    try:
        cdp.call("Page.enable")
        cur_url = str(page.get("url") or "")
        # fetch 最好从 :18080/trade 同源页面发起，避免 CORS/Origin 差异。
        if ":18080/trade" not in cur_url:
            logger.info("浏览器 fetch 前导航到交易页: %s", page_url)
            cdp.call("Page.navigate", {"url": page_url}, timeout=10)
            time.sleep(3)

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
        out = cdp.call(
            "Runtime.evaluate",
            {
                "expression": expr,
                "awaitPromise": True,
                "returnByValue": True,
            },
            timeout=90,
        )
        result = ((out.get("result") or {}).get("value") or {})
        if not isinstance(result, dict):
            raise RuntimeError(f"浏览器 fetch 返回异常: {out}")
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
    ):
        self.base_url = base_url.rstrip("/")
        self.unit_id = unit_id
        self.browser_debug_port = browser_debug_port

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
            if self.browser_debug_port:
                logger.warning(
                    "requests 请求失败，改用浏览器 fetch 兜底: %s %s -> %s",
                    method.upper(),
                    url,
                    e,
                )
                return self._browser_req(method, url, **kwargs)
            raise

        if resp.status_code in (302, 401, 403):
            raise requests.HTTPError(
                f"认证失败 (HTTP {resp.status_code})，Cookie 可能已过期",
                response=resp,
            )
        resp.raise_for_status()
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
            logger.info("页面访问: status=%s url=%s len=%s", resp.status_code, resp.url, len(resp.text or ""))
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
            page_url=self.base_url + "/DaJyjgfbPlantQuery.do?appkey=187",
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
        if resp.status_code >= 400 or resp.status_code == 0:
            logger.error(
                "浏览器 fetch 失败: status=%s statusText=%s text=%s",
                resp.status_code,
                str(result.get("statusText") or "")[:300],
                text[:300],
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

    # ------------------------------------------------------------------
    #  日期切换
    # ------------------------------------------------------------------

    def change_date(self, date_str: str) -> bool:
        """切换市场日期"""
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
        return self._crawl_market_overview_host("DaJyxxPlDa")

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

    def crawl_day_ahead(self) -> list[dict[str, Any]]:
        """日前出清价格/出力 96 点"""
        return self._crawl_unit_detail("DaJyjgfbPlantQuery24.do")

    def crawl_realtime(self) -> list[dict[str, Any]]:
        """实时出清价格/出力 96 点"""
        return self._crawl_unit_detail("YxJyjgfbPlantQuery24.do")

    def crawl_optional_market_data(self) -> dict[str, Any]:
        """爬取 HAR5 中发现的附加市场信息。

        这些数据不直接写入 actual/fcast 核心列，先按原始字段保存，避免
        把日级容量、备用或检修记录错误广播成 96 点序列。
        """
        out: dict[str, Any] = {}
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
