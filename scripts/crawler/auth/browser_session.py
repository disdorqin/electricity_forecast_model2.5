#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PMOS 浏览器登录态桥接
=====================

用途：
  - 不再依赖“手工复制一段静态 Cookie”。
  - 使用企业电脑本地的独立 Chrome/Edge 用户数据目录保存 PMOS 登录态。
  - 程序启动时通过 Chrome DevTools Protocol 读取当前浏览器 Cookie，
    校验可用后写回 config.json，供现有 requests 爬虫继续工作。
  - 登录态失效时，弹出真实浏览器让甲方完成正常登录/验证；登录成功后
    程序自动提取新 Cookie 并继续执行爬虫。

说明：
  这里不做验证码破解；验证码由真实浏览器/真实用户完成。工程上解决的是
  Cookie/Token 会滚动更新、定时任务无法长期复用单条 Cookie 的问题。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import quote, urlparse

import requests
import websocket

from scripts.crawler.collect.crawl import PmosCrawler

logger = logging.getLogger(__name__)


DEFAULT_TRADE_ENTRY = "DaJyjgfbPlantQuery.do?appkey=187"


def update_config_cookie_local(config_path: str | Path, cookie: str) -> None:
    """轻量写回 cookie，避免 browser 模式强依赖 auth.py 的国密依赖。"""
    path = Path(config_path)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    data["cookie"] = cookie
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        return s.connect_ex((host, port)) == 0


def find_free_port(start: int = 9222, limit: int = 80) -> int:
    for port in range(start, start + limit):
        if not _is_port_open(port):
            return port
    raise RuntimeError(f"未找到可用调试端口: {start}-{start + limit - 1}")


def _candidate_browser_paths() -> Iterable[Path]:
    env_keys = ["PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"]
    bases = [Path(os.environ[k]) for k in env_keys if os.environ.get(k)]
    rels = [
        Path("Google/Chrome/Application/chrome.exe"),
        Path("Microsoft/Edge/Application/msedge.exe"),
    ]
    for base in bases:
        for rel in rels:
            yield base / rel


def candidate_browser_exes(configured: str | None = None) -> list[Path]:
    """按优先级返回可用浏览器列表。

    Chrome 在部分单位电脑上会被策略限制 remote-debugging，或者启动后
    DevTools 端口不开放；此时自动 fallback 到 Edge。
    """
    out: list[Path] = []

    def add(p: Path | str | None) -> None:
        if not p:
            return
        pp = Path(p).expanduser()
        try:
            pp = pp.resolve()
        except Exception:
            pass
        if pp.exists() and pp not in out:
            out.append(pp)

    if configured:
        add(configured)

    for name in ("chrome.exe", "msedge.exe", "chrome", "msedge"):
        add(shutil.which(name))

    for p in _candidate_browser_paths():
        add(p)

    return out


def find_browser_exe(configured: str | None = None) -> Path:
    candidates = candidate_browser_exes(configured)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "未找到 Chrome/Edge。请在 config.json 增加 browser_path，"
        "例如 C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    )


def build_service_url(cfg: dict | None = None) -> str:
    """构造登录成功后应跳转的交易系统入口。

    注意：这里必须带上 config.json 里的 :18080/trade。
    旧写法固定成 http://pmos.sd.sgcc.com.cn/trade/...，没有端口，
    浏览器登录后可能只停在统一认证/门户侧，程序一直等不到交易系统 Cookie。
    """
    cfg = cfg or {}
    if cfg.get("browser_service_url"):
        return str(cfg["browser_service_url"]).strip()
    base_url = (
        cfg.get("base_url")
        or cfg.get("trade_base")
        or "https://pmos.sd.sgcc.com.cn:18080/trade"
    )
    return base_url.rstrip("/") + "/" + DEFAULT_TRADE_ENTRY


def build_login_url(auth_host: str, service_url: str | None = None) -> str:
    service_url = service_url or build_service_url({})
    return auth_host.rstrip("/") + "/?service=" + quote(service_url, safe="")


def launch_browser(
    browser_exe: Path,
    profile_dir: Path,
    debug_port: int,
    login_url: str,
    *,
    profile_name: str | None = None,
    headless: bool = False,
) -> subprocess.Popen:
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(browser_exe),
        f"--remote-debugging-port={debug_port}",
        "--remote-debugging-address=127.0.0.1",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-popup-blocking",
        "--disable-features=Translate,AutomationControlled",
    ]
    if profile_name:
        cmd.append(f"--profile-directory={profile_name}")
    if headless:
        cmd += ["--headless=new", "--disable-gpu"]
    cmd.append(login_url)

    logger.info(
        "[browser] 启动浏览器: %s user-data-dir=%s port=%s",
        browser_exe,
        profile_dir,
        debug_port,
    )
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def wait_debugger(debug_port: int, timeout: int = 60, proc: Optional[subprocess.Popen] = None) -> None:
    deadline = time.time() + timeout
    last_err: Optional[Exception] = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"浏览器进程已退出，code={proc.returncode}，DevTools 端口未开启")
        try:
            r = requests.get(f"http://127.0.0.1:{debug_port}/json/version", timeout=2)
            if r.ok:
                try:
                    info = r.json()
                    logger.info(
                        "[browser] DevTools 已就绪: port=%s browser=%s",
                        debug_port,
                        str(info.get("Browser") or "")[:80],
                    )
                except Exception:
                    logger.info("[browser] DevTools 已就绪: port=%s", debug_port)
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"浏览器 DevTools 端口未就绪: {last_err}")


def open_or_get_page(debug_port: int, login_url: str) -> dict[str, Any]:
    base = f"http://127.0.0.1:{debug_port}"

    # 重要：先复用已有 PMOS 页面，不能每轮都 /json/new。
    # 之前这里先 /json/new，导致等待登录期间反复弹出新页面，用户无法输入。
    try:
        r = requests.get(base + "/json", timeout=3)
        if r.ok:
            pages = [
                p for p in r.json()
                if p.get("type") == "page" and p.get("webSocketDebuggerUrl")
            ]
            if pages:
                # 优先选择已经打开到 PMOS 的页面；否则复用第一个普通页面。
                for p in pages:
                    url = str(p.get("url") or "")
                    if "pmos.sd.sgcc.com.cn" in url or "sgcc.com.cn" in url:
                        return p
                return pages[0]
    except Exception:
        pass

    # 没有页面时才创建一个。Chrome 新版本要求 PUT /json/new；旧版本 GET 也可。
    for method in ("put", "get"):
        try:
            r = getattr(requests, method)(base + "/json/new?" + quote(login_url, safe=":/?=&%"), timeout=3)
            if r.ok:
                page = r.json()
                if page.get("webSocketDebuggerUrl"):
                    return page
        except Exception:
            pass

    raise RuntimeError("浏览器已启动，但没有可连接的页面")


def _all_page_targets(debug_port: int) -> list[dict[str, Any]]:
    """返回全部 Edge/Chrome 页面，避免多标签时误选到隐藏旧页面。"""
    try:
        r = requests.get(f"http://127.0.0.1:{debug_port}/json", timeout=3)
        if not r.ok:
            return []
        return [p for p in r.json() if p.get("type") == "page" and p.get("webSocketDebuggerUrl")]
    except Exception:
        return []


def start_browser_with_debugger(
    cfg: dict,
    profile_dir: Path,
    debug_port: int,
    login_url: str,
    *,
    headless: bool = False,
) -> Optional[subprocess.Popen]:
    """启动一个带 DevTools 的浏览器；Chrome 失败则自动尝试 Edge。"""
    configured = cfg.get("browser_path")
    candidates = candidate_browser_exes(configured)
    if not candidates:
        raise FileNotFoundError(
            "未找到 Chrome/Edge。请在 config.json 增加 browser_path，"
            "例如 C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        )

    ready_timeout = int(cfg.get("browser_debugger_timeout_sec") or 60)
    last_err: Optional[Exception] = None
    for i, browser_exe in enumerate(candidates, 1):
        # 每个浏览器用独立子目录，避免 Chrome profile 锁/损坏影响 Edge fallback。
        browser_profile = profile_dir
        if len(candidates) > 1:
            safe_name = browser_exe.stem.replace(" ", "_")
            browser_profile = profile_dir.parent / f"{profile_dir.name}_{safe_name}"

        logger.info("[browser] 尝试启动浏览器 %s/%s: %s", i, len(candidates), browser_exe)
        profile_name = str(cfg.get("browser_profile_name") or "").strip() or None
        proc = launch_browser(
            browser_exe,
            browser_profile,
            debug_port,
            login_url,
            profile_name=profile_name,
            headless=headless,
        )
        try:
            wait_debugger(debug_port, timeout=ready_timeout, proc=proc)
            return proc
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("[browser] %s DevTools 未就绪: %s", browser_exe, e)
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            time.sleep(1.0)

    raise RuntimeError(
        "Chrome/Edge 都未能开启 DevTools 调试端口。"
        f"最后错误: {last_err}。"
        "可尝试在 config.json 指定 browser_path 为 Edge 的 msedge.exe。"
    )


class CdpClient:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=5)
        self._id = 0

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass

    def call(self, method: str, params: Optional[dict[str, Any]] = None, timeout: int = 5) -> dict[str, Any]:
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

    def evaluate(self, expression: str, *, await_promise: bool = False, timeout: int = 10) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
            timeout=timeout,
        )
        value = (result.get("result") or {}).get("value")
        if (result.get("result") or {}).get("subtype") == "error":
            raise RuntimeError(str(result))
        return value


def _page_cdp(debug_port: int, login_url: str) -> CdpClient:
    page = open_or_get_page(debug_port, login_url)
    return CdpClient(page["webSocketDebuggerUrl"])


def automate_edge_login_form(debug_port: int, cfg: dict[str, Any], login_url: str) -> bool:
    """向 Edge 登录页填入账号密码并点击登录。

    滑块和 CFCA 仍由页面/本机驱动处理；这里不伪造证书，也不绕过服务端认证。
    """
    username = str(cfg.get("username") or "").strip()
    password = str(cfg.get("password") or "")
    if not username or not password:
        logger.warning("[browser] 未配置 username/password，跳过自动填表")
        return False
    pages = _all_page_targets(debug_port) or [open_or_get_page(debug_port, login_url)]
    for page in pages:
        cdp = CdpClient(page["webSocketDebuggerUrl"])
        try:
            # 使用 React/Vue 兼容的原生 setter，避免只改 DOM 而不触发表单状态。
            import json as _json
            u = _json.dumps(username, ensure_ascii=False)
            p = _json.dumps(password, ensure_ascii=False)
            expr = f"""(() => {{
          const setv = (el, val) => {{
            const proto = Object.getPrototypeOf(el);
            const desc = Object.getOwnPropertyDescriptor(proto, 'value');
            if (desc && desc.set) desc.set.call(el, val); else el.value = val;
            el.dispatchEvent(new Event('input', {{bubbles:true}}));
            el.dispatchEvent(new Event('change', {{bubbles:true}}));
          }};
          const roots = [document];
          for (const f of [...window.frames]) {{ try {{ roots.push(f.document); }} catch (_) {{}} }}
          const all = roots.flatMap(d => [...d.querySelectorAll('input')]);
          const user = all.find(x => !/password/i.test(x.type) && /user|account|账号|用户名|登录名/i.test(x.placeholder + ' ' + x.name + ' ' + x.autocomplete))
            || all.find(x => !/password/i.test(x.type));
          const pass = all.find(x => x.type === 'password' || /password|密码/i.test(x.placeholder + ' ' + x.name));
          if (!user || !pass) return {{ok:false, inputs:all.map(x => [x.type,x.name,x.placeholder])}};
          setv(user, {u}); setv(pass, {p});
          const btn = roots.flatMap(d => [...d.querySelectorAll('button, [role=button], input[type=submit]')])
            .find(x => /登录|登 录|login/i.test((x.innerText || x.value || '').trim()));
          if (btn) {{ btn.click(); return {{ok:true, clicked:true}}; }}
          return {{ok:true, clicked:false}};
        }})()"""
            result = cdp.evaluate(expr)
            logger.info("[browser] Edge 登录表单自动填充 page=%s: %s", page.get("url", "")[:100], result)
            if isinstance(result, dict) and result.get("ok"):
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("[browser] Edge 登录表单自动填充失败 page=%s: %s", page.get("url", "")[:100], exc)
        finally:
            cdp.close()
    return False


def browser_runtime_auth_probe(debug_port: int, cfg: dict[str, Any], login_url: str, *, verbose: bool = False) -> bool:
    """在 Edge 上验证真实认证 API 和交易入口，禁止用 URL/Cookie 猜测成功。"""
    page = open_or_get_page(debug_port, login_url)
    cdp = CdpClient(page["webSocketDebuggerUrl"])
    base_url = str(cfg.get("base_url") or "https://pmos.sd.sgcc.com.cn:18080/trade").rstrip("/")
    auth_url = str(cfg.get("auth_host") or "https://pmos.sd.sgcc.com.cn").rstrip("/") + "/px-common-authcenter/auth/v2/information"
    trade_url = base_url + "/DaJyxxPlDa.do?appkey=112"
    import json as _json
    expr = f"""Promise.all([
      fetch({_json.dumps(auth_url)}, {{credentials:'include'}}).then(async r => ({{s:r.status,t:(await r.text()).slice(0,800)}})).catch(e=>({{s:0,t:String(e)}})),
      fetch({_json.dumps(trade_url)}, {{credentials:'include'}}).then(async r => ({{s:r.status,t:(await r.text()).slice(0,1200)}})).catch(e=>({{s:0,t:String(e)}}))
    ])"""
    try:
        result = cdp.evaluate(expr, await_promise=True, timeout=15)
        auth, trade = (result or [{}, {}])[:2]
        auth_text = str(auth.get("t") or "").lower()
        trade_text = str(trade.get("t") or "").lower()
        ok = int(auth.get("s") or 0) == 200 and int(trade.get("s") or 0) == 200 and not any(x in trade_text for x in ("top.location.href", "loginform", "captcha"))
        if verbose:
            logger.info("[browser] 真实认证探针: information=%s trade=%s len=%s/%s -> %s", auth.get("s"), trade.get("s"), len(auth_text), len(trade_text), "PASS" if ok else "WAIT")
        return ok
    finally:
        cdp.close()


def get_cookies_via_cdp(debug_port: int, auth_host: str, login_url: str | None = None) -> list[dict[str, Any]]:
    page = open_or_get_page(debug_port, login_url or build_login_url(auth_host))
    cdp = CdpClient(page["webSocketDebuggerUrl"])
    try:
        # 只读取 Cookie，不在轮询中反复导航页面，避免打断用户输入。
        cdp.call("Network.enable")

        try:
            result = cdp.call("Network.getAllCookies")
            return result.get("cookies") or []
        except Exception:
            result = cdp.call("Storage.getCookies")
            return result.get("cookies") or []
    finally:
        cdp.close()


def cookie_string_from_cdp(cookies: list[dict[str, Any]]) -> str:
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for c in cookies:
        name = str(c.get("name") or "").strip()
        value = str(c.get("value") or "")
        domain = str(c.get("domain") or "")
        if not name:
            continue
        # 只拿国网页面相关 Cookie，避免把其它浏览器 Cookie 写进 config。
        if "sgcc.com.cn" not in domain and "pmos" not in domain.lower():
            continue
        if name in seen:
            continue
        seen.add(name)
        items.append((name, value))

    # 关键 Cookie 优先，便于日志排查。
    priority = {
        "Admin-Token": 0,
        "X-Ticket": 1,
        "JSESSIONID": 2,
        "XHXT_SESSIONID": 3,
        "ClientTag": 4,
        "CurrentRoute": 5,
        "X-Token": 6,
        "Gray-Tag": 7,
    }
    items.sort(key=lambda kv: (priority.get(kv[0], 99), kv[0]))
    return "; ".join(f"{k}={v}" for k, v in items)


def summarize_cookie(cookie: str) -> str:
    names = []
    for part in cookie.split(";"):
        part = part.strip()
        if "=" in part:
            names.append(part.split("=", 1)[0])
    return ",".join(names[:20])


def _cookie_dict(cookie: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in cookie.split(";"):
        item = item.strip()
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _cookie_string_from_dict(cookie_map: dict[str, str]) -> str:
    priority = {
        "Admin-Token": 0,
        "X-Ticket": 1,
        "JSESSIONID": 2,
        "XHXT_SESSIONID": 3,
        "ClientTag": 4,
        "CurrentRoute": 5,
        "X-Token": 6,
        "Gray-Tag": 7,
    }
    items = [(k, v) for k, v in cookie_map.items() if k and v is not None]
    items.sort(key=lambda kv: (priority.get(kv[0], 99), kv[0]))
    return "; ".join(f"{k}={v}" for k, v in items)


def enrich_trade_cookie(cookie: str, cfg: dict, *, verbose: bool = False) -> str:
    """用 requests 访问交易系统入口，把服务端 Set-Cookie 合并回来。

    CDP 从浏览器拿到的 Cookie 有时只有门户侧 Admin-Token/X-Ticket，
    但交易接口实际还依赖 JSESSIONID。校验阶段访问 /trade 页面后，
    服务端会补发 JSESSIONID；这里必须把它写回 config.json。
    """
    if not cookie:
        return cookie

    base_url = cfg.get("base_url") or "https://pmos.sd.sgcc.com.cn:18080/trade"
    origin = base_url.rsplit("/", 1)[0]
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    })
    original = _cookie_dict(cookie)
    for k, v in original.items():
        sess.cookies.set(k, v)

    seed_urls = [
        base_url.rstrip("/") + "/DaJyjgfbPlantQuery.do?appkey=187",
        base_url.rstrip("/") + "/main/index.do",
        base_url.rstrip("/") + "/DaJyxxPlDa.do?appkey=112",
    ]
    for url in seed_urls:
        try:
            sess.get(
                url,
                timeout=15,
                verify=False,
                allow_redirects=True,
                headers={"Referer": origin + "/"},
            )
        except Exception as e:  # noqa: BLE001
            if verbose:
                logger.info("[browser] 交易 Cookie 补齐访问失败: %s -> %s", url, e)

    merged = dict(original)
    for c in sess.cookies:
        if c.name and c.value is not None:
            merged[c.name] = c.value
    enriched = _cookie_string_from_dict(merged)
    if verbose or summarize_cookie(enriched) != summarize_cookie(cookie):
        logger.info("[browser] Cookie 补齐后字段: %s", summarize_cookie(enriched))
    return enriched


def _has_session_cookie(cookie: str) -> bool:
    names = _cookie_dict(cookie)
    return any(
        name in names and bool(names.get(name))
        for name in ("Admin-Token", "X-Ticket", "JSESSIONID", "XHXT_SESSIONID")
    )


def _current_page_urls(debug_port: int) -> list[str]:
    try:
        r = requests.get(f"http://127.0.0.1:{debug_port}/json", timeout=3)
        if not r.ok:
            return []
        return [
            str(p.get("url") or "")
            for p in r.json()
            if p.get("type") == "page"
        ]
    except Exception:
        return []


def _looks_like_trade_url(url: str, cfg: dict) -> bool:
    """浏览器 URL 是否已经进入交易系统，而不是还停在统一登录页。"""
    if not url:
        return False
    try:
        p = urlparse(url)
    except Exception:
        return False
    host = (p.hostname or "").lower()
    path = (p.path or "").lower()
    full = url.lower()
    base_url = str(cfg.get("base_url") or "https://pmos.sd.sgcc.com.cn:18080/trade").lower()
    base_path = urlparse(base_url).path.rstrip("/").lower() or "/trade"

    if "pmos.sd.sgcc.com.cn" not in host:
        return False
    # 只认真实路径里的 /trade，不认登录页 service 参数里被编码的 %2Ftrade%2F。
    if path.startswith(base_path + "/") or path == base_path:
        return True
    # 部分前端路由可能把路径放在 hash 后面，但仍应已经在 18080/trade 页面。
    return ":18080/trade" in full and "%2ftrade%2f" not in full


def browser_page_indicates_logged_in(debug_port: int, cookie: str, cfg: dict, *, verbose: bool = False) -> bool:
    """兜底判断：真实浏览器已经进入交易系统页面 + 已读到会话 Cookie。

    有些 PMOS 页面不稳定返回 CSRF/主页标记，导致纯 requests 校验过严。
    对 browser 模式，用户已经在真实浏览器完成登录时，可以先写回 Cookie，
    后面让正式爬虫接口去验证数据请求是否成功。
    """
    urls = _current_page_urls(debug_port)
    has_session = _has_session_cookie(cookie)
    trade_url = next((u for u in urls if _looks_like_trade_url(u, cfg)), "")
    ok = bool(has_session and trade_url)
    if verbose:
        logger.info(
            "[browser] 浏览器兜底判断: has_session=%s trade_url=%s -> %s",
            has_session,
            trade_url[:160] if trade_url else "-",
            "OK" if ok else "WAIT",
        )
    return ok


def inject_cookie_to_browser(debug_port: int, cookie: str, cfg: dict, *, verbose: bool = False) -> None:
    """把 config.json 里的 Cookie 注入到当前 DevTools 浏览器。

    requests 能用的 Cookie 不等于浏览器 profile 里也有这份 Cookie。
    后续如果要用浏览器 fetch 兜底，必须先把 Cookie 写进浏览器上下文。
    """
    if not cookie:
        return
    auth_host = str(cfg.get("auth_host") or "https://pmos.sd.sgcc.com.cn").rstrip("/")
    base_url = str(cfg.get("base_url") or "https://pmos.sd.sgcc.com.cn:18080/trade").rstrip("/")
    urls = [
        auth_host + "/",
        base_url + "/",
        base_url + "/DaJyjgfbPlantQuery.do?appkey=187",
    ]
    login_url = build_login_url(auth_host, build_service_url(cfg))
    page = open_or_get_page(debug_port, login_url)
    cdp = CdpClient(page["webSocketDebuggerUrl"])
    count = 0
    try:
        cdp.call("Network.enable")
        for name, value in _cookie_dict(cookie).items():
            if not name:
                continue
            for url in urls:
                try:
                    ret = cdp.call(
                        "Network.setCookie",
                        {
                            "name": name,
                            "value": value,
                            "url": url,
                            "secure": url.lower().startswith("https://"),
                        },
                    )
                    if ret.get("success"):
                        count += 1
                except Exception as e:  # noqa: BLE001
                    if verbose:
                        logger.info("[browser] 注入 Cookie 失败: %s @ %s -> %s", name, url, e)
        if verbose:
            logger.info("[browser] 已向浏览器注入 Cookie: names=%s writes=%s", summarize_cookie(cookie), count)
    finally:
        cdp.close()


def validate_cookie(cookie: str, cfg: dict, *, verbose: bool = False) -> bool:
    if not cookie or len(cookie) < 20:
        if verbose:
            logger.info("[browser] Cookie 校验未通过：尚未读取到有效 Cookie")
        return False
    base_url = cfg.get("base_url") or "https://pmos.sd.sgcc.com.cn:18080/trade"

    # 不能只依赖 CSRF token。该站部分页面没有显式 CSRF，但 Cookie 已可用于爬虫。
    # 这里直接访问交易主页：没有跳回登录页/认证中心，就认为登录态基本有效。
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    })
    for k, v in _cookie_dict(cookie).items():
        sess.cookies.set(k, v)

    url = base_url.rstrip("/") + "/main/index.do"
    try:
        resp = sess.get(url, timeout=15, verify=False, allow_redirects=True)
        final_url = str(resp.url or "")
        text = resp.text or ""
        low = (final_url + "\n" + text[:4000]).lower()

        login_markers = [
            "/px-common-authcenter/",
            "captcha",
            "loginform",
            "请输入账号",
            "请输入登录系统的密码",
            "滑块",
            # PMOS 过期会返回 135 字节的 SSO 跳转脚本；不能仅因 URL 文本包含 /trade/ 就判定有效。
            "top.location.href",
            "window.location.href",
        ]
        trade_markers = [
            "/trade/",
            "changeDate.do",
            "main/index.do",
            "DaJyxxPlDa.do",
            "DaJyjgfbPlantQuery.do",
        ]

        has_session_cookie = _has_session_cookie(cookie)
        looks_login = any(str(m).lower() in low for m in login_markers)
        looks_trade = any(str(m).lower() in low for m in trade_markers)
        trade_root = base_url.rstrip("/").lower()
        final_is_trade = final_url.lower().startswith(trade_root)

        ok = (
            resp.status_code == 200
            and has_session_cookie
            and not looks_login
            and final_is_trade
            and looks_trade
        )
        if verbose:
            logger.info(
                "[browser] Cookie校验: status=%s final_url=%s has_session=%s looks_login=%s looks_trade=%s final_is_trade=%s len=%s -> %s",
                resp.status_code, final_url[:160], has_session_cookie, looks_login,
                looks_trade, final_is_trade, len(text), "OK" if ok else "WAIT",
            )
        return ok
    except Exception as e:  # noqa: BLE001
        if verbose:
            logger.info("[browser] Cookie 校验失败: %s", e)
        return False


def ensure_browser_cookie(
    cfg: dict,
    config_path: str | Path,
    *,
    base_dir: str | Path,
    timeout_sec: int = 300,
    headless: bool = False,
) -> str:
    """
    返回可用于 requests 爬虫的 Cookie 字符串。

    优先使用 config.json 现有 cookie；不可用则启动本地浏览器读取/刷新登录态。
    """
    auth_host = cfg.get("auth_host") or "https://pmos.sd.sgcc.com.cn"
    profile_dir = Path(
        cfg.get("browser_profile_dir")
        or Path(base_dir) / "pmos_browser_profile"
    ).expanduser()
    debug_port = int(cfg.get("browser_debug_port") or find_free_port())
    service_url = build_service_url(cfg)
    login_url = build_login_url(auth_host, service_url)

    proc: Optional[subprocess.Popen] = None

    def _ensure_debugger() -> None:
        nonlocal proc
        if not _is_port_open(debug_port):
            proc = start_browser_with_debugger(
                cfg,
                profile_dir,
                debug_port,
                login_url,
                headless=headless,
            )
        else:
            logger.info("[browser] 复用已打开的 DevTools 端口: %s", debug_port)
            wait_debugger(
                debug_port,
                timeout=int(cfg.get("browser_debugger_timeout_sec") or 60),
            )
        os.environ["PMOS_BROWSER_DEBUG_PORT"] = str(debug_port)

    existing = str(cfg.get("cookie") or "").strip()
    if validate_cookie(existing, cfg):
        # 即使 Cookie 有效，也启动/复用浏览器 DevTools。
        # 这样后续 requests 接口被服务端断开时，可以兜底用真实浏览器 fetch。
        _ensure_debugger()
        enriched = enrich_trade_cookie(existing, cfg, verbose=True)
        inject_cookie_to_browser(debug_port, enriched, cfg, verbose=True)
        if enriched != existing:
            update_config_cookie_local(config_path, enriched)
        logger.info("[browser] config.json 现有 Cookie 仍有效，直接复用")
        return enriched

    _ensure_debugger()

    logger.info("[browser] 登录入口: %s", login_url)
    logger.info("[browser] 目标交易入口: %s", service_url)
    try:
        # 表单自动化在本模块中，CFCA 探测/原生窗口辅助在 cfca_runtime 中。
        from scripts.crawler.auth.cfca_runtime import probe_local_cryptokit

        probe_local_cryptokit(int(cfg.get("cfca_port") or 7693))
        # 先自动填写账号密码；页面验证码和 CFCA 仍由真实 Edge/本机驱动完成。
        automate_edge_login_form(debug_port, cfg, login_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[browser] 自动登录初始化失败，将继续监听登录态: %s", exc)
    logger.info("[browser] 已启动 Edge 自动认证流程，等待滑块与 CFCA/UKey 完成")
    deadline = time.time() + timeout_sec
    last_cookie = ""
    last_diag = 0.0
    last_native_assist = 0.0
    while time.time() < deadline:
        try:
            cookies = get_cookies_via_cdp(debug_port, auth_host, login_url)
            cookie = cookie_string_from_cdp(cookies)
            if cookie and cookie != last_cookie:
                last_cookie = cookie
                logger.info("[browser] 读取到 Cookie 字段: %s", summarize_cookie(cookie))
            now = time.time()
            verbose = now - last_diag >= 15
            if verbose:
                last_diag = now
                urls = _current_page_urls(debug_port)
                if urls:
                    logger.info("[browser] 当前浏览器页面: %s", " | ".join(u[:140] for u in urls[:3]))
                if not cookie:
                    logger.info("[browser] 尚未读取到 PMOS Cookie；请确认登录页域名是 pmos.sd.sgcc.com.cn")
            if validate_cookie(cookie, cfg, verbose=verbose):
                # requests 校验通过后再次做浏览器真实 API 探针，避免门户 Cookie 假阳性。
                if not browser_runtime_auth_probe(debug_port, cfg, login_url, verbose=True):
                    logger.warning("[browser] requests 已通过但浏览器真实交易探针未通过，继续等待")
                    time.sleep(2.0)
                    continue
                cookie = enrich_trade_cookie(cookie, cfg, verbose=True)
                inject_cookie_to_browser(debug_port, cookie, cfg, verbose=True)
                update_config_cookie_local(config_path, cookie)
                logger.info("[browser] Edge+CFCA 登录和交易探针均通过，Cookie 已写回 %s", config_path)
                return cookie
            # 自动处理可能出现的原生证书/PIN 窗口；每 2 秒最多扫描一次。
            now2 = time.time()
            if now2 - last_native_assist >= 2.0:
                last_native_assist = now2
                try:
                    from scripts.crawler.auth.cfca_runtime import assist_native_ukey_dialog
                    assist_native_ukey_dialog(cfg, timeout_sec=2)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[cfca] 原生窗口扫描异常: %s", exc)
        except Exception as e:  # noqa: BLE001
            logger.debug("[browser] 等待登录态: %s", e)
        time.sleep(2.0)

    if proc and proc.poll() is not None:
        logger.warning("[browser] 浏览器进程已退出，code=%s", proc.returncode)
    raise TimeoutError(
        f"{timeout_sec}s 内未获得有效 PMOS 登录态。"
        "请确认弹出的浏览器已完成登录并能打开交易主页。"
    )


if __name__ == "__main__":
    # 简单调试入口：python scripts/crawler/auth/browser_session.py
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = Path(__file__).resolve().parents[4]
    config = root / "scripts" / "crawler" / "config.json"
    if not config.exists():
        print(f"缺少 {config}")
        sys.exit(2)
    cfg_obj = json.loads(config.read_text(encoding="utf-8"))
    print(ensure_browser_cookie(cfg_obj, config, base_dir=root))
