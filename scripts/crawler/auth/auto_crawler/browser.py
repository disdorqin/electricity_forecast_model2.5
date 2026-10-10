from __future__ import annotations

import json
import logging
import os
import plistlib
import re
import shutil
import socket
import subprocess
import sys
import time
import base64
from pathlib import Path
from typing import Any

import requests
import websocket

from .config import AuthConfig

# 版本演进索引（仅标关键新增能力，避免逐行打标签）：
# [V3-V5 legacy baseline] 早期浏览器/CDP基础控制能力。
# [V8] 新启动浏览器 bootstrap 健康检查与 Chrome→Edge 回退框架。
# [V9] Windows 浏览器安装路径发现修正。
# [V10-r2] 固定检查 C:\\Program Files / C:\\Program Files (x86) 的 Chrome/Edge 路径。
# [V10-r4] PMOS bootstrap render-readiness gate and Edge App Paths/process discovery fallback。
# 后续新增浏览器恢复逻辑必须继续用 [Vx-rN] 注释标记。

logger = logging.getLogger(__name__)


class BrowserResolutionError(RuntimeError):
    pass


class BrowserControlError(RuntimeError):
    """The reused DevTools/browser target is no longer controllable."""


_BOOTSTRAP_RENDER_PROBE = r"""
(() => {
  const body = document.body;
  const text = body ? String(body.innerText || body.textContent || '').trim() : '';
  const exists = (selector) => Boolean(document.querySelector(selector));
  const visible = (selector) => {
    const element = document.querySelector(selector);
    if (!element) return false;
    const style = window.getComputedStyle(element);
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      Number(style.opacity || 1) > 0;
  };
  const route = `${location.hash || ''} ${location.pathname || ''}`.toLowerCase();
  const loginText = /(登录|用户名|密码|证书|滑块|统一认证|ukey|cfca)/i.test(text);
  return {
    href: String(location.href || ''),
    readyState: String(document.readyState || ''),
    bodyTextLength: text.length,
    bodyText: text.slice(0, 240),
    hasPassword: exists('input[type="password"]') || visible('input[type="password"]'),
    hasLogin: loginText || exists('form, input[name*="user" i], input[name*="login" i]'),
    hasCfca: exists('[class*="cfca" i], [id*="cfca" i], [class*="certificate" i], [id*="certificate" i]'),
    hasSlider: exists('[class*="slider" i], [id*="slider" i], [class*="verify" i], [id*="verify" i]'),
    knownRoute: /(outnet|dashboard|zcq|trade|qctc)/i.test(route)
  };
})()
"""


# [2026-09-28] 「死页面」强特征：favicon.ico / 404 / 错误页虽在 PMOS 域名下，
# 但绝不能被当成可复用会话，否则会在 _run_attempt 里死等 600s（实测根因）。
_DEAD_PAGE_URL_PREFIXES = ("chrome-error://", "about:blank", "data:")
_DEAD_PAGE_MARKERS = (
    "404 not found", "not found nginx", "whitelabel error", "internal server error",
    "bad gateway", "service unavailable", "this site can't be reached",
    "无法访问", "找不到", "此网站",
)


def _looks_like_dead_page(url: str, text: str = "") -> bool:
    """Whether a page is an obvious static asset / error page (unusable for reuse)."""
    u = (url or "").strip().lower()
    if not u:
        return True
    if u.startswith(_DEAD_PAGE_URL_PREFIXES):
        return True
    core = u.split("?")[0]
    if core.endswith(".ico") or "favicon" in core:
        return True
    haystack = f"{u} {text or ''}".lower()
    return any(marker in haystack for marker in _DEAD_PAGE_MARKERS)


def is_bootstrap_render_ready(probe: Any) -> bool:
    """Return whether a PMOS runtime probe has a meaningful rendered signal."""
    if not isinstance(probe, dict):
        return False
    href = str(probe.get("href") or "")
    if "pmos.sd.sgcc.com.cn" not in href.lower():
        return False
    # 光有域名不够：favicon.ico / 404 / 错误页也在同域名下 —— 必须排除，
    # 否则「域名对 + 正文非空」会把死页面判成健康，触发 600s 死等。
    if _looks_like_dead_page(href, str(probe.get("bodyText") or "")):
        return False
    if probe.get("hasPassword") or probe.get("hasLogin") \
            or probe.get("hasCfca") or probe.get("hasSlider") \
            or probe.get("knownRoute"):
        return True
    try:
        body_length = int(probe.get("bodyTextLength") or 0)
    except (TypeError, ValueError):
        body_length = 0
    return body_length > 0


def _windows_default_browser_command() -> str:
    import winreg

    user_choice = (
        r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations"
        r"\https\UserChoice"
    )
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, user_choice) as key:
        prog_id, _ = winreg.QueryValueEx(key, "ProgId")
    command_key = rf"{prog_id}\shell\open\command"
    for hive, prefix in (
        (winreg.HKEY_CLASSES_ROOT, ""),
        (winreg.HKEY_CURRENT_USER, "Software\\Classes\\"),
    ):
        try:
            with winreg.OpenKey(hive, prefix + command_key) as key:
                command, _ = winreg.QueryValueEx(key, None)
                return str(command)
        except OSError:
            continue
    raise BrowserResolutionError(f"无法解析 Windows 默认浏览器命令: {prog_id}")


def _extract_executable(command: str) -> Path:
    match = re.match(r'^\s*"([^"]+\.exe)"', command, re.I)
    if not match:
        match = re.match(r"^\s*([^\s]+\.exe)", command, re.I)
    if not match:
        raise BrowserResolutionError(f"无法从默认浏览器命令提取程序路径: {command!r}")
    return Path(os.path.expandvars(match.group(1))).expanduser()


def resolve_default_browser(configured: str = "") -> Path:
    """只解析一个浏览器；绝不自动回退并启动第二个浏览器。"""
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser()
    elif sys.platform == "win32":
        path = _extract_executable(_windows_default_browser_command())
    elif sys.platform == "darwin":
        pref = Path.home() / "Library/Preferences/com.apple.LaunchServices/com.apple.launchservices.secure.plist"
        with pref.open("rb") as handle:
            handlers = plistlib.load(handle).get("LSHandlers", [])
        bundle = next(
            (item.get("LSHandlerRoleAll") for item in reversed(handlers)
             if item.get("LSHandlerURLScheme") in {"https", "http"}),
            "",
        )
        app_names = {
            "com.microsoft.edgemac": "Microsoft Edge",
            "com.google.chrome": "Google Chrome",
            "com.brave.browser": "Brave Browser",
        }
        name = app_names.get(str(bundle).lower(), "")
        path = Path(f"/Applications/{name}.app/Contents/MacOS/{name}") if name else Path()
    else:
        found = shutil.which("xdg-settings")
        if not found:
            raise BrowserResolutionError("无法解析系统默认浏览器；请配置 browser_path")
        desktop = subprocess.check_output(
            [found, "get", "default-web-browser"], text=True, timeout=5
        ).strip()
        path = Path(shutil.which(desktop.removesuffix(".desktop")) or "")

    if not path or not path.is_file():
        raise BrowserResolutionError(f"默认浏览器不存在: {path or '-'}；请配置 browser_path")
    name = path.name.lower()
    if not any(token in name for token in ("edge", "chrome", "chromium", "brave")):
        raise BrowserResolutionError(
            f"默认浏览器 {path} 不支持本程序所需的 Chromium DevTools；请配置 Edge/Chrome 的 browser_path"
        )
    return path.resolve()


def _windows_app_paths(executable_name: str) -> list[Path]:
    """Read-only Windows App Paths lookup; registry failures are non-fatal."""
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []

    subkey = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\" + executable_name
    views = [0]
    for attr in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
        value = getattr(winreg, attr, 0)
        if value not in views:
            views.append(value)
    discovered: list[Path] = []
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in views:
            try:
                with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ | view) as key:
                    raw, _ = winreg.QueryValueEx(key, None)
                path = Path(os.path.expandvars(str(raw))).expanduser()
                if not path.suffix.lower() == ".exe":
                    path = _extract_executable(str(raw))
                path = path.resolve()
                if path.is_file() and path not in discovered:
                    discovered.append(path)
                    logger.info(
                        "browser.discovery source=app-paths executable=%s path=%s",
                        executable_name, path,
                    )
            except (OSError, ValueError, BrowserResolutionError):
                continue
    return discovered


def _windows_running_browser_path(executable_name: str) -> Path | None:
    """Read a running Chromium process executable path without taking it over."""
    if sys.platform != "win32":
        return None
    process_name = Path(executable_name).stem
    command = (
        f"(Get-Process -Name '{process_name}' -ErrorAction SilentlyContinue "
        "| Where-Object {$_.Path} | Select-Object -First 1 -ExpandProperty Path)"
    )
    try:
        output = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not output:
        return None
    path = Path(output.splitlines()[0].strip()).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    if path.is_file():
        logger.info(
            "browser.discovery source=running-process executable=%s path=%s",
            executable_name, path,
        )
        return path
    return None


def browser_executable_candidates(configured: str = "") -> list[Path]:
    """[V8] bootstrap候选；[V9]路径修复；[V10-r2] Edge固定路径发现；[V10-r4]真实路径兜底。"""
    candidates: list[Path] = []

    def add(path: Path | str | None) -> None:
        if not path:
            return
        p = Path(os.path.expandvars(str(path))).expanduser()
        try:
            p = p.resolve()
        except Exception:
            pass
        if p.is_file() and p not in candidates:
            candidates.append(p)

    # 显式 browser_path 永远优先；未配置时 Windows 固定 Chrome→Edge。
    # 这是部署机器当前约定，避免系统默认浏览器临时变化影响爬虫行为。
    if configured:
        try:
            add(resolve_default_browser(configured))
        except BrowserResolutionError:
            raise

    if sys.platform == "win32":
        bases: list[Path] = []

        def add_base(path: str | Path | None) -> None:
            if not path:
                return
            p = Path(str(path))
            if p not in bases:
                bases.append(p)

        # 部署机存在 PROGRAMFILES(X86) 指向 D:，但 Edge 实际装在
        # C:\Program Files (x86)。因此不能只信环境变量；同时检查
        # SystemDrive 下的标准 Program Files 目录。
        for key in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            add_base(os.environ.get(key))
        # 公司的部分镜像环境会把 PROGRAMFILES / PROGRAMFILES(X86) 甚至
        # SystemDrive 指到不存在的盘符；C 盘标准安装位置仍必须无条件检查。
        # 不扫描磁盘，只补两个固定目录，且不改变 Chrome→Edge 顺序。
        add_base(Path(r"C:\Program Files"))
        add_base(Path(r"C:\Program Files (x86)"))
        system_drive = os.environ.get("SystemDrive") or "C:"
        add_base(Path(system_drive) / "Program Files")
        add_base(Path(system_drive) / "Program Files (x86)")

        # [V10-r4] Windows Edge discovery fallback via App Paths / running process path.
        # 每个浏览器独立收集，确保最终顺序仍是 Chrome 再 Edge。
        for executable_name, relative_path in (
            ("chrome.exe", "Google/Chrome/Application/chrome.exe"),
            ("msedge.exe", "Microsoft/Edge/Application/msedge.exe"),
        ):
            before = len(candidates)
            for base in bases:
                add(base / relative_path)
            add(shutil.which(executable_name))
            for path in _windows_app_paths(executable_name):
                add(path)
            if not any(path.name.lower() == executable_name for path in candidates[before:]):
                add(_windows_running_browser_path(executable_name))

        # 若 Chrome/Edge 都未从常规安装位置发现，再把系统默认 Chromium
        # 作为最后候选；不会改变 Chrome→Edge 的正常优先级。
        if not configured:
            try:
                add(resolve_default_browser(""))
            except BrowserResolutionError:
                pass
    else:
        add(shutil.which("google-chrome"))
        add(shutil.which("chromium"))
        add(shutil.which("microsoft-edge"))

    if not candidates:
        raise BrowserResolutionError("未找到可用 Chrome/Edge；请配置 browser_path")
    return candidates


def ensure_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"调试端口 {port} 已被占用；请关闭旧爬虫浏览器或更换 debug_port")


def probe_cdp_port(port: int) -> dict[str, Any] | None:
    """探测一个 Chromium DevTools 端口，并返回版本和页面摘要。"""
    try:
        version = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=0.35)
        if not version.ok:
            return None
        info = version.json() if version.headers.get("content-type", "").lower().find("json") >= 0 else {}
        pages_resp = requests.get(f"http://127.0.0.1:{port}/json", timeout=0.5)
        pages = pages_resp.json() if pages_resp.ok else []
        pages = [
            p for p in pages
            if p.get("type") == "page" and p.get("webSocketDebuggerUrl")
        ]
        # [2026-09-28] 排除 favicon/404/错误页：它们也在 PMOS 域名下，若计入
        # pmos_page_count 会让「已发现可复用 CDP」误判成立 → 后续死等 600s。
        pages = [
            p for p in pages
            if not _looks_like_dead_page(str(p.get("url") or ""), str(p.get("title") or ""))
        ]
        pmos_pages = [p for p in pages if "pmos.sd.sgcc.com.cn" in str(p.get("url", ""))]
        return {
            "port": port,
            "browser": str(info.get("Browser") or ""),
            "websocket": bool(info.get("webSocketDebuggerUrl")),
            "page_count": len(pages),
            "pmos_page_count": len(pmos_pages),
            "pmos_urls": [str(p.get("url") or "")[:180] for p in pmos_pages[:3]],
        }
    except (OSError, ValueError, requests.RequestException):
        return None


def browser_process_snapshot() -> dict[str, int]:
    """仅用于诊断普通浏览器进程；普通进程没有 CDP 时不可直接接管。"""
    if sys.platform != "win32":
        return {}
    result: dict[str, int] = {}
    for image in ("msedge.exe", "chrome.exe", "brave.exe", "chromium.exe"):
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
                text=True,
                timeout=3,
                stderr=subprocess.DEVNULL,
            )
            count = sum(1 for line in output.splitlines() if image.lower() in line.lower())
            if count:
                result[image] = count
        except (OSError, subprocess.SubprocessError):
            continue
    return result


def discover_existing_cdp(config: AuthConfig) -> dict[str, Any] | None:
    """优先发现已有的 PMOS CDP 浏览器，而不是仅检查固定 9222。"""
    if not config.browser_reuse:
        return None
    start = max(1, int(config.debug_port_scan_start or config.debug_port))
    end = max(start, int(config.debug_port_scan_end or config.debug_port))
    candidates = [int(config.debug_port)] + [p for p in range(start, end + 1) if p != int(config.debug_port)]
    for port in candidates:
        found = probe_cdp_port(port)
        if found and found.get("pmos_page_count", 0) > 0:
            logger.info("browser.cdp_discovered port=%s browser=%s pmos_pages=%s",
                        port, found.get("browser"), found.get("pmos_page_count"))
            return found
    snapshot = browser_process_snapshot()
    if snapshot:
        logger.info("browser.process_detected ordinary_processes=%s; no controllable PMOS CDP found", snapshot)
    return None


def choose_free_debug_port(config: AuthConfig) -> int:
    """选择可启动新浏览器的端口，优先使用配置端口。"""
    start = max(1, int(config.debug_port_scan_start or config.debug_port))
    end = max(start, int(config.debug_port_scan_end or start + 20))
    for port in [int(config.debug_port)] + [p for p in range(start, end + 1) if p != int(config.debug_port)]:
        try:
            ensure_free_port(port)
            return port
        except RuntimeError:
            continue
    raise RuntimeError(f"未找到可用 DevTools 端口: {start}-{end}")


def launch_browser(config: AuthConfig, executable: Path, profile_dir: Path) -> subprocess.Popen:
    ensure_free_port(config.debug_port)
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        f"--remote-debugging-port={config.debug_port}",
        "--remote-debugging-address=127.0.0.1",
        "--remote-allow-origins=*",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-popup-blocking",
        "--new-window",
        config.login_url,
    ]
    # [AUX-V1-r11e] 可选浏览器启动参数（如绕过 Chrome 153 的 Local Network
    # Access 限制，使 CFCA UKey 插件可访问本机 127.0.0.1:7693）。从
    # config.extra["extra_browser_args"] 读取；96 的 config 不设此键则为空，
    # 行为完全不变，不影响主爬虫。
    extra_args = [str(a) for a in (config.extra.get("extra_browser_args") or []) if a]
    if extra_args:
        command.extend(extra_args)
    if config.browser_profile_name:
        command.insert(-1, f"--profile-directory={config.browser_profile_name}")
    logger.info("browser.start executable=%s profile=%s port=%s", executable, profile_dir, config.debug_port)
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


class CdpSession:
    def __init__(self, config: AuthConfig):
        self.config = config

    def is_ready(self) -> bool:
        try:
            return bool(requests.get(self._http("/json/version"), timeout=2).ok)
        except requests.RequestException:
            return False

    def wait_ready(self, proc: subprocess.Popen | None = None) -> None:
        deadline = time.monotonic() + self.config.browser_ready_timeout_sec
        launcher_exited_logged = False
        while time.monotonic() < deadline:
            # Edge/Chrome 可能把 URL 交给已有主进程后退出本次启动器。
            # DevTools 是否可访问才是浏览器可控性的权威判断。
            if proc is not None and proc.poll() is not None and not launcher_exited_logged:
                logger.info("browser.launcher_exited code=%s; waiting_for_devtools=true", proc.returncode)
                launcher_exited_logged = True
            try:
                response = requests.get(self._http("/json/version"), timeout=2)
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise TimeoutError("浏览器 DevTools 启动超时")

    def wait_bootstrap(self, proc: subprocess.Popen | None = None) -> dict[str, Any]:
        """[V8] bootstrap fallback：只验证“浏览器已打开 PMOS 页面”，不判断登录/UKey/业务接口。

        成功条件：
        1) DevTools 可访问；
        2) 至少存在一个可控页面；
        3) 页面运行时 location.href 已进入 pmos.sd.sgcc.com.cn，
           而不是 chrome-error:// / edge:// / about:blank 等启动错误页。
        """
        timeout_sec = max(3, int(self.config.browser_bootstrap_timeout_sec))
        deadline = time.monotonic() + timeout_sec
        launcher_exited_logged = False
        last_urls: list[str] = []
        last_runtime_url = ""
        pmos_runtime_seen = False
        last_probe: dict[str, Any] = {}

        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None and not launcher_exited_logged:
                logger.info(
                    "browser.bootstrap_launcher_exited code=%s; probing_devtools=true",
                    proc.returncode,
                )
                # [AUX-V1-r11f] 判据 A：退出码 0 表示新进程并非崩溃，而是发现同一
                # --user-data-dir 已被另一个浏览器实例占用后直接退出（Chrome/Edge 的
                # profile 单实例锁）。此时它不会监听本次分配的调试端口，于是表现为
                # 「20s 内未打开可控 PMOS 页面」——极易被误诊成网络故障。
                if proc.returncode == 0:
                    logger.warning(
                        "browser.bootstrap_profile_maybe_locked port=%s profile=%s; "
                        "启动器以 code=0 立即退出，多半是同一 user-data-dir 已被另一个浏览器"
                        "实例占用（新进程不监听本端口）。请先关闭残留的 chrome.exe / "
                        "msedge.exe；切勿改用临时 profile——临时 profile 下 CFCA/UKey "
                        "原生弹窗不出现，认证会卡死在 CERTIFICATE。",
                        self.config.debug_port,
                        getattr(self.config, "browser_profile_dir", "-"),
                    )
                launcher_exited_logged = True

            probe = probe_cdp_port(self.config.debug_port)
            if probe:
                try:
                    pages = self.pages()
                    last_urls = [str(p.get("url") or "")[:180] for p in pages[:5]]
                    if pages:
                        # [V10-r4] PMOS bootstrap render-readiness gate: URL-only is insufficient.
                        probe_data = self.bootstrap_render_probe(timeout=3)
                        last_probe = probe_data if isinstance(probe_data, dict) else {}
                        runtime_url = str(last_probe.get("href") or "")
                        last_runtime_url = runtime_url[:240]
                        if "pmos.sd.sgcc.com.cn" in runtime_url.lower():
                            pmos_runtime_seen = True
                        if is_bootstrap_render_ready(last_probe):
                            logger.info(
                                "browser.bootstrap_ready port=%s runtime_url=%s body_text_length=%s",
                                self.config.debug_port,
                                last_runtime_url,
                                last_probe.get("bodyTextLength", 0),
                            )
                            return {
                                "port": self.config.debug_port,
                                "runtime_url": last_runtime_url,
                                "browser": probe.get("browser", ""),
                            }
                except Exception as exc:  # 页面可能仍在刚创建/导航中。
                    logger.debug("browser.bootstrap_page_waiting: %s", exc)

            time.sleep(0.25)

        if pmos_runtime_seen:
            logger.warning(
                "BROWSER_BOOTSTRAP_RENDER_STALLED port=%s runtime_url=%s probe=%s",
                self.config.debug_port, last_runtime_url or "-", last_probe,
            )
            detail = (
                f"runtime_url={last_runtime_url or '-'} "
                f"targets={last_urls or '-'} probe={last_probe or '-'} "
                f"port={self.config.debug_port}"
            )
            raise TimeoutError(f"BROWSER_BOOTSTRAP_RENDER_STALLED: {detail}")

        detail = (
            f"runtime_url={last_runtime_url or '-'} "
            f"targets={last_urls or '-'} port={self.config.debug_port}"
        )
        raise TimeoutError(
            f"浏览器启动后 {timeout_sec}s 内未打开可控 PMOS 页面；{detail}"
        )

    def _http(self, path: str) -> str:
        return f"http://127.0.0.1:{self.config.debug_port}{path}"

    def pages(self) -> list[dict[str, Any]]:
        response = requests.get(self._http("/json"), timeout=3)
        response.raise_for_status()
        return [
            page for page in response.json()
            if page.get("type") == "page" and page.get("webSocketDebuggerUrl")
        ]

    def target_page(self, *, require_pmos: bool = False) -> dict[str, Any]:
        pages = self.pages()
        matches = [p for p in pages if "pmos.sd.sgcc.com.cn" in str(p.get("url", ""))]
        if matches:
            return matches[-1]
        if require_pmos:
            raise RuntimeError("未找到 PMOS 浏览器标签页")
        # 独立 profile 只由本程序创建。PMOS 导航尚未完成时，先连接新窗口，
        # 让状态机继续等待，避免启动瞬间把临时空白页当作异常。
        if pages:
            return pages[-1]
        raise RuntimeError("浏览器尚未创建可控制的标签页")

    def evaluate(self, expression: str, *, await_promise: bool = False, timeout: int = 10) -> Any:
        return self._command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
            timeout=timeout,
        ).get("result", {}).get("value")

    def bootstrap_render_probe(self, *, timeout: int = 5) -> dict[str, Any]:
        """Return the bounded DOM/runtime probe used by bootstrap and reuse health gates."""
        result = self.evaluate(_BOOTSTRAP_RENDER_PROBE, timeout=timeout)
        return result if isinstance(result, dict) else {}

    def navigate(self, url: str) -> None:
        """Navigate the currently selected CDP page without opening a browser."""
        self._command("Page.navigate", {"url": str(url)}, timeout=10)

    def _command(self, method: str, params: dict[str, Any], *, timeout: int = 10) -> dict[str, Any]:
        ws = websocket.create_connection(self.target_page()["webSocketDebuggerUrl"], timeout=timeout)
        try:
            ws.send(json.dumps({"id": 1, "method": method, "params": params}))
            while True:
                payload = json.loads(ws.recv())
                if payload.get("id") != 1:
                    continue
                if payload.get("error"):
                    raise RuntimeError(f"CDP {method} 失败: {payload['error']}")
                outer = payload.get("result", {})
                if method == "Runtime.evaluate" and outer.get("exceptionDetails"):
                    raise RuntimeError(str(outer["exceptionDetails"]))
                return outer
        finally:
            ws.close()

    def capture_png(self) -> bytes:
        result = self._command("Page.captureScreenshot", {"format": "png"}, timeout=20)
        data = result.get("data")
        if not data:
            raise RuntimeError("CDP 未返回浏览器截图")
        return base64.b64decode(data)

    def drag_mouse(self, start_x: float, start_y: float, end_x: float, end_y: float, duration_ms: int) -> None:
        """通过 Chromium 输入通道拖动，轨迹由滑块求解器给出。"""
        steps = max(12, min(60, duration_ms // 20))
        self._command("Input.dispatchMouseEvent", {
            "type": "mousePressed", "x": start_x, "y": start_y, "button": "left", "clickCount": 1,
        })
        for step in range(1, steps + 1):
            ratio = step / steps
            # 平滑 ease-in-out，避免单次坐标跳跃；不伪造浏览器或设备指纹。
            eased = ratio * ratio * (3 - 2 * ratio)
            self._command("Input.dispatchMouseEvent", {
                "type": "mouseMoved",
                "x": start_x + (end_x - start_x) * eased,
                "y": start_y + (end_y - start_y) * eased,
                "button": "left",
            })
            time.sleep(duration_ms / steps / 1000)
        self._command("Input.dispatchMouseEvent", {
            "type": "mouseReleased", "x": end_x, "y": end_y, "button": "left", "clickCount": 1,
        })

    def cookies(self) -> str:
        ws = websocket.create_connection(self.target_page()["webSocketDebuggerUrl"], timeout=5)
        try:
            ws.send(json.dumps({"id": 1, "method": "Network.getAllCookies", "params": {}}))
            while True:
                payload = json.loads(ws.recv())
                if payload.get("id") == 1:
                    cookies = payload.get("result", {}).get("cookies", [])
                    break
        finally:
            ws.close()
        selected: dict[str, str] = {}
        for item in cookies:
            domain = str(item.get("domain", "")).lower()
            if "sgcc.com.cn" in domain and item.get("name"):
                selected[str(item["name"])] = str(item.get("value", ""))
        return "; ".join(f"{key}={value}" for key, value in sorted(selected.items()))
