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

logger = logging.getLogger(__name__)


class BrowserResolutionError(RuntimeError):
    pass


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


def ensure_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"调试端口 {port} 已被占用；请关闭旧爬虫浏览器或更换 debug_port")


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
        config.login_url,
    ]
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

    def wait_ready(self, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.config.browser_ready_timeout_sec
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"浏览器提前退出，code={proc.returncode}")
            try:
                response = requests.get(self._http("/json/version"), timeout=2)
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise TimeoutError("浏览器 DevTools 启动超时")

    def _http(self, path: str) -> str:
        return f"http://127.0.0.1:{self.config.debug_port}{path}"

    def pages(self) -> list[dict[str, Any]]:
        response = requests.get(self._http("/json"), timeout=3)
        response.raise_for_status()
        return [
            page for page in response.json()
            if page.get("type") == "page" and page.get("webSocketDebuggerUrl")
        ]

    def target_page(self) -> dict[str, Any]:
        pages = self.pages()
        matches = [p for p in pages if "pmos.sd.sgcc.com.cn" in str(p.get("url", ""))]
        if not matches:
            raise RuntimeError("未找到 PMOS 浏览器标签页")
        return matches[-1]

    def evaluate(self, expression: str, *, await_promise: bool = False, timeout: int = 10) -> Any:
        return self._command(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
            timeout=timeout,
        ).get("result", {}).get("value")

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
