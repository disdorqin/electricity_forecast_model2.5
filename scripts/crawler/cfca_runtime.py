# -*- coding: utf-8 -*-
"""公司电脑上的 CFCA/UKey 运行时辅助。

该模块不读取、导出或复制 UKey 私钥，只做三件事：
1. 检查 CryptoKit 本地服务是否存在；
2. 将账号密码填入 Edge 登录页；
3. 在驱动弹出原生证书/PIN 窗口时，按配置执行有限的 UI 操作。

不同单位的 CFCA 驱动窗口标题和控件实现可能不同，因此所有操作都必须
写入追加日志；无法识别时保持窗口可见并返回 False，而不是误判登录成功。
"""
from __future__ import annotations

import ctypes
import logging
import os
import socket
import time
from ctypes import wintypes
from typing import Any

logger = logging.getLogger(__name__)


def probe_local_cryptokit(port: int = 7693) -> bool:
    """探测 HAR 中确认的 CryptoKit 本地端口，不发送证书操作命令。"""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
            logger.info("[cfca] CryptoKit 本地服务可连接: 127.0.0.1:%s", port)
            return True
    except OSError as exc:
        logger.warning("[cfca] CryptoKit 本地服务不可连接: 127.0.0.1:%s (%s)", port, exc)
        return False


def configured_pin(cfg: dict[str, Any]) -> str:
    """优先环境变量，兼容定时任务；不在日志中输出 PIN。"""
    env_name = str(cfg.get("ukey_pin_env") or "PMOS_UKEY_PIN").strip()
    value = os.environ.get(env_name, "")
    if not value:
        value = str(cfg.get("ukey_pin") or "")
    return value.strip()


def _window_titles() -> list[tuple[int, str]]:
    user32 = ctypes.windll.user32
    result: list[tuple[int, str]] = []
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(max(n + 1, 2))
        user32.GetWindowTextW(hwnd, buf, len(buf))
        title = buf.value.strip()
        if title:
            result.append((int(hwnd), title))
        return True

    user32.EnumWindows(enum_proc(callback), 0)
    return result


def _send_vk(vk: int) -> None:
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, 2, 0)


def _send_text(text: str) -> None:
    # PIN 通常为数字；使用 Unicode 输入避免剪贴板污染公司电脑。
    user32 = ctypes.windll.user32
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002
    for ch in text:
        user32.keybd_event(0, ord(ch), KEYEVENTF_UNICODE, 0)
        user32.keybd_event(0, ord(ch), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0)


def assist_native_ukey_dialog(cfg: dict[str, Any], timeout_sec: int = 90) -> bool:
    """尝试处理 CFCA 原生窗口；无法识别时不阻塞主进程。"""
    pin = configured_pin(cfg)
    if not pin:
        logger.info("[cfca] 未配置 UKey PIN，等待浏览器/驱动自行完成证书认证")
        return False
    keywords = ("CFCA", "CryptoKit", "UKey", "UK", "证书", "数字证书", "密码", "PIN")
    deadline = time.time() + max(5, int(timeout_sec))
    handled = False
    while time.time() < deadline:
        for hwnd, title in _window_titles():
            if not any(k.lower() in title.lower() for k in keywords):
                continue
            try:
                user32 = ctypes.windll.user32
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.2)
                # 证书选择窗口通常直接回车选择默认证书；PIN 窗口先输入 PIN。
                if any(k.lower() in title.lower() for k in ("密码", "pin", "ukey", "uk")):
                    _send_text(pin)
                    _send_vk(0x0D)  # ENTER
                    logger.info("[cfca] 已向疑似 UKey/PIN 窗口提交 PIN（值不写日志）")
                else:
                    _send_vk(0x0D)
                    logger.info("[cfca] 已向疑似证书选择窗口选择默认证书")
                handled = True
                time.sleep(1.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[cfca] 原生窗口辅助失败 title=%s: %s", title, exc)
        if handled:
            return True
        time.sleep(0.5)
    return False
