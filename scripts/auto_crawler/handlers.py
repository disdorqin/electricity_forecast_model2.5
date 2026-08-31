from __future__ import annotations

import ctypes
import importlib
import logging
import socket
import sys
from ctypes import wintypes
from typing import Protocol

from .browser import CdpSession
from .config import AuthConfig
from .page import PageSnapshot

logger = logging.getLogger(__name__)


class InteractionHandler(Protocol):
    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool: ...


class ManualSliderHandler:
    def __init__(self):
        self._announced = False

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if not self._announced:
            logger.warning("auth.waiting_for_human action=slider")
            self._announced = True
        return False


class ManualPinHandler:
    def __init__(self):
        self._announced = False

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if not self._announced:
            logger.warning("auth.waiting_for_human action=ukey_pin")
            self._announced = True
        return False


class WindowsPinHandler:
    """使用精确窗口标题和子控件提交 PIN，不使用全局键盘或剪贴板。"""

    WM_SETTEXT = 0x000C
    BM_CLICK = 0x00F5

    def __init__(self):
        self._submitted = False

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if self._submitted:
            return True
        if sys.platform != "win32":
            raise RuntimeError("Windows PIN 自动处理器只能在 Windows 上运行")
        pin = config.resolved_pin
        if not pin:
            logger.warning("pin.auto_disabled reason=missing_environment_variable env=%s", config.pin_env)
            return False
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, config.ukey_window_title)
        if not hwnd:
            return False
        children: list[tuple[int, str, str]] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def collect(child: int, _param: int) -> bool:
            cls = ctypes.create_unicode_buffer(128)
            text = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(child, cls, len(cls))
            user32.GetWindowTextW(child, text, len(text))
            children.append((int(child), cls.value, text.value))
            return True

        user32.EnumChildWindows(hwnd, callback_type(collect), 0)
        edits = [item for item in children if item[1].lower() == "edit"]
        confirms = [item for item in children if item[2].replace(" ", "") in {"确定", "确认"}]
        if len(edits) != 1 or len(confirms) != 1:
            logger.error("pin.window_ambiguous edits=%s confirms=%s", len(edits), len(confirms))
            return False
        user32.SendMessageW(edits[0][0], self.WM_SETTEXT, 0, pin)
        user32.SendMessageW(confirms[0][0], self.BM_CLICK, 0, 0)
        logger.info("pin.submitted window=%s", config.ukey_window_title)
        self._submitted = True
        return True


def probe_cfca_service(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _load_plugin(spec: str) -> InteractionHandler:
    if ":" not in spec:
        raise ValueError("插件格式必须为 package.module:factory")
    module_name, factory_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory()


def build_slider_handler(config: AuthConfig) -> InteractionHandler:
    if config.slider_handler == "manual":
        return ManualSliderHandler()
    if config.slider_handler == "plugin":
        return _load_plugin(config.slider_plugin)
    raise ValueError(f"不支持 slider_handler={config.slider_handler!r}")


def build_pin_handler(config: AuthConfig) -> InteractionHandler:
    if config.pin_handler == "manual":
        return ManualPinHandler()
    if config.pin_handler == "windows":
        return WindowsPinHandler()
    if config.pin_handler == "plugin":
        return _load_plugin(config.pin_plugin)
    raise ValueError(f"不支持 pin_handler={config.pin_handler!r}")
