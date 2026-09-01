from __future__ import annotations

import ctypes
import importlib
import json
import logging
import socket
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .browser import CdpSession
from .config import AuthConfig
from .page import PageSnapshot

logger = logging.getLogger(__name__)


class InteractionHandler(Protocol):
    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool: ...


@dataclass(frozen=True)
class SliderDrag:
    """滑块插件的唯一返回值；offset 是从滑块中心到目标中心的像素距离。"""

    offset_x: float
    confidence: float
    reason: str = ""


class SliderSolver(Protocol):
    def solve(self, *, screenshot_png: bytes, geometry: dict, config: AuthConfig) -> SliderDrag | None: ...


class ManualSliderHandler:
    def __init__(self):
        self._announced = False

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if not self._announced:
            logger.warning("auth.waiting_for_human action=slider")
            self._announced = True
        return False


class BrowserSliderHandler:
    """收集现场样本、调用本地识别插件，并在浏览器内执行插件给出的拖拽。"""

    def __init__(self, solver: SliderSolver):
        self.solver = solver
        self.attempts = 0

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if self.attempts >= config.slider_max_attempts:
            logger.warning("slider.auto_paused attempts=%s", self.attempts)
            return False
        geometry = _slider_geometry(session)
        if not geometry:
            logger.error("slider.geometry_missing")
            return False
        screenshot = session.capture_png()
        _write_slider_artifact(screenshot, geometry, config)
        drag = self.solver.solve(screenshot_png=screenshot, geometry=geometry, config=config)
        if not drag or drag.confidence <= 0:
            logger.warning("slider.solve_unavailable")
            return False
        max_offset = geometry["track_width"] - geometry["handle_width"]
        offset = max(0.0, min(float(drag.offset_x), max_offset))
        self.attempts += 1
        logger.info("slider.drag attempt=%s offset=%.1f confidence=%.3f reason=%s",
                    self.attempts, offset, drag.confidence, drag.reason)
        session.drag_mouse(
            geometry["handle_center_x"], geometry["handle_center_y"],
            geometry["handle_center_x"] + offset, geometry["handle_center_y"],
            config.slider_drag_duration_ms,
        )
        return True


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


def _load_slider_solver(spec: str) -> SliderSolver:
    if ":" not in spec:
        raise ValueError("滑块识别插件格式必须为 package.module:factory")
    module_name, factory_name = spec.split(":", 1)
    solver = getattr(importlib.import_module(module_name), factory_name)()
    if not callable(getattr(solver, "solve", None)):
        raise TypeError("滑块识别插件必须实现 solve(screenshot_png, geometry, config)")
    return solver


def _slider_geometry(session: CdpSession) -> dict | None:
    return session.evaluate("""(() => {
      const visible = el => !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
      const hint = [...document.querySelectorAll('*')].find(x => visible(x) && /向右滑动完成验证/.test(x.textContent || ''));
      if (!hint) return null;
      let track = hint.closest('.el-slider, .slider, [class*=slider]') || hint.parentElement;
      let handle = track && track.querySelector('.el-slider__button, .slider-handle, [class*=handle], [class*=btn]');
      if (!handle && track) handle = [...track.querySelectorAll('*')].find(x => visible(x) && x.getBoundingClientRect().width >= 20 && x.getBoundingClientRect().width <= 80);
      if (!track || !handle) return null;
      const t = track.getBoundingClientRect(), h = handle.getBoundingClientRect();
      if (t.width < 80 || h.width < 10) return null;
      return {track_x:t.x, track_y:t.y, track_width:t.width, track_height:t.height,
        handle_x:h.x, handle_y:h.y, handle_width:h.width, handle_height:h.height,
        handle_center_x:h.x+h.width/2, handle_center_y:h.y+h.height/2,
        viewport_width:innerWidth, viewport_height:innerHeight};
    })()""")


def _write_slider_artifact(screenshot: bytes, geometry: dict, config: AuthConfig) -> None:
    directory = Path(config.slider_artifact_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (directory / f"slider_{stamp}.png").write_bytes(screenshot)
    (directory / f"slider_{stamp}.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")


def build_slider_handler(config: AuthConfig) -> InteractionHandler:
    if config.slider_handler == "manual":
        return ManualSliderHandler()
    if config.slider_handler == "plugin":
        return _load_plugin(config.slider_plugin)
    if config.slider_handler == "browser":
        return BrowserSliderHandler(_load_slider_solver(config.slider_plugin))
    raise ValueError(f"不支持 slider_handler={config.slider_handler!r}")


def build_pin_handler(config: AuthConfig) -> InteractionHandler:
    if config.pin_handler == "manual":
        return ManualPinHandler()
    if config.pin_handler == "windows":
        return WindowsPinHandler()
    if config.pin_handler == "plugin":
        return _load_plugin(config.pin_plugin)
    raise ValueError(f"不支持 pin_handler={config.pin_handler!r}")
