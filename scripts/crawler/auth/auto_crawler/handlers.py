from __future__ import annotations

import base64
import ctypes
import hashlib
import importlib
from io import BytesIO
import json
import logging
import socket
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

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


class TemplateSliderSolver:
    """用当前验证码的背景图和拼图块实时模板匹配，不依赖历史答案或训练集。"""

    def solve(self, *, screenshot_png: bytes, geometry: dict, config: AuthConfig) -> SliderDrag | None:
        images = geometry.get("images") or []
        decoded: list[tuple[dict, bytes]] = []
        for item in images:
            source = str(item.get("src") or "")
            if not source.startswith("data:image") or "," not in source:
                continue
            try:
                decoded.append((item, base64.b64decode(source.split(",", 1)[1])))
            except (ValueError, base64.binascii.Error):
                continue
        if len(decoded) < 2:
            logger.warning("slider.template_missing_images count=%s", len(decoded))
            return None

        try:
            parsed = [(item, raw, Image.open(BytesIO(raw))) for item, raw in decoded]
            background_item, background_raw, background = max(parsed, key=lambda x: x[2].size[0])
            candidates = [entry for entry in parsed if entry[0] is not background_item]
            piece_item, piece_raw, piece = min(candidates, key=lambda x: x[2].size[0])
            background_rgb = np.asarray(background.convert("RGB"), dtype=np.float64)
            piece_rgba = np.asarray(piece.convert("RGBA"), dtype=np.float64)
        except Exception as exc:
            logger.warning("slider.template_decode_failed error=%s: %s", type(exc).__name__, exc)
            return None

        image_height, image_width = background_rgb.shape[:2]
        piece_height, piece_width = piece_rgba.shape[:2]
        if piece_width < 10 or piece_height != image_height or image_width <= piece_width:
            logger.warning("slider.template_dimensions_invalid background=%sx%s piece=%sx%s",
                           image_width, image_height, piece_width, piece_height)
            return None
        mask = piece_rgba[:, :, 3] > 50
        if not mask.any():
            logger.warning("slider.template_piece_is_transparent")
            return None
        piece_rgb = piece_rgba[:, :, :3]
        scores = np.empty(image_width - piece_width + 1, dtype=np.float64)
        for x in range(len(scores)):
            region = background_rgb[:piece_height, x:x + piece_width, :]
            scores[x] = (np.abs(region - piece_rgb) * mask[:, :, None]).sum()
        best_x = int(np.argmin(scores))
        # 与旧 HAR 验证实现一致：插值提高精度，再扣除前端上报的约 1.7px 校准量。
        best_subpixel = float(best_x)
        if 0 < best_x < len(scores) - 1:
            left, middle, right = scores[best_x - 1], scores[best_x], scores[best_x + 1]
            denom = 2.0 * (left + right - 2.0 * middle)
            if abs(denom) > 1e-12:
                best_subpixel += max(-1.0, min(1.0, (left - right) / denom))
        captcha_x = max(0.0, best_subpixel - 1.7)
        css_width = float(background_item.get("width") or image_width)
        offset_x = captcha_x * css_width / image_width
        ordered = np.partition(scores, 1)
        confidence = float((ordered[1] - ordered[0]) / max(ordered[0], 1.0))
        reason = "template gap=%.2f/%s css_offset=%.2f confidence=%.3f" % (
            captcha_x, image_width, offset_x, confidence,
        )
        return SliderDrag(offset_x=offset_x, confidence=max(confidence, 0.001), reason=reason)


class ManualSliderHandler:
    def __init__(self):
        self._announced = False

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if not self._announced:
            logger.warning("auth.waiting_for_human action=slider")
            self._announced = True
        return False


class CaptureSliderHandler(ManualSliderHandler):
    """人工验证时保存当前滑块的最小诊断样本，供自动识别器接入和校准。"""

    def __init__(self):
        super().__init__()
        self._captured = False

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if not self._captured:
            try:
                screenshot = session.capture_png()
                metadata = _slider_capture_metadata(session)
                directory = _write_slider_artifact(screenshot, metadata, config)
                logger.info("slider.sample_saved directory=%s images=%s", directory,
                            len(metadata.get("images", [])))
                self._captured = True
            except Exception as exc:
                logger.warning("slider.sample_capture_failed error=%s: %s", type(exc).__name__, exc)
        return super().handle(session, snapshot, config)


class BrowserSliderHandler:
    """收集现场样本、调用本地识别插件，并在浏览器内执行插件给出的拖拽。"""

    def __init__(self, solver: SliderSolver):
        self.solver = solver
        self.attempts = 0
        self._last_challenge = ""
        self._last_drag_at = 0.0
        self._manual_fallback_announced = False
        self._geometry_missing_logged = False

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if self.attempts >= config.slider_max_attempts:
            logger.warning("slider.auto_paused attempts=%s", self.attempts)
            return False
        metadata = _slider_capture_metadata(session)
        geometry = _slider_geometry(session) or _slider_geometry_from_images(metadata)
        if not geometry:
            if not self._geometry_missing_logged:
                logger.error("slider.geometry_missing; waiting_for_manual_slider")
                self._geometry_missing_logged = True
            return False
        self._geometry_missing_logged = False
        geometry["images"] = metadata.get("images", [])
        challenge = hashlib.sha256(
            "|".join(str(item.get("src") or "") for item in geometry["images"]).encode("utf-8")
        ).hexdigest()[:16]
        now = time.monotonic()
        if challenge == self._last_challenge and now - self._last_drag_at < config.slider_result_wait_sec:
            return False
        if challenge == self._last_challenge:
            if not self._manual_fallback_announced:
                logger.warning("auth.waiting_for_human action=slider reason=template_result_not_changed challenge=%s",
                               challenge)
                self._manual_fallback_announced = True
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
        self._last_challenge = challenge
        self._last_drag_at = now
        self._manual_fallback_announced = False
        logger.info("slider.drag attempt=%s offset=%.1f confidence=%.3f reason=%s",
                    self.attempts, offset, drag.confidence,
                    "%s geometry=%s" % (drag.reason, geometry.get("source", "dom")))
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
    """只向精确 UKey 窗口的编辑框输入 PIN，并以确认键或 Enter 提交。

    2026-09-16 现场反馈「UKey 还是要人工输入」的根因是配置里没有 PIN
    （日志 ``pin.handler_fallback=manual reason=pin_not_configured``），
    不是自动输入逻辑本身失效。这里同时加固了几处会在真机上静默失败的地方：

    * 标题不完全一致时，按 UKey/口令 关键字模糊查找顶层窗口；
    * 只认**可见**的 Edit 控件（隐藏辅助输入框曾导致 ``len(edits) != 1`` 直接放弃）；
    * 写入后回读校验，读不到长度就退回逐字符 ``WM_CHAR``；
    * 找不到「确定/确认/提交」按钮时退回 Enter，而不是直接报 ambiguous；
    * 弹窗被拒后允许重试，不再一次提交失败就永久停在等待状态。
    """

    WM_SETTEXT = 0x000C
    WM_GETTEXTLENGTH = 0x000E
    BM_CLICK = 0x00F5
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    WM_CHAR = 0x0102
    VK_RETURN = 0x0D
    MAX_ATTEMPTS = 3

    def __init__(self):
        self._submitted = False
        self._attempts = 0
        self._missing_pin_announced = False
        self._ambiguous_announced = False

    @staticmethod
    def _find_pin_window(user32, title: str) -> int:
        """先精确匹配标题，失败再按 UKey/口令 关键字兜底查找（唯一命中才采用）。"""
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return int(hwnd)
        found: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def visit(item: int, _param: int) -> bool:
            text = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(item, text, len(text))
            name = text.value or ""
            if ("UKey" in name or "ukey" in name.lower() or "口令" in name) and user32.IsWindowVisible(item):
                found.append(int(item))
            return True

        user32.EnumWindows(callback_type(visit), 0)
        return found[0] if len(found) == 1 else 0

    @staticmethod
    def _enum_children(user32, hwnd: int) -> list[tuple[int, str, str]]:
        children: list[tuple[int, str, str]] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def collect(child: int, _param: int) -> bool:
            cls = ctypes.create_unicode_buffer(128)
            text = ctypes.create_unicode_buffer(512)
            user32.GetClassNameW(child, cls, len(cls))
            user32.GetWindowTextW(child, text, len(text))
            children.append((int(child), cls.value, text.value))
            return True

        user32.EnumChildWindows(hwnd, callback_type(collect), 0)
        return children

    def _type_pin(self, user32, edit: int, pin: str) -> bool:
        """写入 PIN 并回读校验；回读不到就逐字符补发 WM_CHAR。"""
        user32.SendMessageW(edit, self.WM_SETTEXT, 0, pin)
        length = int(user32.SendMessageW(edit, self.WM_GETTEXTLENGTH, 0, 0) or 0)
        if length <= 0:
            for ch in pin:
                user32.PostMessageW(edit, self.WM_CHAR, ord(ch), 0)
            length = int(user32.SendMessageW(edit, self.WM_GETTEXTLENGTH, 0, 0) or 0)
        return length > 0

    def handle(self, session: CdpSession, snapshot: PageSnapshot, config: AuthConfig) -> bool:
        if sys.platform != "win32":
            raise RuntimeError("Windows PIN 自动处理器只能在 Windows 上运行")
        pin = config.resolved_pin
        if not pin:
            if not self._missing_pin_announced:
                logger.warning(
                    "pin.auto_disabled reason=pin_not_configured env=%s; waiting_for_manual_ukey_input"
                    "（在 config.json 填 ukey_pin，或设置该环境变量后即可自动输入）",
                    config.pin_env,
                )
                self._missing_pin_announced = True
            return False
        if self._submitted and self._attempts >= self.MAX_ATTEMPTS:
            return True

        user32 = ctypes.windll.user32
        hwnd = self._find_pin_window(user32, config.ukey_window_title)
        if not hwnd:
            # 已经提交过且弹窗消失，说明 PIN 被接受。
            return bool(self._submitted)
        if self._submitted:
            logger.warning("pin.window_still_present attempt=%s，重新尝试输入 PIN", self._attempts)

        children = self._enum_children(user32, hwnd)
        edits = [
            item for item in children
            if item[1].lower() == "edit" and user32.IsWindowVisible(item[0])
        ]
        confirms = [
            item for item in children
            if item[2].replace(" ", "") in {"确定", "确认", "提交", "OK"}
            and user32.IsWindowVisible(item[0])
        ]
        if not edits:
            logger.error("pin.window_no_edit hwnd=%s children=%s", hwnd, children[:12])
            return False
        if len(edits) > 1 and not self._ambiguous_announced:
            self._ambiguous_announced = True
            logger.warning(
                "pin.window_multiple_edits edits=%s confirms=%s children=%s",
                len(edits), len(confirms), children[:12],
            )
        if config.pin_submit_mode not in {"click", "enter"}:
            raise ValueError("pin_submit_mode 仅支持 click 或 enter")

        self._attempts += 1
        if not self._type_pin(user32, edits[0][0], pin):
            logger.error("pin.settext_failed hwnd=%s children=%s", hwnd, children[:12])
            return False
        user32.SetFocus(edits[0][0])
        if config.pin_submit_mode == "click" and confirms:
            user32.SendMessageW(confirms[0][0], self.BM_CLICK, 0, 0)
            mode = "click"
        else:
            if config.pin_submit_mode == "click":
                logger.warning("pin.confirm_button_missing fallback=enter confirms=%s children=%s",
                               len(confirms), children[:12])
            # 定向发送给该原生对话框；不影响当前电脑的其他应用或键盘焦点。
            user32.PostMessageW(hwnd, self.WM_KEYDOWN, self.VK_RETURN, 0)
            user32.PostMessageW(hwnd, self.WM_CHAR, self.VK_RETURN, 0)
            user32.PostMessageW(hwnd, self.WM_KEYUP, self.VK_RETURN, 0)
            mode = "enter"
        logger.info("pin.submitted window=%s title=%s mode=%s attempt=%s",
                    hwnd, config.ukey_window_title, mode, self._attempts)
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
      const visible = el => {
        if (!el || !(el.offsetWidth || el.offsetHeight || el.getClientRects().length)) return false;
        const s = getComputedStyle(el), r = el.getBoundingClientRect();
        return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity || 1) > .01
          && r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
      };
      const norm = s => (s || '').replace(/\\s+/g, '');
      const hint = [...document.querySelectorAll('*')].find(x => visible(x) && norm(x.innerText) === '向右滑动完成验证');
      if (!hint) return null;
      // 旧版/新版 PMOS 的 class 名不稳定；按提示文字的最近“滑轨尺寸”祖先定位。
      const ancestors = [];
      for (let node = hint; node && node !== document.body; node = node.parentElement) {
        if (!visible(node)) continue;
        const r = node.getBoundingClientRect();
        if (r.width >= 200 && r.width <= 700 && r.height >= 36 && r.height <= 110) ancestors.push(node);
      }
      const track = ancestors[0];
      const t = track?.getBoundingClientRect();
      let handle = track && [track, ...track.querySelectorAll('*')].filter(visible).map(x => [x, x.getBoundingClientRect()])
        .filter(([, r]) => r.width >= 30 && r.width <= 110 && r.height >= t.height * .5 && r.height <= t.height * 1.5
          && r.left >= t.left - 3 && r.left <= t.left + 15)
        .sort((a, b) => a[1].left - b[1].left)[0]?.[0];
      if (!track || !handle) return null;
      const h = handle.getBoundingClientRect();
      if (t.width < 80 || h.width < 10) return null;
      return {track_x:t.x, track_y:t.y, track_width:t.width, track_height:t.height,
        handle_x:h.x, handle_y:h.y, handle_width:h.width, handle_height:h.height,
        handle_center_x:h.x+h.width/2, handle_center_y:h.y+h.height/2,
        viewport_width:innerWidth, viewport_height:innerHeight, source:'dom'};
    })()""")


def _slider_geometry_from_images(metadata: dict) -> dict | None:
    """DOM 控件类名变化时，以当前验证码图片的实际布局推导滑轨坐标。"""
    images = metadata.get("images") or []
    if not images:
        return None
    try:
        background = max(images, key=lambda item: float(item.get("width") or 0))
        x, y = float(background["x"]), float(background["y"])
        width, height = float(background["width"]), float(background["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width < 100 or height < 80:
        return None
    # PMOS 滑轨与验证码背景同宽、紧接在背景下方；全部是相对当前图片的 CSS 像素。
    # 这会随显示缩放、分辨率和窗口位置实时变化，不使用屏幕绝对坐标。
    track_height = max(40.0, min(80.0, height * 0.42))
    handle_width = max(40.0, min(80.0, height * 0.42))
    track_y = y + height + max(4.0, height * 0.04)
    return {
        "track_x": x,
        "track_y": track_y,
        "track_width": width,
        "track_height": track_height,
        "handle_x": x,
        "handle_y": track_y,
        "handle_width": handle_width,
        "handle_height": track_height,
        "handle_center_x": x + handle_width / 2,
        "handle_center_y": track_y + track_height / 2,
        "source": "captcha_image_relative",
    }


def _slider_capture_metadata(session: CdpSession) -> dict:
    """仅导出当前滑块弹层的布局和图片来源，不保存账号、密码或整页 HTML。"""
    return session.evaluate("""(() => {
      const visible = el => {
        if (!el || !(el.offsetWidth || el.offsetHeight || el.getClientRects().length)) return false;
        const s = getComputedStyle(el), r = el.getBoundingClientRect();
        return s.display !== 'none' && s.visibility !== 'hidden' && Number(s.opacity || 1) > .01
          && r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
      };
      const norm = s => (s || '').replace(/\\s+/g, '');
      const hint = [...document.querySelectorAll('*')].find(x => visible(x)
        && norm(x.innerText) === '向右滑动完成验证');
      const root = hint?.closest('.el-dialog, [role="dialog"], [class*=captcha], [class*=verify]') || hint?.parentElement;
      const images = root ? [...root.querySelectorAll('img')].filter(visible).map(img => {
        const r = img.getBoundingClientRect();
        return {src: img.currentSrc || img.src || '', width:r.width, height:r.height, x:r.x, y:r.y};
      }) : [];
      const backgrounds = root ? [...root.querySelectorAll('*')].filter(visible).map(el => {
        const r = el.getBoundingClientRect(), bg = getComputedStyle(el).backgroundImage;
        return bg && bg !== 'none' ? {background:bg, width:r.width, height:r.height, x:r.x, y:r.y} : null;
      }).filter(Boolean) : [];
      return {captured_at:new Date().toISOString(), hint_found:!!hint, root_class:root?.className || '', images, backgrounds};
    })()""") or {}


def _write_slider_artifact(screenshot: bytes, geometry: dict, config: AuthConfig) -> Path:
    directory = Path(config.slider_artifact_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (directory / f"slider_{stamp}.png").write_bytes(screenshot)
    (directory / f"slider_{stamp}.json").write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


def build_slider_handler(config: AuthConfig) -> InteractionHandler:
    if config.slider_handler == "manual":
        return ManualSliderHandler()
    if config.slider_handler == "capture":
        return CaptureSliderHandler()
    if config.slider_handler == "plugin":
        return _load_plugin(config.slider_plugin)
    if config.slider_handler == "browser":
        return BrowserSliderHandler(_load_slider_solver(config.slider_plugin))
    if config.slider_handler == "template":
        return BrowserSliderHandler(TemplateSliderSolver())
    raise ValueError(f"不支持 slider_handler={config.slider_handler!r}")


def build_pin_handler(config: AuthConfig) -> InteractionHandler:
    if config.pin_handler == "manual":
        return ManualPinHandler()
    if config.pin_handler == "windows":
        # 不把 PIN 写入配置时，自动降级为人工输入；否则 Windows 处理器会
        # 在每个轮询周期重复告警，但并不会比人工处理更进一步。
        if not config.resolved_pin:
            logger.warning(
                "pin.handler_fallback=manual reason=pin_not_configured env=%s "
                "action=fill_config_ukey_pin_or_set_env（否则每次登录都要人工在 UKey 弹窗里输 PIN）",
                config.pin_env,
            )
            return ManualPinHandler()
        return WindowsPinHandler()
    if config.pin_handler == "plugin":
        return _load_plugin(config.pin_plugin)
    raise ValueError(f"不支持 pin_handler={config.pin_handler!r}")
