from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .browser import CdpSession, launch_browser, resolve_default_browser
from .config import AuthConfig
from .handlers import InteractionHandler, build_pin_handler, build_slider_handler, probe_cfca_service
from .page import PageState, PmosPage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthenticationResult:
    cookie: str
    browser_path: str
    elapsed_sec: float


class AuthenticationStateMachine:
    def __init__(
        self,
        config: AuthConfig,
        *,
        slider_handler: InteractionHandler | None = None,
        pin_handler: InteractionHandler | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.slider_handler = slider_handler or build_slider_handler(config)
        self.pin_handler = pin_handler or build_pin_handler(config)
        self.clock = clock
        self.sleeper = sleeper

    def run(self) -> AuthenticationResult:
        started = self.clock()
        executable = resolve_default_browser(self.config.browser_path)
        profile = Path(self.config.browser_profile_dir).expanduser() if self.config.browser_profile_dir else (
            Path.home() / ".pmos-auto-crawler-profile"
        )
        proc = launch_browser(self.config, executable, profile)
        session = CdpSession(self.config)
        session.wait_ready(proc)
        page = PmosPage(session)
        deadline = started + self.config.login_timeout_sec
        next_login_attempt = 0.0
        next_cfca_attempt = 0.0
        next_login_check = 0.0
        last_state: PageState | None = None
        cfca_submitted = False
        login_submitted = False
        gateway_recovered = False
        transient_since: float | None = None

        launcher_exited_logged = False
        while self.clock() < deadline:
            if proc.poll() is not None and not launcher_exited_logged:
                logger.info("auth.browser_launcher_exited code=%s; continuing_with_devtools=true", proc.returncode)
                launcher_exited_logged = True
            try:
                snapshot = page.snapshot()
                transient_since = None
            except Exception as exc:  # 浏览器导航/DevTools 在低配设备上可短暂不可用。
                now = self.clock()
                transient_since = transient_since if transient_since is not None else now
                waited = now - transient_since
                logger.warning("auth.page_waiting retry_in=%.2fs waited=%.1fs error=%s: %s",
                               self.config.poll_interval_sec, waited, type(exc).__name__, exc)
                if waited >= self.config.transient_error_timeout_sec:
                    raise TimeoutError(
                        f"浏览器页面连续 {waited:.0f}s 不可控制；最后错误={type(exc).__name__}: {exc}"
                    ) from exc
                self.sleeper(self.config.poll_interval_sec)
                continue
            if snapshot.state != last_state:
                logger.info("auth.state from=%s to=%s url=%s %s", last_state, snapshot.state.value,
                            snapshot.url[:160], snapshot.detail)
                last_state = snapshot.state

            # 顺序非常重要：证书弹层 > 可见滑块 > 登录表单。初始登录页即使预置隐藏
            # 滑块 DOM，也会先提交账号密码；实际滑块弹出后 slider_visible 才会接管。
            if snapshot.state == PageState.GATEWAY_ERROR:
                if not gateway_recovered:
                    gateway_recovered = page.recover_from_gateway_error()
                    logger.warning("auth.gateway_502 recovered_by_history_back=%s", gateway_recovered)
                else:
                    raise RuntimeError("认证后交易入口持续返回 502；已自动回退一次，请检查 PMOS 网关")
            elif snapshot.certificate_visible:
                if not cfca_submitted and self.clock() >= next_cfca_attempt:
                    available = probe_cfca_service(self.config.cfca_port)
                    logger.info("cfca.service available=%s port=%s", available, self.config.cfca_port)
                    cfca_result = page.select_cfca_and_verify()
                    cfca_submitted = bool(cfca_result.get("ok"))
                    logger.info("cfca.web_submit ok=%s reason=%s", cfca_submitted,
                                cfca_result.get("reason"))
                    next_cfca_attempt = self.clock() + self.config.cfca_retry_interval_sec
                if cfca_submitted:
                    self.pin_handler.handle(session, snapshot, self.config)
            elif snapshot.slider_visible:
                self.slider_handler.handle(session, snapshot, self.config)
            elif snapshot.login_form and not login_submitted and self.clock() >= next_login_attempt:
                result = page.submit_login(self.config.resolved_username, self.config.resolved_password)
                logger.info("auth.login_submit ok=%s reason=%s", result.get("ok"), result.get("reason"))
                # 一次登录提交会触发页面重绘与验证码生成。此后即使滑块 DOM 短暂
                # 不可见，也绝不再次点击登录，避免人工拖动时刷新验证码。
                login_submitted = bool(result.get("ok"))
                next_login_attempt = self.clock() + self.config.login_retry_interval_sec
            elif snapshot.state == PageState.LOGGED_IN or (
                cfca_submitted and not snapshot.certificate_visible and not snapshot.slider_visible
            ):
                if self.clock() >= next_login_check:
                    try:
                        cookie = session.cookies()
                        if self.check_login(cookie):
                            return AuthenticationResult(cookie, str(executable), self.clock() - started)
                    except Exception as exc:
                        logger.warning("auth.login_check_waiting error=%s: %s", type(exc).__name__, exc)
                    next_login_check = self.clock() + self.config.login_check_interval_sec
            self.sleeper(self.config.poll_interval_sec)
        raise TimeoutError(f"PMOS 登录在 {self.config.login_timeout_sec}s 内未完成，最后状态={last_state}")

    def check_login(self, cookie: str) -> bool:
        """认证完成只校验会话 Cookie，不再访问会造成 502 的旧交易入口。"""
        if not cookie:
            return False
        cookie_names = {part.split("=", 1)[0].strip() for part in cookie.split(";") if "=" in part}
        has_session = bool(cookie_names & {"Admin-Token", "X-Ticket", "JSESSIONID", "XHXT_SESSIONID"})
        logger.info("auth.session_cookie present=%s", has_session)
        return has_session
