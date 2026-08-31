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
        last_state: PageState | None = None
        cfca_submitted = False

        while self.clock() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"浏览器在认证完成前退出，code={proc.returncode}")
            snapshot = page.snapshot()
            if snapshot.state != last_state:
                logger.info("auth.state from=%s to=%s url=%s", last_state, snapshot.state.value, snapshot.url[:160])
                last_state = snapshot.state

            if snapshot.state == PageState.LOGIN_READY and self.clock() >= next_login_attempt:
                result = page.submit_login(self.config.resolved_username, self.config.resolved_password)
                logger.info("auth.login_submit ok=%s reason=%s", result.get("ok"), result.get("reason"))
                next_login_attempt = self.clock() + self.config.login_retry_interval_sec
            elif snapshot.state == PageState.SLIDER:
                self.slider_handler.handle(session, snapshot, self.config)
            elif snapshot.state == PageState.CERTIFICATE:
                if not cfca_submitted:
                    available = probe_cfca_service(self.config.cfca_port)
                    logger.info("cfca.service available=%s port=%s", available, self.config.cfca_port)
                    cfca_submitted = page.select_cfca_and_verify()
                    logger.info("cfca.web_submit ok=%s", cfca_submitted)
                self.pin_handler.handle(session, snapshot, self.config)
            elif snapshot.state == PageState.LOGGED_IN:
                cookie = session.cookies()
                if self.check_login(page, cookie):
                    return AuthenticationResult(cookie, str(executable), self.clock() - started)
            self.sleeper(self.config.poll_interval_sec)
        raise TimeoutError(f"PMOS 登录在 {self.config.login_timeout_sec}s 内未完成，最后状态={last_state}")

    def check_login(self, page: PmosPage, cookie: str) -> bool:
        """同时要求认证信息与交易接口可访问，避免只凭 URL/Cookie 误判。"""
        if not cookie:
            return False
        cookie_names = {part.split("=", 1)[0].strip() for part in cookie.split(";") if "=" in part}
        has_session = bool(cookie_names & {"Admin-Token", "X-Ticket", "JSESSIONID", "XHXT_SESSIONID"})
        probe_ok = page.probe_authenticated(self.config.trade_base, self.config.success_probe_paths)
        logger.info("auth.probe pass=%s has_session_cookie=%s", probe_ok, has_session)
        return bool(probe_ok and has_session)
