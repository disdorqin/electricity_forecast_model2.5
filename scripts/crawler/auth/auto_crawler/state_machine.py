from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .browser import (
    CdpSession,
    choose_free_debug_port,
    discover_existing_cdp,
    launch_browser,
    resolve_default_browser,
)
from .config import AuthConfig
from .handlers import InteractionHandler, build_pin_handler, build_slider_handler, probe_cfca_service
from .page import PageState, PmosPage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthenticationResult:
    cookie: str
    browser_path: str
    elapsed_sec: float
    debug_port: int = 0


class AuthenticationStateMachine:
    def __init__(
        self,
        config: AuthConfig,
        *,
        slider_handler: InteractionHandler | None = None,
        pin_handler: InteractionHandler | None = None,
        reporter=None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.slider_handler = slider_handler or build_slider_handler(config)
        self.pin_handler = pin_handler or build_pin_handler(config)
        self.reporter = reporter
        self.clock = clock
        self.sleeper = sleeper

    def run(self) -> AuthenticationResult:
        started = self.clock()
        executable = resolve_default_browser(self.config.browser_path)
        profile = self._profile_dir()
        existing = discover_existing_cdp(self.config)
        attempts: list[tuple[AuthConfig, Path, bool]] = []

        if existing:
            attempts.append((replace(self.config, debug_port=int(existing["port"])), profile, True))
            self._stage("browser_cdp", "PASS", mode="reuse", port=existing["port"], probe=existing)
        else:
            self._stage("browser_cdp", "PARTIAL", mode="new_required",
                        reason="未发现包含PMOS页面的可控CDP端口")

        # 没有可复用 CDP 时直接启动；已有 CDP 但认证失败时再追加一次独立浏览器。
        if not existing or self.config.browser_fallback:
            fallback_profile = profile
            if existing:
                fallback_profile = profile.parent / f"{profile.name}_fallback_{int(time.time())}"
            try:
                port = choose_free_debug_port(self.config)
                attempts.append((replace(self.config, debug_port=port), fallback_profile, False))
            except Exception as exc:
                if not existing:
                    raise
                logger.warning("auth.new_browser_candidate_unavailable: %s", exc)

        last_error: Exception | None = None
        for attempt_no, (attempt_config, attempt_profile, reused) in enumerate(attempts, 1):
            proc = None
            try:
                session = CdpSession(attempt_config)
                logger.info("auth.login_entry url=%s service=%s",
                            attempt_config.login_url, attempt_config.service_url)
                self._event("INFO", "AUTH_LOGIN_ENTRY", "使用带交易回跳的统一认证入口",
                            url=attempt_config.login_url, service=attempt_config.service_url)
                if reused:
                    logger.info("auth.reuse_existing_devtools port=%s", attempt_config.debug_port)
                else:
                    proc = launch_browser(attempt_config, executable, attempt_profile)
                    session.wait_ready(proc)
                    self._stage("browser_cdp", "PASS", mode="new", port=attempt_config.debug_port,
                                profile=str(attempt_profile))
                return self._run_attempt(attempt_config, executable, session, proc, started)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                mode = "reuse" if reused else "new"
                logger.exception("auth.attempt_failed attempt=%s mode=%s", attempt_no, mode)
                self._event("ERROR", "AUTH_ATTEMPT_FAILED", str(exc), attempt=attempt_no,
                            mode=mode, port=attempt_config.debug_port)
                if attempt_no < len(attempts):
                    logger.warning("auth.fallback_next_attempt next=%s", attempt_no + 1)
                    self._stage("browser_cdp", "RETRY", previous_mode=mode,
                                reason=f"{type(exc).__name__}: {exc}")
                    continue
                raise
        raise last_error or RuntimeError("没有可用的浏览器认证尝试")

    def _profile_dir(self) -> Path:
        if self.config.browser_profile_dir:
            profile = Path(self.config.browser_profile_dir).expanduser()
            if not profile.is_absolute():
                base = Path(self.config.extra.get("_config_dir") or (
                    Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
                ))
                profile = base / profile
            return profile.resolve()
        return (Path.home() / ".pmos-auto-crawler-profile").resolve()

    def _stage(self, name: str, status: str, **details) -> None:
        if self.reporter is not None:
            self.reporter.stage(name, status, **details)

    def _event(self, level: str, code: str, message: str, **details) -> None:
        if self.reporter is not None:
            self.reporter.event(level, code, message, **details)

    def _run_attempt(self, config: AuthConfig, executable, session: CdpSession,
                     proc, started: float) -> AuthenticationResult:
        """执行一次认证尝试；异常交给 run() 决定是否切换新浏览器。"""
        page = PmosPage(session)
        deadline = self.clock() + config.login_timeout_sec
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
            if proc is not None and proc.poll() is not None and not launcher_exited_logged:
                logger.info("auth.browser_launcher_exited code=%s; continuing_with_devtools=true", proc.returncode)
                launcher_exited_logged = True
            try:
                snapshot = page.snapshot()
                transient_since = None
            except Exception as exc:  # 浏览器导航/DevTools 短暂不可用时继续等待。
                now = self.clock()
                transient_since = transient_since if transient_since is not None else now
                waited = now - transient_since
                logger.warning("auth.page_waiting retry_in=%.2fs waited=%.1fs error=%s: %s",
                               config.poll_interval_sec, waited, type(exc).__name__, exc)
                if waited >= config.transient_error_timeout_sec:
                    raise TimeoutError(
                        f"浏览器页面连续 {waited:.0f}s 不可控制；最后错误={type(exc).__name__}: {exc}"
                    ) from exc
                self.sleeper(config.poll_interval_sec)
                continue

            if snapshot.state != last_state:
                logger.info("auth.state from=%s to=%s url=%s %s", last_state, snapshot.state.value,
                            snapshot.url[:160], snapshot.detail)
                last_state = snapshot.state
                self._event("INFO", "AUTH_STATE", snapshot.state.value,
                            url=snapshot.url[:180], detail=snapshot.detail)

            # 证书弹层 > 可见滑块 > 登录表单，避免隐藏滑块触发重复提交。
            if snapshot.state == PageState.GATEWAY_ERROR:
                if not gateway_recovered:
                    gateway_recovered = page.recover_from_gateway_error()
                    logger.warning("auth.gateway_502 recovered_by_history_back=%s", gateway_recovered)
                    self._event("WARN", "GATEWAY_502", "认证页面出现502，已尝试返回恢复",
                                recovered=gateway_recovered, url=snapshot.url[:180])
                else:
                    raise RuntimeError("认证后交易入口持续返回 502；已自动回退一次，请检查 PMOS 网关")
            elif snapshot.certificate_visible:
                if not cfca_submitted and self.clock() >= next_cfca_attempt:
                    available = probe_cfca_service(config.cfca_port)
                    logger.info("cfca.service available=%s port=%s", available, config.cfca_port)
                    cfca_result = page.select_cfca_and_verify()
                    cfca_submitted = bool(cfca_result.get("ok"))
                    logger.info("cfca.web_submit ok=%s reason=%s", cfca_submitted,
                                cfca_result.get("reason"))
                    next_cfca_attempt = self.clock() + config.cfca_retry_interval_sec
                if cfca_submitted:
                    self.pin_handler.handle(session, snapshot, config)
            elif snapshot.slider_visible:
                self.slider_handler.handle(session, snapshot, config)
            elif snapshot.login_form and self.clock() >= next_login_attempt:
                if login_submitted:
                    # 第一次 DOM 点击不一定真正触发前端提交；登录页仍在时允许
                    # 按配置间隔重试，避免“ok=True 后永久卡在登录页”。
                    logger.warning("auth.login_submit_retry reason=login_form_still_visible")
                    self._event("WARN", "AUTH_LOGIN_RETRY",
                                "登录表单仍可见，重新触发登录提交")
                result = page.submit_login(config.resolved_username, config.resolved_password)
                logger.info("auth.login_submit ok=%s reason=%s", result.get("ok"), result.get("reason"))
                login_submitted = bool(result.get("ok"))
                next_login_attempt = self.clock() + config.login_retry_interval_sec
            elif snapshot.state == PageState.LOGGED_IN or (
                cfca_submitted and not snapshot.certificate_visible and not snapshot.slider_visible
            ):
                if self.clock() >= next_login_check:
                    try:
                        cookie = session.cookies()
                        if self.check_login(cookie):
                            self._stage("auth_cookie", "PASS", browser=str(executable),
                                        cookie_present=True, cookie_length=len(cookie))
                            return AuthenticationResult(
                                cookie=cookie,
                                browser_path=str(executable),
                                elapsed_sec=self.clock() - started,
                                debug_port=config.debug_port,
                            )
                    except Exception as exc:
                        logger.warning("auth.login_check_waiting error=%s: %s", type(exc).__name__, exc)
                    next_login_check = self.clock() + config.login_check_interval_sec
            self.sleeper(config.poll_interval_sec)
        raise TimeoutError(f"PMOS 登录在 {config.login_timeout_sec}s 内未完成，最后状态={last_state}")

    def check_login(self, cookie: str) -> bool:
        """认证完成只校验会话 Cookie；真实接口有效性由 collect 的 QCTC 请求确认。"""
        if not cookie:
            return False
        cookie_names = {part.split("=", 1)[0].strip() for part in cookie.split(";") if "=" in part}
        has_session = bool(cookie_names & {"Admin-Token", "X-Ticket", "JSESSIONID", "XHXT_SESSIONID"})
        logger.info("auth.session_cookie present=%s", has_session)
        return has_session
