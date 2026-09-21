from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .browser import (
    CdpSession,
    browser_executable_candidates,
    choose_free_debug_port,
    discover_existing_cdp,
    launch_browser,
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
        profile = self._profile_dir()
        existing = discover_existing_cdp(self.config)

        # 先保留原来的“复用已有可控 PMOS 浏览器”成功路径。
        if existing:
            reused_config = replace(self.config, debug_port=int(existing["port"]))
            reused_session = CdpSession(reused_config)
            reused_executable = Path(self.config.browser_path or "existing-cdp")
            self._stage("browser_cdp", "PASS", mode="reuse", port=existing["port"], probe=existing)
            try:
                logger.info("auth.reuse_existing_devtools port=%s", reused_config.debug_port)
                self._event(
                    "INFO", "AUTH_LOGIN_ENTRY", "使用带交易回跳的统一认证入口",
                    url=reused_config.login_url, service=reused_config.service_url,
                )
                return self._run_attempt(
                    reused_config, reused_executable, reused_session, None, started
                )
            except Exception as exc:  # noqa: BLE001
                # 保留既有 browser_fallback 语义：已有旧 CDP 已不可用时，
                # 可以重新启动一次独立浏览器。但真正 Chrome→Edge 的切换
                # 只允许发生在下面的 bootstrap 阶段。
                logger.exception("auth.reused_browser_failed")
                self._event(
                    "ERROR", "AUTH_ATTEMPT_FAILED", str(exc),
                    attempt=1, mode="reuse", port=reused_config.debug_port,
                )
                if not self.config.browser_fallback:
                    raise
                self._stage(
                    "browser_cdp", "RETRY", previous_mode="reuse",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                profile = profile.parent / f"{profile.name}_fallback_{int(time.time())}"
        else:
            self._stage(
                "browser_cdp", "PARTIAL", mode="new_required",
                reason="未发现包含PMOS页面的可控CDP端口",
            )

        attempt_config, executable, session, proc, attempt_profile = self._launch_initial_browser(
            profile
        )
        logger.info(
            "auth.login_entry url=%s service=%s browser=%s",
            attempt_config.login_url, attempt_config.service_url, executable,
        )
        self._event(
            "INFO", "AUTH_LOGIN_ENTRY", "使用带交易回跳的统一认证入口",
            url=attempt_config.login_url, service=attempt_config.service_url,
            browser=str(executable),
        )
        self._stage(
            "browser_cdp", "PASS", mode="new", port=attempt_config.debug_port,
            profile=str(attempt_profile), browser=str(executable),
        )

        # 从这里开始已经确认浏览器成功打开 PMOS。后续登录、滑块、UKey、
        # QCTC 等任何失败都直接按原认证逻辑处理，绝不再切换 Edge/Chrome。
        return self._run_attempt(attempt_config, executable, session, proc, started)

    def _launch_initial_browser(
        self, profile: Path
    ) -> tuple[AuthConfig, Path, CdpSession, object, Path]:
        """仅在启动/打开PMOS阶段按候选浏览器回退；认证开始后不再切换。"""
        candidates = browser_executable_candidates(self.config.browser_path)
        if not self.config.browser_fallback:
            candidates = candidates[:1]
        logger.info(
            "browser.bootstrap_candidates total=%d candidates=%s",
            len(candidates), [str(p) for p in candidates],
        )

        last_error: Exception | None = None
        last_port = 0
        for index, executable in enumerate(candidates):
            port_seed = self.config
            if last_port:
                next_port = min(
                    max(last_port + 1, int(self.config.debug_port_scan_start)),
                    int(self.config.debug_port_scan_end),
                )
                port_seed = replace(
                    self.config,
                    debug_port=next_port,
                    debug_port_scan_start=next_port,
                )
            port = choose_free_debug_port(port_seed)
            last_port = port
            attempt_config = replace(self.config, debug_port=port)

            attempt_profile = profile
            if index:
                attempt_profile = profile.parent / f"{profile.name}_{executable.stem.lower()}"

            proc = None
            try:
                logger.info(
                    "browser.bootstrap_candidate index=%s/%s executable=%s port=%s",
                    index + 1, len(candidates), executable, port,
                )
                self._event(
                    "INFO", "BROWSER_BOOTSTRAP_CANDIDATE", "尝试启动认证浏览器",
                    index=index + 1, total=len(candidates),
                    browser=str(executable), port=port,
                )
                proc = launch_browser(attempt_config, executable, attempt_profile)
                session = CdpSession(attempt_config)
                probe = session.wait_bootstrap(proc)
                self._event(
                    "INFO", "BROWSER_SELECTED", "浏览器已成功打开PMOS，进入原认证流程",
                    browser=str(executable), port=port, probe=probe,
                )
                return attempt_config, executable, session, proc, attempt_profile
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "browser.bootstrap_failed executable=%s port=%s error=%s: %s",
                    executable, port, type(exc).__name__, exc,
                )
                self._event(
                    "WARN", "BROWSER_BOOTSTRAP_FAILED",
                    "浏览器未能在启动阶段打开可控PMOS页面",
                    browser=str(executable), port=port,
                    error=f"{type(exc).__name__}: {exc}",
                )
                try:
                    if proc is not None and proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
                if index + 1 < len(candidates):
                    self._event(
                        "WARN", "BROWSER_BOOTSTRAP_FALLBACK",
                        "仅因启动阶段失败，尝试下一个浏览器",
                        previous_browser=str(executable),
                        next_browser=str(candidates[index + 1]),
                    )

        raise last_error or RuntimeError("没有可用浏览器能打开PMOS登录页")

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
        """执行认证流程；进入本函数后浏览器已选定，后续异常绝不切换浏览器。"""
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
