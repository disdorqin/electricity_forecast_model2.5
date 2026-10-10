from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .browser import (
    BrowserControlError,
    CdpSession,
    browser_executable_candidates,
    choose_free_debug_port,
    discover_existing_cdp,
    is_bootstrap_render_ready,
    launch_browser,
)
from .config import AuthConfig
from .handlers import InteractionHandler, build_pin_handler, build_slider_handler, probe_cfca_service
from .page import PageState, PmosPage

# 版本演进索引：
# [V3-V5 legacy baseline] 原 PMOS 登录/滑块/UKey 认证状态机主链。
# [V8] 新浏览器 bootstrap 阶段候选回退；认证开始后不切浏览器。
# [V10-r1] stale session 同浏览器恢复 + browser-control-only fallback 边界。
# [V10-r3] existing CDP runtime 健康门禁：错误页/blank 不允许复用。
# [V10-r4] existing CDP reuse 同步使用 bootstrap render-readiness 判定。

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthenticationResult:
    cookie: str
    browser_path: str
    elapsed_sec: float
    debug_port: int = 0


class StaleSessionRecoveryError(RuntimeError):
    """The reused browser lost its session twice in one authentication run."""


class SessionExpiredError(RuntimeError):
    """[AUX-V1-r11f] 页面提示「网页/会话已失效」，且自动刷新未能恢复。

    仅当调用方通过 ``AuthConfig.extra["session_expired_refresh"]`` 显式启用时
    才会产生；AUX 三层防护据此从 L1（复用）降级到「重开浏览器」。96 主爬虫不
    设置该开关，永远不会遇到此异常，行为零变化。
    """


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

        reuse_rejected = False
        reused_config = None
        reused_session = None

        # [V10-r3] existing CDP runtime health gate: reject stale/error Chrome before reuse.
        # /json 的 target metadata 只能作为候选线索；必须通过 Runtime.evaluate 读取
        # 当前页面真实 location.href，避免把 chrome-error/about:blank 当作可复用会话。
        if existing:
            reused_config = replace(self.config, debug_port=int(existing["port"]))
            reused_session = CdpSession(reused_config)
            runtime_url = ""
            runtime_probe = {}
            reuse_error: Exception | None = None
            try:
                # [V10-r4] PMOS bootstrap render-readiness gate: URL-only is insufficient.
                runtime_probe = reused_session.bootstrap_render_probe(timeout=5)
                runtime_url = str(runtime_probe.get("href") or "")
            except Exception as exc:  # noqa: BLE001
                reuse_error = exc
            if reuse_error is not None or not is_bootstrap_render_ready(runtime_probe):
                reuse_rejected = True
                reason = (
                    f"{type(reuse_error).__name__}: {reuse_error}"
                    if reuse_error is not None
                    else f"runtime_url={runtime_url or '-'} render_probe={runtime_probe or '-'}"
                )
                logger.warning(
                    "auth.reuse_existing_devtools_unhealthy port=%s runtime_url=%s reason=%s",
                    reused_config.debug_port, runtime_url[:240] or "-", reason,
                )
                self._event(
                    "WARN", "BROWSER_REUSE_UNHEALTHY",
                    "已有 CDP target 元数据疑似 PMOS，但运行时页面未通过渲染健康门禁；转入独立 bootstrap",
                    port=reused_config.debug_port,
                    runtime_url=runtime_url[:240], reason=reason,
                )
                self._stage(
                    "browser_cdp", "RETRY", previous_mode="reuse",
                    reason=f"BROWSER_REUSE_UNHEALTHY: {reason}",
                )
                # 不关闭用户浏览器、不清 Cookie；新浏览器使用独立 profile，避免污染旧 CDP。
                profile = profile.parent / f"{profile.name}_fallback_{int(time.time())}"

        # 保留既有健康 CDP 的成功路径；认证失败后的 stale-session/browser-control
        # fallback 语义不因 V10-r3 健康门禁而改变。
        if existing and not reuse_rejected:
            reused_executable = Path(self.config.browser_path or "existing-cdp")
            self._stage(
                "browser_cdp", "PASS", mode="reuse", port=existing["port"],
                probe=existing,
            )
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
                logger.exception("auth.reused_browser_failed")
                self._event(
                    "ERROR", "AUTH_ATTEMPT_FAILED", str(exc),
                    attempt=1, mode="reuse", port=reused_config.debug_port,
                )
                if not self.config.browser_fallback or not self._is_browser_control_error(exc):
                    raise
                self._event(
                    "ERROR", "BROWSER_CONTROL_LOST",
                    "复用的 DevTools 浏览器已不可控，允许启动新的浏览器",
                    reason=f"{type(exc).__name__}: {exc}",
                    port=reused_config.debug_port,
                )
                self._stage(
                    "browser_cdp", "RETRY", previous_mode="reuse",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                profile = profile.parent / f"{profile.name}_fallback_{int(time.time())}"
        elif not existing:
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
        """[V8] bootstrap fallback；认证开始后不再切换浏览器。"""
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
                if "BROWSER_BOOTSTRAP_RENDER_STALLED" in str(exc):
                    self._event(
                        "WARN", "BROWSER_BOOTSTRAP_RENDER_STALLED",
                        "PMOS URL 已到达但页面在 bootstrap 窗口内持续空白",
                        browser=str(executable), port=port,
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

    @staticmethod
    def _is_browser_control_error(exc: BaseException) -> bool:
        """[V10-r1] Return true only for a lost/uncontrollable DevTools target.

        Authentication, slider, UKey, QCTC context and HTTP authorization
        failures are deliberately not browser fallback triggers.
        """
        if isinstance(exc, BrowserControlError):
            return True
        module = type(exc).__module__.lower()
        if module.startswith("websocket") or module.startswith("requests"):
            return True
        text = str(exc).lower()
        explicit = (
            "devtools", "cdp", "websocket", "web socket", "/json",
            "target closed", "target gone", "target not found",
            "浏览器页面连续", "不可控制", "浏览器无法打开", "连接已中断",
            "未找到 pmos 浏览器标签页", "尚未创建可控制的标签页",
        )
        return any(marker in text for marker in explicit)

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
        # [V10-r1] stale session recovery is same-browser only; it never selects another browser.
        stale_session_since: float | None = None
        stale_session_recovered = False
        stale_timeout = min(
            12.0,
            max(8.0, float(config.extra.get("stale_session_timeout_sec", 10.0))),
        )
        # [AUX-V1-r11f] L1-B2 会话失效自动刷新：门户在会话过期时会在页面上给出
        # 「网页已失效」之类提示，人工按 F5 即可恢复。这里把该人工动作自动化。
        # 开关默认关闭（96 不启用），启用后 AUX 可在刷新无效时降级重开浏览器。
        session_refresh_enabled = bool(
            config.extra.get("session_expired_refresh", False)
        )
        session_refresh_max = max(
            1, int(config.extra.get("session_expired_refresh_max", 2) or 2)
        )
        session_refresh_count = 0
        next_session_refresh = 0.0

        while self.clock() < deadline:
            if proc is not None and proc.poll() is not None and not launcher_exited_logged:
                logger.info("auth.browser_launcher_exited code=%s; continuing_with_devtools=true", proc.returncode)
                launcher_exited_logged = True
            try:
                snapshot = page.snapshot()
                transient_since = None
                # [AUX-V1-r11f] 判据 D1 命中：页面出现「已失效」文案。登录表单可见时
                # 说明本就需要重新登录，不属于「失效刷新」场景，故不触发刷新。
                if (
                    session_refresh_enabled
                    and snapshot.session_expired
                    and snapshot.state != PageState.LOGIN_READY
                ):
                    if session_refresh_count >= session_refresh_max:
                        self._event(
                            "ERROR", "AUTH_SESSION_EXPIRED_EXHAUSTED",
                            "页面提示会话已失效且刷新次数用尽，交回上层重开浏览器",
                            refresh_count=session_refresh_count,
                            url=snapshot.url[:180],
                        )
                        raise SessionExpiredError(
                            f"页面会话已失效；已刷新 {session_refresh_count} 次仍未恢复"
                        )
                    if self.clock() >= next_session_refresh:
                        session_refresh_count += 1
                        self._event(
                            "WARN", "AUTH_SESSION_EXPIRED_REFRESH",
                            "页面提示会话已失效，执行等价人工 F5 的刷新恢复",
                            refresh=session_refresh_count, max=session_refresh_max,
                            url=snapshot.url[:180], detail=snapshot.detail,
                        )
                        logger.warning(
                            "auth.session_expired_refresh count=%s/%s url=%s",
                            session_refresh_count, session_refresh_max, snapshot.url[:160],
                        )
                        try:
                            session.navigate(snapshot.url or config.login_url)
                        except Exception as nav_exc:  # noqa: BLE001
                            logger.warning(
                                "auth.session_expired_navigate_failed error=%s", nav_exc,
                            )
                        next_session_refresh = self.clock() + 5.0
                        last_state = None
                        cfca_submitted = False
                        login_submitted = False
                        gateway_recovered = False
                        self.sleeper(2.0)
                        continue
            except Exception as exc:  # 浏览器导航/DevTools 短暂不可用时继续等待。
                if self._is_browser_control_error(exc):
                    raise BrowserControlError(
                        f"DevTools/CDP 页面不可控: {type(exc).__name__}: {exc}"
                    ) from exc
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
            if snapshot.state != PageState.LOGGED_IN:
                stale_session_since = None

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
                            stale_session_since = None
                            self._stage("auth_cookie", "PASS", browser=str(executable),
                                        cookie_present=True, cookie_length=len(cookie))
                            return AuthenticationResult(
                                cookie=cookie,
                                browser_path=str(executable),
                                elapsed_sec=self.clock() - started,
                                debug_port=config.debug_port,
                            )
                        now = self.clock()
                        if stale_session_since is None:
                            stale_session_since = now
                        elif now - stale_session_since >= stale_timeout:
                            self._event(
                                "WARN", "AUTH_STALE_SESSION_DETECTED",
                                "浏览器仍显示已登录但会话 Cookie 已失效",
                                elapsed_sec=round(now - stale_session_since, 3),
                                recovery_attempted=stale_session_recovered,
                            )
                            if stale_session_recovered:
                                self._event(
                                    "ERROR", "AUTH_STALE_SESSION_RECOVERY_FAIL",
                                    "同一浏览器会话二次失效，停止认证",
                                )
                                raise StaleSessionRecoveryError(
                                    "登录态二次失效，已完成一次同浏览器恢复"
                                )
                            self._event(
                                "WARN", "AUTH_STALE_SESSION_RECOVERY_START",
                                "在同一 CDP 浏览器中重新打开统一认证入口",
                                url=config.login_url,
                            )
                            session.navigate(config.login_url)
                            stale_session_recovered = True
                            stale_session_since = None
                            last_state = None
                            next_login_attempt = 0.0
                            next_cfca_attempt = 0.0
                            next_login_check = 0.0
                            cfca_submitted = False
                            login_submitted = False
                            gateway_recovered = False
                            self._event(
                                "INFO", "AUTH_STALE_SESSION_RECOVERY_OK",
                                "已在同一 CDP 浏览器中重置认证状态",
                            )
                            continue
                    except Exception as exc:
                        if isinstance(exc, StaleSessionRecoveryError):
                            raise
                        if self._is_browser_control_error(exc):
                            raise BrowserControlError(
                                f"DevTools/CDP 会话不可控: {type(exc).__name__}: {exc}"
                            ) from exc
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
