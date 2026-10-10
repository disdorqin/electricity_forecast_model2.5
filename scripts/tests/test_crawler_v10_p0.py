from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from websocket import WebSocketTimeoutException

from scripts.crawler.auth.auto_crawler.config import AuthConfig
from scripts.crawler.auth.auto_crawler.page import PageSnapshot, PageState
from scripts.crawler.auth.auto_crawler.state_machine import (
    AuthenticationStateMachine,
    StaleSessionRecoveryError,
)
from scripts.crawler.collect import crawl as core
from scripts.crawler.collect import crawl_96_local as local
from scripts.crawler.runtime_lock import RuntimeLock, RuntimeLockError


class RuntimeLockP0Test(unittest.TestCase):
    def test_first_holds_second_fails_then_release_reacquires(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".crawler.lock"
            first = RuntimeLock(path).acquire()
            try:
                with self.assertRaises(RuntimeLockError):
                    RuntimeLock(path).acquire()
            finally:
                first.release()
            RuntimeLock(path).acquire().release()


class QctcSoftGateP0Test(unittest.TestCase):
    def test_context_missing_is_soft_and_real_business_request_decides(self) -> None:
        reporter = MagicMock()
        spider = core.PmosCrawler(
            browser_debug_port=9222,
            data_api_mode="qctc",
            reporter=reporter,
        )
        with patch.object(spider, "ensure_qctc_context", return_value=False), \
             patch.object(spider, "_abort_qctc_auth") as abort:
            self.assertTrue(spider.fetch_csrf_token())
        abort.assert_not_called()
        codes = [call.args[1] for call in reporter.event.call_args_list]
        self.assertIn("QCTC_CONTEXT_SOFT_MISSING", codes)

    def test_qctc_403_is_a_hard_auth_rejection(self) -> None:
        spider = core.PmosCrawler(browser_debug_port=9222, data_api_mode="qctc")
        response = type("Response", (), {
            "status_code": 403,
            "text": "forbidden",
            "reason": "Forbidden",
        })()
        spider._browser_req = lambda *args, **kwargs: response
        with self.assertRaises(core.QCTCAuthRejected):
            spider._qctc_get(
                "informationDisclosure/ForecastData/getLoadData",
                params={"pdate": "2026-01-01", "versions": ""},
                page_url=spider.qctc_forecast_page,
                web_path="/qctc-trade/informationDisclosure/forecast10424",
            )


class BrowserRecoveryP0Test(unittest.TestCase):
    def test_context_missing_is_not_a_browser_recovery_trigger(self) -> None:
        self.assertFalse(
            AuthenticationStateMachine._is_browser_control_error(
                RuntimeError("QCTC认证上下文未建立")
            )
        )

    def test_real_cdp_disconnect_is_a_browser_recovery_trigger(self) -> None:
        self.assertTrue(
            AuthenticationStateMachine._is_browser_control_error(
                RuntimeError("QCTC CDP连接已中断")
            )
        )

    def test_date_retry_does_not_treat_generic_http_failure_as_browser_loss(self) -> None:
        self.assertFalse(local._is_recoverable_browser_control_error(
            requests.ConnectionError("remote QCTC API temporarily unavailable")
        ))
        self.assertTrue(local._is_recoverable_browser_control_error(
            requests.ConnectionError("HTTPConnectionPool(host='127.0.0.1') /json")
        ))

    def test_reused_auth_failure_does_not_start_a_second_browser(self) -> None:
        machine = AuthenticationStateMachine(AuthConfig(browser_reuse=True))
        reused_session = MagicMock()
        reused_session.bootstrap_render_probe.return_value = {
            "href": "https://pmos.sd.sgcc.com.cn/#/dashboard",
            "bodyTextLength": 12,
        }
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.discover_existing_cdp",
            return_value={"port": 9222},
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.CdpSession",
            return_value=reused_session,
        ), patch.object(
            machine,
            "_run_attempt",
            side_effect=RuntimeError("登录态二次失效，已完成一次同浏览器恢复"),
        ), patch.object(
            machine,
            "_launch_initial_browser",
            side_effect=AssertionError("authentication failure must not launch another browser"),
        ):
            with self.assertRaisesRegex(RuntimeError, "二次失效"):
                machine.run()

    def _assert_unhealthy_reuse_enters_bootstrap(self, runtime_url=None, evaluate_error=None) -> None:
        machine = AuthenticationStateMachine(AuthConfig(browser_reuse=True))
        reused_session = MagicMock()
        if evaluate_error is not None:
            reused_session.bootstrap_render_probe.side_effect = evaluate_error
        else:
            reused_session.bootstrap_render_probe.return_value = {
                "href": runtime_url,
                "bodyTextLength": 0,
            }
        new_config = AuthConfig(debug_port=9223)
        new_session = object()
        expected = object()
        reporter = MagicMock()
        machine.reporter = reporter
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.discover_existing_cdp",
            return_value={"port": 9222, "url": "https://pmos.sd.sgcc.com.cn/#/dashboard"},
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.CdpSession",
            return_value=reused_session,
        ), patch.object(
            machine,
            "_launch_initial_browser",
            return_value=(new_config, Path("edge.exe"), new_session, None, Path("new-profile")),
        ) as launch, patch.object(
            machine, "_run_attempt", return_value=expected,
        ) as attempt:
            result = machine.run()

        self.assertIs(result, expected)
        launch.assert_called_once()
        attempt.assert_called_once()
        self.assertIs(attempt.call_args.args[2], new_session)
        reused_session.bootstrap_render_probe.assert_called_once_with(timeout=5)
        event_codes = [call.args[1] for call in reporter.event.call_args_list]
        self.assertIn("BROWSER_REUSE_UNHEALTHY", event_codes)

    def test_existing_pmos_metadata_but_chrome_error_runtime_is_not_reused(self) -> None:
        self._assert_unhealthy_reuse_enters_bootstrap("chrome-error://chromewebdata/")

    def test_existing_pmos_metadata_but_about_blank_runtime_is_not_reused(self) -> None:
        self._assert_unhealthy_reuse_enters_bootstrap("about:blank")

    def test_existing_cdp_runtime_evaluate_timeout_is_not_reused(self) -> None:
        self._assert_unhealthy_reuse_enters_bootstrap(
            evaluate_error=WebSocketTimeoutException("Connection timed out")
        )

    def test_healthy_pmos_runtime_keeps_original_reuse_path(self) -> None:
        machine = AuthenticationStateMachine(AuthConfig(browser_reuse=True))
        reused_session = MagicMock()
        reused_session.bootstrap_render_probe.return_value = {
            "href": "https://pmos.sd.sgcc.com.cn/#/dashboard",
            "bodyTextLength": 12,
        }
        expected = object()
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.discover_existing_cdp",
            return_value={"port": 9222, "url": "https://pmos.sd.sgcc.com.cn/#/dashboard"},
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.CdpSession",
            return_value=reused_session,
        ), patch.object(
            machine, "_run_attempt", return_value=expected,
        ) as attempt, patch.object(
            machine, "_launch_initial_browser",
        ) as launch:
            result = machine.run()

        self.assertIs(result, expected)
        launch.assert_not_called()
        attempt.assert_called_once()
        self.assertIs(attempt.call_args.args[2], reused_session)


class StaleSessionRecoveryP0Test(unittest.TestCase):
    @staticmethod
    def _snapshot() -> PageSnapshot:
        return PageSnapshot(
            PageState.LOGGED_IN,
            "https://pmos.sd.sgcc.com.cn/#/dashboard",
            "portal_dashboard",
        )

    def test_reused_dashboard_cookie_loss_navigates_same_cdp_once(self) -> None:
        class Clock:
            value = 0.0

            def __call__(self):
                self.value += 3.0
                return self.value

        class Session:
            def __init__(self):
                self.navigation = []
                self.cookie_calls = 0

            def cookies(self):
                self.cookie_calls += 1
                return "tracking=expired" if self.cookie_calls <= 3 else "JSESSIONID=ok"

            def navigate(self, url):
                self.navigation.append(url)

        session = Session()
        machine = AuthenticationStateMachine(
            AuthConfig(
                login_check_interval_sec=0,
                poll_interval_sec=0,
                extra={"stale_session_timeout_sec": 8},
            ),
            clock=Clock(),
            sleeper=lambda _seconds: None,
        )
        events = []
        machine.reporter = MagicMock()
        machine.reporter.event.side_effect = lambda level, code, message, **details: events.append(code)
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.PmosPage.snapshot",
            return_value=self._snapshot(),
        ):
            result = machine._run_attempt(
                machine.config,
                Path("existing-cdp"),
                session,
                None,
                started=0.0,
            )
        self.assertEqual(result.cookie, "JSESSIONID=ok")
        self.assertEqual(session.navigation, [machine.config.login_url])
        self.assertEqual(events.count("AUTH_STALE_SESSION_RECOVERY_START"), 1)
        self.assertIn("AUTH_STALE_SESSION_RECOVERY_OK", events)

    def test_stale_session_recovery_is_attempted_at_most_once(self) -> None:
        class Clock:
            value = 0.0

            def __call__(self):
                self.value += 3.0
                return self.value

        class Session:
            navigation = []

            @staticmethod
            def cookies():
                return "tracking=expired"

            @staticmethod
            def navigate(url):
                Session.navigation.append(url)

        machine = AuthenticationStateMachine(
            AuthConfig(
                login_check_interval_sec=0,
                poll_interval_sec=0,
                extra={"stale_session_timeout_sec": 8},
            ),
            clock=Clock(),
            sleeper=lambda _seconds: None,
        )
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.PmosPage.snapshot",
            return_value=self._snapshot(),
        ), self.assertRaises(StaleSessionRecoveryError):
            machine._run_attempt(machine.config, Path("existing-cdp"), Session(), None, 0.0)
        self.assertEqual(len(Session.navigation), 1)


if __name__ == "__main__":
    unittest.main()
