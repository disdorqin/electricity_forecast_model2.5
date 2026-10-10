from __future__ import annotations

import json
import os
import tempfile
import unittest
import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from scripts.crawler.auth.auto_crawler import browser as browser_module
from scripts.crawler.auth.auto_crawler.browser import CdpSession
from scripts.crawler.auth.auto_crawler.browser import BrowserResolutionError
from scripts.crawler.auth.auto_crawler.config import AuthConfig
from scripts.crawler.auth.auto_crawler.handlers import BrowserSliderHandler, CaptureSliderHandler, ManualPinHandler, ManualSliderHandler, TemplateSliderSolver, _slider_geometry_from_images, build_pin_handler, build_slider_handler
from scripts.crawler.auth.auto_crawler.main import BUILD_MARKER, default_config_path, ssl_check
from scripts.crawler.auth.auto_crawler.page import PageState, PmosPage
from scripts.crawler.auth.auto_crawler.state_machine import AuthenticationResult, AuthenticationStateMachine


class AuthConfigTest(unittest.TestCase):
    def test_secrets_are_read_from_environment_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"username_env": "TEST_PMOS_USER", "password_env": "TEST_PMOS_PASS"}))
            with patch.dict(os.environ, {"TEST_PMOS_USER": "alice", "TEST_PMOS_PASS": "secret"}, clear=False):
                config = AuthConfig.from_file(path)
                self.assertEqual(config.resolved_username, "alice")
                self.assertEqual(config.resolved_password, "secret")
                self.assertNotIn("secret", path.read_text())

    def test_private_config_is_a_fallback_when_environment_is_absent(self) -> None:
        config = AuthConfig(username="alice", password="secret", ukey_pin="123456")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.resolved_username, "alice")
            self.assertEqual(config.resolved_password, "secret")
            self.assertEqual(config.resolved_pin, "123456")

    def test_slow_machine_retry_settings_can_be_overridden(self) -> None:
        config = AuthConfig(cfca_retry_interval_sec=5.0, login_check_interval_sec=8.0,
                            transient_error_timeout_sec=120)
        self.assertEqual(config.cfca_retry_interval_sec, 5.0)
        self.assertEqual(config.login_check_interval_sec, 8.0)
        self.assertEqual(config.transient_error_timeout_sec, 120)

    def test_pin_submit_mode_is_configurable(self) -> None:
        self.assertEqual(AuthConfig(pin_submit_mode="enter").pin_submit_mode, "enter")

    def test_login_url_preserves_transaction_service_context(self) -> None:
        config = AuthConfig()
        self.assertEqual(
            config.login_url,
            "https://pmos.sd.sgcc.com.cn/?service=https%3A%2F%2Fpmos.sd.sgcc.com.cn%3A18080%2Ftrade%2FDaJyjgfbPlantQuery.do%3Fappkey%3D187",
        )

    def test_login_url_can_be_overridden_for_site_changes(self) -> None:
        config = AuthConfig(extra={"browser_login_url": "https://example.test/login"})
        self.assertEqual(config.login_url, "https://example.test/login")

    def test_default_config_is_the_local_config_json(self) -> None:
        self.assertEqual(default_config_path().name, "config.json")
        self.assertTrue(default_config_path().is_file())

    def test_frozen_mode_uses_executable_sibling_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / "pmos_auto_auth.exe"
            exe.touch()
            config = exe.with_name("config.json")
            config.write_text("{}", encoding="utf-8")
            with patch.object(__import__("sys"), "frozen", True, create=True), patch.object(__import__("sys"), "executable", str(exe)):
                self.assertEqual(default_config_path(), config.resolve())

    def test_ssl_version_check_does_not_open_network_connection(self) -> None:
        with patch("scripts.crawler.auth.auto_crawler.main.ssl.OPENSSL_VERSION", "OpenSSL 3.0.13 test"), \
             patch("scripts.crawler.auth.auto_crawler.main.socket.create_connection") as connect:
            self.assertEqual(ssl_check("https://pmos.sd.sgcc.com.cn", probe_network=False), 0)
            connect.assert_not_called()

    def test_build_marker_identifies_current_auth_state_machine(self) -> None:
        self.assertIn("template-slider", BUILD_MARKER)
        self.assertIn("ukey-pin", BUILD_MARKER)


class HandlerTest(unittest.TestCase):
    def test_manual_handlers_are_default(self) -> None:
        config = AuthConfig()
        self.assertIsInstance(build_slider_handler(config), ManualSliderHandler)
        self.assertIsInstance(build_pin_handler(config), ManualPinHandler)

    def test_capture_slider_handler_is_opt_in(self) -> None:
        self.assertIsInstance(build_slider_handler(AuthConfig(slider_handler="capture")), CaptureSliderHandler)

    def test_template_slider_handler_is_opt_in(self) -> None:
        self.assertIsInstance(build_slider_handler(AuthConfig(slider_handler="template")), BrowserSliderHandler)

    def test_template_solver_uses_current_image_pair(self) -> None:
        background = np.random.default_rng(7).integers(0, 256, size=(16, 50, 3), dtype=np.uint8)
        piece = background[:, 30:42, :]

        def png(image: Image.Image) -> str:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        result = TemplateSliderSolver().solve(
            screenshot_png=b"",
            geometry={"images": [
                {"src": png(Image.fromarray(background, "RGB")), "width": 100},
                {"src": png(Image.fromarray(piece, "RGB").convert("RGBA")), "width": 24},
            ]},
            config=AuthConfig(),
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.offset_x, (30 - 1.7) * 2, places=1)

    def test_image_relative_geometry_scales_with_current_captcha_layout(self) -> None:
        geometry = _slider_geometry_from_images({"images": [{"x": 412, "y": 188, "width": 330, "height": 155}]})
        self.assertIsNotNone(geometry)
        self.assertEqual(geometry["track_x"], 412)
        self.assertEqual(geometry["track_width"], 330)
        self.assertEqual(geometry["source"], "captcha_image_relative")

    def test_plugin_mode_requires_plugin_spec(self) -> None:
        with self.assertRaises(ValueError):
            build_slider_handler(AuthConfig(slider_handler="plugin"))

    def test_windows_pin_without_pin_falls_back_to_manual(self) -> None:
        self.assertIsInstance(
            build_pin_handler(AuthConfig(pin_handler="windows")), ManualPinHandler
        )


class BrowserDiscoveryTest(unittest.TestCase):
    @staticmethod
    def _fake_installed_paths(path: Path) -> bool:
        normalized = str(path).replace("/", "\\").lower()
        return normalized in {
            r"c:\program files\google\chrome\application\chrome.exe",
            r"c:\program files (x86)\microsoft\edge\application\msedge.exe",
        }

    def _candidate_patches(self):
        env = {
            # 故意模拟公司镜像把环境变量指向不存在的盘符；标准 C 盘安装
            # 仍应被 discovery 找到。
            "PROGRAMFILES": r"Z:\Program Files",
            "PROGRAMFILES(X86)": r"Y:\Program Files (x86)",
            "LOCALAPPDATA": r"Z:\LocalAppData",
            "SystemDrive": "Y:",
        }
        return (
            patch.object(browser_module.sys, "platform", "win32"),
            patch.dict(os.environ, env, clear=False),
            patch.object(browser_module.Path, "is_file", self._fake_installed_paths),
            patch.object(browser_module.shutil, "which", return_value=None),
            patch.object(browser_module, "_windows_app_paths", return_value=[]),
            patch.object(browser_module, "_windows_running_browser_path", return_value=None),
            patch.object(
                browser_module,
                "_windows_default_browser_command",
                side_effect=BrowserResolutionError("test default browser unavailable"),
            ),
        )

    def test_windows_discovery_checks_standard_c_edge_when_programfiles_is_wrong(self) -> None:
        patches = self._candidate_patches()
        for item in patches:
            item.start()
        try:
            candidates = browser_module.browser_executable_candidates("")
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertIn(
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe").resolve(),
            candidates,
        )

    def test_windows_default_candidate_order_is_chrome_then_edge(self) -> None:
        patches = self._candidate_patches()
        for item in patches:
            item.start()
        try:
            candidates = browser_module.browser_executable_candidates("")
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual([path.name.lower() for path in candidates], ["chrome.exe", "msedge.exe"])

    def test_explicit_browser_path_has_priority_but_keeps_other_fallback(self) -> None:
        patches = self._candidate_patches()
        for item in patches:
            item.start()
        try:
            edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
            candidates = browser_module.browser_executable_candidates(str(edge))
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(candidates[0], edge.resolve())
        self.assertIn(Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe").resolve(), candidates)

    def test_app_paths_fallback_finds_edge_when_fixed_paths_are_missing(self) -> None:
        edge = Path(r"D:\Apps\Microsoft\Edge\msedge.exe")
        normalized_edge = str(edge).replace("/", "\\").lower()

        def is_file(path: Path) -> bool:
            return str(path).replace("/", "\\").lower() == normalized_edge

        patches = (
            patch.object(browser_module.sys, "platform", "win32"),
            patch.dict(os.environ, {
                "PROGRAMFILES": r"Z:\Program Files",
                "PROGRAMFILES(X86)": r"Y:\Program Files (x86)",
                "LOCALAPPDATA": r"Z:\LocalAppData",
                "SystemDrive": "Y:",
            }, clear=False),
            patch.object(browser_module.Path, "is_file", is_file),
            patch.object(browser_module.shutil, "which", return_value=None),
            patch.object(
                browser_module, "_windows_app_paths",
                side_effect=lambda name: [edge] if name == "msedge.exe" else [],
            ),
            patch.object(browser_module, "_windows_running_browser_path", return_value=None),
            patch.object(
                browser_module, "_windows_default_browser_command",
                side_effect=BrowserResolutionError("test default browser unavailable"),
            ),
        )
        for item in patches:
            item.start()
        try:
            candidates = browser_module.browser_executable_candidates("")
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertIn(edge.resolve(), candidates)

    def test_running_process_path_fallback_finds_edge_after_app_paths(self) -> None:
        edge = Path(r"D:\Apps\Microsoft\Edge\msedge.exe")
        normalized_edge = str(edge).replace("/", "\\").lower()

        def is_file(path: Path) -> bool:
            return str(path).replace("/", "\\").lower() == normalized_edge

        patches = (
            patch.object(browser_module.sys, "platform", "win32"),
            patch.dict(os.environ, {
                "PROGRAMFILES": r"Z:\Program Files",
                "PROGRAMFILES(X86)": r"Y:\Program Files (x86)",
                "LOCALAPPDATA": r"Z:\LocalAppData",
                "SystemDrive": "Y:",
            }, clear=False),
            patch.object(browser_module.Path, "is_file", is_file),
            patch.object(browser_module.shutil, "which", return_value=None),
            patch.object(browser_module, "_windows_app_paths", return_value=[]),
            patch.object(
                browser_module, "_windows_running_browser_path",
                side_effect=lambda name: edge if name == "msedge.exe" else None,
            ),
            patch.object(
                browser_module, "_windows_default_browser_command",
                side_effect=BrowserResolutionError("test default browser unavailable"),
            ),
        )
        for item in patches:
            item.start()
        try:
            candidates = browser_module.browser_executable_candidates("")
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertIn(edge.resolve(), candidates)


class BrowserStartupTest(unittest.TestCase):
    def test_pmos_url_with_empty_body_stalls_bootstrap(self) -> None:
        session = CdpSession(AuthConfig(browser_bootstrap_timeout_sec=3, debug_port=9222))
        page = {"url": "https://pmos.sd.sgcc.com.cn/#/dashboard", "webSocketDebuggerUrl": "ws://test"}
        blank = {"href": page["url"], "bodyTextLength": 0, "hasLogin": False}
        with patch.object(browser_module, "probe_cdp_port", return_value={"browser": "Chrome"}), \
             patch.object(session, "pages", return_value=[page]), \
             patch.object(session, "evaluate", return_value=blank), \
             patch.object(browser_module.time, "monotonic", side_effect=[0.0, 1.0, 4.0]), \
             patch.object(browser_module.time, "sleep"):
            with self.assertRaisesRegex(TimeoutError, "BROWSER_BOOTSTRAP_RENDER_STALLED"):
                session.wait_bootstrap()

    def test_pmos_blank_then_login_form_bootstrap_succeeds(self) -> None:
        page = {"url": "https://pmos.sd.sgcc.com.cn/?service=login", "webSocketDebuggerUrl": "ws://test"}
        probes = [
            {"href": page["url"], "bodyTextLength": 0},
            {"href": page["url"], "bodyTextLength": 18, "hasPassword": True, "hasLogin": True},
        ]
        session = CdpSession(AuthConfig(browser_bootstrap_timeout_sec=3, debug_port=9222))
        with patch.object(browser_module, "probe_cdp_port", return_value={"browser": "Chrome"}), \
             patch.object(session, "pages", return_value=[page]), \
             patch.object(session, "evaluate", side_effect=probes), \
             patch.object(browser_module.time, "monotonic", side_effect=[0.0, 1.0, 2.0]), \
             patch.object(browser_module.time, "sleep"):
            result = session.wait_bootstrap()
        self.assertEqual(result["runtime_url"], page["url"])

    def test_pmos_dashboard_with_business_content_bootstrap_succeeds(self) -> None:
        page = {"url": "https://pmos.sd.sgcc.com.cn/#/dashboard", "webSocketDebuggerUrl": "ws://test"}
        session = CdpSession(AuthConfig(browser_bootstrap_timeout_sec=3, debug_port=9222))
        with patch.object(browser_module, "probe_cdp_port", return_value={"browser": "Chrome"}), \
             patch.object(session, "pages", return_value=[page]), \
             patch.object(session, "evaluate", return_value={
                 "href": page["url"], "bodyTextLength": 22, "bodyText": "交易首页",
             }), \
             patch.object(browser_module.time, "monotonic", side_effect=[0.0, 1.0]), \
             patch.object(browser_module.time, "sleep"):
            result = session.wait_bootstrap()
        self.assertEqual(result["runtime_url"], page["url"])

    def test_pmos_gateway_error_text_remains_bootstrap_ready_for_existing_handling(self) -> None:
        page = {"url": "https://pmos.sd.sgcc.com.cn/#/dashboard", "webSocketDebuggerUrl": "ws://test"}
        session = CdpSession(AuthConfig(browser_bootstrap_timeout_sec=3, debug_port=9222))
        with patch.object(browser_module, "probe_cdp_port", return_value={"browser": "Chrome"}), \
             patch.object(session, "pages", return_value=[page]), \
             patch.object(session, "evaluate", return_value={
                 "href": page["url"], "bodyTextLength": 3, "bodyText": "502",
             }), \
             patch.object(browser_module.time, "monotonic", side_effect=[0.0, 1.0]):
            result = session.wait_bootstrap()
        self.assertEqual(result["runtime_url"], page["url"])

    def test_exited_launcher_does_not_prevent_devtools_readiness(self) -> None:
        class ExitedLauncher:
            returncode = 0

            @staticmethod
            def poll():
                return 0

        class ReadyResponse:
            ok = True

        with patch("scripts.crawler.auth.auto_crawler.browser.requests.get", return_value=ReadyResponse()):
            CdpSession(AuthConfig()).wait_ready(ExitedLauncher())

    def test_chrome_bootstrap_failure_falls_back_to_edge_before_auth(self) -> None:
        chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        launched = []

        class FakeProc:
            returncode = None

            @staticmethod
            def poll():
                return None

            @staticmethod
            def terminate():
                return None

        class FakeSession:
            def __init__(self, config):
                self.config = config

            def wait_bootstrap(self, _proc):
                if self.config.debug_port == 9222:
                    raise TimeoutError("chrome-error://chromewebdata/")
                return {
                    "port": self.config.debug_port,
                    "runtime_url": "https://pmos.sd.sgcc.com.cn/#/login",
                    "browser": "Microsoft Edge",
                }

        machine = AuthenticationStateMachine(AuthConfig(browser_reuse=False))
        expected = AuthenticationResult(
            cookie="JSESSIONID=edge",
            browser_path=str(edge),
            elapsed_sec=1.0,
            debug_port=9223,
        )
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.discover_existing_cdp",
            return_value=None,
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.browser_executable_candidates",
            return_value=[chrome, edge],
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.choose_free_debug_port",
            side_effect=[9222, 9223],
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.launch_browser",
            side_effect=lambda config, executable, profile: (
                launched.append((str(executable), config.debug_port, str(profile))) or FakeProc()
            ),
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.CdpSession",
            FakeSession,
        ), patch.object(
            machine, "_run_attempt", return_value=expected,
        ) as auth_attempt:
            result = machine.run()

        self.assertEqual(result, expected)
        self.assertEqual([item[1] for item in launched], [9222, 9223])
        self.assertIn("chrome.exe", launched[0][0].lower())
        self.assertIn("msedge.exe", launched[1][0].lower())
        self.assertNotEqual(launched[0][2], launched[1][2])
        auth_attempt.assert_called_once()
        self.assertEqual(auth_attempt.call_args.args[0].debug_port, 9223)

    def test_unhealthy_existing_chrome_then_bootstrap_failure_reaches_edge(self) -> None:
        chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        launched = []
        sessions = []

        class FakeProc:
            returncode = None

            @staticmethod
            def poll():
                return None

            @staticmethod
            def terminate():
                return None

        class FakeSession:
            def __init__(self, config):
                self.config = config
                sessions.append(self)

            def bootstrap_render_probe(self, *, timeout=5):
                self.assert_timeout = timeout
                if self.config.debug_port == 9221:
                    return {
                        "href": "chrome-error://chromewebdata/",
                        "bodyTextLength": 0,
                    }
                raise AssertionError("bootstrap sessions must be checked by wait_bootstrap")

            def wait_bootstrap(self, _proc):
                if self.config.debug_port == 9222:
                    raise TimeoutError("BROWSER_BOOTSTRAP_RENDER_STALLED: chrome page stayed blank")
                return {
                    "port": self.config.debug_port,
                    "runtime_url": "https://pmos.sd.sgcc.com.cn/#/login",
                    "browser": "Microsoft Edge",
                }

        machine = AuthenticationStateMachine(AuthConfig(browser_reuse=True))
        expected = AuthenticationResult(
            cookie="JSESSIONID=edge",
            browser_path=str(edge),
            elapsed_sec=1.0,
            debug_port=9223,
        )
        reporter = MagicMock()
        machine.reporter = reporter
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.discover_existing_cdp",
            return_value={"port": 9221, "url": "https://pmos.sd.sgcc.com.cn/#/dashboard"},
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.browser_executable_candidates",
            return_value=[chrome, edge],
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.choose_free_debug_port",
            side_effect=[9222, 9223],
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.launch_browser",
            side_effect=lambda config, executable, profile: (
                launched.append((str(executable), config.debug_port, str(profile))) or FakeProc()
            ),
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.CdpSession",
            FakeSession,
        ), patch.object(
            machine, "_run_attempt", return_value=expected,
        ) as auth_attempt:
            result = machine.run()

        self.assertEqual(result, expected)
        self.assertEqual([item[1] for item in launched], [9222, 9223])
        self.assertIn("chrome.exe", launched[0][0].lower())
        self.assertIn("msedge.exe", launched[1][0].lower())
        self.assertNotEqual(launched[0][2], launched[1][2])
        self.assertEqual([session.config.debug_port for session in sessions], [9221, 9222, 9223])
        auth_attempt.assert_called_once()
        self.assertEqual(auth_attempt.call_args.args[0].debug_port, 9223)
        event_codes = [call.args[1] for call in reporter.event.call_args_list]
        self.assertIn("BROWSER_REUSE_UNHEALTHY", event_codes)
        self.assertIn("BROWSER_BOOTSTRAP_FALLBACK", event_codes)
        self.assertIn("BROWSER_BOOTSTRAP_RENDER_STALLED", event_codes)

    def test_auth_failure_after_chrome_bootstrap_does_not_switch_to_edge(self) -> None:
        chrome = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
        edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
        launched = []

        class FakeProc:
            returncode = None

            @staticmethod
            def poll():
                return None

        class FakeSession:
            def __init__(self, config):
                self.config = config

            def wait_bootstrap(self, _proc):
                return {
                    "port": self.config.debug_port,
                    "runtime_url": "https://pmos.sd.sgcc.com.cn/#/login",
                    "browser": "Google Chrome",
                }

        machine = AuthenticationStateMachine(AuthConfig(browser_reuse=False))
        with patch(
            "scripts.crawler.auth.auto_crawler.state_machine.discover_existing_cdp",
            return_value=None,
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.browser_executable_candidates",
            return_value=[chrome, edge],
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.choose_free_debug_port",
            return_value=9222,
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.launch_browser",
            side_effect=lambda config, executable, profile: (
                launched.append(str(executable)) or FakeProc()
            ),
        ), patch(
            "scripts.crawler.auth.auto_crawler.state_machine.CdpSession",
            FakeSession,
        ), patch.object(
            machine, "_run_attempt", side_effect=RuntimeError("UKey认证失败"),
        ):
            with self.assertRaisesRegex(RuntimeError, "UKey认证失败"):
                machine.run()

        self.assertEqual(len(launched), 1)
        self.assertIn("chrome.exe", launched[0].lower())


class PageStateTest(unittest.TestCase):
    def test_page_state_values_are_stable_for_plugins(self) -> None:
        self.assertEqual(PageState.SLIDER.value, "slider")
        self.assertEqual(PageState.CERTIFICATE.value, "certificate")
        self.assertEqual(PageState.LOGGED_IN.value, "logged_in")

    def test_certificate_is_detected_before_login_form(self) -> None:
        class FakeSession:
            @staticmethod
            def evaluate(*_args, **_kwargs):
                return {"url": "https://pmos.sd.sgcc.com.cn/#/outNet", "ready": "complete",
                        "hasPassword": True, "slider": False, "cfca": True, "text": ""}

        snapshot = PmosPage(FakeSession()).snapshot()
        self.assertEqual(snapshot.state, PageState.CERTIFICATE)

    def test_non_pmos_startup_tab_is_loading_not_terminal_error(self) -> None:
        class FakeSession:
            @staticmethod
            def evaluate(*_args, **_kwargs):
                return {"url": "edge://newtab/", "ready": "complete", "hasPassword": False,
                        "slider": False, "cfca": False, "text": ""}

        snapshot = PmosPage(FakeSession()).snapshot()
        self.assertEqual(snapshot.state, PageState.LOADING)

    def test_login_form_wins_when_slider_is_not_visible(self) -> None:
        class FakeSession:
            @staticmethod
            def evaluate(*_args, **_kwargs):
                return {"url": "https://pmos.sd.sgcc.com.cn/#/outNet", "ready": "complete",
                        "hasPassword": True, "slider": False, "cfca": False, "text": ""}

        self.assertEqual(PmosPage(FakeSession()).snapshot().state, PageState.LOGIN_READY)

    def test_existing_legacy_zcq_home_is_authenticated(self) -> None:
        class FakeSession:
            @staticmethod
            def evaluate(*_args, **_kwargs):
                return {"url": "https://pmos.sd.sgcc.com.cn:18080/zcq/main/index.do",
                        "ready": "complete", "hasPassword": False, "slider": False,
                        "cfca": False, "text": "您好，某公司 返回首页 常用菜单"}

        snapshot = PmosPage(FakeSession()).snapshot()
        self.assertEqual(snapshot.state, PageState.LOGGED_IN)
        self.assertEqual(snapshot.detail, "legacy_zcq_logged_in")

    def test_login_check_requires_session_cookie_without_legacy_trade_probe(self) -> None:
        machine = AuthenticationStateMachine(AuthConfig())
        self.assertTrue(machine.check_login("JSESSIONID=abc"))
        self.assertFalse(machine.check_login("tracking=abc"))

    def test_gateway_502_is_not_treated_as_authenticated_page(self) -> None:
        class FakeSession:
            @staticmethod
            def evaluate(*_args, **_kwargs):
                return {"url": "https://pmos.sd.sgcc.com.cn:18080/trade/x", "ready": "complete",
                        "hasPassword": False, "slider": False, "cfca": False, "text": "502 Bad Gateway nginx"}

        self.assertEqual(PmosPage(FakeSession()).snapshot().state, PageState.GATEWAY_ERROR)


if __name__ == "__main__":
    unittest.main()
