from __future__ import annotations

import json
import os
import tempfile
import unittest
import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scripts.crawler.auth.auto_crawler.browser import CdpSession
from scripts.crawler.auth.auto_crawler.config import AuthConfig
from scripts.crawler.auth.auto_crawler.handlers import BrowserSliderHandler, CaptureSliderHandler, ManualPinHandler, ManualSliderHandler, TemplateSliderSolver, _slider_geometry_from_images, build_pin_handler, build_slider_handler
from scripts.crawler.auth.auto_crawler.main import BUILD_MARKER, default_config_path, ssl_check
from scripts.crawler.auth.auto_crawler.page import PageState, PmosPage
from scripts.crawler.auth.auto_crawler.state_machine import AuthenticationStateMachine


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


class BrowserStartupTest(unittest.TestCase):
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
