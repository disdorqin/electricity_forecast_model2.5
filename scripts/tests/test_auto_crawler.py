from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.auto_crawler.config import AuthConfig
from scripts.auto_crawler.handlers import ManualPinHandler, ManualSliderHandler, build_pin_handler, build_slider_handler
from scripts.auto_crawler.main import default_config_path
from scripts.auto_crawler.page import PageState, PmosPage
from scripts.auto_crawler.state_machine import AuthenticationStateMachine


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

    def test_login_url_contains_encoded_trade_service(self) -> None:
        config = AuthConfig()
        self.assertIn("service=https%3A%2F%2Fpmos.sd.sgcc.com.cn%3A18080%2Ftrade", config.login_url)

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


class HandlerTest(unittest.TestCase):
    def test_manual_handlers_are_default(self) -> None:
        config = AuthConfig()
        self.assertIsInstance(build_slider_handler(config), ManualSliderHandler)
        self.assertIsInstance(build_pin_handler(config), ManualPinHandler)

    def test_plugin_mode_requires_plugin_spec(self) -> None:
        with self.assertRaises(ValueError):
            build_slider_handler(AuthConfig(slider_handler="plugin"))


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

    def test_login_check_requires_probe_and_session_cookie(self) -> None:
        class FakePage:
            @staticmethod
            def probe_authenticated(*_args, **_kwargs):
                return True

        machine = AuthenticationStateMachine(AuthConfig())
        self.assertTrue(machine.check_login(FakePage(), "JSESSIONID=abc"))
        self.assertFalse(machine.check_login(FakePage(), "tracking=abc"))


if __name__ == "__main__":
    unittest.main()
