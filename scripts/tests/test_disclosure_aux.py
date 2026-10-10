"""[AUX-V1-r3] isolated contract tests (no live PMOS calls)."""

from __future__ import annotations

import calendar
import json
import re
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import requests

from scripts.crawler.collect.disclosure_aux import (
    SOURCE_REGISTRY,
    PmosDisclosureAuxCrawler,
    SourceResult,
    STATUS_COMPLETE,
    STATUS_EMPTY_VALID,
    STATUS_FAILED_SOURCE,
    STATUS_PARTIAL,
    STATUS_SKIPPED_NOT_READY,
    parse_contract,
    parse_event,
    parse_unit,
    record_key,
    _safe_diag_text,
)
from scripts.crawler.collect.crawl_disclosure_aux import (
    _diagnostic_phase,
    _auth,
    _collect_with_auth_recovery,
    _default_aux_config,
    build_parser,
    configure_aux_logging,
    load_aux_config,
    _monthly_sources_to_skip,
)
from scripts.crawler.auth.auto_crawler.config import AuthConfig
from scripts.crawler.runtime_lock import RuntimeLock, RuntimeLockError
from scripts.crawler.sync_db import disclosure_aux as aux_db


class _Response:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return json.loads(self.text)


class _HtmlResponse:
    status_code = 503
    headers = {"content-type": "text/html"}
    text = "<html><h1>503 Service Unavailable</h1></html>"

    def json(self):
        raise ValueError("not JSON")


class _FakeCrawler(PmosDisclosureAuxCrawler):
    def __init__(self, root: Path, response):
        super().__init__(cookie="", output_dir=root)
        self.response = response

    def _req(self, method, url, **kwargs):  # noqa: ARG002
        return self.response

    def _aux_request(self, spec, params):  # noqa: ARG002
        return self.response


class DisclosureAuxTests(unittest.TestCase):
    def test_diagnostic_phase_records_running_and_pass_or_failure(self):
        reporter = Mock()
        with _diagnostic_phase("synthetic", reporter=reporter, interval_sec=0.05):
            pass
        reporter.stage.assert_any_call("synthetic", "RUNNING")
        reporter.stage.assert_any_call("synthetic", "PASS", elapsed_sec=ANY)

        reporter.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with _diagnostic_phase("synthetic", reporter=reporter, interval_sec=0.05):
                raise RuntimeError("probe failure")
        reporter.stage.assert_any_call("synthetic", "RUNNING")
        reporter.stage.assert_any_call(
            "synthetic", "FAIL", error_type="RuntimeError", error="probe failure",
            elapsed_sec=ANY,
        )

    def test_diagnostic_phase_emits_heartbeat_for_long_phase(self):
        with self.assertLogs("crawl_disclosure_aux", level="WARNING") as captured:
            with _diagnostic_phase("synthetic", interval_sec=0.01):
                time.sleep(0.03)
        self.assertTrue(any("phase still running phase=synthetic" in line for line in captured.output))

    def test_diagnostic_error_text_redacts_credentials_and_bounds_length(self):
        value = _safe_diag_text("HTTP error token=secret cookie:abc password=hunter2 " + ("x" * 600))
        self.assertNotIn("secret", value)
        self.assertNotIn("abc", value)
        self.assertNotIn("hunter2", value)
        self.assertIn("<redacted>", value)
        self.assertLessEqual(len(value), 400)

    def test_missing_aux_config_is_created_safe_and_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "config.json").write_text("{}", encoding="utf-8")
            (base / "db_config.json").write_text("{}", encoding="utf-8")
            config_path = base / "config_disclosure_aux.json"
            config = load_aux_config(config_path)
            self.assertTrue(config_path.exists())
            self.assertEqual(config["auth_config_path"], "config.json")
            self.assertEqual(config["db_config_path"], "db_config.json")
            self.assertEqual(config["output_dir"], "output_aux")
            self.assertFalse(config["db_upload"])
            self.assertEqual(config, _default_aux_config(base))

    def test_config_has_no_credentials(self):
        path = Path("dist/crawler/辅助信息披露爬虫/config_disclosure_aux.example.json")
        data = path.read_text(encoding="utf-8").lower()
        self.assertNotIn('"password"', data)
        self.assertNotIn('"cookie"', data)
        self.assertNotIn('"token"', data)

    def test_auth_force_new_disables_existing_browser_reuse(self):
        """[AUX-V1-r4] auth recovery uses the shared state machine, new browser mode."""
        reporter = Mock()
        config = AuthConfig(browser_reuse=True)
        auth_result = SimpleNamespace(cookie="cookie", debug_port=9333)
        with patch("scripts.crawler.collect.crawl_disclosure_aux.AuthConfig.from_file", return_value=config), \
             patch("scripts.crawler.collect.crawl_disclosure_aux.AuthenticationStateMachine") as machine:
            machine.return_value.run.return_value = auth_result
            result = _auth(Path("auth.json"), reporter, force_new=True)
        self.assertIs(result, auth_result)
        forced_config = machine.call_args.args[0]
        self.assertFalse(forced_config.browser_reuse)
        # [AUX-V1-r11f] 重开层沿用原 profile 目录（临时 profile 会吞掉 CFCA/UKey
        # 原生弹窗、卡在 CERTIFICATE），不再注入 _aux_recovery_ 临时目录。
        self.assertNotIn("_aux_recovery_", str(forced_config.browser_profile_dir or ""))
        reporter.stage.assert_any_call(
            "browser_cdp", "RETRY", mode="force_new",
            reason="existing_qctc_session_rejected",
        )
        reporter.stage.assert_any_call("auth", "RUNNING")
        reporter.stage.assert_any_call("auth", "PASS", elapsed_sec=0.0)

    def test_auth_rejection_triggers_exactly_one_force_new_retry(self):
        """[AUX-V1-r4] 401 recovery is bounded and retry uses refreshed crawler."""
        from scripts.crawler.collect.crawl_disclosure_aux import AuxAuthRejected

        reporter = Mock()
        old_crawler = Mock()
        old_crawler.collect.side_effect = AuxAuthRejected("HTTP 401")
        recovered = Mock()
        recovered.collect.return_value = ["ok"]
        auth_result = SimpleNamespace(cookie="new", debug_port=9333)
        with tempfile.TemporaryDirectory() as temp, \
             patch("scripts.crawler.collect.crawl_disclosure_aux._auth", return_value=auth_result) as auth, \
             patch("scripts.crawler.collect.crawl_disclosure_aux._new_crawler", return_value=recovered), \
             patch("scripts.crawler.collect.crawl_disclosure_aux.AuxAuthRejected", AuxAuthRejected):
            crawler, results = _collect_with_auth_recovery(
                old_crawler, auth_path=Path(temp) / "auth.json", reporter=reporter,
                output_dir=Path(temp), source="all", business_date="2026-09-23", unitid=None,
            )
        self.assertIs(crawler, recovered)
        self.assertEqual(results, ["ok"])
        auth.assert_called_once_with(Path(temp) / "auth.json", reporter, force_new=True)
        recovered.fetch_csrf_token.assert_called_once_with()
        recovered.collect.assert_called_once_with(source="all", business_date="2026-09-23", unitid=None)

    def test_auth_rejection_after_force_new_does_not_loop(self):
        """[AUX-V1-r4] a second rejection bubbles out; no third browser is launched."""
        from scripts.crawler.collect.crawl_disclosure_aux import AuxAuthRejected

        reporter = Mock()
        old_crawler = Mock()
        old_crawler.collect.side_effect = AuxAuthRejected("HTTP 401")
        recovered = Mock()
        recovered.collect.side_effect = AuxAuthRejected("HTTP 401")
        with tempfile.TemporaryDirectory() as temp, \
             patch("scripts.crawler.collect.crawl_disclosure_aux._auth", return_value=SimpleNamespace(cookie="new", debug_port=9333)) as auth, \
             patch("scripts.crawler.collect.crawl_disclosure_aux._new_crawler", return_value=recovered), \
             patch("scripts.crawler.collect.crawl_disclosure_aux.AuxAuthRejected", AuxAuthRejected):
            with self.assertRaises(AuxAuthRejected):
                _collect_with_auth_recovery(
                    old_crawler, auth_path=Path(temp) / "auth.json", reporter=reporter,
                    output_dir=Path(temp), source="all", business_date="2026-09-23", unitid=None,
                )
        auth.assert_called_once()
        recovered.collect.assert_called_once()

    def test_shared_lock_blocks_second_instance(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".crawler.lock"
            first = RuntimeLock(path).acquire()
            try:
                with self.assertRaises(RuntimeLockError):
                    RuntimeLock(path).acquire()
            finally:
                first.release()

    def test_registry_routes_all_groups(self):
        self.assertTrue(SOURCE_REGISTRY)
        self.assertTrue({spec.group for spec in SOURCE_REGISTRY.values()} >= {"unit", "constraint", "event", "curve", "stat", "contract"})
        self.assertEqual(SOURCE_REGISTRY["fh_char_raw"].target_table, None)

    def test_unit_dictionary_sources_are_raw_only(self):
        """[AUX-V1-r2] dictionary endpoints never become unit entities."""
        with tempfile.TemporaryDirectory() as temp:
            payload = {"code": 0, "data": [{"id": "TYPE-1", "name": "火电"}]}
            for name in ("unit_type", "unit_gengroup"):
                spec = SOURCE_REGISTRY[name]
                self.assertTrue(spec.raw_only)
                self.assertIsNone(spec.target_table)
                crawler = _FakeCrawler(Path(temp), _Response(payload))
                result = crawler.capture_source(spec, business_date="2026-09-22", params={"pdate": "2026-09-22"})
                self.assertEqual(result.status, STATUS_COMPLETE)
                self.assertEqual(result.rows, [])
            self.assertGreaterEqual(len(list((Path(temp) / "raw" / "2026-09-22").glob("*.json"))), 2)

    def test_dependency_source_never_sends_blank_unitid(self):
        """[AUX-V1-r2] dependency guard runs before transport."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.session.request = Mock(side_effect=AssertionError("HTTP must not be sent"))
            result = crawler.collect(source="unit_constraint", business_date="2026-09-22")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].status, STATUS_FAILED_SOURCE)
            self.assertIn("AUX_DEPENDENCY_MISSING unitid", result[0].error)
            crawler.browser_debug_port = 9222
            crawler._browser_req = Mock(return_value=_Response({"code": 0, "data": []}))
            crawler.collect(source="unit_constraint", business_date="2026-09-22", unitid="U1")
            self.assertTrue(crawler._browser_req.called)

    def test_hourly_legacy_dependency_never_sends_blank_unitid(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.session.request = Mock(side_effect=AssertionError("HTTP must not be sent"))
            result = crawler.collect(source="generation_hourly_net", business_date="2026-09-22")
            self.assertEqual(result[0].status, STATUS_FAILED_SOURCE)
            self.assertIn("AUX_DEPENDENCY_MISSING unitid", result[0].error)

    def test_group_and_all_selection_exclude_disabled_sources(self):
        """[AUX-V1-r2] group/all are enabled-source allowlists."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response({"code": 0, "data": []}))
            event_names = {item.name for item in crawler.collect(source="event", business_date="2026-09-22")}
            curve_names = {item.name for item in crawler.collect(source="curve", business_date="2026-09-22")}
            all_names = {item.name for item in crawler.collect(source="all", business_date="2026-09-22")}
            self.assertNotIn("maintenance_plan", event_names)
            self.assertNotIn("maintenance_init", event_names)
            self.assertNotIn("fh_char_raw", curve_names)
            self.assertNotIn("zd_llx_raw", curve_names)
            self.assertNotIn("max_min_raw", curve_names)
            self.assertNotIn("unit_constraint", all_names)
            self.assertNotIn("unit_component", all_names)
            self.assertNotIn("unit_constraint_jjcq", all_names)
            self.assertIn("unit_info", all_names)
            self.assertIn("unit_month_limit", all_names)
            self.assertNotIn("unit_master", all_names)
            self.assertNotIn("run_line", all_names)

    def test_all_designed_runs_registry_and_reports_not_ready_without_http(self):
        """[AUX-V1-r10] one CLI mode sweeps registered sources, but not unsafe requests."""
        with tempfile.TemporaryDirectory() as temp:
            class RecordingCrawler(_FakeCrawler):
                def __init__(self, root, response):
                    super().__init__(root, response)
                    self.captured = []

                def capture_source(self, spec, *, business_date=None, params=None):
                    self.captured.append(spec.name)
                    return super().capture_source(spec, business_date=business_date, params=params)

            crawler = RecordingCrawler(Path(temp), _Response({"code": 0, "data": []}))
            results = crawler.collect(source="all-designed", business_date="2026-09-24")
            result_names = {item.name for item in results}
            self.assertEqual(result_names, set(SOURCE_REGISTRY))
            self.assertNotIn("maintenance_plan", crawler.captured)
            self.assertNotIn("maintenance_init", crawler.captured)
            self.assertNotIn("unit_count_stat", crawler.captured)
            self.assertNotIn("unit_constraint", crawler.captured)
            self.assertNotIn("generation_hourly_net", crawler.captured)
            self.assertNotIn("fh_char_raw", crawler.captured)  # type argument is not yet in the contract builder
            # [AUX-V1-r11b] unrouted qctc_pm_trade_inside sources must not send requests either
            for name in ("unit_type", "unit_master", "special_unit_tag", "run_line"):
                self.assertNotIn(name, crawler.captured)
            for item in results:
                skipped = {"maintenance_plan", "maintenance_init", "unit_count_stat", "unit_constraint",
                           "unit_component", "unit_constraint_jjcq", "generation_hourly_net", "fh_char_raw"}
                if item.name in skipped or "qctc_pm_trade_inside" in SOURCE_REGISTRY[item.name].path:
                    self.assertEqual(item.status, STATUS_SKIPPED_NOT_READY)
                    self.assertIn("no live request sent", item.error)
            self.assertIn("unit_info", crawler.captured)
            self.assertIn("unit_month_limit", crawler.captured)

    def test_all_designed_monthly_sources_are_scheduled_once_per_month(self):
        seen = set()
        self.assertEqual(_monthly_sources_to_skip("all-designed", "2026-09-24", seen), set())
        monthly = {spec.name for spec in SOURCE_REGISTRY.values() if spec.resolution == "monthly"}
        self.assertEqual(_monthly_sources_to_skip("all-designed", "2026-09-23", seen), monthly)
        # Existing modes are unchanged.
        self.assertEqual(_monthly_sources_to_skip("all", "2026-09-23", seen), set())

    def test_all_designed_is_explicit_and_legacy_all_stays_allowlisted(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["--source", "all-designed"]).source, "all-designed")
        self.assertEqual(parser.parse_args(["--source", "all"]).source, "all")

    def test_exact_source_name_is_explicit_debug_opt_in(self):
        """[AUX-V1-r2] disabled source requires an exact name."""
        parser = build_parser()
        self.assertEqual(parser.parse_args(["--source", "maintenance_plan"]).source, "maintenance_plan")
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response({"code": 0, "data": []}))
            result = crawler.collect(source="maintenance_plan", business_date="2026-09-22")
            self.assertEqual([item.name for item in result], ["maintenance_plan"])

    def test_har_route_and_confirmed_methods(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response({"code": 0, "data": []}))
            self.assertEqual(crawler._source_url(SOURCE_REGISTRY["run_line"]), "https://pmos.sd.sgcc.com.cn:18080/qctc/qctc_pm_trade_inside/trade/daRqxxpl/getRunLine")
            already = SimpleNamespace(request_path="/qctc/qctc_pm_trade_outside/x", path="/qctc/qctc_pm_trade_outside/x")
            self.assertNotIn("/qctc/qctc/", crawler._source_url(already))
            self.assertEqual(crawler._source_url(SOURCE_REGISTRY["unit_info"]), "https://pmos.sd.sgcc.com.cn:18080/qctc/qctc_pm_trade_outside/trade/DaJyjgfbPlantQuery/getUnitInfo")
            self.assertEqual(crawler._source_url(SOURCE_REGISTRY["unit_month_limit"]), "https://pmos.sd.sgcc.com.cn:18080/zcq/jysbys/ydfdcsxyhcx.do?method=getarcdetailNxdcFd")
            self.assertEqual(crawler._source_url(SOURCE_REGISTRY["generation_contract_limit"]), "https://pmos.sd.sgcc.com.cn:18080/zcq/jysbys/fdczxsbedcx.do?method=getTableRows")
        for name in ("unit_master", "unit_type", "unit_gengroup", "special_unit_tag", "transmission_maintenance", "reserve_security", "run_line", "debug_line"):
            self.assertEqual(SOURCE_REGISTRY[name].method, "GET")
        self.assertEqual(SOURCE_REGISTRY["net_contract_day"].method, "POST")
        self.assertTrue(SOURCE_REGISTRY["net_contract_day"].params_in_query)
        self.assertIn("appkey=18", SOURCE_REGISTRY["net_contract_day"].page_url)
        self.assertIn("appkey=93", SOURCE_REGISTRY["generation_contract_limit"].page_url)
        self.assertIn("appkey=94", SOURCE_REGISTRY["generation_hourly_net"].page_url)
        self.assertIn("appkey=81", SOURCE_REGISTRY["unit_month_limit"].page_url)
        self.assertEqual(SOURCE_REGISTRY["unit_month_limit"].method, "POST")
        self.assertTrue(SOURCE_REGISTRY["unit_month_limit"].params_in_query)
        self.assertEqual(SOURCE_REGISTRY["generation_contract_limit"].method, "POST")

    def test_qctc_get_uses_browser_same_origin_as_primary(self):
        """[AUX-V1-r10-diag2] QCTC must follow the working 96 browser transport."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            response = _Response({"code": 0, "data": []})
            crawler._browser_req = Mock(return_value=response)
            crawler.session.request = Mock(side_effect=AssertionError("QCTC must not use Python first"))
            crawler._aux_request(SOURCE_REGISTRY["run_line"], {"pdate": "2026-09-22"})
            self.assertEqual(crawler._browser_req.call_args.args[0], "GET")
            self.assertEqual(crawler._browser_req.call_args.kwargs["page_url"], SOURCE_REGISTRY["run_line"].page_url)
            self.assertEqual(crawler._browser_req.call_args.kwargs["params"], {"pdate": "2026-09-22"})

    def test_legacy_get_keeps_python_transport_then_browser_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            response = _Response({"code": 0, "data": []})
            request = Mock(return_value=response)
            crawler.session.request = request
            crawler._aux_request(SOURCE_REGISTRY["generation_contract_limit"], {"dmonth": "2026-09"})
            self.assertTrue(request.called)

    def test_legacy_browser_fallback_navigates_matching_appkey_route(self):
        """[AUX-V1-r10-route1] Different legacy sources use different page contexts."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.session.request = Mock(side_effect=requests.Timeout("legacy direct unavailable"))
            crawler._browser_req = Mock(return_value=_Response({"code": 0, "data": []}))
            crawler._navigate_legacy_route_page = Mock()
            crawler._aux_request(SOURCE_REGISTRY["generation_contract_limit"], {"dmonth": "2026-09"})
            crawler._navigate_legacy_route_page.assert_called_once_with(SOURCE_REGISTRY["generation_contract_limit"])
            self.assertIn("appkey=93", crawler._browser_req.call_args.kwargs["page_url"])

    def test_unit_month_limit_reuses_live_zcq_csrf_contract(self):
        """[AUX-V1-r8] Document navigation supplies token; AJAX POST stays unchanged."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.browser_debug_port = 9222
            page_url = "https://pmos.sd.sgcc.com.cn:18080/zcq/jysbys/ydfdcsxyhcx.do?appkey=81"
            html = (
                '<meta name="_csrf" content="csrf-value-123" />'
                '<meta name="_csrf_header" content="X-CSRF-TOKEN" />'
            )
            page_response = SimpleNamespace(status_code=200, text=html)
            api_response = _Response({"data": [{"unitname": "unit-1", "jzdlzb": "0.8"}]})
            crawler._legacy_zcq_document_get = Mock(return_value=page_response)
            crawler._browser_req = Mock(return_value=api_response)
            crawler.session.request = Mock(side_effect=requests.Timeout("direct transport unavailable"))

            response = crawler._aux_request(
                SOURCE_REGISTRY["unit_month_limit"],
                {"dmonth": "2026-09", "smonth": "2026-09"},
            )

            self.assertIs(response, api_response)
            crawler._legacy_zcq_document_get.assert_called_once_with(page_url)
            request_headers = crawler.session.request.call_args.kwargs["headers"]
            self.assertEqual(request_headers["X-CSRF-TOKEN"], "csrf-value-123")
            self.assertEqual(request_headers["Referer"], page_url)
            self.assertEqual(crawler._browser_req.call_args.args[0], "POST")
            self.assertEqual(crawler._browser_req.call_args.kwargs["params"], {"dmonth": "2026-09", "smonth": "2026-09"})
            self.assertEqual(crawler._browser_req.call_args.kwargs["headers"]["X-CSRF-TOKEN"], "csrf-value-123")

    def test_unit_month_limit_page_uses_real_document_navigation(self):
        """[AUX-V1-r8] Navigate the existing authenticated PMOS target and read DOM HTML."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.browser_debug_port = 9222
            page_url = "https://pmos.sd.sgcc.com.cn:18080/zcq/jysbys/ydfdcsxyhcx.do?appkey=81"
            html = '<html><meta name="_csrf" content="csrf-nav"><meta name="_csrf_header" content="X-CSRF-TOKEN"></html>'
            page = {"webSocketDebuggerUrl": "ws://127.0.0.1/devtools/page/1"}
            cdp = Mock()
            cdp.call.side_effect = [
                {},  # Page.enable
                {},  # Page.navigate (inside _navigate_for_document)
                {"result": {"value": {"url": page_url, "ready": "complete"}}},  # location.href poll
                {"result": {"value": {"status": 200, "url": page_url, "content_type": "text/html", "html": html}}},
            ]
            with patch("scripts.crawler.collect.disclosure_aux._get_pmos_page", return_value=page) as get_page, \
                 patch("scripts.crawler.collect.disclosure_aux._CdpClient", return_value=cdp):
                response = crawler._legacy_zcq_document_get(page_url)

            get_page.assert_called_once_with(crawler.browser_debug_port, prefer_qctc=True)
            # [AUX-V1-r11b] the navigation must wait for the real target URL and
            # then read the page DOM, not reuse whatever same-origin page was open.
            calls = [call.args[0] for call in cdp.call.call_args_list]
            self.assertIn("Page.navigate", calls)
            self.assertEqual(calls.count("Runtime.evaluate"), 2)
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/html", response.headers["content-type"])
            self.assertIn("csrf-nav", response.text)
            cdp.close.assert_called_once()

    def test_unit_month_limit_fails_closed_if_csrf_missing(self):
        """[AUX-V1-r5] Do not send unverified legacy POST without page CSRF."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler._legacy_zcq_document_get = Mock(return_value=SimpleNamespace(status_code=200, text="<html></html>"))
            crawler.session.request = Mock(side_effect=AssertionError("POST must not be sent"))
            with self.assertRaisesRegex(RuntimeError, "LEGACY_ZCQ_CSRF_MISSING"):
                crawler._aux_request(
                    SOURCE_REGISTRY["unit_month_limit"],
                    {"dmonth": "2026-09", "smonth": "2026-09"},
                )
            crawler.session.request.assert_not_called()

    def test_confirmed_legacy_post_keeps_params_in_query(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            response = _Response({"code": 0, "data": []})
            request = Mock(return_value=response)
            crawler.session.request = request
            crawler._legacy_zcq_document_get = Mock(return_value=SimpleNamespace(
                status_code=200,
                text='<meta name="_csrf" content="token-1"><meta name="_csrf_header" content="X-CSRF-TOKEN">',
            ))
            crawler._aux_request(SOURCE_REGISTRY["unit_month_limit"], {"dmonth": "2026-09", "smonth": "2026-09"})
            kwargs = request.call_args.kwargs
            self.assertEqual(request.call_args.args[0], "POST")
            self.assertEqual(kwargs["params"], {"dmonth": "2026-09", "smonth": "2026-09"})
            self.assertNotIn("data", kwargs)

    def test_404_does_not_guess_second_route(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.session.request = Mock(side_effect=AssertionError("QCTC must not use Python transport"))
            crawler._browser_req = Mock(return_value=_Response({"code": 404}, status=404))
            result = crawler.capture_source(SOURCE_REGISTRY["run_line"], business_date="2026-09-22", params={"pdate": "2026-09-22"})
            self.assertEqual(result.status, STATUS_FAILED_SOURCE)
            crawler._browser_req.assert_called_once()

    def test_non_json_gateway_failure_keeps_body_and_http_status(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _HtmlResponse())
            result = crawler.capture_source(SOURCE_REGISTRY["run_line"], business_date="2026-09-23", params={"pdate": "2026-09-23"})
            self.assertEqual(result.status, STATUS_FAILED_SOURCE)
            self.assertEqual(result.http_status, 503)
            self.assertIn("HTTP 503 returned non-JSON response", result.error)
            self.assertNotIn("JSONDecodeError", result.error)
            raw = result.raw["raw_json"]
            self.assertIn("503 Service Unavailable", raw["_aux_non_json_response"])
            self.assertFalse(raw["_aux_response_truncated"])

    def test_capture_status_and_raw_first_parser_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response({"code": 0, "data": [{"unitid": "U1", "unitname": "x", "fuel_type": "火电"}]}))
            result = crawler.capture_source(SOURCE_REGISTRY["unit_master"], business_date="2026-09-22", params={})
            self.assertEqual(result.status, STATUS_COMPLETE)
            self.assertEqual(len(result.rows), 1)
            raw_files = list((Path(temp) / "raw" / "2026-09-22").glob("*.json"))
            self.assertEqual(len(raw_files), 1)

            bad = _FakeCrawler(Path(temp), _Response({"code": 0, "data": "not-a-list"}))
            result = bad.capture_source(SOURCE_REGISTRY["unit_master"], business_date="2026-09-22", params={})
            self.assertEqual(result.status, "PARTIAL")
            self.assertTrue(list((Path(temp) / "raw" / "2026-09-22").glob("*.json")))

            partial = _FakeCrawler(Path(temp), _Response({"code": 0, "partial": True, "data": [{"unitid": "U2"}]}))
            result = partial.capture_source(SOURCE_REGISTRY["unit_master"], business_date="2026-09-22", params={})
            self.assertEqual(result.status, "PARTIAL")

    def test_401_stops_source(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response({"code": 401}, status=401))
            with self.assertRaises(Exception) as ctx:
                crawler.capture_source(SOURCE_REGISTRY["unit_master"], business_date="2026-09-22")
            self.assertIn("401", str(ctx.exception))

    def test_event_not_broadcast_to_96(self):
        rows = parse_event({"code": 0, "data": [{"Unitid": "U1", "Capacity": 500, "startTime": "2026-09-22"}]}, event_type="special_unit_tag", source_api="/x")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["capacity_mw"], 500.0)

    def test_nested_curve_parallel_arrays(self):
        from scripts.crawler.collect.disclosure_aux import parse_curve
        rows = parse_curve({"code": 0, "data": {"pointList": ["p1", "p2"], "typeList": ["a", "b"], "valList": [[1, 2], [3, 4]]}}, curve_type="run_line", source_api="/run")
        self.assertEqual(len(rows), 4)
        self.assertEqual([row["value_mw"] for row in rows], [1.0, 2.0, 3.0, 4.0])

    def test_nested_deg_line_and_event_wrappers(self):
        from scripts.crawler.collect.disclosure_aux import parse_curve
        rows = parse_curve({"code": 0, "data": {"pdateList": ["p1"], "valueList": [8]}}, curve_type="debug_line", source_api="/deg")
        self.assertEqual(rows[0]["value_mw"], 8.0)
        rows = parse_event({"code": 0, "data": {"tableData": [{"Unitid": "U1"}], "kjCount": 2, "tjCount": 1}}, event_type="special_unit_tag", source_api="/tsjz")
        self.assertEqual(rows[0]["kj_count"], 2)

    def test_bounded_pagination_stops_on_repeated_page(self):
        class PagingCrawler(PmosDisclosureAuxCrawler):
            def __init__(self, root):
                super().__init__(cookie="", output_dir=root)
                self.calls = 0

            def capture_source(self, spec, *, business_date=None, params=None):
                self.calls += 1
                return SourceResult(spec.name, spec.group, STATUS_COMPLETE, 200, "0", [{"record_key": str(i)} for i in range(1000)], {"raw_json": {"data": [{"id": 1}]}, "raw_hash": "x", "captured_at": "now"})

        with tempfile.TemporaryDirectory() as temp:
            crawler = PagingCrawler(Path(temp))
            results = crawler.collect(source="net_contract_day", business_date="2026-09-22")
            self.assertEqual(crawler.calls, 2)
            self.assertEqual(results[-1].status, "PARTIAL")

    def test_all_designed_transport_failures_are_isolated(self):
        """[AUX-V1-r10-route2] Full sweep records bad routes and continues."""
        class FailingSweepCrawler(PmosDisclosureAuxCrawler):
            def __init__(self, root):
                super().__init__(cookie="", output_dir=root)
                self.calls = 0

            def capture_source(self, spec, *, business_date=None, params=None):  # noqa: ARG002
                self.calls += 1
                self._transport_failure_count += 1
                return SourceResult(
                    spec.name, spec.group, STATUS_FAILED_SOURCE, None, None, [],
                    {"raw_json": {"error": "synthetic transport"}},
                    "synthetic transport", True,
                )

        with tempfile.TemporaryDirectory() as temp:
            crawler = FailingSweepCrawler(Path(temp))
            results = crawler.collect(source="all-designed", business_date="2026-09-24")
            self.assertGreaterEqual(crawler.calls, 3)
            self.assertTrue(results)

    def test_unit_master_page_size_100_and_records_total(self):
        """[AUX-V1-r2] bounded offset pagination honors recordsTotal."""
        class UnitPagingCrawler(PmosDisclosureAuxCrawler):
            def __init__(self, root, total=150):
                super().__init__(cookie="", output_dir=root)
                self.calls = []
                self.total = total

            def capture_source(self, spec, *, business_date=None, params=None):
                self.calls.append(dict(params or {}))
                start = int((params or {}).get("start", 0))
                remaining = max(0, self.total - start)
                count = min(100, remaining)
                rows = [{"unit_id": f"U{start + i}"} for i in range(count)]
                return SourceResult(spec.name, spec.group, STATUS_COMPLETE, 200, "0", rows, {"raw_json": {"recordsTotal": self.total}})

        with tempfile.TemporaryDirectory() as temp:
            crawler = UnitPagingCrawler(Path(temp))
            result = crawler.collect(source="unit_master", business_date="2026-09-22")
            self.assertEqual(len(crawler.calls), 2)
            self.assertEqual(crawler.calls[0]["start"], 0)
            self.assertEqual(crawler.calls[0]["length"], 100)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].status, STATUS_COMPLETE)
            self.assertEqual(len(result[0].rows), 150)

            original = SOURCE_REGISTRY["unit_master"]
            SOURCE_REGISTRY["unit_master"] = replace(original, max_pages=2)
            try:
                partial = UnitPagingCrawler(Path(temp), total=250)
                partial_result = partial.collect(source="unit_master", business_date="2026-09-22")
                self.assertEqual(partial_result[0].status, STATUS_PARTIAL)
                self.assertIn("pagination_pending", partial_result[0].error)
            finally:
                SOURCE_REGISTRY["unit_master"] = original

    def test_unit_uses_platform_type_not_name_guess(self):
        rows = parse_unit({"code": 0, "data": [{"unitid": "U1", "unitname": "火电字样但平台类型未知", "fuel_type": None, "jzlx": "PV"}]}, business_date="2026-09-22", source_api="/x")
        self.assertEqual(rows[0]["unit_type"], "PV")
        self.assertIsNone(rows[0]["fuel_type"])

    def test_contract_ratio_undefined(self):
        rows = parse_contract({"code": 0, "data": [{"unitid": "U1", "jzdlzb": "0.4"}]}, record_type="unit_month_limit", source_api="/x")
        self.assertEqual(rows[0]["ratio_status"], "UNDEFINED")
        self.assertEqual(rows[0]["jzdlzb"], 0.4)

    def test_har_contract_payload_is_structured_without_fire_claim(self):
        payload = {"data": [{"plantname": "示例机组", "unitname": "示例", "plantid": "P1", "jhydl": "181.8", "jzdlzb": "0.8"}]}
        rows = parse_contract(payload, record_type="unit_month_limit", source_api="/zcq/jysbys/ydfdcsxyhcx.do")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["jzdlzb"], 0.8)
        self.assertEqual(rows[0]["ratio_status"], "UNDEFINED")
        self.assertIn("business definition pending", rows[0]["ratio_definition"])

    def test_confirmed_month_source_capture_is_structured(self):
        payload = {"code": 0, "data": [{"plantname": "示例机组", "unitname": "示例", "plantid": "P1", "dmonth": "202609", "jhydl": "181.8", "jzdlzb": "0.8"}]}
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response(payload))
            result = crawler.capture_source(SOURCE_REGISTRY["unit_month_limit"], business_date="2026-09-22", params={"dmonth": "2026-09", "smonth": "2026-09"})
            self.assertEqual(result.status, STATUS_COMPLETE)
            self.assertEqual(len(result.rows), 1)
            self.assertEqual(result.rows[0]["jzdlzb"], 0.8)

    def test_record_key_stable(self):
        self.assertEqual(record_key("a", "", 1), record_key("a", "", 1))
        self.assertNotEqual(record_key("a", "1"), record_key("a", "", "1"))

    def test_ddl_is_aux_only(self):
        text = aux_db.ddl_text()
        ok, errors = aux_db.validate_ddl(text)
        self.assertTrue(ok, errors)
        executable = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))
        self.assertNotIn("epf_pmos_96_full", executable)
        self.assertEqual(aux_db.ALLOWED_TABLES, frozenset({"epf_pmos_aux_records"}))
        statements = aux_db._statements(text)
        self.assertEqual(len(statements), 1)
        self.assertTrue(statements[0].lstrip().upper().startswith("CREATE TABLE"))

    def test_db_writer_uses_one_table_and_preserves_source_fields(self):
        cursor = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=cursor)
        context.__exit__ = Mock(return_value=False)
        conn = Mock()
        conn.cursor.return_value = context
        result = SourceResult(
            "unit_month_limit", "contract", STATUS_COMPLETE, 200, "0",
            [{"record_key": "a" * 64, "record_type": "unit_month_limit", "jzdlzb": 0.8, "ratio_status": "UNDEFINED"}],
            {"run_id": "test-run", "request_key": "b" * 64, "request_params": {"dmonth": "2026-09"},
             "source_api": "/monthly", "source_status": STATUS_COMPLETE, "source_row_count": 1,
             "raw_hash": "c" * 64, "raw_json": {"data": [{"jzdlzb": "0.8"}]},
             "business_date": "2026-09-23", "captured_at": "2026-09-23T23:20:54.034803+08:00"},
        )
        self.assertEqual(aux_db.upsert_source_result(conn, result), 2)
        sql_calls = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertEqual(len(sql_calls), 2)
        self.assertTrue(all("`epf_pmos_aux_records`" in sql for sql in sql_calls))
        self.assertTrue(all("epf_pmos_aux_contract" not in sql for sql in sql_calls))
        structured_call = cursor.execute.call_args_list[1]
        column_sql = structured_call.args[0].split(" (", 1)[1].split(") VALUES", 1)[0]
        columns = re.findall(r"`([^`]+)`", column_sql)
        values = dict(zip(columns, structured_call.args[1]))
        self.assertEqual(json.loads(values["record_json"])["jzdlzb"], 0.8)
        self.assertEqual(values["record_kind"], "structured")
        expected_captured_at = datetime(2026, 9, 23, 23, 20, 54, 34803)
        for call in cursor.execute.call_args_list:
            columns_sql = call.args[0].split(" (", 1)[1].split(") VALUES", 1)[0]
            names = re.findall(r"`([^`]+)`", columns_sql)
            stored = dict(zip(names, call.args[1]))
            self.assertEqual(stored["captured_at"], expected_captured_at)
            self.assertIsNone(stored["captured_at"].tzinfo)
        self.assertEqual(json.loads(values["record_json"])["captured_at"], "2026-09-23T23:20:54.034803+08:00")

    def test_unknown_200_is_not_empty_and_unverified_is_raw_only(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response({"code": 0, "message": "ok"}))
            result = crawler.capture_source(SOURCE_REGISTRY["run_line"], business_date="2026-09-22")
            self.assertEqual(result.status, "PARTIAL")
        self.assertTrue(SOURCE_REGISTRY["maintenance_plan"].raw_only)
        self.assertFalse(SOURCE_REGISTRY["maintenance_plan"].enabled_by_default)

    def test_aux_log_name(self):
        with tempfile.TemporaryDirectory() as temp:
            path = configure_aux_logging(Path(temp))
            self.assertEqual(path.name, "aux_crawler.log")
            logging = __import__("logging")
            logging.getLogger("test_aux").info("r1 log")
            self.assertTrue(path.exists())
            for handler in list(logging.getLogger().handlers):
                if getattr(handler, "baseFilename", "") == str(path):
                    logging.getLogger().removeHandler(handler)
                    handler.close()


class InformationDisclosureR11Tests(unittest.TestCase):
    """[AUX-V1-r11] HAR20-confirmed informationDisclosure contract regression."""

    EXPECTED = {
        "dcst_forecast_load": ("ForecastData", "getLoadData", "forecast10424", ("pdate", "versions"), "HAR_NETWORK"),
        "dcst_forecast_tieline": ("ForecastData", "getTieLineData", "forecast10424", ("pdate", "versions"), "HAR_NETWORK"),
        "dcst_forecast_unit_overhaul": ("ForecastData", "getUnitOverhaulData", "forecast10424", ("pdate", "versions"), "HAR_NETWORK"),
        "dcst_tmp_table_cols": ("RealityTmpData", "getTableCols", "actualTemporary10425", ("pdate",), "HAR_NETWORK"),
        "dcst_tmp_load": ("RealityTmpData", "getLoadData", "actualTemporary10425", ("pdate",), "HAR_NETWORK"),
        "dcst_tmp_update_time": ("RealityTmpData", "getUpdateTime", "actualTemporary10425", ("pdate",), "HAR_NETWORK"),
        # [AUX-V1-r13] 原 5 条虚构 RealityTmpData 路径改指 r12 explore 实测 200 的
        # ForecastData/* 端点（evidence 升级来源为 explore 真机录制）。
        "dcst_tmp_block": ("ForecastData", "getBlockData", "forecast10424", ("pdate", "versions"), "EXPLORE_NETWORK"),
        "dcst_tmp_spare": ("ForecastData", "getSpareData", "forecast10424", ("pdate", "versions"), "EXPLORE_NETWORK"),
        "dcst_tmp_unit_overhaul": ("ForecastData", "getUnitOverhaulData", "forecast10424", ("pdate", "versions"), "EXPLORE_NETWORK"),
        "dcst_tmp_open_stop": ("ForecastData", "getOpenAndStopUnitData", "forecast10424", ("pdate", "versions"), "EXPLORE_NETWORK"),
        "dcst_tmp_trans_overhaul": ("ForecastData", "getPowerTransmissionAndTransformationOverhaulData", "forecast10424", ("pdate", "versions"), "EXPLORE_NETWORK"),
    }

    def test_registered_gateway_paths_match_har20(self):
        self.assertEqual(len(self.EXPECTED), 11)
        for name, (module, method, page, contract, evidence) in self.EXPECTED.items():
            spec = SOURCE_REGISTRY[name]
            self.assertEqual(spec.evidence_level, evidence)
            self.assertEqual(spec.method, "GET")
            self.assertEqual(spec.group, "disclosure")
            self.assertEqual(spec.param_contract, contract)
            self.assertEqual(
                spec.path,
                f"/qctc/qctc_pm_trade_outside/informationDisclosure/{module}/{method}",
            )
            self.assertIn(f"/informationDisclosure/{page}", spec.page_url)

    def test_repointed_overhaul_defaults_match_forecast_module(self):
        """[AUX-V1-r13] trans/block/spare/open_stop 默认采集；unit_overhaul 与
        dcst_forecast_unit_overhaul 同端点，保留为禁用别名。"""
        self.assertFalse(SOURCE_REGISTRY["dcst_tmp_unit_overhaul"].enabled_by_default)
        self.assertTrue(SOURCE_REGISTRY["dcst_forecast_unit_overhaul"].enabled_by_default)
        for name in ("dcst_tmp_block", "dcst_tmp_spare", "dcst_tmp_open_stop", "dcst_tmp_trans_overhaul"):
            self.assertTrue(SOURCE_REGISTRY[name].enabled_by_default)

    def test_legacy_inside_guess_is_never_default_enabled(self):
        """The invented *_inside/trade/daRqxxpl paths are not routed; keep them off."""
        for name in ("special_unit_tag", "transmission_maintenance", "reserve_security",
                     "run_line", "debug_line", "zd_llx_raw", "max_min_raw"):
            spec = SOURCE_REGISTRY[name]
            self.assertIn("qctc_pm_trade_inside", spec.path)
            self.assertFalse(spec.enabled_by_default)

    def test_forecast_request_carries_web_path_and_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.browser_debug_port = 9222
            captured: dict = {}

            def fake_route_request(spec, method, url, **kwargs):
                captured.update({"spec": spec.name, "method": method, "url": url, "kwargs": kwargs})
                return _Response({"code": 0, "data": {"TableData": []}})

            crawler._browser_route_request = fake_route_request
            crawler._aux_request(
                SOURCE_REGISTRY["dcst_forecast_tieline"],
                {"pdate": "2026-09-26", "versions": ""},
            )
            self.assertIn("/qctc/qctc_pm_trade_outside/informationDisclosure/ForecastData/getTieLineData", captured["url"])
            self.assertEqual(captured["kwargs"]["headers"]["X-Web-Path"],
                             "/qctc-trade/informationDisclosure/forecast10424")
            self.assertEqual(captured["kwargs"]["headers"]["Accept"], "application/json, text/plain, */*")
            self.assertEqual(captured["kwargs"]["params"], {"pdate": "2026-09-26", "versions": ""})

    def test_reality_tmp_request_has_no_versions_and_no_csrf_header(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.browser_debug_port = 9222
            captured: dict = {}

            def fake_route_request(spec, method, url, **kwargs):
                captured.update({"url": url, "kwargs": kwargs})
                return _Response({"code": 0, "data": {"tableData": []}})

            crawler._browser_route_request = fake_route_request
            crawler._aux_request(SOURCE_REGISTRY["dcst_tmp_load"], {"pdate": "2026-09-01"})
            self.assertIn("/informationDisclosure/RealityTmpData/getLoadData", captured["url"])
            self.assertEqual(captured["kwargs"]["headers"]["X-Web-Path"],
                             "/qctc-trade/informationDisclosure/actualTemporary10425")
            self.assertNotIn("X-CSRF-TOKEN", captured["kwargs"]["headers"])
            self.assertEqual(captured["kwargs"]["params"], {"pdate": "2026-09-01"})

    def test_parse_disclosure_normalises_har20_shapes(self):
        from scripts.crawler.collect.disclosure_aux import parse_disclosure

        rows = parse_disclosure({"code": 0, "data": {"tableData": [
            {"pdate": "2026-09-01", "type": "实际", "mold": "正备用", "periodname": "00:15", "power": "11709.000"}]}},
            source_api="/a", business_date="2026-09-01")
        self.assertEqual(rows[0]["period_name"], "00:15")
        self.assertEqual(rows[0]["power"], 11709.0)
        self.assertEqual(rows[0]["mold"], "正备用")

        rows = parse_disclosure({"code": 0, "data": [{"NUM": "1", "pdate": "2026-09-01", "power": ""}]},
                                source_api="/b", business_date="2026-09-01")
        self.assertEqual(len(rows), 1)

        rows = parse_disclosure({"code": 0, "data": {"TableData": [{"periodname": "1", "zdfh": "1"}]}}, source_api="/c")
        self.assertEqual(len(rows), 1)

        rows = parse_disclosure({"code": 0, "data": {"blockTreeData": [{"id": "2490", "label": "正常方式控美屯双线"}]}},
                                source_api="/d")
        self.assertEqual(rows[0]["tree_id"], "2490")
        self.assertEqual(rows[0]["label"], "正常方式控美屯双线")

        self.assertEqual(parse_disclosure({"code": 0, "data": {}}, source_api="/e"), [])

    def test_block_wrapper_is_not_misread_as_empty(self):
        """HAR20 blockTreeData wrapper must not be reported as an empty result."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = _FakeCrawler(Path(temp), _Response({
                "code": 0,
                "data": {"blockTreeData": [{"id": "2490", "label": "正常方式控美屯双线"}]},
            }))
            result = crawler.capture_source(SOURCE_REGISTRY["dcst_tmp_block"], business_date="2026-09-01")
            self.assertEqual(result.status, STATUS_COMPLETE)
            self.assertEqual(len(result.rows), 1)


class InformationDisclosureR11bTests(unittest.TestCase):
    """[AUX-V1-r11b] cleanup regression: unrouted sources, unitid fallback, 504 retry."""

    def test_unrouted_inside_sources_are_unverified(self):
        """All qctc_pm_trade_inside entries must be UNVERIFIED so all-designed skips them."""
        inside = [s for s in SOURCE_REGISTRY.values() if "qctc_pm_trade_inside" in s.path]
        # [AUX-V1-r11m] maintenance_plan/maintenance_init 已迁往
        # qctc-pm-trade-zcq-out-sxed 模块，inside 计数由 14 变 12。
        self.assertEqual(len(inside), 12)
        for spec in inside:
            self.assertEqual(spec.evidence_level, "UNVERIFIED")
            self.assertFalse(spec.enabled_by_default)

    def test_unitid_fallback_from_crawler_config(self):
        """[AUX-V1-r11b] dependency sources must not skip when --unitid is omitted."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp), unit_id="UNIT-FROM-CONFIG")
            crawler.browser_debug_port = 9222
            crawler._browser_req = Mock(return_value=_Response({"code": 0, "data": []}))
            results = crawler.collect(source="unit_constraint", business_date="2026-09-26")
            self.assertEqual(len(results), 1)
            self.assertNotEqual(results[0].status, STATUS_FAILED_SOURCE)
            self.assertTrue(crawler._browser_req.called)

    def test_504_is_retried_before_failing(self):
        """[AUX-V1-r11b] 504 is a gateway timeout, not a rejection: retry twice."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.browser_debug_port = 9222
            responses = [
                _Response({"error": "timeout"}, status=504),
                _Response({"error": "timeout"}, status=504),
                _Response({"code": 0, "data": {"TableData": [{"periodname": "1"}]}}, status=200),
            ]
            calls = {"n": 0}

            def fake_route_request(spec, method, url, **kwargs):
                calls["n"] += 1
                return responses[min(calls["n"] - 1, len(responses) - 1)]

            crawler._browser_route_request = fake_route_request
            with patch("scripts.crawler.collect.disclosure_aux.time.sleep"):
                response = crawler._aux_request(SOURCE_REGISTRY["dcst_tmp_spare"], {"pdate": "2026-09-26"})
            self.assertEqual(calls["n"], 3)
            self.assertEqual(response.status_code, 200)

    def test_504_exhausted_still_reports_failure(self):
        """Retries must not turn a persistent gateway timeout into success."""
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.browser_debug_port = 9222
            crawler._browser_route_request = Mock(
                return_value=_Response({"error": "timeout"}, status=504)
            )
            with patch("scripts.crawler.collect.disclosure_aux.time.sleep"):
                response = crawler._aux_request(SOURCE_REGISTRY["dcst_tmp_spare"], {"pdate": "2026-09-26"})
            self.assertEqual(response.status_code, 504)
            self.assertEqual(crawler._browser_route_request.call_count, 3)


class R13MaintenanceAndContractCurveTests(unittest.TestCase):
    """[AUX-V1-r13] 检修计划（ForecastData/* 重指）+ 火电合约占比（曲线源）回归。"""

    def test_contract_curve_sources_contract(self):
        for name, method, page in (
            ("zcq_contract_curve24", "get24CjTableData", "appkey=21"),
            ("zcq_contract_curve96", "get96CjTableData", "appkey=15"),
        ):
            spec = SOURCE_REGISTRY[name]
            self.assertEqual(spec.method, "POST")
            self.assertTrue(spec.params_in_query)
            self.assertTrue(spec.enabled_by_default)
            self.assertEqual(spec.evidence_level, "EXPLORE_NETWORK")
            self.assertEqual(spec.group, "contract")
            self.assertEqual(spec.target_table, "epf_pmos_aux_records")
            self.assertIn(f"method={method}", spec.path)
            self.assertIn(page, spec.page_url)
            self.assertEqual(spec.pagination_mode, "offset")

    def test_net_contract_day_enabled_and_csrf_member(self):
        from scripts.crawler.collect.disclosure_aux import _ZCQ_CSRF_SOURCES

        self.assertTrue(SOURCE_REGISTRY["net_contract_day"].enabled_by_default)
        self.assertIn("net_contract_day", _ZCQ_CSRF_SOURCES)
        self.assertIn("unit_month_limit", _ZCQ_CSRF_SOURCES)

    def test_contract_curve_params_cover_whole_month(self):
        from scripts.crawler.collect.disclosure_aux import _contract_curve_params

        params = _contract_curve_params("2026-10-08", "UNIT-1")
        last = calendar.monthrange(2026, 10)[1]
        self.assertEqual(params, {
            "dyid": "UNIT-1", "userProp": "1", "sDate": "2026-10-01",
            "eDate": f"2026-10-{last:02d}", "jylx": "ALL",
            "draw": 1, "start": 0, "length": 50,
        })

    def test_curve_rows_get_unique_keys_and_unit_fallback(self):
        payload = {"recordsFiltered": 2, "data": [
            {"id": 1, "pdate": "20261001", "point": 1, "cjdl": 0.431, "cjjj": 115.0},
            {"id": 2, "pdate": "20261001", "point": 2, "cjdl": 0.5, "cjjj": 116.0},
        ]}
        rows = parse_contract(
            payload, record_type="zcq_contract_curve24",
            source_api="/zcq/dlxxxqcx/dlxxxqYhCx.do?method=get24CjTableData",
            unit_id="UNIT-1", use_row_date=True,
        )
        self.assertEqual(rows[0]["unit_id"], "UNIT-1")
        self.assertEqual(rows[0]["business_date"], "20261001")
        self.assertEqual(rows[0]["quantity"], 0.431)
        self.assertEqual(rows[0]["price"], 115.0)
        self.assertNotEqual(rows[0]["record_key"], rows[1]["record_key"])
        # 旧 contract 源行为不变：行自带 unitid、key 用 business_date。
        legacy = parse_contract(
            {"code": 0, "data": [{"unitid": "U1", "jzdlzb": "0.4"}]},
            record_type="unit_month_limit", source_api="/x", business_date="2026-10-08",
        )
        self.assertEqual(legacy[0]["unit_id"], "U1")
        self.assertEqual(legacy[0]["business_date"], "2026-10-08")

    def test_curve_collect_uses_config_unitid_when_cli_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", unit_id="UNIT-1", output_dir=Path(temp))
            captured: dict[str, object] = {}

            def fake_capture(spec, *, business_date=None, params=None):
                captured.update(dict(params or {}))
                return SourceResult(
                    spec.name, spec.group, STATUS_EMPTY_VALID, 200, "0", [],
                    {"raw_json": {"recordsFiltered": 0}, "business_date": business_date},
                )

            crawler.capture_source = Mock(side_effect=fake_capture)
            results = crawler.collect(source="zcq_contract_curve24", business_date="2026-10-08")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, STATUS_EMPTY_VALID)
            self.assertEqual(captured["dyid"], "UNIT-1")
            self.assertNotIn("AUX_DEPENDENCY_MISSING", results[0].error)

    def test_curve_request_sends_query_params_with_page_csrf(self):
        from scripts.crawler.collect.disclosure_aux import _contract_curve_params

        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            crawler.browser_debug_port = 9222
            page_url = SOURCE_REGISTRY["zcq_contract_curve24"].page_url
            crawler._legacy_zcq_document_get = Mock(return_value=SimpleNamespace(
                status_code=200,
                text='<meta name="_csrf" content="token-c24"><meta name="_csrf_header" content="X-CSRF-TOKEN">',
            ))
            response = _Response({"recordsFiltered": 0, "data": []})
            crawler.session.request = Mock(return_value=response)
            result = crawler._aux_request(
                SOURCE_REGISTRY["zcq_contract_curve24"],
                _contract_curve_params("2026-10-08", "UNIT-1"),
            )
            self.assertIs(result, response)
            crawler._legacy_zcq_document_get.assert_called_once_with(page_url)
            kwargs = crawler.session.request.call_args.kwargs
            self.assertEqual(crawler.session.request.call_args.args[0], "POST")
            self.assertEqual(kwargs["headers"]["X-CSRF-TOKEN"], "token-c24")
            self.assertEqual(kwargs["headers"]["Referer"], page_url)
            self.assertEqual(kwargs["params"]["dyid"], "UNIT-1")
            self.assertEqual(kwargs["params"]["sDate"], "2026-10-01")
            self.assertNotIn("data", kwargs)

    def test_csrf_cache_is_per_page(self):
        with tempfile.TemporaryDirectory() as temp:
            crawler = PmosDisclosureAuxCrawler(cookie="", output_dir=Path(temp))
            page_a = SimpleNamespace(status_code=200, text='<meta name="_csrf" content="t-a"><meta name="_csrf_header" content="X-CSRF-TOKEN">')
            page_b = SimpleNamespace(status_code=200, text='<meta name="_csrf" content="t-b"><meta name="_csrf_header" content="X-CSRF-TOKEN">')
            crawler._legacy_zcq_document_get = Mock(side_effect=[page_a, page_b])
            url_a = SOURCE_REGISTRY["unit_month_limit"].page_url
            url_b = SOURCE_REGISTRY["zcq_contract_curve96"].page_url
            first = crawler._zcq_csrf_headers(url_a)
            self.assertEqual(crawler._zcq_csrf_headers(url_a), first)
            self.assertEqual(crawler._zcq_csrf_headers(url_b)["X-CSRF-TOKEN"], "t-b")
            self.assertEqual(crawler._legacy_zcq_document_get.call_count, 2)
            self.assertEqual(crawler._zcq_csrf_headers(url_a)["X-CSRF-TOKEN"], "t-a")


if __name__ == "__main__":
    unittest.main()
