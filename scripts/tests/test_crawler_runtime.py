from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.crawler.collect import crawl as core
from scripts.crawler.collect import crawl_96_local as local
from scripts.crawler.observability import RunReport, cookie_summary
from scripts.crawler.sync_db import run_crawler as dbsync


class RuntimePathTest(unittest.TestCase):
    def test_source_runtime_stays_out_of_repository_root(self) -> None:
        base = Path("repo-root")
        self.assertEqual(
            local._resolve_runtime_output_dir(base, frozen=False),
            base / "outputs" / "crawl" / "runtime_96",
        )

    def test_frozen_runtime_stays_next_to_exe(self) -> None:
        base = Path("deploy-dir")
        self.assertEqual(
            local._resolve_runtime_output_dir(base, frozen=True),
            base / "output_96",
        )


class ReportTest(unittest.TestCase):
    def test_report_is_cumulative_and_does_not_store_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            first = RunReport(path, build_version="test", args={"password": "secret", "date": "2026-01-01"})
            first.stage("auth_cookie", "PASS", cookie="Admin-Token=secret")
            first.finish("PASS")
            second = RunReport(path, build_version="test-2")
            second.finish("PARTIAL")

            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(value["runs"]), 2)
            self.assertEqual(value["runs"][-1]["status"], "PARTIAL")
            self.assertNotIn("secret", path.read_text(encoding="utf-8"))
            self.assertTrue(cookie_summary("Admin-Token=secret")["present"])


class PartialTableTest(unittest.TestCase):
    def test_partial_day_keeps_96_structural_rows_and_empty_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "pmos_96_全量.csv"
            with patch.object(local, "TABLE_FILE", table):
                local.append_day_to_table(
                    "2026-01-01",
                    [{"Periodid": "00:15", "systemload": "123"}],
                    [], [], [], [], [],
                )
            with table.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 96)
            self.assertEqual(rows[0]["时段"], "00:15")
            self.assertEqual(rows[0]["直调负荷预测"], "123")
            self.assertEqual(rows[0]["日前一次出清价格"], "")
            self.assertEqual(rows[0]["实时出清价格"], "")
            self.assertEqual(rows[-1]["时段"], "24:00")


class NextDayForecastPrefetchTest(unittest.TestCase):
    def test_qctc_forecast_for_explicit_date_does_not_change_current_market_date(self) -> None:
        spider = core.PmosCrawler(browser_debug_port=9222, data_api_mode="qctc")
        spider.market_date = "2026-09-17"
        calls = []

        def fake_get(path, *, params=None, page_url=None, web_path=None):
            calls.append((path, params, page_url, web_path))
            return {
                "code": 0,
                "data": {
                    "TableData": [{
                        "periodname": "00:15",
                        "zdfh": "1", "dfdczj": "2", "llxfh": "3",
                        "fd": "4", "gf": "5", "hd": "6", "zbjz": "7", "syzj": "8",
                    }]
                },
            }

        spider._qctc_get = fake_get
        rows = spider.crawl_market_forecast_for_date("2026-09-18")
        self.assertEqual(calls[0][1]["pdate"], "2026-09-18")
        self.assertEqual(spider.market_date, "2026-09-17")
        self.assertEqual(rows[0]["systemload"], "1")
        self.assertEqual(rows[0]["syjzzj"], "8")

    def test_complete_next_day_forecast_writes_only_forecast_columns(self) -> None:
        class FakeSpider:
            def crawl_market_forecast_for_date(self, _target):
                rows = []
                for pno in range(1, 97):
                    pid = "24:00" if pno == 96 else f"{pno // 4:02d}:{(pno % 4) * 15:02d}"
                    row = {"Periodid": pid}
                    row.update({field: str(1000 + pno) for field in local.FORECAST_ZH})
                    rows.append(row)
                return rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "pmos_96_全量.csv"
            raw_dir = root / "next_forecast"
            raw_dir.mkdir()
            with (
                patch.object(local, "TABLE_FILE", table),
                patch.object(local, "NEXT_FORECAST_RAW_DIR", raw_dir),
            ):
                info, _, _ = local.prefetch_next_day_forecast(
                    {}, spider=FakeSpider(), target_date="2026-09-18"
                )
            self.assertTrue(info["complete"])
            with table.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 96)
            self.assertNotEqual(rows[0]["直调负荷预测"], "")
            self.assertEqual(rows[0]["直调负荷实际"], "")
            self.assertEqual(rows[0]["日前出清价格"], "")
            self.assertEqual(rows[0]["实时出清价格"], "")
            self.assertTrue((raw_dir / "2026-09-18.json").exists())

    def test_incomplete_next_day_forecast_keeps_raw_but_not_formal_table(self) -> None:
        class FakeSpider:
            def crawl_market_forecast_for_date(self, _target):
                return [{"Periodid": "00:15", "systemload": "1"}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "pmos_96_全量.csv"
            raw_dir = root / "next_forecast"
            raw_dir.mkdir()
            with (
                patch.object(local, "TABLE_FILE", table),
                patch.object(local, "NEXT_FORECAST_RAW_DIR", raw_dir),
            ):
                info, _, _ = local.prefetch_next_day_forecast(
                    {}, spider=FakeSpider(), target_date="2026-09-18"
                )
            self.assertFalse(info["complete"])
            self.assertFalse(table.exists())
            self.assertTrue((raw_dir / "2026-09-18.json").exists())


class DisclosureSourceSeparationTest(unittest.TestCase):
    def test_daily_capture_keeps_final_tmp_and_boundary_separate(self) -> None:
        class FakeSpider:
            base_url = "https://pmos.test"

            def change_date(self, _date):
                return True

            def crawl_market_overview(self):
                return [{"Periodid": "00:15", "systemload": "100", "dfdcload": "10", "excload": "20", "fdload": "30", "gfload": "40"}]

            def crawl_market_overview_actual_final(self):
                return [{"Periodid": "00:15", "systemload": "90", "dfdcload": "9", "excload": "19", "fdload": "29", "gfload": "39"}]

            def crawl_market_overview_actual_temporary(self):
                return [{"Periodid": "00:15", "systemload": "91", "dfdcload": "9.1", "excload": "19.1", "fdload": "29.1", "gfload": "39.1"}]

            def crawl_market_boundary(self):
                return [{"Periodid": "00:15", "qwfh": "110", "systemload": "101", "excload": "21", "fdload": "31", "gfload": "41"}]

            def crawl_day_ahead_first(self):
                return []

            def crawl_day_ahead(self):
                return []

            def crawl_realtime(self):
                return []

            def crawl_optional_market_data(self):
                return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            table = root / "pmos_96_全量.csv"
            with (
                patch.object(local, "RAW_DIR", raw_dir),
                patch.object(local, "TABLE_FILE", table),
                patch.object(local.time, "sleep", return_value=None),
            ):
                result = local.crawl_one_day("2026-09-19", {}, spider=FakeSpider())
            payload = json.loads((raw_dir / "2026-09-19.json").read_text(encoding="utf-8"))
            self.assertEqual(result["actual_final"], 1)
            self.assertEqual(result["actual_temporary"], 1)
            self.assertEqual(result["forecast_boundary"], 1)
            self.assertEqual(payload["actual"][0]["systemload"], "90")
            self.assertEqual(payload["actual_final"][0]["systemload"], "90")
            self.assertEqual(payload["actual_temporary"][0]["systemload"], "91")
            self.assertEqual(payload["forecast_boundary"][0]["systemload"], "101")
            with table.open(encoding="utf-8-sig", newline="") as handle:
                first = next(csv.DictReader(handle))
            self.assertEqual(first["直调负荷实际"], "90")
            self.assertEqual(first["直调负荷临时实际"], "91")
            self.assertEqual(first["边界直调负荷预测"], "101")


class DayAheadFirstPriceTest(unittest.TestCase):
    def test_first_and_second_day_ahead_use_independent_qctc_endpoints(self) -> None:
        spider = core.PmosCrawler(browser_debug_port=9222, data_api_mode="qctc")
        spider.market_date = "2026-09-01"
        calls = []

        def fake_get(path, *, params=None, page_url=None, web_path=None):
            calls.append((path, page_url, params))
            price = "111.1" if "FirQuery" in path else "222.2"
            return {"code": 0, "data": {"data": [{"periodid": "00:15", "cqPrice": price}]}}

        spider._qctc_get = fake_get
        first = spider.crawl_day_ahead_first()
        second = spider.crawl_day_ahead()
        self.assertEqual(first[0]["cqPrice"], "111.1")
        self.assertEqual(second[0]["cqPrice"], "222.2")
        self.assertIn("DaJyjgfbPlantFirQuery/getDetail96", calls[0][0])
        self.assertIn("DaJyjgfbPlantQuery/getDetail96", calls[1][0])
        self.assertIn("onceCqrqjyjg", calls[0][1])
        self.assertIn("towFdcResultQuery", calls[1][1])

    def test_csv_keeps_first_and_second_day_ahead_prices_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "pmos_96_全量.csv"
            with patch.object(local, "TABLE_FILE", table):
                local.append_day_to_table(
                    "2026-09-01", [], [],
                    [{"periodid": "00:15", "cqPrice": "111.1"}],
                    [{"periodid": "00:15", "cqPrice": "222.2"}],
                    [], [],
                )
            with table.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["日前一次出清价格"], "111.1")
            self.assertEqual(row["日前出清价格"], "222.2")

    def test_existing_canonical_table_adds_disclosure_extensions_without_rename(self) -> None:
        executed = []

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None):
                executed.append(sql)

            def fetchone(self):
                return {"table": "epf_pmos_96_full"}

            def fetchall(self):
                return [
                    {"Field": col}
                    for col in (*dbsync.BASE_DATASET_COLUMNS, "日前一次出清价格")
                ]

        class Conn:
            def cursor(self):
                return Cursor()

            def commit(self):
                pass

        dbsync.ensure_full_table_schema(Conn())
        joined = "\n".join(executed)
        self.assertNotIn("RENAME TABLE", joined)
        for column in dbsync.DISCLOSURE_EXTENSION_COLUMNS:
            self.assertIn(f"ADD COLUMN `{column}`", joined)

    def test_upsert_writes_temporary_actual_and_boundary_independently(self) -> None:
        executed = []

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None):
                executed.append((sql, params))

        class Conn:
            def cursor(self):
                return Cursor()

        counts = dbsync.upsert_full_dataset_table(
            Conn(),
            "2026-09-19",
            "UNIT_TEST",
            [{
                "Periodid": "00:15",
                "systemload": "100",
                "qwfh": "110",
            }],
            [{
                "Periodid": "00:15",
                "systemload": "90",
                "qwfh": "105",
            }],
            [], [],
            actual_temporary_rows=[{
                "Periodid": "00:15",
                "systemload": "91",
                "qwfh": "106",
            }],
            boundary_rows=[{
                "Periodid": "00:15",
                "systemload": "101",
                "qwfh": "111",
                "fdload": "31",
            }],
        )
        first_sql, first_params = executed[0]
        self.assertIn("`全网负荷预测`", first_sql)
        self.assertIn("`全网负荷实际`", first_sql)
        self.assertIn("`直调负荷临时实际`", first_sql)
        self.assertIn("`全网负荷临时实际`", first_sql)
        self.assertIn("`边界直调负荷预测`", first_sql)
        self.assertIn("`边界全网负荷预测`", first_sql)
        self.assertEqual(counts["actual_temporary_periods"], 1)
        self.assertEqual(counts["forecast_boundary_periods"], 1)
        self.assertIn(91.0, first_params)
        self.assertIn(101.0, first_params)

    def test_existing_canonical_table_adds_first_price_without_rename(self) -> None:
        executed = []

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None):
                executed.append(sql)

            def fetchone(self):
                return {"table": "epf_pmos_96_full"}

            def fetchall(self):
                return [{"Field": col} for col in dbsync.FULL_DATASET_COLUMNS]

        class Conn:
            def cursor(self):
                return Cursor()

            def commit(self):
                pass

        dbsync.ensure_full_table_schema(Conn())
        joined = "\n".join(executed)
        self.assertIn("ADD COLUMN `日前一次出清价格`", joined)
        self.assertNotIn("RENAME TABLE", joined)
        self.assertNotIn("日前一次出清价格", dbsync.FULL_DATASET_COLUMNS)


class QctcBrowserSelectionTest(unittest.TestCase):
    def test_qctc_context_recognizes_any_18080_route_but_not_portal(self) -> None:
        self.assertTrue(core._is_qctc_context_page_url("https://pmos.sd.sgcc.com.cn:18080/home"))
        self.assertTrue(core._is_qctc_context_page_url(
            "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/informationDisclosure/forecast10424"
        ))
        self.assertFalse(core._is_qctc_context_page_url("https://pmos.sd.sgcc.com.cn/#/dashboard"))

    def test_qctc_fetch_prefers_existing_qctc_tab(self) -> None:
        pages = [
            {"type": "page", "url": "https://pmos.sd.sgcc.com.cn/#/dashboard", "webSocketDebuggerUrl": "ws://old"},
            {"type": "page", "url": "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/informationDisclosure/forecast10424", "webSocketDebuggerUrl": "ws://qctc"},
        ]
        response = type("Response", (), {"json": lambda self: pages, "raise_for_status": lambda self: None})()
        with patch.object(core.requests, "get", return_value=response):
            selected = core._get_pmos_page(9222, prefer_qctc=True)
        self.assertEqual(selected["webSocketDebuggerUrl"], "ws://qctc")

    def test_existing_home_bearer_context_is_ready_without_forecast_navigation(self) -> None:
        page = {
            "type": "page",
            "id": "qctc-home",
            "url": "https://pmos.sd.sgcc.com.cn:18080/home",
            "webSocketDebuggerUrl": "ws://qctc-home",
        }
        state = {
            "url": page["url"],
            "origin": "https://pmos.sd.sgcc.com.cn:18080",
            "path": "/home",
            "ready": "complete",
            "qctc_route": False,
            "bearer_present": True,
            "session_storage_keys": ["token", "userInfo", "tokenTime", "roles"],
            "local_storage_keys": [],
        }

        class FakeCdp:
            instances = []

            def __init__(self, _ws: str):
                self.calls = []
                FakeCdp.instances.append(self)

            def call(self, method, params=None, timeout=30):
                self.calls.append((method, params))
                return {}

            def close(self):
                pass

        with (
            patch.object(core, "_page_targets", return_value=[page]),
            patch.object(core, "_CdpClient", FakeCdp),
            patch.object(core, "_browser_page_state", return_value=state),
        ):
            spider = core.PmosCrawler(browser_debug_port=9222, data_api_mode="qctc")
            self.assertTrue(spider.ensure_qctc_context())
        self.assertFalse(any(
            call[0] == "Page.navigate"
            for client in FakeCdp.instances for call in client.calls
        ))

    def test_existing_bearer_context_is_ready_without_forecast_navigation(self) -> None:
        page = {
            "type": "page",
            "id": "qctc",
            "url": "https://pmos.sd.sgcc.com.cn:18080/qctc-trade/dayAheadTransaction/declare/fillDataMaintenance31237",
            "webSocketDebuggerUrl": "ws://qctc",
        }
        state = {
            "url": page["url"],
            "origin": "https://pmos.sd.sgcc.com.cn:18080",
            "path": "/qctc-trade/dayAheadTransaction/declare/fillDataMaintenance31237",
            "ready": "complete",
            "qctc_route": False,
            "bearer_present": True,
            "session_storage_keys": ["token", "userInfo"],
            "local_storage_keys": [],
        }

        class FakeCdp:
            instances = []

            def __init__(self, _ws: str):
                self.calls = []
                FakeCdp.instances.append(self)

            def call(self, method, params=None, timeout=30):
                self.calls.append((method, params))
                return {}

            def close(self):
                pass

        with (
            patch.object(core, "_page_targets", return_value=[page]),
            patch.object(core, "_CdpClient", FakeCdp),
            patch.object(core, "_browser_page_state", return_value=state),
        ):
            spider = core.PmosCrawler(browser_debug_port=9222, data_api_mode="qctc")
            self.assertTrue(spider.ensure_qctc_context())
        self.assertFalse(any(
            call[0] == "Page.navigate"
            for client in FakeCdp.instances for call in client.calls
        ))

    def test_portal_ticket_flow_switches_to_new_target_without_exposing_ticket(self) -> None:
        portal = {
            "type": "page", "id": "portal",
            "url": "https://pmos.sd.sgcc.com.cn/#/dashboard",
            "webSocketDebuggerUrl": "ws://portal",
        }
        qctc = {
            "type": "page", "id": "qctc",
            "url": "https://pmos.sd.sgcc.com.cn:18080/home",
            "webSocketDebuggerUrl": "ws://qctc",
        }
        target_calls = [0]

        def targets(_port):
            target_calls[0] += 1
            return [portal] if target_calls[0] == 1 else [portal, qctc]

        portal_state = {
            "url": portal["url"], "origin": "https://pmos.sd.sgcc.com.cn",
            "path": "/#/dashboard", "ready": "complete", "qctc_route": False,
            "bearer_present": False, "session_storage_keys": [], "local_storage_keys": [],
        }
        qctc_state = {
            "url": qctc["url"], "origin": "https://pmos.sd.sgcc.com.cn:18080",
            "path": "/home", "ready": "complete", "qctc_route": False,
            "bearer_present": True, "session_storage_keys": ["token"], "local_storage_keys": [],
        }
        states = iter([portal_state, qctc_state])

        class FakeCdp:
            instances = []

            def __init__(self, _ws: str):
                self.calls = []
                FakeCdp.instances.append(self)

            def call(self, method, params=None, timeout=30):
                self.calls.append((method, params))
                if method == "Runtime.evaluate" and params and "sso/token" in params.get("expression", ""):
                    return {"result": {"value": {
                        "ok": True, "opened_new_window": True,
                        "http_status": 200, "business_status": 0,
                    }}}
                return {}

            def close(self):
                pass

        with (
            patch.object(core, "_page_targets", side_effect=targets),
            patch.object(core, "_CdpClient", FakeCdp),
            patch.object(core, "_browser_page_state", side_effect=lambda _cdp: next(states)),
        ):
            spider = core.PmosCrawler(browser_debug_port=9222, data_api_mode="qctc", qctc_auth_wait_sec=2)
            self.assertTrue(spider.ensure_qctc_context())

        calls = [call for client in FakeCdp.instances for call in client.calls]
        expressions = [call[1].get("expression", "") for call in calls if call[0] == "Runtime.evaluate"]
        self.assertTrue(any("/px-common-authcenter/sso/token" in expression for expression in expressions))
        self.assertTrue(any(core.QCTC_SSO_SERVICE_URL in expression and "?ticket=" in expression
                            for expression in expressions))
        self.assertFalse(any(call[0] == "Page.navigate" for call in calls))
        self.assertNotIn("secret-ticket", json.dumps(calls))

    def test_qctc_401_aborts_without_further_date_retry(self) -> None:
        class Reporter:
            def __init__(self):
                self.events = []
                self.finished = None

            def event(self, level, code, message, **details):
                self.events.append(code)

            def stage(self, name, status, **details):
                pass

            def finish(self, status, **details):
                self.finished = status

        response = type("Response", (), {
            "status_code": 401,
            "text": '{"code":2,"msg":"登录信息失效"}',
            "reason": "Unauthorized",
        })()
        reporter = Reporter()
        spider = core.PmosCrawler(
            browser_debug_port=9222, data_api_mode="qctc", reporter=reporter
        )
        spider._browser_req = lambda *args, **kwargs: response
        with self.assertRaises(core.QCTCAuthRejected):
            spider._qctc_get(
                "informationDisclosure/ForecastData/getLoadData",
                params={"pdate": "2026-01-01", "versions": ""},
                page_url=spider.qctc_forecast_page,
                web_path="/qctc-trade/informationDisclosure/forecast10424",
            )
        self.assertIn("QCTC_AUTH_REJECTED", reporter.events)
        self.assertEqual(reporter.finished, "FAIL")

    def test_ticket_helper_success_constructs_sso_url_but_returns_status_only(self) -> None:
        class FakeCdp:
            def call(self, method, params=None, timeout=30):
                self.expression = params["expression"]
                return {"result": {"value": {
                    "ok": True, "opened_new_window": True,
                    "http_status": 200, "business_status": 0,
                    "data": "secret-ticket",
                }}}

        cdp = FakeCdp()
        result = core._open_qctc_via_portal_ticket(cdp)
        self.assertEqual(result["http_status"], 200)
        self.assertTrue(result["ok"])
        self.assertNotIn("data", result)
        self.assertNotIn("secret-ticket", json.dumps(result))
        self.assertIn("/px-common-authcenter/sso/token", cdp.expression)
        self.assertIn(core.QCTC_SSO_SERVICE_URL, cdp.expression)
        self.assertIn("?ticket=", cdp.expression)


if __name__ == "__main__":
    unittest.main()
