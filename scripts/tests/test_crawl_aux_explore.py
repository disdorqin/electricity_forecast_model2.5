"""[AUX-V1-r12] AUX 探索模式回归测试（无真实 PMOS 调用）。

覆盖两层真实调用路径：
1. 入口分派 —— ``apps/crawl_aux.py`` 收到 ``--explore`` 必须走探索模式且**不进采集实现**；
2. 录制器纯逻辑 —— 静态过滤、凭证脱敏、按接口去重、清单与单包产物。
"""

from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from unittest.mock import patch

from scripts.crawler.collect.crawl_disclosure_aux_explore import (
    ExploreSession,
    _endpoint_key,
    _looks_static,
    _redact_headers,
    _redact_params_text,
    make_zip,
)


def _request_event(url: str, *, method: str = "GET", post: str = "") -> dict:
    return {
        "method": "Network.requestWillBeSent",
        "sessionId": "S1",
        "params": {
            "requestId": f"r{abs(hash((url, post))) % 100000}",
            "type": "XHR",
            "frameId": "F1",
            "request": {
                "url": url,
                "method": method,
                "postData": post,
                "headers": {"Cookie": "Admin-Token=SECRETVALUE", "Accept": "application/json"},
            },
        },
    }


class ExploreFilterTests(unittest.TestCase):
    def test_static_assets_are_skipped(self):
        self.assertTrue(_looks_static("https://pmos.sd.sgcc.com.cn:18080/a/app.js?v=1"))
        self.assertTrue(_looks_static("https://x/y.png"))
        self.assertTrue(_looks_static("https://x/y", "text/css"))
        self.assertFalse(_looks_static("https://x/qctc/getData"))
        self.assertFalse(_looks_static("https://x/qctc/getData", "application/json"))

    def test_credentials_never_survive_redaction(self):
        headers = _redact_headers({"Cookie": "Admin-Token=ABC", "X-CSRF-Token": "XYZ", "Referer": "https://h/p"})
        self.assertEqual(headers["Cookie"], "<redacted>")
        self.assertEqual(headers["X-CSRF-Token"], "<redacted>")
        self.assertEqual(headers["Referer"], "https://h/p")
        self.assertNotIn("ABC", json.dumps(headers))
        json_body = _redact_params_text('{"password":"p@ss","pdate":"2026-10-01"}')
        self.assertIn("2026-10-01", json_body)
        self.assertNotIn("p@ss", json_body)
        form = _redact_params_text("user=alice&password=secret123&pdate=2026-10-01")
        self.assertIn("pdate=2026-10-01", form)
        self.assertNotIn("secret123", form)

    def test_endpoint_key_ignores_parameter_values(self):
        # 同一接口不同日期必须归为一个接口——这是「点 30 天只留 1 份样例」的依据。
        first = _endpoint_key("POST", "https://h/qctc/listData?pdate=2026-10-01", '{"pdate":"2026-10-01"}')
        second = _endpoint_key("POST", "https://h/qctc/listData?pdate=2026-09-02", '{"pdate":"2026-09-02"}')
        self.assertEqual(first, second)


class ExploreSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self._tmp.name) / "explore" / "run1"
        self.session = ExploreSession(self.out_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self, url: str, post: str = "", *, method: str = "GET") -> tuple[str, str]:
        event = _request_event(url, method=method, post=post)
        self.session.handle_network_event(event["method"], event["sessionId"], event["params"])
        return event["sessionId"], event["params"]["requestId"]

    def test_static_response_is_dropped_after_headers(self):
        key = self._record("https://h/api/getData")
        self.session.handle_network_event("Network.responseReceived", key[0], {
            "requestId": key[1],
            "response": {"url": "https://h/api/getData", "status": 200, "mimeType": "application/json"},
        })
        self.assertIn(key, self.session.records)
        # 之后发现其实是脚本资源（HAR 里最常见的噪音路径），记录必须被剔除
        key2 = self._record("https://h/api/chunk")
        self.session.handle_network_event("Network.responseReceived", key2[0], {
            "requestId": key2[1],
            "response": {"url": "https://h/api/chunk.js", "status": 200, "mimeType": "application/javascript"},
        })
        self.assertNotIn(key2, self.session.records)
        self.assertGreater(self.session.skipped_static, 0)

    def test_one_body_per_endpoint_and_manifest_output(self):
        business = json.dumps({
            "code": 200,
            "data": {"tableData": [
                {"pdate": "2026-10-01", "contractRatio": 0.62, "unitName": "某火电厂"},
                {"pdate": "2026-09-30", "contractRatio": 0.58, "unitName": "另一火电厂"},
            ]},
        })
        first = self._record("https://h/qctc/netContract/dayList?pdate=2026-10-01", method="GET")
        second = self._record("https://h/qctc/netContract/dayList?pdate=2026-09-30", method="GET")
        for key in (first, second):
            self.session.handle_network_event("Network.loadingFinished", key[0],
                                              {"requestId": key[1], "encodedDataLength": 4096})
        self.assertEqual(len(self.session.body_requests), 2)
        self.session.store_body(first, {"base64Encoded": False, "body": business})
        self.session.store_body(second, {"base64Encoded": False, "body": business})
        # 第二份是同一接口的不同日期：只写 1 个响应体文件
        meta = self.session.finalize(debug_port=9222, stop_reason="用户在控制台按回车结束", elapsed_sec=12.0)
        self.assertEqual(meta["business_requests"], 2)
        self.assertEqual(meta["distinct_endpoints"], 1)
        self.assertEqual(len(list((self.out_dir / "bodies").glob("*"))), 1)
        manifest = (self.out_dir / "发现清单.md").read_text(encoding="utf-8")
        self.assertIn("netContract/dayList", manifest)
        self.assertIn("contractRatio", manifest)
        self.assertIn("火电合约占比", manifest)
        self.assertIn("未登记", manifest)
        self.assertNotIn("Admin-Token=SECRETVALUE", manifest)
        index = (self.out_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(index), 2)
        self.assertIn("<redacted>", index[0])

    def test_failed_body_is_recorded_not_fatal(self):
        key = self._record("https://h/qctc/slowEndpoint")
        self.session.handle_network_event("Network.loadingFinished", key[0],
                                          {"requestId": key[1], "encodedDataLength": 10})
        self.session.store_body(key, None, error="No data found for resource with given identifier")
        meta = self.session.finalize(debug_port=9222, stop_reason="x", elapsed_sec=1.0)
        self.assertEqual(meta["body_errors"], 1)
        self.assertEqual(meta["distinct_endpoints"], 1)
        self.assertIn("未取到响应体", (self.out_dir / "发现清单.md").read_text(encoding="utf-8"))

    def test_zip_contains_manifest_and_no_logs_or_config(self):
        key = self._record("https://h/qctc/getSomething")
        self.session.handle_network_event("Network.loadingFinished", key[0],
                                          {"requestId": key[1], "encodedDataLength": 8})
        self.session.store_body(key, {"base64Encoded": False, "body": '{"data":[]}'})
        self.session.finalize(debug_port=9222, stop_reason="x", elapsed_sec=1.0)
        (self.out_dir.parent / "aux_crawler.log").write_text("sensitive log\n", encoding="utf-8")
        (self.out_dir.parent / "config_disclosure_aux.json").write_text("{}\n", encoding="utf-8")
        archive = make_zip(self.out_dir)
        self.assertTrue(archive.exists())
        self.assertEqual(archive.parent, self.out_dir.parent)
        with zipfile.ZipFile(archive) as handle:
            members = handle.namelist()
            self.assertIn("发现清单.md", members)  # 清单在 zip 根部，可直接双击打开
            self.assertIn("getSomething", handle.read("发现清单.md").decode("utf-8"))
        joined = " ".join(members)
        self.assertNotIn("aux_crawler.log", joined)
        self.assertNotIn("config_disclosure_aux.json", joined)
        self.assertIn("explore_meta.json", joined)
        self.assertIn("requests.jsonl", joined)


class AppEntryDispatchTests(unittest.TestCase):
    """[AUX-V1-r12] 入口分派：--explore 绝不能落到采集实现上。"""

    def test_explore_flag_routes_to_explore_and_not_impl(self):
        from scripts.crawler.apps import crawl_aux

        with patch.object(crawl_aux, "_EXPLORE_AVAILABLE", True), \
             patch.object(crawl_aux, "_explore_main", return_value=0) as explore_main, \
             patch.object(crawl_aux, "_impl_main") as impl_main, \
             patch.object(crawl_aux, "_preflight") as preflight, \
             patch.object(crawl_aux, "_banner"), \
             patch.object(crawl_aux, "_ensure_logging"):
            code = crawl_aux.main(["--explore", "--explore-max-sec", "120"])
        self.assertEqual(code, 0)
        explore_main.assert_called_once_with(["--explore", "--explore-max-sec", "120"])
        impl_main.assert_not_called()
        preflight.assert_called_once()

    def test_explore_unavailable_fails_without_running_collection(self):
        from scripts.crawler.apps import crawl_aux

        with patch.object(crawl_aux, "_EXPLORE_AVAILABLE", False), \
             patch.object(crawl_aux, "_impl_main") as impl_main, \
             patch.object(crawl_aux, "_preflight"), \
             patch.object(crawl_aux, "_banner"), \
             patch.object(crawl_aux, "_ensure_logging"):
            code = crawl_aux.main(["--explore"])
        self.assertEqual(code, 2)
        impl_main.assert_not_called()


class ConsoleSafetyTests(unittest.TestCase):
    def test_no_console_input_does_not_finish_the_recording(self):
        """EOF/无 stdin 必须走「等到最长秒数」，不能把探索模式秒结束。"""
        from scripts.crawler.collect import crawl_disclosure_aux_explore as explore

        report = mock.Mock()
        with patch.object(explore.sys, "stdin", io.StringIO("")):  # 立即 EOF
            event = explore._start_console_trigger(report)
            deadline = time.monotonic() + 3.0
            while not report.event.called and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertFalse(event.is_set())
        codes = [call.args[1] for call in report.event.call_args_list]
        self.assertIn("EXPLORE_NO_CONSOLE", codes)

    def test_enter_key_stops_the_recording(self):
        from scripts.crawler.collect import crawl_disclosure_aux_explore as explore

        report = mock.Mock()
        with patch.object(explore.sys, "stdin", io.StringIO("\n")):
            event = explore._start_console_trigger(report)
            deadline = time.monotonic() + 3.0
            while not event.is_set() and time.monotonic() < deadline:
                time.sleep(0.02)
        self.assertTrue(event.is_set())
        report.event.assert_not_called()

    def test_cprint_survives_gbk_console(self):
        from scripts.crawler.collect.crawl_disclosure_aux_explore import _cprint

        class _GbkStdout:
            encoding = "gbk"

            def __init__(self) -> None:
                self.chunks = []

            def write(self, text):
                self.chunks.append(text)
                return len(text)

            def flush(self) -> None:
                pass

        stream = _GbkStdout()
        with patch("sys.stdout", stream):
            _cprint("探索结束（回车）｜清单：发现清单.md → ok")
        self.assertIn("探索结束", "".join(stream.chunks))


if __name__ == "__main__":
    unittest.main()
