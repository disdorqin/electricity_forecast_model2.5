"""
国网PMOS爬虫核心 — 认证、爬取
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# 96 个 15 分钟时段标签
TIME_LABELS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15)][1:] + ["24:00"]


def period_no_from_time(time_str: str) -> int:
    """将时间标签转为 96 点序号 (00:15→1, 24:00→96)"""
    parts = time_str.split(":")
    return (int(parts[0]) * 60 + int(parts[1])) // 15


def parse_number(val: Any) -> Optional[float]:
    """解析数值，去除千分位逗号"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", "").replace(" ", "").strip()
    if not s or s in ("-", "--", "null", "None", ""):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


class PmosCrawler:
    """国网 PMOS 爬虫"""

    def __init__(
        self,
        base_url: str = "https://pmos.sd.sgcc.com.cn:18080/trade",
        cookie: str = "",
        unit_id: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.unit_id = unit_id

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "X-Requested-With": "XMLHttpRequest",
        })

        # 解析 Cookie 字符串为 requests CookieJar
        for item in cookie.split(";"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                self.session.cookies.set(k.strip(), v.strip())

        self.csrf_token: Optional[str] = None
        self._verify = False  # 禁用 SSL 验证（国网内网证书问题）

    # ------------------------------------------------------------------
    #  底层请求
    # ------------------------------------------------------------------

    def _req(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", 60)
        kwargs.setdefault("verify", self._verify)
        headers = kwargs.pop("headers", {})
        if method.upper() == "POST" and self.csrf_token:
            headers.setdefault("x-csrf-token", self.csrf_token)
        kwargs["headers"] = headers

        resp = self.session.request(method, url, **kwargs)

        if resp.status_code in (302, 401, 403):
            raise requests.HTTPError(
                f"认证失败 (HTTP {resp.status_code})，Cookie 可能已过期",
                response=resp,
            )
        resp.raise_for_status()
        return resp

    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    # ------------------------------------------------------------------
    #  认证
    # ------------------------------------------------------------------

    def fetch_csrf_token(self) -> bool:
        """从主页提取 CSRF token"""
        try:
            resp = self._req("GET", f"{self.base_url}/main/index.do")
            html = resp.text

            patterns = [
                r'name="_csrf".*?content="([^"]+)"',
                r"name=\"_csrf\".*?content='([^']+)'",
                r"csrfToken\s*[=:]\s*['\"]([^'\"]+)['\"]",
            ]
            for pat in patterns:
                m = re.search(pat, html, re.IGNORECASE)
                if m:
                    self.csrf_token = m.group(1)
                    logger.info("CSRF token acquired from HTML")
                    return True

            for c in self.session.cookies:
                if "csrf" in c.name.lower():
                    self.csrf_token = c.value
                    logger.info("CSRF token acquired from cookie: %s", c.name)
                    return True

            logger.warning("CSRF token not found in any location")
            return False
        except Exception as e:
            logger.warning("fetch_csrf_token failed: %s", e)
            return False

    # ------------------------------------------------------------------
    #  日期切换
    # ------------------------------------------------------------------

    def change_date(self, date_str: str) -> bool:
        """切换市场日期"""
        try:
            self._req(
                "GET",
                f"{self.base_url}/main/home/changeDate.do",
                params={"pdate": date_str, "_": self._ts()},
            )
            return True
        except Exception as e:
            logger.warning("changeDate(%s) failed: %s", date_str, e)
            return False

    # ------------------------------------------------------------------
    #  数据爬取
    # ------------------------------------------------------------------

    def crawl_market_overview(self) -> list[dict[str, Any]]:
        """爬取全省市场特征 96 点数据"""
        columns = [
            "Periodid", "systemload", "dfdcload", "excload",
            "fdload", "gfload", "sytsjz", "selfunit", "syjzzj",
        ]
        data = {
            "method": "getNewDetailGridList",
            "draw": "1",
            "start": "0",
            "length": "-1",  # -1 → 返回全部
            "search[value]": "",
            "search[regex]": "false",
            "_": self._ts(),
        }
        for i, col in enumerate(columns):
            data[f"columns[{i}][data]"] = col
            data[f"columns[{i}][name]"] = ""
            data[f"columns[{i}][searchable]"] = "true"
            data[f"columns[{i}][orderable]"] = "false"
            data[f"columns[{i}][search][value]"] = ""
            data[f"columns[{i}][search][regex]"] = "false"
        data["order[0][column]"] = "0"
        data["order[0][dir]"] = "asc"

        resp = self._req(
            "POST",
            f"{self.base_url}/DaJyxxPlDa.do?method=getNewDetailGridList",
            data=data,
            headers={
                "Referer": f"{self.base_url}/DaJyxxPlDa.do?appkey=112",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            },
        )
        result = resp.json()
        rows = result.get("data", []) if isinstance(result, dict) else result
        logger.info("market_overview: %d rows", len(rows))
        return rows

    def _crawl_unit_detail(self, endpoint: str) -> list[dict[str, Any]]:
        """爬取机组级 96 点明细数据（日前/实时共用）"""
        cols = ["periodid", "power", "energy", "cqPrice", "bq", "kt"]
        parts = [f"method=getDetail", f"unitid={self.unit_id}", "draw=1"]
        for i, c in enumerate(cols):
            parts.append(
                f"columns[{i}][data]={c}"
                f"&columns[{i}][searchable]=true"
                f"&columns[{i}][orderable]=false"
                f"&columns[{i}][search][value]="
                f"&columns[{i}][search][regex]=false"
            )
        parts.append("start=0&length=-1")
        parts.append("search[value]=&search[regex]=false")
        parts.append(f"_={self._ts()}")

        url = f"{self.base_url}/{endpoint}?{'&'.join(parts)}"
        resp = self._req("GET", url)
        result = resp.json()
        rows = result.get("data", []) if isinstance(result, dict) else result
        logger.info("%s: %d rows", endpoint.split(".")[0], len(rows))
        return rows

    def crawl_day_ahead(self) -> list[dict[str, Any]]:
        """日前出清价格/出力 96 点"""
        return self._crawl_unit_detail("DaJyjgfbPlantQuery24.do")

    def crawl_realtime(self) -> list[dict[str, Any]]:
        """实时出清价格/出力 96 点"""
        return self._crawl_unit_detail("YxJyjgfbPlantQuery24.do")
