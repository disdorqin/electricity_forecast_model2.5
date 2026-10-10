"""启动前健康检查：判断「现有浏览器能否直接复用」。

为什么需要：业界经验指出——**在动手之前先做一次轻量健康检查**，能避免把
「端口存在但页面不对」误判为可用（只检查 127.0.0.1:9222 有响应，并不代表
它挂载的是我们要的页面）。

配合「优先复用」策略：健康则直接复用（省掉一次完整登录），不健康才重开。

只依赖标准库（urllib + socket），不引入第三方依赖。

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

from .browser_env import probe_port

_DEFAULT_TIMEOUT = 3.0
PMOS_HOST = "pmos.sd.sgcc.com.cn"

#: 「死页面」的强特征 —— favicon.ico / 404 / 错误页虽然也在 PMOS 域名下，
#: 但绝不能被当成可复用会话（否则认证层会误判健康并死等 600s）。
_DEAD_URL_PREFIXES = ("chrome-error://", "about:blank", "data:")
_DEAD_TITLE_MARKERS = ("404", "not found", "500", "502", "503", "bad request",
                       "error page", "internal server error", "whitelabel")


def _looks_like_dead_page(url: str, title: str = "") -> bool:
    """判断一个页面是否是「明显不可用」的静态资源/错误页。"""
    u = (url or "").strip().lower()
    if not u:
        return True
    if u.startswith(_DEAD_URL_PREFIXES):
        return True
    core = u.split("?")[0]
    if core.endswith(".ico") or "favicon" in core:
        return True
    t = (title or "").strip().lower()
    if any(marker in t for marker in _DEAD_TITLE_MARKERS):
        return True
    return False


@dataclass(frozen=True)
class CdpProbe:
    port: int
    reachable: bool = False
    browser: str = ""
    page_count: int = 0
    pmos_url: str = ""
    first_url: str = ""     # 第一个 page 目标的 URL（含死页面，供诊断）
    error: str = ""

    @property
    def is_pmos(self) -> bool:
        return PMOS_HOST in (self.pmos_url or "").lower()

    @property
    def healthy(self) -> bool:
        """可复用的判定：端口可达 + 有**有效**页面 + 页面在 PMOS 域。"""
        return self.reachable and self.page_count > 0 and self.is_pmos

    def summary(self) -> dict:
        return {
            "port": self.port, "reachable": self.reachable, "browser": self.browser,
            "pages": self.page_count, "pmos": self.is_pmos, "healthy": self.healthy,
            "url": self.pmos_url[:180], "first_url": self.first_url[:180],
            "error": self.error,
        }


def _get_json(port: int, path: str, timeout: float) -> object:
    url = f"http://127.0.0.1:{int(port)}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 —— 本地回环
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def probe_cdp(port: int, *, timeout: float = _DEFAULT_TIMEOUT) -> CdpProbe:
    """探测单个调试端口：是否可达、挂着哪些页面、是否有 PMOS 页面。"""
    if not probe_port(port, timeout=min(timeout, 1.5)):
        return CdpProbe(port=port, reachable=False, error="port_closed")

    browser = ""
    try:
        version = _get_json(port, "/json/version", timeout)
        if isinstance(version, dict):
            browser = str(version.get("Browser") or "")
    except Exception as exc:  # noqa: BLE001
        return CdpProbe(port=port, reachable=True, error=f"version_failed:{type(exc).__name__}")

    pmos_url = ""
    first_url = ""
    pages = 0
    try:
        targets = _get_json(port, "/json", timeout)
        if isinstance(targets, list):
            for target in targets:
                if str(target.get("type") or "") != "page":
                    continue
                url = str(target.get("url") or "")
                title = str(target.get("title") or "")
                if not first_url:
                    first_url = url
                # [2026-09-28] 静态资源/错误页(favicon.ico 404 等)不算「有效页面」，
                # 否则会被当成可复用会话，让认证层在 _run_attempt 里死等 600s。
                if _looks_like_dead_page(url, title):
                    continue
                pages += 1
                if PMOS_HOST in url.lower() and not pmos_url:
                    pmos_url = url
    except Exception as exc:  # noqa: BLE001
        return CdpProbe(port=port, reachable=True, browser=browser,
                        error=f"targets_failed:{type(exc).__name__}")

    return CdpProbe(port=port, reachable=True, browser=browser,
                    page_count=pages, pmos_url=pmos_url, first_url=first_url)


def discover_healthy_cdp(ports: list[int], *, timeout: float = _DEFAULT_TIMEOUT) -> CdpProbe | None:
    """在候选端口里找第一个"健康可复用"的 CDP；找不到返回 None。

    返回 None 即意味着应走「重开浏览器」路径（而不是继续复用）。
    """
    for port in ports:
        probe = probe_cdp(int(port), timeout=timeout)
        if probe.healthy:
            return probe
    return None


def probe_all(ports: list[int], *, timeout: float = _DEFAULT_TIMEOUT) -> list[CdpProbe]:
    """探测全部候选端口（用于诊断报告，观察环境现状）。"""
    return [probe_cdp(int(p), timeout=timeout) for p in ports]


def find_poisoned_cdp(ports: list[int], *, timeout: float = _DEFAULT_TIMEOUT) -> list[CdpProbe]:
    """找出「端口可达、但挂着的是死页面(favicon/404/错误页)」的 CDP —— 毒瘤孤儿。

    与 :func:`discover_healthy_cdp` 互补：后者只回答「能不能复用」，本函数回答
    「哪些占用必须清理」。这种孤儿有两个危害（2026-09-28 实锤）：
    ① 认证层会把它误判成可复用会话 → 死等 600s；
    ② 它持有 profile 单实例锁 → 新浏览器起不来(bootstrap_profile_maybe_locked)。
    """
    return [p for p in probe_all(ports, timeout=timeout) if p.reachable and not p.healthy]


# ─────────────────────────────────────────────────────────────────────────
# 后端会话探活：回答「会话还活着吗 / 接口 404 吗」
#
# 为什么需要：历史故障显示，reuse 模式只校验「Cookie 存在 + URL 是 dashboard」
# 就判定已登录，但**后端 session 可能早已过期**，于是带着失效会话一路跑到第一个
# 业务源才报 401。这里在开跑前用一次轻量业务响应做判定，把问题提前暴露。
#
# 具体实现由入口层发起请求（它持有 crawler），本模块只负责判定与结构化，
# 因此 resilience 不耦合任何采集实现。
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SessionProbe:
    """一次后端探活的结果。"""

    alive: bool
    code: str = ""          # 根因码
    reason: str = ""
    status: int | None = None

    def summary(self) -> dict:
        return {"alive": self.alive, "root_cause": self.code,
                "reason": self.reason, "status": self.status}


def probe_session_from_response(
    status: int | None,
    text: str = "",
    *,
    row_count: int | None = None,
) -> SessionProbe:
    """用「开跑前的第一个轻量业务响应」判定后端会话是否仍然有效。

    :param status: HTTP 状态码（0 表示传输层失败）
    :param text: 响应文本（用于匹配"登录信息失效"等门户文案）
    :param row_count: 若已知数据行数，用于区分正常空数据与真实失败
    """
    from .diagnose import diagnose_response  # 局部导入避免循环

    diagnosis = diagnose_response(status, text=text, row_count=row_count)
    alive = diagnosis.code not in (
        "AUTH_REJECTED_401", "SESSION_EXPIRED", "HTTP_NOT_FOUND_404",
        "CDP_PORT_DEAD", "NETWORK_UNREACHABLE",
    )
    return SessionProbe(alive=alive, code=diagnosis.code,
                        reason=diagnosis.reason, status=status)
