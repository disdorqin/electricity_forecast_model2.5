"""根因诊断器：把「异常 / HTTP 响应 / 页面状态 / 环境探测」翻译成根因码。

这是防御机制的第一入口，也是最重要的一环——**先诊断，再决定是否重试**。
调用方拿到 `Diagnosis` 后只需按 `diagnosis.action` 执行，不要自己猜。

自包含实现：不 import auth/collect，通过「异常类型名 + 文本特征」判定，
因此可被任何爬虫复用（96 / AUX / 未来 XXX）。

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

from typing import Any, Iterable

from .codes import Diagnosis

# ── 判据关键词（语义与 auth/auto_crawler/page.py 保持一致，此处独立维护） ──

SESSION_EXPIRED_MARKERS: tuple[str, ...] = (
    "已失效", "会话已过期", "会话超时", "登录已失效", "登录信息失效",
    "登录状态已失效", "请重新登录", "重新登录", "身份验证已过期", "用户未登录",
)

NETWORK_UNREACHABLE_MARKERS: tuple[str, ...] = (
    "无法访问此网站", "网页无法访问", "找不到该网页", "没有互联网连接", "网络连接中断",
    "err_internet_disconnected", "err_name_not_resolved", "err_connection_timed_out",
    "err_connection_refused", "err_network_changed", "err_address_unreachable",
    "err_proxy_connection_failed",
)

#: CDP target 被销毁（导航/重定向/标签关闭）
CDP_TARGET_LOST_MARKERS: tuple[str, ...] = (
    "inspected target navigated or closed", "target closed", "target gone",
    "target not found", "尚未创建可控制的标签页", "未找到 pmos 浏览器标签页",
    "浏览器页面连续", "不可控制",
)

#: 浏览器启动阶段的超时
BOOTSTRAP_TIMEOUT_MARKERS: tuple[str, ...] = (
    "浏览器启动后", "未打开可控 pmos 页面", "bootstrap_failed", "bootstrap failed",
)

#: profile 单实例锁
PROFILE_LOCK_MARKERS: tuple[str, ...] = (
    "singletonlock", "singletoncookie", "singletonsocket",
    "bootstrap_launcher_exited", "profile_maybe_locked", "无法打开用户数据目录",
)

#: 认证/证书阶段
CFCA_MARKERS: tuple[str, ...] = (
    "certificate", "cfca", "ukey", "验证ukey用户口令", "pin",
)

LOGIN_TIMEOUT_MARKERS: tuple[str, ...] = (
    "pmos 登录在", "登录流程超时", "login_timeout", "600s 内未完成", "登录超时",
)

AUTH_REJECT_MARKERS: tuple[str, ...] = (
    "401", "403", "authentication_rejected", "unauthorized", "forbidden",
)

GATEWAY_MARKERS: tuple[str, ...] = (
    "502 bad gateway", "503", "504", "gateway timeout", "service unavailable",
    "httpconnectionpool",  # 兜底：池化连接错误的通用特征，具体仍按端口判定
)

_UNREACHABLE_URL_PREFIXES = ("chrome-error://", "about:blank", "edge-error://")


def _hits(text: str, markers: Iterable[str]) -> list[str]:
    lowered = (text or "").lower()
    return [m for m in markers if m.lower() in lowered]


def _looks_like_local_cdp(text: str) -> bool:
    """判据：错误是否指向本地 CDP 调试端口（127.0.0.1:922x）。"""
    lowered = (text or "").lower()
    if "127.0.0.1" not in lowered and "localhost" not in lowered:
        return False
    return "port=" in lowered or ":922" in lowered


def _mk(code: str, reason: str, **evidence: Any) -> Diagnosis:
    return Diagnosis(code=code, reason=reason, evidence={k: v for k, v in evidence.items() if v not in (None, "", [], {})})


# ─────────────────────────────────────────────────────────────────────────
# 一、异常诊断
# ─────────────────────────────────────────────────────────────────────────

def diagnose_exception(exc: BaseException, *, context: str = "") -> Diagnosis:
    """把一个异常翻译成根因码。判定顺序从「最具体」到「最泛化」。"""
    name = type(exc).__name__
    text = f"{name}: {exc}"
    low = text.lower()

    # 1) 本地 CDP 端点不可达 → 浏览器进程死亡（最容易被误判为"网络问题"）
    if _looks_like_local_cdp(low):
        return _mk("CDP_PORT_DEAD", f"本地 CDP 端口不可达：{name}", exception=name, context=context)

    # 2) profile 单实例锁
    if _hits(low, PROFILE_LOCK_MARKERS):
        return _mk("PROFILE_LOCKED", f"疑似 profile 被占用：{name}", exception=name, context=context)

    # 3) CDP target 丢失（导航竞态）
    if _hits(low, CDP_TARGET_LOST_MARKERS):
        return _mk("CDP_TARGET_LOST", f"CDP 目标失效：{name}", exception=name, context=context)

    # 4) 启动阶段超时
    if _hits(low, BOOTSTRAP_TIMEOUT_MARKERS) or ("timeout" in low and "browser" in low):
        return _mk("BROWSER_BOOTSTRAP_TIMEOUT", f"浏览器启动超时：{name}", exception=name, context=context)

    # 5) 认证被拒（401/403）
    if _hits(low, AUTH_REJECT_MARKERS):
        return _mk("AUTH_REJECTED_401", f"服务端拒绝认证：{name}", exception=name, context=context)

    # 6) 会话失效（业务/页面提示）
    if _hits(low, SESSION_EXPIRED_MARKERS):
        return _mk("SESSION_EXPIRED", f"会话已失效：{name}", exception=name, context=context)

    # 7) CFCA / UKey 阶段卡死
    if _hits(low, CFCA_MARKERS):
        return _mk("CFCA_PIN_STUCK", f"证书/UKey 阶段异常：{name}", exception=name, context=context)

    # 8) 登录超时
    if _hits(low, LOGIN_TIMEOUT_MARKERS):
        return _mk("LOGIN_TIMEOUT", f"登录流程超时：{name}", exception=name, context=context)

    # 9) 网络不可达
    if _hits(low, NETWORK_UNREACHABLE_MARKERS):
        return _mk("NETWORK_UNREACHABLE", f"网络不可达：{name}", exception=name, context=context)

    # 10) 服务端网关类
    if _hits(low, GATEWAY_MARKERS) and any(
        code in low for code in ("502", "503", "504", "gateway")
    ):
        return _mk("HTTP_GATEWAY_5XX", f"服务端网关错误：{name}", exception=name, context=context)

    # 11) 通用连接类错误（无本地端点特征）→ 归网络
    if isinstance(exc, (ConnectionError, TimeoutError)) or "connection" in low or "timed out" in low:
        return _mk("NETWORK_UNREACHABLE", f"连接/超时异常：{name}（无本地端点特征）",
                   exception=name, context=context)

    return _mk("UNKNOWN", f"未能归类：{name}", exception=name, context=context)


# ─────────────────────────────────────────────────────────────────────────
# 二、HTTP 响应诊断
# ─────────────────────────────────────────────────────────────────────────

def diagnose_response(
    status: int | None,
    text: str = "",
    *,
    row_count: int | None = None,
    context: str = "",
) -> Diagnosis:
    """把一次 HTTP 响应翻译成根因码。

    `row_count` 用于区分「业务正常空数据」与「真实失败」——这是最容易被误统计的一类。
    """
    status = int(status or 0)
    low = (text or "").lower()

    if status == 0:
        # 传输层失败：若文本指向本地 CDP 端点则是浏览器问题，否则按网络处理
        if _looks_like_local_cdp(low):
            return _mk("CDP_PORT_DEAD", "传输失败且指向本地 CDP 端点", context=context, status=status)
        if _hits(low, NETWORK_UNREACHABLE_MARKERS):
            return _mk("NETWORK_UNREACHABLE", "传输失败：网络不可达", context=context, status=status)
        return _mk("NETWORK_UNREACHABLE", "传输失败（无响应）", context=context, status=status)

    if status in (401, 403):
        return _mk("AUTH_REJECTED_401", f"HTTP {status}：认证被拒", context=context, status=status)
    if status == 404:
        return _mk("HTTP_NOT_FOUND_404", "HTTP 404：资源/接口不存在", context=context, status=status)
    if status == 429:
        return _mk("HTTP_RATE_LIMITED_429", "HTTP 429：被限流", context=context, status=status)
    if 500 <= status < 600:
        return _mk("HTTP_GATEWAY_5XX", f"HTTP {status}：服务端错误", context=context, status=status)

    if _hits(low, SESSION_EXPIRED_MARKERS):
        return _mk("SESSION_EXPIRED", "响应文本提示会话失效", context=context, status=status)

    if status == 200:
        if row_count is not None and row_count == 0:
            return _mk("NO_DATA", "HTTP 200 且业务数据为空（正常空数据）",
                       context=context, status=status, row_count=row_count)
        return _mk("NO_DATA", "HTTP 200 且无失败特征", context=context, status=status)

    # 其他 2xx/3xx 视为可用（由解析器进一步判定）
    if 200 <= status < 400:
        return _mk("NO_DATA", f"HTTP {status}：按可用处理", context=context, status=status)

    return _mk("UNKNOWN", f"HTTP {status}：未归类", context=context, status=status)


# ─────────────────────────────────────────────────────────────────────────
# 三、页面状态诊断（配合 auth 页面快照的附加字段使用）
# ─────────────────────────────────────────────────────────────────────────

def diagnose_page(
    *,
    url: str = "",
    text: str = "",
    session_expired: bool = False,
    network_unreachable: bool = False,
    certificate_visible: bool = False,
    gateway_error: bool = False,
    context: str = "",
) -> Diagnosis:
    """把页面状态翻译成根因码。字段直接取自共享状态机的 `PageSnapshot`（只读）。"""
    low_url = (url or "").lower()

    if network_unreachable or any(low_url.startswith(p) for p in _UNREACHABLE_URL_PREFIXES):
        return _mk("NETWORK_UNREACHABLE", "页面为浏览器错误页/空白页", context=context, url=url)
    if network_unreachable or _hits(text, NETWORK_UNREACHABLE_MARKERS):
        return _mk("NETWORK_UNREACHABLE", "页面文本命中网络错误", context=context, url=url)
    if gateway_error or "502 bad gateway" in (text or "").lower():
        return _mk("HTTP_GATEWAY_5XX", "页面为网关错误页(502)", context=context, url=url)
    if session_expired or _hits(text, SESSION_EXPIRED_MARKERS):
        return _mk("SESSION_EXPIRED", "页面提示会话已失效", context=context, url=url)
    if certificate_visible:
        return _mk("CFCA_PIN_STUCK", "页面停留在证书(CFCA)选择阶段", context=context, url=url)
    return _mk("UNKNOWN", "页面状态无明确失败特征", context=context, url=url)


# ─────────────────────────────────────────────────────────────────────────
# 四、环境诊断（profile 锁 / 存活进程 / 端口）
# ─────────────────────────────────────────────────────────────────────────

def diagnose_environment(
    *,
    lock_files: Iterable[str] = (),
    live_holders: Iterable[int] = (),
    port_reachable: bool = False,
    port: int | None = None,
) -> Diagnosis:
    """把环境探测结果翻译成根因码。

    语义区分（关键）：
      - 有存活进程持有 profile → `PROFILE_BUSY_BY_LIVE_PROCESS`（**不杀进程**，让调用方换浏览器/等待）
      - 仅残留锁文件、无存活进程 → `PROFILE_LOCKED`（可安全清理锁文件后复用）
      - 端口可达 → 正常（`REUSE`）
    """
    holders = list(live_holders)
    locks = list(lock_files)

    if port_reachable:
        return _mk("UNKNOWN", "端口可达", port=port)  # 调用方应以 action=REUSE 处理；见 codes.UNKNOWN 兜底
    if holders:
        return _mk("PROFILE_BUSY_BY_LIVE_PROCESS", "存在存活进程持有该 profile",
                   pids=holders[:10], port=port)
    if locks:
        return _mk("PROFILE_LOCKED", "存在残留锁文件且无存活进程持有",
                   lock_files=locks[:10], port=port)
    return _mk("BROWSER_BOOTSTRAP_TIMEOUT", "无锁、无持有者、端口不可达（疑启动失败）", port=port)
