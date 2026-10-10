"""根因码与处置动作定义（防御机制的基础词汇表）。

设计原则（用户拍板 2026-09-27）：
    **诊断优先于重试**。找不到核心问题时，重试一百次也没用。因此每个失败都必须
    先归类到一个明确的「根因码」，再由根因码决定「是否值得重试、该做什么动作」。

本模块自包含（不 import auth/collect），以便被任何爬虫复用。
关键词表与 `auth/auto_crawler/page.py` 的检测判据保持语义一致，但此处独立维护，
避免 resilience 反向依赖认证模块。

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Layer(str, Enum):
    """失败发生在哪一层。用于日志归类与统计。"""

    ENV = "ENV"            # 环境：profile 锁、残留进程、磁盘
    BROWSER = "BROWSER"    # 浏览器/CDP：启动失败、target 丢失、进程死亡
    AUTH = "AUTH"          # 认证：会话失效、401、CFCA/UKey
    NETWORK = "NETWORK"    # 网络：不可达、超时、DNS
    HTTP = "HTTP"          # 服务端：502/503/504/404
    PARSE = "PARSE"        # 解析：结构变更、字段缺失
    UNKNOWN = "UNKNOWN"


class Action(str, Enum):
    """诊断结论对应的处置动作。**这是本模块的输出**，调用方据此行动。"""

    REUSE = "REUSE"                    # 复用现有浏览器，直接继续
    CLEAN_LOCK = "CLEAN_LOCK"          # 清理无主锁文件后复用（不杀进程）
    RELAUNCH = "RELAUNCH"              # 重开浏览器（同 profile、同浏览器）
    ROTATE_BROWSER = "ROTATE_BROWSER"  # 换下一个候选浏览器
    REAUTH = "REAUTH"                  # 重新走认证流程（**不是重试**）
    BACKOFF_RETRY = "BACKOFF_RETRY"    # 退避后重试
    RETRY_ONCE = "RETRY_ONCE"          # 只重试一次（时间敏感型）
    WAIT = "WAIT"                      # 等待（限流/网关冷却）
    RECORD_DLQ = "RECORD_DLQ"          # 记入待补清单，本轮不再尝试
    SKIP_NORMAL = "SKIP_NORMAL"        # 正常空数据，不算失败
    INVESTIGATE = "INVESTIGATE"        # 需人工介入（不空转）
    ABORT = "ABORT"                    # 放弃本轮


#: 根因码 → 元数据。code 是稳定的机器可读标识，落日志/报告，便于统计与复盘。
@dataclass(frozen=True)
class CodeInfo:
    code: str
    layer: Layer
    retryable: bool
    action: Action
    summary: str


_CODE_LIST: tuple[CodeInfo, ...] = (
    # ── 环境层 ──────────────────────────────────────────────────────────
    CodeInfo(
        "PROFILE_LOCKED", Layer.ENV, True, Action.CLEAN_LOCK,
        "用户数据目录被占用/残留锁文件（Chrome/Edge 单实例锁）；清理无主锁文件后复用",
    ),
    CodeInfo(
        "PROFILE_BUSY_BY_LIVE_PROCESS", Layer.ENV, True, Action.RELAUNCH,
        "确有存活进程持有该 profile；不杀进程，改换浏览器或等其退出",
    ),
    # ── 浏览器层 ────────────────────────────────────────────────────────
    CodeInfo(
        "BROWSER_BOOTSTRAP_TIMEOUT", Layer.BROWSER, True, Action.RETRY_ONCE,
        "启动后限时内未出现可控 PMOS 页面；延长超时后只重试一次",
    ),
    CodeInfo(
        "CDP_PORT_DEAD", Layer.BROWSER, True, Action.RELAUNCH,
        "CDP 调试端口不可达（连接被拒）→ 浏览器进程已死亡",
    ),
    CodeInfo(
        "CDP_TARGET_LOST", Layer.BROWSER, True, Action.BACKOFF_RETRY,
        "CDP target 被导航/重定向销毁（Inspected target navigated or closed）",
    ),
    CodeInfo(
        "BROWSER_LAUNCHER_EXITED_ZERO", Layer.BROWSER, True, Action.CLEAN_LOCK,
        "启动器以 code=0 立即退出（新进程发现同 profile 已有实例）",
    ),
    # ── 认证层 ──────────────────────────────────────────────────────────
    CodeInfo(
        "SESSION_EXPIRED", Layer.AUTH, False, Action.REAUTH,
        "页面/接口提示会话已失效；须重新认证，重试无效",
    ),
    CodeInfo(
        "AUTH_REJECTED_401", Layer.AUTH, False, Action.REAUTH,
        "服务端返回 401/403；须重新认证，重试无效",
    ),
    CodeInfo(
        "CFCA_PIN_STUCK", Layer.AUTH, False, Action.INVESTIGATE,
        "卡在 CERTIFICATE（CFCA/UKey 原生弹窗未出现）；多为本地网络权限或插件环境问题",
    ),
    CodeInfo(
        "LOGIN_TIMEOUT", Layer.AUTH, False, Action.ROTATE_BROWSER,
        "登录流程超时（滑块反复失败/证书未过）；换浏览器重试整条登录链路",
    ),
    # ── 网络层 ──────────────────────────────────────────────────────────
    CodeInfo(
        "NETWORK_UNREACHABLE", Layer.NETWORK, True, Action.BACKOFF_RETRY,
        "网络不可达（浏览器错误页/DNS/连接超时）",
    ),
    # ── HTTP 层 ─────────────────────────────────────────────────────────
    CodeInfo(
        "HTTP_GATEWAY_5XX", Layer.HTTP, True, Action.BACKOFF_RETRY,
        "502/503/504 等服务端临时故障（含 nginx 超时）",
    ),
    CodeInfo(
        "HTTP_NOT_FOUND_404", Layer.HTTP, False, Action.RECORD_DLQ,
        "接口/资源不存在；不可重试，记入待补清单待人工核实",
    ),
    CodeInfo(
        "HTTP_RATE_LIMITED_429", Layer.HTTP, True, Action.WAIT,
        "被限流；应按 Retry-After 等待后重试",
    ),
    CodeInfo(
        "NO_DATA", Layer.HTTP, False, Action.SKIP_NORMAL,
        "业务正常空数据（该日期本无数据），不算失败",
    ),
    # ── 解析层 ──────────────────────────────────────────────────────────
    CodeInfo(
        "PARSE_ERROR", Layer.PARSE, False, Action.RECORD_DLQ,
        "响应无法解析/关键字段缺失；疑似页面结构变更，须人工介入",
    ),
    # ── 兜底 ────────────────────────────────────────────────────────────
    CodeInfo(
        "UNKNOWN", Layer.UNKNOWN, False, Action.INVESTIGATE,
        "无法归类；保守处理（不盲目重试），完整证据落盘",
    ),
)

#: 机器可读码 → CodeInfo
CODES: dict[str, CodeInfo] = {info.code: info for info in _CODE_LIST}


def info(code: str) -> CodeInfo:
    """按码取元数据；未知码回退到 UNKNOWN（绝不抛异常，诊断路径必须稳）。"""
    return CODES.get(code, CODES["UNKNOWN"])


def should_retry(code: str) -> bool:
    """该根因是否值得重试。调用方在重试前必须先问这里。"""
    return info(code).retryable


def action_for(code: str) -> Action:
    """该根因对应的处置动作。"""
    return info(code).action


@dataclass(frozen=True)
class Diagnosis:
    """一次失败的结构化诊断结论（防御机制的核心数据流）。"""

    code: str
    reason: str = ""
    evidence: dict = field(default_factory=dict)
    # 以下三项默认从 codes 推导，必要时可由诊断器覆盖。
    layer: str = ""
    retryable: bool = False
    action: str = ""

    def __post_init__(self) -> None:  # dataclass(frozen=True) 用 object.__setattr__
        meta = info(self.code)
        if not self.layer:
            object.__setattr__(self, "layer", meta.layer.value)
        if not self.action:
            object.__setattr__(self, "action", meta.action.value)
        # retryable 显式传 True 才覆盖；默认取元数据（False 是安全默认）。
        if self.retryable is False:
            object.__setattr__(self, "retryable", meta.retryable)

    def as_log_fields(self) -> dict:
        """给结构化日志用的扁平字段。"""
        return {
            "root_cause": self.code,
            "layer": self.layer,
            "retryable": self.retryable,
            "action": self.action,
            "reason": self.reason,
        }
