"""防御机制（resilience）：让爬虫在边界条件下不崩、不白跑、不静默丢数据。

设计宗旨（用户拍板 2026-09-27）：
    **诊断优先于重试**——找不到核心问题时，重试一百次也没用。因此每次失败都会先
    归到一个「根因码」，再由根因码决定是否值得重试、该执行什么动作。

模块组成：
    ``codes``       根因码与处置动作（词汇表）
    ``diagnose``    把异常/响应/页面/环境 翻译成根因码（★核心）
    ``browser_env`` profile 锁检测与安全清理、残留进程识别（优先复用，不杀进程）
    ``health``      启动前健康检查（判断现有 CDP 能否复用）
    ``backoff``     指数退避 + 全抖动；重试预算
    ``circuit``     熔断器（CLOSED / OPEN / HALF_OPEN）
    ``checkpoint``  自动断点续跑 + 失败清单（本地 JSON）
    ``guardian``    统一编排入口（决策中枢）

复用方式（96 / AUX / 未来任何爬虫）::

    from scripts.crawler.resilience import Guardian, diagnose_exception

本包**不修改** auth / collect / sync_db 任何既有文件，仅被它们按需调用。

[RESILIENCE-v1] 新增包。
"""

from __future__ import annotations

from .backoff import Backoff, RetryBudget, compute_delay
from .checkpoint import PendingItem, PendingQueue, ResumePlan, plan_resume
from .circuit import CircuitBreaker, CircuitRegistry, CircuitState
from .codes import CODES, Action, CodeInfo, Diagnosis, Layer, action_for, info, should_retry
from .diagnose import (
    diagnose_environment,
    diagnose_exception,
    diagnose_page,
    diagnose_response,
)
from .doctor import CheckItem, DoctorReport, run_doctor
from .health import (
    CdpProbe,
    SessionProbe,
    discover_healthy_cdp,
    probe_cdp,
    probe_session_from_response,
)
from .guardian import (
    DECISION_CLEAN_THEN_RELAUNCH,
    DECISION_RELAUNCH,
    DECISION_REUSE,
    STRATEGY_ALWAYS_FRESH,
    STRATEGY_REUSE_FIRST,
    Guardian,
    PreflightResult,
)

__all__ = [
    # 诊断
    "Diagnosis", "Action", "Layer", "CodeInfo", "CODES",
    "info", "should_retry", "action_for",
    "diagnose_exception", "diagnose_response", "diagnose_page", "diagnose_environment",
    # 健康检查与会话探活
    "CdpProbe", "SessionProbe", "probe_cdp", "discover_healthy_cdp",
    "probe_session_from_response",
    # 环境自检
    "run_doctor", "DoctorReport", "CheckItem",
    # 退避与熔断
    "Backoff", "RetryBudget", "compute_delay",
    "CircuitBreaker", "CircuitRegistry", "CircuitState",
    # 断点与清单
    "ResumePlan", "plan_resume", "PendingQueue", "PendingItem",
    # 编排
    "Guardian", "PreflightResult",
    "STRATEGY_REUSE_FIRST", "STRATEGY_ALWAYS_FRESH",
    "DECISION_REUSE", "DECISION_CLEAN_THEN_RELAUNCH", "DECISION_RELAUNCH",
]
