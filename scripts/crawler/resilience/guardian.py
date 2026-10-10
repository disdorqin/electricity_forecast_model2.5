"""统一编排入口（决策中枢）。

定位：`resilience` 各能力件（诊断 / 退避 / 熔断 / 环境 / 健康 / 清单）在此汇合，
对外只暴露「开跑前先问什么、失败后该做什么」，**不负责具体执行浏览器动作**
（重开/换浏览器/重认证需由入口层用 auth 模块执行）。这样既集中了决策逻辑，
又保持了与 auth 的解耦，任何爬虫都能复用。

用法（入口层）::

    guardian = Guardian(profile_dir=..., port_candidates=[9222, 9223], reporter=r)
    plan = guardian.preflight()
    if plan.decision == "REUSE": 复用 plan.reuse_probe.port
    elif plan.decision == "CLEAN_THEN_RELAUNCH": 锁已清，去重开浏览器
    else: 直接重开
    ...
    try: ...
    except Exception as exc:
        d = diagnose_exception(exc)
        action = guardian.on_failure(d, key="2024-08-16:dcst_tmp_load")

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import browser_env, health
from .backoff import Backoff
from .checkpoint import PendingQueue
from .circuit import CircuitBreaker
from .codes import Action, Diagnosis, info
from .diagnose import diagnose_exception

#: 策略：优先复用现有浏览器（默认）；若复用问题过多可切到每次全新进程
STRATEGY_REUSE_FIRST = "REUSE_FIRST"
STRATEGY_ALWAYS_FRESH = "ALWAYS_FRESH"

#: preflight 决策结果
DECISION_REUSE = "REUSE"
DECISION_CLEAN_THEN_RELAUNCH = "CLEAN_THEN_RELAUNCH"
DECISION_RELAUNCH = "RELAUNCH"


@dataclass
class PreflightResult:
    strategy: str
    decision: str
    diagnosis: Diagnosis
    profile_dir: str = ""
    lock_files: list[str] = field(default_factory=list)
    live_holders: list[int] = field(default_factory=list)
    lock_cleanup: dict = field(default_factory=dict)
    reuse_port: int | None = None
    reuse_browser: str = ""
    terminated: list[dict] = field(default_factory=list)  # 回收的毒瘤孤儿浏览器动作

    def summary(self) -> dict:
        return {
            "strategy": self.strategy, "decision": self.decision,
            "root_cause": self.diagnosis.code, "action": self.diagnosis.action,
            "profile_dir": self.profile_dir, "lock_files": self.lock_files,
            "live_holders": self.live_holders, "lock_cleanup": self.lock_cleanup,
            "reuse_port": self.reuse_port, "reuse_browser": self.reuse_browser,
            "terminated": self.terminated,
        }


class Guardian:
    """一次运行内的防御编排器（应在入口层创建一个实例并复用）。"""

    def __init__(
        self,
        *,
        profile_dir: str | None = None,
        port_candidates: list[int] | None = None,
        strategy: str = STRATEGY_REUSE_FIRST,
        reporter=None,
        auto_clean_locks: bool = True,
        pending_queue: PendingQueue | None = None,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 60.0,
        report_interval: int = 3,
    ) -> None:
        self.profile_dir = str(profile_dir) if profile_dir else ""
        self.port_candidates = [int(p) for p in (port_candidates or [])]
        self.strategy = strategy
        self.reporter = reporter
        self.auto_clean_locks = bool(auto_clean_locks)
        self.pending = pending_queue if pending_queue is not None else PendingQueue()
        self.report_interval = max(1, int(report_interval))

        self._source_circuits: dict[str, CircuitBreaker] = {}
        self.global_circuit = CircuitBreaker(
            failure_threshold=max(2, circuit_failure_threshold),
            recovery_timeout=circuit_recovery_timeout,
            name="global",
        )
        self.backoff = Backoff(base=2.0, cap=60.0, max_attempts=5)
        self._consecutive_failures = 0
        self._failure_codes: dict[str, int] = {}
        self._reuse_port: int | None = None

    # ── 开跑前 ──────────────────────────────────────────────────────────
    def preflight(self) -> PreflightResult:
        """环境预检 + 复用判定。这是「优先复用」策略的落点。"""
        profile = browser_env.inspect_profile(self.profile_dir, port=None) if self.profile_dir \
            else browser_env.ProfileLockReport(profile_dir="")

        reuse_probe = None
        terminated: list[dict] = []
        if self.strategy == STRATEGY_REUSE_FIRST and self.port_candidates:
            reuse_probe = health.discover_healthy_cdp(self.port_candidates)
            if reuse_probe is None:
                # [2026-09-28 关键修复] 无健康可复用 → 先回收「毒瘤孤儿」：
                # 调试端口活着、但活动页是 favicon/404 等死页面的残留浏览器。
                # 否则：① 认证层会误判「可复用」而死等 600s；② 它占着 profile
                # 单实例锁，新浏览器起不来(bootstrap_profile_maybe_locked)。
                for probe in health.find_poisoned_cdp(self.port_candidates):
                    act = browser_env.terminate_debug_browser(
                        probe.port, profile_dir=self.profile_dir,
                    )
                    terminated.append({
                        "port": probe.port,
                        "first_url": probe.first_url,
                        **act,
                    })
                if terminated and self.profile_dir:
                    # 回收后重探锁：优雅关闭一般已自清，强杀可能残留 Singleton* → 交下面清理
                    profile = browser_env.inspect_profile(self.profile_dir, port=None)

        cleanup: dict = {}
        if reuse_probe is not None:
            decision = DECISION_REUSE
            diagnosis = Diagnosis(code="UNKNOWN", reason="发现健康可复用的 CDP 会话", action=Action.REUSE.value)
        else:
            # 不可复用：若只有残留锁（无存活进程），先清锁再重开
            if self.auto_clean_locks and profile.has_orphan_locks:
                cleanup = browser_env.clean_orphan_locks(self.profile_dir, dry_run=False)
                decision = DECISION_CLEAN_THEN_RELAUNCH
                diagnosis = profile.diagnose()
            else:
                decision = DECISION_RELAUNCH
                diagnosis = profile.diagnose()

        self._reuse_port = reuse_probe.port if reuse_probe else None
        result = PreflightResult(
            strategy=self.strategy, decision=decision, diagnosis=diagnosis,
            profile_dir=self.profile_dir, lock_files=profile.lock_files,
            live_holders=profile.live_holders, lock_cleanup=cleanup,
            reuse_port=reuse_probe.port if reuse_probe else None,
            reuse_browser=reuse_probe.browser if reuse_probe else "",
            terminated=terminated,
        )
        self._report("preflight", result.summary())
        return result

    # ── 失败后 ──────────────────────────────────────────────────────────
    def on_failure(self, diagnosis: Diagnosis, *, key: str = "", kind: str = "item") -> str:
        """登记一次失败，返回建议动作（来自根因码元数据）。"""
        code = diagnosis.code
        self._failure_codes[code] = self._failure_codes.get(code, 0) + 1
        self._consecutive_failures += 1

        retryable = diagnosis.retryable

        # 熔断：只有「可重试型」失败才计入（401/404 这类熔断也救不了）
        self.global_circuit.record_failure(retryable=retryable)
        if code in ("CDP_PORT_DEAD", "PROFILE_LOCKED", "BROWSER_BOOTSTRAP_TIMEOUT"):
            self.global_circuit.force_open(reason=code)

        # 不可重试 → 记入待补清单，避免静默丢数据
        if not retryable and key and Action(diagnosis.action) in (Action.RECORD_DLQ, Action.INVESTIGATE):
            self.pending.add(kind=kind, key=key, code=code, reason=diagnosis.reason)

        self._report("failure", {**diagnosis.as_log_fields(), "key": key, "consecutive": self._consecutive_failures})
        return diagnosis.action

    def on_success(self) -> None:
        self._consecutive_failures = 0
        self.global_circuit.record_success()

    def note_success_for(self, name: str) -> None:
        self._circuit_for(name).record_success()

    # ── 查询 ────────────────────────────────────────────────────────────
    def _circuit_for(self, name: str, retryable: bool = True) -> CircuitBreaker:
        breaker = self._source_circuits.get(name)
        if breaker is None:
            breaker = CircuitBreaker(name=name, failure_threshold=5, recovery_timeout=60.0)
            self._source_circuits[name] = breaker
        return breaker

    def allow_source(self, name: str) -> bool:
        """该数据源当前是否放行（源级熔断）。"""
        return self._circuit_for(name).allow()

    def should_stop(self) -> bool:
        """整轮是否应中止（全局熔断已跳闸）。"""
        return not self.global_circuit.allow()

    def next_backoff(self) -> float:
        """取下一次重试的退避秒数（指数 + 全抖动）。"""
        return self.backoff.next_delay()

    def reset_backoff(self) -> None:
        self.backoff.reset()

    @property
    def reuse_port(self) -> int | None:
        return self._reuse_port

    def summary(self) -> dict:
        return {
            "failures_by_root_cause": dict(self._failure_codes),
            "consecutive_failures": self._consecutive_failures,
            "global_circuit": self.global_circuit.stats,
            "source_circuits": [c.stats for c in self._source_circuits.values()],
            "pending": self.pending.summary(),
        }

    # ── 内部 ────────────────────────────────────────────────────────────
    def _report(self, stage: str, payload: dict) -> None:
        if self.reporter is None:
            return
        try:
            stage_fn = getattr(self.reporter, "stage", None)
            if callable(stage_fn):
                stage_fn(f"resilience:{stage}", "INFO", **{k: v for k, v in payload.items()})
        except Exception:  # noqa: BLE001 —— 上报失败绝不能影响主流程
            pass


def diagnose(exc: BaseException, *, context: str = "") -> Diagnosis:
    """便捷转发（入口层少 import 一个模块）。"""
    return diagnose_exception(exc, context=context)


def action_hint(code: str) -> str:
    """按根因码取建议动作（供日志/报表展示）。"""
    return info(code).action.value
