"""熔断器（Circuit Breaker）。

解决的问题：当故障是**系统性**的（浏览器死了、目标站点挂了），继续逐个源发请求只会
不断失败——表现为日志刷屏、每个源白等超时、资源空转。熔断器在连续失败达阈值后
"跳闸"，直接快速失败并进入冷却；冷却结束进入半开状态，用少量试探请求判断是否恢复。

三态（业界标准）：
    CLOSED    正常放行
    OPEN      快速失败（不发起真实请求），持续 ``recovery_timeout`` 秒
    HALF_OPEN 放行少量试探；连续成功 ``half_open_successes`` 次则回到 CLOSED，

配合用法：源级熔断（单个数据源连续失败）+ 全局熔断（整轮失败率过高则中止）。

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

import threading
import time
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """线程安全的熔断器。

    :param failure_threshold: 连续失败多少次后跳闸（默认 5）
    :param recovery_timeout: 跳闸后冷却秒数（默认 60）
    :param half_open_successes: 半开状态下连续成功多少次才完全恢复（默认 2）
    :param name: 便于日志识别的名字（如源名 / "global"）
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_successes: int = 2,
        name: str = "circuit",
        clock=time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = float(recovery_timeout)
        self.half_open_successes = max(1, int(half_open_successes))
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._opened_at = 0.0
        self._trips = 0          # 累计跳闸次数（可观测指标）
        self._blocked = 0        # 累计被快速失败的请求数

    # ── 查询 ────────────────────────────────────────────────────────────
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "trips": self._trips,
                "blocked": self._blocked,
            }

    def _maybe_half_open(self) -> None:
        """冷却期满则从 OPEN 转入 HALF_OPEN（调用方需持锁）。"""
        if self._state is CircuitState.OPEN and (
            self._clock() - self._opened_at >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._successes = 0
            self._failures = 0

    # ── 放行判定 ────────────────────────────────────────────────────────
    def allow(self) -> bool:
        """是否放行这次请求。OPEN 状态下返回 False（快速失败）。"""
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.OPEN:
                self._blocked += 1
                return False
            if self._state is CircuitState.HALF_OPEN:
                # 半开：只放行「成功阈值」数量的试探请求，其余快速失败
                if self._successes + 1 > self.half_open_successes:
                    self._blocked += 1
                    return False
            return True

    # ── 结果回执 ────────────────────────────────────────────────────────
    def record_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._successes += 1
                if self._successes >= self.half_open_successes:
                    self._state = CircuitState.CLOSED
                    self._failures = 0
                    self._successes = 0
            else:
                self._failures = 0

    def record_failure(self, *, retryable: bool = True) -> None:
        """记录一次失败。

        :param retryable: 只有「可重试型」失败才计入跳闸计数——认证 401、404 这类
            不可重试的错误不应触发熔断（熔断也无济于事）。这与 `codes.retryable`
            的语义保持严格一致。
        """
        if not retryable:
            return
        with self._lock:
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN:
                # 半开又失败 → 重新跳闸并重置冷却
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._trips += 1
                self._successes = 0
            elif self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._trips += 1

    def force_open(self, reason: str = "") -> None:
        """外部判定为系统性故障时直接跳闸（如「浏览器已死亡」）。"""
        with self._lock:
            self._state = CircuitState.OPEN
            self._opened_at = self._clock()
            self._trips += 1

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._successes = 0


class CircuitRegistry:
    """按名字管理多个熔断器（如「每个数据源一个」）。"""

    def __init__(self, **defaults) -> None:
        self._defaults = defaults
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(name)
            if breaker is None:
                breaker = CircuitBreaker(name=name, **self._defaults)
                self._breakers[name] = breaker
            return breaker

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [b.stats for b in self._breakers.values()]
