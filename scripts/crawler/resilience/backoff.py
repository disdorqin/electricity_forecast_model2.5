"""指数退避 + 全抖动（Exponential Backoff with Full Jitter）。

为什么要抖动：多个 worker/多个源同时失败时，固定间隔会让它们在**同一时刻集体重试**
（惊群效应），既浪费资源又更容易触发风控。全抖动把等待时间随机化到 [0, 指数值]，
从根本上打散重试尖峰。

替代原先写死的 `time.sleep(6)`。

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

import random
import threading
import time

DEFAULT_BASE = 1.0
DEFAULT_CAP = 60.0


def compute_delay(attempt: int, *, base: float = DEFAULT_BASE, cap: float = DEFAULT_CAP,
                  jitter: bool = True, rng: random.Random | None = None) -> float:
    """计算第 `attempt` 次重试（从 0 开始）应等待的秒数。

    - 指数部分：``base * 2**attempt``，上限 ``cap``
    - 全抖动：在 ``[0, 指数值]`` 内均匀随机（业界推荐做法）
    """
    attempt = max(0, int(attempt))
    exponential = min(float(cap), float(base) * (2 ** attempt))
    if not jitter:
        return exponential
    rand = rng or random
    return rand.uniform(0.0, exponential)


class Backoff:
    """有状态退避器：为一条独立的失败流维护尝试计数。

    典型用法::

        bo = Backoff(base=2.0, cap=60.0, max_attempts=5)
        while bo.allow():
            if ok: bo.reset(); break
            time.sleep(bo.next_delay())
    """

    def __init__(self, *, base: float = DEFAULT_BASE, cap: float = DEFAULT_CAP,
                 max_attempts: int = 5, jitter: bool = True) -> None:
        self.base = float(base)
        self.cap = float(cap)
        self.max_attempts = max(1, int(max_attempts))
        self.jitter = bool(jitter)
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    def allow(self) -> bool:
        """是否还允许再试一次。"""
        return self._attempt < self.max_attempts

    def next_delay(self) -> float:
        """取本次应等待的秒数，并推进计数。"""
        delay = compute_delay(self._attempt, base=self.base, cap=self.cap, jitter=self.jitter)
        self._attempt += 1
        return delay

    def sleep(self, sleeper=time.sleep) -> float:
        """按策略睡眠并返回实际睡眠秒数（便于测试注入）。"""
        delay = self.next_delay()
        sleeper(delay)
        return delay

    def reset(self) -> None:
        self._attempt = 0


class RetryBudget:
    """重试预算：限制一段区间内的总重试次数，防止系统级"重试放大"。

    业界经验：高并发系统若不做预算控制，故障时重试流量可能翻数倍。
    这里按「每 N 次正常操作允许若干次重试」的简单配额实现。
    """

    def __init__(self, *, per_window: int = 100, allowance: int = 20,
                 window_sec: float = 60.0, clock=time.monotonic) -> None:
        self.per_window = max(1, int(per_window))
        self.allowance = max(0, int(allowance))
        self.window_sec = float(window_sec)
        self._clock = clock
        self._lock = threading.Lock()
        self._ops = 0
        self._used = 0
        self._window_start = clock()

    def _roll(self) -> None:
        now = self._clock()
        if now - self._window_start >= self.window_sec:
            self._window_start = now
            self._ops = 0
            self._used = 0

    def record_operation(self) -> None:
        with self._lock:
            self._roll()
            self._ops += 1

    def try_consume(self) -> bool:
        """申请一次重试额度；用尽则返回 False（调用方应放弃重试）。"""
        with self._lock:
            self._roll()
            cap = int(self.allowance + self._ops * 0.2)  # 允许 20% 的弹性
            if self._used >= cap:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        return self._used
