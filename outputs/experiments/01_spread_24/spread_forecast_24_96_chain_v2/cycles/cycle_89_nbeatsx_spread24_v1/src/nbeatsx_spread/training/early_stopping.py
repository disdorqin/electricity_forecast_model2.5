from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 8
    min_delta: float = 1e-4
    best: float = float("inf")
    checks_without_improvement: int = 0

    def update(self, value: float) -> bool:
        threshold = self.best * (1.0 - self.min_delta) if self.best < float("inf") else float("inf")
        if value < threshold:
            self.best = value
            self.checks_without_improvement = 0
            return True
        self.checks_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.checks_without_improvement >= self.patience
