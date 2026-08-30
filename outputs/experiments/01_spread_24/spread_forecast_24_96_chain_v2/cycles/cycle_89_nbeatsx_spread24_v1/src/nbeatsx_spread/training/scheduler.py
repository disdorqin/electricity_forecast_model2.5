from __future__ import annotations

from collections.abc import Sequence


def scheduled_learning_rate(
    initial: float,
    step: int,
    schedule_total_steps: int = 1200,
    gamma: float = 0.5,
    decays: int = 3,
    nominal_decay_steps: Sequence[int] | None = None,
) -> float:
    """Return a non-compressed step-halving learning rate."""
    if initial <= 0 or step < 0 or schedule_total_steps <= 0:
        raise ValueError("initial, step and schedule_total_steps must be valid")
    milestones = tuple(nominal_decay_steps or (
        max(1, schedule_total_steps // decays) * i for i in range(1, decays + 1)
    ))
    if tuple(sorted(milestones)) != milestones:
        raise ValueError("nominal decay steps must be sorted")
    return float(initial * gamma ** sum(step >= point for point in milestones))
