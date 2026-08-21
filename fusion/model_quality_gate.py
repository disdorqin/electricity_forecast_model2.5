"""Quality gates for learned fusion weights."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


@dataclass(frozen=True)
class WeightGateResult:
    active_weights: dict[str, float]
    pruned_models: tuple[str, ...]
    threshold: float
    fallback_used: bool
    fallback_model: str | None


def gate_weights(
    weights: Mapping[str, float],
    *,
    threshold: float = 0.05,
    min_active_models: int = 1,
) -> WeightGateResult:
    """Prune low-weight models independently for one task/period.

    If the threshold would remove every model, the highest finite-weight
    candidate is retained as a safety fallback. Returned values are raw
    active weights; the fusion layer renormalizes them after availability
    checks.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"weight gate threshold must be in [0, 1), got {threshold}")
    if min_active_models < 1:
        raise ValueError("min_active_models must be >= 1")

    finite = {
        str(model): float(weight)
        for model, weight in weights.items()
        if isfinite(float(weight))
    }
    if not finite:
        raise ValueError("weight gate received no finite model weights")

    # threshold=0 is the explicit opt-out switch. Preserve historical
    # weighting behavior, including negative BGEW values.
    if threshold == 0.0:
        return WeightGateResult(dict(finite), (), threshold, False, None)

    ranked = sorted(finite, key=lambda model: (-finite[model], model))
    active = [model for model in ranked if finite[model] >= threshold]
    fallback_used = False
    fallback_model = None
    if len(active) < min_active_models:
        active = ranked[:min_active_models]
        fallback_used = True
        fallback_model = active[0]

    active_weights = {model: finite[model] for model in active}
    if sum(max(value, 0.0) for value in active_weights.values()) <= 0.0:
        fallback_model = fallback_model or active[0]
        active_weights = {fallback_model: 1.0}
        active = [fallback_model]
        fallback_used = True

    pruned = tuple(model for model in finite if model not in active)
    return WeightGateResult(
        active_weights=active_weights,
        pruned_models=pruned,
        threshold=threshold,
        fallback_used=fallback_used,
        fallback_model=fallback_model,
    )
