"""Canonical production model pools.

This module is the single source of truth for the models that may enter the
ledger production pipeline.  Historical/experimental adapters may still
exist, but they must not silently become production candidates.
"""

from __future__ import annotations

from typing import Iterable


DAYAHEAD_MODELS: tuple[str, ...] = ("lightgbm", "timesfm", "timemixer")

# LightGBM realtime is intentionally disabled. TimesFM realtime is a current
# production candidate and must stay aligned with the ledger contracts.
REALTIME_MODELS: tuple[str, ...] = ("timesfm", "sgdfnet", "timemixer", "rt916")
DISABLED_REALTIME_MODELS: frozenset[str] = frozenset({"lightgbm"})


def models_for_task(task: str) -> tuple[str, ...]:
    """Return the canonical production candidate pool for a task."""
    if task == "dayahead":
        return DAYAHEAD_MODELS
    if task == "realtime":
        return REALTIME_MODELS
    raise ValueError(f"Unknown task: {task!r}")


def validate_pool(task: str, models: Iterable[str]) -> None:
    """Fail fast if a caller tries to add a disabled/unknown candidate."""
    allowed = set(models_for_task(task))
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(
            f"Models {unknown} are not in the production {task} candidate pool; "
            f"allowed={sorted(allowed)}"
        )
