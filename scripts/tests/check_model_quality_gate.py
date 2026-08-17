"""Synthetic checks for learned-weight model pruning."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fusion.model_quality_gate import gate_weights
from fusion.model_pool import models_for_task
from runners.adapters.lightgbm_v1 import LightGBMV1Adapter


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    if not condition:
        raise SystemExit(1)


def main() -> int:
    result = gate_weights(
        {"strong": 0.80, "medium": 0.18, "weak": 0.02},
        threshold=0.05,
    )
    check("low-weight model is pruned", result.pruned_models == ("weak",))
    check("strong and medium remain", set(result.active_weights) == {"strong", "medium"})
    check("no fallback when a candidate survives", not result.fallback_used)

    fallback = gate_weights(
        {"a": 0.02, "b": 0.01},
        threshold=0.05,
    )
    check("highest-weight candidate is retained", set(fallback.active_weights) == {"a"})
    check("all-pruned case is explicitly marked fallback", fallback.fallback_used)

    disabled = gate_weights(
        {"a": 0.6, "b": 0.4},
        threshold=0.0,
    )
    check("threshold zero disables pruning", not disabled.pruned_models)
    rt_pool = models_for_task("realtime")
    check("TimesFM remains in realtime pool", "timesfm" in rt_pool)
    check("LightGBM is absent from realtime pool", "lightgbm" not in rt_pool)
    try:
        LightGBMV1Adapter().predict("2026-01-01", target="realtime")
    except ValueError as exc:
        check("LightGBMV1Adapter realtime path is blocked", "realtime" in str(exc).lower())
    else:
        raise AssertionError("LightGBMV1Adapter realtime path was not blocked")
    print("RESULT: 8/8 scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
