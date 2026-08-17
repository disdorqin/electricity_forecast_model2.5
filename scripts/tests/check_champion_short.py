#!/usr/bin/env python
"""Regression checks for the explicit champion_short learner branch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from fusion.learners.champion_short import ChampionShortConfig, fit_champion_short
from utils.resolution import HOURLY, QUARTER


def _table(resolution, task: str, days: list[str], models: list[str], seed: int, target: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for day in days:
        for slot in range(1, resolution.slots_per_day + 1):
            ds = resolution.timestamp_from_business(day, slot)
            truth = 50.0 + 5.0 * np.sin(slot / resolution.slots_per_day * 2.0 * np.pi)
            for model_index, model in enumerate(models):
                rows.append({
                    "task": task,
                    "model_name": model,
                    "target_day": day,
                    "business_day": day,
                    "ds": ds,
                    "hour_business": slot if resolution is HOURLY else (slot - 1) // 4 + 1,
                    "business_period": slot,
                    "period": resolution.infer_period(slot),
                    "y_pred": truth + rng.normal(0.0, 1.0 + model_index * 0.2),
                    "y_true": np.nan if target else truth,
                })
    return pd.DataFrame(rows)


def _check(resolution, task: str, models: list[str]) -> None:
    days = [f"2026-01-{day:02d}" for day in range(1, 15)]
    history = _table(resolution, task, days, models, seed=42)
    # build_ledger_training_table omits business_period; verify the 96-point
    # learner falls back to ds rather than collapsing to hour_business.
    history = history.drop(columns=["business_period"])
    target = _table(resolution, task, ["2026-01-15"], models, seed=43, target=True)
    weights, report, trace = fit_champion_short(
        history,
        target,
        task=task,
        expected_models=models,
        resolution=resolution,
        config=ChampionShortConfig(),
    )
    assert len(report) == len(resolution.period_names)
    assert len(trace) == len(models) * len(resolution.period_names)
    for values in weights.values():
        assert np.isclose(sum(values.values()), 1.0)

    incomplete = target[target["model_name"] != models[-1]].copy()
    try:
        fit_champion_short(
            history,
            incomplete,
            task=task,
            expected_models=models,
            resolution=resolution,
            config=ChampionShortConfig(),
        )
    except ValueError as exc:
        assert "missing_models" in str(exc)
    else:
        raise AssertionError("incomplete target predictions were not rejected")


def main() -> None:
    _check(HOURLY, "dayahead", ["lightgbm", "timesfm", "timemixer"])
    _check(QUARTER, "realtime", ["timesfm", "sgdfnet", "timemixer", "rt916"])
    print("CHAMPION_SHORT_RESOLUTION_TEST: PASS")


if __name__ == "__main__":
    main()
