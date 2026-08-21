"""Regression test: hourly weight learning must reject 96-point rows."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fusion.model_pool import DAYAHEAD_MODELS
from pipelines.ledger_weight import select_complete_training_days
from utils.resolution import HOURLY


def build_pred(day: str, n_rows: int) -> pd.DataFrame:
    rows = []
    for model in DAYAHEAD_MODELS:
        for i in range(n_rows):
            hour = i % 24 + 1
            rows.append({
                "task": "dayahead", "model_name": model, "target_day": day,
                "business_day": day, "business_period": i + 1,
                "hour_business": hour, "period": "1_8" if hour <= 8 else "9_16" if hour <= 16 else "17_24",
                "ds": f"{day} {hour:02d}:00:00", "y_pred": 100.0,
            })
    return pd.DataFrame(rows)


def build_actual(day: str, n_rows: int) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        hour = i % 24 + 1
        rows.append({
            "task": "dayahead", "target_day": day, "business_day": day,
            "business_period": i + 1, "hour_business": hour,
            "period": "1_8" if hour <= 8 else "9_16" if hour <= 16 else "17_24",
            "ds": f"{day} {hour:02d}:00:00", "y_true": 100.0,
        })
    return pd.DataFrame(rows)


with tempfile.TemporaryDirectory(prefix="mixed_resolution_weight_guard_") as tmp:
    root = Path(tmp)
    for task in ("dayahead",):
        (root / task / "prediction").mkdir(parents=True)
        (root / task / "actual").mkdir(parents=True)
        # D-1 is a 96-row mixed-resolution day; D-2 is a valid hourly day.
        pd.concat([build_pred("2026-07-02", 96), build_pred("2026-07-01", 24)]).to_parquet(
            root / task / "prediction/prediction_ledger.parquet", index=False
        )
        pd.concat([build_actual("2026-07-02", 96), build_actual("2026-07-01", 24)]).to_parquet(
            root / task / "actual/actual_ledger.parquet", index=False
        )
    result = select_complete_training_days(
        "dayahead", "2026-07-03", root, list(DAYAHEAD_MODELS),
        required_days=1, max_lookback_days=3, resolution=HOURLY,
    )
    assert result["status"] == "PASS"
    assert result["selected_days"] == ["2026-07-01"]
    skipped = {item["day"]: item["reason"] for item in result["skipped_days"]}
    assert skipped["2026-07-02"] == "prediction incomplete"
    print("MIXED_RESOLUTION_WEIGHT_GUARD: PASS")
