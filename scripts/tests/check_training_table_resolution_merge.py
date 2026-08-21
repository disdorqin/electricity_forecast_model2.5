"""Regression test: training-table joins must not explode for hourly ledgers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.prediction_ledger import build_ledger_training_table


def make_rows(day: str, slots: int, *, prediction: bool) -> pd.DataFrame:
    rows = []
    for model in ("lightgbm", "timesfm", "timemixer") if prediction else (None,):
        for slot in range(1, slots + 1):
            row = {
                "task": "dayahead",
                "target_day": day,
                "business_day": day,
                "hour_business": slot,
                "business_period": slot if slots == 96 else None,
                "period": "1_8" if slot <= slots // 3 else "9_16" if slot <= 2 * slots // 3 else "17_24",
                "ds": f"{day} {slot:02d}:00:00",
            }
            if prediction:
                row.update({"model_name": model, "y_pred": float(slot)})
            else:
                row["y_true"] = float(slot)
            rows.append(row)
    return pd.DataFrame(rows)


for slots, expected_models in ((24, 3), (96, 3)):
    pred = make_rows("2026-08-13", slots, prediction=True)
    act = make_rows("2026-08-13", slots, prediction=False)
    table = build_ledger_training_table(pred, act, "2026-08-14", window_days=1)
    assert len(table) == slots * expected_models, (slots, len(table))
    assert table["y_true"].notna().all()

print("TRAINING_TABLE_RESOLUTION_MERGE: PASS")
