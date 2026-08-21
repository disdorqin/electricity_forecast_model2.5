"""Regression check: 96-point slot metadata must not become a model weight."""

from __future__ import annotations

import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.weights import fit_weights_from_long_table
from utils.resolution import QUARTER


def main() -> int:
    rows = []
    for day in pd.date_range("2026-08-01", periods=2, freq="D"):
        for slot in range(1, 5):
            ds = day + pd.Timedelta(minutes=15 * slot)
            truth = 100.0 + slot
            for model, offset in (("lightgbm", 0.0), ("timesfm", 3.0)):
                rows.append({
                    "task": "dayahead",
                    "model_name": model,
                    "target_day": day.strftime("%Y-%m-%d"),
                    "ds": ds,
                    "period": "1_32",
                    "hour_business": 1,
                    "business_period": slot,
                    "y_true": truth,
                    "y_pred": truth + offset,
                })
    weights, report = fit_weights_from_long_table(pd.DataFrame(rows), resolution=QUARTER)
    models = set(weights["model_name"].tolist())
    assert models == {"lightgbm", "timesfm"}, models
    assert "business_period" not in models
    assert len(report) == 1
    assert abs(float(weights.groupby(["task", "period"])["weight"].sum().iloc[0]) - 1.0) < 1e-6
    print("SMAPE_WEIGHT_RESOLUTION_METADATA_TEST: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
