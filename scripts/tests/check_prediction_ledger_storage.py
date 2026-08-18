"""Regression test for fragmented range ledger storage and compaction."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.prediction_ledger import (
    append_predictions_to_ledger,
    compact_ledger,
    load_prediction_ledger,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efm3-ledger-") as td:
        root = Path(td)
        rows = []
        for period in range(1, 97):
            rows.append({
                "task": "dayahead",
                "model_name": "lightgbm",
                "forecast_date": "2026-01-01",
                "target_day": "2026-01-01",
                "business_day": "2026-01-01",
                "business_period": period,
                "hour_business": (period - 1) // 4 + 1,
                "period": "1_32" if period <= 32 else "33_64" if period <= 64 else "65_96",
                "ds": f"2026-01-01 {period:02d}:00:00",
                "y_pred": float(period),
            })
        frame = pd.DataFrame(rows)
        result = append_predictions_to_ledger(
            frame, root, "dayahead", fragmented=True
        )
        assert result["storage"] == "fragmented"
        loaded = load_prediction_ledger(root, "dayahead", ["2026-01-01"])
        assert len(loaded) == 96
        assert not (root / "dayahead" / "prediction" / "prediction_ledger.parquet").exists()
        compacted = compact_ledger(root, "dayahead", "prediction")
        assert compacted["status"] == "compacted"
        assert len(load_prediction_ledger(root, "dayahead", ["2026-01-01"])) == 96
        assert not list((root / "dayahead" / "prediction" / "parts").glob("*.parquet"))
    print("check_prediction_ledger_storage: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
