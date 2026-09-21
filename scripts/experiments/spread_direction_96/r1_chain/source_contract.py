from __future__ import annotations

import json
from pathlib import Path


def validate_deployable_prediction_source(prediction_path: Path) -> dict:
    """Reject paper/oracle prediction sources from deployable 96→24 experiments."""
    manifest_path = prediction_path.parent / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing source manifest for prediction source: {prediction_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "experiment": "r1_inspired_96_ahead",
        "deployable_source": True,
        "forecast_origin": "D-1 14:00",
        "target_day_actual_features": False,
        "target_day_da_input": False,
        "dminus1_post14_realized_input": False,
        "production_chain_touched": False,
    }
    bad = {}
    for key, expected in required.items():
        actual = manifest.get(key)
        if actual != expected:
            bad[key] = {"expected": expected, "actual": actual}
    if bad:
        raise RuntimeError(
            f"non-deployable or unaudited 96 prediction source rejected: {prediction_path}; mismatches={bad}"
        )
    return manifest
