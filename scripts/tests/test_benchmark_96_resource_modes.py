from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
from scripts.server.benchmark_96_resource_modes import (
    _downstream_command,
    _prediction_command,
    compare_prediction_outputs,
    promotion_gate,
)


EXPECTED = {
    "dayahead": list(DAYAHEAD_MODELS),
    "realtime": list(REALTIME_MODELS),
}


def _write_predictions(root: Path, target_date: str, delta: float = 0.0) -> None:
    for task, models in EXPECTED.items():
        pred_dir = root / "runs" / target_date / task / "prediction"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for model_index, model in enumerate(models):
            frame = pd.DataFrame(
                {
                    "business_period": np.arange(1, 97),
                    "y_pred": np.arange(96, dtype=float) + model_index + delta,
                }
            )
            frame.to_csv(pred_dir / f"{model}_predictions.csv", index=False)


def _mode(wall: float, elapsed: float) -> dict:
    model_elapsed = {
        f"{task}/{model}": elapsed
        for task, models in EXPECTED.items()
        for model in models
    }
    return {
        "monitor": {
            "wall_seconds": wall,
            "return_code": 0,
            "oom_detected": False,
            "gpu_metrics_available": True,
            "sample_count": 10,
        },
        "model_elapsed_seconds": model_elapsed,
    }


def test_prediction_equivalence_detects_value_change(tmp_path: Path):
    date = "2026-08-16"
    legacy = tmp_path / "legacy"
    split = tmp_path / "split"
    _write_predictions(legacy, date)
    _write_predictions(split, date)

    same = compare_prediction_outputs(
        legacy, split, date, atol=1e-5, rtol=1e-6
    )
    assert same["pass"]

    changed = split / "runs" / date / "realtime" / "prediction" / "timesfm_predictions.csv"
    frame = pd.read_csv(changed)
    frame.loc[0, "y_pred"] += 0.01
    frame.to_csv(changed, index=False)

    different = compare_prediction_outputs(
        legacy, split, date, atol=1e-5, rtol=1e-6
    )
    assert not different["pass"]
    assert not different["details"]["realtime/timesfm"]["pass"]


def test_promotion_gate_never_promotes_without_downstream_equivalence():
    legacy = _mode(100.0, 10.0)
    split = _mode(80.0, 9.0)

    gate = promotion_gate(
        legacy=legacy,
        split=split,
        prediction_equivalent=True,
        ledger_equivalent=True,
        downstream_compared=False,
        downstream_equivalent=False,
        model_completeness=True,
        min_speedup_percent=10.0,
        max_model_slowdown_percent=20.0,
    )
    assert gate["decision"] == "KEEP_LEGACY_DEFAULT"
    assert not gate["eligible_for_default_promotion_review"]
    assert not gate["checks"]["downstream_compared"]


def test_promotion_gate_can_become_eligible_only_after_all_checks_pass():
    legacy = _mode(100.0, 10.0)
    split = _mode(80.0, 9.0)

    gate = promotion_gate(
        legacy=legacy,
        split=split,
        prediction_equivalent=True,
        ledger_equivalent=True,
        downstream_compared=True,
        downstream_equivalent=True,
        model_completeness=True,
        min_speedup_percent=10.0,
        max_model_slowdown_percent=20.0,
    )
    assert gate["eligible_for_default_promotion_review"]
    assert gate["decision"] == "ELIGIBLE_FOR_DEFAULT_PROMOTION_REVIEW"
    assert gate["speedup_percent"] == 20.0


def test_resource_ab_uses_two_cpu_workers_only_for_split_process():
    args = SimpleNamespace(
        date="2026-08-16",
        data_path=Path("model.parquet"),
        actual_data_path=Path("actual.csv"),
        training_months=12,
        rt916_train_steps=24,
        seed=42,
    )
    legacy_prediction = _prediction_command(args, "legacy", Path("legacy"))
    split_prediction = _prediction_command(args, "split_process", Path("split"))
    assert legacy_prediction[legacy_prediction.index("--max-cpu-workers") + 1] == "1"
    assert split_prediction[split_prediction.index("--max-cpu-workers") + 1] == "2"

    legacy_downstream = _downstream_command(args, "legacy", Path("legacy"))
    split_downstream = _downstream_command(args, "split_process", Path("split"))
    assert legacy_downstream[legacy_downstream.index("--max-cpu-workers") + 1] == "1"
    assert split_downstream[split_downstream.index("--max-cpu-workers") + 1] == "2"
