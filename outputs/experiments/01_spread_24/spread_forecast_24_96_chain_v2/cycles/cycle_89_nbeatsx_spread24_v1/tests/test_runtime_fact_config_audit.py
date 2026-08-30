import json
from pathlib import Path

from nbeatsx_spread.training.config import config_execution_audit


def test_minimum_origin_audit_uses_observed_runtime_fact():
    config = json.loads((Path(__file__).parents[1] / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    runtime = {
        "precision": "float32", "dropout_theta": .05, "dropout_exogenous": .05,
        "train_count": 245, "validation_count": 28, "parameter_warning_threshold": 2000000,
        "nominal_lr_decay_steps": [300, 600, 900], "weight_decay": 0.0, "batch_size": 32,
        "patience_checks": 8, "gradient_clip_norm": 1.0, "seed": 42,
        "activation": "Softplus", "initialization": "orthogonal",
    }
    rows = {row["field"]: row for row in config_execution_audit(config, runtime)}
    assert rows["min_train_daily_origins"]["runtime_value"] == 245
    assert rows["min_train_daily_origins"]["relation"] == ">="
    assert rows["min_train_daily_origins"]["status"] == "MATCH"

