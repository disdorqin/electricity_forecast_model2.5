import json
from pathlib import Path

from nbeatsx_spread.training.config import config_execution_audit, training_config_from_business


def test_business_config_is_executed_without_amp():
    config = json.loads((Path(__file__).parents[1] / "configs" / "business_strict34_core.json").read_text(encoding="utf-8"))
    cfg = training_config_from_business(config)
    assert cfg.max_steps == 1200
    assert cfg.nominal_lr_decay_steps == (300, 600, 900)
    runtime = {
        "precision": "float32", "dropout_theta": 0.05, "dropout_exogenous": 0.05, "train_count": 180,
        "validation_count": 21, "parameter_warning_threshold": 2000000,
        "nominal_lr_decay_steps": [300, 600, 900], "weight_decay": 0.0,
        "batch_size": 32, "patience_checks": 8, "gradient_clip_norm": 1.0,
        "seed": 42, "activation": "Softplus", "initialization": "orthogonal",
        "parameter_count": 1206722,
    }
    assert all(row["status"] == "MATCH" for row in config_execution_audit(config, runtime))
