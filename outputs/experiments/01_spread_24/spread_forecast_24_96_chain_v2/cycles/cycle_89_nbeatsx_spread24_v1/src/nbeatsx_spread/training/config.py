from __future__ import annotations

from typing import Any

from .trainer import TrainingConfig


def training_config_from_business(config: dict[str, Any], *, overrides: dict[str, Any] | None = None) -> TrainingConfig:
    """Parse the frozen business JSON into the sole runtime training config."""
    t = config["training"]
    values: dict[str, Any] = {
        "batch_size": int(t["batch_size_daily_origins"]),
        "learning_rate": float(t["learning_rate_initial"]),
        "weight_decay": float(t["weight_decay"]),
        "max_steps": int(t["max_steps"]),
        "min_steps": int(t["min_steps"]),
        "eval_every": int(t["val_check_steps"]),
        "patience_checks": int(t["early_stopping_checks"]),
        "gradient_clip_norm": float(t["gradient_clip_norm"]),
        "seed": int(t["seed"]),
        "schedule_total_steps": int(t["max_steps"]),
        "nominal_lr_decay_steps": tuple(int(x) for x in t["nominal_lr_decay_steps"]),
        "lr_decay_gamma": float(t["lr_decay_gamma"]),
    }
    if overrides:
        allowed = {"max_steps", "min_steps", "eval_every", "patience_checks", "schedule_total_steps"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"unsupported training override(s): {sorted(unknown)}")
        values.update(overrides)
    return TrainingConfig(**values)


def config_execution_audit(config: dict[str, Any], runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a machine-readable configured-vs-runtime execution audit.

    Equality is appropriate for execution parameters. Minimum-origin fields
    are constraints, so their audit records the observed runtime fact and
    checks actual >= configured rather than echoing the configured floor.
    """
    t, a = config["training"], config["architecture"]
    pairs = {
        "mixed_precision_business": (t["mixed_precision_business"], runtime["precision"]),
        "dropout_theta": (a["dropout_theta"], runtime["dropout_theta"]),
        "dropout_exogenous": (a["dropout_exogenous"], runtime["dropout_exogenous"]),
        "parameter_warning_threshold": (a["parameter_warning_threshold"], runtime["parameter_warning_threshold"]),
        "nominal_lr_decay_steps": (t["nominal_lr_decay_steps"], runtime["nominal_lr_decay_steps"]),
        "weight_decay": (t["weight_decay"], runtime["weight_decay"]),
        "batch_size": (t["batch_size_daily_origins"], runtime["batch_size"]),
        "early_stop_patience": (t["early_stopping_checks"], runtime["patience_checks"]),
        "gradient_clip_norm": (t["gradient_clip_norm"], runtime["gradient_clip_norm"]),
        "seed": (t["seed"], runtime["seed"]),
        "activation": (a["activation"], runtime["activation"]),
        "initialization": (a.get("initialization", "orthogonal"), runtime["initialization"]),
    }
    rows = [
        {
            "field": field,
            "configured_value": configured,
            "runtime_value": actual,
            "relation": "==",
            "status": "MATCH" if configured == actual else "FAIL",
        }
        for field, (configured, actual) in pairs.items()
    ]
    for field, configured, actual in (
        ("min_train_daily_origins", int(config["min_train_daily_origins"]), int(runtime["train_count"])),
        ("min_validation_daily_origins", int(config["min_validation_daily_origins"]), int(runtime["validation_count"])),
    ):
        rows.append({
            "field": field,
            "configured_value": configured,
            "runtime_value": actual,
            "relation": ">=",
            "status": "MATCH" if actual >= configured else "FAIL",
        })
    parameter_count = runtime.get("parameter_count")
    if parameter_count is None:
        rows.append({
            "field": "parameter_count_guard",
            "configured_value": int(a["parameter_warning_threshold"]),
            "runtime_value": None,
            "relation": "<=",
            "status": "FAIL",
        })
    else:
        rows.append({
            "field": "parameter_count_guard",
            "configured_value": int(a["parameter_warning_threshold"]),
            "runtime_value": int(parameter_count),
            "relation": "<=",
            # Compact feature packages legitimately increase the first
            # exogenous projection width. This remains an auditable warning,
            # rather than silently changing the frozen chassis.
            "status": "MATCH" if int(parameter_count) <= int(a["parameter_warning_threshold"]) else "WARNING",
        })
    return rows
