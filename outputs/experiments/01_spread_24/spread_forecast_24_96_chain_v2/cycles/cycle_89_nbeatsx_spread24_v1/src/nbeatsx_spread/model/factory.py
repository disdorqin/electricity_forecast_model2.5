from __future__ import annotations

from typing import Any

from .nbeatsx import NBEATSx


def build_model(config: dict[str, Any], *, paper: bool = False) -> NBEATSx:
    """Build a model from either the frozen paper or business architecture config."""
    if paper:
        if config.get("input_size") != 168 or config.get("horizon") != 24:
            raise ValueError("paper profile must use L=168,H=24")
        d = float(config.get("dropout", 0.0))
        return NBEATSx(168, 24, int(config.get("n_features", 1)), tuple(config.get("stack_types", ["identity"])), (1,) * len(config.get("stack_types", ["identity"])), int(config.get("hidden_units", 256)), int(config.get("n_layers", 2)), int(config.get("exogenous_n_channels", 8)), int(config.get("kernel_size", 3)), str(config.get("activation", "softplus")), d, d, bool(config.get("batch_normalization", False)), str(config.get("initialization", "orthogonal")))
    arch = config["architecture"]
    tr = config["training"]
    return NBEATSx(int(config["input_size"]), int(config["horizon"]), len(config["feature_profile"]["temporal_covariates"]), tuple(arch["stack_types"]), tuple(arch["n_blocks"]), tuple(arch["hidden_units"]), tuple(arch["n_layers"]), int(arch["exogenous_encoder_channels"]), int(arch["exogenous_kernel_size"]), str(arch["activation"]), float(arch["dropout_theta"]), float(arch["dropout_exogenous"]), bool(arch["batch_normalization"]), str(arch.get("initialization", "orthogonal")), bool(arch["decomposition_enabled"]))
