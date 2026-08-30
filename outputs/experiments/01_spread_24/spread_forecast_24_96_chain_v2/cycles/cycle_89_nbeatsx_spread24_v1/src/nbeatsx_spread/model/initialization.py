from __future__ import annotations

from torch import nn


def initialize_linear(module: nn.Module, method: str) -> None:
    """Apply one of the paper-search initializers to every Linear layer."""
    if method not in {"orthogonal", "he_normal", "glorot_normal"}:
        raise ValueError(f"unsupported initialization: {method}")
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            if method == "orthogonal":
                nn.init.orthogonal_(layer.weight)
            elif method == "he_normal":
                nn.init.kaiming_normal_(layer.weight)
            else:
                nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
