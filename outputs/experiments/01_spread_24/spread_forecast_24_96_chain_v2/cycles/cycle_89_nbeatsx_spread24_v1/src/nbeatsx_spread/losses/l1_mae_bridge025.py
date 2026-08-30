"""Magnitude-only L1 objective for the frozen loss study."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class D24MAEBridge025(nn.Module):
    """Use full D-day MAE plus 0.25-weighted bridge MAE.

    The loss is evaluated in normalized target units.  It contains no sign
    surrogate, no target-day truth and no trainable weighting parameter.
    """

    def __init__(self, bridge_weight: float = 0.25) -> None:
        super().__init__()
        if bridge_weight < 0:
            raise ValueError("bridge_weight must be non-negative")
        self.bridge_weight = float(bridge_weight)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        bridge_mask: Tensor,
        score_mask: Tensor,
        progress: float = 1.0,
    ) -> Tensor:
        """Return ``MAE(D-day) + 0.25 * MAE(bridge)`` for ``[B,H]`` tensors."""
        del progress
        if pred.shape != target.shape or pred.shape != bridge_mask.shape or pred.shape != score_mask.shape:
            raise ValueError("L1 tensors must share [B,H]")
        if pred.ndim != 2:
            raise ValueError("L1 objective expects [B,H] tensors")
        residual = torch.abs(pred - target)
        score_denom = score_mask.sum().clamp_min(1.0)
        bridge_denom = bridge_mask.sum().clamp_min(1.0)
        d24 = (residual * score_mask).sum() / score_denom
        bridge = (residual * bridge_mask).sum() / bridge_denom
        return d24 + self.bridge_weight * bridge
