from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_mae(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError("prediction and target shape mismatch")
    if mask is None:
        mask = torch.ones_like(target)
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


class PaperMAE(nn.Module):
    """Exact pointwise MAE with an explicit mask."""

    def forward(self, pred: Tensor, target: Tensor, mask: Tensor | None = None, **_: object) -> Tensor:
        return masked_mae(pred, target, mask)
