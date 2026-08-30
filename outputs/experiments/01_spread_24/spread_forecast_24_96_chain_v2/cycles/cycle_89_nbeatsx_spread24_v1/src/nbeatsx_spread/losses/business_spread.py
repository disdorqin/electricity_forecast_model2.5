from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class LossComponents:
    total: Tensor
    magnitude: Tensor
    bridge_magnitude: Tensor
    balanced_sign: Tensor
    raw_sign: Tensor
    magnitude_weight: float
    balanced_weight: float
    raw_weight: float


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


class StableDirectionalPseudoHuber(nn.Module):
    """Warm-started robust magnitude + differentiable direction surrogate."""

    def __init__(self, target_scale: float = 1.0, delta: float = 1.0, bridge_weight: float = 0.25, temperature: float = 0.35, reliability_full_weight: float = 0.25, phase_1_until: float = 0.20, ramp_end: float = 0.40):
        super().__init__()
        if target_scale <= 0 or delta <= 0 or temperature <= 0:
            raise ValueError("loss scales and temperature must be positive")
        self.target_scale, self.delta, self.bridge_weight = target_scale, delta, bridge_weight
        self.temperature, self.reliability_full_weight = temperature, reliability_full_weight
        self.phase_1_until, self.ramp_end = phase_1_until, ramp_end

    def _weights(self, progress: float) -> tuple[float, float, float]:
        if progress <= self.phase_1_until:
            return 1.0, 0.0, 0.0
        if progress < self.ramp_end:
            a = (progress - self.phase_1_until) / (self.ramp_end - self.phase_1_until)
            return 1.0 - 0.30 * a, 0.20 * a, 0.10 * a
        return 0.70, 0.20, 0.10

    def forward(self, pred: Tensor, target: Tensor, bridge_mask: Tensor, score_mask: Tensor, progress: float = 1.0, return_components: bool = False) -> Tensor | LossComponents:
        if pred.shape != target.shape or pred.shape != bridge_mask.shape or pred.shape != score_mask.shape:
            raise ValueError("loss tensors must share [B,H]")
        z_pred, z_true = pred / self.target_scale, target / self.target_scale
        residual = z_pred - z_true
        ph = self.delta ** 2 * (torch.sqrt(1.0 + (residual / self.delta) ** 2) - 1.0)
        magnitude = _masked_mean(ph, score_mask) + self.bridge_weight * _masked_mean(ph, bridge_mask)
        reliability = (torch.abs(z_true) / self.reliability_full_weight).clamp(0.0, 1.0) * score_mask
        pos = (z_true > 0).float() * score_mask
        neg = (z_true < 0).float() * score_mask
        pos_loss = _masked_mean(F.softplus(-z_pred / self.temperature) * reliability, pos)
        neg_loss = _masked_mean(F.softplus(z_pred / self.temperature) * reliability, neg)
        balanced = 0.5 * (pos_loss + neg_loss)
        sign = torch.where(z_true >= 0, torch.ones_like(z_true), -torch.ones_like(z_true))
        raw = _masked_mean(F.softplus(-sign * z_pred / self.temperature) * reliability, score_mask)
        wm, wb, wr = self._weights(float(progress))
        total = wm * magnitude + wb * balanced + wr * raw
        result = LossComponents(total, magnitude, self.bridge_weight * _masked_mean(ph, bridge_mask), balanced, raw, wm, wb, wr)
        return result if return_components else result.total
