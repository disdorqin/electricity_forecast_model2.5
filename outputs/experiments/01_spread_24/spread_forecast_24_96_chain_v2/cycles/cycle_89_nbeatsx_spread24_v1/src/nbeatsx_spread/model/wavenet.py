from __future__ import annotations

import math

import torch
from torch import nn

from .tcn import Chomp1d


class ExogenousWaveNet(nn.Module):
    """Reference WaveNet-style exogenous basis with causal dilated convolutions."""

    def __init__(self, in_features: int, out_features: int, levels: int = 4, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, in_features, 1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))
        layers: list[nn.Module] = []
        for i in range(levels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            layers.extend([nn.Conv1d(in_features if i == 0 else out_features, out_features, kernel_size if i == 0 else 3, padding=padding, dilation=dilation), Chomp1d(padding), nn.ReLU()])
            if i == 0:
                layers.append(nn.Dropout(dropout))
        self.wavenet = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wavenet(x * self.weight)
