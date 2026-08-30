from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Causal residual block copied in structure from the official TCN reference."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + (x if self.downsample is None else self.downsample(x)))


class TemporalConvNet(nn.Module):
    """Paper-style stacked dilated causal TCN over [batch, channel, time]."""

    def __init__(self, num_inputs: int, num_channels: list[int], kernel_size: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            layers.append(TemporalBlock(num_inputs if i == 0 else num_channels[i - 1], out_channels, kernel_size, 2 ** i, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TCN expects [B,C,T]")
        return self.network(x)
