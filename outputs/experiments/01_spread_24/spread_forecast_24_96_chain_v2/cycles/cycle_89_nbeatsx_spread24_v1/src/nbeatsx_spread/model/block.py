from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from .basis import IdentityBasis, SeasonalityBasis, TrendBasis
from .initialization import initialize_linear
from .tcn import TemporalConvNet
from .wavenet import ExogenousWaveNet


def _activation(name: str) -> nn.Module:
    values = {"relu": nn.ReLU, "softplus": nn.Softplus, "selu": nn.SELU, "prelu": nn.PReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
    key = name.lower()
    if key not in values:
        raise ValueError(f"unsupported activation: {name}")
    return values[key]()


class ExogenousBasis(nn.Module):
    """Project convolutional basis channels to backcast and forecast."""

    def __init__(self, encoder: nn.Module, channels: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.encoder, self.channels = encoder, channels
        self.backcast_size, self.forecast_size = backcast_size, forecast_size

    def forward(self, theta: Tensor, x_back: Tensor, x_future: Tensor) -> Tuple[Tensor, Tensor]:
        x_all = torch.cat([x_back, x_future], dim=-1)
        basis = self.encoder(x_all)
        assert basis.shape[-1] == self.backcast_size + self.forecast_size
        back_basis, fore_basis = basis[..., : self.backcast_size], basis[..., self.backcast_size :]
        assert theta.shape[1] == 2 * self.channels
        backcast = torch.einsum("bc,bct->bt", theta[:, self.channels :], back_basis)
        forecast = torch.einsum("bc,bct->bt", theta[:, : self.channels], fore_basis)
        return backcast, forecast


class NBEATSxBlock(nn.Module):
    """An NBEATSx block: MLP theta projection followed by a paper basis."""

    def __init__(self, backcast_size: int, forecast_size: int, n_features: int, stack_type: str, hidden: int = 256, n_layers: int = 2, channels: int = 8, kernel_size: int = 3, activation: str = "softplus", dropout_theta: float = 0.05, dropout_exogenous: float = 0.05, batch_normalization: bool = False, initialization: str = "orthogonal", trend_degree: int = 2, seasonality_harmonics: int = 2):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.stack_type = stack_type.lower()
        context_size = backcast_size + (backcast_size + forecast_size) * n_features
        if self.stack_type == "identity":
            basis: nn.Module = IdentityBasis(backcast_size, forecast_size)
            theta_size = backcast_size + forecast_size
        elif self.stack_type == "trend":
            basis = TrendBasis(trend_degree, backcast_size, forecast_size)
            theta_size = 2 * (trend_degree + 1)
        elif self.stack_type == "seasonality":
            basis = SeasonalityBasis(seasonality_harmonics, backcast_size, forecast_size)
            theta_size = 2 * basis.n_basis
        elif self.stack_type == "exogenous_tcn":
            encoder = TemporalConvNet(n_features, [channels] * 4, kernel_size=kernel_size, dropout=dropout_exogenous)
            basis = ExogenousBasis(encoder, channels, backcast_size, forecast_size)
            theta_size = 2 * channels
        elif self.stack_type == "exogenous_wavenet":
            encoder = ExogenousWaveNet(n_features, channels, levels=4, kernel_size=kernel_size, dropout=dropout_exogenous)
            basis = ExogenousBasis(encoder, channels, backcast_size, forecast_size)
            theta_size = 2 * channels
        else:
            raise ValueError(f"unknown NBEATSx stack type: {stack_type}")
        layers: list[nn.Module] = []
        in_features = context_size
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_features, hidden), _activation(activation)])
            if dropout_theta:
                layers.append(nn.Dropout(dropout_theta))
            if batch_normalization:
                layers.append(nn.BatchNorm1d(hidden))
            in_features = hidden
        layers.append(nn.Linear(in_features, theta_size))
        self.layers = nn.Sequential(*layers)
        self.basis = basis
        initialize_linear(self.layers, initialization)

    def forward(self, residual: Tensor, x_backcast: Tensor, x_future: Tensor) -> Tuple[Tensor, Tensor]:
        if residual.ndim != 2 or x_backcast.ndim != 3 or x_future.ndim != 3:
            raise ValueError("block inputs must be residual [B,L], x_backcast [B,L,C], x_future [B,H,C]")
        context = torch.cat([residual, x_backcast.flatten(1), x_future.flatten(1)], dim=1)
        theta = self.layers(context)
        return self.basis(theta, x_backcast.transpose(1, 2), x_future.transpose(1, 2))
