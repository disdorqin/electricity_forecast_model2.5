from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn


class IdentityBasis(nn.Module):
    """Reference identity basis: theta is split into backcast and forecast."""

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size, self.forecast_size = backcast_size, forecast_size

    def forward(self, theta: Tensor, *_: Tensor) -> Tuple[Tensor, Tensor]:
        assert theta.ndim == 2 and theta.shape[1] == self.backcast_size + self.forecast_size
        return theta[:, : self.backcast_size], theta[:, -self.forecast_size :]


class TrendBasis(nn.Module):
    """Polynomial basis matching the official implementation's normalized grids."""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        p = degree + 1
        back = torch.stack([(torch.arange(backcast_size, dtype=torch.float32) / backcast_size) ** i for i in range(p)])
        fore = torch.stack([(torch.arange(forecast_size, dtype=torch.float32) / forecast_size) ** i for i in range(p)])
        self.register_buffer("backcast_basis", back)
        self.register_buffer("forecast_basis", fore)
        self.n_basis = p

    def forward(self, theta: Tensor, *_: Tensor) -> Tuple[Tensor, Tensor]:
        assert theta.shape[1] == 2 * self.n_basis
        return torch.einsum("bp,pt->bt", theta[:, self.n_basis :], self.backcast_basis), torch.einsum("bp,pt->bt", theta[:, : self.n_basis], self.forecast_basis)


class SeasonalityBasis(nn.Module):
    """Fourier basis matching cchallu/nbeatsx's cosine/sine construction."""

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        if harmonics < 1:
            raise ValueError("harmonics must be positive")
        frequency = torch.cat([torch.zeros(1), torch.arange(harmonics, harmonics / 2 * forecast_size, dtype=torch.float32) / harmonics]).reshape(1, -1)
        back_grid = -2 * math.pi * (torch.arange(backcast_size, dtype=torch.float32)[:, None] / forecast_size) * frequency
        fore_grid = 2 * math.pi * (torch.arange(forecast_size, dtype=torch.float32)[:, None] / forecast_size) * frequency
        back = torch.cat([torch.cos(back_grid).T, torch.sin(back_grid).T], dim=0)
        fore = torch.cat([torch.cos(fore_grid).T, torch.sin(fore_grid).T], dim=0)
        self.register_buffer("backcast_basis", back)
        self.register_buffer("forecast_basis", fore)
        self.n_basis = back.shape[0]

    def forward(self, theta: Tensor, *_: Tensor) -> Tuple[Tensor, Tensor]:
        assert theta.shape[1] == 2 * self.n_basis
        return torch.einsum("bp,pt->bt", theta[:, self.n_basis :], self.backcast_basis), torch.einsum("bp,pt->bt", theta[:, : self.n_basis], self.forecast_basis)
