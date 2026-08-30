from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .block import NBEATSxBlock


@dataclass
class ForecastOutput:
    forecast: Tensor
    block_forecasts: Tensor
    initial_level: Tensor


class NBEATSx(nn.Module):
    """Double-residual NBEATSx with additive block forecasts.

    Each block receives the current target residual plus exogenous history and
    future covariates.  It removes a backcast component from the residual and
    adds its forecast component to the running forecast.  This is the central
    paper-style computation path used by Cycle89.
    """

    def __init__(self, input_size: int, horizon: int, n_features: int, stack_types: tuple[str, ...] = ("identity", "exogenous_tcn"), blocks_per_stack: tuple[int, ...] = (1, 1), hidden_units: int | tuple[int, ...] = 256, n_layers: int | tuple[int, ...] = 2, exogenous_channels: int = 8, kernel_size: int = 3, activation: str = "softplus", dropout_theta: float = 0.05, dropout_exogenous: float = 0.05, batch_normalization: bool = False, initialization: str = "orthogonal", decomposition_enabled: bool = True):
        super().__init__()
        if len(stack_types) != len(blocks_per_stack):
            raise ValueError("stack_types and blocks_per_stack must have equal length")
        hidden = (hidden_units,) * len(stack_types) if isinstance(hidden_units, int) else hidden_units
        layers = (n_layers,) * len(stack_types) if isinstance(n_layers, int) else n_layers
        if len(hidden) != len(stack_types) or len(layers) != len(stack_types):
            raise ValueError("per-stack config length mismatch")
        modules = []
        names = []
        for stack, count, width, depth in zip(stack_types, blocks_per_stack, hidden, layers):
            for block_id in range(count):
                modules.append(NBEATSxBlock(input_size, horizon, n_features, stack, width, depth, exogenous_channels, kernel_size, activation, dropout_theta, dropout_exogenous, batch_normalization, initialization))
                names.append(f"{stack}_{block_id}")
        self.blocks = nn.ModuleList(modules)
        self.block_names = tuple(names)
        self.decomposition_enabled = decomposition_enabled
        self.input_size, self.horizon, self.n_features = input_size, horizon, n_features

    def forward(self, y_backcast: Tensor, x_backcast: Tensor, x_future: Tensor, return_decomposition: bool = False) -> Tensor | ForecastOutput:
        if y_backcast.ndim != 2 or y_backcast.shape[1] != self.input_size:
            raise ValueError(f"y_backcast must be [B,{self.input_size}]")
        if x_backcast.shape[:2] != y_backcast.shape or x_backcast.shape[2] != self.n_features:
            raise ValueError("x_backcast shape mismatch")
        if x_future.ndim != 3 or x_future.shape[1:] != (self.horizon, self.n_features):
            raise ValueError("x_future shape mismatch")
        # The official residual path works in reverse time; the final forecast
        # starts from the last observed level and accumulates block forecasts.
        residual = y_backcast.flip(-1)
        x_back = x_backcast.flip(1)
        forecast = y_backcast[:, -1:].expand(-1, self.horizon)
        components = []
        for block in self.blocks:
            backcast, block_forecast = block(residual, x_back, x_future)
            residual = residual - backcast
            forecast = forecast + block_forecast
            components.append(block_forecast)
        decomposition = torch.stack(components, dim=1) if components else forecast.new_empty((forecast.shape[0], 0, self.horizon))
        if return_decomposition:
            return ForecastOutput(forecast, decomposition, y_backcast[:, -1])
        return forecast

    def decomposed_prediction(self, y_backcast: Tensor, x_backcast: Tensor, x_future: Tensor) -> ForecastOutput:
        return self.forward(y_backcast, x_backcast, x_future, return_decomposition=True)  # type: ignore[return-value]
