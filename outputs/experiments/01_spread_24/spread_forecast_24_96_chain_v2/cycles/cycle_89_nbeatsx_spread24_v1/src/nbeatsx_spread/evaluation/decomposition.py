from __future__ import annotations

import torch


def assert_decomposition_sum(initial_level: torch.Tensor, block_forecasts: torch.Tensor, forecast: torch.Tensor, atol: float = 1e-5) -> None:
    expected = initial_level.unsqueeze(-1) + block_forecasts.sum(dim=1)
    if not torch.allclose(expected, forecast, atol=atol, rtol=atol):
        raise AssertionError("forecast is not equal to initial level plus additive block decomposition")
