import torch

from nbeatsx_spread.model.nbeatsx import NBEATSx


def test_double_residual_and_additive_forecast():
    model = NBEATSx(8, 4, 2, ("identity",), (1,), 16, 1, 4, 3, "softplus", 0.0)
    y, xb, xf = torch.randn(3, 8), torch.randn(3, 8, 2), torch.randn(3, 4, 2)
    out = model(y, xb, xf, True)
    assert out.block_forecasts.shape == (3, 1, 4)
    assert torch.allclose(out.forecast, out.initial_level[:, None] + out.block_forecasts[:, 0], atol=1e-6)
