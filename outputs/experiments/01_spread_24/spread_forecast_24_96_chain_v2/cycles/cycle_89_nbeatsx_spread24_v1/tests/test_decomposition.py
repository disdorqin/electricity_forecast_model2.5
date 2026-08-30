import torch

from nbeatsx_spread.evaluation.decomposition import assert_decomposition_sum
from nbeatsx_spread.model.nbeatsx import NBEATSx


def test_decomposition_sum():
    model = NBEATSx(8, 4, 2, ("identity", "exogenous_tcn"), (1, 1), 16, 1, 4, 3, "softplus", 0.0)
    out = model(torch.randn(2, 8), torch.randn(2, 8, 2), torch.randn(2, 4, 2), True)
    assert_decomposition_sum(out.initial_level, out.block_forecasts, out.forecast)
