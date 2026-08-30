import torch

from nbeatsx_spread.model.tcn import TemporalConvNet


def test_tcn_preserves_time_length():
    x = torch.randn(3, 9, 202)
    y = TemporalConvNet(9, [8, 8], kernel_size=3, dropout=0.0)(x)
    assert y.shape == (3, 8, 202)
