import torch

from nbeatsx_spread.model.wavenet import ExogenousWaveNet


def test_wavenet_preserves_time_length():
    y = ExogenousWaveNet(9, 8, levels=3, kernel_size=3, dropout=0.0)(torch.randn(2, 9, 202))
    assert y.shape == (2, 8, 202)
