import torch

from nbeatsx_spread.training.device import _deterministic_smoke


def test_same_device_determinism_smoke():
    result = _deterministic_smoke(torch.device("cpu"), 42)
    assert result["passed"] is True

