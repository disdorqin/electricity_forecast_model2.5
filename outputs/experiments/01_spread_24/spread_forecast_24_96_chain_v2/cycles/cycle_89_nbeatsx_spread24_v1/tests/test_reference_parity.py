import torch

from nbeatsx_spread.model.nbeatsx import NBEATSx
from nbeatsx_spread.training.reproducibility import seed_everything
from nbeatsx_spread.model.basis import IdentityBasis, SeasonalityBasis, TrendBasis


def test_fixed_seed_forward_parity():
    seed_everything(42); a = NBEATSx(8, 4, 2, ("identity",), (1,), 16, 1, 4, 3, "softplus", 0.0)
    seed_everything(42); b = NBEATSx(8, 4, 2, ("identity",), (1,), 16, 1, 4, 3, "softplus", 0.0)
    x = (torch.randn(2, 8), torch.randn(2, 8, 2), torch.randn(2, 4, 2))
    assert torch.equal(a(*x), b(*x))


def _official():
    import importlib
    import numpy as np
    import sys
    from pathlib import Path
    np.float = float  # type: ignore[attr-defined]
    sys.path.insert(0, str(Path(__file__).parents[1] / "third_party" / "reference_source"))
    return importlib.import_module("src.nbeats.nbeats_model")


def test_official_basis_numerical_parity():
    """Compare local basis tensors with the locked official implementation."""
    official = _official()
    cases = [
        (IdentityBasis(8, 4), official.IdentityBasis(8, 4), 12),
        (TrendBasis(2, 8, 4), official.TrendBasis(2, 8, 4), 6),
        (SeasonalityBasis(2, 8, 4), official.SeasonalityBasis(2, 8, 4), 2 * official.SeasonalityBasis(2, 8, 4).backcast_basis.shape[0]),
    ]
    for local, reference, width in cases:
        theta = torch.randn(3, width)
        local_back, local_fore = local(theta)
        ref_back, ref_fore = reference(theta, torch.empty(3, 1, 8), torch.empty(3, 1, 4))
        assert torch.allclose(local_back, ref_back, atol=1e-6, rtol=1e-6)
        assert torch.allclose(local_fore, ref_fore, atol=1e-6, rtol=1e-6)


def test_official_tcn_numerical_parity():
    """Copy equivalent official TCN weights and compare float32 outputs."""
    official = _official()
    from nbeatsx_spread.model.tcn import TemporalConvNet
    torch.manual_seed(7)
    reference = official.TemporalConvNet(2, [3, 3], kernel_size=3, dropout=0.0)
    local = TemporalConvNet(2, [3, 3], kernel_size=3, dropout=0.0)
    state = local.state_dict()
    ref_state = reference.state_dict()
    mapping = {
        "network.0.net.0.weight_g": "network.0.conv1.weight_g",
        "network.0.net.0.weight_v": "network.0.conv1.weight_v",
        "network.0.net.4.weight_g": "network.0.conv2.weight_g",
        "network.0.net.4.weight_v": "network.0.conv2.weight_v",
        "network.1.net.0.weight_g": "network.1.conv1.weight_g",
        "network.1.net.0.weight_v": "network.1.conv1.weight_v",
        "network.1.net.4.weight_g": "network.1.conv2.weight_g",
        "network.1.net.4.weight_v": "network.1.conv2.weight_v",
    }
    for key in state:
        if key in mapping:
            state[key] = ref_state[mapping[key]]
        elif key in ref_state:
            state[key] = ref_state[key]
    local.load_state_dict(state)
    x = torch.randn(2, 2, 16)
    assert torch.allclose(local(x), reference(x), atol=1e-6, rtol=1e-6)
