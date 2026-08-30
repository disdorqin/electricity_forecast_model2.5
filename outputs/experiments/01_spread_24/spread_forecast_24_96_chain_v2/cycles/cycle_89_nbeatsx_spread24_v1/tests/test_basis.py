import torch

from nbeatsx_spread.model.basis import IdentityBasis, SeasonalityBasis, TrendBasis


def test_identity_basis_shape_and_split():
    theta = torch.arange(7.0).reshape(1, 7)
    back, fore = IdentityBasis(4, 3)(theta)
    assert back.shape == (1, 4) and fore.shape == (1, 3)
    assert torch.equal(back, theta[:, :4])


def test_trend_basis_shape():
    b, f = TrendBasis(2, 5, 3)(torch.ones(2, 6))
    assert b.shape == (2, 5) and f.shape == (2, 3)


def test_seasonality_basis_shape():
    basis = SeasonalityBasis(2, 8, 4)
    b, f = basis(torch.ones(2, basis.n_basis * 2))
    assert b.shape == (2, 8) and f.shape == (2, 4)
