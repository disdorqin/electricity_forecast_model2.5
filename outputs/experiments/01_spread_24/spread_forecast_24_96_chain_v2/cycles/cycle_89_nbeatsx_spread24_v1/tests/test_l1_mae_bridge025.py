from __future__ import annotations

import pytest
import torch

from nbeatsx_spread.losses.l1_mae_bridge025 import D24MAEBridge025


def test_l1_is_d24_mae_plus_quarter_bridge_mae() -> None:
    pred = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    target = torch.tensor([[0.0, 1.0, 1.0, 3.0]])
    bridge = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    score = 1.0 - bridge
    # D-day MAE=(4+4)/2=4; bridge MAE=(1+2)/2=1.5; total=4.375.
    assert D24MAEBridge025()(pred, target, bridge, score).item() == pytest.approx(4.375)


def test_l1_has_finite_backward_and_no_sign_branch() -> None:
    pred = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    bridge = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    score = 1.0 - bridge
    loss = D24MAEBridge025()(pred, target, bridge, score)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_l1_rejects_non_matrix_or_mismatched_inputs() -> None:
    fn = D24MAEBridge025()
    with pytest.raises(ValueError, match=r"\[B,H\]"):
        fn(torch.zeros(4), torch.zeros(4), torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="share"):
        fn(torch.zeros(1, 2), torch.zeros(1, 3), torch.zeros(1, 2), torch.zeros(1, 2))
