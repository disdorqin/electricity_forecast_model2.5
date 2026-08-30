import torch

from nbeatsx_spread.losses.business_spread import StableDirectionalPseudoHuber


def test_directional_loss_is_differentiable_and_warm_started():
    pred = torch.zeros(2, 34, requires_grad=True)
    target = torch.cat([torch.ones(2, 10), -torch.ones(2, 24)], dim=1)
    bridge = torch.cat([torch.ones(10), torch.zeros(24)]).repeat(2, 1)
    score = 1.0 - bridge
    loss = StableDirectionalPseudoHuber(target_scale=10.0)(pred, target, bridge, score, progress=0.3)
    loss.backward()
    assert torch.isfinite(loss) and pred.grad is not None and torch.isfinite(pred.grad).all()
