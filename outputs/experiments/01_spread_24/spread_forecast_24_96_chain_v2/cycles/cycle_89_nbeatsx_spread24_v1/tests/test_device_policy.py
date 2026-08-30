from nbeatsx_spread.training.device import select_device


def test_device_policy_returns_one_deterministic_decision():
    decision = select_device(seed=42)
    assert decision.device in {"cpu", "cuda"}
    assert decision.policy == "cuda_if_deterministic_else_cpu"
    assert decision.deterministic is True

