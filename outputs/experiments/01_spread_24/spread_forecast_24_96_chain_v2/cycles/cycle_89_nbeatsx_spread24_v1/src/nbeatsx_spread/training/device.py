from __future__ import annotations

from dataclasses import dataclass

import torch

from .reproducibility import seed_everything


@dataclass(frozen=True)
class DeviceDecision:
    """Single device decision shared by every run in one panel."""

    device: str
    policy: str
    deterministic: bool
    evidence: dict[str, object]


def _deterministic_smoke(device: torch.device, seed: int) -> dict[str, object]:
    """Check that a tiny same-seed forward/backward is repeatable on device."""
    outputs = []
    for _ in range(2):
        seed_everything(seed, deterministic=True)
        model = torch.nn.Linear(4, 3, device=device)
        x = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
        loss = model(x).square().mean()
        loss.backward()
        outputs.append((model.weight.detach().cpu().clone(), model.bias.detach().cpu().clone()))
    same = all(torch.equal(outputs[0][i], outputs[1][i]) for i in range(2))
    if not same:
        raise RuntimeError("same-device deterministic smoke produced different tensors")
    return {"passed": True, "exact_tensor_match": True, "device": str(device)}


def select_device(policy: str = "cuda_if_deterministic_else_cpu", seed: int = 42) -> DeviceDecision:
    """Select one deterministic device, failing over before any model build."""
    if policy != "cuda_if_deterministic_else_cpu":
        raise ValueError(f"unsupported device policy: {policy}")
    if torch.cuda.is_available():
        try:
            evidence = _deterministic_smoke(torch.device("cuda"), seed)
            return DeviceDecision("cuda", policy, True, {"cuda": evidence, "fallback": False})
        except Exception as exc:
            return DeviceDecision("cpu", policy, True, {"cuda": {"passed": False, "error": repr(exc)}, "fallback": True})
    return DeviceDecision("cpu", policy, True, {"cuda": {"passed": False, "reason": "CUDA unavailable"}, "fallback": False})
