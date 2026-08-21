"""Lightweight TimeMixer CUDA runtime contract check.

This is a configuration test, not a long model-training benchmark.  The
actual one-epoch DA/RT smoke run is recorded by the delivery audit command.
"""

from __future__ import annotations

import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TimeMixer.repro_pipeline import _configure_cuda_determinism
from utils.reproducibility import set_global_seed


def main() -> int:
    set_global_seed(42, False)
    _configure_cuda_determinism(False)
    if torch.cuda.is_available():
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is False

        try:
            _configure_cuda_determinism(True)
        except RuntimeError as exc:
            assert "strict deterministic" in str(exc)
        else:
            raise AssertionError("TimeMixer CUDA deterministic mode must fail fast")

    print(
        "PASS: TimeMixer runtime contract "
        f"(cuda={torch.cuda.is_available()}, "
        f"deterministic={torch.are_deterministic_algorithms_enabled()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
