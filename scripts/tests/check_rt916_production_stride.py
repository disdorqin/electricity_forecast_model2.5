"""RT916 production stride must ignore ambient environment overrides."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from RT916_SpikeFusionNet.pipeline import ModelPipeline, core
from utils.resolution import resolve_resolution


def main() -> int:
    old_env = os.environ.get("RT916_TRAIN_STEPS")
    old_value = core.CONFIG.get("TRAIN_STEPS")
    try:
        os.environ["RT916_TRAIN_STEPS"] = "1"
        ModelPipeline._apply_train_stride(
            resolve_resolution("15min"), {"production_mode": True}
        )
        assert core.CONFIG["TRAIN_STEPS"] == 24
    finally:
        if old_env is None:
            os.environ.pop("RT916_TRAIN_STEPS", None)
        else:
            os.environ["RT916_TRAIN_STEPS"] = old_env
        core.CONFIG["TRAIN_STEPS"] = old_value
    print("check_rt916_production_stride: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
