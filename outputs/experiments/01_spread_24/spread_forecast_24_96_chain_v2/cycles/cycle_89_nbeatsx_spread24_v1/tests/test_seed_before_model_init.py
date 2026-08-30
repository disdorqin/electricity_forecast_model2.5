import json
from pathlib import Path

import torch

from nbeatsx_spread.model.factory import build_model
from nbeatsx_spread.training.provenance import state_dict_hash
from nbeatsx_spread.training.reproducibility import seed_everything


def test_seed_before_model_build_is_repeatable():
    config = json.loads((Path(__file__).parents[1] / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    seed_everything(42)
    first = build_model(config)
    first_hash = state_dict_hash(first)
    seed_everything(42)
    second = build_model(config)
    assert first_hash == state_dict_hash(second)
    assert all(torch.equal(first.state_dict()[k], second.state_dict()[k]) for k in first.state_dict())

