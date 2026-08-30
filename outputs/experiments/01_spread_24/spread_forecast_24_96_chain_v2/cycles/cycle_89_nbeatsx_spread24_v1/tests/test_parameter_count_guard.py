import json
from pathlib import Path

from nbeatsx_spread.model.factory import build_model


def test_parameter_count_is_below_guard():
    config = json.loads((Path(__file__).parents[1] / "configs" / "business_strict34_core.json").read_text(encoding="utf-8"))
    model = build_model(config)
    count = sum(p.numel() for p in model.parameters())
    assert count < config["architecture"]["parameter_warning_threshold"]
