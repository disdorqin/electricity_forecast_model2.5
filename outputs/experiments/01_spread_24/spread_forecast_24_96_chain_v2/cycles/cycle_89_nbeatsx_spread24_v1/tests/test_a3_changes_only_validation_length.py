import json
from pathlib import Path


def test_a3_changes_only_validation_length() -> None:
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    a2 = dict(base, training_history_months=36, validation_history_days=28)
    a3 = dict(base, training_history_months=36, validation_history_days=84)
    assert a2["training_history_months"] == a3["training_history_months"]
    assert a2["validation_history_days"] != a3["validation_history_days"]
    for key in ("architecture", "training", "loss", "input_size", "horizon", "features"):
        if key in a2:
            assert a2[key] == a3[key]
