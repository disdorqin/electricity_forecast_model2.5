import json
from pathlib import Path


def test_history36_val84_train_count() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/a3_history36_val84.json").read_text(encoding="utf-8"))
    assert config["training_history_months"] == 36
    assert config["validation_history_days"] == 84
    assert 36 * 28 - 84 >= 900
