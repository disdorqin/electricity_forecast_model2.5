import json
from pathlib import Path


def test_panel_dates_match_preregistered_config():
    config = json.loads((Path(__file__).parents[1] / "configs/b0_extended_validation_panel.json").read_text(encoding="utf-8"))
    assert config["target_days"] == [
        "2026-06-01", "2026-06-05", "2026-06-10", "2026-06-15",
        "2026-06-20", "2026-06-25", "2026-06-30", "2026-07-01",
        "2026-07-05", "2026-07-10", "2026-07-15", "2026-07-20",
        "2026-07-25", "2026-07-31",
    ]

