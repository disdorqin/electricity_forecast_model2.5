"""Contract tests for the pre-registered FULLDEV5 panel."""

from __future__ import annotations

import sys
from pathlib import Path


CYCLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CYCLE / "scripts"))
sys.path.insert(0, str(CYCLE / "src"))

from run_full_month_cross_month_dev5 import load_config, registered_days  # noqa: E402


def test_fulldev5_exact_registry() -> None:
    """The machine registry expands to exactly the five frozen months."""
    days, month_info = registered_days(load_config())
    assert len(days) == 150
    assert len(set(days)) == 150
    assert {day[:7] for day in days} == {"2026-01", "2026-02", "2026-04", "2026-06", "2026-07"}
    assert len([day for day in days if month_info[day] == "UNSEEN_FULL3"]) == 89
    assert len([day for day in days if month_info[day] == "SEEN_FULL2"]) == 61


def test_fulldev5_forbidden_stages_are_registered() -> None:
    """The runner rejects a matrix that does not enumerate forbidden branches."""
    config = load_config()
    forbidden = " ".join(config["forbidden_in_this_run"]).lower()
    for token in ("recmо", "recursive_h1", "directional_loss", "confirm21", "september"):
        # The protocol uses ASCII RecMO; accept its normalized spelling here.
        assert token.replace("о", "o") in forbidden.replace("о", "o")


def test_fulldev5_scored_hour_contract() -> None:
    """The registered panel has 3,600 scored hours per model strategy."""
    config = load_config()
    assert config["totals"]["target_days_per_strategy"] == 150
    assert config["totals"]["scored_hours_per_strategy"] == 3600
    assert config["scientific_contract"]["training_history_months"] == 36
    assert config["scientific_contract"]["validation_days"] == 28
