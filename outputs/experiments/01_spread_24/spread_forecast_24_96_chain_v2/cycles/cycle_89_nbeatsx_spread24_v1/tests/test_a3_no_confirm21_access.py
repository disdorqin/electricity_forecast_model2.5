from pathlib import Path


def test_a3_no_confirm21_access() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/run_a3_history36_val84.py").read_text(encoding="utf-8").lower()
    assert "confirm21" not in runner
    assert "confirmation_panel" not in runner
    assert "september" not in runner
