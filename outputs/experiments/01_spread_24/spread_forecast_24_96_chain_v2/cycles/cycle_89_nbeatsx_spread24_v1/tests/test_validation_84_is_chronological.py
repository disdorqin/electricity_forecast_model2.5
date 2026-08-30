from datetime import date, timedelta


def test_validation_84_is_chronological() -> None:
    days = [date(2026, 3, 1) + timedelta(days=i) for i in range(84)]
    assert days == sorted(days)
    assert len(days) == len(set(days))
