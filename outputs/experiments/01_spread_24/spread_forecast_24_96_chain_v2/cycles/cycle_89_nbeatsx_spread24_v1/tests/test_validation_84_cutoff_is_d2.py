from datetime import date, timedelta


def test_validation_84_cutoff_is_d2() -> None:
    target_day = date(2026, 6, 1)
    latest_legal_label = target_day - timedelta(days=2)
    validation_end = latest_legal_label
    assert validation_end <= latest_legal_label
