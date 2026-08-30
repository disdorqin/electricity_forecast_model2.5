from nbeatsx_spread.evaluation.panel import daily_metric_row


def test_majority_collapse_is_explicit():
    row = daily_metric_row("2026-06-01", [1, 1, 1, 1], [1, 1, -1, 1])
    assert row["majority_collapse"] is True
    assert row["raw_minus_majority"] == 0.0

