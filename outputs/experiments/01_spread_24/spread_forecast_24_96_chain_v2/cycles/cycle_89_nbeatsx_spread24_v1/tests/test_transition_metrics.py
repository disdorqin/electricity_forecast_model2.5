from nbeatsx_spread.evaluation.panel import daily_metric_row


def test_transition_count_and_exact_timing_metrics():
    row = daily_metric_row("2026-06-01", [1, -1, -1, 1], [1, -1, -1, 1])
    assert row["actual_sign_switch_count"] == 2
    assert row["predicted_sign_switch_count"] == 2
    assert row["transition_precision"] == 1.0
    assert row["transition_recall"] == 1.0
    assert row["transition_f1"] == 1.0


def test_transition_mismatch_is_not_hidden_by_total_count():
    row = daily_metric_row("2026-06-01", [1, 1, -1, -1], [1, -1, -1, 1])
    assert row["actual_sign_switch_count"] == 2
    assert row["predicted_sign_switch_count"] == 1
    assert row["transition_precision"] == 0.0
    assert row["transition_recall"] == 0.0
    assert row["transition_f1"] == 0.0
