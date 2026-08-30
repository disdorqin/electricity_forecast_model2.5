from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, daily_metric_row


def test_daily_macro_preserves_daily_distribution():
    rows = [
        daily_metric_row("2026-06-01", [1, -1], [1, -1]),
        daily_metric_row("2026-06-02", [1, 1], [1, -1]),
    ]
    out = aggregate_daily_metrics(rows)
    assert out["day_count"] == 2
    assert out["balanced"]["mean"] == 0.75
    assert out["raw"]["mean"] == 0.75

