from nbeatsx_spread.evaluation.panel import baseline_row_from_arrays, paired_daily_delta


def test_baseline_delta_requires_exact_same_date():
    nbeats = baseline_row_from_arrays("2026-06-01", [1, -1], [1, -1], "NBEATSx")
    base = baseline_row_from_arrays("2026-06-01", [1, 1], [1, -1], "Cycle88")
    delta = paired_daily_delta(nbeats, base)
    assert delta["delta_raw"] == 0.5

