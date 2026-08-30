import numpy as np

from nbeatsx_spread.evaluation.metrics import metric_by_forecast_offset


def test_h34_offsets_keep_bridge_then_d_day_hours():
    rows = metric_by_forecast_offset(np.zeros((1, 34)), np.ones((1, 34)))
    assert rows[0]["offset"] == 1 and rows[9]["section"] == "bridge"
    assert rows[10]["offset"] == 11 and rows[10]["section"] == "D-day"

