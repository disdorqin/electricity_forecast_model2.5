from nbeatsx_spread.data.origin_index import build_origin_window


def test_origin_is_d_minus_one_14():
    w = build_origin_window("2026-06-01")
    assert w.origin_timestamp == w.backcast_timestamps[-1]
    assert w.origin_timestamp.strftime("%Y-%m-%d %H:%M") == "2026-05-31 14:00"
