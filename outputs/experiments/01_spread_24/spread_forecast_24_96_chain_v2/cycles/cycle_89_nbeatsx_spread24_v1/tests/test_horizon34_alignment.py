from nbeatsx_spread.data.origin_index import build_origin_window


def test_h34_boundary():
    w = build_origin_window("2026-06-01")
    assert len(w.bridge_timestamps) == 10 and len(w.scored_timestamps) == 24
    assert w.bridge_timestamps[-1].strftime("%Y-%m-%d %H:%M") == "2026-06-01 00:00"
    assert w.scored_timestamps[0].strftime("%Y-%m-%d %H:%M") == "2026-06-01 01:00"
