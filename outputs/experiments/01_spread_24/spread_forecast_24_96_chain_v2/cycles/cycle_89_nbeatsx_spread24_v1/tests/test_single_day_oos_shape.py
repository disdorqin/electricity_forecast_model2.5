from pathlib import Path

from nbeatsx_spread.data.business_dataset import build_inference_sample
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource


def test_single_day_oos_input_shape():
    root = next(p for p in Path(__file__).parents if (p / "utils" / "resolution.py").exists())
    sample = build_inference_sample(CanonicalHourlySource.from_csv(root / "data/24/canonical/shandong_pmos_hourly.csv"), "2026-06-01")
    assert sample.y_backcast.shape == (168,)
    assert sample.x_backcast.shape == (168, 9)
    assert sample.x_future.shape == (34, 9)
