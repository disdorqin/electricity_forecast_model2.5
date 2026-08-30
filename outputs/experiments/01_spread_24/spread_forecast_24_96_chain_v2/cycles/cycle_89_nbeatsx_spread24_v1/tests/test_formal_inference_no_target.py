from pathlib import Path

from nbeatsx_spread.data.business_dataset import build_inference_sample
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource


def test_formal_inference_sample_has_no_target_day_label():
    root = next(p for p in Path(__file__).parents if (p / "utils" / "resolution.py").exists())
    source = CanonicalHourlySource.from_csv(root / "data/24/canonical/shandong_pmos_hourly.csv")
    sample = build_inference_sample(source, "2026-06-01")
    assert not hasattr(sample, "y_future")
    assert "y_future" not in sample.tensors()
    assert sample.x_future.shape == (34, 9)
