from __future__ import annotations

from pathlib import Path

import numpy as np

from nbeatsx_spread.audits.counterfactual import run_counterfactual_audit
from nbeatsx_spread.data.business_dataset import build_inference_sample
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource
from nbeatsx_spread.data.covariates import build_feature_matrices, feature_names
from nbeatsx_spread.data.origin_index import build_origin_window


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = next(p for p in ROOT.parents if (p / "utils" / "resolution.py").exists())
DATA = PROJECT_ROOT / "data/24/canonical/shandong_pmos_hourly.csv"


def test_compact_feature_channel_counts() -> None:
    assert len(feature_names("CORE5_RAW")) == 9
    assert len(feature_names("PHYSICAL_SHAPE")) == 17
    assert len(feature_names("CAUSAL_PRICE_STATE")) == 19


def test_physical_shape_uses_origin_safe_forecast_ramps() -> None:
    source = CanonicalHourlySource.from_csv(DATA)
    window = build_origin_window("2026-06-01")
    back, future = build_feature_matrices(source.frame, list(window.backcast_timestamps), list(window.forecast_timestamps), feature_profile="PHYSICAL_SHAPE")
    assert back.shape == (168, 17)
    assert future.shape == (34, 17)
    # F1 ramp channels begin at channel 12; the first future ramp uses the
    # last forecast row at/before origin, never a realized future value.
    assert np.allclose(future[0, 12:17], back[-1, :5] * 0 + (future[0, :5] - back[-1, :5]), atol=1e-5)


def test_price_state_is_broadcast_and_inference_has_no_target_field() -> None:
    source = CanonicalHourlySource.from_csv(DATA)
    sample = build_inference_sample(source, "2026-06-01", feature_profile="CAUSAL_PRICE_STATE")
    assert sample.x_backcast.shape == (168, 19)
    assert sample.x_future.shape == (34, 19)
    assert np.allclose(sample.x_future[:, 9:], sample.x_future[0, 9:])
    assert not hasattr(sample, "y_future")


def test_compact_features_pass_forbidden_truth_counterfactual() -> None:
    source = CanonicalHourlySource.from_csv(DATA)
    for profile in ("PHYSICAL_SHAPE", "CAUSAL_PRICE_STATE"):
        audits = run_counterfactual_audit(source, "2026-06-01", profile)
        assert all(audit.passed for audit in audits)
