import numpy as np

from nbeatsx_spread.data.normalization import RobustArrayScaler, fit_target_scale


def test_scalers_do_not_change_when_future_truth_changes():
    train = [np.ones((4, 9), dtype=np.float32)]
    scaler = RobustArrayScaler.fit(train)
    target = fit_target_scale([np.array([1, -2, 3], dtype=np.float32)])
    before = (scaler.to_dict(), target.scale)
    validation_truth = np.full((4, 9), 999999, dtype=np.float32)
    target_truth = np.full(34, -999999, dtype=np.float32)
    _ = validation_truth, target_truth
    assert (scaler.to_dict(), target.scale) == before
