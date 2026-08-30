import numpy as np

from nbeatsx_spread.data.normalization import RobustArrayScaler


def test_calendar_channels_are_identity_transformed():
    train = np.array([[10, 20, 30, 40, 50, 0.5, -0.5, 1.0, 0.0]], dtype=np.float32)
    scaler = RobustArrayScaler.fit([train], numeric_channels=(0, 1, 2, 3, 4))
    value = np.array([[999, 999, 999, 999, 999, 0.25, -0.25, -1.0, 1.0]], dtype=np.float32)
    transformed = scaler.transform(value)
    np.testing.assert_allclose(transformed[0, 5:], value[0, 5:])
    assert scaler.to_dict()["identity_channels"] == [5, 6, 7, 8]

