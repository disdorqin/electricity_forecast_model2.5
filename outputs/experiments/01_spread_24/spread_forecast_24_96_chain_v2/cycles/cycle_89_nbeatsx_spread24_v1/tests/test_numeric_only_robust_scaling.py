import numpy as np

from nbeatsx_spread.data.normalization import RobustArrayScaler


def test_robust_fit_ignores_calendar_channels():
    a = np.array([[0, 1, 2, 3, 4, 0.0, 1.0, 0.0, -1.0], [10, 11, 12, 13, 14, 1.0, 0.0, -1.0, 0.0]], dtype=np.float32)
    b = a.copy()
    b[:, 5:] = 9999
    one = RobustArrayScaler.fit([a], numeric_channels=(0, 1, 2, 3, 4))
    two = RobustArrayScaler.fit([b], numeric_channels=(0, 1, 2, 3, 4))
    np.testing.assert_allclose(one.median, two.median)
    np.testing.assert_allclose(one.iqr, two.iqr)

