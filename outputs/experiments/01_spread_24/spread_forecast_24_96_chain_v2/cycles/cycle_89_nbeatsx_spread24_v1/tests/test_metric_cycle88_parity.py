import numpy as np

from nbeatsx_spread.evaluation.metrics import compute_metrics


def test_cycle88_toy_formula_parity():
    pred = np.array([2.0, -3.0, 0.0, 4.0, -1.0])
    target = np.array([1.0, -2.0, 5.0, 0.0, -4.0])
    out = compute_metrics(pred, target)
    assert out["mae"] == np.mean(np.abs(pred - target))
    assert out["rmse"] == np.sqrt(np.mean((pred - target) ** 2))
    assert out["direction_accuracy"] == 3 / 4
    assert out["positive_recall"] == 0.5
    assert out["negative_recall"] == 1.0
    assert out["balanced_accuracy"] == 0.75
