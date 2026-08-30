import numpy as np

from nbeatsx_spread.evaluation.metrics import compute_metrics


def test_zero_truth_numeric_but_not_direction():
    out = compute_metrics(np.array([0.0, 0.0, 2.0, -2.0]), np.array([0.0, 1.0, 2.0, -2.0]))
    assert out["sample_count"] == 4
    assert out["direction_sample_count"] == 3
    assert out["mae"] == 0.25
    assert out["positive_recall"] == 0.5
    assert out["negative_recall"] == 1.0


def test_zero_prediction_is_wrong_for_both_signs():
    out = compute_metrics(np.array([0.0, 0.0]), np.array([10.0, -10.0]))
    assert out["direction_accuracy"] == 0.0
    assert out["positive_recall"] == 0.0
    assert out["negative_recall"] == 0.0
