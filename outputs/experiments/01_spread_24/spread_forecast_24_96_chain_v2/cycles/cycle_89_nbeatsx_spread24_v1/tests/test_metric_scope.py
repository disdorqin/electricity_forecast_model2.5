import numpy as np

from nbeatsx_spread.evaluation.metrics import compute_metrics


def test_metric_uses_score_mask_only():
    pred = np.array([[999.0, 1.0, -1.0]])
    target = np.array([[1.0, 1.0, -1.0]])
    metrics = compute_metrics(pred, target, np.array([[0.0, 1.0, 1.0]]))
    assert metrics["direction_accuracy"] == 1.0 and metrics["mae"] == 0.0
