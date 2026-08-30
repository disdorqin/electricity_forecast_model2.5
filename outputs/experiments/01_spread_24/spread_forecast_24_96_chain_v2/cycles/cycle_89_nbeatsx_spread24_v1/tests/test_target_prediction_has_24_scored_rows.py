import numpy as np

from nbeatsx_spread.evaluation.metrics import compute_metrics


def test_target_day_headline_contract_is_24_rows():
    pred = np.zeros(34, dtype=np.float32)
    target = np.ones(34, dtype=np.float32)
    metrics = compute_metrics(pred[10:], target[10:])
    assert metrics["sample_count"] == 24
