"""Regression checks for the signed-spread sMAPE contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments.spread_direction_24.spread_metrics import (
    smape_percent,
    smape_terms_percent,
)


def main() -> None:
    assert smape_percent(np.array([1.0]), np.array([1.0])) == 0.0
    assert np.isclose(smape_percent(np.array([-10.0]), np.array([-5.0])), 66.66666666666667)
    assert np.isclose(smape_percent(np.array([0.0, 1.0]), np.array([0.0, 0.0])), 100.0)
    terms = smape_terms_percent(np.array([-10.0, 10.0]), np.array([10.0, -10.0]))
    assert np.allclose(terms, [200.0, 200.0])
    print("PASS: signed-spread sMAPE zero, negative and opposite-sign cases")


if __name__ == "__main__":
    main()
