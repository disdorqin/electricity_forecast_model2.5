"""Counterfactual contract test for the formal SGDFNet 96-point anchor."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SGDFNet" / "src"))

from sgdfnet.data_contract import (
    ACTUAL_TO_FORECAST_MAP,
    DA_COL,
    FeatureConfig,
    RT_COL,
    TIMESTAMP_COL,
    preprocess_dataframe,
)
from sgdfnet.protocol_b_cutoff import _build_protocol_b_visible_frame


def make_frame(source_day: str = "2026-01-01", target_da: float = 999.0, target_rt: float = 10.0):
    rows = []
    source = pd.Timestamp(source_day)
    for day, is_target in ((source, False), (source + pd.Timedelta(days=1), True)):
        for period in range(1, 97):
            ts = day + pd.Timedelta(minutes=15 * period)
            # p96 is represented by next-day 00:00 and belongs to the business day.
            row = {TIMESTAMP_COL: ts, DA_COL: (target_da if is_target else float(period)),
                   RT_COL: target_rt if is_target else float(period + 100)}
            for actual, forecast in ACTUAL_TO_FORECAST_MAP.items():
                row[actual] = 1000.0 + period if not is_target else 2000.0 + period
                row[forecast] = 3000.0 + period
            rows.append(row)
    return pd.DataFrame(rows)


def anchors(frame):
    visible = _build_protocol_b_visible_frame(frame, pd.Timestamp("2026-01-01"), 15, resolution=96)
    target = visible[visible["business_day"] == pd.Timestamp("2026-01-02")].sort_values("target_hour")
    return target["_sgdfnet_da_anchor"].tolist()


def main() -> int:
    base = make_frame()
    values = anchors(base)
    assert len(values) == 96
    assert values[0] == 1.0 and values[-1] == 96.0

    # Target-day RT/actual and target-day DA are forbidden from influencing it.
    changed = base.copy()
    timestamps = pd.to_datetime(changed[TIMESTAMP_COL])
    target_rows = (timestamps >= pd.Timestamp("2026-01-02 00:15")) & (timestamps <= pd.Timestamp("2026-01-03 00:00"))
    changed.loc[target_rows, RT_COL] = -9999.0
    changed.loc[target_rows, DA_COL] = -8888.0
    changed.loc[target_rows, "actual_direct_load"] = -7777.0
    assert anchors(changed) == values

    # D-1 DA p17 is the only allowed counterfactual and must propagate to D p17.
    changed = base.copy()
    ts = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * 17)
    changed.loc[changed[TIMESTAMP_COL] == ts, DA_COL] = 12345.0
    changed_values = anchors(changed)
    assert changed_values[16] == 12345.0
    assert changed_values[0] == values[0] and changed_values[-1] == values[-1]

    # If a source slot is missing, target-day DA still must not become the
    # fallback anchor.  Only the historical-median fallback is allowed.
    visible = _build_protocol_b_visible_frame(
        base, pd.Timestamp("2026-01-01"), 15, resolution=96
    )
    source_p17 = pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * 17)
    target_p17 = pd.Timestamp("2026-01-02") + pd.Timedelta(minutes=15 * 17)
    visible.loc[visible[TIMESTAMP_COL] == source_p17, DA_COL] = float("nan")
    visible.loc[visible[TIMESTAMP_COL] == target_p17, DA_COL] = 987654.0
    inferred, _ = preprocess_dataframe(visible, FeatureConfig(), resolution=96)
    target_anchor = inferred.loc[
        inferred[TIMESTAMP_COL] == target_p17, "da_anchor"
    ].iloc[0]
    assert target_anchor != 987654.0
    print("check_sgdfnet_anchor_contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
