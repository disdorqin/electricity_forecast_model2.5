"""Delivery contract: formal 96 final uses RT fuse, not classifier output."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipelines.ledger_full import _collect_final_outputs
from utils.resolution import QUARTER


def frame(day: str, value: float) -> pd.DataFrame:
    ts = pd.date_range(pd.Timestamp(day) + pd.Timedelta(minutes=15), periods=96, freq="15min")
    return pd.DataFrame({
        "business_day": day, "ds": ts, "business_period": range(1, 97),
        "period": range(1, 97), "hour_business": [(i - 1) // 4 + 1 for i in range(1, 97)],
        "y_fused": value,
    })


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="efm3-clf-policy-") as tmp:
        root = Path(tmp)
        for task, value in (("dayahead", 100.0), ("realtime", 90.0)):
            path = root / "2026-01-01" / task / "fuse" / "fused_predictions.csv"
            path.parent.mkdir(parents=True)
            frame("2026-01-01", value).to_csv(path, index=False)
        # A previous attempt may have left a stale formal RT final.  The
        # current fuse artifact must overwrite it rather than being ignored.
        stale = root / "2026-01-01" / "realtime" / "final" / "realtime_final_predictions.csv"
        stale.parent.mkdir(parents=True)
        frame("2026-01-01", 1.0).to_csv(stale, index=False)
        result = _collect_final_outputs(root, "2026-01-01", QUARTER)
        assert result["status"] == "complete", result
        assert result.get("submission_realtime_source") != "classifier_corrected"
        submission = pd.read_csv(root / "2026-01-01" / "final" / "submission_ready.csv")
        assert len(submission) == 96
        assert (pd.read_csv(stale)["y_fused"] == 90.0).all()
        assert (submission["realtime_price"] == 90.0).all()
        assert not (root / "2026-01-01" / "final" / "realtime_final_predictions_corrected.csv").exists()
    print("check_classifier_policy_96: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
