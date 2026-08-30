from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve(); CYCLE = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CYCLE / "src"))

from nbeatsx_spread.audits import audit_covariate_availability, audit_holdout, audit_horizon, audit_origin, audit_training_cutoff, run_counterfactual_audit
from nbeatsx_spread.contracts import assert_contract, latest_complete_label_day
from nbeatsx_spread.data.business_dataset import complete_business_days
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource
from nbeatsx_spread.data.origin_index import build_origin_window, strict_train_days


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--target-day", required=True); ap.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv"); args = ap.parse_args()
    assert_contract(); source = CanonicalHourlySource.from_csv(args.data); w = build_origin_window(args.target_day)
    days = strict_train_days(args.target_day, complete_business_days(source))
    audits = [audit_origin(args.target_day, w), audit_horizon(w), audit_covariate_availability(source, args.target_day), audit_training_cutoff(args.target_day, days), *run_counterfactual_audit(source, args.target_day), audit_holdout(False)]
    payload = {"target_day": args.target_day, "forecast_origin": "D-1 14:00", "training_last_day": latest_complete_label_day(args.target_day), "target_day_actual_as_feature": False, "target_day_DA_as_feature": False, "d1_post14_spread_as_feature": False, "final_holdout_touched": False, "audits": [a.as_dict() for a in audits], "leakage_status": "STRICT/PASS" if all(a.passed for a in audits) else "INVALID-LEAKAGE"}
    print(json.dumps(payload, ensure_ascii=False, indent=2)); return 0 if payload["leakage_status"] == "STRICT/PASS" else 1


if __name__ == "__main__": sys.exit(main())
