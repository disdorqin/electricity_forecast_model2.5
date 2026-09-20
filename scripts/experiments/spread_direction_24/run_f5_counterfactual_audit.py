"""Cycle 24: counterfactual information-boundary audit for the spread cube.

The audit rebuilds the feature table after perturbing raw actual values in
three locations.  Features for a target day must be invariant to target-day
actuals and D-1 post-14:00 actuals, while D-1 p1-p14 actuals are allowed to
change visible-context features.  This is an audit only; it writes no
production data or model outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.build_feature_cube import build_slot_table
from utils.resolution import HOURLY


def _same(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> bool:
    return bool(np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True))


def _mutate(raw: pd.DataFrame, mask: pd.Series, actual_cols: list[str], delta: float) -> pd.DataFrame:
    out = raw.copy()
    for c in actual_cols:
        vals = pd.to_numeric(out.loc[mask, c], errors="coerce")
        out.loc[mask, c] = vals + delta
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/24/canonical/shandong_pmos_hourly.csv"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target-day", default="2026-07-15")
    args = ap.parse_args()
    raw = pd.read_csv(args.source, encoding="gb18030")
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw["_bd_audit"] = raw["时刻"].map(HOURLY.business_day_from_timestamp).astype(str)
    raw["_bp_audit"] = raw["时刻"].map(HOURLY.business_period_from_timestamp).astype(int)
    target = pd.Timestamp(args.target_day)
    target_s = target.strftime("%Y-%m-%d")
    d1_s = (target - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    actual_cols = [c for c in raw.columns if str(c).endswith("实际值")]
    # Price labels are not suffixed with 实际值 in the canonical hourly table,
    # but they are the realized RT/DA inputs that drive target_spread/context.
    # Perturb RT only so the realized spread actually changes; DA is a
    # separate target-day contract and is covered by the target-actual case.
    price_cols = [c for c in ("实时电价",) if c in raw.columns]
    mutation_cols = list(dict.fromkeys(actual_cols + price_cols))
    if not mutation_cols:
        raise RuntimeError("no actual-value or realized price columns found")

    base, groups, registry, _, _ = build_slot_table(raw.drop(columns=["_bd_audit", "_bp_audit"]))
    feature_cols = [c for g in groups.values() for c in g]
    f5_cols = list(groups.get("F5", []))
    context_cols = [c for c in feature_cols if c.startswith("ctx_")]
    target_rows = base[base.target_day.eq(target_s)].sort_values("hour_business")
    if len(target_rows) != 24:
        raise RuntimeError(f"target day {target_s} does not have 24 rows")

    cases = []
    mutations = {
        "target_actual": raw["_bd_audit"].eq(target_s),
        "d1_post14_actual": raw["_bd_audit"].eq(d1_s) & raw["_bp_audit"].gt(14),
        "d1_p1_p14_actual": raw["_bd_audit"].eq(d1_s) & raw["_bp_audit"].between(1, 14),
    }
    for name, mask in mutations.items():
        mutated = _mutate(raw.drop(columns=["_bd_audit", "_bp_audit"]), mask, mutation_cols, 1_000_000.0)
        cf, _, _, _, _ = build_slot_table(mutated)
        cf_target = cf[cf.target_day.eq(target_s)].sort_values("hour_business")
        cases.append({
            "case": name,
            "rows_perturbed": int(mask.sum()),
            "all_features_invariant": _same(target_rows, cf_target, feature_cols),
            "f5_invariant": _same(target_rows, cf_target, f5_cols),
            "context_changed": not _same(target_rows, cf_target, context_cols),
            "expected": "invariant" if name != "d1_p1_p14_actual" else "context_change_allowed",
        })
    result = {
        "status": "STRICT/PASS" if cases[0]["all_features_invariant"] and cases[1]["all_features_invariant"] and cases[2]["context_changed"] else "FAIL",
        "audit_type": "counterfactual_information_boundary",
        "forecast_origin": "D-1 14:00",
        "target_day": target_s,
        "target_actual_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "f5_availability": "D-2 or earlier",
        "final_holdout_touched": False,
        "cases": cases,
    }
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(cases).to_csv(out / "counterfactual_cases.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "STRICT/PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
