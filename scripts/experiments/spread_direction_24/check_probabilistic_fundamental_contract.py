"""Gate real probabilistic fundamentals before they can enter research.

The checker is intentionally fail-closed: an absent source manifest is reported
as ``STRICT/PASS_NO_SOURCE`` and never as an eligible probabilistic input.  A
candidate manifest must carry source identity, target day, forecast publish
time, quantile levels and the forecast-origin contract.  No target actuals or
post-origin realized values are accepted.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")
FORBIDDEN = ("actual", "realized", "settled", "post14", "target_spread")


def parse_time(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"invalid timestamp: {value!r}")
    return pd.Timestamp(ts)


def validate_entry(entry: dict) -> list[str]:
    errors: list[str] = []
    required = ("name", "source_id", "target_day", "publish_time", "forecast_origin", "quantile_levels", "columns")
    for key in required:
        if key not in entry:
            errors.append(f"missing:{key}")
    if errors:
        return errors
    try:
        target = parse_time(entry["target_day"])
        publish = parse_time(entry["publish_time"])
        origin = parse_time(entry["forecast_origin"])
        # Entry-level origin must not be later than D-1 14:00.
        cutoff = target.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        if origin > cutoff:
            errors.append("forecast_origin_after_D1_1400")
        if publish > origin:
            errors.append("publish_after_forecast_origin")
        if target.normalize() >= FINAL_HOLDOUT_START:
            errors.append("fresh_final_holdout_in_source")
    except ValueError as exc:
        errors.append(str(exc))
    try:
        levels = [float(x) for x in entry["quantile_levels"]]
        if not levels or any(not 0.0 < x < 1.0 for x in levels) or len(set(levels)) != len(levels):
            errors.append("invalid_quantile_levels")
    except (TypeError, ValueError):
        errors.append("invalid_quantile_levels")
    columns = [str(x) for x in entry.get("columns", [])]
    if not columns:
        errors.append("empty_quantile_columns")
    joined = " ".join([str(entry.get("name", "")), str(entry.get("source_id", "")), *columns]).lower()
    if any(token in joined for token in FORBIDDEN):
        errors.append("forbidden_realized_or_actual_token")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube"))
    ap.add_argument("--source-manifest", type=Path, default=None, help="JSON with {forecasts:[...]}; omit when no external source is available")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cube, out = args.cube.resolve(), args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    registry = json.loads((cube / "feature_registry.json").read_text(encoding="utf-8"))
    uncertainty = []
    for item in registry.get("features", []):
        text = " ".join(str(item.get(k, "")) for k in ("feature", "source", "availability", "transform")).lower()
        if any(t in text for t in ("quantile", "uncert", "q10", "q90")):
            uncertainty.append(item)
    entries = []
    source_present = args.source_manifest is not None
    if source_present:
        payload = json.loads(args.source_manifest.resolve().read_text(encoding="utf-8"))
        entries = payload.get("forecasts", [])
        if not isinstance(entries, list):
            raise RuntimeError("source manifest forecasts must be a list")
    checks = [{"name": e.get("name", ""), "errors": validate_entry(e), "eligible": not validate_entry(e)} for e in entries]
    eligible = [x for x in checks if x["eligible"]]
    status = "STRICT/PASS" if eligible else ("STRICT/PASS_NO_SOURCE" if not source_present else "REJECTED_CONTRACT")
    result = {
        "status": status,
        "eligible": bool(eligible),
        "source_manifest_present": source_present,
        "candidate_count": len(entries),
        "eligible_count": len(eligible),
        "current_cube_uncertainty_like_count": len(uncertainty),
        "current_cube_true_external_quantile_count": 0,
        "required_forecast_origin": "D-1 14:00",
        "required_publish_time": "publish_time <= forecast_origin",
        "required_fields": ["source_id", "target_day", "publish_time", "forecast_origin", "quantile_levels", "columns"],
        "target_day_actual_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "checks": checks,
        "decision": "eligible source may enter a shadow-only B pilot" if eligible else "do not add external probabilistic fundamentals; current F6 remains pseudo historical-error bands",
    }
    (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(checks).to_json(out / "candidate_checks.json", orient="records", force_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if status != "REJECTED_CONTRACT" else 1


if __name__ == "__main__":
    main()
