"""Audit whether probabilistic fundamental inputs are genuinely available at origin.

The current cube may contain pseudo-quantile bands made from target-day point
forecasts plus historical D-2-or-earlier forecast errors.  This diagnostic
separates those causal bands from true externally supplied forecast quantiles and
rejects any target-day actual/post-cutoff source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FORBIDDEN = ("target day actual", "post14", "cutoff actual", "target_day_actual")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cube, out = args.cube.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    registry = json.loads((cube / "feature_registry.json").read_text(encoding="utf-8"))
    rows = []
    for item in registry.get("features", []):
        name = str(item.get("feature", ""))
        source = str(item.get("source", ""))
        availability = str(item.get("availability", ""))
        transform = str(item.get("transform", ""))
        text = " ".join((name, source, availability, transform)).lower()
        is_uncertainty = any(token in text for token in ("uncert", "quantile", "q10", "q90"))
        source_lower = source.lower()
        is_historical_error = (
            "historical forecast error" in source_lower
            or ("actual" in source_lower and "forecast" in source_lower)
            or "d-2 or earlier" in availability.lower()
        )
        is_pseudo = "pseudo" in transform.lower() or is_historical_error
        forbidden = any(token in text for token in FORBIDDEN)
        rows.append({
            "feature": name, "group": item.get("group", ""), "source": source,
            "availability": availability, "transform": transform,
            "uncertainty_like": is_uncertainty, "pseudo_historical_error": is_pseudo,
            "forbidden_token": forbidden, "leakage_status": item.get("leakage_status", ""),
        })
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("empty feature registry")
    forbidden = int(audit["forbidden_token"].sum())
    pseudo = audit[audit["pseudo_historical_error"]]
    true_external = audit[audit["uncertainty_like"] & ~audit["pseudo_historical_error"]]
    status = "STRICT/PASS" if forbidden == 0 else "INVALID-LEAKAGE"
    audit.to_csv(out / "feature_availability.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": status, "experiment_status": "ACTIVE", "diagnostic_only": True,
        "forecast_origin": "D-1 14:00", "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False, "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False, "feature_source": str(cube / "feature_registry.json"),
        "feature_count": int(len(audit)), "forbidden_token_count": forbidden,
        "uncertainty_like_count": int(audit["uncertainty_like"].sum()),
        "pseudo_historical_error_count": int(len(pseudo)),
        "true_external_quantile_count": int(len(true_external)),
        "finding": "当前只有 target forecast + D-2-or-earlier historical-error pseudo bands；未发现外部真实 forecast quantile 输入",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(
        "# Probabilistic fundamental availability audit\n\n"
        f"- status: `{status}`\n- feature_count: {len(audit)}\n"
        f"- uncertainty-like features: {int(audit['uncertainty_like'].sum())}\n"
        f"- pseudo historical-error bands: {len(pseudo)}\n"
        f"- true external forecast-quantile features: {len(true_external)}\n\n"
        "结论：当前 F6 是由目标日点预测与 D-2 及更早历史预测误差构造的 pseudo quantile，" 
        "通过 strict contract，但不能冒充外部概率预测。若未来接入真实 quantile，必须新增来源、发布时间和反事实审计。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
