from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
CYCLE = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CYCLE / "src"))

from nbeatsx_spread.audits import audit_holdout_registry, load_holdout_registry  # noqa: E402
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.evaluation.metrics import compute_metrics, metric_by_forecast_offset  # noqa: E402
from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, baseline_row_from_arrays, daily_metric_row, paired_daily_delta  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import artifact_reuse_audit, environment_identity, git_identity, sha256_file, sha256_json, source_tree_hash  # noqa: E402
from run_business_backtest import run_one  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty tabular artifact with stable column order."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    # Cycle88 exports use a UTF-8 BOM; normalize it at the comparator
    # boundary rather than silently losing the target_day key.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def model_day_arrays(rows: list[dict[str, str]], *, h34: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(rows, key=lambda row: int(row["h34_offset"])) if h34 else sorted(rows, key=lambda row: int(row["business_hour"]))
    return (
        np.asarray([float(row["prediction"]) for row in rows], dtype=float),
        np.asarray([float(row["target"]) for row in rows], dtype=float),
    )


def baseline_day_rows(path: Path, day: str, tree_config: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(path) if row.get("target_day") == day and (tree_config is None or row.get("config") == tree_config)]
    if len(rows) != 24:
        raise RuntimeError(f"comparator {path} has {len(rows)} rows for {day}, expected 24")
    rows = sorted(rows, key=lambda row: int(row["hour_business"]))
    prediction_key = "y_pred" if tree_config is not None else "predicted_spread_DA_minus_RT"
    target_key = "y_true" if tree_config is not None else "target_spread_DA_minus_RT"
    return (
        np.asarray([float(row[prediction_key]) for row in rows], dtype=float),
        np.asarray([float(row[target_key]) for row in rows], dtype=float),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the immutable 14-day Cycle89 MAE panel.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--run-dir", type=Path, default=CYCLE / "runs/b0_extended_panel_14d")
    args = parser.parse_args()
    panel_config = json.loads((CYCLE / "configs/b0_extended_validation_panel.json").read_text(encoding="utf-8"))
    config = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    days = list(panel_config["target_days"])
    if len(days) != 14 or len(set(days)) != 14:
        raise RuntimeError("PANEL_DATE_REGISTRY_INVALID")
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    holdout = audit_holdout_registry(days, registry)
    if not holdout.passed:
        raise RuntimeError(f"FINAL_HOLDOUT_REGISTRY_FAIL: {holdout.detail}")
    source = CanonicalHourlySource.from_csv(args.data)
    decision = select_device(panel_config["device_policy"], seed=int(config["training"]["seed"]))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for day in days:
        day_dir = args.run_dir / day
        # The first panel attempt completed all cold retrains before a
        # comparator BOM parsing error. Reuse only a fully materialized,
        # strict target-day artifact; otherwise train this date from scratch.
        required = (day_dir / "manifest.json", day_dir / "target_day_prediction.csv", day_dir / "predictions.csv")
        reuse_rows = artifact_reuse_audit(
            day_dir,
            config=config,
            source_path=source.path,
            source_code_hash=source_tree_hash(CYCLE / "src"),
            device=decision.device,
            seed=int(config["training"]["seed"]),
        )
        if all(path.exists() for path in required) and all(row["status"] == "MATCH" for row in reuse_rows):
            manifests.append(json.loads((day_dir / "manifest.json").read_text(encoding="utf-8")))
        else:
            manifests.append(run_one(day, source, config, args.run_dir, registry, device=decision.device))

    panel_predictions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    all_pred, all_target = [], []
    for day in days:
        day_dir = args.run_dir / day
        rows34 = read_csv(day_dir / "predictions.csv")
        rows24 = read_csv(day_dir / "target_day_prediction.csv")
        pred24, target24 = model_day_arrays(rows24)
        row = daily_metric_row(day, pred24, target24)
        row["model"] = "NBEATSx_MAE"
        daily_rows.append(row)
        all_pred.extend(pred24.tolist())
        all_target.extend(target24.tolist())
        for raw in rows34:
            panel_predictions.append({
                "target_day": day,
                "model": "NBEATSx_MAE",
                "h34_offset": int(raw["h34_offset"]),
                "section": raw["section"],
                "business_hour": int(raw["business_hour"]),
                "prediction": float(raw["prediction"]),
                "target": float(raw["target"]),
            })

    micro = compute_metrics(np.asarray(all_pred), np.asarray(all_target))
    macro = aggregate_daily_metrics(daily_rows)
    write_csv(args.run_dir / "daily_metrics.csv", daily_rows)
    write_csv(args.run_dir / "panel_predictions.csv", panel_predictions)

    comparator_specs = [
        ("Cycle88_LGBM_v2_full_F0_F9", (CYCLE / panel_config["primary_comparator"]["predictions"]).resolve(), None),
        ("Cycle88_LGBM_baseline_2026", (CYCLE / panel_config["secondary_comparator"]["predictions"]).resolve(), None),
    ]
    trees_path = (CYCLE / panel_config["june_optional_comparator"]["predictions"]).resolve()
    if trees_path.exists():
        comparator_specs.append(("Cycle88_trees_220_June", trees_path, "trees_220"))
    comparison_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    baseline_daily: dict[str, dict[str, dict[str, Any]]] = {}
    for name, path, tree_config in comparator_specs:
        if not path.exists():
            raise FileNotFoundError(path)
        baseline_daily[name] = {}
        compare_days = days if tree_config is None else [day for day in days if day.startswith("2026-06")]
        for day in compare_days:
            pred, target = baseline_day_rows(path, day, tree_config)
            base_row = baseline_row_from_arrays(day, pred, target, name)
            comparison_rows.append(base_row)
            baseline_daily[name][day] = base_row
            delta = paired_daily_delta(daily_rows[days.index(day)], base_row)
            paired_rows.append({"model": name, **delta})

    # Add paired deltas to the NBEATSx rows while retaining one row per model/day.
    primary_name = comparator_specs[0][0]
    for row in daily_rows:
        base = baseline_daily[primary_name][row["target_day"]]
        delta = paired_daily_delta(row, base)
        row.update(delta)
        comparison_rows.append(row)
    required_order = [
        "target_day", "model", "raw", "positive_recall", "negative_recall",
        "balanced", "MAE", "actual_positive_rate", "predicted_positive_rate",
        "majority_baseline", "raw_minus_majority", "balanced_minus_0p5",
        "delta_raw", "delta_balanced", "delta_MAE",
    ]
    for row in comparison_rows:
        row["balanced_minus_0p5"] = float(row["balanced"] - 0.5)
        for field in ("delta_raw", "delta_balanced", "delta_MAE"):
            row.setdefault(field, "")
    write_csv(args.run_dir / "baseline_comparison_daily.csv", [
        {key: row.get(key, "") for key in required_order} for row in comparison_rows
    ])

    baseline_summary: dict[str, Any] = {}
    for name, path, tree_config in comparator_specs:
        pred, target = [], []
        compare_days = days if tree_config is None else [day for day in days if day.startswith("2026-06")]
        for day in compare_days:
            p, y = baseline_day_rows(path, day, tree_config)
            pred.extend(p.tolist())
            target.extend(y.tolist())
        rows = [baseline_daily[name][day] for day in compare_days]
        baseline_summary[name] = {
            "path": str(path),
            "micro_hourly": compute_metrics(np.asarray(pred), np.asarray(target)),
            "daily_macro": aggregate_daily_metrics(rows),
            "paired_delta_summary": aggregate_daily_metrics([
                {"raw": x["delta_raw"], "balanced": x["delta_balanced"], "MAE": x["delta_MAE"],
                 "positive_recall": x["delta_raw"], "negative_recall": x["delta_raw"]}
                for x in paired_rows if x["model"] == name
            ]),
        }
    baseline_summary["NBEATSx_MAE"] = {"micro_hourly": micro, "daily_macro": macro}
    (args.run_dir / "baseline_comparison_summary.json").write_text(json.dumps(baseline_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run_dir / "micro_hourly_metrics.json").write_text(json.dumps({"NBEATSx_MAE": micro, **{k: v["micro_hourly"] for k, v in baseline_summary.items() if k != "NBEATSx_MAE"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.run_dir / "macro_daily_metrics.json").write_text(json.dumps({"NBEATSx_MAE": macro, **{k: v["daily_macro"] for k, v in baseline_summary.items() if k != "NBEATSx_MAE"}}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.run_dir / "majority_collapse_diagnostics.csv", [
        {key: row[key] for key in (
            "target_day", "actual_positive_rate", "predicted_positive_rate",
            "majority_baseline", "raw", "raw_minus_majority", "minority_recall",
            "actual_sign_switch_count", "predicted_sign_switch_count", "majority_collapse",
        )}
        for row in daily_rows
    ])
    collapse_rate = float(sum(bool(row["majority_collapse"]) for row in daily_rows) / len(daily_rows))

    offset_pred = np.asarray([[float(row["prediction"]) for row in sorted(panel_predictions, key=lambda x: x["h34_offset"]) if row["target_day"] == day] for day in days])
    offset_target = np.asarray([[float(row["target"]) for row in sorted(panel_predictions, key=lambda x: x["h34_offset"]) if row["target_day"] == day] for day in days])
    offset_rows = metric_by_forecast_offset(offset_pred, offset_target)
    for row in offset_rows:
        row["h34_offset"] = row.pop("offset")
        row["n"] = row["sample_count"]
        row["business_hour"] = row["h34_offset"] + 14 if row["h34_offset"] <= 10 else row["h34_offset"] - 10
    write_csv(args.run_dir / "metric_by_h34_offset.csv", offset_rows)
    months = {"2026-06": [d for d in days if d.startswith("2026-06")], "2026-07": [d for d in days if d.startswith("2026-07")]}
    monthly_rows = []
    for month, month_days in months.items():
        for model_name, path, tree_config in [("NBEATSx_MAE", None, None), *comparator_specs]:
            if tree_config is not None and month != "2026-06":
                continue
            pred, target = [], []
            for day in month_days:
                if path is None:
                    p, y = model_day_arrays(read_csv(args.run_dir / day / "target_day_prediction.csv"))
                else:
                    p, y = baseline_day_rows(path, day, tree_config)
                pred.extend(p.tolist()); target.extend(y.tolist())
            m = compute_metrics(np.asarray(pred), np.asarray(target))
            monthly_rows.append({"month": month, "model": model_name, **m})
    write_csv(args.run_dir / "monthly_partial_metrics.csv", monthly_rows)
    primary_micro = baseline_summary[primary_name]["micro_hourly"]
    primary_macro = baseline_summary[primary_name]["daily_macro"]
    paired_raw = micro["direction_accuracy"] - primary_micro["direction_accuracy"]
    paired_balanced = macro["balanced"]["mean"] - primary_macro["balanced"]["mean"]
    if paired_raw > 0 and paired_balanced > 0 and collapse_rate <= 0.20:
        interpretation = "STRONG_SIGNAL"
    elif paired_raw >= 0 or paired_balanced >= 0:
        interpretation = "MIXED_SIGNAL"
    else:
        interpretation = "WEAK_SIGNAL"

    root_provenance = {
        "panel_config_sha256": sha256_json(panel_config),
        "business_config_sha256": sha256_json(config),
        "source_data_sha256": sha256_file(source.path) if source.path else None,
        "source_code_sha256": source_tree_hash(CYCLE / "src"),
        "git": git_identity(ROOT),
        "device": environment_identity(device=__import__("torch").device(decision.device), deterministic=True, seed=int(config["training"]["seed"])),
        "device_decision": decision.__dict__,
        "target_days": days,
        "comparator_paths": {name: str(path) for name, path, _ in comparator_specs},
    }
    (args.run_dir / "provenance.json").write_text(json.dumps(root_provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "status": "B0_EXTENDED_PANEL_COMPLETE",
        "panel_profile": panel_config["profile"],
        "target_days": days,
        "date_registry_sha256": sha256_json(days),
        "cold_retrain_per_day": True,
        "target_day_sample_count": 24,
        "scored_points": len(all_pred),
        "leakage_status": "STRICT/PASS",
        "device_decision": decision.__dict__,
        "target_day_manifests": manifests,
        "interpretation": interpretation,
        "majority_collapse_day_count": int(sum(bool(row["majority_collapse"]) for row in daily_rows)),
        "majority_collapse_day_rate": collapse_rate,
        "primary_paired_delta_micro_raw": paired_raw,
        "primary_paired_delta_daily_macro_balanced": paired_balanced,
    }
    (args.run_dir / "extended_panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "days": days, "micro": micro, "device": decision.device}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
