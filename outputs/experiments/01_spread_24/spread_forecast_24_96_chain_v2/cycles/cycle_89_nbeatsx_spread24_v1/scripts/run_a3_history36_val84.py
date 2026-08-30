from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import date, timedelta
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
from nbeatsx_spread.evaluation.metrics import compute_metrics  # noqa: E402
from nbeatsx_spread.evaluation.panel import daily_metric_row, paired_daily_delta  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import sha256_file, sha256_json, source_tree_hash  # noqa: E402
from run_b0_extended_panel import baseline_day_rows, read_csv, write_csv  # noqa: E402
from run_business_backtest import run_one  # noqa: E402
from run_history_window_study import load_matrix, summarize_gradient, target_arrays, target_days_from_matrix  # noqa: E402


def _summary(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    """Aggregate A3 daily rows into pooled and daily-macro metrics."""
    predictions = np.asarray([value for row in rows for value in row["_predictions"]], dtype=float)
    targets = np.asarray([value for row in rows for value in row["_targets"]], dtype=float)
    pooled = compute_metrics(predictions, targets)
    macro: dict[str, Any] = {"day_count": len(rows)}
    for name, key in {"raw": "raw", "balanced": "balanced", "positive_recall": "positive_recall", "negative_recall": "negative_recall", "MAE": "MAE"}.items():
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        macro[name] = {"mean": float(np.mean(values)), "median": float(np.median(values)), "std": float(np.std(values)), "p10": float(np.percentile(values, 10)), "worst": float(np.min(values)), "best": float(np.max(values))}
    values = np.asarray([float(row["minority_recall"]) for row in rows], dtype=float)
    macro["minority_recall"] = {"mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)), "std": float(np.nanstd(values))}
    return pooled, macro


def _reference_daily(path: Path, candidate: str, days: list[str]) -> dict[str, dict[str, Any]]:
    """Read the already-frozen A0/A2 daily comparison rows only."""
    rows = {row["target_day"]: row for row in csv.DictReader(path.open(encoding="utf-8")) if row["history_id"] == candidate}
    if set(rows) != set(days):
        raise RuntimeError(f"{candidate}_DEV14_REFERENCE_INCOMPLETE")
    for row in rows.values():
        for key in ("raw", "balanced", "MAE"):
            row[key] = float(row[key])
    return rows


def _gate_summary(micro: dict[str, float], macro: dict[str, Any], daily: list[dict[str, Any]], a2_daily: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the five pre-registered A3 scientific conditions."""
    collapse_days = sum(bool(row["majority_collapse"]) for row in daily)
    macro_minority = float(macro["minority_recall"]["mean"])
    a2_minority = float(np.mean([float(row["minority_recall"]) for row in a2_daily.values()]))
    gates = {
        "G1_daily_macro_balanced": float(macro["balanced"]["mean"]) >= 0.5227,
        "G2_majority_collapse_days": collapse_days <= 5,
        "G3_micro_raw": float(micro["direction_accuracy"]) >= 0.6586,
        "G4_MAE": float(micro["mae"]) <= 89.10,
        "G5_negative_or_macro_minority": float(micro["negative_recall"]) >= 0.5070 or macro_minority >= a2_minority + 0.03,
    }
    return {"gates": gates, "passed": int(sum(gates.values())), "collapse_days": collapse_days, "macro_minority": macro_minority, "a2_macro_minority": a2_minority}


def main() -> int:
    """Run A3 only on the pre-registered DEV14 panel and write comparison evidence."""
    parser = argparse.ArgumentParser(description="A3 36-month history with 84-day chronological validation")
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--run-dir", type=Path, default=CYCLE / "runs/history_window_study/36m_val84")
    parser.add_argument("--postprocess-only", action="store_true", help="reuse completed A3 day runs after a post-training failure")
    args = parser.parse_args()
    matrix = load_matrix()
    days = target_days_from_matrix(matrix)
    a3_config = json.loads((CYCLE / "configs/a3_history36_val84.json").read_text(encoding="utf-8"))
    if a3_config["id"] != "A3_HISTORY36_VAL84" or a3_config["training_history_months"] != 36 or a3_config["validation_history_days"] != 84:
        raise RuntimeError("A3_CONFIG_NOT_FROZEN")
    base = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    candidate = copy.deepcopy(base)
    candidate.update({"training_history_months": 36, "validation_history_days": 84, "history_study_candidate": "A3_HISTORY36_VAL84"})
    if candidate["training_history_months"] != 36 or candidate["validation_history_days"] != 84:
        raise RuntimeError("A3_RUNTIME_CONFIG_INVALID")
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    holdout = audit_holdout_registry(days, registry)
    if not holdout.passed:
        raise RuntimeError(f"FINAL_HOLDOUT_REGISTRY_FAIL: {holdout.detail}")
    source = CanonicalHourlySource.from_csv(args.data)
    decision = select_device("cuda_if_deterministic_else_cpu", seed=int(candidate["training"]["seed"]))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    daily: list[dict[str, Any]] = []
    gradients: list[dict[str, Any]] = []
    counts: list[dict[str, Any]] = []
    majority: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for day in days:
        if args.postprocess_only:
            manifest = json.loads((args.run_dir / day / "manifest.json").read_text(encoding="utf-8"))
        else:
            manifest = run_one(day, source, candidate, args.run_dir, registry, device=decision.device)
        if manifest["leakage_status"] != "STRICT/PASS" or manifest["target_day_sample_count"] != 24:
            raise RuntimeError(f"A3/{day}: strict OOS manifest gate failed")
        if date.fromisoformat(manifest["training_last_day"]) > date.fromisoformat(day) - timedelta(days=2):
            raise RuntimeError(f"A3/{day}: training cutoff exceeds D-2")
        prediction, target = target_arrays(args.run_dir, day)
        row = daily_metric_row(day, prediction, target)
        row.update({"history_id": "A3_HISTORY36_VAL84", "history_months": 36, "validation_history_days": 84, "_predictions": prediction.tolist(), "_targets": target.tolist()})
        daily.append(row)
        split = json.loads((args.run_dir / day / "split_manifest.json").read_text(encoding="utf-8"))
        counts.append({"history_id": "A3_HISTORY36_VAL84", "history_months": 36, "validation_history_days": 84, "target_day": day, "calibration_start": split["calibration_start"], "calibration_end": split["calibration_end"], "calibration_count": len(split["train_days"]) + len(split["validation_days"]), "train_count": len(split["train_days"]), "validation_count": len(split["validation_days"]), "training_last_day": split["training_last_day"]})
        gradient = summarize_gradient(args.run_dir, [day]); gradient.update({"history_id": "A3_HISTORY36_VAL84", "history_months": 36, "target_day": day}); gradients.append(gradient)
        majority.append({key: row[key] for key in ("history_id", "history_months", "target_day", "actual_positive_rate", "predicted_positive_rate", "majority_baseline", "raw", "raw_minus_majority", "minority_recall", "majority_collapse")})
        transitions.append({key: row[key] for key in ("history_id", "history_months", "target_day", "actual_sign_switch_count", "predicted_sign_switch_count", "transition_precision", "transition_recall", "transition_f1")})
    micro, macro = _summary(daily)
    reference = CYCLE / "runs/history_window_study/comparison/history_window_daily_metrics.csv"
    a0_daily, a2_daily = _reference_daily(reference, "A0_HISTORY_9M", days), _reference_daily(reference, "A2_HISTORY_36M", days)
    paired_a0, paired_a2 = [], []
    for row in daily:
        paired_a0.append({"history_id": row["history_id"], **paired_daily_delta(row, a0_daily[row["target_day"]])})
        paired_a2.append({"history_id": row["history_id"], **paired_daily_delta(row, a2_daily[row["target_day"]])})
    comparator = (CYCLE / "../cycle_88_numeric_spread_da_minus_rt/runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv").resolve()
    if not comparator.exists():
        raise FileNotFoundError(comparator)
    comparator_sha = sha256_file(comparator)
    baseline_daily = {}
    for day in days:
        prediction, target = baseline_day_rows(comparator, day)
        baseline_daily[day] = daily_metric_row(day, prediction, target)
    paired_cycle = [{"history_id": row["history_id"], "baseline_model": "Cycle88_LGBM_v2_full_F0_F9", "baseline_source_path": str(comparator), "baseline_source_sha256": comparator_sha, **paired_daily_delta(row, baseline_daily[row["target_day"]])} for row in daily]
    h34 = []
    for offset in range(1, 35):
        prediction, target = [], []
        for day in days:
            rows = read_csv(args.run_dir / day / "predictions.csv")
            selected = [item for item in rows if int(item["h34_offset"]) == offset]
            if len(selected) != 1:
                raise RuntimeError(f"A3/{day}: H34 offset {offset} missing or duplicated")
            prediction.append(float(selected[0]["prediction"])); target.append(float(selected[0]["target"]))
        p, y = np.asarray(prediction), np.asarray(target)
        h34.append({"history_id": "A3_HISTORY36_VAL84", "history_months": 36, "h34_offset": offset, "section": "bridge" if offset <= 10 else "D-day", "bias": float(np.mean(p - y)), **compute_metrics(p, y)})
    gate = _gate_summary(micro, macro, daily, a2_daily)
    transition_means = {key: float(np.nanmean([float(row[key]) for row in transitions])) for key in ("transition_precision", "transition_recall", "transition_f1")}
    bridge_h34 = [row for row in h34 if row["section"] == "bridge"]
    scored_h34 = [row for row in h34 if row["section"] == "D-day"]
    bridge_mae = float(np.mean([float(row["mae"]) for row in bridge_h34]))
    scored_mae = float(np.mean([float(row["mae"]) for row in scored_h34]))
    best_offset = min(scored_h34, key=lambda row: float(row["mae"]))["h34_offset"]
    worst_offset = max(scored_h34, key=lambda row: float(row["mae"]))["h34_offset"]
    comparison = args.run_dir.parent / "comparison_a3"
    comparison.mkdir(parents=True, exist_ok=True)
    write_csv(comparison / "A3_daily_metrics.csv", [{key: value for key, value in row.items() if not key.startswith("_")} for row in daily])
    (comparison / "A3_micro_metrics.json").write_text(json.dumps(micro, ensure_ascii=False, indent=2), encoding="utf-8")
    (comparison / "A3_macro_metrics.json").write_text(json.dumps(macro, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(comparison / "A3_majority_collapse.csv", majority)
    write_csv(comparison / "A3_transition_metrics.csv", transitions)
    write_csv(comparison / "A3_h34_offset_metrics.csv", h34)
    write_csv(comparison / "A3_training_sample_counts.csv", counts)
    write_csv(comparison / "A3_gradient_summary.csv", gradients)
    write_csv(comparison / "A3_paired_vs_A0.csv", paired_a0)
    write_csv(comparison / "A3_paired_vs_A2.csv", paired_a2)
    write_csv(comparison / "A3_paired_vs_Cycle88.csv", paired_cycle)
    report = ["# A3_HISTORY36_VAL84 Review", "", "status: active", "scope: DEV14 only", "leakage_status: STRICT/PASS", "", "## Frozen change", "", "A3 changes only the validation width relative to A2: 36-month history and latest 84 legal chronological validation days. Model, feature, loss, optimizer, seed, device and precision remain frozen. No feature, rollout, directional-loss or tuning stage was run.", "", "## Metrics", "", f"- Micro: raw={micro['direction_accuracy']:.6f}, positive_recall={micro['positive_recall']:.6f}, negative_recall={micro['negative_recall']:.6f}, balanced={micro['balanced_accuracy']:.6f}, MAE={micro['mae']:.4f}, RMSE={micro['rmse']:.4f}.", f"- Daily macro balanced={macro['balanced']['mean']:.6f} (std={macro['balanced']['std']:.6f}); daily macro minority recall={macro['minority_recall']['mean']:.6f}.", f"- Majority-collapse days={gate['collapse_days']}; transition precision/recall/F1 means={transition_means['transition_precision']:.6f}/{transition_means['transition_recall']:.6f}/{transition_means['transition_f1']:.6f}.", f"- H34 offset MAE/bias/direction is in `A3_h34_offset_metrics.csv`; bridge MAE={bridge_mae:.4f}, scored D-day MAE={scored_mae:.4f}, best scored offset={best_offset}, worst scored offset={worst_offset}.", "", "## Five pre-registered scientific gates", ""]
    if gate["passed"] >= 4 and gate["collapse_days"] < 7:
        status = "ROBUST_LONG_HISTORY_PASS"
    elif micro["direction_accuracy"] >= 0.6586 and micro["mae"] <= 89.10:
        status = "LONG_HISTORY_MAGNITUDE_ONLY"
    else:
        status = "LONG_HISTORY_REGRESSION"
    def mean_delta(rows: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))
    report.extend([f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in gate["gates"].items()])
    report.extend([
        f"- Gates passed: **{gate['passed']}/5**.",
        "",
        "## Paired same-date deltas (A3 minus comparator)",
        f"- Versus A0: Δraw={mean_delta(paired_a0, 'delta_raw'):.6f}, Δbalanced={mean_delta(paired_a0, 'delta_balanced'):.6f}, ΔMAE={mean_delta(paired_a0, 'delta_MAE'):.4f}.",
        f"- Versus A2: Δraw={mean_delta(paired_a2, 'delta_raw'):.6f}, Δbalanced={mean_delta(paired_a2, 'delta_balanced'):.6f}, ΔMAE={mean_delta(paired_a2, 'delta_MAE'):.4f}.",
        f"- Versus Cycle88: Δraw={mean_delta(paired_cycle, 'delta_raw'):.6f}, Δbalanced={mean_delta(paired_cycle, 'delta_balanced'):.6f}, ΔMAE={mean_delta(paired_cycle, 'delta_MAE'):.4f}; source SHA256 `{comparator_sha}`.",
        "",
        "## Decision",
        "",
        f"Final status: **{status}**. A3 does not authorize feature-stage execution in this run.",
    ])
    (comparison / "A3_review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {"status": "A3_HISTORY36_VAL84_COMPLETE", "decision_status": status, "target_days": days, "candidate": "A3_HISTORY36_VAL84", "training_history_months": 36, "validation_history_days": 84, "device_decision": decision.__dict__, "precision": "float32", "seed": 42, "leakage_status": "STRICT/PASS", "gate_summary": gate, "micro": micro, "macro": macro, "comparator_path": str(comparator), "comparator_sha256": comparator_sha, "source_data_sha256": sha256_file(source.path) if source.path else None, "source_code_sha256": source_tree_hash(CYCLE / "src"), "config_sha256": sha256_json(a3_config), "comparison_root": str(comparison), "forbidden_stages_not_run": ["feature_stage", "rollout_stage", "directional_loss", "hyperparameter_search"]}
    (args.run_dir / "A3_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "decision_status": status, "gates_passed": gate["passed"], "device": decision.device}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
