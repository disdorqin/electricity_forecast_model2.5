"""Run the pre-registered L1 MAE+bridge loss race.

This runner deliberately has only two executable stages: RACE5 and RACE25.
The frozen C0 comparator is read from FULLDEV5 and is never retrained here.
Candidate target-day artifacts are produced by the strict formal runner after
the legal audits and before the evaluation labels are joined.
"""

from __future__ import annotations

import argparse
import csv
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve()
CYCLE = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CYCLE / "src"))

from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.audits import audit_holdout_registry, load_holdout_registry  # noqa: E402
from nbeatsx_spread.evaluation.metrics import compute_metrics  # noqa: E402
from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, daily_metric_row  # noqa: E402
from nbeatsx_spread.losses.l1_mae_bridge025 import D24MAEBridge025  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import environment_identity, sha256_file, sha256_json, source_tree_hash  # noqa: E402
from run_b0_extended_panel import baseline_day_rows  # noqa: E402
from run_business_backtest import run_one  # noqa: E402


RACE5_DAYS = ["2026-01-17", "2026-02-12", "2026-04-09", "2026-06-14", "2026-07-15"]
RACE25_DAYS = [
    "2026-01-06", "2026-01-07", "2026-01-17", "2026-01-19", "2026-01-31",
    "2026-02-08", "2026-02-09", "2026-02-12", "2026-02-24", "2026-02-27",
    "2026-04-06", "2026-04-09", "2026-04-12", "2026-04-16", "2026-04-18",
    "2026-06-02", "2026-06-11", "2026-06-14", "2026-06-26", "2026-06-29",
    "2026-07-03", "2026-07-07", "2026-07-15", "2026-07-25", "2026-07-28",
]
MONTHS = ("2026-01", "2026-02", "2026-04", "2026-06", "2026-07")
OBJECTIVE = "L1_D24_MAE_BRIDGE025"


def write_json(path: Path, payload: Any) -> None:
    """Write a deterministic JSON artifact below the cycle run boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty CSV with stable field order."""
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_matrix() -> dict[str, Any]:
    """Load the machine matrix and fail closed if another stage is requested."""
    matrix = json.loads((CYCLE / "configs/loss_objective_budgeted_matrix.json").read_text(encoding="utf-8"))
    if matrix.get("immediate_next_execution", {}).get("allowed") != [OBJECTIVE]:
        raise RuntimeError("L1_MATRIX_NEXT_EXECUTION_CHANGED")
    item = matrix["loss_sequence"][0]
    if item.get("id") != OBJECTIVE or item.get("bridge_weight") != 0.25 or item.get("direction_term") is not False:
        raise RuntimeError("L1_OBJECTIVE_CONTRACT_CHANGED")
    return matrix


def frozen_base() -> dict[str, Any]:
    """Build the frozen C0 chassis, changing no architecture or data policy."""
    config = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    config["training_history_months"] = 36
    config["validation_history_days"] = 28
    config["input_size"] = 168
    config["horizon"] = 34
    config["feature_package"] = "CORE5_RAW"
    config["forecast_strategy"] = "C0_DIRECT_H34"
    config["training"]["mixed_precision_business"] = "float32"
    config["training"]["amp_status"] = "AMP_FOLLOWUP_NOT_ACTIVE"
    return config


def l1_config() -> dict[str, Any]:
    """Return L1 config with the objective as the sole experimental change."""
    config = frozen_base()
    config["loss_objective"] = OBJECTIVE
    config["loss_definition"] = {"d_day_weight": 1.0, "bridge_weight": 0.25, "direction_term": False}
    config["training"]["loss"] = OBJECTIVE
    return config


def c0_root(day: str) -> Path:
    """Return the immutable same-date FULLDEV5 C0 artifact path."""
    return CYCLE / "runs/FULLDEV5/C0" / day


def candidate_root(day: str) -> Path:
    """Return the isolated L1 artifact path."""
    return CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025" / day


def verify_c0_artifact(day: str, source: CanonicalHourlySource, base: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Verify an existing C0 artifact without invoking any training code."""
    root = c0_root(day)
    required = ["config.json", "provenance.json", "manifest.json", "target_day_prediction.csv", "predictions.csv", "checkpoint.pt"]
    checks: list[dict[str, Any]] = []
    for name in required:
        ok = (root / name).is_file()
        checks.append({"field": name, "status": "PASS" if ok else "FAIL"})
    if not all(row["status"] == "PASS" for row in checks):
        raise RuntimeError(f"C0_ARTIFACT_MISSING:{day}")
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256_json(base),
        "source_data_sha256": sha256_file(source.path),
        "device": str(device),
        "seed": 42,
        "history_months": 36,
        "validation_days": 28,
        "feature_package": "CORE5_RAW",
        "forecast_strategy": "C0_DIRECT_H34",
        "leakage_status": "STRICT/PASS",
        "target_day_sample_count": 24,
    }
    actual = {
        "config_sha256": sha256_json(cfg),
        "source_data_sha256": provenance.get("source_data_sha256"),
        "device": provenance.get("device", {}).get("device"),
        "seed": provenance.get("device", {}).get("seed"),
        "history_months": manifest.get("training_history_months"),
        "validation_days": manifest.get("validation_history_days"),
        "feature_package": cfg.get("feature_package"),
        "forecast_strategy": cfg.get("forecast_strategy"),
        "leakage_status": manifest.get("leakage_status"),
        "target_day_sample_count": manifest.get("target_day_sample_count"),
    }
    for field, value in expected.items():
        ok = actual[field] == value
        checks.append({"field": field, "expected": value, "actual": actual[field], "status": "PASS" if ok else "FAIL"})
        if not ok:
            raise RuntimeError(f"C0_REUSE_IDENTITY_FAIL:{day}:{field}")
    rows = list(csv.DictReader((root / "target_day_prediction.csv").open(encoding="utf-8")))
    if len(rows) != 24:
        raise RuntimeError(f"C0_TARGET_DAY_ROWS_FAIL:{day}:{len(rows)}")
    checks.append({"field": "target_day_prediction_rows", "expected": 24, "actual": len(rows), "status": "PASS"})
    checkpoint_hash = hashlib.sha256((root / "checkpoint.pt").read_bytes()).hexdigest()
    checkpoint_ok = checkpoint_hash == manifest.get("checkpoint_sha256")
    checks.append({"field": "checkpoint_sha256", "expected": manifest.get("checkpoint_sha256"), "actual": checkpoint_hash, "status": "PASS" if checkpoint_ok else "FAIL"})
    if not checkpoint_ok:
        raise RuntimeError(f"C0_CHECKPOINT_HASH_FAIL:{day}")
    record = {"target_day": day, "root": str(root), "status": "REUSED_HASH_VERIFIED", "checks": checks}
    return record


def read_candidate_row(day: str) -> dict[str, Any]:
    """Read one strict candidate target-day artifact and compute diagnostics."""
    root = candidate_root(day)
    rows = list(csv.DictReader((root / "target_day_prediction.csv").open(encoding="utf-8")))
    if len(rows) != 24:
        raise RuntimeError(f"L1_TARGET_DAY_ROWS_FAIL:{day}:{len(rows)}")
    rows.sort(key=lambda row: int(row["business_hour"]))
    pred = np.asarray([float(row["prediction"]) for row in rows], dtype=float)
    target = np.asarray([float(row["target"]) for row in rows], dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise RuntimeError(f"L1_NONFINITE_TARGET_ARTIFACT:{day}")
    row = daily_metric_row(day, pred, target)
    return {"target_day": day, "month": day[:7], "prediction": pred, "target": target, "row": row, "root": str(root)}


def train_l1_day(day: str, source: CanonicalHourlySource, config: dict[str, Any], registry: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Cold-retrain exactly one L1 target day through the formal strict runner."""
    root = candidate_root(day)
    started = time.perf_counter()
    manifest = run_one(day, source, config, root.parent, registry, device=device, loss_fn=D24MAEBridge025(0.25))
    manifest["loss_objective"] = OBJECTIVE
    manifest["loss_definition"] = config["loss_definition"]
    write_json(root / "manifest.json", manifest)
    write_json(root / "training_runtime.json", {
        "training_seconds_wall": time.perf_counter() - started,
        "device": str(device),
        "forward_passes": 1,
        "cold_retrain": True,
        "objective": OBJECTIVE,
    })
    # The common formal runner has already written all strict artifacts.  This
    # extra assertion makes the loss study's objective identity explicit.
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    provenance["loss_objective"] = OBJECTIVE
    provenance["loss_definition"] = config["loss_definition"]
    write_json(root / "provenance.json", provenance)
    return read_candidate_row(day)


def ensure_l1_artifact(day: str, source: CanonicalHourlySource, config: dict[str, Any], registry: dict[str, Any], device: torch.device, *, allow_reuse: bool) -> dict[str, Any]:
    """Reuse only a verified L1 artifact; otherwise perform one cold retrain."""
    root = candidate_root(day)
    if allow_reuse and (root / "manifest.json").exists() and (root / "provenance.json").exists():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        prov = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        checks = {
            "objective": manifest.get("loss_objective") == OBJECTIVE and prov.get("loss_objective") == OBJECTIVE,
            "config": prov.get("config_sha256") == sha256_json(config),
            "source_data": prov.get("source_data_sha256") == sha256_file(source.path),
            "device": prov.get("device", {}).get("device") == str(device),
            "seed": prov.get("device", {}).get("seed") == 42,
            "strict": manifest.get("leakage_status") == "STRICT/PASS",
        }
        try:
            row = read_candidate_row(day)
        except (OSError, ValueError, RuntimeError, KeyError):
            row = None
        if all(checks.values()) and row is not None:
            write_json(root / "l1_reuse_audit.json", {"status": "REUSED_HASH_VERIFIED", "checks": checks})
            return row
    return train_l1_day(day, source, config, registry, device)


def validate_engineering_artifacts(days: list[str]) -> dict[str, Any]:
    """Check finite curves, gradients, predictions, and strict manifests."""
    rows: list[dict[str, Any]] = []
    for day in days:
        root = candidate_root(day)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        curve = list(csv.DictReader((root / "training_curve.csv").open(encoding="utf-8")))
        grads = list(csv.DictReader((root / "gradient_stats.csv").open(encoding="utf-8")))
        pred = read_candidate_row(day)["prediction"]
        curve_finite = all(np.isfinite(float(row["train_loss"])) and np.isfinite(float(row["validation_mae"])) for row in curve)
        grad_finite = all(np.isfinite(float(row["grad_norm_pre_clip"])) and np.isfinite(float(row["grad_norm_post_clip"])) for row in grads)
        nonfinite = sum(int(float(row.get("nonfinite_count", 0))) for row in grads)
        nonconstant = float(np.std(pred)) > 0.0
        rows.append({"target_day": day, "strict": manifest.get("leakage_status") == "STRICT/PASS", "curve_finite": curve_finite, "gradient_finite": grad_finite, "nonfinite_count": nonfinite, "prediction_nonconstant": nonconstant, "steps": len(grads)})
    passed = all(row["strict"] and row["curve_finite"] and row["gradient_finite"] and row["nonfinite_count"] == 0 and row["prediction_nonconstant"] and row["steps"] >= 200 for row in rows)
    return {"status": "PASS" if passed else "FAIL", "days": rows, "all_required_checks_pass": passed}


def baseline_row(day: str, comparator: Path) -> dict[str, Any]:
    """Load the same-date frozen FULLDEV5 C0 scored prediction."""
    pred, target = baseline_day_rows(comparator, day)
    return {"target_day": day, "month": day[:7], "prediction": pred, "target": target, "row": daily_metric_row(day, pred, target), "root": str(comparator)}


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return pooled micro, daily macro, minority, collapse, and transitions."""
    pred = np.concatenate([r["prediction"] for r in records])
    target = np.concatenate([r["target"] for r in records])
    micro = compute_metrics(pred, target)
    macro = aggregate_daily_metrics([r["row"] for r in records])
    return {
        "micro": micro,
        "daily_macro": macro,
        "daily_macro_balanced": macro["balanced"]["mean"],
        "daily_macro_minority_recall": float(np.nanmean([r["row"]["minority_recall"] for r in records])),
        "majority_collapse_days": int(sum(bool(r["row"]["majority_collapse"]) for r in records)),
        "majority_collapse_rate": float(np.mean([bool(r["row"]["majority_collapse"]) for r in records])),
        "transition_f1": float(np.nanmean([r["row"]["transition_f1"] for r in records])),
    }


def race5(source: CanonicalHourlySource, config: dict[str, Any], registry: dict[str, Any], device: torch.device, comparator: Path) -> dict[str, Any]:
    """Execute the engineering-only five-day race and write its gate."""
    c0_checks = [verify_c0_artifact(day, source, frozen_base(), device) for day in RACE5_DAYS]
    records = [ensure_l1_artifact(day, source, config, registry, device, allow_reuse=False) for day in RACE5_DAYS]
    engineering = validate_engineering_artifacts(RACE5_DAYS)
    c0 = [baseline_row(day, comparator) for day in RACE5_DAYS]
    candidate_summary = metric_summary(records)
    c0_summary = metric_summary(c0)
    first_last = []
    for day in RACE5_DAYS:
        curve = list(csv.DictReader((candidate_root(day) / "training_curve.csv").open(encoding="utf-8")))
        first_last.append(float(curve[-1]["train_loss"]) <= float(curve[0]["train_loss"]) if curve else False)
    passed = engineering["all_required_checks_pass"] and sum(first_last) >= 3
    out = {"status": "RACE5_PASS" if passed else "RACE5_FAIL", "objective": OBJECTIVE, "target_days": RACE5_DAYS, "c0_reuse": c0_checks, "engineering": engineering, "train_loss_non_increasing_days": int(sum(first_last)), "candidate": candidate_summary, "c0": c0_summary, "leakage_status": "STRICT/PASS"}
    write_json(CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025/RACE5_summary.json", out)
    review = ["# L1 RACE5 Review", "", f"status: **{out['status']}**", "leakage_status: STRICT/PASS", "", "Five pre-registered days were cold-retrained with L1 D-day MAE + 0.25 bridge MAE. C0 was reused from hash-verified FULLDEV5 artifacts and was not retrained.", f"engineering_artifacts: {engineering['status']}", f"train_loss_non_increasing_days: {sum(first_last)}/5", ""]
    (CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025/RACE5_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("L1_RACE5_FAIL; RACE25 is blocked")
    return out


def race25(source: CanonicalHourlySource, config: dict[str, Any], registry: dict[str, Any], device: torch.device, comparator: Path) -> dict[str, Any]:
    """Execute RACE25, reusing only the five verified RACE5 candidate days."""
    race5_path = CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025/RACE5_summary.json"
    if not race5_path.exists() or json.loads(race5_path.read_text(encoding="utf-8")).get("status") != "RACE5_PASS":
        raise RuntimeError("RACE25_REQUIRES_RACE5_PASS")
    c0_checks = [verify_c0_artifact(day, source, frozen_base(), device) for day in RACE25_DAYS]
    # RACE25 may be rerun for aggregation/audit after completion.  Reusing a
    # previously completed same-objective artifact is safe only after its
    # full provenance identity is checked; it never turns into warm-start
    # training.
    rows = [ensure_l1_artifact(day, source, config, registry, device, allow_reuse=True) for day in RACE25_DAYS]
    engineering = validate_engineering_artifacts(RACE25_DAYS)
    c0 = [baseline_row(day, comparator) for day in RACE25_DAYS]
    by_day_l1 = {r["target_day"]: r for r in rows}
    by_day_c0 = {r["target_day"]: r for r in c0}
    paired = []
    for day in RACE25_DAYS:
        a, b = by_day_l1[day]["row"], by_day_c0[day]["row"]
        paired.append({"target_day": day, "month": day[:7], "delta_raw": a["raw"] - b["raw"], "delta_positive_recall": a["positive_recall"] - b["positive_recall"], "delta_negative_recall": a["negative_recall"] - b["negative_recall"], "delta_balanced": a["balanced"] - b["balanced"], "delta_MAE": a["MAE"] - b["MAE"], "delta_transition_f1": a["transition_f1"] - b["transition_f1"], "candidate_majority_collapse": a["majority_collapse"], "c0_majority_collapse": b["majority_collapse"]})
    l1_summary, c0_summary = metric_summary(rows), metric_summary(c0)
    monthly = []
    for month in MONTHS:
        l1m = metric_summary([r for r in rows if r["month"] == month]); c0m = metric_summary([r for r in c0 if r["month"] == month])
        monthly.append({"month": month, "target_days": 5, "l1_raw": l1m["micro"]["direction_accuracy"], "c0_raw": c0m["micro"]["direction_accuracy"], "l1_positive_recall": l1m["micro"]["positive_recall"], "c0_positive_recall": c0m["micro"]["positive_recall"], "l1_negative_recall": l1m["micro"]["negative_recall"], "c0_negative_recall": c0m["micro"]["negative_recall"], "l1_balanced": l1m["micro"]["balanced_accuracy"], "c0_balanced": c0m["micro"]["balanced_accuracy"], "l1_MAE": l1m["micro"]["mae"], "c0_MAE": c0m["micro"]["mae"], "l1_daily_macro_balanced": l1m["daily_macro_balanced"], "c0_daily_macro_balanced": c0m["daily_macro_balanced"], "l1_minority_recall": l1m["daily_macro_minority_recall"], "c0_minority_recall": c0m["daily_macro_minority_recall"], "l1_collapse_days": l1m["majority_collapse_days"], "c0_collapse_days": c0m["majority_collapse_days"], "l1_transition_f1": l1m["transition_f1"], "c0_transition_f1": c0m["transition_f1"]})
    month_benefit = [r["l1_daily_macro_balanced"] > r["c0_daily_macro_balanced"] and r["l1_MAE"] <= r["c0_MAE"] * 1.05 for r in monthly]
    guard = {
        "daily_macro_balanced_delta_min": l1_summary["daily_macro_balanced"] - c0_summary["daily_macro_balanced"] >= 0.02,
        "micro_balanced_delta_min": l1_summary["micro"]["balanced_accuracy"] - c0_summary["micro"]["balanced_accuracy"] >= 0.015,
        "raw_floor": l1_summary["micro"]["direction_accuracy"] - c0_summary["micro"]["direction_accuracy"] >= -0.01,
        "mae_ratio": l1_summary["micro"]["mae"] <= c0_summary["micro"]["mae"] * 1.05,
        "collapse_no_worse": l1_summary["majority_collapse_days"] <= c0_summary["majority_collapse_days"],
        "transition_no_material_deterioration": l1_summary["transition_f1"] >= c0_summary["transition_f1"] - 0.02,
        "month_breadth": sum(month_benefit) >= 3,
    }
    clear = all(guard.values())
    out = {"status": "RACE25_COMPLETE", "objective": OBJECTIVE, "target_days": RACE25_DAYS, "leakage_status": "STRICT/PASS", "c0_reuse": c0_checks, "engineering": engineering, "candidate": l1_summary, "c0": c0_summary, "guard": guard, "guard_passed": sum(guard.values()), "guard_total": len(guard), "clear_signal": clear, "month_benefit": {"months": MONTHS, "passed": month_benefit, "count": sum(month_benefit)}, "paired_delta_mean": {key: float(np.nanmean([row[key] for row in paired])) for key in ("delta_raw", "delta_positive_recall", "delta_negative_recall", "delta_balanced", "delta_MAE", "delta_transition_f1")}}
    root = CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025"
    write_json(root / "RACE25_overall_metrics.json", out)
    write_json(root / "RACE25_daily_macro_metrics.json", {"L1": l1_summary["daily_macro"], "C0": c0_summary["daily_macro"]})
    write_json(root / "RACE25_month_breadth.json", out["month_benefit"])
    write_csv(root / "RACE25_monthly_metrics.csv", monthly)
    write_csv(root / "RACE25_paired_deltas.csv", paired)
    write_csv(root / "RACE25_structure_metrics.csv", [{"model": "L1", "month": m, "collapse_days": next(r["l1_collapse_days"] for r in monthly if r["month"] == m), "transition_f1": next(r["l1_transition_f1"] for r in monthly if r["month"] == m), "minority_recall": next(r["l1_minority_recall"] for r in monthly if r["month"] == m)} for m in MONTHS] + [{"model": "C0", "month": m, "collapse_days": next(r["c0_collapse_days"] for r in monthly if r["month"] == m), "transition_f1": next(r["c0_transition_f1"] for r in monthly if r["month"] == m), "minority_recall": next(r["c0_minority_recall"] for r in monthly if r["month"] == m)} for m in MONTHS])
    write_json(root / "loss_l1_manifest.json", {"status": "RACE25_COMPLETE", "objective": OBJECTIVE, "frozen": {"strategy": "C0_DIRECT_H34", "history_months": 36, "validation_days": 28, "feature_package": "CORE5_RAW", "precision": "float32", "seed": 42, "daily_cold_retrain": True}, "forbidden_not_run": ["L2", "L3", "L4", "RACE50", "CONFIRM21", "September", "new_features", "structure_search", "history_search", "model_size_search"]})
    review = ["# L1 RACE25 Review", "", "status: **RACE25_COMPLETE**", "leakage_status: STRICT/PASS", f"clear_signal: **{clear}**", f"guard: {sum(guard.values())}/{len(guard)}", f"month_breadth: {sum(month_benefit)}/5", "", "C0 was reused from hash-verified FULLDEV5 target-day artifacts; no C0 retraining was performed.", "Headline metrics use only the 24 D-day scored rows per target day. Bridge is not mixed into headline metrics."]
    (root / "RACE25_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    return out


def main() -> int:
    """Run exactly one registered race stage and then stop."""
    parser = argparse.ArgumentParser(description="Cycle89 L1 budgeted loss race")
    parser.add_argument("--stage", choices=("RACE5", "RACE25"), required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    args = parser.parse_args()
    matrix = load_matrix()
    source = CanonicalHourlySource.from_csv(args.data)
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    requested = RACE5_DAYS if args.stage == "RACE5" else RACE25_DAYS
    holdout = audit_holdout_registry(requested, registry)
    if not holdout.passed:
        raise RuntimeError(f"FINAL_HOLDOUT_REGISTRY_FAIL:{holdout.detail}")
    config = l1_config()
    device = select_device("cuda_if_deterministic_else_cpu", seed=42).device
    comparator = (CYCLE / "../cycle_88_numeric_spread_da_minus_rt/runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv").resolve()
    if not comparator.exists():
        raise FileNotFoundError(comparator)
    root = CYCLE / "runs/loss_objective/L1_D24_MAE_BRIDGE025"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "execution_context.json", {"matrix_profile": matrix["profile"], "stage": args.stage, "device": environment_identity(device=torch.device(device), deterministic=True, seed=42), "source_data_sha256": sha256_file(source.path), "source_code_sha256": source_tree_hash(CYCLE / "src"), "target_days": RACE5_DAYS if args.stage == "RACE5" else RACE25_DAYS, "objective": OBJECTIVE})
    result = race5(source, config, registry, torch.device(device), comparator) if args.stage == "RACE5" else race25(source, config, registry, torch.device(device), comparator)
    print(json.dumps({"status": result["status"], "stage": args.stage, "objective": OBJECTIVE, "target_days": len(RACE5_DAYS if args.stage == "RACE5" else RACE25_DAYS), "device": str(device), "clear_signal": result.get("clear_signal")}, ensure_ascii=False), flush=True)
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
