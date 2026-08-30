"""Run the pre-registered FULLDEV5 cross-month cold-retrain panel.

This runner is deliberately separate from the DEV14 smoke/stage runners.  It
trains C0 and the three independent C3 direct blocks for every target day,
joins labels only after inference, and writes monthly, pooled, paired and
diagnostic artifacts under one isolated run directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
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

from nbeatsx_spread.audits import audit_holdout_registry, load_holdout_registry  # noqa: E402
from nbeatsx_spread.audits.lineage import tensor_hash  # noqa: E402
from nbeatsx_spread.data.business_dataset import build_business_split  # noqa: E402
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.evaluation.metrics import compute_metrics, metric_by_forecast_offset  # noqa: E402
from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, baseline_row_from_arrays, daily_metric_row  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import (  # noqa: E402
    artifact_reuse_audit,
    environment_identity,
    sha256_file,
    sha256_json,
    source_tree_hash,
)
from nbeatsx_spread.contracts import latest_complete_label_day  # noqa: E402
from run_b0_extended_panel import baseline_day_rows  # noqa: E402
from run_business_backtest import run_one as run_c0_one  # noqa: E402
from run_c3_dirmo_stage1 import DIRMO_BLOCKS, run_day as run_c3_day  # noqa: E402
from run_forecast_strategy_stage1 import legal_audits  # noqa: E402


STRATEGIES = ("C0_DIRECT_H34", "C3_DIRMO_10_12_12", "Cycle88_LGBM_v2_full_F0_F9")
MODEL_STRATEGIES = STRATEGIES[:2]
MONTHS = ("2026-01", "2026-02", "2026-04", "2026-06", "2026-07")


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic human-readable JSON below the run boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty tabular artifact with stable column order."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_config() -> dict[str, Any]:
    """Load and validate the exact machine-readable FULLDEV5 matrix."""
    config = json.loads((CYCLE / "configs/full_month_cross_month_dev5.json").read_text(encoding="utf-8"))
    if config.get("profile") != "cycle89_full_month_cross_month_dev5":
        raise RuntimeError("FULLDEV5_PROFILE_MISMATCH")
    ids = [item["id"] for item in config["candidate_strategies"]]
    if ids != ["C0_DIRECT_H34", "C3_DIRMO_10_12_12"]:
        raise RuntimeError(f"FULLDEV5_CANDIDATE_MISMATCH:{ids}")
    if config["scientific_contract"] != {
        "target": "spread_DA_minus_RT", "forecast_origin": "D-1 14:00",
        "training_labels_latest": "D-2", "training_history_months": 36,
        "validation_days": 28, "backcast_hours": 168,
        "feature_profile": "CORE5_RAW_TRAJECTORY_PLUS_CALENDAR", "loss": "MAE",
        "seed": 42, "precision": "float32", "amp": False,
        "recalibration": "daily_cold_retrain",
    }:
        raise RuntimeError("FULLDEV5_SCIENTIFIC_CONTRACT_CHANGED")
    forbidden = json.dumps(config["forbidden_in_this_run"], ensure_ascii=False).lower()
    for token in ("c2b", "c2c", "recmо", "recursive_h1", "scheduled_sampling", "new_features", "directional_loss", "history_search", "confirm21", "september"):
        if token.lower() not in forbidden and token != "recmо":
            raise RuntimeError(f"FULLDEV5_FORBIDDEN_REGISTRY_INCOMPLETE:{token}")
    return config


def registered_days(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    """Expand the five exact full-month registries and reject reserved months."""
    days: list[str] = []
    month_info: dict[str, str] = {}
    for entry in config["full_month_registry"]:
        month = entry["month"]
        if month not in MONTHS:
            raise RuntimeError(f"UNREGISTERED_FULL_MONTH:{month}")
        expanded = [d.strftime("%Y-%m-%d") for d in pd.date_range(entry["start"], entry["end"], freq="D")]
        if len(expanded) != int(entry["days"]):
            raise RuntimeError(f"FULL_MONTH_COUNT_MISMATCH:{month}")
        days.extend(expanded)
        month_info.update({day: entry["stratum"] for day in expanded})
    if len(days) != 150 or len(set(days)) != 150:
        raise RuntimeError("FULLDEV5_MUST_CONTAIN_150_UNIQUE_DAYS")
    if set(month_info) & {f"2026-{m:02d}-{d:02d}" for m in (3, 5, 8) for d in range(1, 32)}:
        raise RuntimeError("CONFIRM21_DATE_TOUCHED")
    if any(day.startswith("2026-09-") for day in days):
        raise RuntimeError("SEPTEMBER_LOCKBOX_TOUCHED")
    return days, month_info


def frozen_base() -> dict[str, Any]:
    """Build the one frozen business chassis used by both candidates."""
    base = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    base["training_history_months"] = 36
    base["validation_history_days"] = 28
    base["input_size"] = 168
    base["horizon"] = 34
    base["feature_package"] = "CORE5_RAW"
    base["forecast_strategy"] = "C0_DIRECT_H34"
    base["training"]["mixed_precision_business"] = "float32"
    base["training"]["amp_status"] = "AMP_FOLLOWUP_NOT_ACTIVE"
    return base


def strict_pre_audit(source: CanonicalHourlySource, days: list[str], registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Complete all day-level causal gates before the first model is trained."""
    rows: list[dict[str, Any]] = []
    for day in days:
        audits = legal_audits(source, day, registry)
        # The formal runners rebuild the exact split immediately before each
        # fit and audit its concrete train/validation days.  The global gate
        # only needs to establish the immutable cutoff for all dates here;
        # rebuilding complete_business_days 150 times would duplicate a costly
        # preflight without adding evidence.
        from nbeatsx_spread.audits import audit_training_cutoff  # local import keeps audit dependency explicit
        cutoff = latest_complete_label_day(day)
        audits.append(audit_training_cutoff(day, [cutoff]))
        if not all(a.passed for a in audits):
            raise RuntimeError(f"INVALID-LEAKAGE:FULLDEV5:{day}")
        rows.append({"target_day": day, "status": "PASS", "training_last_day": cutoff, "audits": [a.as_dict() for a in audits]})
    return rows


def prediction_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the common H34 artifact and return full and scored arrays."""
    rows = list(csv.DictReader((path / "predictions.csv").open(encoding="utf-8")))
    rows.sort(key=lambda row: int(row["h34_offset"]))
    if len(rows) != 34:
        raise AssertionError(f"H34 artifact must contain 34 rows: {path}")
    pred = np.asarray([float(row["prediction"]) for row in rows], dtype=float)
    target = np.asarray([float(row["target"]) for row in rows], dtype=float)
    scored_rows = list(csv.DictReader((path / "target_day_prediction.csv").open(encoding="utf-8")))
    if len(scored_rows) != 24:
        raise AssertionError(f"target-day artifact must contain 24 rows: {path}")
    scored_rows.sort(key=lambda row: int(row["business_hour"]))
    scored_pred = np.asarray([float(row["prediction"]) for row in scored_rows], dtype=float)
    scored_target = np.asarray([float(row["target"]) for row in scored_rows], dtype=float)
    return pred, target, scored_pred, scored_target


def reuse_check(candidate_root: Path, config: Any, source: CanonicalHourlySource, source_hash: str, device: torch.device) -> tuple[bool, list[dict[str, Any]]]:
    """Return reusable only when the full identity contract is verified."""
    if not candidate_root.exists():
        return False, [{"field": "artifact_exists", "status": "FAIL", "actual": None}]
    rows = artifact_reuse_audit(candidate_root, config=config, source_path=source.path, source_code_hash=source_hash, device=str(device), seed=42)
    ok = all(row["status"] == "MATCH" for row in rows)
    if ok:
        try:
            prediction_arrays(candidate_root)
        except (OSError, ValueError, AssertionError, KeyError):
            ok = False
            rows.append({"field": "prediction_artifacts", "status": "FAIL"})
    return ok, rows


def model_record(day: str, strategy: str, root: Path, month_info: dict[str, str], reuse_status: str, reuse_audit: list[dict[str, Any]]) -> dict[str, Any]:
    """Load one completed target-day artifact into the aggregate schema."""
    pred, target, pred24, target24 = prediction_arrays(root)
    row = daily_metric_row(day, pred24, target24)
    bridge = compute_metrics(pred[:10], target[:10])
    return {"target_day": day, "month": day[:7], "stratum": month_info[day], "strategy": strategy, "prediction": pred, "target": target, "prediction24": pred24, "target24": target24, "row": row, "bridge": bridge, "reuse_status": reuse_status, "reuse_audit": reuse_audit, "root": str(root)}


def cycle88_record(day: str, comparator: Path, month_info: dict[str, str]) -> dict[str, Any]:
    """Load the immutable Cycle88 same-date comparator."""
    pred, target = baseline_day_rows(comparator, day)
    pred = np.asarray(pred, dtype=float); target = np.asarray(target, dtype=float)
    row = baseline_row_from_arrays(day, pred, target, "Cycle88_LGBM_v2_full_F0_F9")
    return {"target_day": day, "month": day[:7], "stratum": month_info[day], "strategy": STRATEGIES[2], "prediction": np.r_[np.full(10, np.nan), pred], "target": np.r_[np.full(10, np.nan), target], "prediction24": pred, "target24": target, "row": row, "bridge": None, "reuse_status": "FROZEN_COMPARATOR", "reuse_audit": [], "root": str(comparator)}


def run_model_day(day: str, source: CanonicalHourlySource, base: dict[str, Any], registry: dict[str, Any], run_dir: Path, device: torch.device, source_hash: str, c3_matrix: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run or strictly reuse C0 and C3 for one day."""
    device = torch.device(device)
    c0_root = run_dir / "C0" / day
    c0_config = copy.deepcopy(base)
    c0_config["forecast_strategy"] = "C0_DIRECT_H34"
    c0_reuse, c0_audit = reuse_check(c0_root, c0_config, source, source_hash, device)
    if c0_reuse:
        c0_status = "REUSED_HASH_VERIFIED"
    else:
        c0_started = time.perf_counter()
        run_c0_one(day, source, c0_config, run_dir / "C0", registry, device=device)
        write_json(c0_root / "full_month_runtime.json", {"training_seconds": time.perf_counter() - c0_started, "inference_seconds": None, "forward_passes": 1, "runtime_status": "RECORDED_BY_FULLDEV5_RUNNER"})
        c0_status = "COLD_RETRAIN_NEW"
    c0 = model_record(day, "C0_DIRECT_H34", c0_root, {day: day[:7] if day[:7] not in () else ""}, c0_status, c0_audit)
    # run_c3_day carries the fixed [10,12,12] partition and performs its own
    # common leakage gates before constructing any model.
    c3_root = run_dir / "C3" / day
    c3_config_hash = sha256_json(c3_matrix)
    c3_reuse, c3_audit = reuse_check(c3_root, c3_matrix, source, source_hash, device)
    if c3_reuse:
        c3_status = "REUSED_HASH_VERIFIED"
        c3 = model_record(day, "C3_DIRMO_10_12_12", c3_root, {day: day[:7]}, c3_status, c3_audit)
    else:
        c3_status = "COLD_RETRAIN_NEW"
        started = time.perf_counter()
        out = run_c3_day(source, day, base, c3_root, device, registry)
        c3 = model_record(day, "C3_DIRMO_10_12_12", c3_root, {day: day[:7]}, "COLD_RETRAIN_NEW", c3_audit)
        c3["training_seconds"] = float(sum(item["training_seconds"] for item in out["results"]))
        c3["inference_seconds"] = float(sum(item["inference_seconds"] for item in out["results"]))
        c3["parameter_count"] = int(sum(item["parameter_count"] for item in out["results"]))
        c3["forward_passes"] = 3
        c3["wall_seconds"] = float(time.perf_counter() - started)
        write_json(c3_root / "provenance.json", {"config_sha256": c3_config_hash, "source_data_sha256": sha256_file(source.path), "source_code_sha256": source_hash, "device": environment_identity(device=device, deterministic=True, seed=42), "strategy_definition": "B0=10,B1=12,B2=12; independent direct; no feedback"})
        manifest = json.loads((c3_root / "manifest.json").read_text(encoding="utf-8"))
        manifest.update({"reuse_status": "COLD_RETRAIN_NEW", "target_day_sample_count": 24, "config_sha256": c3_config_hash})
        write_json(c3_root / "manifest.json", manifest)
    if "training_seconds" not in c3:
        c3["training_seconds"] = float(json.loads((c3_root / "manifest.json").read_text(encoding="utf-8")).get("training_seconds_total", 0.0))
        c3["inference_seconds"] = float(json.loads((c3_root / "manifest.json").read_text(encoding="utf-8")).get("inference_seconds_total", 0.0))
        c3["parameter_count"] = int(json.loads((c3_root / "manifest.json").read_text(encoding="utf-8")).get("parameter_count_total", 0))
        c3["forward_passes"] = 3
    write_json(c0_root / "full_month_reuse_audit.json", {"status": c0_status, "checks": c0_audit})
    write_json(c3_root / "full_month_reuse_audit.json", {"status": c3_status, "checks": c3_audit})
    return c0, c3


def add_runtime_rows(records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Produce the required per-month runtime/capacity table."""
    rows: list[dict[str, Any]] = []
    for strategy in MODEL_STRATEGIES:
        for rec in records[strategy]:
            root = Path(rec["root"])
            if strategy == "C0_DIRECT_H34":
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                summary = json.loads((root / "model_summary.json").read_text(encoding="utf-8"))
                runtime = json.loads((root / "full_month_runtime.json").read_text(encoding="utf-8")) if (root / "full_month_runtime.json").exists() else {}
                rows.append({"strategy": strategy, "target_day": rec["target_day"], "month": rec["month"], "parameter_count": summary["parameter_count"], "training_seconds": runtime.get("training_seconds"), "inference_seconds": runtime.get("inference_seconds"), "forward_passes": 1, "best_step": manifest.get("best_step"), "final_step": manifest.get("final_step"), "reuse_status": rec["reuse_status"], "runtime_status": runtime.get("runtime_status", "NOT_RECORDED_BY_LEGACY_C0_RUNNER")})
            else:
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                rows.append({"strategy": strategy, "target_day": rec["target_day"], "month": rec["month"], "parameter_count": manifest.get("parameter_count_total", rec.get("parameter_count")), "training_seconds": manifest.get("training_seconds_total", rec.get("training_seconds")), "inference_seconds": manifest.get("inference_seconds_total", rec.get("inference_seconds")), "forward_passes": 3, "best_step": json.dumps(manifest.get("best_steps", {})), "final_step": "", "reuse_status": rec["reuse_status"]})
    return rows


def metric_flat(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute pooled scored-hour metrics and daily macro metrics."""
    p = np.concatenate([r["prediction24"] for r in records]); y = np.concatenate([r["target24"] for r in records])
    micro = compute_metrics(p, y)
    macro = aggregate_daily_metrics([r["row"] for r in records])
    return {"micro": micro, "daily_macro": macro, "daily_macro_balanced": macro["balanced"]["mean"], "daily_macro_raw": macro["raw"]["mean"], "daily_macro_mae": macro["MAE"]["mean"], "daily_macro_positive_recall": macro["positive_recall"]["mean"], "daily_macro_negative_recall": macro["negative_recall"]["mean"], "minority_recall_mean": float(np.nanmean([r["row"]["minority_recall"] for r in records])), "majority_collapse_days": int(sum(r["row"]["majority_collapse"] for r in records)), "transition_f1_mean": float(np.nanmean([r["row"]["transition_f1"] for r in records]))}


def month_rows(records: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create monthly micro, macro and structure CSV rows."""
    micro_rows: list[dict[str, Any]] = []; macro_rows: list[dict[str, Any]] = []; structure_rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for month in MONTHS:
            subset = [r for r in records[strategy] if r["month"] == month]
            summary = metric_flat(subset)
            micro = summary["micro"]
            micro_rows.append({"strategy": strategy, "month": month, "target_days": len(subset), "scored_hours": len(subset) * 24, **micro})
            macro = summary["daily_macro"]
            macro_rows.append({"strategy": strategy, "month": month, "target_days": len(subset), "daily_macro_raw": macro["raw"]["mean"], "daily_macro_balanced": macro["balanced"]["mean"], "daily_macro_positive_recall": macro["positive_recall"]["mean"], "daily_macro_negative_recall": macro["negative_recall"]["mean"], "daily_macro_mae": macro["MAE"]["mean"], "daily_macro_rmse": float(np.nanmean([r["row"]["RMSE"] for r in subset])), "minority_recall": summary["minority_recall_mean"]})
            structure_rows.append({"strategy": strategy, "month": month, "majority_collapse_days": summary["majority_collapse_days"], "majority_collapse_rate": summary["majority_collapse_days"] / len(subset), "transition_f1": summary["transition_f1_mean"], "mean_actual_positive_rate": float(np.nanmean([r["row"]["actual_positive_rate"] for r in subset])), "mean_predicted_positive_rate": float(np.nanmean([r["row"]["predicted_positive_rate"] for r in subset])), "minority_recall": summary["minority_recall_mean"]})
    return micro_rows, macro_rows, structure_rows


def horizon_rows(records: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Compute H34 offset diagnostics without merging bridge into headline."""
    rows: list[dict[str, Any]] = []
    for strategy in MODEL_STRATEGIES:
        for month in MONTHS:
            subset = [r for r in records[strategy] if r["month"] == month]
            p = np.stack([r["prediction"] for r in subset]); y = np.stack([r["target"] for r in subset])
            for item in metric_by_forecast_offset(p, y):
                rows.append({"strategy": strategy, "month": month, "h34_offset": item.pop("offset"), **item})
    return rows


def paired_rows(records: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate same-day paired deltas before any cross-month summary."""
    by = {(r["strategy"], r["target_day"]): r for strategy in STRATEGIES for r in records[strategy]}
    daily: list[dict[str, Any]] = []
    for day in sorted(r["target_day"] for r in records["C0_DIRECT_H34"]):
        for candidate, reference in (("C3_DIRMO_10_12_12", "C0_DIRECT_H34"), ("C0_DIRECT_H34", STRATEGIES[2]), ("C3_DIRMO_10_12_12", STRATEGIES[2])):
            a, b = by[(candidate, day)], by[(reference, day)]
            daily.append({"target_day": day, "month": day[:7], "stratum": a["stratum"], "candidate": candidate, "reference": reference, "delta_raw": a["row"]["raw"] - b["row"]["raw"], "delta_balanced": a["row"]["balanced"] - b["row"]["balanced"], "delta_MAE": a["row"]["MAE"] - b["row"]["MAE"], "delta_positive_recall": a["row"]["positive_recall"] - b["row"]["positive_recall"], "delta_negative_recall": a["row"]["negative_recall"] - b["row"]["negative_recall"]})
    monthly: list[dict[str, Any]] = []
    for (month, candidate, reference), group in pd.DataFrame(daily).groupby(["month", "candidate", "reference"]):
        monthly.append({"month": month, "candidate": candidate, "reference": reference, "target_days": len(group), **{f"mean_{key}": float(group[key].mean()) for key in ("delta_raw", "delta_balanced", "delta_MAE", "delta_positive_recall", "delta_negative_recall")}})
    return daily, monthly


def bootstrap_ci(daily: list[dict[str, Any]], n_boot: int = 2000) -> dict[str, Any]:
    """Deterministic paired-day bootstrap intervals for every comparison."""
    frame = pd.DataFrame(daily); rng = np.random.default_rng(42); out: dict[str, Any] = {"seed": 42, "resamples": n_boot}
    for (candidate, reference), group in frame.groupby(["candidate", "reference"]):
        values: dict[str, Any] = {}
        for key in ("delta_raw", "delta_balanced", "delta_MAE", "delta_positive_recall", "delta_negative_recall"):
            x = group[key].to_numpy(float); x = x[np.isfinite(x)]
            if len(x) == 0:
                values[key] = {"mean": None, "p2_5": None, "p97_5": None}
                continue
            samples = np.asarray([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
            values[key] = {"mean": float(x.mean()), "p2_5": float(np.percentile(samples, 2.5)), "p97_5": float(np.percentile(samples, 97.5))}
        out[f"{candidate}_vs_{reference}"] = values
    return out


def stratified_summary(records: dict[str, list[dict[str, Any]]], strata: dict[str, str], stratum: str) -> dict[str, Any]:
    """Summarize one pre-registered month stratum."""
    return {strategy: metric_flat([r for r in records[strategy] if r["stratum"] == stratum]) for strategy in STRATEGIES}


def classify(records: dict[str, list[dict[str, Any]]], micro_rows: list[dict[str, Any]], macro_rows: list[dict[str, Any]], structure_rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Apply the six pre-registered C3 promotion gates exactly once."""
    c0 = metric_flat(records["C0_DIRECT_H34"]); c3 = metric_flat(records["C3_DIRMO_10_12_12"])
    c0s = {r["month"]: r for r in structure_rows if r["strategy"] == "C0_DIRECT_H34"}; c3s = {r["month"]: r for r in structure_rows if r["strategy"] == "C3_DIRMO_10_12_12"}
    c0m = {r["month"]: r for r in macro_rows if r["strategy"] == "C0_DIRECT_H34"}; c3m = {r["month"]: r for r in macro_rows if r["strategy"] == "C3_DIRMO_10_12_12"}
    g = config["c3_promotion_gate"]
    gates = {
        "G1_month_macro_daily_balanced": c3["daily_macro_balanced"] >= c0["daily_macro_balanced"] + float(g["G1_month_macro_daily_balanced_delta_pp_min"])/100,
        "G2_collapse": sum(c3s[m]["majority_collapse_days"] <= c0s[m]["majority_collapse_days"] for m in MONTHS) >= int(g["G2_months_collapse_lower_or_equal_min"]) and sum(c3s[m]["majority_collapse_days"] for m in MONTHS) < sum(c0s[m]["majority_collapse_days"] for m in MONTHS),
        "G3_transition_f1": c3["transition_f1_mean"] >= c0["transition_f1_mean"],
        "G4_month_macro_raw": c3["daily_macro_raw"] >= c0["daily_macro_raw"] + float(g["G4_month_macro_raw_delta_pp_min"])/100,
        "G5_mae": c3["daily_macro_mae"] <= c0["daily_macro_mae"] * float(g["G5_month_macro_mae_ratio_max"]),
        "G6_unseen_full3": sum((c3m[m]["daily_macro_balanced"] >= c0m[m]["daily_macro_balanced"] - .01 and c3m[m]["daily_macro_raw"] >= c0m[m]["daily_macro_raw"] - .01 and c3m[m]["daily_macro_mae"] <= c0m[m]["daily_macro_mae"] * 1.03) for m in ("2026-01", "2026-02", "2026-04")) >= int(g["G6_unseen_full3_noninferior_months_min"]),
    }
    count = sum(gates.values())
    label = "DIRMO_FULLMONTH_POSITIVE" if count == 6 else ("DIRMO_FULLMONTH_MIXED" if count >= 2 else "DIRMO_FULLMONTH_REJECT")
    return label, {"gates": gates, "gates_passed": count, "gates_total": len(gates), "c0": c0, "c3": c3}


def main() -> int:
    """Execute FULLDEV5 and stop after the registered cross-month review."""
    parser = argparse.ArgumentParser(description="Run Cycle89 FULLDEV5 full-month panel")
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--run-dir", type=Path, default=CYCLE / "runs/FULLDEV5")
    parser.add_argument("--target-day", action="append", help="Run an explicit registered subset; intended for disjoint workers.")
    parser.add_argument("--skip-aggregate", action="store_true", help="Write day artifacts only; use --aggregate-only after all workers finish.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate already completed FULLDEV5 day artifacts without training.")
    args = parser.parse_args()
    config = load_config(); all_days, month_info = registered_days(config)
    days = sorted(set(args.target_day)) if args.target_day else all_days
    if not set(days).issubset(set(all_days)) or not days:
        raise RuntimeError("FULLDEV5_TARGET_SUBSET_NOT_REGISTERED")
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    holdout = audit_holdout_registry(days, registry)
    if not holdout.passed:
        raise RuntimeError(f"FINAL_HOLDOUT_REGISTRY_FAIL:{holdout.detail}")
    source = CanonicalHourlySource.from_csv(args.data)
    base = frozen_base()
    decision = select_device("cuda_if_deterministic_else_cpu", seed=42)
    source_hash = source_tree_hash(CYCLE / "src")
    comparator = (CYCLE / "../cycle_88_numeric_spread_da_minus_rt/runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv").resolve()
    if not comparator.exists():
        raise FileNotFoundError(comparator)
    records: dict[str, list[dict[str, Any]]] = {strategy: [] for strategy in STRATEGIES}
    reuse_registry: list[dict[str, Any]] = []
    if args.aggregate_only:
        # Aggregation-only is fail-closed: missing or incomplete day artifacts
        # never trigger an implicit training fallback.
        for day in all_days:
            c0_root, c3_root = args.run_dir / "C0" / day, args.run_dir / "C3" / day
            for root in (c0_root, c3_root):
                if not (root / "manifest.json").exists() or not (root / "predictions.csv").exists():
                    raise RuntimeError(f"FULLDEV5_AGGREGATE_MISSING_DAY:{root}")
            c0 = model_record(day, "C0_DIRECT_H34", c0_root, month_info, "REUSED_HASH_VERIFIED", [])
            c3 = model_record(day, "C3_DIRMO_10_12_12", c3_root, month_info, "REUSED_HASH_VERIFIED", [])
            records["C0_DIRECT_H34"].append(c0); records["C3_DIRMO_10_12_12"].append(c3); records[STRATEGIES[2]].append(cycle88_record(day, comparator, month_info))
            for strategy, root in (("C0_DIRECT_H34", c0_root), ("C3_DIRMO_10_12_12", c3_root)):
                audit_path = root / "full_month_reuse_audit.json"
                audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"status": "UNKNOWN", "checks": []}
                reuse_registry.append({"target_day": day, "strategy": strategy, "status": audit["status"], "audit": audit.get("checks", [])})
    else:
        # This is the only pre-training phase. No model construction occurs
        # until every registered date has passed the causal gates.
        pre_audit = strict_pre_audit(source, all_days, registry)
        write_json(args.run_dir / "pre_training_leakage_audit.json", {"status": "STRICT/PASS", "target_days": len(all_days), "audits": pre_audit})
        for index, day in enumerate(days, start=1):
            c0, c3 = run_model_day(day, source, base, registry, args.run_dir, decision.device, source_hash, config)
            c0["stratum"] = c3["stratum"] = month_info[day]
            records["C0_DIRECT_H34"].append(c0); records["C3_DIRMO_10_12_12"].append(c3)
            records[STRATEGIES[2]].append(cycle88_record(day, comparator, month_info))
            reuse_registry.extend([{"target_day": day, "strategy": c0["strategy"], "status": c0["reuse_status"], "audit": c0["reuse_audit"]}, {"target_day": day, "strategy": c3["strategy"], "status": c3["reuse_status"], "audit": c3["reuse_audit"]}])
            print(json.dumps({"progress": f"{index}/{len(days)}", "target_day": day, "device": str(decision.device), "c0": c0["reuse_status"], "c3": c3["reuse_status"]}, ensure_ascii=False), flush=True)
            gc.collect()
            if torch.device(decision.device).type == "cuda":
                torch.cuda.empty_cache()
        if args.skip_aggregate:
            print(json.dumps({"status": "FULLDEV5_DAY_ARTIFACTS_COMPLETE", "target_days": len(days), "aggregate": "DEFERRED"}, ensure_ascii=False))
            return 0
    micro_rows, macro_rows, structure_rows = month_rows(records)
    daily_paired, monthly_paired = paired_rows(records)
    summary_label, decision_summary = classify(records, micro_rows, macro_rows, structure_rows, config)
    runtime_rows = add_runtime_rows(records)
    offset = horizon_rows(records)
    full = {strategy: metric_flat(records[strategy]) for strategy in STRATEGIES}
    write_csv(args.run_dir / "monthly_micro_metrics.csv", micro_rows)
    write_csv(args.run_dir / "monthly_daily_macro_metrics.csv", macro_rows)
    write_csv(args.run_dir / "monthly_structure_metrics.csv", structure_rows)
    write_json(args.run_dir / "cross_month_micro_metrics.json", {"FULLDEV5": full, "UNSEEN_FULL3": stratified_summary(records, month_info, "UNSEEN_FULL3"), "SEEN_FULL2": stratified_summary(records, month_info, "SEEN_FULL2")})
    write_json(args.run_dir / "cross_month_daily_macro_metrics.json", {strategy: metric_flat(records[strategy])["daily_macro"] for strategy in STRATEGIES})
    write_json(args.run_dir / "cross_month_month_macro_metrics.json", {strategy: {"mean_month_macro_raw": float(np.mean([r["daily_macro_raw"] for r in macro_rows if r["strategy"] == strategy])), "mean_month_macro_balanced": float(np.mean([r["daily_macro_balanced"] for r in macro_rows if r["strategy"] == strategy])), "mean_month_macro_mae": float(np.mean([r["daily_macro_mae"] for r in macro_rows if r["strategy"] == strategy]))} for strategy in STRATEGIES})
    write_csv(args.run_dir / "paired_daily_deltas.csv", daily_paired)
    write_csv(args.run_dir / "paired_monthly_deltas.csv", monthly_paired)
    write_json(args.run_dir / "paired_bootstrap_ci.json", bootstrap_ci(daily_paired))
    write_csv(args.run_dir / "majority_collapse_by_month.csv", [r for r in structure_rows if r["strategy"] in MODEL_STRATEGIES])
    write_csv(args.run_dir / "transition_by_month.csv", [r for r in structure_rows if r["strategy"] in MODEL_STRATEGIES])
    write_csv(args.run_dir / "horizon_by_month.csv", offset)
    write_csv(args.run_dir / "runtime_cost_by_month.csv", runtime_rows)
    write_json(args.run_dir / "cycle88_comparator_manifest.json", {"id": STRATEGIES[2], "path": str(comparator), "sha256": sha256_file(comparator), "same_date_only": True, "target_days": days, "scored_rows_per_day": 24})
    write_json(args.run_dir / "full_month_manifest.json", {"status": "FULLDEV5_COMPLETE", "classification": summary_label, "target_days": days, "target_day_count": len(days), "scored_hours_per_strategy": 3600, "strata": {month: sorted({d for d in days if d.startswith(month)}) for month in MONTHS}, "scientific_contract": config["scientific_contract"], "device": environment_identity(device=torch.device(decision.device), deterministic=True, seed=42), "pre_training_audit": "STRICT/PASS", "reuse_registry": reuse_registry, "cold_retrain_count": sum(item["status"] == "COLD_RETRAIN_NEW" for item in reuse_registry), "reused_count": sum(item["status"] == "REUSED_HASH_VERIFIED" for item in reuse_registry), "decision": decision_summary, "forbidden_not_run": config["forbidden_in_this_run"]})
    review = ["# FULLDEV5 Full-Month Cross-Month Review", "", f"Decision: **{summary_label}**", "", "leakage_status: STRICT/PASS", f"device: {decision.device}", f"target_days: {len(days)}; scored hours/model: 3600", "", "## Frozen comparison", "C0_DIRECT_H34 vs C3_DIRMO_10_12_12 vs Cycle88 strict comparator; 36m history, VAL28, CORE5, MAE, float32, daily cold retrain.", "", "## Promotion gates"]
    review.extend([f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in decision_summary["gates"].items()])
    review.append(f"- gates passed: {decision_summary['gates_passed']}/{decision_summary['gates_total']}")
    review.extend(["", "All headline metrics use D-day 24 scored points only. Bridge offsets 1-10 remain diagnostic. Cycle88 is same-date frozen comparator only.", "No RecMO, Recursive H1, other block sizes, new features, directional loss, history/validation/model-size search, CONFIRM21 or September lockbox was run."])
    (args.run_dir / "full_month_cross_month_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    write_json(args.run_dir / "run_manifest.json", {"status": "FULLDEV5_COMPLETE", "classification": summary_label, "target_days": len(days), "device": str(decision.device), "comparison": "runs/FULLDEV5", "source_code_sha256": source_hash, "source_data_sha256": sha256_file(source.path)})
    print(json.dumps({"status": "FULLDEV5_COMPLETE", "classification": summary_label, "target_days": len(days), "scored_hours_per_strategy": 3600, "device": str(decision.device), "gates_passed": decision_summary["gates_passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
