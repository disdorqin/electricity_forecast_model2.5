"""Run the frozen DEV14 compact-feature rescue study.

Only F0, F1 and F2 are accepted.  Each target day is cold-retrained with the
36-month/28-day A2 chassis; target-day labels are joined only after forward.
"""
from __future__ import annotations

import argparse
import copy
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
from nbeatsx_spread.data.covariates import feature_names  # noqa: E402
from nbeatsx_spread.evaluation.metrics import compute_metrics  # noqa: E402
from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, baseline_row_from_arrays, daily_metric_row, paired_daily_delta  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import sha256_file, sha256_json, source_tree_hash  # noqa: E402
from run_b0_extended_panel import baseline_day_rows, read_csv, write_csv  # noqa: E402
from run_business_backtest import run_one  # noqa: E402


FEATURE_PROFILES = {
    "F0_A2_CORE5": "CORE5_RAW",
    "F1_A2_PHYSICAL_SHAPE": "PHYSICAL_SHAPE",
    "F2_A2_CAUSAL_PRICE_STATE": "CAUSAL_PRICE_STATE",
}
FORBIDDEN = {"F1_plus_F2", "forecast_error_state", "B208_full", "rollout", "directional_loss", "history_search", "validation_window_search", "CONFIRM21", "September_lockbox"}


def load_matrix() -> dict[str, Any]:
    """Load and validate the machine-readable compact-feature matrix."""
    matrix = json.loads((CYCLE / "configs/compact_feature_rescue_matrix.json").read_text(encoding="utf-8"))
    ids = {item["id"] for item in matrix["candidates"]}
    if ids != set(FEATURE_PROFILES):
        raise RuntimeError(f"COMPACT_MATRIX_NOT_EXACTLY_F0_F1_F2: {sorted(ids)}")
    if any(any(term.lower() in json.dumps(item, ensure_ascii=False).lower() for term in FORBIDDEN) for item in matrix.get("candidates", [])):
        raise RuntimeError("FORBIDDEN_FEATURE_STAGE_IN_MATRIX")
    if matrix.get("development_panel") != "DEV14":
        raise RuntimeError("COMPACT_STUDY_NOT_DEV14")
    return matrix


def target_days(matrix: dict[str, Any]) -> list[str]:
    """Return exactly the pre-registered fourteen development dates."""
    # The machine matrix freezes the panel identity (DEV14); the date list is
    # inherited from the already materialized, frozen A0/A1/A2 DEV14 registry.
    registry_path = CYCLE / "runs/history_window_study/history_window_manifest.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"frozen DEV14 registry missing: {registry_path}")
    days = list(json.loads(registry_path.read_text(encoding="utf-8"))["target_days"])
    if len(days) != 14 or len(set(days)) != 14:
        raise RuntimeError("DEV14_REGISTRY_INVALID")
    return days


def candidate_config(base: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Create one frozen A2 config with only the requested feature package changed."""
    profile = FEATURE_PROFILES[candidate_id]
    config = copy.deepcopy(base)
    config["training_history_months"] = 36
    config["validation_history_days"] = 28
    config["feature_package"] = profile
    config["feature_study_candidate"] = candidate_id
    config["feature_profile"]["name"] = profile
    config["feature_profile"]["temporal_covariates"] = list(feature_names(profile))
    config["feature_profile"]["b208_used"] = False
    return config


def target_arrays(path: Path, day: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the post-forward 24-point target-day artifact."""
    rows = read_csv(path / day / "target_day_prediction.csv")
    if len(rows) != 24:
        raise RuntimeError(f"{day}: target_day_prediction rows={len(rows)}, expected 24")
    rows.sort(key=lambda row: int(row["business_hour"]))
    return np.asarray([float(row["prediction"]) for row in rows]), np.asarray([float(row["target"]) for row in rows])


def transition_rows(candidate_id: str, daily: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep transition diagnostics separate from headline metrics."""
    return [{"feature_id": candidate_id, "target_day": row["target_day"], **{key: row[key] for key in ("actual_sign_switch_count", "predicted_sign_switch_count", "transition_precision", "transition_recall", "transition_f1")}} for row in daily]


def h34_rows(candidate_id: str, root: Path, days: list[str]) -> list[dict[str, Any]]:
    """Aggregate all 34 offsets while preserving bridge/D-day labels."""
    result = []
    for offset in range(1, 35):
        pred, target = [], []
        for day in days:
            rows = [row for row in read_csv(root / day / "predictions.csv") if int(row["h34_offset"]) == offset]
            if len(rows) != 1:
                raise RuntimeError(f"{candidate_id}/{day}: offset {offset} is not unique")
            pred.append(float(rows[0]["prediction"]))
            target.append(float(rows[0]["target"]))
        p, y = np.asarray(pred), np.asarray(target)
        result.append({"feature_id": candidate_id, "offset": offset, "section": "bridge" if offset <= 10 else "D-day", "bias": float(np.mean(p - y)), **compute_metrics(p, y)})
    return result


def feature_summary(candidate_id: str, daily: list[dict[str, Any]], pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Return pooled metrics and pre-registered daily diagnostics."""
    micro = compute_metrics(pred, target)
    macro = aggregate_daily_metrics(daily)
    collapse = sum(bool(row["majority_collapse"]) for row in daily)
    transitions = [float(row["transition_f1"]) for row in daily]
    minority = [float(row["minority_recall"]) for row in daily]
    return {
        "feature_id": candidate_id,
        "feature_profile": FEATURE_PROFILES[candidate_id],
        **micro,
        "daily_macro_raw": macro["raw"]["mean"],
        "daily_macro_balanced": macro["balanced"]["mean"],
        "daily_macro_balanced_std": macro["balanced"]["std"],
        "daily_macro_positive_recall": macro["positive_recall"]["mean"],
        "daily_macro_negative_recall": macro["negative_recall"]["mean"],
        "daily_macro_minority_recall": float(np.nanmean(minority)),
        "daily_macro_mae": macro["MAE"]["mean"],
        "majority_collapse_days": collapse,
        "majority_collapse_rate": collapse / len(daily),
        "transition_f1_mean": float(np.nanmean(transitions)),
        "transition_recall_mean": float(np.nanmean([float(row["transition_recall"]) for row in daily])),
        "predicted_positive_rate_mean": float(np.mean([float(row["predicted_positive_rate"]) for row in daily])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Cycle89 F0/F1/F2 DEV14 compact-feature study.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--run-dir", type=Path, default=CYCLE / "runs/feature_study")
    args = parser.parse_args()
    matrix = load_matrix()
    days = target_days(matrix)
    base = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    holdout = audit_holdout_registry(days, registry)
    if not holdout.passed:
        raise RuntimeError(f"FINAL_HOLDOUT_REGISTRY_FAIL: {holdout.detail}")
    source = CanonicalHourlySource.from_csv(args.data)
    device = select_device(base.get("device_policy", "cuda_if_deterministic_else_cpu"), seed=int(base["training"]["seed"]))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    all_daily, all_counts, all_gradients, all_majority, all_transition = [], [], [], [], []
    summaries, offset_metrics = [], []

    for candidate_id, profile in FEATURE_PROFILES.items():
        config = candidate_config(base, candidate_id)
        root = args.run_dir / candidate_id
        daily, predictions, targets = [], [], []
        for day in days:
            manifest = run_one(day, source, config, root, registry, device=device.device)
            if manifest.get("leakage_status") != "STRICT/PASS" or manifest.get("target_day_sample_count") != 24:
                raise RuntimeError(f"{candidate_id}/{day}: strict OOS gate failed")
            pred, target = target_arrays(root, day)
            row = daily_metric_row(day, pred, target)
            row.update({"feature_id": candidate_id, "feature_profile": profile})
            daily.append(row); all_daily.append(row); predictions.extend(pred.tolist()); targets.extend(target.tolist())
            split = json.loads((root / day / "split_manifest.json").read_text(encoding="utf-8"))
            all_counts.append({"feature_id": candidate_id, "target_day": day, "train_count": split["train_count"], "validation_count": split["validation_count"], "training_last_day": split["training_last_day"], "feature_profile": split["feature_profile"]})
            training_manifest = json.loads((root / day / "manifest.json").read_text(encoding="utf-8"))
            grad = read_csv(root / day / "gradient_stats.csv")
            grad_values = np.asarray([float(x["grad_norm_pre_clip"]) for x in grad])
            all_gradients.append({"feature_id": candidate_id, "target_day": day, "median_grad_norm": float(np.median(grad_values)), "p90_grad_norm": float(np.percentile(grad_values, 90)), "max_grad_norm": float(np.max(grad_values)), "clip_fraction": float(np.mean([int(x["clipped"]) for x in grad])), "best_step": training_manifest["best_step"]})
            all_majority.append({"feature_id": candidate_id, "target_day": day, "majority_collapse": row["majority_collapse"], "raw": row["raw"], "balanced": row["balanced"], "minority_recall": row["minority_recall"], "predicted_positive_rate": row["predicted_positive_rate"]})
        p, y = np.asarray(predictions), np.asarray(targets)
        summaries.append(feature_summary(candidate_id, daily, p, y))
        all_transition.extend(transition_rows(candidate_id, daily))
        offset_metrics.extend(h34_rows(candidate_id, root, days))

    comparison = args.run_dir / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    comparator = (CYCLE / "../cycle_88_numeric_spread_da_minus_rt/runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv").resolve()
    if not comparator.exists():
        raise FileNotFoundError(comparator)
    comparator_hash = sha256_file(comparator)
    baseline = {day: baseline_row_from_arrays(day, *baseline_day_rows(comparator, day), "Cycle88_LGBM_v2_full_F0_F9") for day in days}
    paired = []
    for row in all_daily:
        base_row = baseline[row["target_day"]]
        paired.append({"feature_id": row["feature_id"], "baseline_model": "Cycle88_LGBM_v2_full_F0_F9", "baseline_sha256": comparator_hash, **paired_daily_delta(row, base_row)})

    # Frozen A2 is a same-date reference artifact, not re-used as a feature-study run.
    a2_path = CYCLE / "runs/history_window_study/comparison/history_window_daily_metrics.csv"
    if not a2_path.exists():
        raise FileNotFoundError(a2_path)
    with a2_path.open(encoding="utf-8", newline="") as handle:
        a2_rows = [row for row in csv.DictReader(handle) if row.get("history_id") == "A2_HISTORY_36M" and row.get("target_day") in days]
    if len(a2_rows) != 14:
        raise RuntimeError(f"A2_SAME_DATE_REFERENCE_ROWS={len(a2_rows)}")
    a2_daily = {row["target_day"]: row for row in a2_rows}
    a2_paired = []
    for row in all_daily:
        a2 = a2_daily[row["target_day"]]
        a2_paired.append({"feature_id": row["feature_id"], "target_day": row["target_day"], "delta_raw_vs_A2": float(row["raw"] - float(a2["raw"])), "delta_balanced_vs_A2": float(row["balanced"] - float(a2["balanced"])), "delta_MAE_vs_A2": float(row["MAE"] - float(a2["MAE"])), "A2_reference": "history_window_study/comparison/history_window_daily_metrics.csv"})

    write_csv(comparison / "feature_daily_metrics.csv", all_daily)
    write_csv(comparison / "feature_micro_metrics.csv", summaries)
    write_csv(comparison / "feature_macro_metrics.csv", [{
        "feature_id": row["feature_id"],
        "feature_profile": row["feature_profile"],
        "daily_macro_raw": row["daily_macro_raw"],
        "daily_macro_balanced": row["daily_macro_balanced"],
        "daily_macro_balanced_std": row["daily_macro_balanced_std"],
        "daily_macro_positive_recall": row["daily_macro_positive_recall"],
        "daily_macro_negative_recall": row["daily_macro_negative_recall"],
        "daily_macro_minority_recall": row["daily_macro_minority_recall"],
        "daily_macro_mae": row["daily_macro_mae"],
    } for row in summaries])
    write_csv(comparison / "feature_paired_vs_A2.csv", a2_paired)
    write_csv(comparison / "feature_paired_vs_Cycle88.csv", paired)
    write_csv(comparison / "feature_training_sample_counts.csv", all_counts)
    write_csv(comparison / "feature_gradient_summary.csv", all_gradients)
    write_csv(comparison / "feature_majority_collapse.csv", all_majority)
    write_csv(comparison / "feature_transition_metrics.csv", all_transition)
    write_csv(comparison / "feature_h34_offset_metrics.csv", offset_metrics)

    summary_by_id = {row["feature_id"]: row for row in summaries}
    a2_summary = {
        "daily_macro_balanced": float(np.mean([float(row["balanced"]) for row in a2_rows])),
        "daily_macro_minority_recall": float(np.mean([float(row["minority_recall"]) for row in a2_rows])),
        "majority_collapse_days": sum(row["majority_collapse"] == "True" for row in a2_rows),
        "micro_raw": float(np.mean([float(row["raw"]) for row in a2_rows])),
        "micro_balanced": float(np.mean([float(row["balanced"]) for row in a2_rows])),
        "micro_MAE": float(np.mean([float(row["MAE"]) for row in a2_rows])),
        "transition_f1_mean": float(np.mean([float(row["transition_f1"]) for row in a2_rows])),
    }
    review_lines = ["# Cycle89 Compact Feature Rescue Review", "", "status: active", "leakage_status: STRICT/PASS", "", "## Frozen execution", "", "- DEV14 only; F0/F1/F2 each independently cold-retrained on the A2 36m/28d chassis.", "- F1+F2, forecast-error features, B208, rollout, directional loss, history/validation search, CONFIRM21 and September lockbox were not run.", "- Headline metrics are D-day 24 points; H34 bridge metrics are diagnostic only.", "", "## Candidate summary", "", "| feature | micro raw | micro balanced | micro MAE | macro balanced | macro minority recall | collapse days | transition F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        review_lines.append(f"| {row['feature_id']} | {row['direction_accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['mae']:.2f} | {row['daily_macro_balanced']:.4f} | {row['daily_macro_minority_recall']:.4f} | {row['majority_collapse_days']} | {row['transition_f1_mean']:.4f} |")
    review_lines += ["", "## A2 same-date reference", "", f"- A2 macro balanced={a2_summary['daily_macro_balanced']:.4f}; macro minority recall={a2_summary['daily_macro_minority_recall']:.4f}; collapse days={a2_summary['majority_collapse_days']}; transition F1={a2_summary['transition_f1_mean']:.4f}.", "- A2 is the frozen history-study same-date reference; it is not mixed with F0/F1/F2 artifacts.", "", "## Selection gate", "", "A package is a rescue candidate only if it improves daily-macro balanced, reduces collapse, improves macro minority recall and transition F1 without sacrificing raw or MAE against A2. No package is promoted automatically from a single DEV14 screen."]
    for candidate_id, row in summary_by_id.items():
        if candidate_id == "F0_A2_CORE5":
            continue
        raw_ok = row["direction_accuracy"] >= 0.6686
        mae_ok = row["mae"] <= 89.09
        bal_ok = row["daily_macro_balanced"] >= 0.5401
        minority_ok = row["daily_macro_minority_recall"] >= a2_summary["daily_macro_minority_recall"] + 0.05
        collapse_ok = row["majority_collapse_days"] <= 5
        transition_ok = row["transition_f1_mean"] > a2_summary["transition_f1_mean"] and row["predicted_positive_rate_mean"] <= 0.75
        review_lines.append(f"- {candidate_id}: raw={'PASS' if raw_ok else 'FAIL'}, MAE={'PASS' if mae_ok else 'FAIL'}, macro balanced={'PASS' if bal_ok else 'FAIL'}, minority recall={'PASS' if minority_ok else 'FAIL'}, collapse={'PASS' if collapse_ok else 'FAIL'}, transition={'PASS' if transition_ok else 'FAIL'}.")
    winner = max((row for row in summaries if row["feature_id"] != "F0_A2_CORE5"), key=lambda row: (row["daily_macro_balanced"], row["daily_macro_minority_recall"], -row["mae"]))["feature_id"]
    review_lines += ["", f"## Feature review conclusion", "", f"On this DEV14 screen, the strongest diagnostic package by daily-macro balanced then minority recall is **{winner}**. This is a review result, not permission to run a forbidden stage or a confirmation-set claim.", "", "## Artifacts", "", "- comparison/feature_daily_metrics.csv", "- comparison/feature_micro_metrics.csv", "- comparison/feature_macro_metrics.csv", "- comparison/feature_paired_vs_A2.csv", "- comparison/feature_paired_vs_Cycle88.csv", "- comparison/feature_h34_offset_metrics.csv"]
    (comparison / "feature_review.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    manifest = {
        "status": "COMPACT_FEATURE_STUDY_COMPLETE", "phase": "DEV14_COMPACT_FEATURE_ONLY", "target_days": days,
        "candidates": list(FEATURE_PROFILES), "forbidden_stages_not_run": sorted(FORBIDDEN), "device": device.__dict__,
        "source_data_sha256": sha256_file(source.path) if source.path else None, "source_code_sha256": source_tree_hash(CYCLE / "src"),
        "config_sha256": sha256_json(base), "cycle88_comparator": str(comparator), "cycle88_comparator_sha256": comparator_hash,
        "a2_reference": str(a2_path), "summaries": summaries, "comparison_root": str(comparison), "leakage_status": "STRICT/PASS",
    }
    (args.run_dir / "feature_study_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "candidates": list(FEATURE_PROFILES), "device": device.device}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
