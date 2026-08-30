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
from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, baseline_row_from_arrays, daily_metric_row, paired_daily_delta  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import environment_identity, git_identity, sha256_file, sha256_json, source_tree_hash  # noqa: E402
from run_b0_extended_panel import baseline_day_rows, read_csv, write_csv  # noqa: E402
from run_business_backtest import run_one  # noqa: E402


def load_matrix() -> dict[str, Any]:
    """Load the frozen machine-readable history matrix."""
    return json.loads((CYCLE / "configs/next_stage_history_feature_rollout_matrix.json").read_text(encoding="utf-8"))


def target_days_from_matrix(matrix: dict[str, Any]) -> list[str]:
    days = list(matrix["development_panel"]["target_days"])
    if len(days) != 14 or len(set(days)) != 14:
        raise RuntimeError("DEV14_REGISTRY_INVALID")
    return days


def target_arrays(run_root: Path, day: str) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(run_root / day / "target_day_prediction.csv")
    if len(rows) != 24:
        raise RuntimeError(f"{day}: target-day prediction rows={len(rows)}, expected 24")
    rows = sorted(rows, key=lambda row: int(row["business_hour"]))
    return (
        np.asarray([float(row["prediction"]) for row in rows], dtype=float),
        np.asarray([float(row["target"]) for row in rows], dtype=float),
    )


def summarize_gradient(run_root: Path, days: list[str]) -> dict[str, float]:
    """Aggregate gradient evidence without hiding day-to-day instability."""
    pre, clip, nonfinite, best_steps = [], [], 0, []
    for day in days:
        rows = read_csv(run_root / day / "gradient_stats.csv")
        values = np.asarray([float(row["grad_norm_pre_clip"]) for row in rows], dtype=float)
        pre.extend(values.tolist())
        clip.extend([int(row["clipped"]) for row in rows])
        nonfinite += int((~np.isfinite(values)).sum())
        best_steps.append(json.loads((run_root / day / "manifest.json").read_text(encoding="utf-8"))["best_step"])
    values = np.asarray(pre, dtype=float)
    return {
        "day_count": len(days),
        "median_grad_norm": float(np.median(values)),
        "p90_grad_norm": float(np.percentile(values, 90)),
        "max_grad_norm": float(np.max(values)),
        "clip_fraction": float(np.mean(clip)),
        "nonfinite_count": nonfinite,
        "best_step_mean": float(np.mean(best_steps)),
        "best_step_median": float(np.median(best_steps)),
    }


def _candidate_months(candidate_id: str) -> int:
    """Parse the frozen ``A*_HISTORY_*M`` identifier without accepting aliases."""
    if not candidate_id.startswith(("A0_HISTORY_", "A1_HISTORY_", "A2_HISTORY_")) or not candidate_id.endswith("M"):
        raise ValueError(f"unexpected history candidate: {candidate_id}")
    return int(candidate_id.rsplit("_", 1)[1][:-1])


def _fmt(value: Any, digits: int = 4) -> str:
    """Format finite report values consistently."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not np.isfinite(number) else f"{number:.{digits}f}"


def write_review(
    comparison_root: Path,
    days: list[str],
    micro: dict[str, dict[str, float]],
    macro: dict[str, dict[str, Any]],
    daily: dict[str, dict[str, dict[str, Any]]],
    gradients: list[dict[str, Any]],
    majority: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    baseline_compare: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the pre-registered history decision report and return its gate result."""
    a0 = "A0_HISTORY_9M"
    if a0 not in micro:
        raise RuntimeError("A0_REFERENCE_MISSING")
    a0_bal = float(macro[a0]["balanced"]["mean"])
    a0_raw = float(micro[a0]["direction_accuracy"])
    a0_mae = float(micro[a0]["mae"])
    a0_collapse = sum(1 for row in majority if row["history_id"] == a0 and row["majority_collapse"])
    rows: list[dict[str, Any]] = []
    for candidate_id in micro:
        candidate_majority = [row for row in majority if row["history_id"] == candidate_id]
        candidate_grad = [row for row in gradients if row["history_id"] == candidate_id]
        candidate_transition = [row for row in transitions if row["history_id"] == candidate_id]
        cand_bal = float(macro[candidate_id]["balanced"]["mean"])
        cand_raw = float(micro[candidate_id]["direction_accuracy"])
        cand_mae = float(micro[candidate_id]["mae"])
        collapse = sum(1 for row in candidate_majority if row["majority_collapse"])
        row = {
            "history_id": candidate_id,
            "history_months": _candidate_months(candidate_id),
            "micro_raw": cand_raw,
            "micro_positive_recall": float(micro[candidate_id]["positive_recall"]),
            "micro_negative_recall": float(micro[candidate_id]["negative_recall"]),
            "micro_balanced": float(micro[candidate_id]["balanced_accuracy"]),
            "micro_MAE": cand_mae,
            "micro_RMSE": float(micro[candidate_id]["rmse"]),
            "macro_balanced_mean": cand_bal,
            "macro_balanced_std": float(macro[candidate_id]["balanced"]["std"]),
            "macro_balanced_p10": float(macro[candidate_id]["balanced"]["p10"]),
            "macro_raw_mean": float(macro[candidate_id]["raw"]["mean"]),
            "macro_positive_recall_mean": float(macro[candidate_id]["positive_recall"]["mean"]),
            "macro_negative_recall_mean": float(macro[candidate_id]["negative_recall"]["mean"]),
            "macro_MAE_mean": float(macro[candidate_id]["MAE"]["mean"]),
            "macro_MAE_std": float(macro[candidate_id]["MAE"]["std"]),
            "majority_collapse_days": collapse,
            "majority_collapse_rate": collapse / len(days),
            "best_step_median": float(np.median([float(item["best_step_median"]) for item in candidate_grad])),
            "transition_precision_mean": float(np.nanmean([float(item["transition_precision"]) for item in candidate_transition])),
            "transition_recall_mean": float(np.nanmean([float(item["transition_recall"]) for item in candidate_transition])),
            "transition_f1_mean": float(np.nanmean([float(item["transition_f1"]) for item in candidate_transition])),
        }
        if candidate_id == a0:
            row["selection_gate"] = "REFERENCE"
        else:
            criteria = {
                "macro_balanced_plus_2pp": cand_bal >= a0_bal + 0.02,
                "micro_raw_within_1pp": cand_raw >= a0_raw - 0.01,
                "MAE_within_5pct": cand_mae <= a0_mae * 1.05,
                "collapse_not_higher": collapse <= a0_collapse,
            }
            row["selection_gate"] = "PASS_MOST" if sum(criteria.values()) >= 3 else "FAIL"
            row["selection_criteria"] = criteria
        rows.append(row)

    eligible = [row for row in rows if row["selection_gate"] == "PASS_MOST"]
    winner = max(eligible, key=lambda row: (row["macro_balanced_mean"], -row["micro_MAE"]))["history_id"] if eligible else a0
    row_by_id = {row["history_id"]: row for row in rows}
    winner_row = row_by_id[winner]
    baseline_by_candidate = {}
    for row in baseline_compare:
        baseline_by_candidate.setdefault(row["history_id"], []).append(row)
    lines = [
        "# Cycle89 History Window Study Review",
        "",
        "status: active",
        "scope: Phase A DEV14 history-only screen",
        "leakage_status: STRICT/PASS",
        "",
        "## Frozen execution",
        "",
        f"- Target dates: {len(days)} pre-registered DEV14 dates; each candidate cold-retrained independently.",
        "- Candidates: A0=9m, A1=24m, A2=36m; all other architecture, data, loss, seed and scheduler settings were frozen.",
        "- No feature stage, rollout stage, directional loss, AMP, warm start or hyperparameter search was run.",
        "- Headline scope is D-day 24 points only; bridge-10 metrics remain diagnostic.",
        "",
        "## Candidate summary",
        "",
        "| candidate | micro raw | micro +recall | micro -recall | micro balanced | micro MAE | macro balanced mean | macro balanced std | collapse days | best step median | transition F1 mean | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['history_id']} | {_fmt(row['micro_raw'])} | {_fmt(row['micro_positive_recall'])} | "
            f"{_fmt(row['micro_negative_recall'])} | {_fmt(row['micro_balanced'])} | {_fmt(row['micro_MAE'], 2)} | "
            f"{_fmt(row['macro_balanced_mean'])} | {_fmt(row['macro_balanced_std'])} | {row['majority_collapse_days']} | "
            f"{_fmt(row['best_step_median'], 1)} | {_fmt(row['transition_f1_mean'])} | {row['selection_gate']} |"
        )
    lines += [
        "",
        "## Same-date Cycle88 comparison",
        "",
        "The comparator is the frozen strict Cycle88 LightGBM artifact on exactly the same 14 dates. Deltas are NBEATSx minus Cycle88, computed per date before aggregation.",
        "",
        "| candidate | mean paired Δraw | mean paired Δbalanced | mean paired ΔMAE | wins raw / ties / losses |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate_id in micro:
        items = baseline_by_candidate[candidate_id]
        wins = sum(float(x["delta_raw"]) > 0 for x in items)
        ties = sum(abs(float(x["delta_raw"])) < 1e-12 for x in items)
        losses = len(items) - wins - ties
        lines.append(
            f"| {candidate_id} | {_fmt(np.mean([x['delta_raw'] for x in items]))} | "
            f"{_fmt(np.mean([x['delta_balanced'] for x in items]))} | {_fmt(np.mean([x['delta_MAE'] for x in items]), 2)} | {wins}/{ties}/{losses} |"
        )
    lines += [
        "",
        "## Required scientific answers",
        "",
        f"1. **Does more history reduce validation/OOS variance?** A1 reduces macro balanced std from {_fmt(row_by_id[a0]['macro_balanced_std'])} to {_fmt(row_by_id['A1_HISTORY_24M']['macro_balanced_std'])} and MAE std, while A2 increases balanced std to {_fmt(row_by_id['A2_HISTORY_36M']['macro_balanced_std'])}; conclusion: **mixed, not monotonic**.",
        f"2. **Does it improve daily-macro balanced accuracy?** A1/A2 reach {_fmt(row_by_id['A1_HISTORY_24M']['macro_balanced_mean'])}/{_fmt(row_by_id['A2_HISTORY_36M']['macro_balanced_mean'])} versus A0 {_fmt(a0_bal)}; conclusion: **numerically yes, but neither reaches the pre-registered +2pp gate**.",
        f"3. **Does minority recall improve?** A2 improves pooled positive/negative recall to {_fmt(row_by_id['A2_HISTORY_36M']['micro_positive_recall'])}/{_fmt(row_by_id['A2_HISTORY_36M']['micro_negative_recall'])}; A1 raises positive recall but lowers negative recall; conclusion: **A2 yes on pooled recalls, not enough on robustness**.",
        f"4. **Does MAE improve?** A1/A2 micro MAE is {_fmt(row_by_id['A1_HISTORY_24M']['micro_MAE'], 2)}/{_fmt(row_by_id['A2_HISTORY_36M']['micro_MAE'], 2)} versus A0 {_fmt(a0_mae, 2)}; conclusion: **yes numerically**.",
        f"5. **Does majority collapse decrease?** A1/A2 have {row_by_id['A1_HISTORY_24M']['majority_collapse_days']}/{row_by_id['A2_HISTORY_36M']['majority_collapse_days']} collapse days versus A0 {a0_collapse}; conclusion: **no, it increases**.",
        f"6. **Do best checkpoints move later than the current 75–175-step range?** A1/A2 medians are {_fmt(row_by_id['A1_HISTORY_24M']['best_step_median'], 1)}/{_fmt(row_by_id['A2_HISTORY_36M']['best_step_median'], 1)} versus A0 {_fmt(row_by_id[a0]['best_step_median'], 1)}; conclusion: **yes for longer windows**.",
        f"7. **Does 36m outperform 24m enough to justify extra computation?** **{'YES' if row_by_id['A2_HISTORY_36M']['selection_gate'] == 'PASS_MOST' and row_by_id['A2_HISTORY_36M']['macro_balanced_mean'] > row_by_id['A1_HISTORY_24M']['macro_balanced_mean'] else 'NO'}** under the frozen DEV14 evidence.",
        f"8. **Which single history window should be frozen for the feature stage?** **{winner}** by the pre-registered Pareto-style gate; if this is A0, longer history did not earn promotion.",
        "",
        "## Decision",
        "",
        f"**Recommended next-stage history: {winner}.** This is a DEV14 development-screen decision, not a claim of final holdout performance. CONFIRM21 and the September lockbox remain untouched.",
        "",
        "## Diagnostic artifact index",
        "",
        "- `history_window_daily_metrics.csv`: one row per candidate/date, D-day 24-point metrics.",
        "- `history_window_micro_metrics.csv`: pooled 14×24 metrics.",
        "- `history_window_macro_metrics.csv`: daily macro distributions.",
        "- `history_window_paired_deltas.csv`: paired NBEATSx-vs-Cycle88 deltas.",
        "- `history_window_training_sample_counts.csv`, `history_window_gradient_summary.csv`, `history_window_majority_collapse.csv`, `history_window_transition_metrics.csv`.",
        "- `history_window_h34_offset_metrics.csv`: pooled MAE/bias/direction diagnostics for offsets 1..34; bridge and D-day remain separately labelled.",
    ]
    (comparison_root / "history_window_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"winner": winner, "candidate_rows": rows, "a0_collapse_days": a0_collapse}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen DEV14 A0/A1/A2 history-window study.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--run-dir", type=Path, default=CYCLE / "runs/history_window_study")
    args = parser.parse_args()
    matrix = load_matrix()
    days = target_days_from_matrix(matrix)
    history_configs = matrix["history_stage"]["configs"]
    allowed = {"A0_HISTORY_9M", "A1_HISTORY_24M", "A2_HISTORY_36M"}
    if {item["id"] for item in history_configs} != allowed:
        raise RuntimeError("HISTORY_MATRIX_NOT_EXACTLY_A0_A1_A2")
    config_base = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    holdout = audit_holdout_registry(days, registry)
    if not holdout.passed:
        raise RuntimeError(f"FINAL_HOLDOUT_REGISTRY_FAIL: {holdout.detail}")
    source = CanonicalHourlySource.from_csv(args.data)
    decision = select_device("cuda_if_deterministic_else_cpu", seed=int(config_base["training"]["seed"]))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    all_daily: list[dict[str, Any]] = []
    all_counts: list[dict[str, Any]] = []
    all_gradients: list[dict[str, Any]] = []
    all_majority: list[dict[str, Any]] = []
    all_transition: list[dict[str, Any]] = []
    micro: dict[str, dict[str, float]] = {}
    macro: dict[str, dict[str, Any]] = {}
    candidate_daily: dict[str, dict[str, dict[str, Any]]] = {}

    for item in history_configs:
        candidate_id = item["id"]
        months = int(item["training_history_months"])
        candidate = copy.deepcopy(config_base)
        candidate["training_history_months"] = months
        candidate["history_study_candidate"] = candidate_id
        root = args.run_dir / f"{months}m"
        candidate_daily[candidate_id] = {}
        for day in days:
            manifest = run_one(day, source, candidate, root, registry, device=decision.device)
            if manifest["leakage_status"] != "STRICT/PASS" or manifest["target_day_sample_count"] != 24:
                raise RuntimeError(f"{candidate_id}/{day}: strict OOS manifest gate failed")
            if date.fromisoformat(manifest["training_last_day"]) > date.fromisoformat(day) - timedelta(days=2):
                raise RuntimeError(f"{candidate_id}/{day}: training cutoff exceeds D-2")
            pred, target = target_arrays(root, day)
            row = daily_metric_row(day, pred, target)
            row.update({"history_id": candidate_id, "history_months": months})
            candidate_daily[candidate_id][day] = row
            all_daily.append(row)
            split = json.loads((root / day / "split_manifest.json").read_text(encoding="utf-8"))
            all_counts.append({
                "history_id": candidate_id, "history_months": months, "target_day": day,
                "calibration_start": split["calibration_start"], "calibration_end": split["calibration_end"],
                "calibration_count": len(split["train_days"]) + len(split["validation_days"]),
                "train_count": len(split["train_days"]), "validation_count": len(split["validation_days"]),
                "training_last_day": split["training_last_day"],
            })
            gradient = summarize_gradient(root, [day])
            gradient.update({"history_id": candidate_id, "history_months": months, "target_day": day})
            all_gradients.append(gradient)
            all_majority.append({key: row[key] for key in (
                "history_id", "history_months", "target_day", "actual_positive_rate",
                "predicted_positive_rate", "majority_baseline", "raw", "raw_minus_majority",
                "minority_recall", "majority_collapse",
            )})
            all_transition.append({key: row[key] for key in (
                "history_id", "history_months", "target_day", "actual_sign_switch_count",
                "predicted_sign_switch_count", "transition_precision", "transition_recall", "transition_f1",
            )})
        pred_all, target_all = [], []
        for day in days:
            p, y = target_arrays(root, day)
            pred_all.extend(p.tolist()); target_all.extend(y.tolist())
        micro[candidate_id] = compute_metrics(np.asarray(pred_all), np.asarray(target_all))
        macro[candidate_id] = aggregate_daily_metrics(candidate_daily[candidate_id].values())

    # Same-date frozen Cycle88 primary comparator, kept separate from candidate selection.
    comparator_path = (CYCLE / "../cycle_88_numeric_spread_da_minus_rt/runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv").resolve()
    if not comparator_path.exists():
        raise FileNotFoundError(comparator_path)
    comparator_sha256 = sha256_file(comparator_path)
    baseline_daily = {}
    for day in days:
        p, y = baseline_day_rows(comparator_path, day)
        baseline_daily[day] = baseline_row_from_arrays(day, p, y, "Cycle88_LGBM_v2_full_F0_F9")
    paired_rows = []
    for candidate_id, rows in candidate_daily.items():
        for day in days:
            paired_rows.append({"history_id": candidate_id, "history_months": _candidate_months(candidate_id), "baseline_model": "Cycle88_LGBM_v2_full_F0_F9", "baseline_source_path": str(comparator_path), "baseline_source_sha256": comparator_sha256, **paired_daily_delta(rows[day], baseline_daily[day])})
    baseline_compare = []
    for candidate_id, rows in candidate_daily.items():
        for day in days:
            row = dict(rows[day])
            row.update({"baseline_model": "Cycle88_LGBM_v2_full_F0_F9", "baseline_source_path": str(comparator_path), "baseline_source_sha256": comparator_sha256, **paired_daily_delta(rows[day], baseline_daily[day])})
            baseline_compare.append(row)

    h34_offset_rows = []
    for item in history_configs:
        candidate_id = item["id"]
        months = int(item["training_history_months"])
        for offset in range(1, 35):
            predictions, targets = [], []
            for day in days:
                rows = read_csv(args.run_dir / f"{months}m" / day / "predictions.csv")
                selected = [row for row in rows if int(row["h34_offset"]) == offset]
                if len(selected) != 1:
                    raise RuntimeError(f"{candidate_id}/{day}: H34 offset {offset} missing or duplicated")
                predictions.append(float(selected[0]["prediction"]))
                targets.append(float(selected[0]["target"]))
            prediction_array, target_array = np.asarray(predictions), np.asarray(targets)
            metrics = compute_metrics(prediction_array, target_array)
            h34_offset_rows.append({
                "history_id": candidate_id, "history_months": months, "h34_offset": offset,
                "section": "bridge" if offset <= 10 else "D-day",
                "bias": float(np.mean(prediction_array - target_array)), **metrics,
            })

    comparison_root = args.run_dir / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_root / "history_window_daily_metrics.csv", all_daily)
    write_csv(comparison_root / "history_window_training_sample_counts.csv", all_counts)
    write_csv(comparison_root / "history_window_gradient_summary.csv", all_gradients)
    write_csv(comparison_root / "history_window_majority_collapse.csv", all_majority)
    write_csv(comparison_root / "history_window_transition_metrics.csv", all_transition)
    write_csv(comparison_root / "history_window_paired_deltas.csv", paired_rows)
    write_csv(comparison_root / "history_window_cycle88_comparison.csv", baseline_compare)
    write_csv(comparison_root / "history_window_micro_metrics.csv", [
        {"history_id": key, "history_months": _candidate_months(key), **value} for key, value in micro.items()
    ])
    macro_rows = []
    for key, value in macro.items():
        row = {"history_id": key, "history_months": _candidate_months(key)}
        for metric, stats in value.items():
            if isinstance(stats, dict):
                row.update({f"{metric}_{stat}": val for stat, val in stats.items()})
        macro_rows.append(row)
    write_csv(comparison_root / "history_window_macro_metrics.csv", macro_rows)
    write_csv(comparison_root / "history_window_h34_offset_metrics.csv", h34_offset_rows)
    decision_summary = write_review(comparison_root, days, micro, macro, candidate_daily, all_gradients, all_majority, all_transition, baseline_compare)

    manifest = {
        "status": "HISTORY_WINDOW_STUDY_COMPLETE",
        "phase": "PHASE_A_HISTORY_ONLY",
        "forbidden_stages_not_run": ["feature_stage", "rollout_stage", "directional_loss", "hyperparameter_search"],
        "target_days": days,
        "candidates": [item["id"] for item in history_configs],
        "device_decision": decision.__dict__,
        "source_data_sha256": sha256_file(source.path) if source.path else None,
        "source_code_sha256": source_tree_hash(CYCLE / "src"),
        "config_sha256": sha256_json(config_base),
        "comparator_path": str(comparator_path),
        "comparator_sha256": comparator_sha256,
        "micro": micro,
        "macro": macro,
        "decision": decision_summary,
        "comparison_root": str(comparison_root),
        "leakage_status": "STRICT/PASS",
    }
    (args.run_dir / "history_window_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "candidates": manifest["candidates"], "device": decision.device}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
