from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sgdfnet.metrics import build_metrics_frame, build_segment_metrics, build_tail_metrics, capped_smape
from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment


DEFAULT_EXPERIMENTS: list[dict[str, object]] = [
    {
        "name": "baseline",
        "config": "SGDFNet/configs/cutoff_recovery_2026_baseline.yaml",
        "family": "baseline",
        "stage": "baseline",
        "hypothesis": "Current corrected cutoff-safe baseline establishes the no-leakage reference.",
        "changed_factor": "none",
    },
    {
        "name": "diag_a_prune_actualside",
        "config": "SGDFNet/configs/cutoff_recovery_2026_diag_a_prune_actualside.yaml",
        "family": "A",
        "stage": "diagnosis",
        "hypothesis": "If visible actual-side features are still not helping enough, removing them should expose whether they are noisy under cutoff-safe conditions.",
        "changed_factor": "prune_actualside_history_and_residual_features",
    },
    {
        "name": "diag_b2_global_fill_bias",
        "config": "SGDFNet/configs/cutoff_recovery_2026_diag_b2_global_fill_bias.yaml",
        "family": "B",
        "stage": "diagnosis",
        "hypothesis": "A train-safe global fill bias may recover some information lost when DA directly fills the blocked same-day window.",
        "changed_factor": "global_da_fill_bias_from_val",
    },
    {
        "name": "diag_b3_segment_fill_bias",
        "config": "SGDFNet/configs/cutoff_recovery_2026_diag_b3_segment_fill_bias.yaml",
        "family": "B",
        "stage": "diagnosis",
        "hypothesis": "Segment-specific DA fill bias may better match 1_8 / 9_16 / 17_24 asymmetry than one global fill shift.",
        "changed_factor": "segment_da_fill_bias_from_val",
    },
    {
        "name": "diag_c1_weekly_history",
        "config": "SGDFNet/configs/cutoff_recovery_2026_diag_c1_weekly_history.yaml",
        "family": "C",
        "stage": "diagnosis",
        "hypothesis": "Weekly same-hour history can restore stable pre-cutoff signal without leaking target-month actuals.",
        "changed_factor": "weekly_history_features",
    },
    {
        "name": "diag_c2_segment_local",
        "config": "SGDFNet/configs/cutoff_recovery_2026_diag_c2_segment_local.yaml",
        "family": "C",
        "stage": "diagnosis",
        "hypothesis": "Segment-local rolling stats may restore hour/segment structure lost after cutoff recomputation.",
        "changed_factor": "segment_local_stats",
    },
    {
        "name": "diag_d2_segment_conditioned",
        "config": "SGDFNet/configs/cutoff_recovery_2026_diag_d2_segment_conditioned.yaml",
        "family": "D",
        "stage": "diagnosis",
        "hypothesis": "A lightweight segment-conditioned model may repair 9_16 behavior without changing the no-leakage feature contract.",
        "changed_factor": "segment_conditioned_model",
    },
]

TARGETED_EXPERIMENTS: list[dict[str, object]] = [
    {
        "name": "target_b2_global_fill_bias_trainval",
        "config": "SGDFNet/configs/cutoff_recovery_2026_target_b2_global_fill_bias_trainval.yaml",
        "family": "B",
        "stage": "targeted",
        "hypothesis": "Using train+val to estimate a stable global fill bias may improve over the val-only fill correction.",
        "changed_factor": "global_da_fill_bias_from_train_val",
    },
    {
        "name": "target_b3_segment_fill_bias_trainval",
        "config": "SGDFNet/configs/cutoff_recovery_2026_target_b3_segment_fill_bias_trainval.yaml",
        "family": "B",
        "stage": "targeted",
        "hypothesis": "A train+val segment fill bias may outperform val-only segment fill correction if fill bias is the main failure source.",
        "changed_factor": "segment_da_fill_bias_from_train_val",
    },
    {
        "name": "target_c3_forecast_pressure",
        "config": "SGDFNet/configs/cutoff_recovery_2026_target_c3_forecast_pressure.yaml",
        "family": "C",
        "stage": "targeted",
        "hypothesis": "Forecast pressure interactions may restore some hard-hour predictive structure without using hidden actuals.",
        "changed_factor": "forecast_pressure_interactions",
    },
    {
        "name": "target_d3_risk_hour_weight",
        "config": "SGDFNet/configs/cutoff_recovery_2026_target_d3_risk_hour_weight.yaml",
        "family": "D",
        "stage": "targeted",
        "hypothesis": "A mild 9/10/15 risk-hour weighting may recover capped-SMAPE where the corrected baseline under-focuses business-critical hours.",
        "changed_factor": "risk_hour_weighting",
    },
]


def _floor50_smape(actual: pd.Series, pred: pd.Series) -> float:
    return capped_smape(actual.to_numpy(dtype=float), pred.to_numpy(dtype=float))


def _load_run_outputs(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig", parse_dates=["timestamp", "decision_day", "target_day"])
    monthly_summary = pd.read_csv(run_dir / "monthly_summary.csv", encoding="utf-8-sig")
    with open(run_dir / "metrics_summary.json", "r", encoding="utf-8") as f:
        metrics_summary = json.load(f)
    with open(run_dir / "split_audit.json", "r", encoding="utf-8") as f:
        split_audit = json.load(f)
    return predictions, monthly_summary, metrics_summary, split_audit


def _compute_monthly_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month, month_df in predictions.groupby("target_month"):
        row: dict[str, object] = {"month": month, "rows": int(len(month_df))}
        row.update(build_metrics_frame(month_df))
        seg_916 = month_df[month_df["segment"] == "9_16"].copy()
        row["segment_9_16_rt_capped_smape"] = _floor50_smape(seg_916["rt_actual"], seg_916["rt_hat"]) if not seg_916.empty else float("nan")
        row["segment_9_16_rt_smape"] = build_metrics_frame(seg_916)["rt_smape"] if not seg_916.empty else float("nan")
        risk = month_df[month_df["hour"].isin([9, 10, 15])].copy()
        row["risk_9_10_15_rt_capped_smape"] = _floor50_smape(risk["rt_actual"], risk["rt_hat"]) if not risk.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def _compute_experiment_summary(predictions: pd.DataFrame, monthly_scores: pd.DataFrame) -> dict[str, object]:
    overall = build_metrics_frame(predictions)
    segment_metrics = build_segment_metrics(predictions)
    tail_metrics = build_tail_metrics(predictions)

    segment_916 = predictions[predictions["segment"] == "9_16"].copy()
    segment_916_capped = _floor50_smape(segment_916["rt_actual"], segment_916["rt_hat"]) if not segment_916.empty else float("nan")
    risk_915 = predictions[predictions["hour"].isin([9, 10, 15])].copy()
    risk_915_capped = _floor50_smape(risk_915["rt_actual"], risk_915["rt_hat"]) if not risk_915.empty else float("nan")
    non_risk = predictions[~predictions["hour"].isin([9, 10, 15])].copy()
    non_risk_capped = _floor50_smape(non_risk["rt_actual"], non_risk["rt_hat"]) if not non_risk.empty else float("nan")

    hour_rows = []
    for hour in [9, 10, 15]:
        hour_df = predictions[predictions["hour"] == hour].copy()
        hour_rows.append(
            {
                "hour": hour,
                "rt_capped_smape": _floor50_smape(hour_df["rt_actual"], hour_df["rt_hat"]) if not hour_df.empty else float("nan"),
                "rows": int(len(hour_df)),
            }
        )

    top10 = tail_metrics[tail_metrics["tail_quantile"] == 0.9]
    top10_tail_delta_mae = float(top10["tail_delta_mae"].iloc[0]) if not top10.empty else float("nan")
    top10_tail_rt_smape = float(top10["tail_rt_smape"].iloc[0]) if not top10.empty else float("nan")

    return {
        "overall": overall,
        "segment_9_16_rt_capped_smape": segment_916_capped,
        "segment_9_16_rt_smape": build_metrics_frame(segment_916)["rt_smape"] if not segment_916.empty else float("nan"),
        "risk_9_10_15_rt_capped_smape": risk_915_capped,
        "non_risk_rt_capped_smape": non_risk_capped,
        "hour_metrics": hour_rows,
        "monthly_scores": monthly_scores.to_dict(orient="records"),
        "worst_month_9_16_rt_capped_smape": float(monthly_scores["segment_9_16_rt_capped_smape"].max()) if not monthly_scores.empty else float("nan"),
        "top10_tail_delta_mae": top10_tail_delta_mae,
        "top10_tail_rt_smape": top10_tail_rt_smape,
        "tail_metrics": tail_metrics.to_dict(orient="records"),
        "segment_metrics": segment_metrics.to_dict(orient="records"),
    }


def _family_diagnosis_note(record: dict[str, object], baseline: dict[str, object]) -> str:
    delta = float(record["overall_rt_capped_smape"] - baseline["overall_rt_capped_smape"])
    family = record["family"]
    if family == "A":
        return f"Actual-side pruning changed capped SMAPE by {delta:+.3f}; this is a direct probe of visible-history dependence."
    if family == "B":
        return f"DA fill strategy changed capped SMAPE by {delta:+.3f}; this isolates same-day blocked-window fill sensitivity."
    if family == "C":
        return f"History/feature redesign changed capped SMAPE by {delta:+.3f}; this tests whether no-leakage signal recovery is mostly feature-limited."
    if family == "D":
        return f"Lightweight policy/model adjustment changed capped SMAPE by {delta:+.3f}; this tests whether the remaining gap is mainly model-allocation rather than visibility."
    return f"Capped SMAPE changed by {delta:+.3f}."


def _decision_for_record(record: dict[str, object], baseline: dict[str, object]) -> tuple[str, str]:
    delta = float(record["overall_rt_capped_smape"] - baseline["overall_rt_capped_smape"])
    if record["name"] == "baseline":
        return "KEEP", "Corrected fixed no-leakage baseline."
    if delta <= -2.0:
        return "KEEP", "Improved overall corrected baseline by at least 2.0 capped-SMAPE points."
    if delta <= -0.5:
        return "RETRY", "Shows meaningful signal versus the corrected baseline and may justify a more focused follow-up."
    return "REJECT", "Did not improve the corrected no-leakage baseline enough to justify another round on the same mechanism."


def _write_markdown_report(path: Path, title: str, lines: list[str]) -> None:
    path.write_text("\n".join([title, "", *lines]).strip() + "\n", encoding="utf-8")


def _run_one_experiment(spec: dict[str, object], batch_dir: Path) -> dict[str, object]:
    config_path = Path(spec["config"])
    run_dir = run_protocol_b_cutoff_experiment(config_path)
    predictions, monthly_summary, metrics_summary, split_audit = _load_run_outputs(run_dir)
    monthly_scores = _compute_monthly_scores(predictions)
    monthly_scores.to_csv(run_dir / "monthly_scores_2026.csv", index=False, encoding="utf-8-sig")
    summary = _compute_experiment_summary(predictions, monthly_scores)
    record = {
        "name": spec["name"],
        "family": spec["family"],
        "stage": spec["stage"],
        "hypothesis": spec["hypothesis"],
        "changed_factor": spec["changed_factor"],
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "rows": int(len(predictions)),
        "coverage_start": str(predictions["timestamp"].min()),
        "coverage_end": str(predictions["timestamp"].max()),
        "protocol_tag": "B_D15_cutoff_walk_forward",
        "leakage_statement": "fixed no-leakage; D15 Protocol B walk-forward; no future actuals after decision cutoff",
        "overall_rt_smape": float(summary["overall"]["rt_smape"]),
        "overall_rt_capped_smape": float(summary["overall"]["rt_capped_smape"]),
        "overall_rt_mae": float(summary["overall"]["rt_mae"]),
        "overall_delta_mae": float(summary["overall"]["delta_mae"]),
        "direction_accuracy": float(summary["overall"]["direction_accuracy"]),
        "positive_direction_recall": float(summary["overall"]["positive_direction_recall"]),
        "segment_9_16_rt_capped_smape": float(summary["segment_9_16_rt_capped_smape"]),
        "segment_9_16_rt_smape": float(summary["segment_9_16_rt_smape"]),
        "risk_9_10_15_rt_capped_smape": float(summary["risk_9_10_15_rt_capped_smape"]),
        "non_risk_rt_capped_smape": float(summary["non_risk_rt_capped_smape"]),
        "worst_month_9_16_rt_capped_smape": float(summary["worst_month_9_16_rt_capped_smape"]),
        "top10_tail_delta_mae": float(summary["top10_tail_delta_mae"]),
        "top10_tail_rt_smape": float(summary["top10_tail_rt_smape"]),
        "monthly_scores_path": str((run_dir / "monthly_scores_2026.csv").resolve()),
        "split_audit_path": str((run_dir / "split_audit.json").resolve()),
        "metrics_summary_path": str((run_dir / "metrics_summary.json").resolve()),
        "hour_9_rt_capped_smape": float(next((r["rt_capped_smape"] for r in summary["hour_metrics"] if r["hour"] == 9), float("nan"))),
        "hour_10_rt_capped_smape": float(next((r["rt_capped_smape"] for r in summary["hour_metrics"] if r["hour"] == 10), float("nan"))),
        "hour_15_rt_capped_smape": float(next((r["rt_capped_smape"] for r in summary["hour_metrics"] if r["hour"] == 15), float("nan"))),
    }
    with open(run_dir / "recovery_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "record": record,
                "monthly_scores": summary["monthly_scores"],
                "tail_metrics": summary["tail_metrics"],
                "segment_metrics": summary["segment_metrics"],
                "split_audit": split_audit,
                "metrics_summary": metrics_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return record


def _family_best_rows(leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, fam_df in leaderboard.groupby("family"):
        fam_non_baseline = fam_df[fam_df["name"] != "baseline"].copy()
        if fam_non_baseline.empty:
            continue
        rows.append(fam_non_baseline.sort_values("overall_rt_capped_smape", ascending=True).iloc[0].to_dict())
    return pd.DataFrame(rows)


def _select_targeted_specs(
    diagnostic_leaderboard: pd.DataFrame,
    baseline_row: dict[str, object],
) -> list[dict[str, object]]:
    family_best = _family_best_rows(diagnostic_leaderboard)
    if family_best.empty:
        return []

    signal_families = set(
        family_best.loc[
            family_best["overall_rt_capped_smape"] <= float(baseline_row["overall_rt_capped_smape"]) - 0.5,
            "family",
        ].tolist()
    )
    if not signal_families:
        return []
    return [spec for spec in TARGETED_EXPERIMENTS if spec["family"] in signal_families]


def _load_specs(spec_path: str | None) -> list[dict[str, object]]:
    if spec_path is None:
        return DEFAULT_EXPERIMENTS + TARGETED_EXPERIMENTS
    with open(spec_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["experiments"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run constrained SGDFNet cutoff-safe 2026 recovery experiments.")
    parser.add_argument("--spec", help="Optional JSON experiment spec. Defaults to built-in constrained ladder.")
    parser.add_argument(
        "--output-dir",
        default="outputs/RT916_SpikeMarketLab/cutoff_recovery_batch",
        help="Root directory for recovery batch summaries.",
    )
    args = parser.parse_args()

    batch_root = Path(args.output_dir)
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_dir = batch_root / f"cutoff_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_specs(args.spec)
    if args.spec is None:
        diagnostic_specs = DEFAULT_EXPERIMENTS
        targeted_specs_seed = TARGETED_EXPERIMENTS
    else:
        diagnostic_specs = specs
        targeted_specs_seed = []

    records: list[dict[str, object]] = []
    for spec in diagnostic_specs:
        records.append(_run_one_experiment(spec, batch_dir))

    leaderboard = pd.DataFrame(records)
    baseline_row = leaderboard[leaderboard["name"] == "baseline"].iloc[0].to_dict()
    leaderboard["delta_vs_baseline_rt_capped_smape"] = leaderboard["overall_rt_capped_smape"] - float(baseline_row["overall_rt_capped_smape"])
    leaderboard["delta_vs_baseline_9_16_rt_capped_smape"] = leaderboard["segment_9_16_rt_capped_smape"] - float(baseline_row["segment_9_16_rt_capped_smape"])
    decisions = [_decision_for_record(row._asdict() if hasattr(row, "_asdict") else row.to_dict(), baseline_row) for _, row in leaderboard.iterrows()]
    leaderboard["decision"] = [d[0] for d in decisions]
    leaderboard["decision_reason"] = [d[1] for d in decisions]
    leaderboard["diagnosis_note"] = [
        _family_diagnosis_note(row._asdict() if hasattr(row, "_asdict") else row.to_dict(), baseline_row) for _, row in leaderboard.iterrows()
    ]

    targeted_specs = []
    if args.spec is None:
        targeted_specs = _select_targeted_specs(leaderboard, baseline_row)
        for spec in targeted_specs:
            records.append(_run_one_experiment(spec, batch_dir))
        if targeted_specs:
            leaderboard = pd.DataFrame(records)
            baseline_row = leaderboard[leaderboard["name"] == "baseline"].iloc[0].to_dict()
            leaderboard["delta_vs_baseline_rt_capped_smape"] = leaderboard["overall_rt_capped_smape"] - float(baseline_row["overall_rt_capped_smape"])
            leaderboard["delta_vs_baseline_9_16_rt_capped_smape"] = leaderboard["segment_9_16_rt_capped_smape"] - float(baseline_row["segment_9_16_rt_capped_smape"])
            decisions = [_decision_for_record(row._asdict() if hasattr(row, "_asdict") else row.to_dict(), baseline_row) for _, row in leaderboard.iterrows()]
            leaderboard["decision"] = [d[0] for d in decisions]
            leaderboard["decision_reason"] = [d[1] for d in decisions]
            leaderboard["diagnosis_note"] = [
                _family_diagnosis_note(row._asdict() if hasattr(row, "_asdict") else row.to_dict(), baseline_row) for _, row in leaderboard.iterrows()
            ]

    leaderboard = leaderboard.sort_values(["overall_rt_capped_smape", "segment_9_16_rt_capped_smape"], ascending=[True, True]).reset_index(drop=True)
    leaderboard.to_csv(batch_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")

    non_baseline = leaderboard[leaderboard["name"] != "baseline"].copy()
    best_candidate = non_baseline.iloc[0].to_dict() if not non_baseline.empty else baseline_row
    best_improvement = float(baseline_row["overall_rt_capped_smape"]) - float(best_candidate["overall_rt_capped_smape"])
    root_cause = "visible-feature loss problem"
    if best_candidate["family"] == "B":
        root_cause = "DA fill strategy problem dominates the corrected gap"
    elif best_candidate["family"] in {"C", "D"} and best_improvement > 0.5:
        root_cause = "visible-feature loss is the dominant issue, with some model/policy contribution"
    elif best_improvement <= 0.5:
        root_cause = "model capacity is not the first bottleneck; corrected visible-feature loss remains the main issue"

    diagnosis_lines = [
        f"- Data coverage for this loop is fixed to `2026-01-01` through `2026-05-12 00:00:00` in the current workspace dataset.",
        f"- Corrected baseline overall capped SMAPE: `{float(baseline_row['overall_rt_capped_smape']):.4f}`.",
        f"- Best diagnostic/targeted candidate: `{best_candidate['name']}` with overall capped SMAPE `{float(best_candidate['overall_rt_capped_smape']):.4f}`.",
        f"- Best absolute improvement vs corrected baseline: `{best_improvement:.4f}` capped-SMAPE points.",
        f"- Provisional root-cause judgment: `{root_cause}`.",
        f"- Targeted follow-ups executed: `{', '.join(spec['name'] for spec in targeted_specs) if targeted_specs else 'none; diagnostics did not clear the signal threshold'}`.",
        "",
        "## Family-by-family readout",
    ]
    for family in ["A", "B", "C", "D"]:
        fam = leaderboard[leaderboard["family"] == family]
        if fam.empty:
            continue
        best_fam = fam.iloc[0]
        diagnosis_lines.append(
            f"- `{family}` best = `{best_fam['name']}` | overall capped `{float(best_fam['overall_rt_capped_smape']):.4f}` | "
            f"delta vs baseline `{float(best_fam['delta_vs_baseline_rt_capped_smape']):+.4f}` | 9_16 capped `{float(best_fam['segment_9_16_rt_capped_smape']):.4f}`."
        )
    _write_markdown_report(batch_dir / "diagnosis_summary.md", "# Diagnosis Summary", diagnosis_lines)

    notes_lines = [
        "- All experiments in this batch use the corrected fixed no-leakage path: `B_D15_cutoff_walk_forward`.",
        "- No experiment reuses the old leaked realtime path.",
        "- No external data is introduced.",
        "- Each config changes one main factor family relative to the corrected baseline.",
        "",
        "## Experiment notes",
    ]
    for _, row in leaderboard.iterrows():
        notes_lines.append(
            f"- `{row['name']}` | family `{row['family']}` | stage `{row['stage']}` | decision `{row['decision']}` | "
            f"overall capped `{float(row['overall_rt_capped_smape']):.4f}` | delta `{float(row['delta_vs_baseline_rt_capped_smape']):+.4f}` | "
            f"note: {row['diagnosis_note']}"
        )
    _write_markdown_report(batch_dir / "experiment_notes.md", "# Experiment Notes", notes_lines)

    best_candidate_decision = {
        "baseline_name": baseline_row["name"],
        "baseline_run_dir": baseline_row["run_dir"],
        "baseline_overall_rt_capped_smape": float(baseline_row["overall_rt_capped_smape"]),
        "best_candidate_name": best_candidate["name"],
        "best_candidate_family": best_candidate["family"],
        "best_candidate_run_dir": best_candidate["run_dir"],
        "best_candidate_overall_rt_capped_smape": float(best_candidate["overall_rt_capped_smape"]),
        "best_candidate_delta_vs_baseline": float(best_candidate["delta_vs_baseline_rt_capped_smape"]),
        "root_cause_judgment": root_cause,
        "continue_second_round": bool(best_improvement >= 0.5),
        "second_round_condition": "worth continuing only if a family shows a stable positive signal; otherwise stop and report structural corrected-gap reasons",
        "targeted_specs_executed": [spec["name"] for spec in targeted_specs],
        "coverage_note": "workspace data ends at 2026-05-12 00:00:00; May is partial in this batch",
        "old_leaked_reference_only": 15.0,
    }
    with open(batch_dir / "best_candidate_decision.json", "w", encoding="utf-8") as f:
        json.dump(best_candidate_decision, f, ensure_ascii=False, indent=2)

    with open(batch_dir / "batch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol_tag": "B_D15_cutoff_walk_forward",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "experiments": specs,
                "artifact_files": [
                    "leaderboard.csv",
                    "experiment_notes.md",
                    "diagnosis_summary.md",
                    "best_candidate_decision.json",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(batch_dir)


if __name__ == "__main__":
    main()
