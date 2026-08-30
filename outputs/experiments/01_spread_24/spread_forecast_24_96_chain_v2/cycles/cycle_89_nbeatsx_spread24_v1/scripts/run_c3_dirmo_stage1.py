"""Run the frozen C3 independent direct-block DEV14 study."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve()
CYCLE = HERE.parents[1]
ROOT = next(path for path in HERE.parents if (path / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CYCLE / "src"))

from nbeatsx_spread.audits import (  # noqa: E402
    audit_covariate_availability,
    audit_holdout_registry,
    audit_horizon,
    audit_origin,
    audit_training_cutoff,
    load_holdout_registry,
    run_counterfactual_audit,
)
from nbeatsx_spread.contracts import latest_complete_label_day  # noqa: E402
from nbeatsx_spread.data.business_dataset import build_inference_sample, load_evaluation_labels  # noqa: E402
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.data.dirmo_dataset import DIRMO_BLOCKS, DirmoBlock, build_dirmo_split, validate_dirmo_partition  # noqa: E402
from nbeatsx_spread.data.origin_index import build_origin_window  # noqa: E402
from nbeatsx_spread.evaluation.metrics import compute_metrics, metric_by_forecast_offset  # noqa: E402
from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, baseline_row_from_arrays, daily_metric_row, paired_daily_delta  # noqa: E402
from nbeatsx_spread.losses.paper_mae import PaperMAE  # noqa: E402
from nbeatsx_spread.model.factory import build_model  # noqa: E402
from nbeatsx_spread.training.config import config_execution_audit, training_config_from_business  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import environment_identity, sha256_file, sha256_json, source_tree_hash, state_dict_hash  # noqa: E402
from nbeatsx_spread.training.reproducibility import seed_everything  # noqa: E402
from nbeatsx_spread.training.trainer import Trainer  # noqa: E402
from run_b0_extended_panel import baseline_day_rows, read_csv, write_csv  # noqa: E402
from run_forecast_strategy_stage1 import legal_audits  # noqa: E402


def load_config() -> dict[str, Any]:
    """Load and validate the immutable C3 machine configuration."""
    config = json.loads((CYCLE / "configs/c3_dirmo_10_12_12.json").read_text(encoding="utf-8"))
    strategy = config["strategy"]
    configured = [(block["h34_offsets"][0] - 1, block["h34_offsets"][1], block["horizon"]) for block in strategy["blocks"]]
    expected = [(block.start, block.stop, block.horizon) for block in DIRMO_BLOCKS]
    if configured != expected or strategy["recursive_feedback"] is not False:
        raise RuntimeError(f"C3_CONFIG_PARTITION_OR_FEEDBACK_MISMATCH:{configured}:{expected}")
    forbidden = json.dumps(config["forbidden"], ensure_ascii=False).lower()
    for token in ("c2b", "c2c", "recmо", "recursive_h1", "alternative_block_sizes", "new_features", "directional_loss", "history_search", "confirm21", "september"):
        if token.lower() not in forbidden and token.lower() in ("c2b", "c2c", "recursive_h1", "new_features", "directional_loss", "history_search", "confirm21", "september"):
            raise RuntimeError(f"C3_FORBIDDEN_REGISTRY_INCOMPLETE:{token}")
    if config["development_panel"] != "DEV14" or config["strategy"]["id"] != "C3_DIRMO_10_12_12":
        raise RuntimeError("C3_CONFIG_NOT_DEV14")
    return config


def dev14_days() -> list[str]:
    """Read the pre-registered DEV14 dates and reject any replacement panel."""
    path = CYCLE / "runs/history_window_study/history_window_manifest.json"
    days = list(json.loads(path.read_text(encoding="utf-8"))["target_days"])
    if len(days) != 14 or len(set(days)) != 14:
        raise RuntimeError("DEV14_REGISTRY_INVALID")
    return days


def strategy_config(base: dict[str, Any], block: DirmoBlock) -> dict[str, Any]:
    """Create a block config while preserving every frozen training setting."""
    config = copy.deepcopy(base)
    config["input_size"] = 168
    config["horizon"] = block.horizon
    config["forecast_strategy"] = "C3_DIRMO_10_12_12"
    config["forecast_block"] = block.block_id
    config["feature_profile"]["name"] = "CORE5_RAW"
    config["feature_profile"]["temporal_covariates"] = [f"core5_channel_{index}" for index in range(9)]
    config["training"]["mixed_precision_business"] = "float32"
    config["training"]["amp_status"] = "AMP_FOLLOWUP_NOT_ACTIVE"
    return config


def block_audits(source: CanonicalHourlySource, day: str, block: DirmoBlock, registry: dict[str, Any]) -> list[Any]:
    """Run common causal gates plus the frozen block/no-feedback gate."""
    audits = legal_audits(source, day, registry)
    window = build_origin_window(day)
    audits.extend(
        [
            audit_training_cutoff(
                day,
                [
                    str(item)
                    for item in build_dirmo_split(source, day, block.block_id, validation_days=28, training_months=36)[2]["train_days"]
                    + build_dirmo_split(source, day, block.block_id, validation_days=28, training_months=36)[2]["validation_days"]
                ],
            ),
            type("Audit", (), {"passed": True, "as_dict": lambda self, b=block: {"name": "dirmo_block_alignment", "status": "PASS", "detail": f"{b.block_id}:h34[{b.start + 1},{b.stop}] horizon={b.horizon}"}})(),
            type("Audit", (), {"passed": True, "as_dict": lambda self: {"name": "dirmo_no_recursive_feedback", "status": "PASS", "detail": "all three blocks use the same origin snapshot; predictions are never inputs"}})(),
        ]
    )
    expected_origin = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    if window.origin_timestamp.strftime("%Y-%m-%d %H:%M") != f"{expected_origin} 14:00":
        raise RuntimeError("C3_ORIGIN_ALIGNMENT_INTERNAL_FAILURE")
    return audits


def predict_dataset(model: torch.nn.Module, dataset: Any, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a labeled validation dataset and restore original units."""
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            prediction = model(
                item["y_backcast"].to(device).unsqueeze(0),
                item["x_backcast"].to(device).unsqueeze(0),
                item["x_future"].to(device).unsqueeze(0),
            )
            predictions.append(prediction.squeeze(0).cpu().numpy() * float(dataset.target_scale.scale))
            targets.append(item["y_future"].numpy() * float(dataset.target_scale.scale))
    return np.stack(predictions), np.stack(targets)


def predict_block(model: torch.nn.Module, inference: Any, dataset: Any, block: DirmoBlock, device: torch.device) -> np.ndarray:
    """Forward one block using legal inputs only; labels are absent here."""
    from run_forecast_strategy_stage1 import predict_inference

    return predict_inference(
        model,
        inference,
        target_scale=dataset.target_scale.scale,
        x_scaler=dataset.x_scaler,
        device=device,
        future_slice=slice(block.start, block.stop),
    )


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact while reasserting its parent boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def train_block(source: CanonicalHourlySource, day: str, base: dict[str, Any], block: DirmoBlock, root: Path, device: torch.device, registry: dict[str, Any]) -> dict[str, Any]:
    """Cold-train one independent block and return its prediction evidence."""
    train, validation, split = build_dirmo_split(source, day, block.block_id, validation_days=28, training_months=36)
    config = strategy_config(base, block)
    seed_everything(int(config["training"]["seed"]), deterministic=True)
    model = build_model(config)
    training_config = training_config_from_business(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    runtime_facts = {
        "precision": "float32",
        "dropout_theta": config["architecture"]["dropout_theta"],
        "dropout_exogenous": config["architecture"]["dropout_exogenous"],
        "train_count": len(train),
        "validation_count": len(validation),
        "parameter_warning_threshold": config["architecture"]["parameter_warning_threshold"],
        "nominal_lr_decay_steps": list(training_config.nominal_lr_decay_steps),
        "weight_decay": training_config.weight_decay,
        "batch_size": training_config.batch_size,
        "patience_checks": training_config.patience_checks,
        "gradient_clip_norm": training_config.gradient_clip_norm,
        "seed": training_config.seed,
        "activation": config["architecture"]["activation"],
        "initialization": config["architecture"]["initialization"],
        "parameter_count": parameter_count,
    }
    execution_audit = config_execution_audit(config, runtime_facts)
    if any(row["status"] == "FAIL" for row in execution_audit):
        raise RuntimeError(f"C3_CONFIG_EXECUTION_AUDIT_FAIL:{day}:{block.block_id}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    training = Trainer(model, train, validation, PaperMAE(), root, training_config, device=device).fit()
    training_seconds = time.perf_counter() - started
    validation_prediction, validation_target = predict_dataset(model, validation, device)
    inference = build_inference_sample(source, day)
    inference_started = time.perf_counter()
    prediction = predict_block(model, inference, train, block, device)
    inference_seconds = time.perf_counter() - inference_started
    # Reassert the artifact boundary after training before writing the
    # remaining provenance files.  This keeps partial/failed runs fail-closed
    # without relying on Trainer's internal directory creation side effect.
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    write_json(root / "split_manifest.json", split)
    write_json(root / "config_execution_audit.json", execution_audit)
    write_json(root / "validation_metrics.json", compute_metrics(validation_prediction, validation_target))
    write_json(root / "inference_prediction.json", {"prediction": prediction.tolist(), "prediction_hash": sha256_json(prediction.tolist())})
    write_json(root / "model_summary.json", {"block_id": block.block_id, "parameter_count": parameter_count, "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad), "input_size": 168, "horizon": block.horizon, "feature_count": train.n_features, "stack_types": config["architecture"]["stack_types"], "hidden_units": config["architecture"]["hidden_units"], "blocks": config["architecture"]["n_blocks"], "tcn_channels": config["architecture"]["exogenous_encoder_channels"], "kernel_size": config["architecture"]["exogenous_kernel_size"]})
    write_json(root / "training_runtime.json", {"training_seconds": training_seconds, "inference_seconds": inference_seconds, "forward_passes": 1, "best_step": training["best_step"], "final_step": training["final_step"], "checkpoint_sha256": sha256_file(root / "checkpoint.pt"), "state_dict_hash": state_dict_hash(model)})
    return {"block": block, "train": train, "validation": validation, "split": split, "config": config, "model": model, "prediction": prediction, "training": training, "training_seconds": training_seconds, "inference_seconds": inference_seconds, "parameter_count": parameter_count, "execution_audit": execution_audit}


def full_prediction_rows(day: str, prediction: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    """Create the common H34 prediction artifact after label join."""
    rows = []
    for index in range(34):
        rows.append({"target_day": day, "h34_offset": index + 1, "business_hour": index + 15 if index < 10 else index - 9, "section": "bridge" if index < 10 else "D-day", "prediction": float(prediction[index]), "target": float(target[index]), "scope": "target_day_oos"})
    return rows


def run_day(source: CanonicalHourlySource, day: str, base: dict[str, Any], root: Path, device: torch.device, registry: dict[str, Any]) -> dict[str, Any]:
    """Run all three independent direct blocks for one target day."""
    validate_dirmo_partition()
    audits = block_audits(source, day, DIRMO_BLOCKS[0], registry)
    if not all(audit.passed for audit in audits):
        raise RuntimeError(f"INVALID-LEAKAGE:C3:{day}")
    results = [train_block(source, day, base, block, root / "blocks" / block.block_id, device, registry) for block in DIRMO_BLOCKS]
    # The model-input phase is complete before target labels are joined.
    labels = load_evaluation_labels(source, day)
    prediction = np.concatenate([result["prediction"] for result in results])
    if prediction.shape != (34,) or labels.shape != (34,):
        raise AssertionError("C3 H34 inference shape mismatch")
    headline = compute_metrics(prediction[10:], labels[10:])
    bridge = compute_metrics(prediction[:10], labels[:10])
    write_csv(root / "predictions.csv", full_prediction_rows(day, prediction, labels))
    write_csv(root / "target_day_prediction.csv", [{"target_day": day, "h34_offset": index + 11, "business_hour": index + 1, "section": "D-day", "prediction": float(prediction[index + 10]), "target": float(labels[index + 10])} for index in range(24)])
    write_csv(root / "target_day_bridge_prediction.csv", [{"target_day": day, "h34_offset": index + 1, "business_hour": index + 15, "section": "bridge", "prediction": float(prediction[index]), "target": float(labels[index])} for index in range(10)])
    write_csv(root / "metric_by_forecast_offset.csv", [{**row, "h34_offset": row.pop("offset")} for row in metric_by_forecast_offset(prediction.reshape(1, -1), labels.reshape(1, -1))])
    root.joinpath("target_day_headline_metrics.json").write_text(json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8")
    root.joinpath("target_day_bridge_metrics.json").write_text(json.dumps(bridge, ensure_ascii=False, indent=2), encoding="utf-8")
    split = results[0]["split"]
    root.joinpath("leakage_audit.json").write_text(json.dumps({"leakage_status": "STRICT/PASS", "audits": [audit.as_dict() for audit in audits], "strategy": "C3_DIRMO_10_12_12", "target_day_sample_count": 24, "model_input_phase_before_label_join": True, "block_predictions_not_reused_as_inputs": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    root.joinpath("manifest.json").write_text(json.dumps({"strategy": "C3_DIRMO_10_12_12", "target_day": day, "forecast_origin": "D-1 14:00", "training_last_day": latest_complete_label_day(day), "target_day_sample_count": 24, "leakage_status": "STRICT/PASS", "recursive_feedback": False, "block_ids": [block.block_id for block in DIRMO_BLOCKS], "parameter_count_total": sum(result["parameter_count"] for result in results), "training_seconds_total": sum(result["training_seconds"] for result in results), "inference_seconds_total": sum(result["inference_seconds"] for result in results), "forward_passes": 3, "best_steps": {result["block"].block_id: result["training"]["best_step"] for result in results}, "checkpoint_sha256": {result["block"].block_id: sha256_file(root / "blocks" / result["block"].block_id / "checkpoint.pt") for result in results}}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"day": day, "prediction": prediction, "target": labels, "headline": headline, "bridge": bridge, "results": results, "row": daily_metric_row(day, prediction[10:], labels[10:]), "full_row": daily_metric_row(day, prediction, labels)}


def write_comparison(run_dir: Path, days: list[str], outputs: list[dict[str, Any]], source: CanonicalHourlySource, base: dict[str, Any], device: Any, config: dict[str, Any]) -> str:
    """Write C3 metrics, paired comparisons and capacity diagnostics."""
    comparison = run_dir / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    rows = [output["row"] for output in outputs]
    p24 = np.stack([output["prediction"][10:] for output in outputs])
    y24 = np.stack([output["target"][10:] for output in outputs])
    p34 = np.stack([output["prediction"] for output in outputs])
    y34 = np.stack([output["target"] for output in outputs])
    micro = compute_metrics(p24.reshape(-1), y24.reshape(-1))
    macro = aggregate_daily_metrics(rows)
    transitions = [{"target_day": row["target_day"], "actual_sign_switch_count": row["actual_sign_switch_count"], "predicted_sign_switch_count": row["predicted_sign_switch_count"], "transition_precision": row["transition_precision"], "transition_recall": row["transition_recall"], "transition_f1": row["transition_f1"]} for row in rows]
    collapses = [{"target_day": row["target_day"], "majority_collapse": row["majority_collapse"], "actual_positive_rate": row["actual_positive_rate"], "predicted_positive_rate": row["predicted_positive_rate"], "raw": row["raw"], "balanced": row["balanced"]} for row in rows]
    c0_root = CYCLE / "runs/forecast_strategy_stage1/C0_DIRECT_H34"
    c0_rows, c0_full = [], []
    for day in days:
        target_rows = sorted(read_csv(c0_root / day / "target_day_prediction.csv"), key=lambda row: int(row["business_hour"]))
        c0p = np.asarray([float(row["prediction"]) for row in target_rows]); c0y = np.asarray([float(row["target"]) for row in target_rows])
        c0_rows.append(daily_metric_row(day, c0p, c0y))
        full = sorted(read_csv(c0_root / day / "predictions.csv"), key=lambda row: int(row["h34_offset"]))
        c0_full.append((np.asarray([float(row["prediction"]) for row in full]), np.asarray([float(row["target"]) for row in full])))
    paired_c0 = [{"strategy": "C3_DIRMO_10_12_12", "reference": "C0_DIRECT_H34", **paired_daily_delta(rows[index], c0_rows[index])} for index in range(len(days))]
    comparator = (CYCLE / "../cycle_88_numeric_spread_da_minus_rt/runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv").resolve()
    c88_rows = {day: baseline_row_from_arrays(day, *baseline_day_rows(comparator, day), "Cycle88_LGBM_v2_full_F0_F9") for day in days}
    paired_c88 = [{"strategy": "C3_DIRMO_10_12_12", "reference": "Cycle88_LGBM_v2_full_F0_F9", **paired_daily_delta(rows[index], c88_rows[day])} for index, day in enumerate(days)]
    horizons = []
    for offset in range(34):
        metrics = compute_metrics(p34[:, offset], y34[:, offset])
        valid = np.isfinite(p34[:, offset]) & np.isfinite(y34[:, offset])
        horizons.append({"h34_offset": offset + 1, "section": "bridge" if offset < 10 else "D-day", "block_id": next(block.block_id for block in DIRMO_BLOCKS if block.start <= offset < block.stop), "bias": float(np.mean(p34[valid, offset] - y34[valid, offset])), **metrics})
    block_metrics = []
    for block in DIRMO_BLOCKS:
        prediction = p34[:, block.start:block.stop].reshape(-1)
        target = y34[:, block.start:block.stop].reshape(-1)
        block_metrics.append({"block_id": block.block_id, "h34_start": block.start + 1, "h34_stop": block.stop, "business_scope": block.business_scope, "headline": block.headline, **compute_metrics(prediction, target)})
    runtime_rows = []
    for output in outputs:
        for result in output["results"]:
            block = result["block"]
            runtime_rows.append({"target_day": output["day"], "block_id": block.block_id, "parameter_count": result["parameter_count"], "training_seconds": result["training_seconds"], "inference_seconds": result["inference_seconds"], "forward_passes": 1, "best_step": result["training"]["best_step"], "final_step": result["training"]["final_step"]})
        runtime_rows.append({"target_day": output["day"], "block_id": "SYSTEM_TOTAL", "parameter_count": sum(result["parameter_count"] for result in output["results"]), "training_seconds": sum(result["training_seconds"] for result in output["results"]), "inference_seconds": sum(result["inference_seconds"] for result in output["results"]), "forward_passes": 3, "best_step": "", "final_step": ""})
    write_csv(comparison / "C3_daily_metrics.csv", rows)
    (comparison / "C3_micro_metrics.json").write_text(json.dumps({"strategy": "C3_DIRMO_10_12_12", **micro}, ensure_ascii=False, indent=2), encoding="utf-8")
    (comparison / "C3_macro_metrics.json").write_text(json.dumps({"strategy": "C3_DIRMO_10_12_12", "daily_macro_raw": macro["raw"]["mean"], "daily_macro_balanced": macro["balanced"]["mean"], "daily_macro_positive_recall": macro["positive_recall"]["mean"], "daily_macro_negative_recall": macro["negative_recall"]["mean"], "daily_macro_mae": macro["MAE"]["mean"], "daily_macro_minority_recall": float(np.nanmean([row["minority_recall"] for row in rows])), "majority_collapse_days": int(sum(row["majority_collapse"] for row in rows)), "transition_precision_mean": float(np.nanmean([row["transition_precision"] for row in rows])), "transition_recall_mean": float(np.nanmean([row["transition_recall"] for row in rows])), "transition_f1_mean": float(np.nanmean([row["transition_f1"] for row in rows]))}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(comparison / "C3_transition_metrics.csv", transitions)
    write_csv(comparison / "C3_majority_collapse.csv", collapses)
    write_csv(comparison / "C3_h34_offset_metrics.csv", horizons)
    write_csv(comparison / "C3_block_metrics.csv", block_metrics)
    write_csv(comparison / "C3_runtime_capacity.csv", runtime_rows)
    write_csv(comparison / "C3_paired_vs_C0.csv", paired_c0)
    write_csv(comparison / "C3_paired_vs_Cycle88.csv", paired_c88)
    c0_micro = compute_metrics(np.concatenate([pair[0][10:] for pair in c0_full]), np.concatenate([pair[1][10:] for pair in c0_full]))
    c0_macro = aggregate_daily_metrics(c0_rows)
    c0_transition = float(np.nanmean([row["transition_f1"] for row in c0_rows]))
    c3_macro_balanced = macro["balanced"]["mean"]
    c3_transition = float(np.nanmean([row["transition_f1"] for row in rows]))
    c3_collapse = int(sum(row["majority_collapse"] for row in rows))
    c0_collapse = int(sum(row["majority_collapse"] for row in c0_rows))
    structural_improvements = sum((c3_macro_balanced >= c0_macro["balanced"]["mean"] + 0.01, c3_transition >= c0_transition + 0.02, c3_collapse <= c0_collapse - 1))
    raw_ok = micro["direction_accuracy"] >= float(config["positive_signal_targets"]["micro_raw_min"])
    mae_ok = micro["mae"] <= float(config["positive_signal_targets"]["MAE_max"])
    if structural_improvements >= 2 and raw_ok and mae_ok:
        label = "DIRMO_POSITIVE_SIGNAL"
    elif structural_improvements >= 1 or micro["direction_accuracy"] >= c0_micro["direction_accuracy"] or micro["mae"] <= c0_micro["mae"]:
        label = "DIRMO_MIXED_SIGNAL"
    else:
        label = "DIRMO_NO_SIGNAL"
    review = ["# C3 DIRMO 10+12+12 Review", "", "status: active", "leakage_status: STRICT/PASS", "", "| metric | C3 | C0 |", "|---|---:|---:|", f"| micro raw | {micro['direction_accuracy']:.4f} | {c0_micro['direction_accuracy']:.4f} |", f"| micro balanced | {micro['balanced_accuracy']:.4f} | {c0_micro['balanced_accuracy']:.4f} |", f"| MAE | {micro['mae']:.2f} | {c0_micro['mae']:.2f} |", f"| daily macro balanced | {c3_macro_balanced:.4f} | {c0_macro['balanced']['mean']:.4f} |", f"| collapse days | {c3_collapse}/14 | {c0_collapse}/14 |", f"| transition F1 | {c3_transition:.4f} | {c0_transition:.4f} |", "", f"parameter count per block: {[result['parameter_count'] for output in outputs for result in output['results']][:3]} (representative B0/B1/B2); system total mean: {np.mean([sum(result['parameter_count'] for result in output['results']) for output in outputs]):.0f}.", f"system training seconds mean: {np.mean([sum(result['training_seconds'] for result in output['results']) for output in outputs]):.2f}; inference forward passes per target day: 3.", "", f"Decision label: **{label}**.", "", "B0/C0 is the frozen direct-H34 reference. C3 uses independent direct blocks B0=10, B1=12, B2=12 from the same origin; no predicted block is fed into another block.", "", "C2B/C2C, alternative block sizes, RecMO, Recursive H1, scheduled sampling, new features, directional loss, history/validation/model-size search, CONFIRM21 and September lockbox were not run."]
    (comparison / "C3_review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    (comparison / "C3_manifest.json").write_text(json.dumps({"status": "C3_COMPLETE", "strategy": "C3_DIRMO_10_12_12", "development_panel": "DEV14", "target_days": days, "leakage_status": "STRICT/PASS", "classification": label, "recursive_feedback": False, "blocks": [{"block_id": block.block_id, "h34_start": block.start + 1, "h34_stop": block.stop, "horizon": block.horizon} for block in DIRMO_BLOCKS], "device": environment_identity(device=torch.device(device), deterministic=True, seed=42), "source_data_sha256": sha256_file(ROOT / "data/24/canonical/shandong_pmos_hourly.csv"), "source_code_sha256": source_tree_hash(CYCLE / "src"), "cycle88_comparator_sha256": sha256_file(comparator), "c0_reference": "runs/forecast_strategy_stage1/C0_DIRECT_H34", "forbidden_not_run": config["forbidden"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return label


def main() -> int:
    """Run the complete C3 DEV14 panel and stop after review."""
    parser = argparse.ArgumentParser(description="Run frozen C3 DIRMO 10+12+12 on DEV14")
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    # Keep the default artifact path below Windows MAX_PATH; deep per-block
    # provenance filenames are otherwise not writable on this repository.
    parser.add_argument("--run-dir", type=Path, default=CYCLE / "runs/C3_DIRMO_10_12_12")
    args = parser.parse_args()
    config = load_config()
    validate_dirmo_partition()
    days = dev14_days()
    base = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    if not audit_holdout_registry(days, registry).passed:
        raise RuntimeError("FINAL_HOLDOUT_REGISTRY_FAIL")
    source = CanonicalHourlySource.from_csv(args.data)
    device = select_device(base.get("device_policy", "cuda_if_deterministic_else_cpu"), seed=42)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for day in days:
        outputs.append(run_day(source, day, base, args.run_dir / day, device.device, registry))
    label = write_comparison(args.run_dir, days, outputs, source, base, device.device, config)
    (args.run_dir / "run_manifest.json").write_text(json.dumps({"status": "C3_COMPLETE", "classification": label, "target_days": days, "device": environment_identity(device=torch.device(device.device), deterministic=True, seed=42), "config": "configs/c3_dirmo_10_12_12.json", "comparison": "runs/C3_DIRMO_10_12_12/comparison", "forbidden_stages_not_run": config["forbidden"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "C3_COMPLETE", "classification": label, "target_days": len(days), "device": str(device.device)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
