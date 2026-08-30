"""Run the frozen C0/C1/C2A forecast-strategy screen on DEV14."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve()
CYCLE = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(CYCLE / "src"))

from nbeatsx_spread.audits import (audit_covariate_availability, audit_holdout_registry, audit_horizon, audit_origin,
                                   audit_training_cutoff, load_holdout_registry, run_counterfactual_audit)  # noqa: E402
from nbeatsx_spread.audits.lineage import tensor_hash  # noqa: E402
from nbeatsx_spread.contracts import latest_complete_label_day  # noqa: E402
from nbeatsx_spread.data.business_dataset import build_business_split, build_inference_sample, load_evaluation_labels  # noqa: E402
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.data.origin_index import build_origin_window  # noqa: E402
from nbeatsx_spread.data.strategy_dataset import StrategyDataset, build_strategy_split  # noqa: E402
from nbeatsx_spread.evaluation.metrics import compute_metrics, metric_by_forecast_offset  # noqa: E402
from nbeatsx_spread.evaluation.panel import aggregate_daily_metrics, baseline_row_from_arrays, daily_metric_row, paired_daily_delta  # noqa: E402
from nbeatsx_spread.losses.paper_mae import PaperMAE  # noqa: E402
from nbeatsx_spread.model.factory import build_model  # noqa: E402
from nbeatsx_spread.training.config import config_execution_audit, training_config_from_business  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import sha256_file, sha256_json, state_dict_hash, source_tree_hash  # noqa: E402
from nbeatsx_spread.training.reproducibility import seed_everything  # noqa: E402
from nbeatsx_spread.training.trainer import Trainer  # noqa: E402
from run_b0_extended_panel import baseline_day_rows, read_csv, write_csv  # noqa: E402
from run_business_backtest import run_one as run_c0_one  # noqa: E402


STAGE_CANDIDATES = ("C0_DIRECT_H34", "C1_GAP_DIRECT_D24", "C2A_BRIDGE_TF")


def load_matrix() -> dict[str, Any]:
    """Load and validate the frozen strategy matrix."""
    matrix = json.loads((CYCLE / "configs/forecast_strategy_stage1_matrix.json").read_text(encoding="utf-8"))
    ids = {item["id"] for item in matrix["stage1_candidates"]} | {matrix["reference"]["id"]}
    if ids != set(STAGE_CANDIDATES):
        raise RuntimeError(f"STRATEGY_MATRIX_NOT_EXACTLY_STAGE1: {sorted(ids)}")
    forbidden = json.dumps(matrix.get("forbidden_in_stage1", []), ensure_ascii=False)
    if any(x.lower() in forbidden.lower() for x in ("C2B", "C2C", "DIRMO", "RecMO", "Recursive H1", "scheduled sampling", "new features", "directional loss", "history search", "CONFIRM21", "September")):
        # The matrix is expected to list these as forbidden; this assertion
        # prevents a malformed matrix from silently broadening execution.
        expected = {"C2B_BRIDGE_OOF_MATCHED", "C2C_BRIDGE_NOISY", "C3_DIRMO", "C4_RECMO", "C5_RECURSIVE_H1", "scheduled_sampling", "RecNoisy", "new_features", "directional_loss", "history_search", "validation_search", "CONFIRM21", "September_lockbox"}
        if set(matrix.get("forbidden_in_stage1", [])) != expected:
            raise RuntimeError("STRATEGY_FORBIDDEN_MATRIX_MISMATCH")
    if matrix.get("development_panel") != "DEV14":
        raise RuntimeError("STRATEGY_STAGE1_NOT_DEV14")
    return matrix


def dev14_days() -> list[str]:
    """Use the existing frozen DEV14 date registry, never an ad-hoc panel."""
    path = CYCLE / "runs/history_window_study/history_window_manifest.json"
    days = list(json.loads(path.read_text(encoding="utf-8"))["target_days"])
    if len(days) != 14 or len(set(days)) != 14:
        raise RuntimeError("DEV14_REGISTRY_INVALID")
    return days


def strategy_config(base: dict[str, Any], strategy: str, horizon: int, n_features: int = 9) -> dict[str, Any]:
    """Set only the output formulation and its explicit input width."""
    config = copy.deepcopy(base)
    config["input_size"] = 168; config["horizon"] = horizon
    config["forecast_strategy"] = strategy
    config["feature_profile"]["name"] = "CORE5_RAW" if n_features == 9 else "CORE5_RAW_PLUS_BRIDGE_CONTEXT"
    config["feature_profile"]["temporal_covariates"] = [f"core5_channel_{i}" for i in range(9)] + ([f"bridge_context_{i+1}" for i in range(10)] if n_features == 19 else [])
    config["training"]["mixed_precision_business"] = "float32"
    config["training"]["amp_status"] = "AMP_FOLLOWUP_NOT_ACTIVE"
    return config


def legal_audits(source: CanonicalHourlySource, day: str, registry: dict[str, Any]) -> list[Any]:
    """Run common strategy-independent gates before any model is trained."""
    window = build_origin_window(day)
    audits = [audit_origin(day, window), audit_horizon(window), audit_covariate_availability(source, day), *run_counterfactual_audit(source, day), audit_holdout_registry([day], registry)]
    return audits


def predict_dataset(model: torch.nn.Module, dataset: StrategyDataset, device: str | torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Predict a labeled strategy dataset in deterministic evaluation mode."""
    model.eval(); predictions, targets = [], []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            pred = model(item["y_backcast"].to(device).unsqueeze(0), item["x_backcast"].to(device).unsqueeze(0), item["x_future"].to(device).unsqueeze(0))
            predictions.append(pred.squeeze(0).cpu().numpy() * float(dataset.target_scale.scale))
            targets.append(item["y_future"].numpy() * float(dataset.target_scale.scale))
    return np.stack(predictions), np.stack(targets)


def predict_inference(model: torch.nn.Module, inference: Any, *, target_scale: float, x_scaler: Any, device: str | torch.device, future_slice: slice, context: np.ndarray | None = None, context_scale: float | None = None) -> np.ndarray:
    """Forward only legal input tensors; labels are intentionally absent."""
    x_back = x_scaler.transform(inference.x_backcast)
    x_future = x_scaler.transform(inference.x_future[future_slice])
    if context is not None:
        if context_scale is None or context_scale <= 0:
            raise ValueError("invalid bridge context scale")
        normalized = np.asarray(context, dtype=np.float32) / float(context_scale)
        x_back = np.concatenate([x_back, np.repeat(normalized[None, :], len(x_back), axis=0)], axis=1)
        x_future = np.concatenate([x_future, np.repeat(normalized[None, :], len(x_future), axis=0)], axis=1)
    batch = {
        "y_backcast": torch.from_numpy(inference.y_backcast / float(target_scale)).float().unsqueeze(0).to(device),
        "x_backcast": torch.from_numpy(x_back).float().unsqueeze(0).to(device),
        "x_future": torch.from_numpy(x_future).float().unsqueeze(0).to(device),
    }
    model.eval()
    with torch.no_grad():
        return model(**batch).squeeze(0).cpu().numpy() * float(target_scale)


def train_stage(day: str, source: CanonicalHourlySource, base: dict[str, Any], strategy: str, stage: str, root: Path, device: str | torch.device) -> dict[str, Any]:
    """Cold-train one C1 or C2A stage and return its inference evidence."""
    root.mkdir(parents=True, exist_ok=True)
    train, val, split = build_strategy_split(source, day, strategy=strategy, stage=stage, validation_days=28, training_months=36)
    horizon = train.horizon
    n_features = train.n_features
    config = strategy_config(base, strategy, horizon, n_features)
    seed_everything(int(config["training"]["seed"]), deterministic=True)
    model = build_model(config)
    training_config = training_config_from_business(config)
    runtime = {
        "precision": "float32", "dropout_theta": config["architecture"]["dropout_theta"], "dropout_exogenous": config["architecture"]["dropout_exogenous"],
        "train_count": len(train), "validation_count": len(val), "parameter_warning_threshold": config["architecture"]["parameter_warning_threshold"],
        "nominal_lr_decay_steps": list(training_config.nominal_lr_decay_steps), "weight_decay": training_config.weight_decay, "batch_size": training_config.batch_size,
        "patience_checks": training_config.patience_checks, "gradient_clip_norm": training_config.gradient_clip_norm, "seed": training_config.seed,
        "activation": config["architecture"]["activation"], "initialization": config["architecture"]["initialization"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    execution_audit = config_execution_audit(config, runtime)
    if any(row["status"] == "FAIL" for row in execution_audit):
        raise RuntimeError(f"CONFIG_EXECUTION_AUDIT_FAIL:{strategy}/{stage}/{day}")
    result = Trainer(model, train, val, PaperMAE(), root, training_config, device=device).fit()
    val_pred, val_target = predict_dataset(model, val, device)
    inference = build_inference_sample(source, day)
    if stage == "stage1":
        inference_pred = predict_inference(model, inference, target_scale=train.target_scale.scale, x_scaler=train.x_scaler, device=device, future_slice=slice(0, 10))
    else:
        raise AssertionError("stage2 inference requires an explicit bridge context")
    (root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "split_manifest.json").write_text(json.dumps(split, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (root / "config_execution_audit.json").write_text(json.dumps(execution_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (root / "validation_metrics.json").write_text(json.dumps(compute_metrics(val_pred, val_target), ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "inference_prediction.json").write_text(json.dumps({"prediction": inference_pred.tolist(), "prediction_hash": tensor_hash(inference_pred)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"train": train, "val": val, "split": split, "config": config, "model": model, "inference_prediction": inference_pred, "training": result}


def run_c2a_day(day: str, source: CanonicalHourlySource, base: dict[str, Any], root: Path, device: str | torch.device, registry: dict[str, Any]) -> dict[str, Any]:
    """Train bridge stage then teacher-forced stage 2, predicted bridge at OOS."""
    audits = legal_audits(source, day, registry)
    if not all(a.passed for a in audits):
        raise RuntimeError(f"INVALID-LEAKAGE:C2A:{day}")
    stage1 = train_stage(day, source, base, "C2A_BRIDGE_TF", "stage1", root / "stage1", device)
    train2, val2, split2 = build_strategy_split(source, day, strategy="C2A_BRIDGE_TF", stage="stage2", validation_days=28, training_months=36)
    config2 = strategy_config(base, "C2A_BRIDGE_TF", 24, train2.n_features)
    seed_everything(int(config2["training"]["seed"]), deterministic=True)
    model2 = build_model(config2)
    training_config = training_config_from_business(config2)
    runtime = {"precision": "float32", "dropout_theta": config2["architecture"]["dropout_theta"], "dropout_exogenous": config2["architecture"]["dropout_exogenous"], "train_count": len(train2), "validation_count": len(val2), "parameter_warning_threshold": config2["architecture"]["parameter_warning_threshold"], "nominal_lr_decay_steps": list(training_config.nominal_lr_decay_steps), "weight_decay": training_config.weight_decay, "batch_size": training_config.batch_size, "patience_checks": training_config.patience_checks, "gradient_clip_norm": training_config.gradient_clip_norm, "seed": training_config.seed, "activation": config2["architecture"]["activation"], "initialization": config2["architecture"]["initialization"], "parameter_count": sum(p.numel() for p in model2.parameters())}
    execution_audit = config_execution_audit(config2, runtime)
    if any(row["status"] == "FAIL" for row in execution_audit):
        raise RuntimeError(f"CONFIG_EXECUTION_AUDIT_FAIL:C2A/stage2/{day}")
    result2 = Trainer(model2, train2, val2, PaperMAE(), root / "stage2", training_config, device=device).fit()
    inference = build_inference_sample(source, day)
    pred_bridge = stage1["inference_prediction"]
    pred_d24 = predict_inference(model2, inference, target_scale=train2.target_scale.scale, x_scaler=train2.x_scaler, device=device, future_slice=slice(10, 34), context=pred_bridge, context_scale=train2.context_scale)
    labels = load_evaluation_labels(source, day)
    bridge_metrics = compute_metrics(pred_bridge, labels[:10])
    headline = compute_metrics(pred_d24, labels[10:])
    full_pred = np.concatenate([pred_bridge, pred_d24]); full_target = labels
    write_csv(root / "target_day_prediction.csv", [{"target_day": day, "h34_offset": i + 11, "business_hour": i + 1, "section": "D-day", "prediction": float(pred_d24[i]), "target": float(labels[i + 10])} for i in range(24)])
    write_csv(root / "target_day_bridge_prediction.csv", [{"target_day": day, "h34_offset": i + 1, "business_hour": i + 15, "section": "bridge", "prediction": float(pred_bridge[i]), "target": float(labels[i])} for i in range(10)])
    write_csv(root / "predictions.csv", [{"target_day": day, "h34_offset": i + 1, "business_hour": i + 15 if i < 10 else i - 9, "section": "bridge" if i < 10 else "D-day", "prediction": float(full_pred[i]), "target": float(full_target[i]), "scope": "target_day_oos"} for i in range(34)])
    write_csv(root / "metric_by_forecast_offset.csv", [{**row, "h34_offset": row.pop("offset")} for row in metric_by_forecast_offset(full_pred.reshape(1, -1), labels.reshape(1, -1))])
    (root / "target_day_headline_metrics.json").write_text(json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "target_day_bridge_metrics.json").write_text(json.dumps(bridge_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(config2, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "leakage_audit.json").write_text(json.dumps({"leakage_status": "STRICT/PASS", "audits": [a.as_dict() for a in audits], "strategy": "C2A_BRIDGE_TF", "target_day_sample_count": 24}, ensure_ascii=False, indent=2), encoding="utf-8")
    bridge_audit = {"status": "LEGAL_TF_TRAIN_PREDICTED_BRIDGE_INFERENCE", "true_bridge_training_days_latest": split2["training_last_day"], "true_bridge_timestamps": "historical d-1 h15-h24 only", "true_bridge_inference": False, "inference_stage2_bridge_source": "STAGE1_PREDICTED_BRIDGE", "stage1_prediction_hash": tensor_hash(pred_bridge), "stage1_training_last_day": stage1["split"]["training_last_day"], "target_day": day}
    (root / "bridge_teacher_forcing_audit.json").write_text(json.dumps(bridge_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "strategy_manifest.json").write_text(json.dumps({"strategy": "C2A_BRIDGE_TF", "stage1_best_step": result_step(stage1), "stage2_best_step": result2["best_step"], "stage2_best_validation_mae": result2["best_validation_mae"], "target_day_sample_count": 24, "leakage_status": "STRICT/PASS"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "split_manifest.json").write_text(json.dumps(split2, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (root / "config_execution_audit.json").write_text(json.dumps(execution_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"prediction": pred_d24, "target": labels[10:], "full_prediction": full_pred, "full_target": labels, "bridge_prediction": pred_bridge, "bridge_target": labels[:10], "bridge_metrics": bridge_metrics, "headline": headline, "stage1": stage1, "stage2": result2, "split": split2, "bridge_audit": bridge_audit}


def result_step(info: dict[str, Any]) -> int:
    """Extract best step from a Trainer result without exposing trainer state."""
    return int(info["training"]["best_step"])


def read_target_run(root: Path, day: str, *, h34: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(root / day / "predictions.csv")
    if h34:
        rows.sort(key=lambda row: int(row["h34_offset"]))
        prediction = np.asarray([float(x["prediction"]) for x in rows], dtype=np.float32)
        target = np.asarray([float(x["target"]) for x in rows], dtype=np.float32)
        # C1 has no modeled bridge.  Keep a common H34 diagnostic coordinate
        # system by padding its ten unmodeled bridge slots with NaN.
        if len(rows) == 24 and [int(x["h34_offset"]) for x in rows] == list(range(11, 35)):
            prediction = np.concatenate([np.full(10, np.nan, dtype=np.float32), prediction])
            target = np.concatenate([np.full(10, np.nan, dtype=np.float32), target])
        if prediction.shape != (34,) or target.shape != (34,):
            raise ValueError(f"invalid H34 target run shape: {root / day}")
        return prediction, target
    rows = read_csv(root / day / "target_day_prediction.csv")
    rows.sort(key=lambda row: int(row["business_hour"]))
    return np.asarray([float(x["prediction"]) for x in rows]), np.asarray([float(x["target"]) for x in rows])


def write_comparison(args: argparse.Namespace, days: list[str], source: CanonicalHourlySource, matrix: dict[str, Any], device: Any, daily: dict[str, list[dict[str, Any]]], full_predictions: dict[str, list[np.ndarray]], full_targets: dict[str, list[np.ndarray]], c0_root: Path, c2_bridge: list[dict[str, Any]]) -> None:
    """Write common, paired, bridge and horizon diagnostics."""
    comparison = args.run_dir / "comparison"; comparison.mkdir(parents=True, exist_ok=True)
    all_daily = [row for rows in daily.values() for row in rows]
    micro_rows = []
    macro_rows = []
    for strategy in STAGE_CANDIDATES:
        # Headline metrics are always the scored D-day 24 points.  The full
        # arrays are retained only for H34 diagnostics and bridge analysis.
        p = np.concatenate([values[10:] for values in full_predictions[strategy]])
        y = np.concatenate([values[10:] for values in full_targets[strategy]])
        micro = compute_metrics(p, y); macro = aggregate_daily_metrics(daily[strategy])
        micro_rows.append({"strategy": strategy, **micro})
        macro_rows.append({"strategy": strategy, "daily_macro_raw": macro["raw"]["mean"], "daily_macro_balanced": macro["balanced"]["mean"], "daily_macro_positive_recall": macro["positive_recall"]["mean"], "daily_macro_negative_recall": macro["negative_recall"]["mean"], "daily_macro_mae": macro["MAE"]["mean"], "daily_macro_balanced_std": macro["balanced"]["std"], "daily_macro_minority_recall": float(np.nanmean([row["minority_recall"] for row in daily[strategy]])), "majority_collapse_days": int(sum(row["majority_collapse"] for row in daily[strategy])), "transition_f1_mean": float(np.nanmean([row["transition_f1"] for row in daily[strategy]])), "transition_recall_mean": float(np.nanmean([row["transition_recall"] for row in daily[strategy]]))})
    c0_rows = {row["target_day"]: row for row in daily["C0_DIRECT_H34"]}
    paired = []
    for strategy in ("C1_GAP_DIRECT_D24", "C2A_BRIDGE_TF"):
        for row in daily[strategy]:
            paired.append({"strategy": strategy, "reference": "C0_DIRECT_H34", **paired_daily_delta(row, c0_rows[row["target_day"]])})
    comparator = (CYCLE / "../cycle_88_numeric_spread_da_minus_rt/runs/cross_month_2026_01_08_14_lgbm_v2_full/predictions.csv").resolve()
    if not comparator.exists(): raise FileNotFoundError(comparator)
    c88_rows = {day: baseline_row_from_arrays(day, *baseline_day_rows(comparator, day), "Cycle88_LGBM_v2_full_F0_F9") for day in days}
    paired_c88 = []
    for strategy in STAGE_CANDIDATES:
        for row in daily[strategy]: paired_c88.append({"strategy": strategy, "reference": "Cycle88_LGBM_v2_full_F0_F9", **paired_daily_delta(row, c88_rows[row["target_day"]])})
    horizons = []
    for strategy in STAGE_CANDIDATES:
        offsets = range(10, 34) if strategy == "C1_GAP_DIRECT_D24" else range(34)
        for offset in offsets:
            p = np.asarray([full_predictions[strategy][i][offset] for i in range(len(days))]); y = np.asarray([full_targets[strategy][i][offset] for i in range(len(days))]); horizons.append({"strategy": strategy, "h34_offset": offset + 1, "strategy_offset": offset - 9 if strategy == "C1_GAP_DIRECT_D24" else offset + 1, "section": "bridge" if offset < 10 else "D-day", "bias": float(np.mean(p-y)), **compute_metrics(p, y)})
    write_csv(comparison / "strategy_stage1_daily_metrics.csv", all_daily); write_csv(comparison / "strategy_stage1_micro_metrics.csv", micro_rows); write_csv(comparison / "strategy_stage1_macro_metrics.csv", macro_rows); write_csv(comparison / "strategy_stage1_transition_metrics.csv", [{"strategy": s, "target_day": r["target_day"], "actual_sign_switch_count": r["actual_sign_switch_count"], "predicted_sign_switch_count": r["predicted_sign_switch_count"], "transition_precision": r["transition_precision"], "transition_recall": r["transition_recall"], "transition_f1": r["transition_f1"]} for s in STAGE_CANDIDATES for r in daily[s]]); write_csv(comparison / "strategy_stage1_majority_collapse.csv", [{"strategy": s, "target_day": r["target_day"], "majority_collapse": r["majority_collapse"], "actual_positive_rate": r["actual_positive_rate"], "predicted_positive_rate": r["predicted_positive_rate"], "raw": r["raw"], "balanced": r["balanced"]} for s in STAGE_CANDIDATES for r in daily[s]]); write_csv(comparison / "strategy_stage1_paired_deltas.csv", paired + paired_c88); write_csv(comparison / "strategy_stage1_horizon_metrics.csv", horizons)
    c0_bridge_rows = []
    for day in days:
        raw_bridge = read_csv(c0_root / day / "target_day_bridge_prediction.csv")
        raw_bridge.sort(key=lambda row: int(row["h34_offset"]))
        bridge_pred = np.asarray([float(row["prediction"]) for row in raw_bridge])
        bridge_target = np.asarray([float(row["target"]) for row in raw_bridge])
        bridge_metric = compute_metrics(bridge_pred, bridge_target)
        c0_bridge_rows.append({"strategy": "C0_DIRECT_H34", "target_day": day, "stage1_bridge_MAE": bridge_metric["mae"], "stage1_bridge_balanced": bridge_metric["balanced_accuracy"], "stage1_bridge_raw": bridge_metric["direction_accuracy"]})
    bridge_rows = c2_bridge + c0_bridge_rows
    write_csv(comparison / "bridge_stage1_metrics.csv", bridge_rows)
    for row in bridge_rows:
        if row["strategy"] == "C2A_BRIDGE_TF":
            row["d24_mae"] = next(r["MAE"] for r in daily["C2A_BRIDGE_TF"] if r["target_day"] == row["target_day"])
    bridge_c2 = [r for r in bridge_rows if r["strategy"] == "C2A_BRIDGE_TF"]
    bridge_errors = np.asarray([float(r["stage1_bridge_MAE"]) for r in bridge_c2]); d24_errors = np.asarray([float(r["d24_mae"]) for r in bridge_c2]); corr = float(np.corrcoef(bridge_errors, d24_errors)[0,1]) if np.std(bridge_errors)>0 and np.std(d24_errors)>0 else float("nan")
    quartile_edges = np.quantile(bridge_errors, [0.25, 0.5, 0.75])
    for row in bridge_c2:
        row["bridge_error_quartile"] = int(np.digitize(float(row["stage1_bridge_MAE"]), quartile_edges, right=True) + 1)
        row["bridge_d24_error_correlation"] = corr
    write_csv(comparison / "bridge_error_to_d24_error.csv", [{**r, "d24_mae": float(r["d24_mae"])} for r in bridge_c2])
    (comparison / "strategy_stage1_manifest.json").write_text(json.dumps({"status":"FORECAST_STRATEGY_STAGE1_COMPLETE", "target_days":days, "candidates":list(STAGE_CANDIDATES), "device":device.__dict__, "matrix":"configs/forecast_strategy_stage1_matrix.json", "leakage_status":"STRICT/PASS", "forbidden_stages_not_run":json.loads(json.dumps(matrix["forbidden_in_stage1"])), "cycle88_comparator":str(comparator), "cycle88_comparator_sha256":sha256_file(comparator)}, ensure_ascii=False, indent=2), encoding="utf-8")
    # The classification uses the pre-registered four labels only.
    by = {r["strategy"]: r for r in macro_rows}; c0, c1, c2 = by["C0_DIRECT_H34"], by["C1_GAP_DIRECT_D24"], by["C2A_BRIDGE_TF"]
    c1_pos = c1["daily_macro_balanced"] >= 0.54 and sum(r["majority_collapse"] for r in daily["C1_GAP_DIRECT_D24"]) <= 5 and np.mean([r["raw"] for r in daily["C1_GAP_DIRECT_D24"]]) >= 0.66 and c1["daily_macro_mae"] <= 89.1 and np.mean([r["transition_f1"] for r in daily["C1_GAP_DIRECT_D24"]]) >= 0.20
    c2_pos = c2["daily_macro_balanced"] >= 0.54 and sum(r["majority_collapse"] for r in daily["C2A_BRIDGE_TF"]) <= 5 and np.mean([r["raw"] for r in daily["C2A_BRIDGE_TF"]]) >= 0.66 and c2["daily_macro_mae"] <= 89.1 and np.mean([r["transition_f1"] for r in daily["C2A_BRIDGE_TF"]]) >= 0.20
    label = "GAP_DIRECT_POSITIVE" if c1_pos and not c2_pos else "BRIDGE_TF_POSITIVE" if c2_pos and not c1_pos else "BOTH_MIXED" if c1_pos and c2_pos else "NO_FORMULATION_SIGNAL_YET"
    review = ["# Forecast Strategy Stage-1 Review", "", "status: active", "leakage_status: STRICT/PASS", "", "| strategy | raw | balanced | MAE | macro balanced | macro minority recall | collapse | transition F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in macro_rows:
        micro = next(x for x in micro_rows if x["strategy"] == row["strategy"]); review.append(f"| {row['strategy']} | {micro['direction_accuracy']:.4f} | {micro['balanced_accuracy']:.4f} | {micro['mae']:.2f} | {row['daily_macro_balanced']:.4f} | {row['daily_macro_minority_recall']:.4f} | {row['majority_collapse_days']}/14 | {row['transition_f1_mean']:.4f} |")
    review += ["", "## Same-date paired deltas", "", "| strategy | reference | mean Δraw | mean Δbalanced | mean ΔMAE |", "|---|---|---:|---:|---:|"]
    for reference in ("C0_DIRECT_H34", "Cycle88_LGBM_v2_full_F0_F9"):
        for strategy in STAGE_CANDIDATES:
            rows = [row for row in paired + paired_c88 if row["reference"] == reference and row["strategy"] == strategy]
            if rows:
                review.append(f"| {strategy} | {reference} | {np.mean([row['delta_raw'] for row in rows]):+.4f} | {np.mean([row['delta_balanced'] for row in rows]):+.4f} | {np.mean([row['delta_MAE'] for row in rows]):+.2f} |")
    review += ["", "## Bridge diagnostics", "", f"C2A Stage-1 bridge MAE mean: {np.mean([row['stage1_bridge_MAE'] for row in bridge_c2]):.2f}; balanced mean: {np.mean([row['stage1_bridge_balanced'] for row in bridge_c2]):.4f}; raw mean: {np.mean([row['stage1_bridge_raw'] for row in bridge_c2]):.4f}.", f"C0 bridge MAE mean: {np.mean([row['stage1_bridge_MAE'] for row in c0_bridge_rows]):.2f}; C2A bridge-error to D24-MAE correlation: {corr:.4f}.", "", "| bridge-error quartile | days | mean D24 MAE |", "|---:|---:|---:|"]
    for quartile in range(1, 5):
        values = [float(row["d24_mae"]) for row in bridge_c2 if int(row["bridge_error_quartile"]) == quartile]
        if values:
            review.append(f"| {quartile} | {len(values)} | {np.mean(values):.2f} |")
    review += ["", f"Decision label: **{label}**.", "", "C0 is the frozen DIRECT_H34 reference. C1 removes the bridge from the modeled output and directly predicts D h1-h24. C2A uses true historical bridge context in stage-2 training and stage-1 predicted bridge at inference; this is explicitly LEGAL_TF_TRAIN_PREDICTED_BRIDGE_INFERENCE with deployment mismatch.", "", "C2B/C2C, DIRMO, RecMO, recursive H1, scheduled sampling, new features, directional loss, history search, CONFIRM21 and September lockbox were not run."]
    (comparison / "strategy_stage1_review.md").write_text("\n".join(review)+"\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C0/C1/C2A strategy Stage-1 on frozen DEV14.")
    parser.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    parser.add_argument("--run-dir", type=Path, default=CYCLE / "runs/forecast_strategy_stage1")
    args = parser.parse_args(); matrix = load_matrix(); days = dev14_days(); base = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8")); registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    if not audit_holdout_registry(days, registry).passed: raise RuntimeError("FINAL_HOLDOUT_REGISTRY_FAIL")
    source = CanonicalHourlySource.from_csv(args.data); device = select_device(base.get("device_policy", "cuda_if_deterministic_else_cpu"), seed=int(base["training"]["seed"])); args.run_dir.mkdir(parents=True, exist_ok=True)
    c0_source = CYCLE / "runs/feature_study/F0_A2_CORE5"
    c0_root = args.run_dir / "C0_DIRECT_H34"
    if not c0_source.exists(): raise FileNotFoundError(c0_source)
    if c0_root.exists(): shutil.rmtree(c0_root)
    shutil.copytree(c0_source, c0_root)
    daily: dict[str, list[dict[str, Any]]] = {s: [] for s in STAGE_CANDIDATES}; full_predictions: dict[str, list[np.ndarray]] = {s: [] for s in STAGE_CANDIDATES}; full_targets: dict[str, list[np.ndarray]] = {s: [] for s in STAGE_CANDIDATES}; c2_bridge=[]
    for day in days:
        audits = legal_audits(source, day, registry); split_audit = audit_training_cutoff(day, build_business_split(source, day, validation_days=28, training_months=36)[2]["train_days"] + build_business_split(source, day, validation_days=28, training_months=36)[2]["validation_days"]); audits.append(split_audit)
        if not all(a.passed for a in audits): raise RuntimeError(f"INVALID-LEAKAGE:{day}")
        c0p, c0y = read_target_run(c0_root, day); c0fullp, c0fully = read_target_run(c0_root, day, h34=True); c0row=daily_metric_row(day,c0p,c0y); c0row.update({"strategy":"C0_DIRECT_H34","forecast_strategy":"DIRECT_H34"}); daily["C0_DIRECT_H34"].append(c0row); full_predictions["C0_DIRECT_H34"].append(c0fullp); full_targets["C0_DIRECT_H34"].append(c0fully)
        c1_root=args.run_dir/"C1_GAP_DIRECT_D24"; c1_root.mkdir(parents=True,exist_ok=True); train1,val1,split1=build_strategy_split(source,day,strategy="C1_GAP_DIRECT_D24",stage="direct",validation_days=28,training_months=36); cfg1=strategy_config(base,"C1_GAP_DIRECT_D24",24,train1.n_features); seed_everything(42,deterministic=True); model1=build_model(cfg1); tc1=training_config_from_business(cfg1); rt={"precision":"float32","dropout_theta":.05,"dropout_exogenous":.05,"train_count":len(train1),"validation_count":len(val1),"parameter_warning_threshold":2000000,"nominal_lr_decay_steps":list(tc1.nominal_lr_decay_steps),"weight_decay":tc1.weight_decay,"batch_size":tc1.batch_size,"patience_checks":tc1.patience_checks,"gradient_clip_norm":tc1.gradient_clip_norm,"seed":42,"activation":"Softplus","initialization":"orthogonal","parameter_count":sum(p.numel() for p in model1.parameters())}; audit_cfg=config_execution_audit(cfg1,rt); 
        if any(x["status"]=="FAIL" for x in audit_cfg): raise RuntimeError(f"CONFIG_EXECUTION_AUDIT_FAIL:C1:{day}")
        info1=Trainer(model1,train1,val1,PaperMAE(),c1_root/day,tc1,device=device.device).fit(); inf=build_inference_sample(source,day); p1=predict_inference(model1,inf,target_scale=train1.target_scale.scale,x_scaler=train1.x_scaler,device=device.device,future_slice=slice(10,34)); labels=load_evaluation_labels(source,day); row1=daily_metric_row(day,p1,labels[10:]); row1.update({"strategy":"C1_GAP_DIRECT_D24","forecast_strategy":"GAP_DIRECT_D24"}); daily["C1_GAP_DIRECT_D24"].append(row1); full_predictions["C1_GAP_DIRECT_D24"].append(np.concatenate([np.full(10,np.nan),p1])); full_targets["C1_GAP_DIRECT_D24"].append(np.concatenate([np.full(10,np.nan),labels[10:]]));
        write_csv(c1_root/day/"target_day_prediction.csv",[{"target_day":day,"h34_offset":i+11,"business_hour":i+1,"section":"D-day","prediction":float(p1[i]),"target":float(labels[i+10])} for i in range(24)]); write_csv(c1_root/day/"predictions.csv",[{"target_day":day,"h34_offset":i+11,"business_hour":i+1,"section":"D-day","prediction":float(p1[i]),"target":float(labels[i+10]),"scope":"target_day_oos"} for i in range(24)]); (c1_root/day/"config.json").write_text(json.dumps(cfg1,ensure_ascii=False,indent=2),encoding="utf-8"); (c1_root/day/"split_manifest.json").write_text(json.dumps(split1,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); (c1_root/day/"config_execution_audit.json").write_text(json.dumps(audit_cfg,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); (c1_root/day/"leakage_audit.json").write_text(json.dumps({"leakage_status":"STRICT/PASS","audits":[a.as_dict() for a in audits],"strategy":"C1_GAP_DIRECT_D24","target_day_sample_count":24},ensure_ascii=False,indent=2),encoding="utf-8"); (c1_root/day/"manifest.json").write_text(json.dumps({"strategy":"C1_GAP_DIRECT_D24","target_day":day,"forecast_origin":"D-1 14:00","training_last_day":latest_complete_label_day(day),"target_day_sample_count":24,"leakage_status":"STRICT/PASS","best_step":info1["best_step"],"best_validation_mae":info1["best_validation_mae"],"final_step":info1["final_step"]},ensure_ascii=False,indent=2),encoding="utf-8")
        c2root=args.run_dir/"C2A_BRIDGE_TF"/day; result=run_c2a_day(day,source,base,c2root,device.device,registry); c2row=daily_metric_row(day,result["prediction"],result["target"]); c2row.update({"strategy":"C2A_BRIDGE_TF","forecast_strategy":"BRIDGE_TF"}); daily["C2A_BRIDGE_TF"].append(c2row); full_predictions["C2A_BRIDGE_TF"].append(result["full_prediction"]); full_targets["C2A_BRIDGE_TF"].append(result["full_target"]); c2_bridge.append({"strategy":"C2A_BRIDGE_TF","target_day":day,"stage1_bridge_MAE":result["bridge_metrics"]["mae"],"stage1_bridge_balanced":result["bridge_metrics"]["balanced_accuracy"],"stage1_bridge_raw":result["bridge_metrics"]["direction_accuracy"]})
    write_comparison(args,days,source,matrix,device,daily,full_predictions,full_targets,c0_root,c2_bridge); print(json.dumps({"status":"FORECAST_STRATEGY_STAGE1_COMPLETE","candidates":list(STAGE_CANDIDATES),"target_days":len(days),"device":device.device},ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
