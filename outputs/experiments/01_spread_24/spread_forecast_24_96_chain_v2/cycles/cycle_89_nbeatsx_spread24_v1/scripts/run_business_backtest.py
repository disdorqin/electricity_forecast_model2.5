from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve()
CYCLE = HERE.parents[1]
ROOT = next(p for p in HERE.parents if (p / "utils" / "resolution.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CYCLE / "src"))

from nbeatsx_spread.audits import (  # noqa: E402
    audit_covariate_availability, audit_holdout_registry, audit_horizon,
    audit_origin, audit_training_cutoff, load_holdout_registry,
    run_counterfactual_audit,
)
from nbeatsx_spread.audits.lineage import build_input_lineage, tensor_hash  # noqa: E402
from nbeatsx_spread.contracts import latest_complete_label_day  # noqa: E402
from nbeatsx_spread.data.business_dataset import (  # noqa: E402
    build_business_split, build_inference_sample, load_evaluation_labels,
)
from nbeatsx_spread.data.canonical_source import CanonicalHourlySource  # noqa: E402
from nbeatsx_spread.data.covariates import feature_names  # noqa: E402
from nbeatsx_spread.data.origin_index import build_origin_window  # noqa: E402
from nbeatsx_spread.evaluation.metrics import compute_metrics, metric_by_forecast_offset  # noqa: E402
from nbeatsx_spread.losses.paper_mae import PaperMAE  # noqa: E402
from nbeatsx_spread.model.factory import build_model  # noqa: E402
from nbeatsx_spread.training.config import config_execution_audit, training_config_from_business  # noqa: E402
from nbeatsx_spread.training.device import select_device  # noqa: E402
from nbeatsx_spread.training.provenance import environment_identity, git_identity, sha256_file, sha256_json, state_dict_hash, source_tree_hash  # noqa: E402
from nbeatsx_spread.training.reproducibility import seed_everything  # noqa: E402
from nbeatsx_spread.training.trainer import Trainer  # noqa: E402


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _predict_dataset(model: torch.nn.Module, dataset: Any, scale: float, device: str | torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    pred, target, masks = [], [], []
    device = torch.device(device)
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            out = model(
                item["y_backcast"].to(device).unsqueeze(0),
                item["x_backcast"].to(device).unsqueeze(0),
                item["x_future"].to(device).unsqueeze(0),
            )
            pred.append(out.squeeze(0).cpu().numpy() * scale)
            target.append(item["y_future"].cpu().numpy() * scale)
            masks.append(item["score_mask"].cpu().numpy())
    return np.stack(pred), np.stack(target), np.stack(masks)


def _model_summary(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arch = config["architecture"]
    out = {
        "parameter_count": total, "trainable_parameter_count": trainable,
        "stack_types": arch["stack_types"], "hidden_units": arch["hidden_units"],
        "blocks": arch["n_blocks"], "tcn_channels": arch["exogenous_encoder_channels"],
        "kernel_size": arch["exogenous_kernel_size"], "input_size": config["input_size"],
        "horizon": config["horizon"], "feature_count": len(feature_names(config.get("feature_package", "CORE5_RAW"))),
        "feature_package": config.get("feature_package", "CORE5_RAW"),
    }
    if total > int(arch["parameter_warning_threshold"]):
        out["warning"] = "MODEL_CAPACITY_WARNING"
    return out


def run_one(
    target_day: str,
    source: CanonicalHourlySource,
    config: dict[str, Any],
    output_root: Path,
    registry: dict[str, Any],
    training_overrides: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    loss_fn: Any | None = None,
) -> dict[str, Any]:
    run_dir = output_root / target_day
    run_dir.mkdir(parents=True, exist_ok=True)
    window = build_origin_window(target_day)
    audits = [
        audit_origin(target_day, window), audit_horizon(window),
        audit_covariate_availability(source, target_day, config.get("feature_package", "CORE5_RAW")),
        *run_counterfactual_audit(source, target_day, config.get("feature_package", "CORE5_RAW")),
        audit_holdout_registry([target_day], registry),
    ]
    train, val, split = build_business_split(
        source, target_day, validation_days=int(config["validation_history_days"]),
        training_months=int(config["training_history_months"]),
        feature_profile=config.get("feature_package", "CORE5_RAW"),
    )
    audits.append(audit_training_cutoff(target_day, split["train_days"] + split["validation_days"]))
    if len(train) < int(config["min_train_daily_origins"]) or len(val) < int(config["min_validation_daily_origins"]):
        raise RuntimeError("INSUFFICIENT_DAILY_ORIGINS")
    if not all(a.passed for a in audits):
        (run_dir / "leakage_audit.json").write_text(_json({"leakage_status": "INVALID-LEAKAGE", "audits": [a.as_dict() for a in audits]}), encoding="utf-8")
        raise RuntimeError("INVALID-LEAKAGE; training is blocked")
    training_config = training_config_from_business(config, overrides=training_overrides)
    # Model construction must happen after the explicit seed in every fresh
    # process; Trainer reseeds again before the optimizer/data-loader phase.
    seed_everything(training_config.seed, deterministic=True)
    model = build_model(config)
    model_initial_hash = state_dict_hash(model)
    model_summary = _model_summary(model, config)
    runtime = {
        "precision": "float32", "dropout_theta": config["architecture"]["dropout_theta"],
        "dropout_exogenous": config["architecture"]["dropout_exogenous"],
        "train_count": len(train), "validation_count": len(val),
        "parameter_warning_threshold": config["architecture"]["parameter_warning_threshold"],
        "nominal_lr_decay_steps": list(training_config.nominal_lr_decay_steps),
        "weight_decay": training_config.weight_decay, "batch_size": training_config.batch_size,
        "patience_checks": training_config.patience_checks, "gradient_clip_norm": training_config.gradient_clip_norm,
        "seed": training_config.seed, "activation": config["architecture"]["activation"],
        "initialization": config["architecture"]["initialization"],
        "parameter_count": model_summary["parameter_count"],
    }
    execution_audit = config_execution_audit(config, runtime)
    if any(row["status"] == "FAIL" for row in execution_audit):
        raise RuntimeError("CONFIG_EXECUTION_AUDIT_FAIL")
    # ``PaperMAE`` remains the default for the original C0 runner.  Loss
    # studies inject one explicitly without duplicating the strict dataset,
    # audit, inference, and artifact contract in another runner.
    # Formal flow: audit -> train/validate -> freeze best checkpoint -> build
    # origin-safe inference tensor -> forward -> join labels only for scoring.
    objective = PaperMAE() if loss_fn is None else loss_fn
    train_info = Trainer(model, train, val, objective, run_dir, training_config, device=device).fit()
    val_pred, val_target, val_mask = _predict_dataset(model, val, float(split["target_scale"]["scale"]), device)
    val_metrics = compute_metrics(val_pred[:, 10:], val_target[:, 10:], val_mask[:, 10:])
    feature_profile = config.get("feature_package", "CORE5_RAW")
    inference = build_inference_sample(source, target_day, feature_profile=feature_profile)
    x_back = train.x_scaler.transform(inference.x_backcast)
    x_future = train.x_scaler.transform(inference.x_future)
    input_batch = {
        "y_backcast": torch.from_numpy(inference.y_backcast / float(split["target_scale"]["scale"])).float().unsqueeze(0),
        "x_backcast": torch.from_numpy(x_back).float().unsqueeze(0),
        "x_future": torch.from_numpy(x_future).float().unsqueeze(0),
    }
    input_batch = {key: value.to(device) for key, value in input_batch.items()}
    model.eval()
    with torch.no_grad():
        target_prediction = model(**input_batch).squeeze(0).cpu().numpy() * float(split["target_scale"]["scale"])
    labels = load_evaluation_labels(source, target_day)
    headline = compute_metrics(target_prediction[10:], labels[10:])
    bridge = compute_metrics(target_prediction[:10], labels[:10])
    if headline["sample_count"] != 24:
        raise AssertionError("target-day headline must contain exactly 24 scored points")
    _write_csv(run_dir / "validation_predictions.csv", [
        {"target_day": target_day, "sample": i, "h34_offset": j + 1, "prediction": float(val_pred[i, j]), "target": float(val_target[i, j]), "section": "bridge" if j < 10 else "D-day", "business_hour": j - 9 if j >= 10 else j + 15}
        for i in range(len(val_pred)) for j in range(val_pred.shape[1])
    ])
    _write_csv(run_dir / "target_day_prediction.csv", [
        {"target_day": target_day, "h34_offset": j + 11, "prediction": float(target_prediction[j + 10]), "target": float(labels[j + 10]), "business_hour": j + 1, "section": "D-day"}
        for j in range(24)
    ])
    _write_csv(run_dir / "target_day_bridge_prediction.csv", [
        {"target_day": target_day, "h34_offset": j + 1, "prediction": float(target_prediction[j]), "target": float(labels[j]), "business_hour": j + 15, "section": "bridge"}
        for j in range(10)
    ])
    _write_csv(run_dir / "predictions.csv", [
        {"target_day": target_day, "h34_offset": j + 1,
         "prediction": float(target_prediction[j]), "target": float(labels[j]),
         "section": "bridge" if j < 10 else "D-day",
         "business_hour": j + 15 if j < 10 else j - 9, "scope": "target_day_oos"}
        for j in range(34)
    ])
    offset_rows = metric_by_forecast_offset(target_prediction.reshape(1, -1), labels.reshape(1, -1))
    for row in offset_rows:
        row["h34_offset"] = row.pop("offset")
        row["business_hour"] = row["h34_offset"] + 14 if row["h34_offset"] <= 10 else row["h34_offset"] - 10
    _write_csv(run_dir / "metric_by_h34_offset.csv", offset_rows)
    _write_csv(run_dir / "metric_by_forecast_offset.csv", offset_rows)
    (run_dir / "config.json").write_text(_json(config), encoding="utf-8")
    (run_dir / "split_manifest.json").write_text(_json(split), encoding="utf-8")
    environment = environment_identity(device=torch.device(device), deterministic=True, seed=training_config.seed)
    (run_dir / "environment.json").write_text(_json(environment), encoding="utf-8")
    (run_dir / "input_lineage.json").write_text(_json(build_input_lineage(target_day, feature_profile)), encoding="utf-8")
    (run_dir / "config_execution_audit.json").write_text(_json(execution_audit), encoding="utf-8")
    (run_dir / "model_summary.json").write_text(_json(model_summary), encoding="utf-8")
    (run_dir / "validation_metrics.json").write_text(_json(val_metrics), encoding="utf-8")
    (run_dir / "target_day_headline_metrics.json").write_text(_json(headline), encoding="utf-8")
    (run_dir / "target_day_bridge_metrics.json").write_text(_json(bridge), encoding="utf-8")
    (run_dir / "headline_metrics.json").write_text(_json(headline), encoding="utf-8")
    (run_dir / "bridge_metrics.json").write_text(_json(bridge), encoding="utf-8")
    scaler_isolated = train.x_scaler.to_dict() == split["x_scaler"]
    input_hashes = {
        "y_backcast": tensor_hash(inference.y_backcast),
        "x_backcast_raw": tensor_hash(inference.x_backcast),
        "x_future_raw": tensor_hash(inference.x_future),
        "x_future_scaled": tensor_hash(x_future),
    }
    (run_dir / "leakage_audit.json").write_text(_json({
        "leakage_status": "STRICT/PASS", "audits": [a.as_dict() for a in audits],
        "inference_truth_isolation": {"passed": True, "label_join_after_forward": True,
                                      "input_tensor_hashes": input_hashes},
        "scaler_train_only": {"passed": scaler_isolated, "source": "train_split_only"},
        "input_lineage": build_input_lineage(target_day, feature_profile),
    }), encoding="utf-8")
    checkpoint_sha = hashlib.sha256((run_dir / "checkpoint.pt").read_bytes()).hexdigest()
    provenance = {
        "config_sha256": sha256_json(config),
        "source_data_sha256": sha256_file(source.path) if source.path else None,
        "source_code_sha256": source_tree_hash(CYCLE / "src"),
        "git": git_identity(ROOT),
        "device": environment,
        "model_initial_state_sha256": model_initial_hash,
        "model_best_state_sha256": state_dict_hash(model),
        "checkpoint_sha256": checkpoint_sha,
        "split_manifest_sha256": sha256_json(split),
    }
    (run_dir / "provenance.json").write_text(_json(provenance), encoding="utf-8")
    manifest = {
        "target_day": target_day, "forecast_origin": "D-1 14:00",
        "training_history_months": int(config["training_history_months"]),
        "validation_history_days": int(config["validation_history_days"]),
        "training_last_day": latest_complete_label_day(target_day),
        "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False, "final_holdout_touched": False,
        "leakage_status": "STRICT/PASS", "best_step": train_info["best_step"],
        "best_validation_mae": train_info["best_validation_mae"], "final_step": train_info["final_step"],
        "early_stopped": train_info["early_stopped"], "checkpoint_sha256": checkpoint_sha,
        "target_day_sample_count": headline["sample_count"],
    }
    (run_dir / "manifest.json").write_text(_json(manifest), encoding="utf-8")
    return manifest


def requested_days(args: argparse.Namespace) -> list[str]:
    if args.target_day:
        return sorted(set(args.target_day))
    if not args.start_date or not args.end_date:
        raise ValueError("provide --target-day or both --start-date and --end-date")
    return [d.strftime("%Y-%m-%d") for d in __import__("pandas").date_range(args.start_date, args.end_date, freq="D")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent formal strict daily OOS runner.")
    ap.add_argument("--target-day", action="append")
    ap.add_argument("--start-date")
    ap.add_argument("--end-date")
    ap.add_argument("--data", type=Path, default=ROOT / "data/24/canonical/shandong_pmos_hourly.csv")
    ap.add_argument("--run-dir", type=Path, default=CYCLE / "runs/formal_backtest")
    args = ap.parse_args()
    days = requested_days(args)
    config = json.loads((CYCLE / "configs/business_strict34_core.json").read_text(encoding="utf-8"))
    registry = load_holdout_registry(CYCLE / "configs/holdout_registry.json")
    registry_audit = audit_holdout_registry(days, registry)
    if not registry_audit.passed:
        raise RuntimeError(f"FINAL_HOLDOUT_REGISTRY_FAIL: {registry_audit.detail}")
    source = CanonicalHourlySource.from_csv(args.data)
    decision = select_device(config.get("device_policy", "cuda_if_deterministic_else_cpu"), seed=int(config["training"]["seed"]))
    manifests = [run_one(day, source, config, args.run_dir, registry, device=decision.device) for day in days]
    predictions = []
    labels = []
    for day in days:
        rows = list(csv.DictReader((args.run_dir / day / "target_day_prediction.csv").open(encoding="utf-8")))
        predictions.extend(float(row["prediction"]) for row in rows)
        labels.extend(float(row["target"]) for row in rows)
    aggregate_metrics = compute_metrics(np.asarray(predictions), np.asarray(labels))
    (args.run_dir / "aggregate_metrics.json").write_text(_json(aggregate_metrics), encoding="utf-8")
    (args.run_dir / "aggregate_manifest.json").write_text(_json({
        "status": "FORMAL_BACKTEST_PASS",
        "target_days": days,
        "runs": manifests,
        "headline_scope": "target-day D-day 24 points only",
        "sample_count": len(predictions),
        "device_decision": decision.__dict__,
    }), encoding="utf-8")
    print(_json({"status": "FORMAL_BACKTEST_PASS", "target_days": days, "sample_count_per_target_day": 24, "device": decision.device}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
