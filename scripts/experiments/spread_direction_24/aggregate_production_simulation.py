"""Aggregate isolated model batches into one auditable production simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _summary(evaluation: pd.DataFrame, elapsed_by_model: dict[str, float]) -> pd.DataFrame:
    rows = []
    for model_name, group in evaluation.groupby("model_name", sort=True):
        true = group["y_true_spread"].to_numpy(float)
        pred = group["y_pred_spread"].to_numpy(float)
        true_sign = np.sign(true)
        pred_sign = np.sign(pred)
        eligible = true_sign != 0
        correct = eligible & (true_sign == pred_sign)
        pos = true_sign > 0
        neg = true_sign < 0
        pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
        neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
        weights = np.abs(true[eligible])
        rows.append(
            {
                "model_name": model_name,
                "days": int(group["target_day"].nunique()),
                "n_slots": int(len(group)),
                "n_direction_eligible": int(eligible.sum()),
                "n_positive_actual": int(pos.sum()),
                "n_negative_actual": int(neg.sum()),
                "n_zero_actual": int((true_sign == 0).sum()),
                "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
                "positive_accuracy": pos_acc,
                "negative_accuracy": neg_acc,
                "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
                "abs_spread_weighted_direction_accuracy": float(
                    np.average(correct[eligible].astype(float), weights=weights)
                )
                if eligible.any() and weights.sum() > 0
                else math.nan,
                "mae": float(np.mean(np.abs(pred - true))),
                "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
                "elapsed_seconds": float(elapsed_by_model.get(model_name, math.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["direction_accuracy", "balanced_direction_accuracy"], ascending=False
    ).reset_index(drop=True)


def aggregate(output_root: Path, batch_roots: list[Path]) -> dict:
    if not batch_roots:
        raise ValueError("at least one --batch-root is required")
    manifests = []
    evaluations = []
    predictions = []
    models: list[str] = []
    elapsed_by_model: dict[str, float] = {}
    for batch_root in batch_roots:
        manifest = json.loads((batch_root / "range_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"batch is not complete: {batch_root}")
        if not manifest.get("production_simulation"):
            raise ValueError(f"batch is not a production simulation: {batch_root}")
        evaluation_path = batch_root / "ledger" / "evaluation_ledger.parquet"
        prediction_path = batch_root / "ledger" / "prediction_ledger.parquet"
        if not evaluation_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(f"missing batch ledger: {batch_root}")
        frame = pd.read_parquet(evaluation_path)
        pred = pd.read_parquet(prediction_path)
        batch_models = list(manifest["models"])
        if set(frame["model_name"].unique()) != set(batch_models):
            raise ValueError(f"ledger/model manifest mismatch: {batch_root}")
        overlap = set(models) & set(batch_models)
        if overlap:
            raise ValueError(f"duplicate model batches: {sorted(overlap)}")
        models.extend(batch_models)
        evaluations.append(frame)
        predictions.append(pred)
        for day in manifest["daily"]:
            for model in day["ok_models"]:
                elapsed_by_model[model] = elapsed_by_model.get(model, 0.0) + 0.0
        # Use the model-level elapsed values from the daily manifests.
        for day in manifest["selected_dates"]:
            day_manifest = json.loads(
                (batch_root / "runs" / day / "run_manifest.json").read_text(encoding="utf-8")
            )
            for model, detail in day_manifest["models"].items():
                elapsed_by_model.setdefault(model, 0.0)
                elapsed_by_model[model] += float(detail["elapsed_seconds"])
        manifests.append(manifest)

    first = manifests[0]
    for manifest in manifests[1:]:
        for key in ("start", "end", "input_scheme", "experiment_schema_version"):
            if manifest.get(key) != first.get(key):
                raise ValueError(f"batch boundary mismatch for {key}")
        if manifest["source"].get("source_sha256") != first["source"].get("source_sha256"):
            raise ValueError("batch source SHA256 mismatch")

    evaluation = pd.concat(evaluations, ignore_index=True).sort_values(
        ["target_day", "model_name", "hour_business"]
    ).reset_index(drop=True)
    prediction = pd.concat(predictions, ignore_index=True).sort_values(
        ["target_day", "model_name", "hour_business"]
    ).reset_index(drop=True)
    expected = len(first["selected_dates"]) * len(models) * 24
    if len(evaluation) != expected or len(prediction) != expected:
        raise ValueError(f"unexpected aggregate rows: evaluation={len(evaluation)}, prediction={len(prediction)}, expected={expected}")
    if evaluation[["y_pred_spread", "y_true_spread"]].isna().any().any():
        raise ValueError("aggregate ledger contains NaN spread values")

    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output_root / "ledger" / "evaluation_ledger.parquet", evaluation)
    _atomic_parquet(output_root / "ledger" / "prediction_ledger.parquet", prediction)
    _atomic_csv(output_root / "summary" / "model_summary.csv", _summary(evaluation, elapsed_by_model))
    total_elapsed = sum(float(day["elapsed_seconds"]) for manifest in manifests for day in manifest["daily"])
    run_signature = hashlib.sha256(
        json.dumps(
            {
                "batches": [m["run_signature"] for m in manifests],
                "models": models,
                "start": first["start"],
                "end": first["end"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    aggregate_manifest = {
        "pipeline": "spread_direction_24_experiment",
        "simulation_profile": "production_like_24_spread_v1",
        "production_simulation": True,
        "aggregate": True,
        "task": "spread",
        "resolution_namespace": "hourly",
        "future_production_namespace": "outputs/24/feature_store/spread",
        "experiment_schema_version": first["experiment_schema_version"],
        "status": "complete",
        "resolution": first["resolution"],
        "slots_per_day": first["slots_per_day"],
        "target_definition": first["target_definition"],
        "direction_definition": first["direction_definition"],
        "models": models,
        "input_scheme": first["input_scheme"],
        "run_signature": run_signature,
        "start": first["start"],
        "end": first["end"],
        "selected_dates": first["selected_dates"],
        "source": first["source"],
        "information_boundary": first["information_boundary"],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "execution_batches": [str(x) for x in batch_roots],
            "per_model_total_elapsed_seconds": elapsed_by_model,
            "total_day_model_elapsed_seconds": total_elapsed,
        },
        "daily": [
            {"target_day": day, "status": "complete", "ok_models": models, "failed_models": [], "source_batches": [str(x) for x in batch_roots]}
            for day in first["selected_dates"]
        ],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_root / "range_manifest.json", aggregate_manifest)
    return aggregate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-root", required=True, action="append", type=Path)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.output_root, args.batch_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
