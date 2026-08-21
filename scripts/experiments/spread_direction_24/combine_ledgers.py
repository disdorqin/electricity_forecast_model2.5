"""Combine disjoint model ledgers for an isolated spread experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _load(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_parquet(path)
    required = {
        "target_day", "ds", "hour_business", "period", "model_name",
        "y_pred_spread", "y_true_spread", "predicted_direction", "direction_correct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing columns={missing}")
    frame = frame.copy()
    frame["target_day"] = pd.to_datetime(frame["target_day"]).dt.strftime("%Y-%m-%d")
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
    if frame[["y_pred_spread", "y_true_spread"]].isna().any().any():
        raise ValueError(f"{path} contains NaN spread values")
    manifest_path = path.parents[1] / "range_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return frame, manifest


def combine(output_root: Path, ledger_paths: list[Path], input_scheme: str) -> dict:
    if len(ledger_paths) < 2:
        raise ValueError("at least two ledgers are required")
    frames = []
    manifests = []
    seen_models: set[str] = set()
    for path in ledger_paths:
        frame, manifest = _load(path)
        models = set(frame["model_name"].unique())
        overlap = seen_models & models
        if overlap:
            raise ValueError(f"duplicate model names across ledgers: {sorted(overlap)}")
        seen_models |= models
        counts = frame.groupby(["target_day", "model_name"]).size()
        if not (counts == 24).all():
            raise ValueError(f"incomplete model-day in {path}")
        frames.append(frame)
        manifests.append(manifest)

    first = frames[0]
    dates = sorted(first["target_day"].unique())
    truth = first.drop_duplicates(["target_day", "ds", "hour_business", "period"])[
        ["target_day", "ds", "hour_business", "period", "y_true_spread"]
    ].set_index(["target_day", "ds", "hour_business", "period"])
    for frame in frames[1:]:
        other = frame.drop_duplicates(["target_day", "ds", "hour_business", "period"])[
            ["target_day", "ds", "hour_business", "period", "y_true_spread"]
        ].set_index(["target_day", "ds", "hour_business", "period"])
        aligned = other.reindex(truth.index)
        if aligned["y_true_spread"].isna().any() or not aligned["y_true_spread"].eq(truth["y_true_spread"]).all():
            raise ValueError("truth labels differ across ledgers")

    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["target_day", "model_name", "hour_business"]
    ).reset_index(drop=True)
    expected = len(dates) * len(seen_models) * 24
    if len(merged) != expected:
        raise ValueError(f"aggregate rows={len(merged)}, expected={expected}")
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output_root / "ledger" / "evaluation_ledger.parquet", merged)
    prediction_cols = [c for c in merged.columns if c not in {
        "y_true_dayahead", "y_true_realtime", "actual_direction",
        "predicted_direction", "direction_eligible", "direction_correct",
    }]
    _atomic_parquet(output_root / "ledger" / "prediction_ledger.parquet", merged[prediction_cols])
    signature = hashlib.sha256(
        json.dumps({"ledgers": [str(x) for x in ledger_paths], "models": sorted(seen_models)}, sort_keys=True).encode()
    ).hexdigest()[:20]
    manifest = {
        "pipeline": "spread_direction_24_experiment",
        "simulation_profile": "screening_24_spread_v3",
        "aggregate": True,
        "status": "complete",
        "task": "spread",
        "resolution_namespace": "hourly",
        "input_scheme": input_scheme,
        "models": sorted(seen_models),
        "start": dates[0],
        "end": dates[-1],
        "selected_dates": dates,
        "rows": int(len(merged)),
        "run_signature": signature,
        "source_ledgers": [str(x.resolve()) for x in ledger_paths],
        "source_manifests": manifests,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_root / "range_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--input-scheme", required=True)
    parser.add_argument("--ledger", action="append", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(combine(args.output_root, args.ledger, args.input_scheme), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
