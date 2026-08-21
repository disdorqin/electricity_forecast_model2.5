"""Merge improved CPU predictions with unchanged server 96-point models.

The script is deliberately strict: it refuses to merge until every requested
day has complete replacement outputs for the selected LightGBM/TimesFM legs.
The server ledger remains read-only and the merged result is written below
``outputs/experiments``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct execution from the repository root without installing the
# project as a package.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.model_pool import models_for_task


REPLACEMENTS = {
    "dayahead": {"lightgbm", "timesfm"},
    "realtime": {"timesfm"},
}


def days_between(start: str, end: str) -> list[str]:
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    result = []
    while current <= stop:
        result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_new_day(new_runs: Path, task: str, target_day: str) -> pd.DataFrame:
    path = new_runs / target_day / task / "prediction" / "all_model_predictions_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing improved output: {path}")
    frame = pd.read_csv(path)
    required = {"model_name", "business_period", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    expected = REPLACEMENTS[task]
    frame = frame[frame["model_name"].isin(expected)].copy()
    if set(frame["model_name"].unique()) != expected:
        raise ValueError(
            f"{path}: replacement models={sorted(frame['model_name'].unique())}, "
            f"expected={sorted(expected)}"
        )
    if len(frame) != len(expected) * 96:
        raise ValueError(f"{path}: rows={len(frame)}, expected={len(expected) * 96}")
    for model, group in frame.groupby("model_name"):
        slots = set(pd.to_numeric(group["business_period"], errors="coerce").astype(int))
        if slots != set(range(1, 97)) or group["y_pred"].isna().any():
            raise ValueError(f"{path}: incomplete/non-finite {task}/{model}")
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-ledger", required=True, type=Path)
    ap.add_argument("--improved-runs", required=True, type=Path)
    ap.add_argument("--output-ledger", required=True, type=Path)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    target_days = days_between(args.start, args.end)
    output = args.output_ledger
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = []

    for task in ("dayahead", "realtime"):
        pred_path = args.server_ledger / task / "prediction" / "prediction_ledger.parquet"
        actual_path = args.server_ledger / task / "actual" / "actual_ledger.parquet"
        if not pred_path.exists() or not actual_path.exists():
            raise FileNotFoundError(f"server ledger missing for {task}")
        old = pd.read_parquet(pred_path)
        old = old[old["target_day"].astype(str).isin(target_days)].copy()
        old = old[~old["model_name"].isin(REPLACEMENTS[task])]

        improved = []
        for target_day in target_days:
            day = load_new_day(args.improved_runs, task, target_day)
            day["target_day"] = target_day
            day["business_day"] = target_day
            day["forecast_date"] = target_day
            day["task"] = task
            improved.append(day)
        replacement = pd.concat(improved, ignore_index=True)
        merged = pd.concat([old, replacement], ignore_index=True, sort=False)
        merged["business_period"] = pd.to_numeric(merged["business_period"], errors="raise").astype(int)
        merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="raise")
        expected_models = list(models_for_task(task))
        for target_day in target_days:
            day = merged[merged["target_day"].astype(str) == target_day]
            if set(day["model_name"].unique()) != set(expected_models):
                raise ValueError(f"{task}/{target_day}: model set mismatch")
            for model in expected_models:
                part = day[day["model_name"] == model]
                if len(part) != 96 or set(part["business_period"]) != set(range(1, 97)):
                    raise ValueError(f"{task}/{target_day}/{model}: not exactly 96 slots")
                if not np.isfinite(part["y_pred"].to_numpy(float)).all():
                    raise ValueError(f"{task}/{target_day}/{model}: non-finite prediction")
        task_dir = output / task
        prediction_dir = task_dir / "prediction"
        actual_dir = task_dir / "actual"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        actual_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(prediction_dir / "prediction_ledger.parquet", index=False)
        merged.to_csv(prediction_dir / "prediction_ledger.csv", index=False)
        actual = pd.read_parquet(actual_path)
        actual = actual[actual["target_day"].astype(str).isin(target_days)].copy()
        actual.to_parquet(actual_dir / "actual_ledger.parquet", index=False)
        actual.to_csv(actual_dir / "actual_ledger.csv", index=False)
        audit.append({"task": task, "rows": len(merged), "days": len(target_days), "models": expected_models})

    manifest = {
        "status": "complete",
        "resolution": "15min",
        "start": args.start,
        "end": args.end,
        "days": len(target_days),
        "server_ledger": str(args.server_ledger),
        "improved_runs": str(args.improved_runs),
        "replacement_models": {k: sorted(v) for k, v in REPLACEMENTS.items()},
        "audit": audit,
        "input_prediction_sha256": {
            task: sha256(args.server_ledger / task / "prediction" / "prediction_ledger.parquet")
            for task in ("dayahead", "realtime")
        },
    }
    (output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
