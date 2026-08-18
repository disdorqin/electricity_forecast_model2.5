"""Strict audit for 96-point prediction and replay artifacts.

This is intentionally read-only.  It is the post-run gate for the server:
it refuses partial model sets, incomplete p1..p96 data, NaN values, mixed
24/96 outputs, non-materialized prediction runs, and degraded replay output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS


EXPECTED = {"dayahead": list(DAYAHEAD_MODELS), "realtime": list(REALTIME_MODELS)}
SLOTS = 96


def _read_json(path: Path, errors: list[str]) -> dict:
    if not path.exists():
        errors.append(f"missing JSON: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return {}


def _read_csv(path: Path, errors: list[str]) -> pd.DataFrame:
    if not path.exists():
        errors.append(f"missing CSV: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        errors.append(f"cannot read {path}: {exc}")
        return pd.DataFrame()


def _check_96_frame(frame: pd.DataFrame, path: Path, value_col: str, errors: list[str]) -> None:
    if frame.empty:
        return
    slot_col = "business_period" if "business_period" in frame.columns else "period"
    if slot_col not in frame.columns:
        errors.append(f"{path}: missing 96-point slot column")
        return
    slots = pd.to_numeric(frame[slot_col], errors="coerce")
    if len(frame) != SLOTS or slots.isna().any() or sorted(slots.astype(int).unique()) != list(range(1, SLOTS + 1)):
        errors.append(f"{path}: expected exactly p1..p96, rows={len(frame)}")
    if slots.duplicated().any():
        errors.append(f"{path}: duplicate slots")
    if value_col in frame.columns:
        values = pd.to_numeric(frame[value_col], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
            errors.append(f"{path}: {value_col} contains NaN/non-finite values")


def _audit_prediction_day(run_dir: Path, target_day: str, errors: list[str]) -> None:
    manifest = _read_json(run_dir / "run_manifest.json", errors)
    if manifest.get("resolution") != "15min":
        errors.append(f"{run_dir}: manifest resolution is not 15min")
    prediction_stage = manifest.get("stages", {}).get("ledger_predict", manifest)
    if prediction_stage.get("status") != "complete":
        errors.append(f"{run_dir}: prediction stage status={prediction_stage.get('status')}")
    if prediction_stage.get("resource_mode") not in (None, "split_process"):
        errors.append(f"{run_dir}: resource_mode is not split_process")
    if prediction_stage.get("feature_store", {}).get("mode") not in (None, "materialized"):
        errors.append(f"{run_dir}: feature store mode is not materialized")

    for task, models in EXPECTED.items():
        pred_dir = run_dir / task / "prediction"
        for model in models:
            path = pred_dir / f"{model}_predictions.csv"
            frame = _read_csv(path, errors)
            _check_96_frame(frame, path, "y_pred", errors)
        long_path = pred_dir / "all_model_predictions_long.csv"
        long_frame = _read_csv(long_path, errors)
        if not long_frame.empty:
            if len(long_frame) != len(models) * SLOTS:
                errors.append(f"{long_path}: rows={len(long_frame)} expected={len(models) * SLOTS}")
            if "model_name" in long_frame.columns and set(long_frame["model_name"]) != set(models):
                errors.append(f"{long_path}: model set mismatch")
            _check_96_frame(long_frame.drop_duplicates(subset=[c for c in ("business_period", "period") if c in long_frame.columns]), long_path, "y_pred", errors)


def _audit_prediction_ledger(ledger_root: Path, dates: list[str], errors: list[str]) -> None:
    from pipelines.prediction_ledger import load_actual_ledger, load_prediction_ledger

    for task, models in EXPECTED.items():
        pred = load_prediction_ledger(ledger_root, task, dates)
        actual = load_actual_ledger(ledger_root, task, dates)
        if pred.empty:
            errors.append(f"{task}: prediction ledger empty")
        if actual.empty:
            errors.append(f"{task}: actual ledger empty")
        if not pred.empty:
            if set(pred["model_name"].dropna().unique()) != set(models):
                errors.append(f"{task}: ledger model set mismatch")
            key = [c for c in ("model_name", "target_day", "business_period") if c in pred.columns]
            if pred.duplicated(key).any():
                errors.append(f"{task}: duplicate prediction ledger key")
            for (day, model), part in pred.groupby(["target_day", "model_name"]):
                _check_96_frame(part, ledger_root / task / "prediction" / f"{day}-{model}", "y_pred", errors)
        if not actual.empty:
            for day, part in actual.groupby("target_day"):
                _check_96_frame(part, ledger_root / task / "actual" / str(day), "y_true", errors)


def _audit_replay_day(run_dir: Path, errors: list[str]) -> None:
    manifest = _read_json(run_dir / "run_manifest.json", errors)
    if manifest.get("delivery_status") != "NORMAL":
        errors.append(f"{run_dir}: replay delivery_status={manifest.get('delivery_status')}, expected NORMAL")
    stages = manifest.get("stages", {})
    for name in ("ledger_weight", "ledger_fuse", "ledger_classifier", "final_outputs"):
        if stages.get(name, {}).get("status") != "complete":
            errors.append(f"{run_dir}: {name} is not complete")
    for task in EXPECTED:
        fuse_dir = run_dir / task / "fuse"
        for name, value_col in (("fused_predictions.csv", "y_fused"), ("fused_debug.csv", "y_fused")):
            frame = _read_csv(fuse_dir / name, errors)
            _check_96_frame(frame, fuse_dir / name, value_col, errors)
        gate = _read_csv(fuse_dir / "model_quality_gate.csv", errors)
        if not gate.empty:
            if "period" not in gate.columns or gate["period"].isna().any():
                errors.append(f"{fuse_dir / 'model_quality_gate.csv'}: invalid period gate")
            for col in ("weight_gate_threshold", "active_models", "gate_fallback_used"):
                if col not in gate.columns:
                    errors.append(f"{fuse_dir / 'model_quality_gate.csv'}: missing {col}")
        weights = _read_csv(run_dir / task / "weight" / "weights.csv", errors)
        if not weights.empty and weights.select_dtypes(include=[np.number]).isna().any().any():
            errors.append(f"{run_dir / task / 'weight' / 'weights.csv'}: numeric NaN")
    final_dir = run_dir / "final"
    for name, value_col in (("dayahead_final_predictions.csv", "y_fused"), ("realtime_final_predictions.csv", "y_fused"), ("realtime_final_predictions_corrected.csv", "y_fused_corrected")):
        frame = _read_csv(final_dir / name, errors)
        _check_96_frame(frame, final_dir / name, value_col, errors)
    submission = _read_csv(final_dir / "submission_ready.csv", errors)
    if not submission.empty:
        if len(submission) != SLOTS:
            errors.append(f"{final_dir / 'submission_ready.csv'}: rows={len(submission)} expected=96")
        for col in ("dayahead_price", "realtime_price"):
            if col not in submission.columns or pd.to_numeric(submission[col], errors="coerce").isna().any():
                errors.append(f"{final_dir / 'submission_ready.csv'}: invalid {col}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/96/feature_store")
    parser.add_argument("--phase", choices=("prediction", "replay", "all"), default="prediction")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    runs_root = output_root / "runs"
    ledger_root = output_root / "ledger"
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(args.start, args.end, freq="D")]
    errors: list[str] = []
    if args.phase in ("prediction", "all"):
        range_dirs = sorted(output_root.glob("prediction_range_*") )
        if not range_dirs:
            errors.append(f"no prediction range directory under {output_root}")
        else:
            _read_json(range_dirs[-1] / "feature_store_manifest.json", errors)
        for day in dates:
            _audit_prediction_day(runs_root / day, day, errors)
        _audit_prediction_ledger(ledger_root, dates, errors)
    if args.phase in ("replay", "all"):
        for day in dates:
            _audit_replay_day(runs_root / day, errors)

    if errors:
        print(f"96 ARTIFACT AUDIT: FAIL ({len(errors)} errors)")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} more")
        return 1
    print(f"96 ARTIFACT AUDIT: PASS ({args.phase}, {len(dates)} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
