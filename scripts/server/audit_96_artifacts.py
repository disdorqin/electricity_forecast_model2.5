"""Strict audit for 96-point prediction and replay artifacts.

This is intentionally read-only.  It is the post-run gate for the server:
it refuses partial model sets, incomplete p1..p96 data, NaN values, mixed
24/96 outputs, prediction runs without strict production provenance, and degraded replay output.
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
from utils.asof_view_96 import DYNAMIC_PROTOCOL, HISTORICAL_PROXY_PROTOCOL
from scripts.server.run_96_prediction_backtest import _protocol_manifest_audit


EXPECTED = {"dayahead": list(DAYAHEAD_MODELS), "realtime": list(REALTIME_MODELS)}
SLOTS = 96


def _range_manifest_required(start: str, end: str, explicit: bool = False) -> bool:
    """Daily production audits do not require a range-run manifest."""
    return bool(explicit) or pd.Timestamp(start).normalize() != pd.Timestamp(end).normalize()


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


def _audit_prediction_day(
    run_dir: Path,
    target_day: str,
    resource_mode: str,
    errors: list[str],
) -> None:
    protocol_ok, protocol_reasons = _protocol_manifest_audit(
        run_dir.parent, target_day, resource_mode=resource_mode
    )
    if not protocol_ok:
        errors.extend(f"{run_dir}: {reason}" for reason in protocol_reasons)

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


def _audit_prediction_range_manifest(
    output_root: Path,
    start: str,
    end: str,
    resource_mode: str,
    errors: list[str],
    *,
    require_strict_forecast_vintage: bool = False,
) -> None:
    candidates = sorted(
        (output_root / "runs").glob("range_*_predict/prediction_range_manifest.json")
    )
    if not candidates:
        errors.append(f"no prediction_range_manifest.json under {output_root / 'runs'}")
        return

    matched = None
    for path in reversed(candidates):
        payload = _read_json(path, errors)
        if not payload:
            continue
        try:
            covers = (
                pd.Timestamp(payload.get("effective_start")) <= pd.Timestamp(start)
                and pd.Timestamp(payload.get("end")) >= pd.Timestamp(end)
            )
        except Exception:
            covers = False
        if covers:
            matched = (path, payload)
            break
    if matched is None:
        errors.append(f"no prediction range manifest covers {start}..{end}")
        return

    path, payload = matched
    if payload.get("status") != "complete":
        errors.append(f"{path}: status={payload.get('status')!r}, expected complete")
    execution = payload.get("execution", {})
    if execution.get("resource_mode") != resource_mode:
        errors.append(
            f"{path}: resource_mode={execution.get('resource_mode')!r} expected={resource_mode!r}"
        )
    def _int_or_none(value: object) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    expected_cpu_workers = 2 if resource_mode == "split_process" else 1
    if _int_or_none(execution.get("cpu_workers")) != expected_cpu_workers:
        errors.append(
            f"{path}: cpu_workers={execution.get('cpu_workers')!r} "
            f"expected={expected_cpu_workers} for {resource_mode}"
        )
    if _int_or_none(execution.get("gpu_workers")) != 1:
        errors.append(f"{path}: gpu_workers={execution.get('gpu_workers')!r} expected=1")
    if resource_mode == "split_process" and execution.get("dag_aware") is not True:
        errors.append(f"{path}: split_process must record dag_aware=true")
    if payload.get("serving_protocol") not in {DYNAMIC_PROTOCOL, HISTORICAL_PROXY_PROTOCOL}:
        errors.append(f"{path}: serving_protocol={payload.get('serving_protocol')!r}")
    if payload.get("serving_visibility_source") != "FeatureViewBuilder":
        errors.append(
            f"{path}: serving_visibility_source={payload.get('serving_visibility_source')!r}"
        )
    if payload.get("model_input_contract") != "DB sync -> immutable D/T snapshot -> FeatureViewBuilder -> models":
        errors.append(f"{path}: unexpected model_input_contract={payload.get('model_input_contract')!r}")
    if require_strict_forecast_vintage:
        vintage = payload.get("forecast_vintage", {})
        if not bool(vintage.get("strict_historical_vintage_proven")):
            errors.append(
                f"{path}: strict historical forecast vintage is not proven; "
                f"status={vintage.get('status')!r}"
            )


def _audit_prediction_ledger(
    ledger_root: Path,
    dates: list[str],
    errors: list[str],
    *,
    require_target_actual: bool = False,
) -> None:
    from pipelines.prediction_ledger import load_actual_ledger, load_prediction_ledger

    for task, models in EXPECTED.items():
        pred = load_prediction_ledger(ledger_root, task, dates)
        actual = load_actual_ledger(ledger_root, task, dates)
        if pred.empty:
            errors.append(f"{task}: prediction ledger empty")
        if actual.empty and require_target_actual:
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
                path = ledger_root / task / "actual" / str(day)
                if require_target_actual:
                    _check_96_frame(part, path, "y_true", errors)
                    continue
                # Live prediction does not require target-day truth to be
                # complete.  Any rows that are already settled must still be
                # well-formed, unique, finite and inside p1..p96.
                slot_col = (
                    "business_period"
                    if "business_period" in part.columns
                    else "period"
                )
                if slot_col not in part.columns:
                    errors.append(f"{path}: missing 96-point slot column")
                    continue
                slots = pd.to_numeric(part[slot_col], errors="coerce")
                if (
                    slots.isna().any()
                    or not set(slots.astype(int)).issubset(set(range(1, SLOTS + 1)))
                ):
                    errors.append(f"{path}: invalid partial target-day slot set")
                if slots.duplicated().any():
                    errors.append(f"{path}: duplicate slots")
                if "y_true" not in part.columns:
                    errors.append(f"{path}: y_true missing")
                else:
                    values = pd.to_numeric(part["y_true"], errors="coerce")
                    if not np.isfinite(
                        values.to_numpy(dtype=float, na_value=np.nan)
                    ).all():
                        errors.append(f"{path}: y_true contains NaN/non-finite values")


def _audit_replay_day(run_dir: Path, errors: list[str]) -> None:
    manifest = _read_json(run_dir / "run_manifest.json", errors)
    if manifest.get("delivery_status") != "NORMAL":
        errors.append(f"{run_dir}: replay delivery_status={manifest.get('delivery_status')}, expected NORMAL")
    stages = manifest.get("stages", {})
    for name in ("ledger_weight", "ledger_fuse", "final_outputs"):
        if stages.get(name, {}).get("status") != "complete":
            errors.append(f"{run_dir}: {name} is not complete")
    if manifest.get("classifier_policy") != "disabled_by_production_policy":
        if stages.get("ledger_classifier", {}).get("status") != "complete":
            errors.append(f"{run_dir}: ledger_classifier is not complete")
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
    final_files = [("dayahead_final_predictions.csv", "y_fused"), ("realtime_final_predictions.csv", "y_fused")]
    if manifest.get("classifier_policy") != "disabled_by_production_policy":
        final_files.append(("realtime_final_predictions_corrected.csv", "y_fused_corrected"))
    for name, value_col in final_files:
        frame = _read_csv(final_dir / name, errors)
        _check_96_frame(frame, final_dir / name, value_col, errors)
        if name == "realtime_final_predictions_corrected.csv" and not frame.empty and "final_pred" in frame.columns:
            decisions = pd.to_numeric(frame["final_pred"], errors="coerce")
            if decisions.isna().any():
                errors.append(f"{final_dir / name}: final_pred contains NaN/non-numeric")
    submission = _read_csv(final_dir / "submission_ready.csv", errors)
    if not submission.empty:
        if len(submission) != SLOTS:
            errors.append(f"{final_dir / 'submission_ready.csv'}: rows={len(submission)} expected=96")
        for col in ("dayahead_price", "realtime_price"):
            if col not in submission.columns or pd.to_numeric(submission[col], errors="coerce").isna().any():
                errors.append(f"{final_dir / 'submission_ready.csv'}: invalid {col}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/96")
    parser.add_argument("--phase", choices=("prediction", "replay", "all"), default="prediction")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--resource-mode",
        choices=("legacy", "split_process"),
        default="legacy",
        help="Expected scheduler mode recorded by the strict prediction provenance.",
    )
    parser.add_argument(
        "--require-range-manifest",
        action="store_true",
        help=(
            "Require a prediction_range_manifest.json even for a single-day audit. "
            "Multi-day prediction audits require it automatically."
        ),
    )
    parser.add_argument(
        "--require-target-actual",
        action="store_true",
        help=(
            "Require audited target-day actual ledgers to contain exact p1..p96. "
            "Default prediction audit is live-serving mode and permits partial/absent target truth."
        ),
    )
    parser.add_argument(
        "--require-strict-forecast-vintage",
        action="store_true",
        help=(
            "Also require proof that historical target-day forecasts are the exact D-1 publication vintage. "
            "Current legacy latest-state history is expected to fail this gate."
        ),
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    runs_root = output_root / "runs"
    ledger_root = output_root / "ledger"
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(args.start, args.end, freq="D")]
    errors: list[str] = []
    if args.phase in ("prediction", "all"):
        if _range_manifest_required(
            args.start, args.end, explicit=args.require_range_manifest
        ):
            _audit_prediction_range_manifest(
                output_root,
                args.start,
                args.end,
                args.resource_mode,
                errors,
                require_strict_forecast_vintage=args.require_strict_forecast_vintage,
            )
        for day in dates:
            _audit_prediction_day(runs_root / day, day, args.resource_mode, errors)
        _audit_prediction_ledger(
            ledger_root,
            dates,
            errors,
            require_target_actual=args.require_target_actual,
        )
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
