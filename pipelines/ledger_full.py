"""
Ledger full pipeline.

Orchestrates the complete production chain for a single target day D:

  1. ledger_predict   → run all models, append to prediction ledger
  2. ledger_weight    → learn fusion weights from D-30~D-1 ledger
  3. ledger_fuse      → apply weights to produce fused predictions
  4. final outputs    → aggregate dayahead + realtime final files

For the formal 96-point production profile the classifier is soft-disabled;
RT fused output goes directly to final. The classifier source and internal
pipeline remain available for legacy/replay/experiment callers.

This is the production-grade replacement for the old staged `full` pipeline.
No validation tap, no rolling OOF, no online validation.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fusion.model_pool import models_for_task, tasks_for_target

logger = logging.getLogger(__name__)


def prepare_daily_run_dir(
    runs_root: Path,
    target_date: str,
    force: bool = False,
    *,
    preserve_snapshot: bool = False,
) -> Path:
    """Prepare a daily run directory without destroying canonical snapshots.

    Legacy/24-point force keeps the historical full-clear behavior. Formal96
    passes preserve_snapshot=True so successful LIVE snapshots remain durable
    replay assets even when other run artifacts are force-refreshed.
    """
    run_dir = runs_root / target_date
    if force and run_dir.exists():
        logger.info(
            "Force mode: clearing existing run directory %s (preserve_snapshot=%s)",
            run_dir,
            preserve_snapshot,
        )
        for child in run_dir.iterdir():
            if preserve_snapshot and child.name == "snapshot":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _extract_prediction_provenance(payload: dict) -> dict | None:
    """Return the original strict ledger_predict manifest from a daily manifest."""
    if not isinstance(payload, dict):
        return None
    if payload.get("pipeline") == "ledger_predict" and (payload.get("asof_view") or payload.get("dynamic_snapshot") or payload.get("snapshot_id")):
        return payload
    nested = payload.get("prediction_provenance")
    if isinstance(nested, dict) and (nested.get("asof_view") or nested.get("dynamic_snapshot") or nested.get("snapshot_id")):
        return nested
    stage = payload.get("stages", {}).get("ledger_predict", {})
    if isinstance(stage, dict) and (stage.get("asof_view") or stage.get("dynamic_snapshot") or stage.get("snapshot_id")):
        return stage
    return None


def _strict_history_readiness(
    ledger_root: Path,
    target_date: str,
    tasks: tuple[str, ...] = ("dayahead", "realtime"),
    *,
    required_days: int = 30,
    max_lookback_days: int = 90,
) -> dict:
    """Use the production selector as the formal-96 cold-start gate."""
    from pipelines.ledger_weight import select_complete_training_days
    from utils.resolution import QUARTER

    results = {}
    for task in tasks:
        results[task] = select_complete_training_days(
            task=task,
            target_date=target_date,
            ledger_root=ledger_root,
            expected_models=list(models_for_task(task)),
            required_days=required_days,
            max_lookback_days=max_lookback_days,
            resolution=QUARTER,
            history_lag_days=2,
        )
    ready = all(item.get("status") == "PASS" for item in results.values())
    return {
        "status": "PASS" if ready else "FAIL",
        "ready": ready,
        "required_days": required_days,
        "max_lookback_days": max_lookback_days,
        "resolution": "15min",
        "tasks": results,
    }


def _validate_finish_prediction_provenance(
    source: dict | None,
    *,
    target_date: str,
    run_dir: Path,
    ledger_root: Path,
    requested_tasks: tuple[str, ...] = ("dayahead", "realtime"),
) -> dict:
    """Strictly validate a prediction-only artifact before ``--finish`` reuse."""
    reasons: list[str] = []
    from utils.resolution import QUARTER
    from utils.asof_view_96 import DYNAMIC_PROTOCOL, HISTORICAL_PROXY_PROTOCOL

    n_slots = QUARTER.slots_per_day
    if not isinstance(source, dict):
        return {"status": "FAIL", "reason": "INCOMPLETE_PREDICTION_PROVENANCE", "errors": ["source manifest missing"]}

    serving_protocol = source.get("serving_protocol") or source.get("production_contract")
    allowed_protocols = {DYNAMIC_PROTOCOL, HISTORICAL_PROXY_PROTOCOL}
    checks = {
        "status": source.get("status") in {"complete", "complete_with_warnings"},
        "target_date": source.get("target_date") == target_date,
        "resolution": source.get("resolution") == "15min",
        "output_profile": source.get("output_profile") == "production",
        "resource_mode": source.get("resource_mode") == "split_process",
        "requested_tasks": tuple(source.get("requested_tasks", ())) == tuple(requested_tasks),
        "serving_protocol": serving_protocol in allowed_protocols,
        "snapshot_id": bool(source.get("snapshot_id")),
        "snapshot_status": source.get("dynamic_snapshot", {}).get("protocol") in allowed_protocols,
        "feature_view": source.get("feature_view", {}).get("status") == "PASS",
        "target_truth_mask": source.get("feature_view", {}).get("target_truth_mask") is True,
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"{name} contract mismatch")

    pool = source.get("selected_model_pool", {})
    if list(pool.get("dayahead", ())) != list(models_for_task("dayahead")):
        reasons.append("canonical DA model pool mismatch")
    if list(pool.get("realtime", ())) != list(models_for_task("realtime")):
        reasons.append("canonical RT model pool mismatch")

    production = source.get("production_config", {})
    if production.get("rt916_train_steps") != 24:
        reasons.append("RT916 production stride is not 24")

    snapshot_meta = source.get("dynamic_snapshot", {})
    if serving_protocol == HISTORICAL_PROXY_PROTOCOL:
        proxy_checks = {
            "run_mode": source.get("run_mode") == "HISTORICAL_PROXY_V1",
            "snapshot_kind": source.get("snapshot_kind") == "historical_proxy",
            "proxy_cutoff_period": source.get("proxy_cutoff_period") == 56,
            "historical_vintage": source.get("historical_vintage") == "UNVERIFIED_LEGACY_VINTAGE",
            "strict_historical_vintage_proven": source.get("strict_historical_vintage_proven") is False,
        }
        reasons.extend(f"{key} proxy contract mismatch" for key, ok in proxy_checks.items() if not ok)
    for key in ("values_path", "manifest_path"):
        path = snapshot_meta.get(key)
        if not path or not Path(path).exists():
            reasons.append(f"dynamic snapshot {key} missing")
    if snapshot_meta.get("snapshot_id") != source.get("snapshot_id"):
        reasons.append("dynamic snapshot_id mismatch")

    persisted_manifest_path = snapshot_meta.get("manifest_path")
    if persisted_manifest_path and Path(persisted_manifest_path).exists():
        try:
            persisted_snapshot = json.loads(
                Path(persisted_manifest_path).read_text(encoding="utf-8")
            )
            if persisted_snapshot.get("snapshot_id") != source.get("snapshot_id"):
                reasons.append("persisted dynamic snapshot_id mismatch")
            if persisted_snapshot.get("protocol") != serving_protocol:
                reasons.append("persisted dynamic snapshot protocol mismatch")
            if persisted_snapshot.get("target_day") != target_date:
                reasons.append("persisted dynamic snapshot target_day mismatch")
            persisted_values = persisted_snapshot.get("values_path")
            if not persisted_values or not Path(persisted_values).exists():
                reasons.append("persisted dynamic snapshot values_path missing")
            elif snapshot_meta.get("values_path") and (
                Path(persisted_values).resolve()
                != Path(snapshot_meta.get("values_path")).resolve()
            ):
                reasons.append("persisted dynamic snapshot values_path mismatch")
            if persisted_values and Path(persisted_values).exists() and persisted_snapshot.get("values_sha256"):
                import hashlib
                digest = hashlib.sha256(Path(persisted_values).read_bytes()).hexdigest()
                if digest != persisted_snapshot.get("values_sha256"):
                    reasons.append("persisted snapshot values hash mismatch")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            reasons.append(f"persisted dynamic snapshot manifest unreadable: {exc}")

    # The stage-level results carry the SGDFNet anchor audit.  It is required
    # even when the model output CSV itself is present.
    sgdf = source.get("results", {}).get("realtime", {}).get("sgdfnet", {})
    anchor = sgdf.get("anchor_contract", {}) if isinstance(sgdf, dict) else {}
    if anchor.get("anchor_source_day") != (
        str(pd.Timestamp(target_date) - pd.Timedelta(days=1))[:10]
    ):
        reasons.append("SGDFNet decision-day anchor source mismatch")
    if anchor.get("rows") != n_slots or anchor.get("fallback_used") is not False:
        reasons.append("SGDFNet anchor contract incomplete or fallback_used=true")

    # Canonical per-model files are the provenance of the append.  Require
    # exact 96 slots and finite values before accepting any reuse.
    for task in requested_tasks:
        task_dir = run_dir / task / "prediction"
        value_col = "y_pred"
        for model in models_for_task(task):
            path = task_dir / f"{model}_predictions.csv"
            if not path.exists():
                reasons.append(f"missing {task}/{model} prediction CSV")
                continue
            try:
                frame = pd.read_csv(path)
                slots = frame.get("business_period")
                values = pd.to_numeric(frame.get(value_col), errors="coerce")
                if len(frame) != n_slots or slots is None or set(slots.astype(int)) != set(range(1, n_slots + 1)):
                    reasons.append(f"{task}/{model} does not contain exactly 96 slots")
                if values is None or not values.notna().all():
                    reasons.append(f"{task}/{model} prediction contains NaN")
                if "serving_protocol" not in frame.columns or set(frame["serving_protocol"].dropna().astype(str)) != {str(serving_protocol)}:
                    reasons.append(f"{task}/{model} prediction serving_protocol mismatch")
                if "snapshot_id" not in frame.columns or set(frame["snapshot_id"].dropna().astype(str)) != {str(source.get("snapshot_id"))}:
                    reasons.append(f"{task}/{model} prediction snapshot_id mismatch")
            except Exception as exc:
                reasons.append(f"{task}/{model} unreadable: {exc}")

    # Confirm the target-day append itself has all canonical rows.  This avoids
    # trusting a manifest that was complete before an interrupted append.
    try:
        for task in requested_tasks:
            path = ledger_root / task / "prediction" / "prediction_ledger.parquet"
            frame = pd.read_parquet(path)
            frame = frame[(frame.get("task") == task) & (frame.get("target_day") == target_date)]
            for model in models_for_task(task):
                part = frame[frame.get("model_name") == model]
                if len(part) != n_slots or part.get("business_period") is None or set(part["business_period"].astype(int)) != set(range(1, n_slots + 1)):
                    reasons.append(f"{task}/{model} prediction ledger append incomplete")
                if "serving_protocol" not in part.columns or set(part["serving_protocol"].dropna().astype(str)) != {str(serving_protocol)}:
                    reasons.append(f"{task}/{model} ledger serving_protocol mismatch")
                if "snapshot_id" not in part.columns or set(part["snapshot_id"].dropna().astype(str)) != {str(source.get("snapshot_id"))}:
                    reasons.append(f"{task}/{model} ledger snapshot_id mismatch")
    except Exception as exc:
        reasons.append(f"prediction ledger append unreadable: {exc}")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reason": None if not reasons else "INCOMPLETE_PREDICTION_PROVENANCE",
        "errors": reasons,
    }


def _isolate_stale_delivery_artifacts(run_dir: Path, attempt_id: str) -> None:
    """Keep only the immediately previous delivery for same-day reruns.

    Repeated backtests for one business day must not create an unbounded
    ``stale_delivery_<attempt>`` fan-out. A fixed ``stale_delivery_previous``
    slot is overwritten before each new attempt, preserving one rollback copy.
    """
    candidates = [run_dir / "final"]
    candidates.extend(run_dir / name for name in (
        "delivery_report.json", "delivery_report.md", "fallback_report.json",
        "fallback_report.md",
    ))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    archive = run_dir / "runtime" / "diagnostics" / "stale_delivery_previous"
    if archive.exists():
        shutil.rmtree(archive, ignore_errors=True)
    archive.mkdir(parents=True, exist_ok=True)
    meta = {
        "replaced_by_attempt_id": attempt_id,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    }
    (archive / "archive_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for path in existing:
        destination = archive / path.name
        try:
            shutil.move(str(path), str(destination))
        except OSError:
            logger.warning("Could not isolate stale delivery artifact %s", path)


def _build_decision_snapshot(run_dir: Path, tasks: tuple[str, ...]) -> dict:
    """Embed the small final fusion decision into the root manifest.

    Large training/debug tables may later be retained for only a bounded
    period, so the manifest must preserve the exact learned weights and model
    quality gate that produced the final delivery.
    """
    snapshot: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": {},
    }
    for task in tasks:
        task_snapshot: dict[str, Any] = {}
        weights_path = run_dir / task / "weight" / "weights.csv"
        gate_path = run_dir / task / "fuse" / "model_quality_gate.csv"
        if weights_path.exists():
            weights = pd.read_csv(weights_path)
            task_snapshot["weights"] = json.loads(weights.to_json(orient="records"))
        else:
            task_snapshot["weights"] = []
            task_snapshot["weights_missing"] = True
        if gate_path.exists():
            gate = pd.read_csv(gate_path)
            task_snapshot["model_quality_gate"] = json.loads(
                gate.to_json(orient="records")
            )
        else:
            task_snapshot["model_quality_gate"] = []
            task_snapshot["quality_gate_missing"] = True
        snapshot["tasks"][task] = task_snapshot
    return snapshot


def _cleanup_success_run_runtime(run_dir: Path) -> None:
    """Remove resumable stage scratch after a NORMAL delivery only."""
    stage_dir = run_dir / "runtime" / "stage_manifests"
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    runtime_dir = run_dir / "runtime"
    try:
        if runtime_dir.exists() and not any(runtime_dir.iterdir()):
            runtime_dir.rmdir()
    except OSError:
        pass


def run_ledger_full(args: Any) -> dict:
    """Root-owner wrapper that records interrupt/failure state."""
    try:
        return _run_ledger_full_impl(args)
    except KeyboardInterrupt:
        _record_interrupted_attempt(args, "KeyboardInterrupt")
        raise
    except Exception as exc:
        _record_interrupted_attempt(args, f"{type(exc).__name__}: {exc}")
        raise


def _run_ledger_full_impl(args: Any) -> dict:
    """
    Main entry for --pipeline ledger_full.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: date, data_path, epf_v1_root (optional),
        ledger_root, runs_root, max_cpu_workers, max_gpu_workers,
        allow_missing_models, force, strict_classifier.

    Returns
    -------
    dict with full pipeline manifest.
    """
    from utils.resolution import resolve_resolution

    target_date = args.date
    if not target_date:
        raise ValueError("--date is required for ledger_full")

    res = resolve_resolution(getattr(args, "resolution", "hourly"))
    requested_tasks = tasks_for_target(getattr(args, "target", "both"))
    logger.info(
        f"=== ledger_full: {target_date} (res={res.label}, tasks={','.join(requested_tasks)}) ==="
    )

    # Production defaults are domain-scoped; main.py normally resolves these
    # through utils.output_layout. Explicit roots / legacy profile still win.
    domain = "96" if res.label == "15min" else "24"
    default_ledger = "outputs/96/ledger" if res.label == "15min" else "outputs/ledger"
    default_runs = "outputs/96/runs" if res.label == "15min" else "outputs/runs"
    ledger_root = Path(getattr(args, "ledger_root", None) or default_ledger)
    runs_root = Path(getattr(args, "runs_root", None) or default_runs)
    force = getattr(args, "force", False)

    replay_only = bool(getattr(args, "replay_only", False))
    output_profile = str(getattr(args, "output_profile", "production"))
    resource_mode = str(getattr(args, "resource_mode", "legacy"))
    # Preserve the most recent valid prediction provenance before a new full
    # attempt takes ownership of the root manifest.  This is needed even for a
    # non-replay full run: an early strict-history fail must not erase a prior
    # successful --predict artifact and make a later --finish unrecoverable.
    prior_prediction_provenance = None
    prior_manifest_path = runs_root / target_date / "run_manifest.json"
    if prior_manifest_path.exists():
        try:
            prior_payload = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
            prior_prediction_provenance = _extract_prediction_provenance(prior_payload)
            if prior_prediction_provenance is None:
                stage_path = (
                    runs_root / target_date / "runtime" / "stage_manifests"
                    / "ledger_predict.json"
                )
                if stage_path.exists():
                    stage_payload = json.loads(stage_path.read_text(encoding="utf-8"))
                    prior_prediction_provenance = _extract_prediction_provenance(stage_payload)
        except (OSError, json.JSONDecodeError):
            prior_prediction_provenance = None

    formal_96 = res.label == "15min" and output_profile == "production"
    # Prepare (or optionally clear) the run directory before starting. A
    # replay must preserve prediction-stage artifacts. Formal96 force-runs also
    # preserve the immutable snapshot subtree so an earlier successful LIVE
    # snapshot cannot be destroyed by a later retry.
    prepare_daily_run_dir(
        runs_root,
        target_date,
        force=(force and not replay_only),
        preserve_snapshot=formal_96,
    )
    run_dir = runs_root / target_date
    attempt_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    if formal_96:
        setattr(args, "_attempt_id", attempt_id)
        _isolate_stale_delivery_artifacts(run_dir, attempt_id)

    manifest = {
        "pipeline": "ledger_full",
        "target_date": target_date,
        "resolution": res.label,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "attempt_id": attempt_id,
        "stages": {"ledger_predict": {"status": "running"}},
        "warnings": [],
        "errors": [],
        "mode": "replay_only" if replay_only else "full",
        "requested_tasks": list(requested_tasks),
    }
    classifier_disabled = formal_96
    manifest["classifier_policy"] = (
        "disabled_by_production_policy" if classifier_disabled else "enabled_legacy_or_experiment"
    )
    manifest["production_config"] = {
        "formal_96": bool(classifier_disabled),
        "scheduler_mode": resource_mode,
        "cpu_workers": 2 if classifier_disabled and resource_mode == "split_process" else int(getattr(args, "max_cpu_workers", 2)),
        "gpu_workers": 1 if classifier_disabled and resource_mode == "split_process" else int(getattr(args, "max_gpu_workers", 1)),
        "cpu_dag_aware": bool(classifier_disabled and resource_mode == "split_process"),
        "gpu_serial": bool(classifier_disabled and resource_mode == "split_process"),
        "rt916_train_steps": 24 if classifier_disabled else None,
        "rt916_train_steps_role": "production_stride" if classifier_disabled else "legacy/internal",
        "weight_learner": str(getattr(args, "weight_learner", "smape_reg")),
        "weight_optimizer": "SLSQP" if str(getattr(args, "weight_learner", "smape_reg")) == "smape_reg" else None,
        "weight_window_days": int(getattr(args, "validation_days", 30)),
        "weight_max_lookback_days": int(getattr(args, "weight_max_lookback_days", 90)),
        "weight_granularity": str(getattr(args, "weight_granularity", "period")),
        "weight_reg": 0.2 if str(getattr(args, "weight_learner", "smape_reg")) == "smape_reg" else None,
        "weight_bounds": [0.0, 1.0] if str(getattr(args, "weight_learner", "smape_reg")) == "smape_reg" else None,
        "weight_prune_threshold": float(getattr(args, "weight_prune_threshold", 0.05)),
    }
    stage_total = 4 if classifier_disabled else 5
    if prior_prediction_provenance is not None:
        manifest["prediction_provenance"] = prior_prediction_provenance
        # Keep the previous successful Stage1 binding available to the
        # three-way resolver while this new full attempt owns a fresh root
        # manifest with status=running.  It is never treated as the current
        # attempt's output unless the exact snapshot provenance validates.
        manifest["previous_prediction_provenance"] = prior_prediction_provenance

    manifest["output_roots"] = {
        "ledger_root": str(ledger_root),
        "runs_root": str(runs_root),
    }
    if formal_96 or not replay_only:
        stage_manifest_dir = run_dir / "runtime" / "stage_manifests"
        stage_manifest_dir.mkdir(parents=True, exist_ok=True)
        setattr(args, "_root_manifest_owned", True)
        setattr(args, "_weight_manifest_path", stage_manifest_dir / "ledger_weight.json")
        setattr(args, "_fuse_manifest_path", stage_manifest_dir / "ledger_fuse.json")
        setattr(args, "_classifier_manifest_path", stage_manifest_dir / "ledger_classifier.json")
    # Root manifest ownership belongs to ledger_full.  Persist the running
    # attempt before any child model process can start.
    _write_manifest(runs_root, target_date, manifest)

    provenance_required = replay_only and formal_96
    skip_remaining = False
    if provenance_required:
        if prior_prediction_provenance is None:
            manifest["errors"].append(
                "INCOMPLETE_PREDICTION_PROVENANCE: replay-only production 96 requires strict prediction provenance"
            )
            manifest["stages"]["ledger_predict"] = {
                "status": "failed", "reason": "INCOMPLETE_PREDICTION_PROVENANCE",
            }
            manifest["status"] = "failed"
            skip_remaining = True
        else:
            provenance = _validate_finish_prediction_provenance(
                prior_prediction_provenance,
                target_date=target_date,
                run_dir=run_dir,
                ledger_root=ledger_root,
                requested_tasks=tuple(requested_tasks),
            )
            manifest["finish_provenance"] = provenance
            if provenance["status"] != "PASS":
                manifest["errors"].extend(provenance.get("errors", []))
                manifest["errors"].append("INCOMPLETE_PREDICTION_PROVENANCE")
                manifest["stages"]["ledger_predict"] = {
                    "status": "failed", "reason": "INCOMPLETE_PREDICTION_PROVENANCE",
                    "errors": provenance.get("errors", []),
                }
                manifest["status"] = "failed"
                skip_remaining = True
            else:
                manifest["stages"]["ledger_predict"] = {
                    "pipeline": "ledger_predict", "status": "complete",
                    "mode": "replay_only", "reused": True,
                    "provenance_validated": True,
                }

    # Keep the persistent actual ledger moving in live production, but only
    # after replay provenance has passed.  Target T is served on T-1 before
    # that business day is complete, so T-1 full-day truth is not a causal
    # learner input.  Settle only the latest fully closed day (T-2).
    if formal_96 and not skip_remaining:
        from pipelines.ledger_predict import settle_closed_actuals

        actual_source = getattr(args, "actual_data_path", None) or getattr(args, "data_path", None)
        settlement = settle_closed_actuals(
            str(actual_source or ""),
            target_date,
            ledger_root,
            resolution=res,
            tasks=tuple(requested_tasks),
            output_profile=output_profile,
            settlement_lag_days=2,
        )
        manifest["closed_actual_settlement"] = settlement
        if settlement.get("status") != "complete":
            manifest["warnings"].append(
                "closed actual settlement incomplete; learner readiness will skip unavailable days"
            )
        _write_manifest(runs_root, target_date, manifest)

    # Formal combined production runs are fail-closed before any model starts.
    # Prediction-only façade calls never enter ledger_full and therefore remain
    # available to build the strict history needed for bootstrap.
    if (
        not replay_only and formal_96 and set(requested_tasks) == {"dayahead", "realtime"}
    ):
        readiness = _strict_history_readiness(ledger_root, target_date)
        manifest["strict_history_readiness"] = readiness
        if not readiness["ready"]:
            reason = "INSUFFICIENT_STRICT_HISTORY"
            manifest["cold_start_status"] = reason
            manifest["cold_start_policy"] = "fail_closed_no_emergency_fallback"
            manifest["errors"].append(reason)
            manifest["stages"]["ledger_predict"] = {
                "status": "failed", "reason": reason,
                "readiness": readiness,
            }
            manifest["status"] = "failed"
            skip_remaining = True

    # -----------------------------------------------------------------------
    # Stage 1: ledger_predict (or reuse the frozen prediction ledger)
    # -----------------------------------------------------------------------
    if replay_only and not skip_remaining:
        logger.info(
            f"\n{'='*60}\nStage 1/{stage_total}: ledger_predict (replay-only, reused)\n{'='*60}"
        )
        manifest["stages"]["ledger_predict"] = {
            "pipeline": "ledger_predict",
            "status": "complete",
            "mode": "replay_only",
            "reused": True,
            "note": "Prediction and actual ledgers were generated by a prior prediction-only range run.",
        }
    elif not replay_only and not skip_remaining:
        logger.info(f"\n{'='*60}\nStage 1/{stage_total}: ledger_predict\n{'='*60}")
        try:
            from pipelines.ledger_predict import run_ledger_predict
            stage_manifest_path = run_dir / "runtime" / "stage_manifests" / "ledger_predict.json"
            stage_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            setattr(args, "_stage_manifest_path", stage_manifest_path)
            predict_result = run_ledger_predict(args)
            manifest["stages"]["ledger_predict"] = predict_result
            if predict_result.get("status") in {"complete", "complete_with_warnings"}:
                # The current attempt supersedes any provenance inherited from a
                # previous prediction-only run.
                manifest["prediction_provenance"] = predict_result

            # Compact any fragmented compatibility parts before weight learning.
            # Production profile normally appends the canonical ledger atomically,
            # so this is a safe no-op when no parts exist.
            if predict_result.get("status") == "complete" and res.label == "15min":
                from pipelines.prediction_ledger import compact_ledger

                compaction = {}
                for task in requested_tasks:
                    compaction[f"{task}_prediction"] = compact_ledger(
                        ledger_root, task, "prediction"
                    )
                    compaction[f"{task}_actual"] = compact_ledger(
                        ledger_root, task, "actual"
                    )
                predict_result["ledger_compaction"] = compaction
                manifest["stages"]["ledger_predict"] = predict_result

            _write_manifest(runs_root, target_date, manifest)

            if predict_result.get("status") in {"failed", "error"}:
                manifest["status"] = "failed"
                manifest["errors"].append("ledger_predict failed")
                skip_remaining = True
        except Exception as e:
            manifest["stages"]["ledger_predict"] = {"status": "error", "error": str(e)}
            manifest["errors"].append(f"ledger_predict: {e}")
            manifest["status"] = "failed"
            skip_remaining = True
            _write_manifest(runs_root, target_date, manifest)

    # -----------------------------------------------------------------------
    # Stage 2: ledger_weight (skip if previous stage failed)
    # -----------------------------------------------------------------------
    if not skip_remaining:
        logger.info(f"\n{'='*60}\nStage 2/{stage_total}: ledger_weight\n{'='*60}")
        try:
            from pipelines.ledger_weight import run_ledger_weight
            weight_result = run_ledger_weight(args)
            manifest["stages"]["ledger_weight"] = weight_result
            _write_manifest(runs_root, target_date, manifest)

            if weight_result.get("status") == "failed":
                manifest["status"] = "failed"
                # Preserve a machine-actionable cold-start reason at the
                # orchestration level.  Generic stage failure remains the
                # fallback for non-readiness failures and legacy behavior.
                weight_reason = weight_result.get("cold_start_status")
                manifest["errors"].append(weight_reason or "ledger_weight failed")
                skip_remaining = True
        except Exception as e:
            manifest["stages"]["ledger_weight"] = {"status": "error", "error": str(e)}
            manifest["errors"].append(f"ledger_weight: {e}")
            manifest["status"] = "failed"
            skip_remaining = True

    # -----------------------------------------------------------------------
    # Stage 3: ledger_fuse (skip if previous stage failed)
    # -----------------------------------------------------------------------
    if not skip_remaining:
        logger.info(f"\n{'='*60}\nStage 3/{stage_total}: ledger_fuse\n{'='*60}")
        try:
            from pipelines.ledger_fuse import run_ledger_fuse
            fuse_result = run_ledger_fuse(args)
            manifest["stages"]["ledger_fuse"] = fuse_result
            if fuse_result.get("status") == "complete":
                manifest["decision_snapshot"] = _build_decision_snapshot(
                    run_dir, tuple(requested_tasks)
                )
            _write_manifest(runs_root, target_date, manifest)

            if fuse_result.get("status") == "failed":
                manifest["status"] = "failed"
                manifest["errors"].append("ledger_fuse failed")
                skip_remaining = True
        except Exception as e:
            manifest["stages"]["ledger_fuse"] = {"status": "error", "error": str(e)}
            manifest["errors"].append(f"ledger_fuse: {e}")
            manifest["status"] = "failed"
            skip_remaining = True

    # -----------------------------------------------------------------------
    # Stage 4: classifier shadow (skip for formal 96 production)
    # -----------------------------------------------------------------------
    if not skip_remaining and "realtime" in requested_tasks and not classifier_disabled:
        logger.info(f"\n{'='*60}\nStage 4/{stage_total}: ledger_classifier (legacy/shadow)\n{'='*60}")
        # Frozen 96-point replay is an evaluation artifact, not a live
        # degraded-delivery path: a classifier failure must be visible and
        # must fail the replay instead of silently returning uncorrected RT.
        strict_clf = bool(getattr(args, "strict_classifier", False)) or (
            res.label == "15min" and replay_only
        )
        try:
            from copy import copy
            from pipelines.ledger_classifier import run_ledger_classifier
            # Propagate the effective strictness. In particular, 96-point
            # replay must not silently downgrade to uncorrected RT output.
            classifier_args = copy(args)
            classifier_args.strict_classifier = strict_clf
            clf_result = run_ledger_classifier(classifier_args)
            manifest["stages"]["ledger_classifier"] = clf_result
            _write_manifest(runs_root, target_date, manifest)

            clf_status = clf_result.get("status")
            if clf_status == "failed":
                manifest["errors"].append("ledger_classifier failed")
                if strict_clf:
                    manifest["status"] = "failed"
                    skip_remaining = True
            else:
                # Propagate classifier warnings/errors to top-level manifest
                for w in clf_result.get("warnings", []):
                    manifest["warnings"].append(f"ledger_classifier: {w}")
                for e in clf_result.get("errors", []):
                    if strict_clf:
                        manifest["errors"].append(f"ledger_classifier: {e}")
                    else:
                        manifest["warnings"].append(f"ledger_classifier: {e}")
        except Exception as e:
            manifest["stages"]["ledger_classifier"] = {"status": "error", "error": str(e)}
            if strict_clf:
                manifest["errors"].append(f"ledger_classifier: {e}")
                manifest["status"] = "failed"
                skip_remaining = True
            else:
                manifest["warnings"].append(f"ledger_classifier: {e}")
    elif not skip_remaining and classifier_disabled:
        manifest["stages"]["ledger_classifier"] = {
            "status": "disabled_by_production_policy",
            "reason": "formal 96-point production uses uncorrected fused realtime output",
            "source_preserved": True,
        }
        _write_manifest(runs_root, target_date, manifest)
    elif not skip_remaining:
        manifest["stages"]["ledger_classifier"] = {
            "status": "skipped",
            "reason": "dayahead-only task scope",
        }

    # -----------------------------------------------------------------------
    # Stage 5: Final outputs (skip if previous stage failed)
    # -----------------------------------------------------------------------
    if not skip_remaining:
        logger.info(f"\n{'='*60}\nStage {stage_total}/{stage_total}: Final outputs\n{'='*60}")
        try:
            final_result = _collect_final_outputs(
                runs_root, target_date, res, tasks=requested_tasks
            )
            manifest["stages"]["final_outputs"] = final_result
            _write_manifest(runs_root, target_date, manifest)
            if final_result.get("status") != "complete":
                manifest["errors"].append("final_outputs failed integrity checks")
                skip_remaining = True
        except Exception as e:
            manifest["stages"]["final_outputs"] = {"status": "error", "error": str(e)}
            manifest["warnings"].append(f"final_outputs: {e}")
            _write_manifest(runs_root, target_date, manifest)

    # -----------------------------------------------------------------------
    # Unified finalization — write manifest, postflight, fallback, report
    # -----------------------------------------------------------------------
    return _finalize_delivery(args, manifest)


def _cleanup_transient_input(args: Any, manifest: dict) -> None:
    """Reclaim formal attempt scratch only after a successful delivery.

    Controlled failures keep their single attempt sandbox for diagnosis; the
    maintenance TTL owns later recovery. Legacy callers without an attempt
    root retain the historical one-file cleanup behavior.
    """
    attempt_root = getattr(args, "_runtime_attempt_root", None)
    path = getattr(args, "_transient_asof_path", None)
    if attempt_root:
        root = Path(attempt_root)
        success = manifest.get("delivery_status") == "NORMAL"
        if success and root.exists():
            shutil.rmtree(root, ignore_errors=True)
        manifest["runtime_input_cleanup"] = {
            "path": str(path) if path else None,
            "attempt_root": str(root),
            "persistent": False,
            "removed": not root.exists() if success else False,
            "reason": None if success else "retained_after_failed_delivery_for_diagnosis",
        }
        return
    if not path:
        return
    from utils.asof_view_96 import cleanup_transient_asof_96

    cleanup_transient_asof_96(path)
    manifest["runtime_input_cleanup"] = {
        "path": str(path),
        "persistent": False,
        "exists_after_cleanup": Path(path).exists(),
    }


def _finalize_delivery(args: Any, manifest: dict) -> dict:
    """Unified delivery finalization — always called, even on early failure.

    Steps
    -----
    1. Derive ``status`` from errors/warnings.
    2. **Write manifest first** so ``validate_daily_submission`` can read it.
    3. ``validate_daily_submission`` — if no submission_ready.csv exists yet,
       the check will fail, triggering emergency fallback.
    4. If postflight fails → ``try_emergency_fallback``.
    5. Set ``delivery_status``: NORMAL / DEGRADED_DELIVERED / FAILED_NO_DELIVERY.
    6. ``validate_next_day_readiness``.
    7. Write final manifest + delivery report.
    8. Print terminal DAILY DELIVERY REPORT.
    """
    from utils.resolution import resolve_resolution

    res = resolve_resolution(getattr(args, "resolution", "hourly"))
    requested_tasks = tasks_for_target(getattr(args, "target", "both"))
    target_date = manifest["target_date"]
    domain = "96" if res.label == "15min" else "24"
    default_runs = "outputs/96/runs" if res.label == "15min" else "outputs/runs"
    default_ledger = "outputs/96/ledger" if res.label == "15min" else "outputs/ledger"
    runs_root = Path(getattr(args, "runs_root", None) or default_runs)
    ledger_root = Path(getattr(args, "ledger_root", None) or default_ledger)
    from utils.data_layout import data_path as resolve_data_path
    data_path = getattr(args, "data_path", None) or str(resolve_data_path(res.label))

    # 1. Derive status
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()

    errors = manifest.get("errors", [])
    warnings = manifest.get("warnings", [])
    if errors:
        manifest["status"] = "failed"
    elif warnings:
        manifest["status"] = "complete_with_warnings"
    else:
        manifest["status"] = "complete"

    # 2. Write manifest first (postflight/fallback will read it)
    _write_manifest(runs_root, target_date, manifest)

    # Formal 96 contract failures are fail-closed.  In particular, a missing
    # strict weight history or an invalid prediction provenance must never be
    # converted into a plausible-looking historical-median submission.
    if _is_formal96_contract_failure(manifest, res):
        manifest["delivery_status"] = "FAILED_NO_DELIVERY"
        manifest["fallback"] = {
            "fallback_used": False,
            "policy": "disabled_by_production_policy",
        }
        manifest["postflight"] = {
            "status": "FAIL",
            "errors": list(manifest.get("errors", [])),
            "reason": "formal96_contract_failure",
        }
        manifest["next_day_readiness"] = {"status": "SKIPPED", "reason": "no_delivery"}
        _cleanup_transient_input(args, manifest)
        _write_manifest(runs_root, target_date, manifest)
        from pipelines.delivery_report import write_daily_delivery_report, print_daily_delivery_report
        write_daily_delivery_report(runs_root / target_date, manifest)
        print_daily_delivery_report(manifest)
        return manifest

    # Task-scoped commands are complete products on their own: DA stops after
    # fusion, RT stops after the policy-selected final source (uncorrected
    # fused output for formal 96, classifier correction for legacy). They do not
    # build the two-price submission_ready.csv contract or invoke a combined
    # emergency fallback that would fabricate the unrequested task.
    if len(requested_tasks) == 1:
        from pipelines.delivery_report import (
            write_daily_delivery_report,
            print_daily_delivery_report,
        )

        final_stage = manifest.get("stages", {}).get("final_outputs", {})
        task_ok = (
            manifest.get("status") != "failed"
            and final_stage.get("status") == "complete"
        )
        manifest["postflight"] = {
            "status": "PASS" if task_ok else "FAIL",
            "scope": requested_tasks[0],
            "note": "task-scoped final output integrity check",
            "errors": [] if task_ok else list(manifest.get("errors", [])),
        }
        manifest["delivery_status"] = "NORMAL" if task_ok else "FAILED_NO_DELIVERY"
        manifest["fallback"] = {"fallback_used": False, "scope": requested_tasks[0]}
        manifest["next_day_readiness"] = {
            "status": "SKIPPED",
            "scope": requested_tasks[0],
            "reason": "combined two-task readiness is not applicable to task-scoped runs",
        }
        _cleanup_transient_input(args, manifest)
        _write_manifest(runs_root, target_date, manifest)
        write_daily_delivery_report(runs_root / target_date, manifest)
        if manifest.get("delivery_status") == "NORMAL":
            _cleanup_success_run_runtime(runs_root / target_date)
        print_daily_delivery_report(manifest)
        logger.info(
            f"ledger_full {target_date} task={requested_tasks[0]}: "
            f"status={manifest['status']}, delivery={manifest['delivery_status']}"
        )
        return manifest

    # Imports (lazy to avoid circulars at module level)
    from pipelines.delivery_quality import (
        validate_daily_submission,
        validate_next_day_readiness,
    )
    from pipelines.emergency_fallback import try_emergency_fallback
    from pipelines.delivery_report import (
        write_daily_delivery_report,
        print_daily_delivery_report,
    )

    # 3. Postflight
    postflight_result = validate_daily_submission(runs_root, target_date, resolution=res)
    manifest["postflight"] = postflight_result

    if postflight_result["status"] == "PASS":
        manifest["delivery_status"] = "NORMAL"
        manifest["fallback"] = {"fallback_used": False}
    else:
        formal96_production = (
            getattr(res, "label", None) == "15min"
            and (
                manifest.get("production_config", {}).get("formal_96") is True
                or manifest.get("classifier_policy") == "disabled_by_production_policy"
            )
        )
        if formal96_production:
            # Formal96 is fail-closed: never replace a failed model delivery
            # with a plausible-looking historical-median submission.
            manifest["delivery_status"] = "FAILED_NO_DELIVERY"
            manifest["fallback"] = {
                "fallback_used": False,
                "policy": "disabled_by_production_policy",
                "reason": (
                    f"formal96 postflight failed: "
                    f"{len(postflight_result['errors'])} error(s)"
                ),
            }
            logger.error(
                "Formal96 postflight FAILED for %s: %d error(s); "
                "emergency fallback disabled by production policy",
                target_date,
                len(postflight_result["errors"]),
            )
        else:
            logger.warning(
                f"Postflight FAILED for {target_date}: "
                f"{len(postflight_result['errors'])} error(s). "
                "Attempting emergency fallback..."
            )

            # 4. Emergency fallback
            fb_reason = (
                f"normal pipeline {'failed' if manifest.get('status') == 'failed' else 'postflight failed'}: "
                f"{len(postflight_result['errors'])} error(s)"
            )
            fallback_result = try_emergency_fallback(
                target_date, data_path, runs_root, reason=fb_reason,
                resolution=res,
            )

            if fallback_result["success"]:
                # 1. Set delivery_status & fallback BEFORE writing/validating
                manifest["delivery_status"] = "DEGRADED_DELIVERED"
                manifest["fallback"] = {
                    "fallback_used": True,
                    "fallback_method": fallback_result["fallback_method"],
                    "fallback_level": fallback_result["fallback_level"],
                    "reason": fb_reason,
                    "report": fallback_result,
                }
                # 2. Write manifest so validate_daily_submission reads updates
                _write_manifest(runs_root, target_date, manifest)
                # 3. Re-validate (manifest now has DEGRADED_DELIVERED + fallback)
                second_postflight = validate_daily_submission(
                    runs_root, target_date, allow_degraded=True,
                    resolution=res,
                )
                manifest["postflight"] = second_postflight
                # 4. If second postflight fails, downgrade & re-write
                if second_postflight["status"] != "PASS":
                    manifest["delivery_status"] = "FAILED_NO_DELIVERY"
                    _write_manifest(runs_root, target_date, manifest)
            else:
                manifest["delivery_status"] = "FAILED_NO_DELIVERY"
                manifest["fallback"] = {
                    "fallback_used": True,
                    "fallback_method": "historical_same_hour_median",
                    "fallback_level": "failed",
                    "reason": fb_reason,
                    "report": fallback_result,
                    "errors": fallback_result.get("errors", []),
                }

    # 5. Next-day readiness. Formal96 must use the same adaptive selector as
    # tomorrow's learner (30 complete days within the configured lookback),
    # not the legacy "30 contiguous calendar days" validator.
    formal96_production = (
        getattr(res, "label", None) == "15min"
        and (
            manifest.get("production_config", {}).get("formal_96") is True
            or manifest.get("classifier_policy") == "disabled_by_production_policy"
        )
    )
    if formal96_production:
        next_target = (
            pd.Timestamp(target_date) + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        required_days = int(
            manifest.get("production_config", {}).get("weight_window_days", 30)
        )
        max_lookback_days = int(
            manifest.get("production_config", {}).get(
                "weight_max_lookback_days", 90
            )
        )
        strict = _strict_history_readiness(
            ledger_root,
            next_target,
            tasks=tuple(requested_tasks),
            required_days=required_days,
            max_lookback_days=max_lookback_days,
        )
        ndr_errors = []
        for task, task_result in strict.get("tasks", {}).items():
            for err in task_result.get("errors", []):
                ndr_errors.append({"task": task, "error": str(err)})
        manifest["next_day_readiness"] = {
            "status": strict["status"],
            "target_date": next_target,
            "mode": "adaptive_complete_days",
            "history_lag_days": 2,
            "required_days": required_days,
            "max_lookback_days": max_lookback_days,
            "selected_count": {
                task: result.get("selected_count", 0)
                for task, result in strict.get("tasks", {}).items()
            },
            "selected_days": {
                task: result.get("selected_days", [])
                for task, result in strict.get("tasks", {}).items()
            },
            "tasks": strict.get("tasks", {}),
            "errors": ndr_errors,
        }
    else:
        manifest["next_day_readiness"] = validate_next_day_readiness(
            target_date,
            ledger_root,
            resolution=res,
            history_lag_days=1,
        )

    # 6. Remove transient masked input, then write the final audit record.
    _cleanup_transient_input(args, manifest)
    _write_manifest(runs_root, target_date, manifest)
    write_daily_delivery_report(runs_root / target_date, manifest)
    if manifest.get("delivery_status") == "NORMAL":
        _cleanup_success_run_runtime(runs_root / target_date)

    # 7. Terminal report
    print_daily_delivery_report(manifest)

    logger.info(
        f"ledger_full {target_date}: status={manifest['status']}, "
        f"delivery={manifest.get('delivery_status', 'UNSET')}"
    )

    return manifest


def _is_formal96_contract_failure(manifest: dict, resolution) -> bool:
    if getattr(resolution, "label", None) != "15min":
        return False
    if not (
        manifest.get("production_config", {}).get("formal_96") is True
        or manifest.get("classifier_policy") == "disabled_by_production_policy"
    ):
        return False
    # Formal production never emits a degraded delivery.  The named contract
    # failures below provide the audit reason; any other failed formal stage is
    # still fail-closed rather than silently fabricating a fallback.
    if manifest.get("status") == "failed":
        return True
    needles = (
        "INSUFFICIENT_STRICT_HISTORY", "INCOMPLETE_PREDICTION_PROVENANCE",
        "model pool", "schema mismatch", "as-of", "asof", "leakage",
        "RT916", "SGDFNet", "anchor contract",
    )
    errors = [str(item) for item in manifest.get("errors", [])]
    return bool(manifest.get("cold_start_status") or any(
        any(needle.lower() in item.lower() for needle in needles) for item in errors
    ))


def _collect_final_outputs(
    runs_root: Path,
    target_date: str,
    resolution=None,
    tasks=("dayahead", "realtime"),
) -> dict:
    """Collect selected-task outputs to the top-level final directory."""
    result = {"status": "running", "errors": [], "warnings": [], "output_paths": {}}

    run_dir = runs_root / target_date
    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    requested_tasks = tuple(tasks)

    if "dayahead" in requested_tasks:
        # Dayahead final — write to BOTH locations
        da_final = run_dir / "dayahead" / "fuse" / "fused_predictions.csv"
        if da_final.exists():
            shutil.copy2(da_final, final_dir / "dayahead_final_predictions.csv")
            dayahead_final_dir = run_dir / "dayahead" / "final"
            dayahead_final_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(da_final, dayahead_final_dir / "dayahead_final_predictions.csv")
            da_df = pd.read_csv(da_final)
            result["dayahead_final_rows"] = len(da_df)
            result["output_paths"]["dayahead"] = str(final_dir / "dayahead_final_predictions.csv")
            _validate_final(da_df, "dayahead", target_date, result, resolution)
        else:
            result["errors"].append(f"missing dayahead fused output: {da_final}")

    if "realtime" in requested_tasks:
        # Realtime final (uncorrected).  Formal 96 has no classifier stage:
        # fuse is the authoritative final source and must overwrite a stale
        # file left by an earlier attempt.  Never reuse an old RT final when
        # the current fuse artifact is missing; fail closed instead.
        rt_final = run_dir / "realtime" / "final" / "realtime_final_predictions.csv"
        formal96 = resolution is not None and resolution.label == "15min"
        if formal96:
            fused = run_dir / "realtime" / "fuse" / "fused_predictions.csv"
            if fused.exists():
                rt_final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fused, rt_final)
            else:
                result["errors"].append(f"missing realtime fused output: {fused}")
        if rt_final.exists():
            shutil.copy2(rt_final, final_dir / "realtime_final_predictions.csv")
            rt_df = pd.read_csv(rt_final)
            result["realtime_final_rows"] = len(rt_df)
            result["output_paths"]["realtime"] = str(final_dir / "realtime_final_predictions.csv")
            _validate_final(rt_df, "realtime", target_date, result, resolution)
        else:
            result["errors"].append(f"missing realtime final output: {rt_final}")

        # Corrected output is legacy/shadow only; formal 96 production does not
        # execute or consume ExtremePriceClf.
        rt_corrected = run_dir / "realtime" / "final" / "realtime_final_predictions_corrected.csv"
        classifier_shadow_allowed = not (resolution is not None and resolution.label == "15min")
        if rt_corrected.exists() and classifier_shadow_allowed:
            shutil.copy2(rt_corrected, final_dir / "realtime_final_predictions_corrected.csv")
            rt_c_df = pd.read_csv(rt_corrected)
            result["realtime_corrected_rows"] = len(rt_c_df)
            result["output_paths"]["realtime_corrected"] = str(
                final_dir / "realtime_final_predictions_corrected.csv"
            )
            if resolution is not None and len(rt_c_df) != resolution.slots_per_day:
                result["errors"].append(
                    f"realtime corrected rows={len(rt_c_df)} expected={resolution.slots_per_day}"
                )
            corrected_col = (
                "y_fused_corrected" if "y_fused_corrected" in rt_c_df.columns else "y_fused"
            )
            if corrected_col not in rt_c_df.columns or not pd.to_numeric(
                rt_c_df[corrected_col], errors="coerce"
            ).notna().all():
                result["errors"].append("realtime corrected final contains invalid predictions")
        elif classifier_shadow_allowed:
            result["errors"].append(f"missing realtime corrected output: {rt_corrected}")

    # Only the combined scope emits the two-price submission contract.
    if set(requested_tasks) == {"dayahead", "realtime"}:
        _build_submission_ready(final_dir, target_date, result, resolution)
        submission_path = final_dir / "submission_ready.csv"
        result["output_paths"]["submission_ready"] = str(submission_path)
        if not submission_path.exists():
            result["errors"].append(f"missing submission output: {submission_path}")
        elif resolution is not None and getattr(resolution, "label", "hourly") == "15min":
            submission = pd.read_csv(submission_path)
            if len(submission) != resolution.slots_per_day:
                result["errors"].append(
                    f"submission rows={len(submission)} expected={resolution.slots_per_day}"
                )
            for price_col in ("dayahead_price", "realtime_price"):
                if price_col not in submission.columns or submission[price_col].isna().any():
                    result["errors"].append(f"96-point submission has invalid {price_col}")

    result["status"] = "complete" if not result["errors"] else "failed"
    return result


def _validate_final(df: pd.DataFrame, task: str, target_date: str, result: dict, resolution=None):
    """Validate final output: N rows, slots 1..N, no duplicates."""
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    n_expected = res.slots_per_day
    slot_col = res.slot_column
    n = len(df)
    if n != n_expected:
        result.setdefault("errors", []).append(
            f"{task} final: expected {n_expected} rows, got {n}"
        )

    if slot_col in df.columns:
        slots = sorted(df[slot_col].unique())
        if slots != list(range(1, n_expected + 1)):
            result.setdefault("errors", []).append(
                f"{task} final: slots {slots[:5]}{'...' if len(slots) > 5 else ''}, "
                f"expected 1..{n_expected}"
            )

        if df[slot_col].duplicated().any():
            result.setdefault("errors", []).append(
                f"{task} final: duplicate slots detected"
            )

    if "y_fused" not in df.columns:
        result.setdefault("errors", []).append(f"{task} final: y_fused missing")
    elif not pd.to_numeric(df["y_fused"], errors="coerce").notna().all():
        result.setdefault("errors", []).append(f"{task} final: y_fused contains NaN/non-numeric")


def _build_submission_ready(final_dir: Path, target_date: str, result: dict, resolution=None):
    """Build a consolidated submission_ready.csv with dayahead + realtime — fixed columns.

    24 点（hourly）：按 business_day + hour_business merge，6 列契约。
    96 点（15min）：按 business_day + business_period merge（4 行/时不再笛卡尔），
    输出加 period_no(1..96)；hour_business=ceil(period/4) 由 business_period 派生。
    """
    from utils.resolution import HOURLY

    _res = resolution or HOURLY
    da_path = final_dir / "dayahead_final_predictions.csv"
    rt_path = final_dir / "realtime_final_predictions.csv"
    # 24 legacy 主链路：优先用分类器修正后的实时预测（若存在），否则回退未修正版。
    # formal 96 不执行此分支；修正版仅由 legacy ledger_classifier 阶段写入（y_fused_corrected 列）。
    rt_corrected_path = final_dir / "realtime_final_predictions_corrected.csv"
    rt_used_corrected = False
    if rt_corrected_path.exists() and _res.label != "15min":
        _probe = pd.read_csv(rt_corrected_path, nrows=0)
        if "y_fused_corrected" in _probe.columns:
            rt_path = rt_corrected_path
            rt_used_corrected = True

    if not da_path.exists() and not rt_path.exists():
        result.setdefault("warnings", []).append("No data for submission_ready.csv")
        return

    da_df = None
    rt_df = None

    if da_path.exists():
        da_df = pd.read_csv(da_path)
        da_df = da_df.rename(columns={"y_fused": "dayahead_price"})

    if rt_path.exists():
        rt_df = pd.read_csv(rt_path)
        _rt_col = "y_fused_corrected" if rt_used_corrected else "y_fused"
        rt_df = rt_df.rename(columns={_rt_col: "realtime_price"})
    if rt_used_corrected:
        result.setdefault("submission_realtime_source", "classifier_corrected")

    is_96 = _res.label == "15min"
    merge_key = "business_period" if is_96 else "hour_business"
    if is_96:
        for df_ in (da_df, rt_df):
            if df_ is not None and "business_period" not in df_.columns:
                df_["business_period"] = df_["hour_business"].apply(
                    lambda h: int(h) if h is not None else None
                )

    # Build with fixed, clean columns — merge on business_day + merge_key
    if is_96:
        FIXED_COLUMNS = ["business_day", "ds", "business_period", "period", "dayahead_price", "realtime_price"]
    else:
        FIXED_COLUMNS = ["business_day", "ds", "hour_business", "period", "dayahead_price", "realtime_price"]

    if da_df is not None and rt_df is not None:
        da_sub = da_df[["business_day", merge_key, "ds", "period", "dayahead_price"]].copy()
        rt_sub = rt_df[["business_day", merge_key, "realtime_price"]].copy()
        submission = da_sub.merge(rt_sub, on=["business_day", merge_key], how="outer")
        # Drop _x/_y columns if any
        for col in list(submission.columns):
            if col.endswith("_x") or col.endswith("_y"):
                submission = submission.drop(columns=[col])
    elif da_df is not None:
        da_sub = da_df[["business_day", merge_key, "ds", "period", "dayahead_price"]].copy()
        da_sub["realtime_price"] = None
        submission = da_sub
    else:
        rt_sub = rt_df[["business_day", merge_key, "ds", "period", "realtime_price"]].copy()
        rt_sub["dayahead_price"] = None
        submission = rt_sub

    # Enforce fixed column order
    out_cols = [c for c in FIXED_COLUMNS if c in submission.columns]
    submission = submission[out_cols]

    submission.to_csv(final_dir / "submission_ready.csv", index=False)
    result["submission_ready_rows"] = len(submission)
    logger.info(f"submission_ready.csv: {len(submission)} rows")


def _record_interrupted_attempt(args: Any, reason: str) -> None:
    """Leave the current root attempt auditable when an exception is caught."""
    target_date = getattr(args, "date", None)
    if not target_date:
        return
    from utils.resolution import resolve_resolution

    res = resolve_resolution(getattr(args, "resolution", "hourly"))
    default_runs = "outputs/96/runs" if res.label == "15min" else "outputs/runs"
    runs_root = Path(getattr(args, "runs_root", None) or default_runs)
    path = runs_root / target_date / "run_manifest.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.setdefault("pipeline", "ledger_full")
    payload.setdefault("target_date", target_date)
    payload.setdefault("attempt_id", f"interrupted-{uuid.uuid4().hex[:8]}")
    payload["status"] = "interrupted"
    payload["delivery_status"] = "FAILED_NO_DELIVERY"
    payload.setdefault("errors", []).append(reason)
    payload["interrupted_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(runs_root, target_date, payload)


def _write_manifest(runs_root: Path, target_date: str, manifest: dict):
    """Atomically write the root manifest owned by ``ledger_full``."""
    manifest_path = runs_root / target_date / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
    tmp_path.replace(manifest_path)
