"""
Ledger weight pipeline.

For a target day D, reads prediction ledger + actual ledger for
the training window, builds a training table, and learns
per-(task, period) fusion weights using Daily Ledger GEF.

Both dayahead and realtime use adaptive complete-day selection:
scan backwards from D-1, skip incomplete days, collect the most recent 30 complete days.

validate_ledger_window() is retained as an audit-only check (not a hard gate).

Output:
  outputs/runs/{D}/{task}/weight/
    ledger_training_table.csv
    weights.csv
    dynamic_weight_trace.csv
    candidate_metrics.csv
    coverage_report.csv
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.prediction_ledger import (
    load_prediction_ledger,
    load_actual_ledger,
    build_ledger_training_table,
    check_ledger_coverage,
)
from fusion.learners.daily_ledger_gef import DailyLedgerGEF, GEFConfig, NNLSGEF, NNLSConfig
from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS

logger = logging.getLogger(__name__)

_EXPECTED_HOURS = set(range(1, 25))  # business hours 1..24 (hourly 默认)


def _expected_slots(resolution) -> set[int]:
    """resolution 感知的合法槽集合（默认 24 点 {1..24}）。"""
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    return set(range(1, res.slots_per_day + 1))


def _slot_column(resolution) -> str:
    from utils.resolution import HOURLY

    return (resolution or HOURLY).slot_column


# ===========================================================================
# Adaptive complete training day selector
# ===========================================================================


def select_complete_training_days(
    task: str,
    target_date: str,
    ledger_root: Path,
    expected_models: list[str],
    required_days: int = 30,
    max_lookback_days: int = 90,
    resolution=None,
) -> dict:
    """
    Select the most recent *required_days* complete training days for weight
    learning by scanning backwards from D-1.

    A day is **complete** when ALL of the following hold:

    1. Prediction ledger exists and contains all *expected_models* for that day.
    2. Each model has exactly 24 ``hour_business`` values (1..24) with no NaN
       in ``y_pred``.
    3. Actual ledger exists and contains 24 hours for that day with no NaN
       in ``y_true``.
    4. After deduplication the above still holds.

    Parameters
    ----------
    task : str
        ``"dayahead"`` or ``"realtime"``.
    target_date : str
        The prediction day D (YYYY-MM-DD).
    ledger_root : Path
        Root of the ledger directory tree.
    expected_models : list[str]
        Model names that must be present for a day to count as complete.
    required_days : int
        Number of complete days to collect (default 30).
    max_lookback_days : int
        Maximum number of calendar days to scan backwards (default 90).

    Returns
    -------
    dict with keys: status, task, target_date, required_days,
    max_lookback_days, anchor_start, selected_days, selected_count,
    skipped_days, errors.
    """
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    slot_col = res.slot_column
    expected_slots = set(range(1, res.slots_per_day + 1))

    ledger_root = Path(ledger_root)
    D = pd.Timestamp(target_date)

    pred_path = ledger_root / task / "prediction" / "prediction_ledger.parquet"
    act_path = ledger_root / task / "actual" / "actual_ledger.parquet"

    result: dict[str, Any] = {
        "status": "PASS",
        "task": task,
        "target_date": target_date,
        "required_days": required_days,
        "max_lookback_days": max_lookback_days,
        "anchor_start": (D - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "selected_days": [],
        "selected_count": 0,
        "skipped_days": [],
        "errors": [],
    }

    # --- Load ledgers -------------------------------------------------------
    if not pred_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"prediction ledger not found: {pred_path}")
        return result
    if not act_path.exists():
        result["status"] = "FAIL"
        result["errors"].append(f"actual ledger not found: {act_path}")
        return result

    pred_df = pd.read_parquet(pred_path)
    act_df = pd.read_parquet(act_path)

    # Filter to task
    if "task" in pred_df.columns:
        pred_df = pred_df[pred_df["task"] == task]
    if "task" in act_df.columns:
        act_df = act_df[act_df["task"] == task]

    # --- Scan backwards -----------------------------------------------------
    selected: list[str] = []
    skipped: list[dict] = []

    for offset in range(1, max_lookback_days + 1):
        if len(selected) >= required_days:
            break

        day = (D - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")

        # -- prediction check --
        day_pred = pred_df[pred_df.get("target_day", pred_df.get("business_day")) == day] if "target_day" in pred_df.columns else pred_df[pred_df.get("business_day") == day]

        if day_pred.empty:
            skipped.append({"day": day, "reason": "prediction missing", "detail": "0 rows in prediction ledger"})
            logger.info(f"[ledger_weight][{task}] skip {day}: prediction missing")
            continue

        # Check each expected model
        models_present = []
        models_missing = []
        models_nan = []
        all_models_ok = True

        for model in expected_models:
            model_pred = day_pred[day_pred["model_name"] == model] if "model_name" in day_pred.columns else pd.DataFrame()
            if len(model_pred) == 0:
                models_missing.append(model)
                all_models_ok = False
                continue
            # Dedup by slot column
            if slot_col in model_pred.columns:
                model_pred = model_pred.drop_duplicates(subset=[slot_col], keep="last")
            # Strict slot set check: must be exactly 1..N
            if slot_col in model_pred.columns:
                actual_hours = set(model_pred[slot_col].astype(int).tolist())
            else:
                actual_hours = set()
            missing_h = expected_slots - actual_hours
            extra_h = actual_hours - expected_slots
            if missing_h or extra_h:
                parts_h = []
                if missing_h:
                    parts_h.append(f"missing_hours={sorted(missing_h)}")
                if extra_h:
                    parts_h.append(f"extra_hours={sorted(extra_h)}")
                models_missing.append(f"{model} ({'; '.join(parts_h)})")
                all_models_ok = False
                continue
            # Check NaN in y_pred
            if "y_pred" in model_pred.columns and model_pred["y_pred"].isna().any():
                models_nan.append(model)
                all_models_ok = False
                continue
            models_present.append(model)

        if not all_models_ok:
            parts = []
            if models_missing:
                parts.append(f"{','.join(models_missing)} prediction missing")
            if models_nan:
                parts.append(f"{','.join(models_nan)} prediction has NaN")
            detail = "; ".join(parts) if parts else "incomplete prediction"
            # Build a more specific detail
            n_pred_total = len(day_pred)
            skipped.append({"day": day, "reason": "prediction incomplete", "detail": detail})
            logger.info(f"[ledger_weight][{task}] skip {day}: {detail}")
            continue

        # -- actual check --
        if "target_day" in act_df.columns:
            day_act = act_df[act_df["target_day"] == day]
        elif "business_day" in act_df.columns:
            day_act = act_df[act_df["business_day"] == day]
        else:
            day_act = pd.DataFrame()

        if day_act.empty:
            skipped.append({"day": day, "reason": "actual missing", "detail": "0 rows in actual ledger"})
            logger.info(f"[ledger_weight][{task}] skip {day}: actual missing")
            continue

        # Dedup by slot column
        if slot_col in day_act.columns:
            day_act_dedup = day_act.drop_duplicates(subset=[slot_col], keep="last")
        else:
            day_act_dedup = day_act

        # Strict slot set check: must be exactly 1..N
        if slot_col in day_act_dedup.columns:
            actual_act_hours = set(day_act_dedup[slot_col].astype(int).tolist())
        else:
            actual_act_hours = set()
        act_missing_h = expected_slots - actual_act_hours
        act_extra_h = actual_act_hours - expected_slots
        if act_missing_h or act_extra_h:
            parts_h = []
            if act_missing_h:
                parts_h.append(f"missing_hours={sorted(act_missing_h)}")
            if act_extra_h:
                parts_h.append(f"extra_hours={sorted(act_extra_h)}")
            n_act = len(day_act_dedup)
            skipped.append({"day": day, "reason": "actual incomplete", "detail": f"{n_act}/{res.slots_per_day} hours; {'; '.join(parts_h)}"})
            logger.info(f"[ledger_weight][{task}] skip {day}: actual hour set mismatch {'; '.join(parts_h)}")
            continue

        # Check NaN in y_true
        if "y_true" in day_act_dedup.columns and day_act_dedup["y_true"].isna().any():
            n_nan = int(day_act_dedup["y_true"].isna().sum())
            skipped.append({"day": day, "reason": "actual has NaN", "detail": f"{n_nan} NaN values in y_true"})
            logger.info(f"[ledger_weight][{task}] skip {day}: actual has {n_nan} NaN")
            continue

        # Day is complete
        selected.append(day)

    # --- Build result -------------------------------------------------------
    result["selected_days"] = selected
    result["selected_count"] = len(selected)
    result["skipped_days"] = skipped

    if len(selected) < required_days:
        result["status"] = "FAIL"
        result["errors"].append(
            f"cannot collect {required_days} complete training days within "
            f"{max_lookback_days}-day lookback: collected={len(selected)}"
        )
        logger.error(
            f"[ledger_weight][{task}] cannot collect {required_days} complete "
            f"training days within {max_lookback_days}-day lookback"
        )
        logger.error(f"[ledger_weight][{task}] collected={len(selected)}")
        for s in skipped:
            logger.error(f"[ledger_weight][{task}]   {s['day']} {s['reason']} {s['detail']}")
    else:
        latest = selected[0] if selected else "N/A"
        n_skipped = len(skipped)
        logger.info(
            f"[ledger_weight][{task}] selected {len(selected)} complete "
            f"training days for {target_date}"
        )
        logger.info(f"[ledger_weight][{task}] latest complete day: {latest}")
        if n_skipped > 0:
            skip_summary = ", ".join(
                f"{s['day']} {s['reason']} {s['detail']}" for s in skipped[:10]
            )
            logger.info(
                f"[ledger_weight][{task}] skipped {n_skipped} day(s): {skip_summary}"
            )

    return result


# ===========================================================================
# Main entry
# ===========================================================================


def run_ledger_weight(args: Any) -> dict:
    """
    Main entry for --pipeline ledger_weight.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: date, ledger_root, runs_root, window_days (optional).

    Returns
    -------
    dict with weights manifest.
    """
    from utils.resolution import resolve_resolution

    target_date = args.date
    if not target_date:
        raise ValueError("--date is required for ledger_weight")

    res = resolve_resolution(getattr(args, "resolution", "hourly"))
    # 96 点用独立 ledger_96/runs_96；24 点保持 outputs/ledger + outputs/runs
    default_ledger = "outputs/ledger_96" if res.label == "15min" else "outputs/ledger"
    default_runs = "outputs/runs_96" if res.label == "15min" else "outputs/runs"
    ledger_root = Path(getattr(args, "ledger_root", default_ledger))
    runs_root = Path(getattr(args, "runs_root", default_runs))
    window_days = getattr(args, "validation_days", 30)
    recent_week_boost = getattr(args, "recent_week_boost", True)
    recent_week_max_gate = getattr(args, "recent_week_max_gate", 0.85)
    allow_missing = getattr(args, "allow_missing_models", False)
    max_lookback = getattr(args, "weight_max_lookback_days", 90)
    learner = getattr(args, "weight_learner", "nnls") or "nnls"
    # 权重粒度：默认 period（3 段，实证最优——hour/point 因样本稀释降级，混合也未提升）。
    # 可 --weight-granularity {period,hour,point} 实验覆盖。
    granularity = getattr(args, "weight_granularity", "period") or "period"
    da_granularity = granularity
    rt_granularity = granularity

    logger.info(f"=== ledger_weight: {target_date} (window={window_days}d, res={res.label}) ===")

    D = pd.Timestamp(target_date)

    manifest: dict[str, Any] = {
        "pipeline": "ledger_weight",
        "target_date": target_date,
        "window_days": window_days,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "results": {},
        "warnings": [],
        "errors": [],
        "training_day_selection": {},
    }

    try:
        # ------------------------------------------------------------------
        # Audit-only: strict contiguous D-30..D-1 validation (informational)
        # ------------------------------------------------------------------
        from pipelines.delivery_quality import validate_ledger_window

        try:
            ledger_window_check = validate_ledger_window(target_date, ledger_root, days=window_days)
            manifest["ledger_window_check"] = ledger_window_check
        except Exception as exc:
            manifest["warnings"].append(f"strict ledger window audit failed: {exc}")

        # ------------------------------------------------------------------
        # Adaptive complete-day selection for BOTH dayahead and realtime
        # ------------------------------------------------------------------
        da_selection = select_complete_training_days(
            task="dayahead",
            target_date=target_date,
            ledger_root=ledger_root,
            expected_models=DAYAHEAD_MODELS,
            required_days=window_days,
            max_lookback_days=max_lookback,
            resolution=res,
        )

        rt_selection = select_complete_training_days(
            task="realtime",
            target_date=target_date,
            ledger_root=ledger_root,
            expected_models=REALTIME_MODELS,
            required_days=window_days,
            max_lookback_days=max_lookback,
            resolution=res,
        )

        manifest["training_day_selection"]["dayahead"] = da_selection
        manifest["training_day_selection"]["realtime"] = rt_selection

        # Both must PASS
        for task, selection in [("dayahead", da_selection), ("realtime", rt_selection)]:
            if selection["status"] != "PASS":
                msg = selection["errors"][0] if selection.get("errors") else f"{task}: training day selection failed"
                manifest["status"] = "failed"
                manifest["errors"].append(msg)
                manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
                _write_weight_manifest(runs_root, target_date, manifest)
                logger.error(f"[ledger_weight][{task}] {msg}")
                return manifest

        dayahead_days_list = da_selection["selected_days"]
        rt_days_list = rt_selection["selected_days"]

        # Update manifest window info from adaptive selection
        if dayahead_days_list:
            manifest["window_start"] = dayahead_days_list[-1]
            manifest["window_end"] = dayahead_days_list[0]

        # ------------------------------------------------------------------
        # Learn weights
        # ------------------------------------------------------------------
        failed_tasks: list[str] = []

        # Dayahead
        da_result = _learn_weights_for_task(
            task="dayahead",
            target_date=target_date,
            window_days_list=dayahead_days_list,
            ledger_root=ledger_root,
            runs_root=runs_root,
            expected_models=DAYAHEAD_MODELS,
            recent_week_boost=recent_week_boost,
            recent_week_max_gate=recent_week_max_gate,
            resolution=res,
            learner=learner,
            granularity=da_granularity,
        )
        manifest["results"]["dayahead"] = da_result
        if da_result.get("status") != "complete":
            failed_tasks.append(f"dayahead: {da_result.get('error', da_result.get('status'))}")

        # Realtime
        rt_result = _learn_weights_for_task(
            task="realtime",
            target_date=target_date,
            window_days_list=rt_days_list,
            ledger_root=ledger_root,
            runs_root=runs_root,
            expected_models=REALTIME_MODELS,
            recent_week_boost=recent_week_boost,
            recent_week_max_gate=recent_week_max_gate,
            resolution=res,
            learner=learner,
            granularity=rt_granularity,
        )
        manifest["results"]["realtime"] = rt_result
        if rt_result.get("status") != "complete":
            failed_tasks.append(f"realtime: {rt_result.get('error', rt_result.get('status'))}")

        if failed_tasks:
            manifest["status"] = "failed"
            manifest["errors"].extend(failed_tasks)
        else:
            manifest["status"] = "complete_with_warnings" if manifest["warnings"] else "complete"

        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Weight validation (only if not failed)
        if not failed_tasks:
            _validate_weights(manifest)

    except Exception as e:
        manifest["status"] = "failed"
        manifest["errors"].append(str(e))
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.exception(f"ledger_weight failed: {e}")

    _write_weight_manifest(runs_root, target_date, manifest)
    return manifest


def _learn_weights_for_task(
    task: str,
    target_date: str,
    window_days_list: list[str],
    ledger_root: Path,
    runs_root: Path,
    expected_models: list[str],
    recent_week_boost: bool = True,
    recent_week_max_gate: float = 0.85,
    resolution=None,
    learner: str = "nnls",
    granularity: str = "period",
) -> dict:
    """Learn weights for a single task (dayahead or realtime).

    Parameters
    ----------
    window_days_list : list[str]
        Explicit list of training days (newest-first).  May be contiguous
        (dayahead) or non-contiguous (realtime adaptive selection).
    learner : str
        "nnls" (default, 稀疏非负最小二乘, 实证优于 BGEW) 或 "bgew" (旧算法)。
    granularity : str
        "period"(3段) / "hour"(24组, 96点专属) / "point"(96组)。
    """
    from utils.resolution import HOURLY

    res = resolution or HOURLY
    result = {"task": task, "status": "running"}

    # Load ledgers filtered to the selected days
    pred_ledger = load_prediction_ledger(ledger_root, task, window_days_list)
    act_ledger = load_actual_ledger(ledger_root, task, window_days_list)

    logger.info(
        f"[{task}] Loaded: {len(pred_ledger)} prediction rows, "
        f"{len(act_ledger)} actual rows"
    )

    if pred_ledger.empty:
        result["status"] = "failed"
        result["error"] = f"Prediction ledger is empty for {task}"
        return result

    if act_ledger.empty:
        result["status"] = "failed"
        result["error"] = f"Actual ledger is empty for {task}"
        return result

    # Build training table — pass explicit window_days_list
    training = build_ledger_training_table(
        prediction_ledger=pred_ledger,
        actual_ledger=act_ledger,
        target_day=target_date,
        window_days=len(window_days_list),
        recent_week_boost=recent_week_boost,
        recent_week_max_gate=recent_week_max_gate,
        window_days_list=window_days_list,
    )

    # Historical ledgers may still contain disabled candidates. Keep them
    # for audit, but never train current production weights on them.
    training = training[training["model_name"].isin(expected_models)].copy()

    weight_dir = runs_root / target_date / task / "weight"
    weight_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(weight_dir / "ledger_training_table.csv", index=False)

    # Coverage check saved for audit
    coverage = check_ledger_coverage(
        pred_ledger, act_ledger, task, window_days_list, expected_models,
        resolution=res,
    )
    coverage.to_csv(weight_dir / "coverage_report.csv", index=False)

    expected_rows = len(window_days_list) * len(expected_models) * res.slots_per_day
    actual_rows = len(training)
    actual_days = training["target_day"].nunique() if "target_day" in training.columns else 0
    expected_days = len(window_days_list)

    result["training_rows"] = actual_rows
    result["training_days"] = actual_days
    result["coverage"] = {
        "total_expected": expected_rows,
        "total_actual": actual_rows,
        "expected_days": expected_days,
        "actual_days": actual_days,
    }

    if actual_rows != expected_rows or actual_days != expected_days:
        result["status"] = "failed"
        result["error"] = (
            f"{task} ledger training coverage failed: "
            f"expected_rows={expected_rows}, actual_rows={actual_rows}, "
            f"expected_days={expected_days}, actual_days={actual_days}"
        )
        logger.error(result["error"])
        return result

    result["day_gate_min"] = round(float(training["day_gate"].min()), 4)
    result["day_gate_max"] = round(float(training["day_gate"].max()), 4)

    logger.info(
        f"[{task}] Training table: {len(training)} rows, "
        f"day_gate [{result['day_gate_min']}, {result['day_gate_max']}]"
    )

    # Learn weights（传 resolution：96 点用 1_32/33_64/65_96 三段 + 每天 96 行，
    # 否则默认 24 点三段匹配不到 96 点数据 → weights 为空）
    _weights_df = None
    _report = None
    if learner == "bgew":
        gef = DailyLedgerGEF(GEFConfig(window_days=len(window_days_list), resolution=res))
        weights = gef.fit(training)
        _weights_df = gef.get_weights_df()
    elif learner == "smape_reg":
        # SLSQP 软门控（smape + reg*||w-prior||² 目标，直接优化 SMAPE）。
        # 实证（RT 201 单元）：reg=0.2 bound[0,1] → composite 33.54 < NNLS 37.08，
        # 赢等权 77.6%，赢最优单模型 26.9%（NNLS 15.9%）。SMAPE 目标比 NNLS 的 MSE 更匹配评价。
        from fusion.weights import fit_weights_from_long_table
        _weights_df, _report = fit_weights_from_long_table(
            training, reg=0.2, lower_bound=0.0, upper_bound=1.0, resolution=res,
        )
        weights = {}
        for (t, p), grp in _weights_df.groupby(["task", "period"]):
            weights[(t, p)] = dict(zip(grp["model_name"], grp["weight"]))
        result["weight_reg"] = 0.2
        result["weight_bounds"] = [0.0, 1.0]
    else:
        # nnls（默认）：稀疏非负最小二乘。OOF 窗取 min(window_days, 21)，
        # 实证（96 点 2025-12~2026-07）段1/段3 优于等权 ~20%，赢等权 68.6%。
        gef = NNLSGEF(NNLSConfig(window_days=min(len(window_days_list), 21), resolution=res,
                                 granularity=granularity))
        weights = gef.fit(training)
        _weights_df = gef.get_weights_df()

    result["weight_learner"] = learner
    result["weight_granularity"] = granularity

    # Save weights
    weights_df = _weights_df if _weights_df is not None else pd.DataFrame()
    weights_df.to_csv(weight_dir / "weights.csv", index=False)

    # Save trace
    trace_df = pd.DataFrame()
    if learner in ("bgew", "nnls") and gef is not None:
        trace_df = gef.get_trace_df()
    if not trace_df.empty:
        trace_df.to_csv(weight_dir / "dynamic_weight_trace.csv", index=False)

    # Save candidate metrics
    if learner in ("bgew", "nnls") and gef is not None:
        metrics_df = gef.get_candidate_metrics(training)
    else:
        # smape_reg：用 wdf 的逐段 smape（report 已含）
        metrics_df = _report if _report is not None and not _report.empty else pd.DataFrame()
    if not metrics_df.empty:
        metrics_df.to_csv(weight_dir / "candidate_metrics.csv", index=False)

    # Verify weights sum
    for (t, p), wdict in weights.items():
        s = sum(wdict.values())
        if abs(s - 1.0) > 0.01:
            result.setdefault("weight_sum_warnings", []).append(
                f"{t}/{p}: sum={s:.4f}"
            )

    result["status"] = "complete"
    result["n_weights"] = len(weights)
    result["weight_dir"] = str(weight_dir)

    return result


def _write_weight_manifest(runs_root: Path, target_date: str, manifest: dict) -> None:
    """Write ledger_weight manifest, merging into run_manifest.json if present."""
    manifest_path = runs_root / target_date / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing["weight_stage"] = manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, default=str)
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def _format_ledger_window_error(err: Any) -> str:
    """Compactly format a validate_ledger_window error item."""
    if not isinstance(err, dict):
        return str(err)
    parts = []
    for key in ("ledger", "day", "model", "hour_business", "detail", "error"):
        value = err.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else str(err)


def _validate_weights(manifest: dict):
    """Validate weight sums and update order."""
    for task in ["dayahead", "realtime"]:
        task_result = manifest.get("results", {}).get(task, {})
        if not isinstance(task_result, dict):
            continue

        weight_dir = Path(task_result.get("weight_dir", ""))
        if not weight_dir.exists():
            continue

        # Check weights.csv
        weights_path = weight_dir / "weights.csv"
        if weights_path.exists():
            wdf = pd.read_csv(weights_path)
            for (t, p), grp in wdf.groupby(["task", "period"]):
                s = grp["weight"].sum()
                if abs(s - 1.0) > 0.01:
                    manifest.setdefault("warnings", []).append(
                        f"Weight sum {t}/{p}: {s:.4f} != 1.0"
                    )

        # Check update order in trace（BGEW 有 age_days；NNLS trace 用 method/n_obs）
        trace_path = weight_dir / "dynamic_weight_trace.csv"
        if trace_path.exists():
            tdf = pd.read_csv(trace_path)
            if "age_days" in tdf.columns:
                for (t, p), grp in tdf.groupby(["task", "period"]):
                    ages = grp["age_days"].drop_duplicates().sort_values().values
                    expected_order = sorted(ages)
                    if not (ages == expected_order).all():
                        manifest.setdefault("warnings", []).append(
                            f"Trace order for {t}/{p}: {list(ages)}, expected {list(expected_order)}"
                        )
            elif "method" in tdf.columns:
                # NNLS trace：检查每 (task, period) 都有 method 且权重和为 1
                for (t, p), grp in tdf.groupby(["task", "period"]):
                    if grp["method"].isna().any():
                        manifest.setdefault("warnings", []).append(
                            f"NNLS trace missing method for {t}/{p}"
                        )
                wcols = [c for c in tdf.columns if c.startswith("w_")]
                if wcols:
                    row_sum = tdf[wcols].sum(axis=1)
                    bad = (abs(row_sum - 1.0) > 0.01).sum()
                    if bad:
                        manifest.setdefault("warnings", []).append(
                            f"NNLS trace weight sum != 1 for {bad} row(s)"
                        )
