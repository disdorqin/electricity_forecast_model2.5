from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS  # noqa: E402
from pipelines.ledger_full_range import _prediction_day_audit  # noqa: E402
from pipelines.prediction_ledger import load_actual_ledger, load_prediction_ledger  # noqa: E402
from scripts.server.run_96_prediction_backtest import (  # noqa: E402
    _actual_day_audit,
    _protocol_manifest_audit,
)
from utils.resolution import QUARTER  # noqa: E402


EXPECTED = {
    "dayahead": list(DAYAHEAD_MODELS),
    "realtime": list(REALTIME_MODELS),
}
OOM_PATTERN = re.compile(r"(cuda\s+out\s+of\s+memory|out\s+of\s+memory|\boom\b)", re.I)


def _gpu_sample() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}

    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "utilization_percent": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                }
            )
        except ValueError:
            continue
    return {"available": bool(rows), "gpus": rows}


def _process_tree(root: psutil.Process) -> list[psutil.Process]:
    processes = []
    try:
        if root.is_running():
            processes.append(root)
            processes.extend(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    unique: dict[int, psutil.Process] = {}
    for process in processes:
        unique[process.pid] = process
    return list(unique.values())


def _sample_resources(
    root: psutil.Process,
    cpu_state: dict[int, tuple[float, float]],
) -> dict[str, Any]:
    rss_bytes = 0
    cpu_percent = 0.0
    pids: list[int] = []
    now = time.perf_counter()
    for process in _process_tree(root):
        try:
            pids.append(process.pid)
            rss_bytes += int(process.memory_info().rss)
            cpu_times = process.cpu_times()
            cpu_total = float(cpu_times.user + cpu_times.system)
            previous = cpu_state.get(process.pid)
            if previous is not None and now > previous[1]:
                cpu_percent += max(0.0, cpu_total - previous[0]) / (now - previous[1]) * 100.0
            cpu_state[process.pid] = (cpu_total, now)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    gpu = _gpu_sample()
    gpu_util = None
    gpu_mem = None
    if gpu.get("available"):
        gpu_util = max(float(item["utilization_percent"]) for item in gpu["gpus"])
        gpu_mem = max(float(item["memory_used_mib"]) for item in gpu["gpus"])

    return {
        "process_tree_pids": pids,
        "process_tree_rss_mib": rss_bytes / (1024.0 * 1024.0),
        "process_tree_cpu_percent": cpu_percent,
        "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
        "gpu_available": bool(gpu.get("available")),
        "gpu_utilization_percent": gpu_util,
        "gpu_memory_used_mib": gpu_mem,
    }


def _run_monitored(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
    sample_seconds: float = 1.0,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    samples = 0
    max_rss = 0.0
    max_tree_cpu = 0.0
    max_system_cpu = 0.0
    max_gpu_util: float | None = None
    max_gpu_mem: float | None = None
    gpu_available = False
    cpu_state: dict[int, tuple[float, float]] = {}

    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        ps = psutil.Process(process.pid)
        psutil.cpu_percent(interval=None)
        while process.poll() is None:
            sample = _sample_resources(ps, cpu_state)
            samples += 1
            max_rss = max(max_rss, float(sample["process_tree_rss_mib"]))
            max_tree_cpu = max(max_tree_cpu, float(sample["process_tree_cpu_percent"]))
            max_system_cpu = max(max_system_cpu, float(sample["system_cpu_percent"]))
            if sample["gpu_available"]:
                gpu_available = True
                util = sample["gpu_utilization_percent"]
                mem = sample["gpu_memory_used_mib"]
                if util is not None:
                    max_gpu_util = util if max_gpu_util is None else max(max_gpu_util, util)
                if mem is not None:
                    max_gpu_mem = mem if max_gpu_mem is None else max(max_gpu_mem, mem)
            time.sleep(max(0.1, sample_seconds))
        return_code = process.wait()

    wall = time.perf_counter() - started
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "return_code": int(return_code),
        "wall_seconds": round(wall, 3),
        "sample_count": samples,
        "peak_process_tree_rss_mib": round(max_rss, 2),
        "peak_process_tree_cpu_percent": round(max_tree_cpu, 2),
        "peak_system_cpu_percent": round(max_system_cpu, 2),
        "gpu_metrics_available": gpu_available,
        "peak_gpu_utilization_percent": None if max_gpu_util is None else round(max_gpu_util, 2),
        "peak_gpu_memory_used_mib": None if max_gpu_mem is None else round(max_gpu_mem, 2),
        "oom_detected": bool(OOM_PATTERN.search(text)),
        "log": str(log_path),
    }


def _load_prediction_manifest(mode_root: Path, target_date: str) -> dict[str, Any]:
    path = mode_root / "runs" / target_date / "run_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_elapsed(manifest: dict[str, Any]) -> dict[str, float]:
    elapsed: dict[str, float] = {}
    for task, models in EXPECTED.items():
        task_result = manifest.get("results", {}).get(task, {})
        for model in models:
            value = task_result.get(model, {}).get("elapsed_seconds")
            if value is not None:
                elapsed[f"{task}/{model}"] = float(value)
    return elapsed


def _prediction_file_audit(mode_root: Path, target_date: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for task, models in EXPECTED.items():
        pred_dir = mode_root / "runs" / target_date / task / "prediction"
        for model in models:
            path = pred_dir / f"{model}_predictions.csv"
            if not path.exists():
                errors.append(f"missing {task}/{model}")
                continue
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                errors.append(f"cannot read {task}/{model}: {exc}")
                continue
            if len(frame) != 96:
                errors.append(f"{task}/{model}: rows={len(frame)}")
            value = pd.to_numeric(frame.get("y_pred"), errors="coerce")
            if len(value) != 96 or not np.isfinite(value.to_numpy(dtype=float, na_value=np.nan)).all():
                errors.append(f"{task}/{model}: y_pred incomplete/non-finite")
            slot_col = "business_period" if "business_period" in frame.columns else "period"
            if slot_col not in frame.columns:
                errors.append(f"{task}/{model}: slot column missing")
            else:
                slots = pd.to_numeric(frame[slot_col], errors="coerce")
                if slots.isna().any() or sorted(slots.astype(int).unique()) != list(range(1, 97)):
                    errors.append(f"{task}/{model}: slots are not p1..p96")
    return not errors, errors


def _sort_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    for key in ("business_period", "period", "ds", "时刻"):
        if key in frame.columns:
            return frame.sort_values(key).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _numeric_diff(a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float) -> dict[str, Any]:
    abs_diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel_diff = np.divide(abs_diff, denom, out=np.zeros_like(abs_diff), where=denom > 0)
    passed = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
    return {
        "pass": passed,
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "max_rel_diff": float(rel_diff.max()) if rel_diff.size else 0.0,
    }


def compare_prediction_outputs(
    legacy_root: Path,
    split_root: Path,
    target_date: str,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    for task, models in EXPECTED.items():
        for model in models:
            key = f"{task}/{model}"
            a_path = legacy_root / "runs" / target_date / task / "prediction" / f"{model}_predictions.csv"
            b_path = split_root / "runs" / target_date / task / "prediction" / f"{model}_predictions.csv"
            if not a_path.exists() or not b_path.exists():
                details[key] = {"pass": False, "reason": "missing prediction file"}
                passed = False
                continue
            a = _sort_prediction(pd.read_csv(a_path))
            b = _sort_prediction(pd.read_csv(b_path))
            if len(a) != len(b) or "y_pred" not in a.columns or "y_pred" not in b.columns:
                details[key] = {"pass": False, "reason": "shape/y_pred mismatch"}
                passed = False
                continue
            result = _numeric_diff(
                pd.to_numeric(a["y_pred"], errors="coerce").to_numpy(float),
                pd.to_numeric(b["y_pred"], errors="coerce").to_numpy(float),
                atol=atol,
                rtol=rtol,
            )
            details[key] = result
            passed = passed and bool(result["pass"])
    return {"pass": passed, "details": details}


def _ledger_frame(
    root: Path,
    task: str,
    target_date: str,
    *,
    actual: bool,
) -> pd.DataFrame:
    if actual:
        frame = load_actual_ledger(root / "ledger", task, [target_date])
        keys = [c for c in ("target_day", "business_period", "hour_business") if c in frame.columns]
    else:
        frame = load_prediction_ledger(root / "ledger", task, [target_date])
        keys = [c for c in ("model_name", "target_day", "business_period", "hour_business") if c in frame.columns]
    return frame.sort_values(keys).reset_index(drop=True) if keys else frame.reset_index(drop=True)


def compare_ledgers(
    legacy_root: Path,
    split_root: Path,
    target_date: str,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    passed = True
    for task in EXPECTED:
        for actual in (False, True):
            name = f"{task}/{'actual' if actual else 'prediction'}"
            a = _ledger_frame(legacy_root, task, target_date, actual=actual)
            b = _ledger_frame(split_root, task, target_date, actual=actual)
            value_col = "y_true" if actual else "y_pred"
            key_cols = [
                c
                for c in (
                    "task",
                    "model_name",
                    "target_day",
                    "business_day",
                    "business_period",
                    "hour_business",
                )
                if c in a.columns and c in b.columns
            ]
            if len(a) != len(b) or value_col not in a.columns or value_col not in b.columns:
                details[name] = {"pass": False, "reason": "shape/value mismatch", "legacy_rows": len(a), "split_rows": len(b)}
                passed = False
                continue
            keys_equal = a[key_cols].astype(str).equals(b[key_cols].astype(str)) if key_cols else True
            numeric = _numeric_diff(
                pd.to_numeric(a[value_col], errors="coerce").to_numpy(float),
                pd.to_numeric(b[value_col], errors="coerce").to_numpy(float),
                atol=atol,
                rtol=rtol,
            )
            numeric["keys_equal"] = bool(keys_equal)
            numeric["pass"] = bool(numeric["pass"] and keys_equal)
            details[name] = numeric
            passed = passed and bool(numeric["pass"])
    return {"pass": passed, "details": details}


def _compare_csv(path_a: Path, path_b: Path, *, atol: float, rtol: float) -> dict[str, Any]:
    if not path_a.exists() or not path_b.exists():
        return {"pass": False, "reason": "missing file"}
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return {
            "pass": False,
            "reason": "shape/columns mismatch",
            "legacy_shape": list(a.shape),
            "split_shape": list(b.shape),
        }
    sort_cols = [c for c in ("task", "period", "model_name", "business_period", "hour_business", "ds") if c in a.columns]
    if sort_cols:
        a = a.sort_values(sort_cols).reset_index(drop=True)
        b = b.sort_values(sort_cols).reset_index(drop=True)
    details: dict[str, Any] = {}
    passed = True
    for column in a.columns:
        if pd.api.types.is_numeric_dtype(a[column]) or pd.api.types.is_numeric_dtype(b[column]):
            av = pd.to_numeric(a[column], errors="coerce").to_numpy(float)
            bv = pd.to_numeric(b[column], errors="coerce").to_numpy(float)
            if np.isnan(av).any() or np.isnan(bv).any():
                same = bool(np.array_equal(np.isnan(av), np.isnan(bv)))
                finite = np.isfinite(av) & np.isfinite(bv)
                result = _numeric_diff(av[finite], bv[finite], atol=atol, rtol=rtol)
                result["nan_mask_equal"] = same
                result["pass"] = bool(result["pass"] and same)
            else:
                result = _numeric_diff(av, bv, atol=atol, rtol=rtol)
            details[column] = result
            passed = passed and bool(result["pass"])
        else:
            same = a[column].fillna("<NA>").astype(str).equals(b[column].fillna("<NA>").astype(str))
            details[column] = {"pass": bool(same)}
            passed = passed and bool(same)
    return {"pass": passed, "columns": details}


def compare_downstream_outputs(
    legacy_root: Path,
    split_root: Path,
    target_date: str,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    relative_files = []
    for task in EXPECTED:
        relative_files.extend(
            [
                Path(task) / "weight" / "weights.csv",
                Path(task) / "fuse" / "fused_predictions.csv",
                Path(task) / "fuse" / "model_quality_gate.csv",
            ]
        )
    relative_files.extend(
        [
            Path("final") / "dayahead_final_predictions.csv",
            Path("final") / "realtime_final_predictions.csv",
            Path("final") / "submission_ready.csv",
        ]
    )

    details: dict[str, Any] = {}
    passed = True
    for relative in relative_files:
        result = _compare_csv(
            legacy_root / "runs" / target_date / relative,
            split_root / "runs" / target_date / relative,
            atol=atol,
            rtol=rtol,
        )
        details[str(relative)] = result
        passed = passed and bool(result["pass"])

    # Formal 96 production bypasses ExtremePriceClf.  Compare the explicit
    # policy marker instead of requiring a classifier report.
    policy_values = []
    for root in (legacy_root, split_root):
        manifest = root / "runs" / target_date / "run_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        policy_values.append(payload.get("classifier_policy"))
    policy_equal = policy_values == ["disabled_by_production_policy"] * 2
    details["classifier_policy"] = {"pass": policy_equal, "values": policy_values}
    passed = passed and policy_equal
    return {"pass": passed, "details": details}


def _validate_seed_history(
    *,
    ledger_root: Path,
    runs_root: Path,
    target_date: str,
    resource_mode: str,
    days: int = 30,
) -> list[str]:
    errors: list[str] = []
    target = pd.Timestamp(target_date)
    history = [
        (target - pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(days, 0, -1)
    ]
    for day in history:
        protocol_ok, protocol_reasons = _protocol_manifest_audit(
            runs_root, day, resource_mode=resource_mode
        )
        if not protocol_ok:
            errors.extend(f"{day}: {reason}" for reason in protocol_reasons)
        prediction_ok, prediction_reasons = _prediction_day_audit(ledger_root, day, QUARTER)
        actual_ok, actual_reasons = _actual_day_audit(ledger_root, day)
        if not prediction_ok:
            errors.extend(f"{day}: {reason}" for reason in prediction_reasons)
        if not actual_ok:
            errors.extend(f"{day}: {reason}" for reason in actual_reasons)
    return errors


def _prepare_mode_root(
    mode_root: Path,
    *,
    force: bool,
    seed_ledger_root: Path | None,
    seed_cache_root: Path | None,
) -> None:
    if mode_root.exists():
        if not force:
            raise FileExistsError(
                f"A/B root already exists: {mode_root}. Pass --force only for this isolated benchmark root."
            )
        shutil.rmtree(mode_root)
    mode_root.mkdir(parents=True, exist_ok=True)
    if seed_ledger_root is not None:
        shutil.copytree(seed_ledger_root, mode_root / "ledger", dirs_exist_ok=True)
    if seed_cache_root is not None and seed_cache_root.exists():
        shutil.copytree(seed_cache_root, mode_root / "cache", dirs_exist_ok=True)


def _prediction_command(args: argparse.Namespace, mode: str, mode_root: Path) -> list[str]:
    cpu_workers = 2 if mode == "split_process" else 1
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "server" / "run_96_prediction_backtest.py"),
        "--report-start", args.date,
        "--end", args.date,
        "--limit-days", "1",
        "--data-path", str(args.data_path),
        "--actual-data-path", str(args.actual_data_path),
        "--output-root", str(mode_root),
        "--resource-mode", mode,
        "--max-cpu-workers", str(cpu_workers),
        "--max-gpu-workers", "1",
        "--training-months", str(args.training_months),
        "--rt916-train-steps", str(args.rt916_train_steps),
        "--seed", str(args.seed),
        "--force",
    ]


def _downstream_command(args: argparse.Namespace, mode: str, mode_root: Path) -> list[str]:
    cpu_workers = 2 if mode == "split_process" else 1
    return [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--pipeline", "ledger_full_range",
        "--start", args.date,
        "--end", args.date,
        "--resolution", "15min",
        "--output-profile", "production",
        "--data-path", str(args.data_path),
        "--actual-data-path", str(args.actual_data_path),
        "--ledger-root", str(mode_root / "ledger"),
        "--runs-root", str(mode_root / "runs"),
        "--feature-store-root", str(mode_root / "cache"),
        "--resource-mode", mode,
        "--realtime-cutoff-hour", "15",
        "--max-cpu-workers", str(cpu_workers),
        "--max-gpu-workers", "1",
        "--seed", str(args.seed),
        "--replay-only",
        "--require-target-actual",
    ]


def promotion_gate(
    *,
    legacy: dict[str, Any],
    split: dict[str, Any],
    prediction_equivalent: bool,
    ledger_equivalent: bool,
    downstream_compared: bool,
    downstream_equivalent: bool,
    model_completeness: bool,
    min_speedup_percent: float,
    max_model_slowdown_percent: float,
) -> dict[str, Any]:
    legacy_wall = float(legacy["monitor"]["wall_seconds"])
    split_wall = float(split["monitor"]["wall_seconds"])
    speedup = 0.0 if legacy_wall <= 0 else (legacy_wall - split_wall) / legacy_wall * 100.0

    slowdown_details: dict[str, float] = {}
    contention_suspected = False
    for key, legacy_elapsed in legacy.get("model_elapsed_seconds", {}).items():
        split_elapsed = split.get("model_elapsed_seconds", {}).get(key)
        if split_elapsed is None or legacy_elapsed <= 0:
            continue
        slowdown = (float(split_elapsed) - float(legacy_elapsed)) / float(legacy_elapsed) * 100.0
        slowdown_details[key] = round(slowdown, 2)
        if slowdown > max_model_slowdown_percent:
            contention_suspected = True

    resource_metrics_complete = bool(
        legacy["monitor"].get("gpu_metrics_available")
        and split["monitor"].get("gpu_metrics_available")
        and legacy["monitor"].get("sample_count", 0) > 0
        and split["monitor"].get("sample_count", 0) > 0
    )
    stability_ok = bool(
        legacy["monitor"].get("return_code") == 0
        and split["monitor"].get("return_code") == 0
        and not legacy["monitor"].get("oom_detected")
        and not split["monitor"].get("oom_detected")
        and not contention_suspected
    )
    speed_ok = speedup >= min_speedup_percent

    checks = {
        "both_modes_complete_without_oom": stability_ok,
        "resource_metrics_complete": resource_metrics_complete,
        "seven_model_outputs_complete": model_completeness,
        "prediction_values_equivalent": prediction_equivalent,
        "ledger_values_equivalent": ledger_equivalent,
        "downstream_compared": downstream_compared,
        "weight_fuse_final_equivalent": downstream_compared and downstream_equivalent,
        "no_model_contention_suspected": not contention_suspected,
        "speedup_meets_threshold": speed_ok,
    }
    eligible = all(checks.values())
    return {
        "eligible_for_default_promotion_review": eligible,
        "decision": (
            "ELIGIBLE_FOR_DEFAULT_PROMOTION_REVIEW"
            if eligible
            else "KEEP_LEGACY_DEFAULT"
        ),
        "speedup_percent": round(speedup, 2),
        "min_speedup_percent": float(min_speedup_percent),
        "max_model_slowdown_percent": float(max_model_slowdown_percent),
        "contention_suspected": contention_suspected,
        "per_model_slowdown_percent": slowdown_details,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Server A/B benchmark for legacy vs split_process 96-point scheduling."
    )
    parser.add_argument("--date", default="2026-08-16")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "96" / "model_input" / "shandong_pmos_96_model_input_full.parquet",
    )
    parser.add_argument(
        "--actual-data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "96" / "authoritative" / "pmos_96_全量.csv",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--training-months", type=int, default=12)
    parser.add_argument("--rt916-train-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--prediction-atol", type=float, default=1e-5)
    parser.add_argument("--prediction-rtol", type=float, default=1e-6)
    parser.add_argument("--min-speedup-percent", type=float, default=10.0)
    parser.add_argument("--max-model-slowdown-percent", type=float, default=20.0)
    parser.add_argument("--run-downstream", action="store_true")
    parser.add_argument("--seed-ledger-root", type=Path)
    parser.add_argument("--seed-runs-root", type=Path)
    parser.add_argument("--seed-cache-root", type=Path)
    parser.add_argument("--seed-resource-mode", choices=("legacy", "split_process"), default="legacy")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--require-promotion-gate",
        action="store_true",
        help="Return non-zero unless every promotion condition passes.",
    )
    args = parser.parse_args(argv)

    if args.output_root is None:
        args.output_root = (
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "04_pipeline_audits"
            / f"resource_mode_ab_{args.date.replace('-', '')}"
        )
    else:
        args.output_root = args.output_root.resolve()
    args.data_path = args.data_path.resolve()
    args.actual_data_path = args.actual_data_path.resolve()

    if args.run_downstream:
        if args.seed_ledger_root is None or args.seed_runs_root is None:
            parser.error("--run-downstream requires --seed-ledger-root and --seed-runs-root")
        seed_errors = _validate_seed_history(
            ledger_root=args.seed_ledger_root.resolve(),
            runs_root=args.seed_runs_root.resolve(),
            target_date=args.date,
            resource_mode=args.seed_resource_mode,
            days=30,
        )
        if seed_errors:
            raise RuntimeError(
                "STRICT_SEED_HISTORY_NOT_READY: " + " | ".join(seed_errors[:20])
            )

    modes: dict[str, dict[str, Any]] = {}
    for mode in ("legacy", "split_process"):
        mode_root = args.output_root / mode
        _prepare_mode_root(
            mode_root,
            force=args.force,
            seed_ledger_root=None if args.seed_ledger_root is None else args.seed_ledger_root.resolve(),
            seed_cache_root=None if args.seed_cache_root is None else args.seed_cache_root.resolve(),
        )
        monitor = _run_monitored(
            _prediction_command(args, mode, mode_root),
            log_path=mode_root / "benchmark_prediction.log",
            env={**os.environ, "TIMESFM_DEVICE": os.environ.get("TIMESFM_DEVICE", "cpu")},
            sample_seconds=args.sample_seconds,
        )
        manifest = _load_prediction_manifest(mode_root, args.date)
        files_ok, file_errors = _prediction_file_audit(mode_root, args.date)
        protocol_ok, protocol_errors = _protocol_manifest_audit(
            mode_root / "runs", args.date, resource_mode=mode
        )
        modes[mode] = {
            "root": str(mode_root),
            "monitor": monitor,
            "model_elapsed_seconds": _model_elapsed(manifest),
            "model_files_complete": files_ok,
            "model_file_errors": file_errors,
            "protocol_audit": "PASS" if protocol_ok else "FAIL",
            "protocol_errors": protocol_errors,
        }

    legacy_root = Path(modes["legacy"]["root"])
    split_root = Path(modes["split_process"]["root"])
    predictions = compare_prediction_outputs(
        legacy_root,
        split_root,
        args.date,
        atol=args.prediction_atol,
        rtol=args.prediction_rtol,
    )
    ledgers = compare_ledgers(
        legacy_root,
        split_root,
        args.date,
        atol=args.prediction_atol,
        rtol=args.prediction_rtol,
    )

    downstream_compared = False
    downstream = {"pass": False, "status": "NOT_RUN"}
    if (
        args.run_downstream
        and modes["legacy"]["monitor"]["return_code"] == 0
        and modes["split_process"]["monitor"]["return_code"] == 0
        and modes["legacy"]["model_files_complete"]
        and modes["split_process"]["model_files_complete"]
    ):
        downstream_runs = {}
        for mode, mode_root in (("legacy", legacy_root), ("split_process", split_root)):
            downstream_runs[mode] = _run_monitored(
                _downstream_command(args, mode, mode_root),
                log_path=mode_root / "benchmark_downstream.log",
                env={**os.environ, "TIMESFM_DEVICE": os.environ.get("TIMESFM_DEVICE", "cpu")},
                sample_seconds=args.sample_seconds,
            )
        downstream_compared = all(
            item["return_code"] == 0 and not item["oom_detected"]
            for item in downstream_runs.values()
        )
        if downstream_compared:
            downstream = compare_downstream_outputs(
                legacy_root,
                split_root,
                args.date,
                atol=args.prediction_atol,
                rtol=args.prediction_rtol,
            )
            downstream["status"] = "COMPARED"
        else:
            downstream = {
                "pass": False,
                "status": "DOWNSTREAM_RUN_FAILED",
                "runs": downstream_runs,
            }

    model_completeness = bool(
        modes["legacy"]["model_files_complete"]
        and modes["split_process"]["model_files_complete"]
        and modes["legacy"]["protocol_audit"] == "PASS"
        and modes["split_process"]["protocol_audit"] == "PASS"
    )
    gate = promotion_gate(
        legacy=modes["legacy"],
        split=modes["split_process"],
        prediction_equivalent=bool(predictions["pass"]),
        ledger_equivalent=bool(ledgers["pass"]),
        downstream_compared=downstream_compared,
        downstream_equivalent=bool(downstream.get("pass")),
        model_completeness=model_completeness,
        min_speedup_percent=args.min_speedup_percent,
        max_model_slowdown_percent=args.max_model_slowdown_percent,
    )

    report = {
        "status": "COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": args.date,
        "contract": {
            "cutoff_hour": 15,
            "cutoff_period": 60,
            "cpu_workers_by_mode": {"legacy": 1, "split_process": 2},
            "dag_aware_by_mode": {"legacy": False, "split_process": True},
            "gpu_serial_by_mode": {"legacy": True, "split_process": True},
            "max_gpu_workers": 1,
            "production_default_before_ab": "legacy",
            "prediction_atol": args.prediction_atol,
            "prediction_rtol": args.prediction_rtol,
        },
        "modes": modes,
        "prediction_equivalence": predictions,
        "ledger_equivalence": ledgers,
        "downstream_equivalence": downstream,
        "promotion_gate": gate,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "resource_mode_ab_report.json"
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(report_path)
    print(json.dumps({"report": str(report_path), **gate}, ensure_ascii=False, indent=2))

    if args.require_promotion_gate and not gate["eligible_for_default_promotion_review"]:
        return 2
    return 0 if all(modes[mode]["monitor"]["return_code"] == 0 for mode in modes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
