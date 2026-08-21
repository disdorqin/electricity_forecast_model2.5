#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""96点历史账本上的冠军锚定权重学习器实验。

本实验只读取 outputs/ledger_96 下已经生成的预测账本和价格实际账本，
不读取污染宽表重新生成特征，也不修改生产学习器。

协议：
  - 固定最近30个完整历史日；
  - 前23日学习，后7日验证；
  - 日样本按指数衰减加权（half-life=7d）；
  - 候选：冠军单模型、加权NNLS、冠军锚定有界负权岭回归；
  - 验证集不劣于冠军才允许将候选用于目标日；
  - 目标日只做最终评估，不参与学习。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from fusion.learners.daily_ledger_gef import compute_daily_loss, mae_percent, smape_floor50  # noqa: E402
from utils.resolution import resolve_resolution  # noqa: E402


RES = resolve_resolution("15min")
HALF_LIFE_DAYS = 7.0
TRAIN_DAYS = 23
WINDOW_DAYS = 30
GATE_TOL = 0.005
NEGATIVE_CAP = 0.25
MIN_CHAMPION_WEIGHT = 0.50


@dataclass(frozen=True)
class DayMatrix:
    X: np.ndarray
    y: np.ndarray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_task(task: str) -> tuple[dict[str, dict[str, DayMatrix]], list[str], list[str]]:
    """Load only prediction/price ledgers and build complete day-period matrices."""
    pred_path = PROJECT / "outputs" / "ledger_96" / task / "prediction" / "prediction_ledger.parquet"
    act_path = PROJECT / "outputs" / "ledger_96" / task / "actual" / "actual_ledger.parquet"
    pred = pd.read_parquet(pred_path)
    act = pd.read_parquet(act_path)
    pred["target_day"] = pred["target_day"].astype(str)
    act["target_day"] = act["target_day"].astype(str)
    models = sorted(pred["model_name"].dropna().unique().tolist())
    periods = list(RES.period_names)

    pred_groups = {
        key: group.sort_values(RES.slot_column)
        for key, group in pred.groupby(["target_day", "period"], sort=False)
    }
    act_groups = {
        key: group.sort_values(RES.slot_column)
        for key, group in act.groupby(["target_day", "period"], sort=False)
    }

    day_data: dict[str, dict[str, DayMatrix]] = {}
    for day in sorted(set(pred["target_day"]) & set(act["target_day"])):
        per_data: dict[str, DayMatrix] = {}
        for period in periods:
            pg = pred_groups.get((day, period))
            ag = act_groups.get((day, period))
            if pg is None or ag is None:
                continue
            if len(ag) != RES.slots_per_period or ag["y_true"].isna().any():
                continue
            wide = pg.pivot_table(
                index=RES.slot_column,
                columns="model_name",
                values="y_pred",
                aggfunc="first",
            ).reindex(columns=models)
            actual = ag.drop_duplicates(RES.slot_column).set_index(RES.slot_column)["y_true"]
            wide = wide.reindex(actual.index)
            if len(wide) != RES.slots_per_period or wide.isna().any().any():
                continue
            per_data[period] = DayMatrix(
                X=wide.to_numpy(dtype=float),
                y=actual.to_numpy(dtype=float),
            )
        if len(per_data) == len(periods):
            day_data[day] = per_data

    return day_data, models, periods


def day_weights(n_days: int, half_life: float = HALF_LIFE_DAYS) -> np.ndarray:
    age = np.arange(n_days - 1, -1, -1, dtype=float)
    q = np.power(0.5, age / half_life)
    return q / q.mean()


def stack_window(mats: list[DayMatrix], q_days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.vstack([m.X for m in mats])
    y = np.concatenate([m.y for m in mats])
    q = np.repeat(q_days, [m.X.shape[0] for m in mats])
    return X, y, q


def weighted_model_losses(mats: list[DayMatrix], models: list[str], q_days: np.ndarray) -> dict[str, float]:
    denom = float(q_days.sum())
    result: dict[str, float] = {}
    for j, model in enumerate(models):
        losses = [compute_daily_loss(m.y, m.X[:, j], "composite") for m in mats]
        result[model] = float(np.dot(q_days, losses) / denom)
    return result


def champion_index(mats: list[DayMatrix], models: list[str], q_days: np.ndarray) -> int:
    losses = weighted_model_losses(mats, models, q_days)
    return min(range(len(models)), key=lambda j: losses[models[j]])


def fit_weighted_nnls(mats: list[DayMatrix], q_days: np.ndarray) -> np.ndarray:
    """Weighted raw-scale NNLS; unlike the old implementation, no post-standardization mismatch."""
    X, y, q = stack_window(mats, q_days)
    try:
        from scipy.optimize import nnls

        sol, _ = nnls(X * np.sqrt(q)[:, None], y * np.sqrt(q))
    except Exception:
        sol = np.ones(X.shape[1], dtype=float)
    total = float(sol.sum())
    if not np.isfinite(total) or total <= 1e-10:
        return np.ones(X.shape[1], dtype=float) / X.shape[1]
    return sol / total


def fit_signed_anchor(
    mats: list[DayMatrix],
    q_days: np.ndarray,
    champion: int,
    negative_cap: float = NEGATIVE_CAP,
    min_champion_weight: float = MIN_CHAMPION_WEIGHT,
) -> np.ndarray:
    """Fit small signed residual corrections around a champion model."""
    X, y, q = stack_window(mats, q_days)
    other = [j for j in range(X.shape[1]) if j != champion]
    residual_features = X[:, other] - X[:, [champion]]
    residual_target = y - X[:, champion]
    A = residual_features.T @ (q[:, None] * residual_features)
    b = residual_features.T @ (q * residual_target)
    trace_scale = float(np.trace(A)) / max(len(other), 1)
    ridge = max(0.10 * trace_scale, 1e-8)
    try:
        alpha = np.linalg.solve(A + ridge * np.eye(len(other)), b)
    except np.linalg.LinAlgError:
        alpha = np.zeros(len(other), dtype=float)
    alpha = np.clip(alpha, -negative_cap, negative_cap)
    champion_weight = 1.0 - float(alpha.sum())
    if champion_weight < min_champion_weight and float(alpha.sum()) > 0:
        alpha *= (1.0 - min_champion_weight) / float(alpha.sum())
    weights = np.zeros(X.shape[1], dtype=float)
    weights[other] = alpha
    weights[champion] = 1.0 - float(alpha.sum())
    return weights


def one_hot(n: int, index: int) -> np.ndarray:
    w = np.zeros(n, dtype=float)
    w[index] = 1.0
    return w


def evaluate_window(mats: list[DayMatrix], weights: np.ndarray) -> dict[str, float]:
    losses = [compute_daily_loss(m.y, m.X @ weights, "composite") for m in mats]
    smapes = [smape_floor50(m.y, m.X @ weights) for m in mats]
    maes = [mae_percent(m.y, m.X @ weights) for m in mats]
    return {
        "composite": float(np.mean(losses)),
        "smape": float(np.mean(smapes)),
        "mae_percent": float(np.mean(maes)),
        "daily_win_rate_nonzero": float(np.mean(np.asarray(losses) < 1e9)),
    }


def run_task(task: str, step: int, max_target_days: int | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    day_data, models, periods = load_task(task)
    days = sorted(day_data)
    target_days = days[WINDOW_DAYS:]
    if step > 1:
        target_days = target_days[::step]
    if max_target_days is not None:
        target_days = target_days[:max_target_days]

    rows: list[dict[str, Any]] = []
    for target_day in target_days:
        target_idx = days.index(target_day)
        window_days = days[target_idx - WINDOW_DAYS:target_idx]
        train_days = window_days[:TRAIN_DAYS]
        validation_days = window_days[TRAIN_DAYS:]
        q_train = day_weights(len(train_days))
        q_all = day_weights(len(window_days))

        for period in periods:
            train_mats = [day_data[d][period] for d in train_days]
            validation_mats = [day_data[d][period] for d in validation_days]
            all_mats = [day_data[d][period] for d in window_days]
            target_mat = day_data[target_day][period]
            champion = champion_index(train_mats, models, q_train)
            w_champion = one_hot(len(models), champion)
            w_nnls = fit_weighted_nnls(train_mats, q_train)
            w_signed = fit_signed_anchor(train_mats, q_train, champion)

            val_scores = {
                "champion": evaluate_window(validation_mats, w_champion),
                "weighted_nnls": evaluate_window(validation_mats, w_nnls),
                "signed_anchor": evaluate_window(validation_mats, w_signed),
            }
            champion_val = val_scores["champion"]["composite"]
            eligible = {
                name: score["composite"]
                for name, score in val_scores.items()
                if score["composite"] <= champion_val * (1.0 + GATE_TOL)
            }
            selected_strategy = min(eligible, key=eligible.get)

            # 通过验证后才在完整30日上重新拟合；冠军身份只由前23日决定。
            if selected_strategy == "weighted_nnls":
                w_selected = fit_weighted_nnls(all_mats, q_all)
            elif selected_strategy == "signed_anchor":
                w_selected = fit_signed_anchor(all_mats, q_all, champion)
            else:
                w_selected = w_champion
            w_nnls_final = fit_weighted_nnls(all_mats, q_all)
            w_signed_final = fit_signed_anchor(all_mats, q_all, champion)

            target_scores = {
                "champion": evaluate_window([target_mat], w_champion),
                "weighted_nnls": evaluate_window([target_mat], w_nnls_final),
                "signed_anchor": evaluate_window([target_mat], w_signed_final),
                "selected": evaluate_window([target_mat], w_selected),
                "oracle": evaluate_window(
                    [target_mat],
                    one_hot(len(models), int(np.argmin([
                        compute_daily_loss(target_mat.y, target_mat.X[:, j], "composite")
                        for j in range(len(models))
                    ]))),
                ),
            }
            row: dict[str, Any] = {
                "task": task,
                "target_day": target_day,
                "period": period,
                "champion_model": models[champion],
                "selected_strategy": selected_strategy,
                "gate_tol": GATE_TOL,
                "train_days": len(train_days),
                "validation_days": len(validation_days),
                "window_days": len(window_days),
            }
            for name, score in val_scores.items():
                row[f"val_{name}_composite"] = score["composite"]
            for name, score in target_scores.items():
                row[f"target_{name}_composite"] = score["composite"]
                row[f"target_{name}_smape"] = score["smape"]
                row[f"target_{name}_mae_percent"] = score["mae_percent"]
            for name, weights in {
                "selected": w_selected,
                "signed": w_signed_final,
                "nnls": w_nnls_final,
            }.items():
                for model, weight in zip(models, weights):
                    row[f"{name}_w_{model}"] = float(weight)
            rows.append(row)

    metadata = {
        "task": task,
        "models": models,
        "periods": periods,
        "complete_days": len(days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "target_days_evaluated": len(target_days),
        "step": step,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return pd.DataFrame(rows), metadata


def build_manifest(step: int, max_target_days: int | None) -> dict[str, Any]:
    paths = {
        "dayahead_prediction": PROJECT / "outputs/ledger_96/dayahead/prediction/prediction_ledger.parquet",
        "dayahead_actual": PROJECT / "outputs/ledger_96/dayahead/actual/actual_ledger.parquet",
        "realtime_prediction": PROJECT / "outputs/ledger_96/realtime/prediction/prediction_ledger.parquet",
        "realtime_actual": PROJECT / "outputs/ledger_96/realtime/actual/actual_ledger.parquet",
    }
    return {
        "experiment": "champion_anchor_weight_ab",
        "status": "historical-invalid-features-proxy-only",
        "resolution": RES.label,
        "window_days": WINDOW_DAYS,
        "train_days": TRAIN_DAYS,
        "validation_days": WINDOW_DAYS - TRAIN_DAYS,
        "half_life_days": HALF_LIFE_DAYS,
        "gate_tolerance": GATE_TOL,
        "negative_cap": NEGATIVE_CAP,
        "min_champion_weight": MIN_CHAMPION_WEIGHT,
        "step": step,
        "max_target_days": max_target_days,
        "inputs": {
            str(path.relative_to(PROJECT)): {
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for path in paths.values()
        },
        "forbidden_input": "data/96/model_input/shandong_pmos_96_model_input.xlsx",
        "old_baselines": [
            "outputs/experiments/nnls_ab/summary.csv",
            "outputs/experiments/nnls_ab/negative_w_grid.csv",
        ],
    }


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, period), group in result.groupby(["task", "period"]):
        for method in ["champion", "weighted_nnls", "signed_anchor", "selected", "oracle"]:
            col = f"target_{method}_composite"
            if col not in group:
                continue
            rows.append({
                "task": task,
                "period": period,
                "method": method,
                "n": len(group),
                "mean_composite": group[col].mean(),
                "median_composite": group[col].median(),
                "win_rate_vs_champion": (
                    (group[col] < group["target_champion_composite"]).mean()
                    if method != "champion" else 0.0
                ),
                "mean_delta_vs_champion": (
                    (group[col] - group["target_champion_composite"]).mean()
                    if method != "champion" else 0.0
                ),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["dayahead", "realtime", "all"], default="all")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max-target-days", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/experiments/champion_weight_ab",
    )
    args = parser.parse_args()
    if args.step < 1:
        raise ValueError("--step must be >= 1")

    out = PROJECT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.step, args.max_target_days)
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tasks = ["dayahead", "realtime"] if args.task == "all" else [args.task]
    all_results = []
    metadata = {}
    for task in tasks:
        result, meta = run_task(task, args.step, args.max_target_days)
        result.to_csv(out / f"rolling_results_{task}.csv", index=False)
        summary = summarize(result)
        summary.to_csv(out / f"summary_{task}.csv", index=False)
        all_results.append(result)
        metadata[task] = meta
        print(json.dumps(meta, ensure_ascii=False))

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    combined.to_csv(out / "rolling_results_all.csv", index=False)
    summarize(combined).to_csv(out / "summary_all.csv", index=False)
    (out / "runtime.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={out}")
    print(summarize(combined).to_string(index=False))


if __name__ == "__main__":
    main()
