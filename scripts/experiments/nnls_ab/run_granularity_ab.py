#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""比较 period 级与 slot 级冠军学习/融合的历史代理实验。

只读取 96 点预测账本与价格实际账本；不读取错误宽表，不接入生产链路。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from fusion.learners.daily_ledger_gef import compute_daily_loss  # noqa: E402
from scripts.experiments.nnls_ab.run_champion_anchor_ab import (  # noqa: E402
    GATE_TOL,
    HALF_LIFE_DAYS,
    MIN_CHAMPION_WEIGHT,
    NEGATIVE_CAP,
    TRAIN_DAYS,
    WINDOW_DAYS,
    DayMatrix,
    champion_index,
    day_weights,
    evaluate_window,
    fit_signed_anchor,
    fit_weighted_nnls,
    load_task,
    one_hot,
)


def fit_slot_champion(mats: list[DayMatrix], q_days: np.ndarray) -> np.ndarray:
    n_slots, n_models = mats[0].X.shape
    score = np.zeros((n_slots, n_models), dtype=float)
    denom = float(q_days.sum())
    for q, mat in zip(q_days, mats):
        score += float(q) * np.abs(mat.X - mat.y[:, None])
    return np.eye(n_models, dtype=float)[np.argmin(score / denom, axis=1)]


def fit_slot_nnls(mats: list[DayMatrix], q_days: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls

    n_slots, n_models = mats[0].X.shape
    result = np.zeros((n_slots, n_models), dtype=float)
    for slot in range(n_slots):
        X = np.vstack([mat.X[[slot], :] for mat in mats])
        y = np.asarray([mat.y[slot] for mat in mats], dtype=float)
        q = np.sqrt(q_days)
        w, _ = nnls(X * q[:, None], y * q)
        total = float(w.sum())
        result[slot] = w / total if total > 1e-10 else np.ones(n_models) / n_models
    return result


def fit_slot_signed(mats: list[DayMatrix], q_days: np.ndarray) -> np.ndarray:
    n_slots, n_models = mats[0].X.shape
    result = np.zeros((n_slots, n_models), dtype=float)
    for slot in range(n_slots):
        slot_mats = [DayMatrix(X=mat.X[[slot], :], y=mat.y[[slot]]) for mat in mats]
        champion = champion_index(slot_mats, [str(i) for i in range(n_models)], q_days)
        result[slot] = fit_signed_anchor(
            slot_mats, q_days, champion,
            negative_cap=NEGATIVE_CAP,
            min_champion_weight=MIN_CHAMPION_WEIGHT,
        )
    return result


def evaluate_slot(mats: list[DayMatrix], weights: np.ndarray) -> dict[str, float]:
    losses = []
    for mat in mats:
        losses.append(compute_daily_loss(mat.y, np.sum(mat.X * weights, axis=1), "composite"))
    return {"composite": float(np.mean(losses))}


def main() -> None:
    rows = []
    for task in ("dayahead", "realtime"):
        day_data, models, periods = load_task(task)
        days = sorted(day_data)
        for target_day in days[WINDOW_DAYS:]:
            idx = days.index(target_day)
            window = days[idx - WINDOW_DAYS:idx]
            train = window[:TRAIN_DAYS]
            validation = window[TRAIN_DAYS:]
            q_train = day_weights(len(train), HALF_LIFE_DAYS)
            q_all = day_weights(len(window), HALF_LIFE_DAYS)
            for period in periods:
                train_mats = [day_data[d][period] for d in train]
                val_mats = [day_data[d][period] for d in validation]
                all_mats = [day_data[d][period] for d in window]
                target = [day_data[target_day][period]]
                champion = champion_index(train_mats, models, q_train)
                candidates = {
                    "period_champion": ("period", one_hot(len(models), champion)),
                    "period_nnls": ("period", fit_weighted_nnls(all_mats, q_all)),
                    "period_signed": ("period", fit_signed_anchor(all_mats, q_all, champion)),
                    "slot_champion": ("slot", fit_slot_champion(train_mats, q_train)),
                    "slot_nnls": ("slot", fit_slot_nnls(all_mats, q_all)),
                    "slot_signed": ("slot", fit_slot_signed(all_mats, q_all)),
                }
                val_scores = {}
                for name, (kind, w) in candidates.items():
                    val_scores[name] = evaluate_window(val_mats, w) if kind == "period" else evaluate_slot(val_mats, w)
                base = val_scores["period_champion"]["composite"]
                eligible = {
                    name: score["composite"]
                    for name, score in val_scores.items()
                    if score["composite"] <= base * (1 + GATE_TOL)
                }
                selected = min(eligible, key=eligible.get)
                target_scores = {}
                for name, (kind, w) in candidates.items():
                    target_scores[name] = (
                        evaluate_window(target, w) if kind == "period" else evaluate_slot(target, w)
                    )["composite"]
                row = {"task": task, "period": period, "target_day": target_day,
                       "selected": selected, "champion_model": models[champion]}
                for name, value in target_scores.items():
                    row[name] = value
                rows.append(row)
    result = pd.DataFrame(rows)
    out = PROJECT / "outputs/experiments/03_fusion_weighting/champion_weight_ab/granularity_results.csv"
    result.to_csv(out, index=False)
    summary = []
    for (task, period), group in result.groupby(["task", "period"]):
        for method in ["period_champion", "period_nnls", "period_signed", "slot_champion", "slot_nnls", "slot_signed"]:
            summary.append({"task": task, "period": period, "method": method,
                            "n": len(group), "mean_composite": group[method].mean(),
                            "mean_delta_vs_champion": (group[method] - group.period_champion).mean(),
                            "win_rate_vs_champion": (group[method] < group.period_champion).mean()})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out.with_name("granularity_summary.csv"), index=False)
    print(summary_df.to_string(index=False))
    print("\nselection counts")
    print(result.groupby(["task", "period", "selected"]).size().to_string())


if __name__ == "__main__":
    main()
