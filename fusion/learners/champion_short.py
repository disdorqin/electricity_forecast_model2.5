"""短窗口冠军门控学习器（实验接入版）。

该模块只负责从历史 ledger 学习每个 ``(task, period)`` 的目标日权重，
不参与分类器。生产入口需显式选择 ``--weight-learner champion_short``，
默认 NNLS 行为保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fusion.learners.daily_ledger_gef import (
    compute_daily_loss,
    mae_percent,
    smape_floor50,
)


@dataclass(frozen=True)
class ChampionShortConfig:
    window_days: int = 14
    validation_days: int = 7
    half_life_days: float = 7.0
    gate_tolerance: float = 0.005
    meta_gate_tolerance: float = 0.0
    meta_alpha: float = 1000.0
    meta_blend_rho: float = 0.30
    negative_cap: float = 0.25
    min_champion_weight: float = 0.50


def _day_weights(n_days: int, half_life: float) -> np.ndarray:
    age = np.arange(n_days - 1, -1, -1, dtype=float)
    values = np.power(0.5, age / max(float(half_life), 1e-6))
    return values / values.mean()


def _resolve_slot_column(table: pd.DataFrame, preferred: str) -> str:
    """Resolve the row key without collapsing 96-point data to 24 hours.

    ``build_ledger_training_table`` intentionally exposes the common ``ds``
    column but not ``business_period``.  For 15-minute data, falling back to
    ``hour_business`` would merge four distinct slots, so ``ds`` is the safe
    resolution-independent key whenever the preferred slot column is absent.
    """
    if preferred in table.columns:
        return preferred
    if "ds" in table.columns:
        return "ds"
    for candidate in ("business_period", "hour_business"):
        if candidate in table.columns:
            return candidate
    raise KeyError(
        f"No slot key in champion_short table; preferred={preferred!r}, "
        f"columns={list(table.columns)!r}"
    )


def _matrix_by_day(
    table: pd.DataFrame,
    models: list[str],
    periods: list[str],
    slot_column: str,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Build complete ``day/period -> (X, y)`` matrices from long ledger rows."""
    output: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    slot_column = _resolve_slot_column(table, slot_column)
    for (day, period), group in table.groupby(["target_day", "period"], sort=False):
        wide = group.pivot_table(
            index=slot_column,
            columns="model_name",
            values="y_pred",
            aggfunc="first",
        ).reindex(columns=models)
        actual = (
            group.drop_duplicates(slot_column)
            .set_index(slot_column)["y_true"]
        )
        wide = wide.reindex(actual.index)
        if (
            period not in periods
            or len(wide) == 0
            or len(wide) != len(actual)
            or wide.isna().any().any()
            or actual.isna().any()
        ):
            continue
        output.setdefault(str(day), {})[str(period)] = (
            wide.to_numpy(dtype=float),
            actual.to_numpy(dtype=float),
        )
    return output


def _target_matrices(
    table: pd.DataFrame,
    models: list[str],
    periods: list[str],
    slot_column: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    slot_column = _resolve_slot_column(table, slot_column)
    for period, group in table.groupby("period", sort=False):
        wide = group.pivot_table(
            index=slot_column,
            columns="model_name",
            values="y_pred",
            aggfunc="first",
        ).reindex(columns=models)
        if period in periods and not wide.empty and not wide.isna().any().any():
            output[str(period)] = wide.to_numpy(dtype=float)
    return output


def _target_completeness_diagnostics(
    table: pd.DataFrame,
    models: list[str],
    periods: list[str],
    slot_column: str,
    expected_slots_per_period: int,
) -> list[str]:
    """Explain missing target models/slots before refusing to fuse."""
    slot_column = _resolve_slot_column(table, slot_column)
    diagnostics: list[str] = []
    for period in periods:
        group = table[table["period"].astype(str) == str(period)]
        present = set(group["model_name"].astype(str).unique())
        missing = [model for model in models if model not in present]
        bad_slots = {
            model: int(group[group["model_name"] == model][slot_column].nunique())
            for model in models
            if model in present
            and group[group["model_name"] == model][slot_column].nunique()
            != expected_slots_per_period
        }
        if missing or bad_slots:
            detail = [f"period={period}"]
            if missing:
                detail.append(f"missing_models={missing}")
            if bad_slots:
                detail.append(f"bad_slot_counts={bad_slots}")
            diagnostics.append(" ".join(detail))
    return diagnostics


def _champion(mats: list[tuple[np.ndarray, np.ndarray]], q: np.ndarray) -> int:
    losses = np.asarray([
        [compute_daily_loss(y, x[:, j], "composite") for j in range(x.shape[1])]
        for x, y in mats
    ])
    return int(np.argmin((q[:, None] * losses).sum(axis=0) / q.sum()))


def _weighted_loss(mats: list[tuple[np.ndarray, np.ndarray]], weights: np.ndarray) -> float:
    return float(np.mean([
        compute_daily_loss(y, x @ weights, "composite") for x, y in mats
    ]))


def _weighted_model_loss(mats: list[tuple[np.ndarray, np.ndarray]], j: int, q: np.ndarray) -> float:
    values = [compute_daily_loss(y, x[:, j], "composite") for x, y in mats]
    return float(np.dot(q, values) / q.sum())


def _fit_nnls(mats: list[tuple[np.ndarray, np.ndarray]], q: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls

    x = np.vstack([item[0] for item in mats])
    y = np.concatenate([item[1] for item in mats])
    row_q = np.repeat(q, [item[0].shape[0] for item in mats])
    try:
        result, _ = nnls(x * np.sqrt(row_q)[:, None], y * np.sqrt(row_q))
    except Exception:
        result = np.ones(x.shape[1], dtype=float)
    total = float(result.sum())
    return result / total if total > 1e-10 else np.ones(x.shape[1]) / x.shape[1]


def _fit_signed(
    mats: list[tuple[np.ndarray, np.ndarray]],
    q: np.ndarray,
    champion: int,
    config: ChampionShortConfig,
) -> np.ndarray:
    x = np.vstack([item[0] for item in mats])
    y = np.concatenate([item[1] for item in mats])
    row_q = np.repeat(q, [item[0].shape[0] for item in mats])
    others = [j for j in range(x.shape[1]) if j != champion]
    residual_x = x[:, others] - x[:, [champion]]
    residual_y = y - x[:, champion]
    gram = residual_x.T @ (row_q[:, None] * residual_x)
    rhs = residual_x.T @ (row_q * residual_y)
    ridge = max(0.10 * float(np.trace(gram)) / max(len(others), 1), 1e-8)
    try:
        alpha = np.linalg.solve(gram + ridge * np.eye(len(others)), rhs)
    except np.linalg.LinAlgError:
        alpha = np.zeros(len(others), dtype=float)
    alpha = np.clip(alpha, -config.negative_cap, config.negative_cap)
    total_other = float(alpha.sum())
    if total_other > 0 and 1.0 - total_other < config.min_champion_weight:
        alpha *= (1.0 - config.min_champion_weight) / total_other
    weights = np.zeros(x.shape[1], dtype=float)
    weights[others] = alpha
    weights[champion] = 1.0 - float(alpha.sum())
    return weights


def _features(x: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for j in range(x.shape[1]):
        z = x[:, j]
        values.extend([float(z.mean()), float(z.std()), float(np.median(z)), float(z.min()), float(z.max())])
    for j in range(x.shape[1]):
        for k in range(j + 1, x.shape[1]):
            values.extend([float(np.mean(np.abs(x[:, j] - x[:, k]))), float(np.mean(x[:, j] - x[:, k]))])
    return np.asarray(values, dtype=float)


def _fit_meta(
    mats: list[tuple[np.ndarray, np.ndarray]], q: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.vstack([_features(x) for x, _ in mats])
    losses = np.asarray([
        [compute_daily_loss(y, x[:, j], "composite") for j in range(x.shape[1])]
        for x, y in mats
    ])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (features - mean) / scale
    root_q = np.sqrt(q)
    design = z * root_q[:, None]
    response = losses * root_q[:, None]
    coef = np.linalg.solve(
        design.T @ design + alpha * np.eye(design.shape[1]),
        design.T @ response,
    )
    intercept = losses.mean(axis=0) - (mean / scale) @ coef
    return mean, scale, coef, intercept


def _predict_meta(
    mats: list[tuple[np.ndarray, np.ndarray] | np.ndarray],
    fitted: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    mean, scale, coef, intercept = fitted
    features = []
    for item in mats:
        x = item[0] if isinstance(item, tuple) else item
        features.append(_features(x))
    z = (np.vstack(features) - mean) / scale
    return z @ coef + intercept


def _dynamic_blend_loss(
    mats: list[tuple[np.ndarray, np.ndarray]], estimates: np.ndarray, champion: int, rho: float
) -> float:
    values = []
    for (x, y), estimate in zip(mats, estimates):
        selected = int(np.argmin(estimate))
        fused = (1.0 - rho) * x[:, champion] + rho * x[:, selected]
        values.append(compute_daily_loss(y, fused, "composite"))
    return float(np.mean(values))


def _weights_df(weights: dict[tuple[str, str], dict[str, float]]) -> pd.DataFrame:
    rows = []
    for (task, period), values in weights.items():
        for model, weight in values.items():
            rows.append({
                "task": task,
                "period": period,
                "model_name": model,
                "weight": round(float(weight), 8),
            })
    return pd.DataFrame(rows)


def fit_champion_short(
    training_table: pd.DataFrame,
    target_prediction_table: pd.DataFrame,
    *,
    task: str,
    expected_models: list[str],
    resolution: Any,
    config: ChampionShortConfig | None = None,
) -> tuple[dict[tuple[str, str], dict[str, float]], pd.DataFrame, pd.DataFrame]:
    """Fit target-day weights and return weights, candidate report, trace."""
    cfg = config or ChampionShortConfig()
    periods = list(resolution.period_names)
    slot_column = resolution.slot_column
    history = _matrix_by_day(training_table, expected_models, periods, slot_column)
    target = _target_matrices(target_prediction_table, expected_models, periods, slot_column)
    complete_days = sorted({day for day, values in history.items() if len(values) == len(periods)})
    if len(complete_days) != cfg.window_days:
        raise ValueError(
            f"champion_short requires {cfg.window_days} complete days, got {len(complete_days)}"
        )
    if len(target) != len(periods):
        details = _target_completeness_diagnostics(
            target_prediction_table,
            expected_models,
            periods,
            slot_column,
            resolution.slots_per_period,
        )
        suffix = f": {'; '.join(details)}" if details else ""
        raise ValueError(f"target prediction table is incomplete for champion_short{suffix}")

    train_days = complete_days[: cfg.window_days - cfg.validation_days]
    validation_days = complete_days[cfg.window_days - cfg.validation_days :]
    all_days = complete_days
    q_train = _day_weights(len(train_days), cfg.half_life_days)
    q_all = _day_weights(len(all_days), cfg.half_life_days)
    weights: dict[tuple[str, str], dict[str, float]] = {}
    report_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    for period in periods:
        train = [history[day][period] for day in train_days]
        validation = [history[day][period] for day in validation_days]
        all_mats = [history[day][period] for day in all_days]
        champion = _champion(train, q_train)
        one_hot = np.zeros(len(expected_models), dtype=float)
        one_hot[champion] = 1.0
        champion_val = _weighted_loss(validation, one_hot)
        selected = "champion"
        selected_weights = one_hot
        val_selected = champion_val
        target_model = expected_models[champion]

        if task == "realtime" and period in {"1_32", "33_64"}:
            meta_train = _fit_meta(train, q_train, cfg.meta_alpha)
            meta_val_pred = _predict_meta(validation, meta_train)
            val_meta = _dynamic_blend_loss(validation, meta_val_pred, champion, cfg.meta_blend_rho)
            if val_meta <= champion_val * (1.0 + cfg.meta_gate_tolerance):
                meta_all = _fit_meta(all_mats, q_all, cfg.meta_alpha)
                target_estimate = _predict_meta([target[period]], meta_all)[0]
                target_idx = int(np.argmin(target_estimate))
                selected_weights = one_hot.copy()
                selected_weights[target_idx] += cfg.meta_blend_rho
                selected_weights[champion] -= cfg.meta_blend_rho
                selected = "meta_blend_gate"
                val_selected = val_meta
                target_model = expected_models[target_idx]
        elif task == "realtime" and period == "65_96":
            signed_train = _fit_signed(train, q_train, champion, cfg)
            val_signed = _weighted_loss(validation, signed_train)
            if val_signed <= champion_val * (1.0 + cfg.gate_tolerance):
                selected_weights = _fit_signed(all_mats, q_all, champion, cfg)
                selected = "signed_gate"
                val_selected = val_signed
        else:
            nnls_train = _fit_nnls(train, q_train)
            signed_train = _fit_signed(train, q_train, champion, cfg)
            candidates = {
                "champion": (champion_val, one_hot),
                "weighted_nnls": (_weighted_loss(validation, nnls_train), nnls_train),
                "signed_anchor": (_weighted_loss(validation, signed_train), signed_train),
            }
            eligible = {
                name: item
                for name, item in candidates.items()
                if item[0] <= champion_val * (1.0 + cfg.gate_tolerance)
            }
            selected, (val_selected, _) = min(eligible.items(), key=lambda item: item[1][0])
            if selected == "weighted_nnls":
                selected_weights = _fit_nnls(all_mats, q_all)
            elif selected == "signed_anchor":
                selected_weights = _fit_signed(all_mats, q_all, champion, cfg)

        weights[(task, period)] = {
            model: float(value) for model, value in zip(expected_models, selected_weights)
        }
        report_rows.append({
            "task": task,
            "period": period,
            "champion_model": expected_models[champion],
            "selected_strategy": selected,
            "validation_champion_composite": champion_val,
            "validation_selected_composite": val_selected,
            "gate_passed": selected != "champion",
            "target_meta_model": target_model,
            "window_days": cfg.window_days,
            "train_days": len(train_days),
            "validation_days": len(validation_days),
        })
        for model, value in zip(expected_models, selected_weights):
            trace_rows.append({
                "task": task,
                "period": period,
                "model_name": model,
                "weight": float(value),
                "selected_strategy": selected,
            })

    return weights, pd.DataFrame(report_rows), pd.DataFrame(trace_rows)


def weights_to_dataframe(weights: dict[tuple[str, str], dict[str, float]]) -> pd.DataFrame:
    return _weights_df(weights)


def candidate_metrics_from_report(report: pd.DataFrame) -> pd.DataFrame:
    """Return a stable report schema for ledger_weight output."""
    return report.copy()
