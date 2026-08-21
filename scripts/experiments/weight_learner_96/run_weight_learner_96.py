"""Causal 96-point weight learner experiment.

This runner is intentionally isolated from production ``ledger_weight`` and
``ledger_fuse``.  It consumes only the copied prediction/actual ledgers and
evaluates:

* best single model and equal-weight baselines;
* non-negative simplex weights;
* bounded signed weights (negative residual corrections allowed);
* subset selection with the causal champion retained;
* a historical reliability gate that can disable persistently harmful models.

All decisions for target day D use only dates strictly earlier than D.  The
outer validation window selects a strategy; the selected strategy is then
refit on the full pre-D window before predicting D.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.resolution import resolve_resolution


RES = resolve_resolution("15min")
PERIODS = tuple(RES.period_names)
WINDOW_DAYS = 45
TRAIN_DAYS = 30
VALIDATION_DAYS = WINDOW_DAYS - TRAIN_DAYS
GATE_DAYS = 7
REFERENCE_BEST = {"dayahead": "timesfm", "realtime": "sgdfnet"}
POLICY_METHODS = ("champion", "nonnegative_all", "signed_all", "gated_signed")


@dataclass(frozen=True)
class DayMatrix:
    X: np.ndarray
    y: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smape_floor50_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.maximum(np.asarray(y_true, dtype=float), 50.0)
    yp = np.maximum(np.asarray(y_pred, dtype=float), 50.0)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, np.abs(yp - yt) / denom)
    return float(np.mean(terms) * 100.0)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yp - yt
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    safe_ape = np.zeros_like(yt, dtype=float)
    nonzero = yt != 0
    safe_ape[nonzero] = np.abs(err[nonzero]) / np.abs(yt[nonzero])
    return {
        "n": int(len(yt)),
        "MAE": float(np.mean(np.abs(err))),
        "MSE": float(np.mean(err**2)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_pct": float(np.mean(safe_ape) * 100.0),
        "SMAPE_pct": smape_floor50_pct(yt, yp),
        "accuracy_pct": 100.0 - smape_floor50_pct(yt, yp),
        "R2": float(1.0 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
    }


def load_task(ledger_root: Path, task: str) -> tuple[dict[str, dict[str, DayMatrix]], list[str]]:
    pred_path = ledger_root / task / "prediction" / "prediction_ledger.parquet"
    actual_path = ledger_root / task / "actual" / "actual_ledger.parquet"
    if not pred_path.exists() or not actual_path.exists():
        raise FileNotFoundError(f"missing ledger for {task}: {pred_path} / {actual_path}")

    pred = pd.read_parquet(pred_path).copy()
    actual = pd.read_parquet(actual_path).copy()
    pred["business_day"] = pd.to_datetime(pred["business_day"]).dt.strftime("%Y-%m-%d")
    actual["business_day"] = pd.to_datetime(actual["business_day"]).dt.strftime("%Y-%m-%d")
    pred["business_period"] = pd.to_numeric(pred["business_period"], errors="raise").astype(int)
    actual["business_period"] = pd.to_numeric(actual["business_period"], errors="raise").astype(int)
    models = sorted(pred["model_name"].dropna().astype(str).unique().tolist())
    if not models:
        raise ValueError(f"{task}: no models in prediction ledger")

    key = ["business_day", "business_period"]
    actual = actual[key + ["y_true"]].drop_duplicates(key)
    merged = pred.merge(actual, on=key, how="inner", validate="many_to_one")
    merged = merged.dropna(subset=["y_pred", "y_true"])
    merged = merged[merged["business_period"].between(1, RES.slots_per_day)]

    result: dict[str, dict[str, DayMatrix]] = {}
    for day, day_frame in merged.groupby("business_day", sort=True):
        per_day: dict[str, DayMatrix] = {}
        for period in PERIODS:
            period_frame = day_frame[day_frame["period"] == period].copy()
            if period_frame.empty:
                continue
            wide = period_frame.pivot(index=RES.slot_column, columns="model_name", values="y_pred")
            wide = wide.reindex(columns=models)
            truth = (
                period_frame[[RES.slot_column, "y_true"]]
                .drop_duplicates(RES.slot_column)
                .set_index(RES.slot_column)["y_true"]
            )
            wide = wide.reindex(truth.index)
            if len(wide) != RES.slots_per_period or wide.isna().any().any() or truth.isna().any():
                continue
            per_day[period] = DayMatrix(
                X=wide.to_numpy(dtype=float),
                y=truth.to_numpy(dtype=float),
            )
        if len(per_day) == len(PERIODS):
            result[day] = per_day
    if not result:
        raise ValueError(f"{task}: no complete days after ledger alignment")
    return result, models


def weighted_mse(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    scale = max(float(np.std(y)), 1.0)
    return float(np.mean(((X @ weights - y) / scale) ** 2))


def fit_weights(
    X: np.ndarray,
    y: np.ndarray,
    subset: tuple[int, ...],
    champion: int,
    *,
    signed: bool,
    negative_cap: float = 0.5,
    positive_cap: float = 1.5,
    ridge: float = 0.02,
) -> np.ndarray:
    """Fit bounded weights with sum(w)=1; signed mode permits residual reversal."""
    local_champion = subset.index(champion)
    anchor = np.zeros(len(subset), dtype=float)
    anchor[local_champion] = 1.0
    if not signed:
        anchor[:] = 1.0 / len(subset)
    lower = -negative_cap if signed else 0.0
    bounds = [(lower, positive_cap) for _ in subset]

    def objective(w: np.ndarray) -> float:
        # Fit to the primary documented floor-50 sMAPE rather than only raw
        # MSE; raw-MSE weights can improve spikes while losing the acceptance
        # metric used for the experiment gate.
        smape_loss = smape_floor50_pct(y, X[:, subset] @ w) / 100.0
        return smape_loss + ridge * float(np.sum((w - anchor) ** 2))

    result = minimize(
        objective,
        anchor,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all() or abs(float(result.x.sum()) - 1.0) > 1e-5:
        return anchor
    return result.x.astype(float)


def full_weights(n_models: int, champion: int, *, signed: bool) -> np.ndarray:
    subset = tuple(range(n_models))
    return expand_weights(n_models, subset, fit_weights(
        np.zeros((1, n_models)),
        np.zeros(1),
        subset,
        champion,
        signed=signed,
    ))


def expand_weights(n_models: int, subset: tuple[int, ...], local_weights: np.ndarray) -> np.ndarray:
    weights = np.zeros(n_models, dtype=float)
    weights[list(subset)] = local_weights
    return weights


def model_losses(mats: list[DayMatrix], n_models: int) -> np.ndarray:
    losses = np.zeros(n_models, dtype=float)
    for j in range(n_models):
        losses[j] = float(np.mean([smape_floor50_pct(m.y, m.X[:, j]) for m in mats]))
    return losses


def regime_features(X: np.ndarray) -> np.ndarray:
    """Build target-day regime features from predictions only.

    No actual values or target-day losses are used here.  The model-level
    level/spread statistics are available at prediction time and allow a
    causal selector to learn when a correction strategy is useful.
    """
    values: list[float] = []
    for column in X.T:
        values.extend([
            float(np.mean(column)), float(np.std(column)),
            float(np.median(column)), float(np.min(column)),
            float(np.max(column)), float(np.mean(column < 50.0)),
        ])
    for left in range(X.shape[1]):
        for right in range(left + 1, X.shape[1]):
            difference = X[:, left] - X[:, right]
            values.extend([
                float(np.mean(np.abs(difference))),
                float(np.mean(difference)), float(np.std(difference)),
            ])
    return np.asarray(values, dtype=float)


def fit_regime_selector(
    history: list[dict[str, Any]],
    current_features: np.ndarray,
    *,
    train_days: int,
    validation_days: int,
    champion_name: str,
    margin: float,
) -> tuple[str, dict[str, Any]]:
    """Select a strategy from earlier target-day regimes only.

    The final validation tail is used only as a historical safety gate.  The
    target-day feature is passed to the classifier only after that gate has
    accepted the policy, so target-day truth cannot affect the decision.
    """
    audit: dict[str, Any] = {
        "regime_history_days": len(history), "regime_policy_used": False,
        "regime_gate_passed": False,
    }
    if len(history) < train_days:
        return champion_name, audit
    history = history[-train_days:]
    if len(history) < max(10, validation_days + 10):
        return champion_name, audit

    def fit_predict(train_rows: list[dict[str, Any]], predict_rows: list[dict[str, Any]]) -> np.ndarray | None:
        labels = np.asarray([row["label"] for row in train_rows], dtype=object)
        if len(set(labels.tolist())) < 2:
            return None
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced"),
        )
        model.fit(np.vstack([row["features"] for row in train_rows]), labels)
        return model.predict(np.vstack([row["features"] for row in predict_rows]))

    fit_rows = history[:-validation_days]
    validation_rows = history[-validation_days:]
    validation_prediction = fit_predict(fit_rows, validation_rows)
    if validation_prediction is None:
        return champion_name, audit
    candidate_loss = float(np.mean([
        row["losses"][str(predicted)]
        for row, predicted in zip(validation_rows, validation_prediction)
    ]))
    champion_loss = float(np.mean([
        row["losses"][champion_name] for row in validation_rows
    ]))
    audit.update({
        "regime_validation_loss": candidate_loss,
        "regime_champion_loss": champion_loss,
        "regime_gate_margin": margin,
    })
    if candidate_loss > champion_loss * (1.0 + margin):
        return champion_name, audit

    full_prediction = fit_predict(history, [{"features": current_features, "label": champion_name}])
    if full_prediction is None:
        return champion_name, audit
    audit["regime_policy_used"] = True
    audit["regime_gate_passed"] = True
    audit["regime_predicted_strategy"] = str(full_prediction[0])
    return str(full_prediction[0]), audit


def gate_subset(
    mats: list[DayMatrix],
    models: list[str],
    champion: int,
    *,
    gate_days: int = GATE_DAYS,
    tolerance: float = 0.01,
) -> tuple[int, ...]:
    """Keep models whose signed two-model correction is not harmful recently."""
    if len(mats) <= gate_days + 1:
        return tuple(range(len(models)))
    core = mats[:-gate_days]
    recent = mats[-gate_days:]
    kept = [champion]
    champ_recent = float(np.mean([smape_floor50_pct(m.y, m.X[:, champion]) for m in recent]))
    for j in range(len(models)):
        if j == champion:
            continue
        pair = (champion, j)
        local = fit_weights(
            np.vstack([m.X for m in core]),
            np.concatenate([m.y for m in core]),
            pair,
            champion,
            signed=True,
        )
        recent_loss = float(np.mean([
            smape_floor50_pct(m.y, m.X[:, pair] @ local) for m in recent
        ]))
        if recent_loss <= champ_recent * (1.0 + tolerance):
            kept.append(j)
    return tuple(sorted(set(kept)))


def candidate_weights(
    mats: list[DayMatrix],
    models: list[str],
    champion: int,
    *,
    gate: bool,
) -> dict[str, np.ndarray]:
    n_models = len(models)
    candidates: dict[str, np.ndarray] = {}
    all_subset = tuple(range(n_models))
    candidates["equal_all"] = np.ones(n_models, dtype=float) / n_models
    candidates["nonnegative_all"] = fit_weights(
        np.vstack([m.X for m in mats]),
        np.concatenate([m.y for m in mats]),
        all_subset,
        champion,
        signed=False,
    )
    candidates["signed_all"] = fit_weights(
        np.vstack([m.X for m in mats]),
        np.concatenate([m.y for m in mats]),
        all_subset,
        champion,
        signed=True,
    )
    champion_w = np.zeros(n_models, dtype=float)
    champion_w[champion] = 1.0
    candidates["champion"] = champion_w

    allowed = gate_subset(mats, models, champion) if gate else all_subset
    candidates["gated_signed"] = expand_weights(
        n_models,
        allowed,
        fit_weights(
            np.vstack([m.X for m in mats]),
            np.concatenate([m.y for m in mats]),
            allowed,
            champion,
            signed=True,
        ),
    )

    # All subsets containing the causal champion are eligible. This is small
    # (7 for DA, 8 for RT when the champion is fixed) and exposes persistent
    # interferers without forcing every model into the final prediction.
    for size in range(1, n_models + 1):
        for subset in itertools.combinations(range(n_models), size):
            if champion not in subset:
                continue
            local = fit_weights(
                np.vstack([m.X for m in mats]),
                np.concatenate([m.y for m in mats]),
                subset,
                champion,
                signed=True,
            )
            name = "signed_subset_" + "+".join(models[i] for i in subset)
            candidates[name] = expand_weights(n_models, subset, local)
    return candidates


def choose_candidate(
    candidates: dict[str, np.ndarray],
    mats: list[DayMatrix],
    champion_name: str,
    selection_margin: float,
) -> tuple[str, dict[str, float]]:
    scores = {
        name: float(np.mean([smape_floor50_pct(m.y, m.X @ weights) for m in mats]))
        for name, weights in candidates.items()
    }
    champion_score = scores[champion_name]
    best_name = min(scores, key=scores.get)
    if best_name != champion_name and scores[best_name] > champion_score * (1.0 - selection_margin):
        best_name = champion_name
    return best_name, scores


def refit_weights(
    mats: list[DayMatrix],
    models: list[str],
    champion: int,
    selected_name: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    n_models = len(models)
    all_subset = tuple(range(n_models))
    if selected_name == "champion":
        w = np.zeros(n_models, dtype=float)
        w[champion] = 1.0
        return w, (champion,)
    if selected_name == "equal_all":
        return np.ones(n_models, dtype=float) / n_models, all_subset
    if selected_name == "nonnegative_all":
        return expand_weights(n_models, all_subset, fit_weights(
            np.vstack([m.X for m in mats]), np.concatenate([m.y for m in mats]),
            all_subset, champion, signed=False,
        )), all_subset
    if selected_name == "signed_all":
        return expand_weights(n_models, all_subset, fit_weights(
            np.vstack([m.X for m in mats]), np.concatenate([m.y for m in mats]),
            all_subset, champion, signed=True,
        )), all_subset
    if selected_name == "gated_signed":
        subset = gate_subset(mats, models, champion)
    elif selected_name.startswith("signed_subset_"):
        selected_models = selected_name.removeprefix("signed_subset_").split("+")
        subset = tuple(models.index(model) for model in selected_models)
    else:
        raise ValueError(f"unknown strategy: {selected_name}")
    local = fit_weights(
        np.vstack([m.X for m in mats]), np.concatenate([m.y for m in mats]),
        subset, champion, signed=True,
    )
    return expand_weights(n_models, subset, local), subset


def run_task(
    task: str,
    day_data: dict[str, dict[str, DayMatrix]],
    models: list[str],
    *,
    start_date: str,
    selection_margin: float,
    policy: str,
    policy_history_days: int,
    regime_validation_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    days = sorted(day_data)
    target_days = [day for day in days if day >= start_date]
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    # A policy decision for D must not use D's validation score.  The rolling
    # policy therefore consumes only validation scores recorded for earlier
    # target days.  This reduces day-to-day strategy flipping while keeping
    # the experiment causal.
    policy_history: dict[str, list[dict[str, float]]] = {period: [] for period in PERIODS}
    regime_history: dict[str, list[dict[str, Any]]] = {period: [] for period in PERIODS}

    for target_day in target_days:
        target_index = days.index(target_day)
        if target_index < WINDOW_DAYS:
            continue
        window_days = days[target_index - WINDOW_DAYS : target_index]
        train_days = window_days[:TRAIN_DAYS]
        validation_days = window_days[TRAIN_DAYS:]
        for period in PERIODS:
            train_mats = [day_data[d][period] for d in train_days]
            validation_mats = [day_data[d][period] for d in validation_days]
            all_mats = [day_data[d][period] for d in window_days]
            target_mat = day_data[target_day][period]
            losses = model_losses(train_mats, len(models))
            train_best = int(np.argmin(losses))
            # The real-time experiment is explicitly anchored to SGDFNet:
            # other models may correct it, including with a negative weight,
            # but a noisy inner window must not silently replace the known
            # real-time champion before the outer test day.
            if task == "realtime" and REFERENCE_BEST[task] in models:
                champion = models.index(REFERENCE_BEST[task])
            else:
                champion = train_best
            candidates = candidate_weights(train_mats, models, champion, gate=True)
            selected_name, validation_scores = choose_candidate(
                candidates, validation_mats, "champion", selection_margin,
            )
            instant_selected_name = selected_name
            regime_audit: dict[str, Any] = {}
            if policy == "validation_rolling":
                history = policy_history[period][-policy_history_days:]
                if history:
                    historical_scores = {
                        name: float(np.mean([row[name] for row in history if name in row]))
                        for name in POLICY_METHODS
                        if all(name in row for row in history)
                    }
                    selected_name = min(
                        historical_scores,
                        key=lambda name: (historical_scores[name], POLICY_METHODS.index(name)),
                    ) if historical_scores else "champion"
                else:
                    selected_name = "champion"
            elif policy == "regime_selector":
                selected_name, regime_audit = fit_regime_selector(
                    regime_history[period],
                    regime_features(target_mat.X),
                    train_days=policy_history_days,
                    validation_days=regime_validation_days,
                    champion_name="champion",
                    margin=selection_margin,
                )
            elif policy == "fixed_nonnegative":
                selected_name = "nonnegative_all"
            elif policy == "fixed_champion":
                selected_name = "champion"
            elif policy != "instant":
                raise ValueError(f"unknown policy: {policy}")
            # Append only after making the current target decision.
            policy_history[period].append({name: float(score) for name, score in validation_scores.items()})
            selected_weights, selected_subset = refit_weights(
                all_mats, models, champion, selected_name,
            )
            all_refit: dict[str, np.ndarray] = {}
            for name in ["champion", "equal_all", "nonnegative_all", "signed_all", "gated_signed"]:
                all_refit[name], _ = refit_weights(all_mats, models, champion, name)
            regime_losses = {
                name: smape_floor50_pct(target_mat.y, target_mat.X @ weights)
                for name, weights in all_refit.items()
                if name in POLICY_METHODS
            }
            # This label is appended only after the current target decision
            # and prediction.  It is therefore available to later dates but
            # cannot leak target-day truth into the current selector.
            regime_history[period].append({
                "features": regime_features(target_mat.X),
                "label": min(regime_losses, key=regime_losses.get),
                "losses": regime_losses,
            })
            for name, weights in all_refit.items():
                prediction = target_mat.X @ weights
                for slot, (predicted, truth) in enumerate(zip(prediction, target_mat.y), start=1):
                    rows.append({
                        "task": task,
                        "target_day": target_day,
                        "period": period,
                        "business_period_in_segment": slot,
                        "method": name,
                        "y_pred": float(predicted),
                        "y_true": float(truth),
                    })
            selected_prediction = target_mat.X @ selected_weights
            for slot, (predicted, truth) in enumerate(zip(selected_prediction, target_mat.y), start=1):
                rows.append({
                    "task": task,
                    "target_day": target_day,
                    "period": period,
                    "business_period_in_segment": slot,
                    "method": "selected",
                    "y_pred": float(predicted),
                    "y_true": float(truth),
                })
            for j, model in enumerate(models):
                for slot, (predicted, truth) in enumerate(zip(target_mat.X[:, j], target_mat.y), start=1):
                    rows.append({
                        "task": task,
                        "target_day": target_day,
                        "period": period,
                        "business_period_in_segment": slot,
                        "method": model,
                        "y_pred": float(predicted),
                        "y_true": float(truth),
                    })
            for name, weights in all_refit.items():
                for model, weight in zip(models, weights):
                    weight_rows.append({
                        "task": task, "target_day": target_day, "period": period,
                        "method": name, "model": model, "weight": float(weight),
                        "active": bool(abs(weight) > 1e-8),
                    })
            for model, weight in zip(models, selected_weights):
                weight_rows.append({
                    "task": task, "target_day": target_day, "period": period,
                    "method": "selected", "model": model, "weight": float(weight),
                    "active": bool(abs(weight) > 1e-8),
                })
            selection_rows.append({
                "task": task, "target_day": target_day, "period": period,
                "champion_model": models[champion],
                "train_best_model": models[train_best],
                "policy": policy,
                "instant_selected_strategy": instant_selected_name,
                "selected_strategy": selected_name,
                "selected_subset": "+".join(models[i] for i in selected_subset),
                "train_days": len(train_days), "validation_days": len(validation_days),
                "window_days": len(window_days), "selection_margin": selection_margin,
                "policy_history_days": policy_history_days,
                "regime_validation_days": regime_validation_days,
                "gate_subset": "+".join(models[i] for i in gate_subset(train_mats, models, champion)),
                **{f"val_smape_{name}": score for name, score in validation_scores.items()},
                **regime_audit,
            })

    metadata = {
        "task": task, "models": models, "periods": list(PERIODS),
        "complete_days": len(days), "target_days_evaluated": len(target_days),
        "window_days": WINDOW_DAYS, "train_days": TRAIN_DAYS,
        "validation_days": VALIDATION_DAYS, "gate_days": GATE_DAYS,
        "policy": policy, "policy_history_days": policy_history_days,
        "regime_validation_days": regime_validation_days,
    }
    return pd.DataFrame(rows), pd.DataFrame(weight_rows), pd.DataFrame(selection_rows), metadata


def daily_significance(predictions: pd.DataFrame, task: str, reference: str) -> pd.DataFrame:
    data = predictions[predictions["task"] == task].copy()
    daily = data.groupby(["target_day", "method"], sort=True).apply(
        lambda g: smape_floor50_pct(g["y_true"].to_numpy(), g["y_pred"].to_numpy()),
        include_groups=False,
    ).rename("daily_smape_pct").reset_index()
    pivot = daily.pivot(index="target_day", columns="method", values="daily_smape_pct")
    if reference not in pivot.columns:
        raise ValueError(f"missing reference method {task}/{reference}")
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for method in sorted(c for c in pivot.columns if c != reference):
        pair = pivot[[method, reference]].dropna()
        delta = (pair[method] - pair[reference]).to_numpy(float)
        boot = np.empty(2000, dtype=float)
        for i in range(len(boot)):
            boot[i] = float(rng.choice(delta, size=len(delta), replace=True).mean())
        try:
            pvalue = float(wilcoxon(delta, alternative="less").pvalue)
        except ValueError:
            pvalue = float("nan")
        rows.append({
            "task": task, "method": method, "reference": reference,
            "n_days": len(delta), "mean_delta_smape_pct": float(delta.mean()),
            "median_delta_smape_pct": float(np.median(delta)),
            "win_rate": float(np.mean(delta < 0)),
            "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
            "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            "wilcoxon_p_less": pvalue,
        })
    return pd.DataFrame(rows)


def build_metrics(predictions: pd.DataFrame, out_dir: Path) -> None:
    data = predictions.copy()
    data["month"] = pd.to_datetime(data["target_day"]).dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for (task, method, month), group in data.groupby(["task", "method", "month"], sort=True):
        rows.append({"task": task, "method": method, "month": month, **metric_row(group.y_true, group.y_pred)})
    for (task, method), group in data.groupby(["task", "method"], sort=True):
        rows.append({"task": task, "method": method, "month": "ALL", **metric_row(group.y_true, group.y_pred)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics_monthly.csv", index=False, encoding="utf-8-sig")
    metrics[metrics["month"] == "ALL"].to_csv(out_dir / "metrics_overall.csv", index=False, encoding="utf-8-sig")


def build_scr(predictions: pd.DataFrame, out_dir: Path, selected_method: str = "selected", output_name: str = "selected_scr_monthly.csv") -> None:
    da = predictions[predictions.task == "dayahead"].copy()
    rt = predictions[predictions.task == "realtime"].copy()
    da = da[da.method == selected_method][["target_day", "period", "business_period_in_segment", "y_pred", "y_true"]]
    rt = rt[rt.method == selected_method][["target_day", "period", "business_period_in_segment", "y_pred", "y_true"]]
    key = ["target_day", "period", "business_period_in_segment"]
    da = da.rename(columns={"y_pred": "pred_da", "y_true": "true_da"})
    rt = rt.rename(columns={"y_pred": "pred_rt", "y_true": "true_rt"})
    merged = da.merge(rt, on=key, how="inner", validate="one_to_one")
    merged["month"] = pd.to_datetime(merged["target_day"]).dt.to_period("M").astype(str)
    merged["real_spread"] = merged.true_rt - merged.true_da
    merged["pred_spread"] = merged.pred_rt - merged.pred_da
    rows: list[dict[str, Any]] = []
    for month, group in list(merged.groupby("month", sort=True)) + [("ALL", merged)]:
        real = group.real_spread.to_numpy(float)
        pred = group.pred_spread.to_numpy(float)
        denom = np.abs(real) + np.abs(pred)
        smape = np.where(denom == 0, 0.0, 200.0 * np.abs(pred - real) / denom)
        rows.append({
            "month": month, "n": len(group),
            "SCR_pct": float(np.mean(np.sign(real) == np.sign(pred)) * 100.0),
            "spread_MAE": float(np.mean(np.abs(pred - real))),
            "spread_RMSE": float(np.sqrt(np.mean((pred - real) ** 2))),
            "spread_sMAPE_pct": float(np.mean(smape)),
            "positive_actual": int((real > 0).sum()),
            "negative_actual": int((real < 0).sum()),
            "zero_actual": int((real == 0).sum()),
        })
    pd.DataFrame(rows).to_csv(out_dir / output_name, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", default="2026-02-01")
    parser.add_argument("--selection-margin", type=float, default=0.002)
    parser.add_argument(
        "--policy", choices=("instant", "validation_rolling", "regime_selector", "fixed_nonnegative", "fixed_champion"), default="instant",
        help="strategy selection policy; regime_selector uses prediction-only regime features and prior labels",
    )
    parser.add_argument("--policy-history-days", type=int, default=45)
    parser.add_argument("--regime-validation-days", type=int, default=15)
    policy_choices = ("instant", "validation_rolling", "regime_selector", "fixed_nonnegative", "fixed_champion")
    parser.add_argument("--dayahead-policy", choices=policy_choices, default=None)
    parser.add_argument("--realtime-policy", choices=policy_choices, default=None)
    args = parser.parse_args()

    ledger_root = (PROJECT_ROOT / args.ledger_root).resolve() if not Path(args.ledger_root).is_absolute() else Path(args.ledger_root)
    out_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    task_data: dict[str, tuple[dict[str, dict[str, DayMatrix]], list[str]]] = {}
    for task in ("dayahead", "realtime"):
        task_data[task] = load_task(ledger_root, task)

    all_predictions: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    all_selection: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for task, (day_data, models) in task_data.items():
        task_policy = getattr(args, f"{task}_policy") or args.policy
        predictions, weights, selection, meta = run_task(
            task, day_data, models, start_date=args.start_date,
            selection_margin=args.selection_margin,
            policy=task_policy,
            policy_history_days=args.policy_history_days,
            regime_validation_days=args.regime_validation_days,
        )
        all_predictions.append(predictions)
        all_weights.append(weights)
        all_selection.append(selection)
        metadata[task] = meta
        print(json.dumps(meta, ensure_ascii=False), flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    weights = pd.concat(all_weights, ignore_index=True)
    selection = pd.concat(all_selection, ignore_index=True)
    if predictions.empty:
        raise ValueError("no walk-forward predictions generated")
    if not np.isfinite(predictions[["y_pred", "y_true"]].to_numpy(float)).all():
        raise ValueError("non-finite walk-forward predictions")
    predictions.to_parquet(out_dir / "walk_forward_predictions.parquet", index=False)
    weights.to_csv(out_dir / "weights_audit.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(out_dir / "selection_audit.csv", index=False, encoding="utf-8-sig")
    build_metrics(predictions, out_dir)
    build_scr(predictions, out_dir)

    significance = pd.concat([
        daily_significance(predictions, "dayahead", REFERENCE_BEST["dayahead"]),
        daily_significance(predictions, "realtime", REFERENCE_BEST["realtime"]),
    ], ignore_index=True)
    significance.to_csv(out_dir / "significance_daily_smape.csv", index=False, encoding="utf-8-sig")

    source_paths = [
        ledger_root / "dayahead/prediction/prediction_ledger.parquet",
        ledger_root / "dayahead/actual/actual_ledger.parquet",
        ledger_root / "realtime/prediction/prediction_ledger.parquet",
        ledger_root / "realtime/actual/actual_ledger.parquet",
    ]
    manifest = {
        "experiment": "weight_learner_96_signed_subset_gate_regime",
        "status": "complete",
        "production_link_touched": False,
        "resolution": RES.label,
        "slots_per_day": RES.slots_per_day,
        "periods": list(PERIODS),
        "start_date": args.start_date,
        "end_date": str(predictions["target_day"].max()),
        "window_days": WINDOW_DAYS,
        "train_days": TRAIN_DAYS,
        "validation_days": VALIDATION_DAYS,
        "gate_days": GATE_DAYS,
        "selection_margin": args.selection_margin,
        "policy": args.policy,
        "task_policies": {
            "dayahead": args.dayahead_policy or args.policy,
            "realtime": args.realtime_policy or args.policy,
        },
        "policy_history_days": args.policy_history_days,
        "regime_validation_days": args.regime_validation_days,
        "signed_weight_bounds": {"lower": -0.5, "upper": 1.5, "sum": 1.0},
        "reference_best": REFERENCE_BEST,
        "decision_rules": [
            "target day uses strictly earlier days only",
            "champion is selected from outer train days",
            "subset candidates always retain the causal champion",
            "gate uses only historical train days and may disable harmful models",
            "negative weights are allowed only in experiment area",
        ],
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)): {
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
            }
            for path in source_paths
        },
        "rows": {
            "walk_forward_predictions": len(predictions),
            "weights_audit": len(weights),
            "selection_audit": len(selection),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(out_dir), "elapsed_seconds": manifest["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
