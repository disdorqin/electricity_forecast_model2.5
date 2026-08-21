"""Prequential continuous-spread fusion for the segmented hourly experiment.

The learner selects at most three model families globally, then fits one
non-negative simplex weight vector per business-hour segment.  It optimizes
only the sign of the continuous weighted spread; positive/negative hit rates
are retained as diagnostics and never receive separate class weights.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.spread_direction_24.spread_metrics import smape_percent


SEGMENTS = (("1_8", 1, 8), ("9_16", 9, 16), ("17_24", 17, 24))
WEIGHT_STEP = 0.05


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _simplex(n: int) -> list[np.ndarray]:
    if n == 1:
        return [np.array([1.0])]
    units = round(1.0 / WEIGHT_STEP)
    values: list[np.ndarray] = []

    def compositions(total: int, parts: int, prefix: list[int]) -> None:
        if parts == 1:
            values.append(np.asarray(prefix + [total], dtype=float) / units)
            return
        for value in range(total + 1):
            compositions(total - value, parts - 1, prefix + [value])

    compositions(units, n, [])
    return values


def _direction_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    true_sign = np.sign(true)
    pred_sign = np.sign(pred)
    eligible = true_sign != 0
    correct = eligible & (true_sign == pred_sign)
    pos = true_sign > 0
    neg = true_sign < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
        "n_direction_eligible": int(eligible.sum()),
        "n_zero_actual": int((true_sign == 0).sum()),
    }


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    required = {"target_day", "hour_business", "period", "model_name", "y_true_spread", "y_pred_spread"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fusion input missing columns: {missing}")
    frame["target_day"] = frame["target_day"].astype(str)
    frame["hour_business"] = pd.to_numeric(frame["hour_business"], errors="raise").astype(int)
    frame["period"] = frame["period"].astype(str)
    frame["y_true_spread"] = pd.to_numeric(frame["y_true_spread"], errors="coerce")
    frame["y_pred_spread"] = pd.to_numeric(frame["y_pred_spread"], errors="coerce")
    if frame[["y_true_spread", "y_pred_spread"]].isna().any().any():
        raise ValueError("fusion input contains NaN labels or predictions")
    return frame


def _wide(frame: pd.DataFrame, days: list[str], models: tuple[str, ...], period: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    start, end = next((a, b) for name, a, b in SEGMENTS if name == period)
    sub = frame[
        frame["target_day"].isin(days)
        & frame["period"].eq(period)
        & frame["model_name"].isin(models)
    ].copy()
    expected = len(days) * (end - start + 1) * len(models)
    if len(sub) != expected:
        raise ValueError(f"{period}: rows={len(sub)}, expected={expected}; incomplete model/day ledger")
    index = pd.MultiIndex.from_product([days, range(start, end + 1)], names=["target_day", "hour_business"])
    true = (
        sub.drop_duplicates(["target_day", "hour_business"])
        .set_index(["target_day", "hour_business"])["y_true_spread"]
        .reindex(index)
        .to_numpy(float)
    )
    if not np.isfinite(true).all():
        raise ValueError(f"{period}: missing true spread values")
    matrices = []
    for model in models:
        pred = (
            sub[sub["model_name"].eq(model)]
            .set_index(["target_day", "hour_business"])["y_pred_spread"]
            .reindex(index)
            .to_numpy(float)
        )
        if not np.isfinite(pred).all():
            raise ValueError(f"{period}/{model}: missing predictions")
        matrices.append(pred)
    return true, np.column_stack(matrices), pd.DataFrame(
        {"target_day": index.get_level_values(0), "hour_business": index.get_level_values(1), "period": period}
    )


def _score_candidate(true: np.ndarray, pred: np.ndarray, days: list[str], slots: int) -> dict[str, float]:
    metrics = _direction_metrics(true, pred)
    day_scores = []
    for i in range(len(days)):
        day_scores.append(_direction_metrics(true[i * slots : (i + 1) * slots], pred[i * slots : (i + 1) * slots])["direction_accuracy"])
    metrics["daily_accuracy_mean"] = float(np.nanmean(day_scores))
    metrics["daily_accuracy_std"] = float(np.nanstd(day_scores))
    metrics["mae"] = float(np.mean(np.abs(pred - true)))
    metrics["spread_smape_pct"] = smape_percent(true, pred)
    return metrics


def _fit_period(frame: pd.DataFrame, days: list[str], models: tuple[str, ...], period: str) -> tuple[dict, list[dict]]:
    true, matrix, meta = _wide(frame, days, models, period)
    slots = matrix.shape[0] // len(days)
    best = None
    trace = []
    for weights in _simplex(len(models)):
        pred = matrix @ weights
        metrics = _score_candidate(true, pred, days, slots)
        row = {"period": period, "models": ",".join(models), "weights": ",".join(f"{x:.2f}" for x in weights), **metrics}
        trace.append(row)
        key = (
            metrics["direction_accuracy"],
            metrics["balanced_direction_accuracy"],
            -metrics["daily_accuracy_std"],
            -metrics["mae"],
        )
        if best is None or key > best[0]:
            best = (key, weights.copy(), metrics)
    assert best is not None
    return {
        "period": period,
        "models": list(models),
        "weights": {m: float(w) for m, w in zip(models, best[1])},
        "metrics": best[2],
    }, trace


def _fit_global(frame: pd.DataFrame, days: list[str], models: tuple[str, ...]) -> dict[str, float]:
    true_parts = []
    matrix_parts = []
    for period, _, _ in SEGMENTS:
        true, matrix, _ = _wide(frame, days, models, period)
        true_parts.append(true)
        matrix_parts.append(matrix)
    true_all = np.concatenate(true_parts)
    matrix_all = np.vstack(matrix_parts)
    best = None
    for weights in _simplex(len(models)):
        metrics = _direction_metrics(true_all, matrix_all @ weights)
        key = (metrics["direction_accuracy"], metrics["balanced_direction_accuracy"])
        if best is None or key > best[0]:
            best = (key, weights.copy())
    assert best is not None
    return {m: float(w) for m, w in zip(models, best[1])}


def _apply(frame: pd.DataFrame, days: list[str], selected: tuple[str, ...], period_weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for period, _, _ in SEGMENTS:
        true, matrix, meta = _wide(frame, days, selected, period)
        weights = np.array([period_weights[period].get(m, 0.0) for m in selected], dtype=float)
        pred = matrix @ weights
        out = meta.copy()
        out["y_true_spread"] = true
        out["y_pred_spread"] = pred
        out["fusion_model"] = "weighted_continuous_spread"
        out["selected_models"] = ",".join(selected)
        out["weights"] = ",".join(f"{m}:{period_weights[period].get(m, 0.0):.4f}" for m in selected)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def run(args) -> dict:
    frame = _load_table(Path(args.evaluation_ledger))
    days = sorted(frame["target_day"].unique())
    if len(days) < args.dev_days + 1:
        raise ValueError(f"need at least dev_days+1 dates, got {len(days)}")
    dev_days = days[: args.dev_days]
    test_days = days[args.dev_days :]
    models = tuple(sorted(frame["model_name"].unique()))
    if len(models) < 1:
        raise ValueError("no model candidates")
    max_models = min(args.max_models, len(models))
    combo_rows = []
    best_combo = None
    for count in range(1, max_models + 1):
        for combo in itertools.combinations(models, count):
            period_fits = {}
            trace = []
            for period, _, _ in SEGMENTS:
                fit, period_trace = _fit_period(frame, dev_days, combo, period)
                period_fits[period] = fit
                trace.extend(period_trace)
            dev_fused = _apply(frame, dev_days, combo, {p: period_fits[p]["weights"] for p in period_fits})
            metrics = _direction_metrics(dev_fused["y_true_spread"], dev_fused["y_pred_spread"])
            daily = dev_fused.groupby("target_day").apply(
                lambda g: _direction_metrics(g["y_true_spread"], g["y_pred_spread"])["direction_accuracy"],
                include_groups=False,
            )
            row = {
                "models": ",".join(combo),
                "model_count": count,
                **metrics,
                "daily_accuracy_mean": float(daily.mean()),
                "daily_accuracy_std": float(daily.std(ddof=0)),
                "period_weights": json.dumps({p: period_fits[p]["weights"] for p in period_fits}, ensure_ascii=False),
            }
            combo_rows.append(row)
            key = (
                metrics["direction_accuracy"],
                metrics["balanced_direction_accuracy"],
                -float(daily.std(ddof=0)),
                -count,
            )
            if best_combo is None or key > best_combo[0]:
                best_combo = (key, combo, period_fits, trace)
    assert best_combo is not None
    _, selected, fits, trace = best_combo
    weights = {p: fits[p]["weights"] for p in fits}
    dev_out = _apply(frame, dev_days, selected, weights)
    test_out = _apply(frame, test_days, selected, weights)
    all_out = pd.concat([dev_out.assign(split="dev"), test_out.assign(split="test")], ignore_index=True)
    equal_weights = {p: {m: 1.0 / len(selected) for m in selected} for p, _, _ in SEGMENTS}
    global_vector = _fit_global(frame, dev_days, selected)
    global_weights = {p: dict(global_vector) for p, _, _ in SEGMENTS}
    strategy_rows = []
    for strategy, strategy_weights in {
        "selected_per_segment": weights,
        "equal_selected_models": equal_weights,
        "single_global_weight_vector": global_weights,
    }.items():
        for split, split_days in (("dev", dev_days), ("test", test_days)):
            fused = _apply(frame, split_days, selected, strategy_weights)
            strategy_rows.append({"strategy": strategy, "split": split, **_direction_metrics(fused["y_true_spread"], fused["y_pred_spread"]), "days": int(fused["target_day"].nunique()), "slots": int(len(fused)), "mae": float(np.mean(np.abs(fused["y_pred_spread"] - fused["y_true_spread"])))})
    daily_rows = []
    for (split, target_day, period), group in all_out.groupby(["split", "target_day", "period"], sort=True):
        daily_rows.append({"split": split, "target_day": target_day, "period": period, **_direction_metrics(group["y_true_spread"], group["y_pred_spread"])})
    summary_rows = []
    for split, group in all_out.groupby("split", sort=False):
        summary_rows.append({"split": split, **_direction_metrics(group["y_true_spread"], group["y_pred_spread"]), "days": int(group["target_day"].nunique()), "slots": int(len(group)), "mae": float(np.mean(np.abs(group["y_pred_spread"] - group["y_true_spread"])))})
        for period, period_group in group.groupby("period"):
            summary_rows.append({"split": split, "period": period, **_direction_metrics(period_group["y_true_spread"], period_group["y_pred_spread"]), "days": int(period_group["target_day"].nunique()), "slots": int(len(period_group)), "mae": float(np.mean(np.abs(period_group["y_pred_spread"] - period_group["y_true_spread"])))})
    out_root = Path(args.output_root)
    _atomic_csv(out_root / "combo_selection_report.csv", pd.DataFrame(combo_rows).sort_values(["direction_accuracy", "balanced_direction_accuracy"], ascending=False))
    _atomic_csv(out_root / "weight_search_trace.csv", pd.DataFrame(trace))
    _atomic_csv(out_root / "weights.csv", pd.DataFrame([{"period": p, "model_name": m, "weight": w} for p, values in weights.items() for m, w in values.items()]))
    _atomic_csv(out_root / "summary.csv", pd.DataFrame(summary_rows))
    _atomic_csv(out_root / "fusion_strategy_comparison.csv", pd.DataFrame(strategy_rows))
    daily_frame = pd.DataFrame(daily_rows)
    _atomic_csv(out_root / "daily_accuracy.csv", daily_frame)
    stability_rows = []
    rng = np.random.default_rng(42)
    for split, group in daily_frame.groupby("split"):
        values = group["direction_accuracy"].dropna().to_numpy(float)
        boot = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)])
        stability_rows.append({"split": split, "daily_mean_accuracy": float(values.mean()), "daily_std_accuracy": float(values.std()), "bootstrap_ci95_low": float(np.quantile(boot, 0.025)), "bootstrap_ci95_high": float(np.quantile(boot, 0.975)), "daily_observations": int(len(values))})
    _atomic_csv(out_root / "stability_report.csv", pd.DataFrame(stability_rows))
    all_out.to_parquet(out_root / "fused_predictions.parquet", index=False)
    manifest = {
        "status": "complete",
        "method": "nonnegative_continuous_spread_weighted_fusion",
        "direction_metric": "sign(final_spread)==sign(actual_spread); actual zero excluded; predicted zero wrong",
        "selected_models": list(selected),
        "dev_days": dev_days,
        "test_days": test_days,
        "weight_step": WEIGHT_STEP,
        "period_weights": weights,
        "equal_selected_weights": equal_weights,
        "global_weight_vector": global_vector,
        "selection_priority": ["direction_accuracy", "balanced_direction_accuracy", "daily_stability", "simplicity"],
    }
    _atomic_json(out_root / "fusion_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-ledger", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dev-days", type=int, default=30)
    parser.add_argument("--max-models", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
