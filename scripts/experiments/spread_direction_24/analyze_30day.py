"""Audit model complementarity and prequential fusion for the 24-point spread experiment.

All learned decisions are chronological: predictions for day D may only use
evaluation rows whose target_day is earlier than D.  This script never writes
to production ledgers and never retrains a base model.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.spread_metrics import smape_percent


KEYS = ["target_day", "ds", "hour_business", "period"]
REQUIRED = set(KEYS) | {
    "model_name",
    "y_pred_spread",
    "y_true_spread",
    "predicted_direction",
    "direction_correct",
}


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_ledgers(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["source_ledger"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    merged["target_day"] = pd.to_datetime(merged["target_day"]).dt.strftime("%Y-%m-%d")
    merged["ds"] = pd.to_datetime(merged["ds"], errors="raise")
    merged["y_pred_spread"] = pd.to_numeric(merged["y_pred_spread"], errors="raise")
    merged["y_true_spread"] = pd.to_numeric(merged["y_true_spread"], errors="raise")
    if not np.isfinite(merged[["y_pred_spread", "y_true_spread"]].to_numpy()).all():
        raise ValueError("ledger contains NaN or infinite spread values")

    duplicate = merged.duplicated(KEYS + ["model_name"], keep=False)
    if duplicate.any():
        check = merged.loc[duplicate].groupby(KEYS + ["model_name"], dropna=False)
        inconsistent = check[["y_pred_spread", "y_true_spread"]].nunique().max(axis=1) > 1
        if inconsistent.any():
            raise ValueError(f"conflicting duplicate rows: {int(inconsistent.sum())}")
        merged = merged.drop_duplicates(KEYS + ["model_name"], keep="last")

    counts = merged.groupby(["target_day", "model_name"])["hour_business"].agg(
        rows="size", slots="nunique"
    )
    bad = counts[(counts["rows"] != 24) | (counts["slots"] != 24)]
    if not bad.empty:
        raise ValueError(f"incomplete model-days:\n{bad.to_string()}")
    return merged.sort_values(KEYS + ["model_name"]).reset_index(drop=True)


def metric_row(group: pd.DataFrame, name: str, *, model_col: str = "model_name") -> dict:
    true = group["y_true_spread"].to_numpy(float)
    pred = group["y_pred_spread"].to_numpy(float)
    true_sign = np.sign(true)
    pred_sign = np.sign(pred)
    eligible = true_sign != 0
    correct = eligible & (true_sign == pred_sign)
    pos = true_sign > 0
    neg = true_sign < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    weights = np.abs(true[eligible])
    return {
        model_col: name,
        "days": int(group["target_day"].nunique()),
        "n_slots": int(len(group)),
        "n_direction_eligible": int(eligible.sum()),
        "n_positive_actual": int(pos.sum()),
        "n_negative_actual": int(neg.sum()),
        "n_zero_actual": int((true_sign == 0).sum()),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
        "abs_spread_weighted_direction_accuracy": (
            float(np.average(correct[eligible].astype(float), weights=weights))
            if eligible.any() and weights.sum() > 0
            else math.nan
        ),
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "spread_smape_pct": smape_percent(true, pred),
    }


def base_reports(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        [metric_row(group, model) for model, group in ledger.groupby("model_name")]
    ).sort_values(["balanced_direction_accuracy", "direction_accuracy"], ascending=False)

    daily_rows = [
        metric_row(group, model)
        | {"target_day": day}
        for (day, model), group in ledger.groupby(["target_day", "model_name"])
    ]
    daily = pd.DataFrame(daily_rows)
    stability_rows: list[dict] = []
    for model, group in daily.groupby("model_name"):
        acc = group["direction_accuracy"]
        bal = group["balanced_direction_accuracy"]
        worst = acc.nsmallest(min(5, len(acc)))
        stability_rows.append(
            {
                "model_name": model,
                "days": int(len(group)),
                "daily_accuracy_mean": float(acc.mean()),
                "daily_accuracy_std": float(acc.std(ddof=0)),
                "daily_accuracy_min": float(acc.min()),
                "worst5_accuracy_mean": float(worst.mean()),
                "daily_balanced_mean": float(bal.mean()),
                "daily_balanced_std": float(bal.std(ddof=0)),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["daily_balanced_mean", "daily_accuracy_mean"], ascending=False
    )

    period_rows = [
        metric_row(group, model) | {"period": period}
        for (model, period), group in ledger.groupby(["model_name", "period"])
    ]
    periods = pd.DataFrame(period_rows).sort_values(["model_name", "period"])
    return summary.reset_index(drop=True), stability.reset_index(drop=True), periods.reset_index(drop=True)


def complementarity_report(ledger: pd.DataFrame, primary: str) -> pd.DataFrame:
    pivot = ledger.pivot(index=KEYS, columns="model_name", values="y_pred_spread")
    truth = ledger.drop_duplicates(KEYS).set_index(KEYS)["y_true_spread"].reindex(pivot.index)
    if primary not in pivot.columns:
        raise ValueError(f"primary model not found: {primary}")
    true_sign = np.sign(truth)
    p_sign = np.sign(pivot[primary])
    eligible = true_sign != 0
    p_ok = eligible & (p_sign == true_sign)
    rows: list[dict] = []
    for helper in pivot.columns:
        if helper == primary:
            continue
        common = eligible & pivot[helper].notna() & pivot[primary].notna()
        h_sign = np.sign(pivot[helper])
        h_ok = common & (h_sign == true_sign)
        p_common_ok = common & p_ok
        p_errors = common & ~p_ok
        disagreement = common & (h_sign != p_sign)
        rescues = p_errors & h_ok
        rows.append(
            {
                "primary_model": primary,
                "helper_model": helper,
                "common_slots": int(common.sum()),
                "primary_errors": int(p_errors.sum()),
                "helper_rescues": int(rescues.sum()),
                "rescue_rate": float(rescues.sum() / p_errors.sum()) if p_errors.any() else math.nan,
                "simultaneous_errors": int((common & ~p_ok & ~h_ok).sum()),
                "oracle_accuracy": float((p_common_ok | h_ok)[common].mean()),
                "disagreement_rate": float(disagreement.sum() / common.sum()),
                "primary_accuracy_on_disagreement": (
                    float(p_ok[disagreement].mean()) if disagreement.any() else math.nan
                ),
                "helper_accuracy_on_disagreement": (
                    float(h_ok[disagreement].mean()) if disagreement.any() else math.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["oracle_accuracy", "rescue_rate"], ascending=False
    ).reset_index(drop=True)


def _posterior_accuracy(history: pd.DataFrame, model: str, period: str, pred_sign: int) -> tuple[float, int]:
    rows = history[
        history["model_name"].eq(model)
        & history["period"].eq(period)
        & np.sign(history["y_pred_spread"]).eq(pred_sign)
        & np.sign(history["y_true_spread"]).ne(0)
    ]
    n = len(rows)
    correct = (np.sign(rows["y_pred_spread"]) == np.sign(rows["y_true_spread"])).sum()
    return float((correct + 2.0) / (n + 4.0)), int(n)


def _posterior_balanced_accuracy(history: pd.DataFrame, model: str, period: str) -> tuple[float, int]:
    rows = history[history["model_name"].eq(model) & history["period"].eq(period)]
    true_sign = np.sign(rows["y_true_spread"])
    pred_sign = np.sign(rows["y_pred_spread"])
    scores: list[float] = []
    support = 0
    for actual_sign in (1, -1):
        mask = true_sign.eq(actual_sign)
        n = int(mask.sum())
        support += n
        correct = int((pred_sign[mask] == actual_sign).sum())
        scores.append((correct + 2.0) / (n + 4.0))
    return float(np.mean(scores)), support


def _select_balanced_threshold(
    history: pd.DataFrame, model: str, period: str
) -> tuple[float, float, int]:
    """Select a sign threshold using strictly prior rows for one period."""
    rows = history[history["model_name"].eq(model) & history["period"].eq(period)]
    x = rows["y_pred_spread"].to_numpy(float)
    y = np.sign(rows["y_true_spread"].to_numpy(float))
    eligible = y != 0
    x, y = x[eligible], y[eligible]
    if len(x) < 16 or not (y > 0).any() or not (y < 0).any():
        return 0.0, math.nan, int(len(x))
    candidates = np.unique(np.r_[0.0, np.quantile(x, np.linspace(0.05, 0.95, 37))])
    best_threshold = 0.0
    best_score = -math.inf
    for threshold in candidates:
        pred = np.where(x > threshold, 1, -1)
        pos = y > 0
        neg = y < 0
        score = 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == -1).mean())
        if score > best_score or (
            math.isclose(score, best_score) and abs(threshold) < abs(best_threshold)
        ):
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score, int(len(x))


def prequential_fusions(
    ledger: pd.DataFrame,
    *,
    primary: str,
    helper: str,
    baseline: str,
    warmup_days: int,
    min_gate_support: int,
    gate_margin: float,
    lookback_days: int = 0,
) -> pd.DataFrame:
    needed = [primary, helper, baseline]
    available = set(ledger["model_name"])
    missing = set(needed) - available
    if missing:
        raise ValueError(f"fusion models missing: {sorted(missing)}")
    dates = sorted(ledger["target_day"].unique())
    if len(dates) <= warmup_days:
        raise ValueError(f"need more than {warmup_days} days for prequential fusion")

    wide = ledger[ledger["model_name"].isin(needed)].pivot(
        index=KEYS, columns="model_name", values="y_pred_spread"
    )
    truth = ledger.drop_duplicates(KEYS).set_index(KEYS)["y_true_spread"].reindex(wide.index)
    if wide[needed].isna().any().any():
        raise ValueError("fusion candidates do not share a complete model-day grid")

    out: list[dict] = []
    for day_index, day in enumerate(dates):
        if day_index < warmup_days:
            continue
        prior_days = dates[:day_index]
        if lookback_days > 0:
            prior_days = prior_days[-lookback_days:]
        history = ledger[ledger["target_day"].isin(prior_days)]
        today = wide.loc[wide.index.get_level_values("target_day") == day]
        for key, values in today.iterrows():
            period = str(key[3])
            y_true = float(truth.loc[key])
            preds = {model: float(values[model]) for model in needed}
            signs = {model: int(np.sign(value)) for model, value in preds.items()}

            strategies: dict[str, tuple[float, str, int]] = {
                f"single_{primary}": (preds[primary], "primary", 0),
            }
            threshold, threshold_score, threshold_support = _select_balanced_threshold(
                history, primary, period
            )
            calibrated_margin = preds[primary] - threshold
            if calibrated_margin <= 0:
                calibrated_margin = -max(abs(calibrated_margin), 1e-9)
            strategies[f"threshold_{primary}"] = (
                calibrated_margin,
                f"threshold={threshold:.6g};history_balanced={threshold_score:.6g}",
                threshold_support,
            )
            vote = signs[primary] + signs[helper] + signs[baseline]
            vote_sign = int(np.sign(vote)) or signs[primary]
            strategies[f"majority_{primary}_{helper}_{baseline}"] = (
                float(vote_sign),
                "majority_vote",
                0,
            )
            calibrated_primary_sign = 1 if calibrated_margin > 0 else -1
            calibrated_vote = (
                calibrated_primary_sign + signs[helper] + signs[baseline]
            )
            calibrated_vote_sign = int(np.sign(calibrated_vote)) or calibrated_primary_sign
            strategies[f"majority_threshold_{primary}_{helper}_{baseline}"] = (
                float(calibrated_vote_sign),
                f"primary_threshold={threshold:.6g}",
                threshold_support,
            )

            weighted_score = 0.0
            weight_parts: list[str] = []
            for model in needed:
                reliability, support = _posterior_accuracy(history, model, period, signs[model])
                edge = max(0.01, 2.0 * reliability - 1.0)
                hist_abs = history[
                    history["model_name"].eq(model) & history["period"].eq(period)
                ]["y_pred_spread"].abs()
                scale = float(hist_abs.median()) if len(hist_abs) else 1.0
                scale = max(scale, 1e-6)
                confidence = float(np.clip(abs(preds[model]) / scale, 0.25, 3.0))
                weighted_score += edge * confidence * signs[model]
                weight_parts.append(f"{model}:{edge:.3f}x{confidence:.3f}(n={support})")
            dynamic_sign = int(np.sign(weighted_score)) or signs[primary]
            strategies["dynamic_reliability"] = (
                float(dynamic_sign),
                ";".join(weight_parts),
                0,
            )

            period_scores = {
                model: _posterior_balanced_accuracy(history, model, period) for model in needed
            }
            period_winner = max(
                needed,
                key=lambda model: (period_scores[model][0], model == primary),
            )
            strategies["period_champion"] = (
                preds[period_winner],
                f"winner={period_winner};"
                + ";".join(
                    f"{model}:{period_scores[model][0]:.3f}(n={period_scores[model][1]})"
                    for model in needed
                ),
                period_scores[period_winner][1],
            )

            conditional_scores = {
                model: _posterior_accuracy(history, model, period, signs[model]) for model in needed
            }
            conditional_winner = max(
                needed,
                key=lambda model: (conditional_scores[model][0], model == primary),
            )
            strategies["conditional_champion"] = (
                preds[conditional_winner],
                f"winner={conditional_winner};"
                + ";".join(
                    f"{model}:{conditional_scores[model][0]:.3f}(n={conditional_scores[model][1]})"
                    for model in needed
                ),
                conditional_scores[conditional_winner][1],
            )

            pair_pred = preds[primary]
            pair_reason = "keep_primary_agree"
            pair_support = 0
            if signs[helper] != signs[primary]:
                pair_history = history[history["model_name"].isin([primary, helper])].pivot(
                    index=KEYS, columns="model_name", values="y_pred_spread"
                ).dropna(subset=[primary, helper])
                pair_truth = (
                    history.drop_duplicates(KEYS)
                    .set_index(KEYS)["y_true_spread"]
                    .reindex(pair_history.index)
                )
                pair_match = (
                    (pair_history.index.get_level_values("period").astype(str) == period)
                    & np.sign(pair_history[primary]).eq(signs[primary])
                    & np.sign(pair_history[helper]).eq(signs[helper])
                    & np.sign(pair_truth).ne(0)
                )
                pair_support = int(pair_match.sum())
                if pair_support:
                    pair_primary_hits = float(
                        (np.sign(pair_history.loc[pair_match, primary]) == np.sign(pair_truth[pair_match])).mean()
                    )
                    pair_helper_hits = float(
                        (np.sign(pair_history.loc[pair_match, helper]) == np.sign(pair_truth[pair_match])).mean()
                    )
                else:
                    pair_primary_hits = pair_helper_hits = 0.0
                if pair_support >= min_gate_support and pair_helper_hits - pair_primary_hits >= gate_margin:
                    pair_pred = preds[helper]
                    pair_reason = f"override_pair_win:{pair_helper_hits:.3f}>{pair_primary_hits:.3f}"
                else:
                    pair_reason = (
                        f"keep_primary_pair:n={pair_support};helper={pair_helper_hits:.3f};"
                        f"primary={pair_primary_hits:.3f}"
                    )
            strategies[f"pair_gate_{primary}_{helper}"] = (
                pair_pred,
                pair_reason,
                pair_support,
            )

            gate_pred = preds[primary]
            gate_reason = "keep_primary_no_consensus"
            gate_support = 0
            if signs[helper] == signs[baseline] and signs[helper] != signs[primary]:
                hist_wide = history[history["model_name"].isin(needed)].pivot(
                    index=KEYS, columns="model_name", values="y_pred_spread"
                ).dropna(subset=needed)
                hist_truth = (
                    history.drop_duplicates(KEYS)
                    .set_index(KEYS)["y_true_spread"]
                    .reindex(hist_wide.index)
                )
                matching = (
                    (hist_wide.index.get_level_values("period").astype(str) == period)
                    & np.sign(hist_wide[primary]).eq(signs[primary])
                    & np.sign(hist_wide[helper]).eq(signs[helper])
                    & np.sign(hist_wide[baseline]).eq(signs[baseline])
                    & np.sign(hist_truth).ne(0)
                )
                gate_support = int(matching.sum())
                if gate_support:
                    primary_hits = float(
                        (np.sign(hist_wide.loc[matching, primary]) == np.sign(hist_truth[matching])).mean()
                    )
                    consensus_hits = float(
                        (np.sign(hist_wide.loc[matching, helper]) == np.sign(hist_truth[matching])).mean()
                    )
                else:
                    primary_hits = consensus_hits = 0.0
                if gate_support >= min_gate_support and consensus_hits - primary_hits >= gate_margin:
                    gate_pred = preds[helper]
                    gate_reason = (
                        f"override_consensus_win:{consensus_hits:.3f}>{primary_hits:.3f}"
                    )
                else:
                    gate_reason = (
                        f"keep_primary_support_or_margin:n={gate_support};"
                        f"consensus={consensus_hits:.3f};primary={primary_hits:.3f}"
                    )
            strategies[f"gate_{primary}_{helper}_{baseline}"] = (
                gate_pred,
                gate_reason,
                gate_support,
            )

            for strategy, (prediction, reason, support) in strategies.items():
                out.append(
                    {
                        "target_day": key[0],
                        "ds": key[1],
                        "hour_business": int(key[2]),
                        "period": period,
                        "model_name": strategy,
                        "y_pred_spread": prediction,
                        "y_true_spread": y_true,
                        "predicted_direction": int(np.sign(prediction)),
                        "actual_direction": int(np.sign(y_true)),
                        "direction_correct": bool(
                            np.sign(y_true) != 0 and np.sign(prediction) == np.sign(y_true)
                        ),
                        "decision_reason": reason,
                        "gate_support": support,
                        "history_days": day_index,
                    }
                )
    return pd.DataFrame(out)


def run(args: argparse.Namespace) -> dict:
    ledger_paths = [Path(path) for path in args.ledger]
    ledger = load_ledgers(ledger_paths)
    output = Path(args.output_dir)
    summary, stability, periods = base_reports(ledger)
    complement = complementarity_report(ledger, args.primary)
    fusion = prequential_fusions(
        ledger,
        primary=args.primary,
        helper=args.helper,
        baseline=args.baseline,
        warmup_days=args.warmup_days,
        min_gate_support=args.min_gate_support,
        gate_margin=args.gate_margin,
        lookback_days=args.lookback_days,
    )
    fusion_summary = pd.DataFrame(
        [metric_row(group, model) for model, group in fusion.groupby("model_name")]
    ).sort_values(["balanced_direction_accuracy", "direction_accuracy"], ascending=False)
    fusion_daily = pd.DataFrame(
        [
            metric_row(group, model) | {"target_day": day}
            for (day, model), group in fusion.groupby(["target_day", "model_name"])
        ]
    )

    _atomic_csv(output / "base_model_summary.csv", summary)
    _atomic_csv(output / "daily_stability.csv", stability)
    _atomic_csv(output / "period_metrics.csv", periods)
    _atomic_csv(output / "complementarity.csv", complement)
    _atomic_parquet(output / "fusion_prequential_predictions.parquet", fusion)
    _atomic_csv(output / "fusion_summary.csv", fusion_summary)
    _atomic_csv(output / "fusion_daily_metrics.csv", fusion_daily)
    manifest = {
        "analysis": "spread_direction_24_30day",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ledgers": [str(path) for path in ledger_paths],
        "dates": sorted(ledger["target_day"].unique()),
        "models": sorted(ledger["model_name"].unique()),
        "rows": int(len(ledger)),
        "prequential": {
            "primary": args.primary,
            "helper": args.helper,
            "baseline": args.baseline,
            "warmup_days": args.warmup_days,
            "min_gate_support": args.min_gate_support,
            "gate_margin": args.gate_margin,
            "lookback_days": args.lookback_days or "expanding",
            "invariant": "target day uses strictly earlier target_day evaluations only",
        },
        "outputs": {
            "base_model_summary": str(output / "base_model_summary.csv"),
            "complementarity": str(output / "complementarity.csv"),
            "fusion_summary": str(output / "fusion_summary.csv"),
        },
    }
    _atomic_json(output / "analysis_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary", default="sgdfnet")
    parser.add_argument("--helper", default="timemixer")
    parser.add_argument("--baseline", default="spread_lag24")
    parser.add_argument("--warmup-days", type=int, default=10)
    parser.add_argument("--min-gate-support", type=int, default=8)
    parser.add_argument("--gate-margin", type=float, default=0.10)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=0,
        help="strictly prior rolling window; 0 uses all earlier experiment days",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
