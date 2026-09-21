"""Causal hierarchical competence router over already strict expert OOS outputs.

This is an additive pilot on top of the retained Similar-Day route.  It does not
retrain the base experts.  For target day ``D`` it estimates each expert's
competence at three resolutions (global, period and hour) using only complete
OOS labels through ``D-2``.  The estimates are shrunk toward the expert's
global competence before being used as soft mixture weights.  The router is
therefore different from a hard per-slot winner-takes-all selector and does not
use the target-day label for calibration.

The script is research-only; it never writes formal ledger/runs outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pos, neg = y == 1, y == -1
    pr = float((p[pos] == 1).mean()) if pos.any() else float("nan")
    nr = float((p[neg] == -1).mean()) if neg.any() else float("nan")
    return {
        "n": int(len(y)),
        "direction_accuracy": float((p == y).mean()),
        "positive_recall": pr,
        "negative_recall": nr,
        "balanced_accuracy": float(np.nanmean([pr, nr])),
        "all_negative_baseline": float(neg.mean()),
    }


def _beta_rate(correct: pd.Series, prior: float, strength: float) -> tuple[float, int]:
    n = int(correct.size)
    if n == 0:
        return float(prior), 0
    rate = float((correct.sum() + strength * prior) / (n + strength))
    return rate, n


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(x, dtype=float) / max(float(temperature), 1e-6)
    z -= np.nanmax(z)
    w = np.exp(np.clip(z, -50.0, 50.0))
    return w / max(float(w.sum()), 1e-12)


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    required = {"target_day", "hour_business", "period", "target_spread", "variant", "predicted_direction"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"source missing columns: {sorted(missing)}")
    frame["target_day"] = pd.to_datetime(frame["target_day"], errors="coerce").dt.normalize()
    frame["target_spread"] = pd.to_numeric(frame["target_spread"], errors="coerce")
    frame["predicted_direction"] = pd.to_numeric(frame["predicted_direction"], errors="coerce")
    frame = frame.dropna(subset=["target_day", "target_spread", "predicted_direction"]).copy()
    if frame.empty or frame["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("empty source or fresh final holdout touched")
    train_col = "router_training_last_day" if "router_training_last_day" in frame else "training_last_day"
    if train_col in frame:
        frame[train_col] = pd.to_datetime(frame[train_col], errors="coerce").dt.normalize()
        if frame[train_col].isna().any():
            raise RuntimeError("source has invalid training boundary metadata")
        if (frame[train_col] > frame["target_day"] - pd.Timedelta(days=2)).any():
            raise RuntimeError("source violates strict D-2 router boundary")

    variants = sorted(frame["variant"].dropna().unique().tolist())
    if args.experts:
        requested = [x.strip() for x in args.experts.split(",") if x.strip()]
        missing_experts = sorted(set(requested) - set(variants))
        if missing_experts:
            raise RuntimeError(f"requested experts missing: {missing_experts}")
        variants = requested
    if len(variants) < 2:
        raise RuntimeError("hierarchical router needs at least two experts")

    direction = frame.pivot_table(
        index=["target_day", "hour_business", "period"],
        columns="variant", values="predicted_direction", aggfunc="first",
    ).reindex(columns=variants)
    probability = None
    if "prob_positive" in frame.columns:
        probability = frame.pivot_table(
            index=["target_day", "hour_business", "period"],
            columns="variant", values="prob_positive", aggfunc="first",
        ).reindex(index=direction.index, columns=variants)
    labels = frame.drop_duplicates(["target_day", "hour_business", "period"]).set_index(
        ["target_day", "hour_business", "period"]
    )["target_spread"]
    labels = np.sign(labels.reindex(direction.index).to_numpy(float)).astype(int)
    if np.any(labels == 0):
        raise RuntimeError("zero spread labels are not supported by the direction contract")

    all_days = sorted(direction.index.get_level_values("target_day").unique())
    targets = [d for d in all_days if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    rows: list[dict] = []
    audits: list[dict] = []
    base = "SD20_static" if "SD20_static" in variants else variants[0]
    for day in targets:
        cutoff = pd.Timestamp(day) - pd.Timedelta(days=2)
        history_days = [d for d in all_days if d <= cutoff][-args.history_days:]
        if len(history_days) < args.min_history_days:
            continue
        hmask = direction.index.get_level_values("target_day").isin(history_days)
        qmask = direction.index.get_level_values("target_day") == day
        hdir = direction.loc[hmask]
        qdir = direction.loc[qmask]
        hy = labels[hmask]
        qy = labels[qmask]
        if len(qdir) != 24:
            raise RuntimeError(f"{day.date()}: expected 24 target slots, got {len(qdir)}")
        correctness = hdir.to_numpy(float) == hy[:, None]
        global_rates = {}
        global_counts = {}
        for j, expert in enumerate(variants):
            global_rates[expert], global_counts[expert] = _beta_rate(
                pd.Series(correctness[:, j]), 0.5, args.prior_strength
            )
        # Historical competence tables.  Each local estimate is explicitly
        # shrunk toward the expert's global rate, so sparse hour/period cells
        # cannot create an unstable winner.
        hist_index = hdir.index
        hour_values = hist_index.get_level_values("hour_business").to_numpy(int)
        period_values = hist_index.get_level_values("period").to_numpy(str)
        q_hours = qdir.index.get_level_values("hour_business").to_numpy(int)
        q_periods = qdir.index.get_level_values("period").to_numpy(str)
        score_matrix = np.zeros((len(qdir), len(variants)), dtype=float)
        for j, expert in enumerate(variants):
            prior = global_rates[expert]
            for i, (hour, period) in enumerate(zip(q_hours, q_periods)):
                hour_mask = hour_values == hour
                period_mask = period_values == period
                hr, hn = _beta_rate(pd.Series(correctness[hour_mask, j]), prior, args.prior_strength)
                pe, pn = _beta_rate(pd.Series(correctness[period_mask, j]), prior, args.prior_strength)
                # The hour estimate is most specific, period is a stabilizer,
                # and global is a robust prior.  Weights are fixed before the
                # evaluation window and are not tuned against target labels.
                score_matrix[i, j] = (
                    args.hour_weight * hr
                    + args.period_weight * pe
                    + args.global_weight * prior
                )
        weights = np.vstack([_softmax(row, args.temperature) for row in score_matrix])
        vote = (qdir.to_numpy(float) == 1.0).astype(float)
        if probability is not None:
            qp = probability.loc[qmask].to_numpy(float)
            qp = np.where(np.isfinite(qp), qp, vote)
            qp = np.clip(qp, 0.0, 1.0)
            soft_signal = args.probability_weight * np.sum(weights * qp, axis=1) + (1.0 - args.probability_weight) * np.sum(weights * vote, axis=1)
        else:
            soft_signal = np.sum(weights * vote, axis=1)
        pred = np.where(soft_signal >= args.threshold, 1, -1).astype(int)
        # Stable fallback: if the mixture has no separation from 0.5, keep
        # the retained expert instead of manufacturing a weak positive call.
        if args.min_signal_margin > 0:
            uncertain = np.abs(soft_signal - 0.5) < args.min_signal_margin
            pred[uncertain] = qdir[base].to_numpy(int)[uncertain]
        for i, idx in enumerate(qdir.index):
            rows.append({
                "target_day": pd.Timestamp(idx[0]).date().isoformat(),
                "hour_business": int(idx[1]),
                "period": str(idx[2]),
                "y_true": int(qy[i]),
                "baseline_pred": int(qdir[base].iloc[i]),
                "predicted_direction": int(pred[i]),
                "soft_signal": float(soft_signal[i]),
                "selected_expert": variants[int(np.argmax(weights[i]))],
                "max_weight": float(np.max(weights[i])),
                "training_last_day": max(history_days).date().isoformat(),
            })
        audits.append({
            "target_day": pd.Timestamp(day).date().isoformat(),
            "training_last_day": max(history_days).date().isoformat(),
            "required_last_day": cutoff.date().isoformat(),
            "strict_ok": bool(max(history_days) <= cutoff),
            "history_days": len(history_days),
            "mean_max_weight": float(np.max(weights, axis=1).mean()),
            "positive_signal_rate": float((soft_signal >= args.threshold).mean()),
        })

    pred = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if pred.empty or audit.empty or not audit["strict_ok"].all():
        raise RuntimeError("no valid output or strict audit failed")
    summary = []
    for name, col in [("baseline", "baseline_pred"), ("hierarchical_router", "predicted_direction")]:
        summary.append({"variant": name, **metrics(pred.y_true.to_numpy(int), pred[col].to_numpy(int))})
    pred["month"] = pred.target_day.str[:7]
    monthly = []
    for month, group in pred.groupby("month", sort=True):
        for name, col in [("baseline", "baseline_pred"), ("hierarchical_router", "predicted_direction")]:
            monthly.append({"month": month, "variant": name, **metrics(group.y_true.to_numpy(int), group[col].to_numpy(int))})
    pred.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "router_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly).to_csv(out / "monthly.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS",
        "route": "A_hierarchical_soft_competence_router",
        "source": str(source),
        "experts": variants,
        "baseline": base,
        "forecast_origin": "D-1 14:00",
        "training_last_day": "per target <= D-2",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "history_days": args.history_days,
        "min_history_days": args.min_history_days,
        "prior_strength": args.prior_strength,
        "hour_weight": args.hour_weight,
        "period_weight": args.period_weight,
        "global_weight": args.global_weight,
        "temperature": args.temperature,
        "probability_weight": args.probability_weight,
        "threshold": args.threshold,
        "min_signal_margin": args.min_signal_margin,
        "selection": "fixed predeclared weights; no target-day labels used",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(monthly).to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--history-days", type=int, default=90)
    parser.add_argument("--min-history-days", type=int, default=30)
    parser.add_argument("--experts", default="")
    parser.add_argument("--prior-strength", type=float, default=12.0)
    parser.add_argument("--hour-weight", type=float, default=0.45)
    parser.add_argument("--period-weight", type=float, default=0.35)
    parser.add_argument("--global-weight", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--probability-weight", type=float, default=0.35)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-signal-margin", type=float, default=0.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
