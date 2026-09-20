"""Cycle 22 A pilot: causal block-level expert aggregation.

The pilot tests a small temporal-hierarchy-inspired rule on strict OOS expert
predictions.  For each target day and 8-slot block, the selected expert is
chosen using only historical OOS correctness from days <= D-2.  The score is a
fixed sign-aware utility (accuracy plus a fixed positive-event recall term),
so no target-day label or post-cutoff observation can influence selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")
EXPERTS = ["P6_static", "A_competence_soft", "CTX_static", "SD20_static"]


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pos, neg = y > 0, y < 0
    pr = float((p[pos] == 1).mean()) if pos.any() else float("nan")
    nr = float((p[neg] == -1).mean()) if neg.any() else float("nan")
    return {
        "slots": int(len(y)),
        "direction_accuracy": float((p == y).mean()),
        "positive_recall": pr,
        "negative_recall": nr,
        "balanced_accuracy": float(np.nanmean([pr, nr])),
        "all_negative_baseline": float(neg.mean()),
    }


def run(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    x = pd.read_parquet(source)
    required = {"target_day", "hour_business", "target_spread", "variant", "predicted_direction"}
    missing = required.difference(x.columns)
    if missing:
        raise RuntimeError(f"source missing columns: {sorted(missing)}")
    x["target_day"] = pd.to_datetime(x["target_day"]).dt.normalize()
    if x["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("source touches fresh final holdout")
    x["y"] = np.sign(pd.to_numeric(x["target_spread"], errors="coerce")).astype(int)
    x["predicted_direction"] = pd.to_numeric(x["predicted_direction"], errors="coerce").astype(int)
    x = x[x.y != 0].copy()
    available = [e for e in EXPERTS if e in set(x.variant)]
    if len(available) < 2:
        raise RuntimeError(f"need at least two strict experts, found {available}")
    w = x.pivot_table(index=["target_day", "hour_business"], columns="variant", values="predicted_direction", aggfunc="first").reset_index()
    w.columns.name = None
    y = x.drop_duplicates(["target_day", "hour_business"])[["target_day", "hour_business", "y"]]
    w = w.merge(y, on=["target_day", "hour_business"], how="inner").dropna(subset=available).sort_values(["target_day", "hour_business"])
    w["block"] = ((w["hour_business"].astype(int) - 1) // 8 + 1).astype(int)
    all_days = sorted(w.target_day.unique())
    target_days = [d for d in all_days if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    rows, audits = [], []
    for day in target_days:
        cutoff = pd.Timestamp(day) - pd.Timedelta(days=2)
        hist_days = [d for d in all_days if d <= cutoff][-args.history_days:]
        if len(hist_days) < args.min_history_days:
            continue
        hist = w[w.target_day.isin(hist_days)]
        q = w[w.target_day == day].copy()
        if len(q) != 24:
            raise RuntimeError(f"{day.date()}: expected 24 slots")
        pred = np.empty(len(q), dtype=int)
        chosen = {}
        for block, idx in q.groupby("block").groups.items():
            hb = hist[hist.block == block]
            scores = {}
            for e in available:
                yy = hb.y.to_numpy(int)
                pp = hb[e].to_numpy(int)
                correct = pp == yy
                positive_recall = float((pp[yy > 0] == 1).mean()) if (yy > 0).any() else 0.0
                scores[e] = float(correct.mean() + args.positive_recall_weight * positive_recall)
            # Stable tie-break keeps the conservative P6 expert as fallback.
            best = max(available, key=lambda e: (scores[e], e == "P6_static"))
            chosen[int(block)] = {"expert": best, "scores": scores}
            pred[np.asarray(list(idx)) - q.index.min()] = q.loc[idx, best].to_numpy(int)
        yy = q.y.to_numpy(int)
        m = metric(yy, pred)
        rows.append({"target_day": day.date().isoformat(), **m, "selected_by_block": json.dumps(chosen, ensure_ascii=False), "training_last_day": max(hist_days).date().isoformat()})
        audits.append({"target_day": day.date().isoformat(), "training_last_day": max(hist_days).date().isoformat(), "required_last_day": cutoff.date().isoformat(), "strict_ok": bool(max(hist_days) <= cutoff), "history_days": len(hist_days)})
    if not rows:
        raise RuntimeError("no target days passed history gate")
    daily = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if not audit.strict_ok.all():
        raise RuntimeError("strict audit failed")
    daily["month"] = daily.target_day.str[:7]
    monthly = daily.groupby("month", as_index=False).agg(days=("target_day", "nunique"), direction_accuracy=("direction_accuracy", "mean"), positive_recall=("positive_recall", "mean"), negative_recall=("negative_recall", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), all_negative_baseline=("all_negative_baseline", "mean"))
    summary = {k: float(daily[k].mean()) for k in ["direction_accuracy", "positive_recall", "negative_recall", "balanced_accuracy", "all_negative_baseline"]}
    summary.update({"days": int(daily.target_day.nunique()), "gain_vs_all_negative": summary["direction_accuracy"] - summary["all_negative_baseline"]})
    daily.to_csv(out / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(out / "monthly.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "router_training_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "manifest.json").write_text(json.dumps({"status": "STRICT/PASS", "route": "A_causal_block_expert_aggregation", "forecast_origin": "D-1 14:00", "training_last_day": "per target day <= D-2", "target_day_actual_as_feature": False, "target_day_DA_as_feature": False, "d1_post14_spread_as_feature": False, "final_holdout_touched": False, "experts": available, "block_slots": 8, "positive_recall_weight": args.positive_recall_weight, "selection_labels": "historical OOS correctness only"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--history-days", type=int, default=90)
    ap.add_argument("--min-history-days", type=int, default=60)
    ap.add_argument("--positive-recall-weight", type=float, default=0.5)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
