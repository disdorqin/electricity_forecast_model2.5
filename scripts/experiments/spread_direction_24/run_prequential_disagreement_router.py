"""Cycle 10 A pilot: prequential router focused on expert disagreement.

The Cycle 10 diagnostic shows that at least one strict OOS expert is correct on
many disagreement slots, but the route must choose without seeing the target
label. This runner trains a correctness model on historical OOS candidates only
and selects one candidate per target slot. It is deliberately limited to legal
prediction/calendar/disagreement features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def router_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=100,
        learning_rate=0.04, num_leaves=15, max_depth=4, min_child_samples=35,
        reg_lambda=3.0, random_state=seed, n_jobs=4, verbosity=-1,
    )


def make_long(wide: pd.DataFrame, variants: list[str], with_label: bool) -> pd.DataFrame:
    rows = []
    for variant in variants:
        g = wide[["target_day", "hour_business", "dow", "y", variant]].copy()
        g = g.rename(columns={variant: "candidate_pred"})
        g["variant"] = variant
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    vote = wide[variants].to_numpy(float)
    out["vote_positive_fraction"] = np.tile(np.nanmean((vote > 0).astype(float), axis=1), len(variants))
    out["n_unique_predictions"] = np.tile(wide[variants].nunique(axis=1, dropna=True).to_numpy(), len(variants))
    out["candidate_is_majority"] = (out["candidate_pred"] == np.where(out["vote_positive_fraction"] >= 0.5, 1, -1)).astype(int)
    out["candidate_vote_margin"] = np.abs(2.0 * out["vote_positive_fraction"] - 1.0)
    hour = out["hour_business"].to_numpy(float)
    dow = out["dow"].to_numpy(float)
    out["hour_sin"] = np.sin(2 * np.pi * (hour - 1) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * (hour - 1) / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    for candidate in variants:
        out[f"variant__{candidate}"] = (out["variant"] == candidate).astype(int)
    if with_label:
        out["correct"] = (out["candidate_pred"].to_numpy(int) == out["y"].to_numpy(int)).astype(int)
    return out


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


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(source)
    required = {"target_day", "hour_business", "target_spread", "predicted_direction", "variant"}
    if required.difference(ledger.columns):
        raise RuntimeError(f"source missing columns: {sorted(required.difference(ledger.columns))}")
    ledger["target_day"] = pd.to_datetime(ledger["target_day"]).dt.normalize()
    if ledger["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("source touches fresh final holdout")
    ledger["hour_business"] = pd.to_numeric(ledger["hour_business"], errors="coerce").astype(int)
    ledger["predicted_direction"] = pd.to_numeric(ledger["predicted_direction"], errors="coerce").astype(int)
    ledger["y"] = np.sign(pd.to_numeric(ledger["target_spread"], errors="coerce")).astype(int)
    ledger = ledger[ledger["y"] != 0].copy()
    variants = sorted(ledger["variant"].dropna().unique().tolist())
    if len(variants) < 3:
        raise RuntimeError("need at least three strict OOS candidate variants")
    wide = ledger.pivot_table(index=["target_day", "hour_business"], columns="variant", values="predicted_direction", aggfunc="first").reset_index()
    wide.columns.name = None
    y = ledger.drop_duplicates(["target_day", "hour_business"])[["target_day", "hour_business", "y"]]
    wide = wide.merge(y, on=["target_day", "hour_business"], how="inner")
    wide["dow"] = wide["target_day"].dt.dayofweek
    wide = wide.dropna(subset=variants + ["y"]).sort_values(["target_day", "hour_business"]).reset_index(drop=True)
    all_days = sorted(wide["target_day"].unique())
    target_days = [d for d in all_days if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    feature_cols = [
        "candidate_pred", "vote_positive_fraction", "n_unique_predictions",
        "candidate_is_majority", "candidate_vote_margin", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
    ] + [f"variant__{v}" for v in variants]
    daily_rows, prediction_rows, audits = [], [], []
    for target_day in target_days:
        cutoff = target_day - pd.Timedelta(days=2)
        history_days = [d for d in all_days if d <= cutoff][-args.history_days:]
        if len(history_days) < args.min_history_days:
            continue
        assert max(history_days) <= cutoff
        hist = wide[wide["target_day"].isin(history_days)].copy()
        query = wide[wide["target_day"] == target_day].copy()
        if len(query) != 24:
            raise RuntimeError(f"{target_day.date()}: expected 24 query slots, got {len(query)}")
        train_long = make_long(hist, variants, with_label=True)
        query_long = make_long(query, variants, with_label=False)
        if train_long["correct"].nunique() < 2:
            raise RuntimeError(f"{target_day.date()}: correctness labels have one class")
        model = router_model(args.seed)
        sample_weight = np.where(train_long["y"].to_numpy(int) > 0, args.positive_slot_weight, 1.0)
        model.fit(train_long[feature_cols], train_long["correct"], sample_weight=sample_weight)
        query_long["p_correct"] = model.predict_proba(query_long[feature_cols])[:, 1]
        query_long = query_long.sort_values(["target_day", "hour_business", "p_correct"], ascending=[True, True, False])
        selected = query_long.groupby(["target_day", "hour_business"], as_index=False).head(1).copy()
        selected = selected.sort_values("hour_business")
        y_true = selected["y"].to_numpy(int)
        pred = selected["candidate_pred"].to_numpy(int)
        m = metric(y_true, pred)
        m.update({"target_day": target_day.date().isoformat(), "selected_variant_mode": selected["variant"].mode().iat[0]})
        daily_rows.append(m)
        for _, row in selected.iterrows():
            prediction_rows.append({
                "target_day": target_day.date().isoformat(), "hour_business": int(row["hour_business"]),
                "y_true": int(row["y"]), "predicted_direction": int(row["candidate_pred"]),
                "selected_variant": row["variant"], "p_correct": float(row["p_correct"]),
                "n_unique_predictions": int(row["n_unique_predictions"]),
                "training_last_day": max(history_days).date().isoformat(),
            })
        audits.append({
            "target_day": target_day.date().isoformat(),
            "training_last_day": max(history_days).date().isoformat(),
            "required_last_day": cutoff.date().isoformat(),
            "strict_ok": bool(max(history_days) <= cutoff),
            "history_days": len(history_days), "training_rows": len(train_long),
        })
    if not daily_rows:
        raise RuntimeError("no target days passed prequential history gate")
    daily = pd.DataFrame(daily_rows)
    predictions = pd.DataFrame(prediction_rows)
    audit = pd.DataFrame(audits)
    if not audit["strict_ok"].all():
        raise RuntimeError("router strict audit failed")
    daily["month"] = daily["target_day"].str[:7]
    monthly = daily.groupby("month", as_index=False).agg(
        days=("target_day", "nunique"), direction_accuracy=("direction_accuracy", "mean"),
        positive_recall=("positive_recall", "mean"), negative_recall=("negative_recall", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), all_negative_baseline=("all_negative_baseline", "mean"),
    )
    summary = {k: float(daily[k].mean()) for k in ("direction_accuracy", "positive_recall", "negative_recall", "balanced_accuracy", "all_negative_baseline")}
    summary["days"] = int(daily["target_day"].nunique())
    summary["route"] = "A_prequential_disagreement_router"
    summary["gain_vs_all_negative"] = summary["direction_accuracy"] - summary["all_negative_baseline"]
    daily.to_csv(output / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "router_training_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS", "route": "A_prequential_disagreement_router",
        "forecast_origin": "D-1 14:00", "router_training_labels": "historical strict OOS correctness <= D-2",
        "training_last_day": "per target day <= D-2", "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False, "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False, "candidate_variants": variants,
        "router_features": feature_cols, "history_days": args.history_days,
        "screen_range": [args.start, args.end], "pilot_only": True,
        "positive_slot_weight": args.positive_slot_weight,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--history-days", type=int, default=90)
    parser.add_argument("--min-history-days", type=int, default=60)
    parser.add_argument("--positive-slot-weight", type=float, default=1.0,
                        help="historical correctness weight for positive-spread slots; must be fixed before confirmation")
    parser.add_argument("--seed", type=int, default=20260823)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
