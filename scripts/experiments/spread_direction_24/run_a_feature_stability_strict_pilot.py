"""Cycle 17 A pilot: strict feature-package stability under the D-1 14:00 contract.

This is a small walk-forward classifier probe.  It compares the retained P6+SD20
package with causal forecast-regime descriptors (F7), dropping only features whose
missingness is too high inside the current strict training window.  It is not a
production model and does not read the final holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    add_similar_day_features,
    dedupe,
    p6_features,
    strict_train_days,
)


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def metrics(frame: pd.DataFrame) -> dict:
    y = frame["y_true"].to_numpy(int)
    p = frame["predicted_direction"].to_numpy(int)
    pos, neg = y == 1, y == -1
    pr = float((p[pos] == 1).mean()) if pos.any() else 0.0
    nr = float((p[neg] == -1).mean()) if neg.any() else 0.0
    return {
        "days": int(frame["target_day"].nunique()),
        "n_slots": int(len(frame)),
        "direction_accuracy": float((p == y).mean()),
        "positive_recall": pr,
        "negative_recall": nr,
        "balanced_accuracy": float((pr + nr) / 2.0),
        "all_negative_baseline": float(neg.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--train-window", type=int, default=90)
    parser.add_argument("--missing-rate", type=float, default=0.10)
    parser.add_argument("--packages", default="A_p6_sd20,A_p6_sd20_f7_clean")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--positive-weight", type=float, default=1.0,
                        help="additional causal training weight for positive labels; select only on development data")
    parser.add_argument("--period-specialist", action="store_true",
                        help="fit independent classifiers for 1-8, 9-16 and 17-24 slots")
    parser.add_argument("--positive-magnitude-weight", type=float, default=0.0,
                        help="extra weight ramp for large positive training spreads; 0 disables it")
    parser.add_argument("--two-stage-positive", action="store_true",
                        help="fit positive-spike and regular-positive-vs-negative stages causally")
    parser.add_argument("--spike-quantile", type=float, default=0.75,
                        help="historical positive-spread quantile defining the spike state")
    parser.add_argument("--forecast-anomaly", action="store_true",
                        help="add same-slot forecast anomaly/z-score features using only D-2-or-earlier forecasts")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end >= FINAL_HOLDOUT_START:
        raise RuntimeError("fresh final holdout remains sealed")
    cube = args.cube.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    slot["target_day"] = pd.to_datetime(slot["target_day"]).dt.normalize()
    anomaly_features = []
    if args.forecast_anomaly:
        # Hourly-only 24-point research feature: target-day forecast compared
        # with same-slot forecast history. The two-row shift excludes D-1 and D.
        slot = slot.sort_values(["hour_business", "target_day"]).copy()
        forecast_cols = [c for c in json.loads((cube / "feature_groups.json").read_text(encoding="utf-8")).get("F2", []) if c in slot.columns]
        for col in forecast_cols:
            shifted = slot.groupby("hour_business", sort=False)[col].transform(lambda s: s.shift(2))
            med = shifted.groupby(slot["hour_business"], sort=False).transform(lambda s: s.rolling(28, min_periods=7).median())
            std = shifted.groupby(slot["hour_business"], sort=False).transform(lambda s: s.rolling(28, min_periods=7).std())
            safe_std = std.replace(0, np.nan)
            stem = f"anom_{col}"
            slot[f"{stem}_delta"] = pd.to_numeric(slot[col], errors="coerce") - med
            slot[f"{stem}_z"] = slot[f"{stem}_delta"] / safe_std
            anomaly_features.extend([f"{stem}_delta", f"{stem}_z"])
        slot = slot.sort_index()
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    base = p6_features(groups)
    slot, sd = add_similar_day_features(slot, groups, k_values=(20,), lookback_days=365)
    audit = sd["audit"]
    if audit.empty or not audit["causal_ok"].all():
        raise RuntimeError("Similar-Day causal audit failed")
    if (pd.to_datetime(audit["latest_candidate_day"]) > pd.to_datetime(audit["required_latest_candidate_le"])).any():
        raise RuntimeError("Similar-Day candidate exceeds D-2")
    sd20 = [c for c in sd["features"] if c.startswith("sd20_")]
    f7 = list(groups.get("F7", []))
    f6 = list(groups.get("F6", []))
    f5 = list(groups.get("F5", []))
    f8 = list(groups.get("F8", []))
    packages = {
        "A_p6_sd20": dedupe(base + sd20),
        "A_p6_sd20_forecast_anomaly": dedupe(base + sd20 + anomaly_features),
        "A_p6_sd20_f7_clean": dedupe(base + sd20 + f7),
        "A_p6_sd20_f6_uncert": dedupe(base + sd20 + f6),
        "A_p6_sd20_f5_error": dedupe(base + sd20 + f5),
        "A_p6_sd20_f5_error_f7_clean": dedupe(base + sd20 + f5 + f7),
        "A_p6_sd20_f8_rawctx": dedupe(base + sd20 + f8),
    }
    requested = [x.strip() for x in args.packages.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(packages))
    if unknown:
        raise RuntimeError(f"unknown packages {unknown}")
    all_days = sorted(slot["target_day"].dropna().unique())
    target_days = [d for d in all_days if start <= d <= end]
    rows, feature_audits = [], []
    for package in requested:
        features = packages[package]
        for target_day in target_days:
            train_days = strict_train_days([d.strftime("%Y-%m-%d") for d in all_days], target_day.strftime("%Y-%m-%d"), args.train_window)
            train_days = [pd.Timestamp(d) for d in train_days]
            if len(train_days) < min(60, args.train_window):
                continue
            if max(train_days) > target_day - pd.Timedelta(days=2):
                raise RuntimeError(f"strict training boundary violated for {target_day.date()}")
            tr = slot[slot["target_day"].isin(train_days)].copy()
            te = slot[slot["target_day"].eq(target_day)].sort_values("hour_business").copy()
            if len(te) != 24:
                raise RuntimeError(f"{target_day.date()}: expected 24 target slots, got {len(te)}")
            selected = list(features)
            dropped = []
            if package.endswith("f7_clean"):
                missing_rate = tr[features].apply(pd.to_numeric, errors="coerce").isna().mean()
                selected = [c for c in features if float(missing_rate[c]) <= args.missing_rate]
                dropped = [c for c in features if c not in selected]
            yte = (te["target_spread"].to_numpy(float) >= 0).astype(int)
            prob = np.full(len(te), 0.5, dtype=float)
            blocks = [("all", np.ones(len(te), dtype=bool))]
            if args.period_specialist:
                h = te["hour_business"].to_numpy(int)
                blocks = [("1_8", (h >= 1) & (h <= 8)), ("9_16", (h >= 9) & (h <= 16)), ("17_24", (h >= 17) & (h <= 24))]
            for block_name, te_mask in blocks:
                htr = tr["hour_business"].to_numpy(int)
                if block_name == "1_8": tr_mask = (htr >= 1) & (htr <= 8)
                elif block_name == "9_16": tr_mask = (htr >= 9) & (htr <= 16)
                elif block_name == "17_24": tr_mask = (htr >= 17) & (htr <= 24)
                else: tr_mask = np.ones(len(tr), dtype=bool)
                tr_block, te_block = tr.loc[tr_mask], te.loc[te_mask]
                xtr = tr_block[selected].apply(pd.to_numeric, errors="coerce")
                xte = te_block[selected].apply(pd.to_numeric, errors="coerce")
                med = xtr.median().fillna(0.0)
                xtr, xte = xtr.fillna(med), xte.fillna(med)
                spread_tr = tr_block["target_spread"].to_numpy(float)

                def fit_binary(labels: np.ndarray, magnitude_values: np.ndarray | None = None) -> np.ndarray:
                    pos_n, neg_n = max(int(labels.sum()), 1), max(int((1 - labels).sum()), 1)
                    total = pos_n + neg_n
                    weights = {0: total / (2.0 * neg_n), 1: total / (2.0 * pos_n) * args.positive_weight}
                    model = lgb.LGBMClassifier(
                        objective="binary", class_weight=weights, n_estimators=args.n_estimators,
                        learning_rate=0.04, num_leaves=31, min_child_samples=35,
                        reg_lambda=1.0, random_state=args.seed, n_jobs=4, verbosity=-1,
                    )
                    sample_weight = np.ones(len(labels), dtype=float)
                    if args.positive_magnitude_weight > 0 and magnitude_values is not None and np.any(labels == 1):
                        positive_values = np.maximum(magnitude_values, 0.0)
                        positive_scale = max(float(np.nanquantile(positive_values[labels == 1], 0.90)), 1e-6)
                        sample_weight[labels == 1] += args.positive_magnitude_weight * np.clip(
                            positive_values[labels == 1] / positive_scale, 0.0, 1.0
                        )
                    model.fit(xtr, labels, sample_weight=sample_weight)
                    return model.predict_proba(xte)[:, 1]

                ytr = (spread_tr >= 0).astype(int)
                if args.two_stage_positive:
                    positive_values = spread_tr[spread_tr > 0]
                    if len(positive_values) < 10:
                        raise RuntimeError(f"{target_day.date()} {block_name}: too few positive labels for two-stage model")
                    spike_cut = float(np.nanquantile(positive_values, args.spike_quantile))
                    spike_y = (spread_tr >= spike_cut).astype(int)
                    regular_y = ((spread_tr > 0) & (spread_tr < spike_cut)).astype(int)
                    p_spike = fit_binary(spike_y)
                    p_regular = fit_binary(regular_y)
                    prob[te_mask] = p_spike + (1.0 - p_spike) * p_regular
                else:
                    prob[te_mask] = fit_binary(ytr, spread_tr)
            rows.append(pd.DataFrame({
                "target_day": target_day.date().isoformat(),
                "hour_business": te["hour_business"].to_numpy(int),
                "variant": package,
                "y_true": yte * 2 - 1,
                "predicted_direction": np.where(prob >= 0.5, 1, -1),
                "prob_positive": prob,
                "training_last_day": max(train_days).date().isoformat(),
            }))
            feature_audits.append({
                "target_day": target_day.date().isoformat(), "variant": package,
                "training_last_day": max(train_days).date().isoformat(),
                "feature_count": len(features), "selected_feature_count": len(selected),
                "dropped_feature_count": len(dropped), "dropped_features": ",".join(dropped),
                "latest_candidate_day_max": str(audit.loc[audit["target_day"].eq(target_day.strftime("%Y-%m-%d")), "latest_candidate_day"].max()),
            })
    ledger = pd.concat(rows, ignore_index=True)
    summary_rows = []
    for variant, group in ledger.groupby("variant", sort=False):
        summary_rows.append({"variant": variant, **metrics(group)})
        for month, month_group in group.assign(month=group["target_day"].str[:7]).groupby("month"):
            summary_rows.append({"variant": variant, "month": month, **metrics(month_group)})
    summary = pd.DataFrame(summary_rows)
    ledger.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_audits).to_csv(out / "feature_selection_audit.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "similar_day_causal_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS", "experiment_status": "CANDIDATE",
        "route": "A_strict_DSA", "forecast_origin": "D-1 14:00",
        "training_last_day": "per target day <= D-2", "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False, "d1_post14_spread_as_feature": False,
        "similar_day_latest_candidate": "per target day <= D-2", "final_holdout_touched": False,
        "screen_range": [args.start, args.end], "train_window": args.train_window,
        "missing_rate_filter": args.missing_rate, "positive_weight": args.positive_weight,
        "period_specialist": args.period_specialist,
        "positive_magnitude_weight": args.positive_magnitude_weight,
        "two_stage_positive": args.two_stage_positive, "spike_quantile": args.spike_quantile,
        "forecast_anomaly": args.forecast_anomaly,
        "forecast_anomaly_features": anomaly_features,
        "packages": {k: v for k, v in packages.items() if k in requested},
        "feature_source": str(cube),
        "f5_contract": "historical forecast errors only; D-2 or earlier, no target-day realized values",
        "note": "low-cost pilot only; no production integration",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[summary["month"].isna() if "month" in summary else slice(None)].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
