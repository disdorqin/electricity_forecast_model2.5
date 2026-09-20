from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from common import DEFAULT_96, atomic_csv, atomic_json, aggregate_96_to_hourly, load_shandong_96  # noqa: E402

PAPER_AUC = {
    "logistic": {-30: 0.710, -45: 0.745, -60: 0.765},
    "random_forest": {-30: 0.722, -45: 0.751, -60: 0.766},
    "gradient_boosting": {-30: 0.722, -45: 0.755, -60: 0.769},
    "dnn": {-30: 0.700, -45: 0.723, -60: 0.748},
}
THRESHOLDS = [-30.0, -45.0, -60.0]


def prepare_hourly(path: Path) -> pd.DataFrame:
    q = load_shandong_96("2022-01-01", "2026-08-17", path)
    h = aggregate_96_to_hourly(q)
    supply_cols = [c for c in ["地方电厂出力预测", "外电预测", "风电预测", "光伏预测", "核电预测", "自备电厂预测", "试验机组预测"] if c in h]
    supply = h[supply_cols].sum(axis=1).replace(0, np.nan)
    h["load_grid"] = h["直调负荷预测"] / supply
    h["load_grid_sq"] = h["load_grid"] ** 2
    h["price_error_sq"] = (h["实时出清价格"] - h["日前出清价格"]) ** 2
    h["load_error_sq"] = (h["直调负荷实际"] - h["直调负荷预测"]) ** 2
    return h.sort_values("timestamp").reset_index(drop=True)


def daily_backward(h: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    target_days = sorted(h["market_date"].unique())
    for day in target_days:
        d = pd.Timestamp(day)
        prediction_time = d - pd.Timedelta(days=2) + pd.Timedelta(hours=18)
        start = prediction_time - pd.Timedelta(hours=24)
        hist = h[(h["timestamp"] > start) & (h["timestamp"] <= prediction_time)]
        if len(hist) < 20:
            continue
        rows.append({
            "market_date": day,
            "past_spikes": float((hist["dart_da_minus_rt"] < threshold).sum()),
            "past_price_error": float(hist["price_error_sq"].sum()),
            "past_load_error": float(hist["load_error_sq"].sum()),
        })
    return pd.DataFrame(rows)


def model_features(h: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[str]]:
    b = daily_backward(h, threshold)
    d = h.merge(b, on="market_date", how="left")
    # Paper logistic uses month/hour buckets and load/grid squared term.
    d["month_bucket"] = pd.cut(d["month"], bins=[0, 2, 5, 9, 12], labels=False, include_lowest=True)
    hour = d["hour_business"] - 1
    d["hour_bucket"] = pd.cut(hour, bins=[-1, 5, 10, 13, 16, 19, 23], labels=False, include_lowest=True)
    d["target"] = (d["dart_da_minus_rt"] < threshold).astype(int)
    # Exact NYISO HDD/CDD and transfer-capacity inputs are unavailable locally.
    features = ["load_grid", "load_grid_sq", "month_bucket", "hour_bucket", "is_weekend", "past_spikes", "past_price_error", "past_load_error"]
    return d.dropna(subset=features).reset_index(drop=True), features


def models(seed: int = 42):
    return {
        "logistic": Pipeline([("scale", StandardScaler()), ("m", LogisticRegression(max_iter=1000, class_weight=None, random_state=seed))]),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=4, random_state=seed),
        "gradient_boosting": LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=30, verbosity=-1, n_jobs=4, random_state=seed),
        "dnn": Pipeline([("scale", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, early_stopping=True, validation_fraction=0.15, random_state=seed))]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_96.relative_to(Path(__file__).resolve().parents[4])))
    ap.add_argument("--output", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/r2_dart_spikes")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    h = prepare_hourly(root / args.data)
    rows = []
    preds = []
    for thr in THRESHOLDS:
        d, features = model_features(h, thr)
        # Paper: initial 3 years, expanding one year at a time. Local analogous split: 2022-24 -> 2025, 2022-25 -> 2026 YTD.
        for test_year in [2025, 2026]:
            train = d[pd.to_datetime(d["market_date"]).dt.year < test_year]
            test = d[pd.to_datetime(d["market_date"]).dt.year == test_year]
            if len(train) < 1000 or len(test) < 100:
                continue
            for name, m in models().items():
                m.fit(train[features], train["target"])
                prob = m.predict_proba(test[features])[:, 1]
                y = test["target"].to_numpy(int)
                aucv = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else math.nan
                ll = float(-log_loss(y, prob, labels=[0, 1]))
                rows.append({
                    "threshold": thr, "test_year": test_year, "model": name,
                    "n_train": len(train), "n_test": len(test), "event_rate": float(y.mean()),
                    "auc": aucv, "avg_loglik": ll, "paper_aggregated_auc": PAPER_AUC[name][int(thr)],
                })
                p = test[["market_date", "hour_business", "timestamp", "dart_da_minus_rt"]].copy()
                p["threshold"] = thr; p["model"] = name; p["prob_spike"] = prob; p["target"] = y
                preds.append(p)
    result = pd.DataFrame(rows)
    atomic_csv(out / "metrics.csv", result)
    if preds:
        pd.concat(preds, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    agg = result.groupby(["threshold", "model"], as_index=False).apply(
        lambda g: pd.Series({"auc_mean": np.average(g["auc"], weights=g["n_test"]), "paper_auc": g["paper_aggregated_auc"].iloc[0]})
    ).reset_index(drop=True)
    atomic_csv(out / "aggregate_metrics.csv", agg)
    manifest = {
        "paper": "Foreseeing the worst: Forecasting electricity DART spikes",
        "paper_exact_protocol": {"market": "NYISO Long Island", "resolution": "hourly", "dart": "DA-RT", "thresholds": THRESHOLDS, "models": list(models()), "evaluation": ["AUC", "average log-likelihood"], "expanding_window": "3-year initial then annual"},
        "fidelity": "algorithm-family and timing reproduction on Shandong analog; result-level paper replication is impossible without the paper's NYISO grid-transfer and OpenWeather forecast archive (HDD/CDD).",
        "local_substitutions": {"load_grid": "Shandong forecast load / forecast supply proxy", "HDD_CDD": "unavailable and intentionally not fabricated", "backward_features": "past spikes, DA price error, DA load error reproduced"},
        "paper_targets_auc": PAPER_AUC,
        "local_aggregate": agg.to_dict("records"),
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(out / "manifest.json", manifest)
    print(result.sort_values(["threshold", "test_year", "auc"], ascending=[False, True, False]).to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
