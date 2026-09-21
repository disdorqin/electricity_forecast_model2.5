from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aggregate_to_24 import load_hourly_truth, vote_direction  # noqa: E402
from source_contract import validate_deployable_prediction_source  # noqa: E402
from utils.resolution import HOURLY  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def direction_metrics(y_true: np.ndarray, pred_dir: np.ndarray) -> dict[str, float | int]:
    yt = np.sign(np.asarray(y_true, float))
    yp = np.asarray(pred_dir, int)
    eligible = yt != 0
    pos = yt > 0
    neg = yt < 0
    correct = yt == yp
    pa = float(correct[pos].mean()) if pos.any() else math.nan
    na = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "n_nonzero": int(eligible.sum()),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
    }


def build_features(pred_path: Path, truth_path: Path) -> pd.DataFrame:
    p = pd.read_parquet(pred_path)
    p["market_date"] = p["market_date"].astype(str)
    p["quarter_in_hour"] = ((pd.to_numeric(p["slot"]) - 1) % 4 + 1).astype(int)
    p["hour_business"] = ((pd.to_numeric(p["slot"]) - 1) // 4 + 1).astype(int)
    pv = p.pivot(index=["market_date", "hour_business"], columns="quarter_in_hour", values="q50")
    pv.columns = [f"q{i}" for i in pv.columns]
    pv = pv.reset_index()
    required_q = ["q1", "q2", "q3", "q4"]
    if pv[required_q].isna().any().any():
        raise RuntimeError(f"incomplete quarter predictions in {pred_path}")

    x = pv.copy()
    q = x[required_q]
    x["q_mean"] = q.mean(axis=1)
    x["q_median"] = q.median(axis=1)
    x["q_std"] = q.std(axis=1, ddof=0)
    x["q_min"] = q.min(axis=1)
    x["q_max"] = q.max(axis=1)
    x["q_range"] = x["q_max"] - x["q_min"]
    x["q_mean_abs"] = q.abs().mean(axis=1)
    x["q_max_abs"] = q.abs().max(axis=1)
    x["q_first"] = x["q1"]
    x["q_last"] = x["q4"]
    x["q_slope"] = x["q4"] - x["q1"]
    x["q_d12"] = x["q2"] - x["q1"]
    x["q_d23"] = x["q3"] - x["q2"]
    x["q_d34"] = x["q4"] - x["q3"]
    signs = np.sign(q.to_numpy(float))
    x["q_pos_count"] = (signs > 0).sum(axis=1)
    x["q_neg_count"] = (signs < 0).sum(axis=1)
    x["q_vote_margin"] = np.abs(x["q_pos_count"] - x["q_neg_count"])
    x["q_unanimous"] = ((x["q_pos_count"] == 4) | (x["q_neg_count"] == 4)).astype(int)
    x["q_sign_changes"] = (signs[:, 1:] != signs[:, :-1]).sum(axis=1)
    x["mean_direction"] = np.sign(x["q_mean"]).astype(int)
    x["median_direction"] = np.sign(x["q_median"]).astype(int)
    vote = p.groupby(["market_date", "hour_business"])["q50"].apply(vote_direction)
    x = x.merge(vote.rename("vote_direction").reset_index(), on=["market_date", "hour_business"], how="left", validate="one_to_one")

    dates = pd.to_datetime(x["market_date"])
    x["hour_sin"] = np.sin(2 * np.pi * (x["hour_business"] - 1) / 24)
    x["hour_cos"] = np.cos(2 * np.pi * (x["hour_business"] - 1) / 24)
    x["dow_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
    x["dow_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
    x["month_sin"] = np.sin(2 * np.pi * (dates.dt.month - 1) / 12)
    x["month_cos"] = np.cos(2 * np.pi * (dates.dt.month - 1) / 12)
    x["period_id"] = np.where(x["hour_business"] <= 8, 0, np.where(x["hour_business"] <= 16, 1, 2))
    x["period"] = x["hour_business"].map(HOURLY.infer_period)

    start, end = str(x["market_date"].min()), str(x["market_date"].max())
    truth = load_hourly_truth(truth_path, start, end)
    counts = truth.groupby("target_day")["hourly_spread"].agg(["size", "count"])
    complete_days = set(counts.index[(counts["size"] == 24) & (counts["count"] == 24)].astype(str))
    x = x[x["market_date"].isin(complete_days)].copy()
    truth = truth[truth["target_day"].isin(complete_days)].copy()
    x = x.rename(columns={"market_date": "target_day"}).merge(truth, on=["target_day", "hour_business"], how="inner", validate="one_to_one")
    x["label"] = (x["hourly_spread"] > 0).astype(int)
    x["true_direction"] = np.sign(x["hourly_spread"]).astype(int)
    return x


def lgb_model(seed: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        verbosity=-1,
        n_jobs=4,
        random_state=seed,
    )


def feature_cols() -> list[str]:
    return [
        "q1", "q2", "q3", "q4", "q_mean", "q_median", "q_std", "q_min", "q_max", "q_range",
        "q_mean_abs", "q_max_abs", "q_first", "q_last", "q_slope", "q_d12", "q_d23", "q_d34",
        "q_pos_count", "q_neg_count", "q_vote_margin", "q_unanimous", "q_sign_changes",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "period_id",
    ]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> dict[str, np.ndarray]:
    feats = feature_cols()
    y = train["label"].to_numpy(int)
    out: dict[str, np.ndarray] = {}

    logit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=seed)),
    ])
    logit.fit(train[feats], y)
    out["logistic"] = logit.predict_proba(test[feats])[:, 1]

    global_lgb = lgb_model(seed).fit(train[feats], y)
    out["lgb_global"] = global_lgb.predict_proba(test[feats])[:, 1]

    period_prob = np.full(len(test), np.nan, float)
    for period in ("1_8", "9_16", "17_24"):
        tr = train[train["period"].eq(period)]
        mask = test["period"].eq(period).to_numpy()
        if tr["label"].nunique() < 2:
            continue
        model = lgb_model(seed + {"1_8": 11, "9_16": 23, "17_24": 37}[period]).fit(tr[feats], tr["label"].to_numpy(int))
        period_prob[mask] = model.predict_proba(test.loc[mask, feats])[:, 1]
    if np.isnan(period_prob).any():
        raise RuntimeError("period model left NaN probabilities")
    out["lgb_period"] = period_prob
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-predictions", nargs="+", required=True)
    ap.add_argument("--test-predictions", required=True)
    ap.add_argument("--hourly-canonical", default="data/24/canonical/shandong_pmos_hourly.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-start", default="")
    ap.add_argument("--train-end", default="")
    ap.add_argument("--test-start", default="")
    ap.add_argument("--test-end", default="")
    args = ap.parse_args()

    truth_path = ROOT / args.hourly_canonical
    train_source_manifests = [validate_deployable_prediction_source(ROOT / p) for p in args.train_predictions]
    test_source_manifest = validate_deployable_prediction_source(ROOT / args.test_predictions)
    train_parts = [build_features(ROOT / p, truth_path) for p in args.train_predictions]
    train = pd.concat(train_parts, ignore_index=True).drop_duplicates(["target_day", "hour_business"], keep="last")
    test = build_features(ROOT / args.test_predictions, truth_path)
    if args.train_start:
        train = train[train["target_day"] >= args.train_start].copy()
    if args.train_end:
        train = train[train["target_day"] <= args.train_end].copy()
    if args.test_start:
        test = test[test["target_day"] >= args.test_start].copy()
    if args.test_end:
        test = test[test["target_day"] <= args.test_end].copy()
    # Never allow target-period overlap.
    overlap = set(train["target_day"]) & set(test["target_day"])
    if overlap:
        raise RuntimeError(f"train/test day overlap: {sorted(overlap)[:3]}")
    train = train[train["true_direction"].ne(0)].copy()
    test_eval = test[test["true_direction"].ne(0)].copy()

    probs = fit_predict(train, test_eval, args.seed)
    ledger = test_eval[["target_day", "hour_business", "period", "hourly_spread", "mean_direction", "median_direction", "vote_direction"]].copy()
    rows = []
    for base in ("mean", "median", "vote"):
        pred = ledger[f"{base}_direction"].to_numpy(int)
        rows.append({"model": base, **direction_metrics(ledger["hourly_spread"], pred)})
    for name, prob in probs.items():
        ledger[f"{name}_prob"] = prob
        ledger[f"{name}_direction"] = np.where(prob >= 0.5, 1, -1)
        rows.append({"model": name, **direction_metrics(ledger["hourly_spread"], ledger[f"{name}_direction"])})
    summary = pd.DataFrame(rows)

    monthly_rows = []
    ledger["month"] = ledger["target_day"].str.slice(0, 7)
    for month, g in ledger.groupby("month", sort=True):
        for model in ["mean", "median", "vote", *probs.keys()]:
            col = f"{model}_direction"
            monthly_rows.append({"month": month, "model": model, **direction_metrics(g["hourly_spread"], g[col])})

    period_rows = []
    for period, g in ledger.groupby("period", sort=False):
        for model in ["mean", "median", "vote", *probs.keys()]:
            col = f"{model}_direction"
            period_rows.append({"period": period, "model": model, **direction_metrics(g["hourly_spread"], g[col])})

    outdir = ROOT / args.output
    atomic_parquet(outdir / "ledger.parquet", ledger)
    atomic_csv(outdir / "summary.csv", summary)
    atomic_csv(outdir / "monthly_metrics.csv", pd.DataFrame(monthly_rows))
    atomic_csv(outdir / "period_metrics.csv", pd.DataFrame(period_rows))
    manifest = {
        "status": "complete",
        "experiment": "r1_96_to_24_learned_aggregator",
        "train_prediction_sources": args.train_predictions,
        "test_prediction_source": args.test_predictions,
        "source_contract": {"deployable_sources_only": True, "train_forecast_origins": sorted({m["forecast_origin"] for m in train_source_manifests}), "test_forecast_origin": test_source_manifest["forecast_origin"]},
        "train_days": int(train["target_day"].nunique()),
        "test_days": int(test_eval["target_day"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test_eval)),
        "features": feature_cols(),
        "threshold": 0.5,
        "date_filters": {"train_start": args.train_start, "train_end": args.train_end, "test_start": args.test_start, "test_end": args.test_end},
        "production_chain_touched": False,
    }
    atomic_json(outdir / "manifest.json", manifest)
    print(summary.sort_values("direction_accuracy", ascending=False).to_string(index=False))
    print("\nPERIOD")
    print(pd.DataFrame(period_rows).sort_values(["period", "direction_accuracy"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
