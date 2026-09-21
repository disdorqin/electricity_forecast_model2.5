from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.resolution import HOURLY  # noqa: E402
from source_contract import validate_deployable_prediction_source  # noqa: E402


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


def direction_metrics(y_true: np.ndarray, pred_direction: np.ndarray) -> dict[str, float | int]:
    yt = np.sign(np.asarray(y_true, float))
    yp = np.asarray(pred_direction, int)
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


def load_hourly_truth(path: Path, start: str, end: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="gb18030", usecols=["时刻", "日前电价", "实时电价"])
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw["target_day"] = raw["时刻"].map(HOURLY.business_day_from_timestamp)
    raw["hour_business"] = raw["时刻"].map(HOURLY.business_period_from_timestamp).astype(int)
    raw["hourly_spread"] = pd.to_numeric(raw["实时电价"], errors="coerce") - pd.to_numeric(raw["日前电价"], errors="coerce")
    raw = raw[(raw["target_day"] >= start) & (raw["target_day"] <= end)].copy()
    return raw[["target_day", "hour_business", "hourly_spread"]]


def vote_direction(values: pd.Series) -> int:
    signs = np.sign(pd.to_numeric(values, errors="coerce").to_numpy(float))
    pos = int((signs > 0).sum())
    neg = int((signs < 0).sum())
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    mean = float(np.nanmean(pd.to_numeric(values, errors="coerce")))
    return 1 if mean > 0 else (-1 if mean < 0 else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--hourly-canonical", default="data/24/canonical/shandong_pmos_hourly.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pred_path = ROOT / args.predictions
    outdir = ROOT / args.output
    source_manifest = validate_deployable_prediction_source(pred_path)
    p = pd.read_parquet(pred_path)
    p["market_date"] = p["market_date"].astype(str)
    if p.empty:
        raise ValueError("empty 96-point prediction table")
    required = {"market_date", "slot", "spread", "q50"}
    missing = required - set(p.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    p["hour_business"] = ((pd.to_numeric(p["slot"]) - 1) // 4 + 1).astype(int)

    hourly96 = p.groupby(["market_date", "hour_business"], as_index=False).agg(
        n_quarters=("slot", "size"),
        true96_mean_spread=("spread", "mean"),
        pred_mean=("q50", "mean"),
        pred_median=("q50", "median"),
        pred_q_min=("q50", "min"),
        pred_q_max=("q50", "max"),
        pred_q_std=("q50", lambda s: float(np.std(pd.to_numeric(s, errors="coerce"), ddof=0))),
    )
    vote = p.groupby(["market_date", "hour_business"])["q50"].apply(vote_direction).rename("pred_vote_direction").reset_index()
    hourly96 = hourly96.merge(vote, on=["market_date", "hour_business"], how="left", validate="one_to_one")
    if not (hourly96["n_quarters"] == 4).all():
        bad = hourly96.loc[hourly96["n_quarters"] != 4].head().to_dict("records")
        raise RuntimeError(f"incomplete 4x15min aggregation: {bad}")

    start = str(hourly96["market_date"].min())
    end = str(hourly96["market_date"].max())
    truth24 = load_hourly_truth(ROOT / args.hourly_canonical, start, end)
    complete_counts = truth24.groupby("target_day")["hourly_spread"].agg(["size", "count"])
    complete_days = complete_counts.index[(complete_counts["size"] == 24) & (complete_counts["count"] == 24)].astype(str).tolist()
    source_days = sorted(hourly96["market_date"].astype(str).unique())
    excluded_days = sorted(set(source_days) - set(complete_days))
    hourly96_eval = hourly96[hourly96["market_date"].astype(str).isin(complete_days)].copy()
    truth24_eval = truth24[truth24["target_day"].isin(complete_days)].copy()
    merged = hourly96_eval.rename(columns={"market_date": "target_day"}).merge(
        truth24_eval, on=["target_day", "hour_business"], how="inner", validate="one_to_one"
    )
    expected = len(complete_days) * 24
    if len(merged) != expected:
        raise RuntimeError(f"24 truth alignment incomplete: complete_days={len(complete_days)}, expected={expected}, merged={len(merged)}")

    merged["period"] = merged["hour_business"].map(HOURLY.infer_period)
    merged["pred_mean_direction"] = np.sign(merged["pred_mean"]).astype(int)
    merged["pred_median_direction"] = np.sign(merged["pred_median"]).astype(int)
    merged["true96_mean_direction"] = np.sign(merged["true96_mean_spread"]).astype(int)
    merged["true24_direction"] = np.sign(merged["hourly_spread"]).astype(int)

    truth_agreement = direction_metrics(
        merged["hourly_spread"].to_numpy(float), merged["true96_mean_direction"].to_numpy(int)
    )
    truth_value_mae = float(np.mean(np.abs(merged["true96_mean_spread"] - merged["hourly_spread"])))

    rows = []
    for name, col in (
        ("mean", "pred_mean_direction"),
        ("median", "pred_median_direction"),
        ("vote", "pred_vote_direction"),
    ):
        rows.append({"aggregation": name, "truth": "canonical_24", **direction_metrics(merged["hourly_spread"], merged[col])})
        rows.append({"aggregation": name, "truth": "aggregated_96", **direction_metrics(merged["true96_mean_spread"], merged[col])})
    summary = pd.DataFrame(rows)

    monthly_rows = []
    merged["month"] = merged["target_day"].str.slice(0, 7)
    for month, g in merged.groupby("month", sort=True):
        for name, col in (("mean", "pred_mean_direction"), ("median", "pred_median_direction"), ("vote", "pred_vote_direction")):
            monthly_rows.append({"month": month, "aggregation": name, **direction_metrics(g["hourly_spread"], g[col])})

    period_rows = []
    for period, g in merged.groupby("period", sort=False):
        for name, col in (("mean", "pred_mean_direction"), ("median", "pred_median_direction"), ("vote", "pred_vote_direction")):
            period_rows.append({"period": period, "aggregation": name, **direction_metrics(g["hourly_spread"], g[col])})

    atomic_parquet(outdir / "hourly_ledger.parquet", merged)
    atomic_csv(outdir / "summary.csv", summary)
    atomic_csv(outdir / "monthly_metrics.csv", pd.DataFrame(monthly_rows))
    atomic_csv(outdir / "period_metrics.csv", pd.DataFrame(period_rows))
    manifest = {
        "status": "complete",
        "experiment": "r1_96_to_24_simple_aggregation",
        "source_predictions": args.predictions,
        "source_contract": {"deployable_source": True, "forecast_origin": source_manifest["forecast_origin"]},
        "source_days": int(len(source_days)),
        "evaluated_complete_24_days": int(merged["target_day"].nunique()),
        "excluded_incomplete_24_truth_days": excluded_days,
        "source_96_rows": int(len(p)),
        "hourly_rows": int(len(merged)),
        "truth_alignment": {
            "canonical24_vs_mean96_direction": truth_agreement,
            "canonical24_vs_mean96_value_mae": truth_value_mae,
        },
        "production_chain_touched": False,
    }
    atomic_json(outdir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    print("\nPERIOD")
    print(pd.DataFrame(period_rows).to_string(index=False))


if __name__ == "__main__":
    main()
