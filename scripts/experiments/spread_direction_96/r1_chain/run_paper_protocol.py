from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from r1_core import R1QuantileModel, build_paper_features, load_shandong_96, regression_metrics


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/96/authoritative/pmos_96_全量.csv")
    ap.add_argument("--output", default="outputs/experiments/02_spread_96/spread_direction_96/r1_chain/paper_protocol_v1")
    ap.add_argument("--train-start", default="2024-01-01")
    ap.add_argument("--train-end", default="2024-09-30")
    ap.add_argument("--test-start", default="2024-12-01")
    ap.add_argument("--test-end", default="2024-12-31")
    ap.add_argument("--median-only", action="store_true", help="Fit q50 only for fast direction-accuracy evaluation")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[4]
    data_path = root / args.data
    outdir = root / args.output
    t0 = time.perf_counter()

    raw = load_shandong_96(data_path, args.train_start, args.test_end)
    data = build_paper_features(raw)
    train = data[(data["market_date"] >= args.train_start) & (data["market_date"] <= args.train_end)].copy()
    test = data[(data["market_date"] >= args.test_start) & (data["market_date"] <= args.test_end)].copy()
    if len(test) == 0:
        raise ValueError("empty test set")

    model = R1QuantileModel((0.50,)) if args.median_only else R1QuantileModel()
    model.fit(train)
    pred = model.predict_quantiles(test)
    metrics = regression_metrics(test["spread"].to_numpy(float), pred["q50"].to_numpy(float))
    pred["true_direction"] = np.sign(pred["spread"]).astype(int)
    pred["pred_direction"] = np.sign(pred["q50"]).astype(int)
    pred["direction_correct"] = pred["true_direction"] == pred["pred_direction"]

    period_rows = []
    for period_name, lo, hi in (("p01_32", 1, 32), ("p33_64", 33, 64), ("p65_96", 65, 96)):
        g = pred[pred["slot"].between(lo, hi)]
        m = regression_metrics(g["spread"].to_numpy(float), g["q50"].to_numpy(float))
        period_rows.append({"period": period_name, **m})

    daily = pred.groupby("market_date", as_index=False).agg(
        n=("direction_correct", "size"),
        direction_accuracy=("direction_correct", "mean"),
    )
    monthly_rows = []
    pred["month"] = pred["market_date"].astype(str).str.slice(0, 7)
    for month, g in pred.groupby("month", sort=True):
        m = regression_metrics(g["spread"].to_numpy(float), g["q50"].to_numpy(float))
        monthly_rows.append({"month": month, **m})

    atomic_parquet(outdir / "predictions.parquet", pred)
    atomic_csv(outdir / "period_metrics.csv", pd.DataFrame(period_rows))
    atomic_csv(outdir / "daily_metrics.csv", daily)
    atomic_csv(outdir / "monthly_metrics.csv", pd.DataFrame(monthly_rows))
    manifest = {
        "status": "complete",
        "experiment": "spread_direction_96_r1_paper_protocol",
        "dataset": "Shandong 96-point authoritative",
        "resolution": "15min / 96 points",
        "target": "RT-DA spread",
        "protocol": "R1 paper-style contemporaneous/lagged 15min protocol; NOT D-1 14:00 deployable",
        "train": [args.train_start, args.train_end],
        "test": [args.test_start, args.test_end],
        "median_only": bool(args.median_only),
        "metrics": metrics,
        "data_sha256": sha256(data_path),
        "python": platform.python_version(),
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(outdir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(pd.DataFrame(period_rows).to_string(index=False))


if __name__ == "__main__":
    main()
