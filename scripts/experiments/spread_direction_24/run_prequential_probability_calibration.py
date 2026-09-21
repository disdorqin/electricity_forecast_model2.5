"""Prequential probability calibration with strict D-2 history.

For every target day D, calibration and threshold selection consume only historical
strict OOS rows with target_day <= D-2. The current target labels are never used to
fit the calibrator or select the threshold. This is the Cycle 14 alternative to
repeatedly tuning one fixed development/confirmation block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")
DEFAULT_THRESHOLDS = np.arange(0.30, 0.701, 0.025)


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pos, neg = y > 0, y < 0
    pr = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    nr = float((pred[neg] == -1).mean()) if neg.any() else float("nan")
    return {
        "slots": int(len(y)), "direction_accuracy": float((pred == y).mean()),
        "positive_recall": pr, "negative_recall": nr,
        "balanced_accuracy": float(np.nanmean([pr, nr])),
        "all_negative_baseline": float(neg.mean()),
    }


def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    if np.unique(y).size < 2:
        return None
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y)
        return model
    if method == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return model
    raise ValueError(f"unknown calibrator: {method}")


def transform(model, method: str, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    if model is None:
        return p
    if method == "isotonic":
        return np.asarray(model.predict(p), float)
    return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def choose_threshold(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> tuple[float, dict[str, float]]:
    best = None
    for threshold in thresholds:
        pred = np.where(p >= threshold, 1, -1)
        metric = score(y, pred)
        key = (metric["balanced_accuracy"], metric["positive_recall"], -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold, metric)
    assert best is not None
    return float(best[1]), best[2]


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if args.variant and "variant" in frame.columns:
        frame = frame[frame["variant"].eq(args.variant)].copy()
    required = {"target_day", args.label_col, args.score_col}
    if required.difference(frame.columns):
        raise RuntimeError(f"source missing columns: {sorted(required.difference(frame.columns))}")
    frame["target_day"] = pd.to_datetime(frame["target_day"], errors="coerce").dt.normalize()
    frame[args.label_col] = pd.to_numeric(frame[args.label_col], errors="coerce")
    frame[args.score_col] = pd.to_numeric(frame[args.score_col], errors="coerce")
    if "training_last_day" in frame.columns:
        frame["training_last_day"] = pd.to_datetime(frame["training_last_day"], errors="coerce").dt.normalize()
        valid = frame["training_last_day"].isna() | (frame["training_last_day"] <= frame["target_day"] - pd.Timedelta(days=2))
        if not valid.all():
            raise RuntimeError("source training_last_day violates D-2")
    frame = frame.dropna(subset=["target_day", args.label_col, args.score_col]).sort_values("target_day")
    if frame.empty or frame["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("empty source or final holdout touched")
    frame["y"] = np.sign(frame[args.label_col].to_numpy(float)).astype(int)
    frame = frame[frame["y"] != 0].copy()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()] if args.thresholds else DEFAULT_THRESHOLDS.tolist()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    all_days = sorted(frame["target_day"].unique())
    target_days = [d for d in all_days if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    daily_rows, prediction_rows, audits = [], [], []
    for method in methods:
        for window in windows:
            variant_name = f"{method}_w{window}"
            for target_day in target_days:
                cutoff = target_day - pd.Timedelta(days=2)
                history_days = [d for d in all_days if d <= cutoff][-window:]
                if len(history_days) < args.min_history_days:
                    continue
                history = frame[frame["target_day"].isin(history_days)]
                current = frame[frame["target_day"] == target_day]
                y_hist = (history["y"].to_numpy(int) > 0).astype(int)
                y_current = history["y"].to_numpy(int)
                calibrator = fit_calibrator(method, history[args.score_col].to_numpy(float), y_hist)
                p_hist = transform(calibrator, method, history[args.score_col].to_numpy(float))
                threshold, hist_metric = choose_threshold(y_current, p_hist, thresholds)
                p_current = transform(calibrator, method, current[args.score_col].to_numpy(float))
                y_true = current["y"].to_numpy(int)
                pred = np.where(p_current >= threshold, 1, -1)
                m = score(y_true, pred)
                m.update({"target_day": target_day.date().isoformat(), "variant": variant_name, "threshold": threshold})
                daily_rows.append(m)
                for i, (_, row) in enumerate(current.iterrows()):
                    prediction_rows.append({
                        "target_day": target_day.date().isoformat(), "variant": variant_name,
                        "y_true": int(y_true[i]), "p_raw": float(row[args.score_col]),
                        "p_calibrated": float(p_current[i]), "pred": int(pred[i]),
                        "threshold": threshold, "training_last_day": cutoff.date().isoformat(),
                    })
                audits.append({
                    "target_day": target_day.date().isoformat(), "variant": variant_name,
                    "training_last_day": max(history_days).date().isoformat(),
                    "required_last_day": cutoff.date().isoformat(), "strict_ok": bool(max(history_days) <= cutoff),
                    "history_days": len(history_days), "threshold": threshold,
                    "history_balanced_accuracy": hist_metric["balanced_accuracy"],
                })
    if not daily_rows:
        raise RuntimeError("no target days passed prequential history gate")
    daily = pd.DataFrame(daily_rows)
    predictions = pd.DataFrame(prediction_rows)
    audit = pd.DataFrame(audits)
    if not audit["strict_ok"].all():
        raise RuntimeError("prequential calibration strict audit failed")
    daily["month"] = daily["target_day"].str[:7]
    monthly = daily.groupby(["variant", "month"], as_index=False).agg(
        days=("target_day", "nunique"), direction_accuracy=("direction_accuracy", "mean"),
        positive_recall=("positive_recall", "mean"), negative_recall=("negative_recall", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), all_negative_baseline=("all_negative_baseline", "mean"),
    )
    robustness = monthly.groupby("variant", as_index=False).agg(
        months=("month", "nunique"), mean_month_acc=("direction_accuracy", "mean"),
        mean_month_bal=("balanced_accuracy", "mean"), mean_positive_recall=("positive_recall", "mean"),
        mean_negative_recall=("negative_recall", "mean"), mean_all_negative=("all_negative_baseline", "mean"),
    )
    robustness["mean_gain_vs_all_negative"] = robustness["mean_month_acc"] - robustness["mean_all_negative"]
    summary_rows = []
    for variant, group in predictions.groupby("variant", sort=True):
        summary = score(group["y_true"].to_numpy(int), group["pred"].to_numpy(int))
        summary.update({"variant": variant, "days": int(group["target_day"].nunique())})
        summary_rows.append(summary)
    daily.to_csv(output / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(output / "robustness.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "calibration_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS", "route": args.route, "source": str(source),
        "forecast_origin": "D-1 14:00", "calibration_labels": "historical strict OOS <= D-2",
        "training_last_day": "per target day <= D-2", "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False, "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False, "methods": methods, "windows": windows,
        "threshold_candidates": thresholds, "screen_range": [args.start, args.end], "pilot_only": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(robustness.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--label-col", default="target_spread")
    parser.add_argument("--score-col", default="prob_positive")
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--windows", default="60,120")
    parser.add_argument("--min-history-days", type=int, default=30)
    parser.add_argument("--methods", default="isotonic,platt")
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
