"""Strict prequential period calibration for the positive-spread event.

This experiment is deliberately a post-model correction only.  For each target
day and business period it fits a one-dimensional calibrator on historical OOS
probabilities whose labels are complete by D-2, then freezes a threshold before
scoring D.  It never reads target-day labels during fitting or threshold choice.
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


def fit_transform(method: str, train_p: np.ndarray, train_y: np.ndarray, query_p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(train_p, float), 1e-6, 1 - 1e-6)
    q = np.clip(np.asarray(query_p, float), 1e-6, 1 - 1e-6)
    if np.unique(train_y).size < 2:
        return q
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, train_y)
        return np.asarray(model.predict(q), float)
    if method == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        z = np.log(p / (1 - p)).reshape(-1, 1)
        model.fit(z, train_y)
        return model.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1]
    raise ValueError(method)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pos, neg = y == 1, y == -1
    pr = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    nr = float((pred[neg] == -1).mean()) if neg.any() else float("nan")
    return {
        "n": int(len(y)), "direction_accuracy": float((pred == y).mean()),
        "positive_recall": pr, "negative_recall": nr,
        "balanced_accuracy": float(np.nanmean([pr, nr])),
        "all_negative_baseline": float(neg.mean()),
    }


def choose_threshold(y: np.ndarray, p: np.ndarray, candidates: np.ndarray) -> tuple[float, dict[str, float]]:
    best = None
    for t in candidates:
        pred = np.where(p >= t, 1, -1)
        m = metrics(y, pred)
        # Balanced first, then positive recall, then raw accuracy.  This prevents
        # the event specialist from silently becoming an all-negative rule.
        key = (m["balanced_accuracy"], m["positive_recall"], m["direction_accuracy"], -abs(float(t) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t), m)
    assert best is not None
    return best[1], best[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--variant")
    ap.add_argument("--label-col", default="target_spread")
    ap.add_argument("--score-col", default="prob_positive")
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--methods", default="platt,isotonic")
    ap.add_argument("--windows", default="30,60,120")
    ap.add_argument("--min-history-days", type=int, default=30)
    ap.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    args = ap.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if args.variant and "variant" in frame:
        frame = frame[frame["variant"].eq(args.variant)].copy()
    required = {"target_day", args.label_col, args.score_col, "hour_business"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"source missing columns: {sorted(missing)}")
    frame["target_day"] = pd.to_datetime(frame["target_day"], errors="coerce").dt.normalize()
    frame[args.label_col] = pd.to_numeric(frame[args.label_col], errors="coerce")
    frame[args.score_col] = pd.to_numeric(frame[args.score_col], errors="coerce")
    frame = frame.dropna(subset=["target_day", args.label_col, args.score_col]).copy()
    frame["y"] = np.sign(frame[args.label_col].to_numpy(float)).astype(int)
    frame = frame[frame["y"] != 0].copy()
    frame["period"] = pd.cut(frame["hour_business"].astype(int), [0, 8, 16, 24], labels=["1_8", "9_16", "17_24"])
    if frame.empty or frame["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("empty source or final holdout touched")
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    windows = [int(x) for x in args.windows.split(",") if x.strip()]
    thresholds = np.array([float(x) for x in args.thresholds.split(",") if x.strip()], dtype=float)
    all_days = sorted(frame["target_day"].unique())
    targets = [d for d in all_days if pd.Timestamp(args.start) <= d <= pd.Timestamp(args.end)]
    rows, audits = [], []
    for method in methods:
        for window in windows:
            name = f"period_{method}_w{window}"
            for day in targets:
                cutoff = day - pd.Timedelta(days=2)
                hist_days = [d for d in all_days if d <= cutoff][-window:]
                if len(hist_days) < args.min_history_days:
                    continue
                current = frame[frame.target_day.eq(day)].copy()
                for period in ["1_8", "9_16", "17_24"]:
                    hist = frame[frame.target_day.isin(hist_days) & frame.period.eq(period)]
                    cur = current[current.period.eq(period)].sort_values("hour_business")
                    if len(cur) == 0 or hist["y"].nunique() < 2:
                        continue
                    y_hist = (hist.y.to_numpy(int) > 0).astype(int)
                    p_hist = fit_transform(method, hist[args.score_col].to_numpy(float), y_hist, hist[args.score_col].to_numpy(float))
                    threshold, hist_metric = choose_threshold(hist.y.to_numpy(int), p_hist, thresholds)
                    p_cur = fit_transform(method, hist[args.score_col].to_numpy(float), y_hist, cur[args.score_col].to_numpy(float))
                    pred = np.where(p_cur >= threshold, 1, -1)
                    for i, (_, r) in enumerate(cur.iterrows()):
                        rows.append({"target_day": day.date().isoformat(), "hour_business": int(r.hour_business),
                                     "period": period, "variant": name, "y_true": int(r.y),
                                     "p_calibrated": float(p_cur[i]), "pred": int(pred[i]), "threshold": threshold,
                                     "training_last_day": cutoff.date().isoformat()})
                    audits.append({"target_day": day.date().isoformat(), "period": period, "variant": name,
                                   "latest_training_day": max(hist_days).date().isoformat(),
                                   "required_latest_day": cutoff.date().isoformat(),
                                   "strict_ok": bool(max(hist_days) <= cutoff), "history_days": len(hist_days),
                                   "positive_rate_history": float(y_hist.mean()), "threshold": threshold,
                                   "history_balanced_accuracy": hist_metric["balanced_accuracy"]})
    pred = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if pred.empty or not audit["strict_ok"].all():
        raise RuntimeError("no valid output or D-2 audit failed")
    pred["month"] = pred.target_day.str[:7]
    summary = []
    for v, g in pred.groupby("variant", sort=False):
        summary.append({"variant": v, **metrics(g.y_true.to_numpy(int), g.pred.to_numpy(int))})
    monthly = []
    for (v, m), g in pred.groupby(["variant", "month"], sort=False):
        monthly.append({"variant": v, "month": m, **metrics(g.y_true.to_numpy(int), g.pred.to_numpy(int))})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly).to_csv(out / "monthly.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "period_calibration_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {"status": "STRICT/PASS", "route": args.route, "source": str(source),
                "forecast_origin": "D-1 14:00", "training_last_day": "per target day <= D-2",
                "periods": ["1_8", "9_16", "17_24"], "calibration_labels": "historical OOS only <= D-2",
                "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
                "d1_post14_spread_as_feature": False, "final_holdout_touched": False,
                "methods": methods, "windows": windows, "pilot_only": True}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
