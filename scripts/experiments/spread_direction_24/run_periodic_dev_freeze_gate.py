"""Development-frozen period-specific threshold gate.

Unlike a global threshold sweep, this pilot allows the three business periods to
use separate thresholds selected once on the development block. Confirmation is
strictly report-only. The source must already be strict OOS; this script does not
retrain the forecaster.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


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


def add_period(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "period" in out.columns:
        out["period_gate"] = out["period"].astype(str)
    elif "hour_business" in out.columns:
        h = pd.to_numeric(out["hour_business"], errors="coerce")
        out["period_gate"] = pd.cut(h, [0, 8, 16, 24], labels=["1_8", "9_16", "17_24"]).astype(str)
    elif "时刻" in out.columns:
        h = pd.to_datetime(out["时刻"], errors="coerce").dt.hour
        out["period_gate"] = pd.cut(h, [-1, 7, 15, 23], labels=["1_8", "9_16", "17_24"]).astype(str)
    else:
        raise RuntimeError("source must contain period/hour_business/时刻")
    return out


def choose(dev: pd.DataFrame, thresholds: list[float]) -> dict[str, float]:
    selected = {}
    for period, group in dev.groupby("period_gate", sort=True):
        candidates = []
        y = np.sign(group["label"].to_numpy(float)).astype(int)
        for threshold in thresholds:
            pred = np.where(group["score"] >= threshold, 1, -1)
            m = score(y, pred)
            candidates.append((m["balanced_accuracy"], m["positive_recall"], -abs(threshold), threshold))
        selected[str(period)] = max(candidates)[-1]
    return selected


def evaluate(frame: pd.DataFrame, thresholds: dict[str, float]) -> tuple[dict[str, float], pd.DataFrame]:
    frame = frame.copy()
    frame["selected_threshold"] = frame["period_gate"].map(thresholds)
    frame["pred"] = np.where(frame["score"] >= frame["selected_threshold"], 1, -1)
    y = np.sign(frame["label"].to_numpy(float)).astype(int)
    overall = score(y, frame["pred"].to_numpy(int))
    rows = []
    for period, group in frame.groupby("period_gate", sort=True):
        rows.append({"period": period, "threshold": thresholds[str(period)], **score(
            np.sign(group["label"].to_numpy(float)).astype(int), group["pred"].to_numpy(int)
        )})
    return overall, pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if args.variant and "variant" in frame.columns:
        frame = frame[frame["variant"].eq(args.variant)].copy()
    for col in (args.label_col, args.score_col):
        if col not in frame.columns:
            raise RuntimeError(f"source missing {col}")
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["target_day"] = pd.to_datetime(frame["target_day"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["target_day", args.label_col, args.score_col]).copy()
    if frame.empty or frame["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("empty source or final holdout touched")
    frame = frame.rename(columns={args.label_col: "label", args.score_col: "score"})
    frame = add_period(frame).sort_values(["target_day", "period_gate"])
    dev = frame[frame["target_day"].between(pd.Timestamp(args.dev_start), pd.Timestamp(args.dev_end))].copy()
    confirm = frame[frame["target_day"].between(pd.Timestamp(args.confirm_start), pd.Timestamp(args.confirm_end))].copy()
    if dev.empty or confirm.empty:
        raise RuntimeError("development or confirmation block empty")
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    selected = choose(dev, thresholds)
    dev_overall, dev_period = evaluate(dev, selected)
    confirm_overall, confirm_period = evaluate(confirm, selected)
    dev_period["phase"] = "development"
    confirm_period["phase"] = "confirmation"
    pd.concat([dev_period, confirm_period], ignore_index=True).to_csv(output / "period_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"phase": "development", **dev_overall}, {"phase": "confirmation", **confirm_overall}]).to_csv(output / "phase_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"period": p, "selected_threshold": t} for p, t in selected.items()]).to_csv(output / "selected_thresholds.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS", "route": args.route, "source": str(source),
        "forecast_origin": "D-1 14:00", "threshold_selected_on": "development labels only",
        "development_window": [args.dev_start, args.dev_end], "confirmation_window": [args.confirm_start, args.confirm_end],
        "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False, "final_holdout_touched": False,
        "selected_thresholds": selected, "threshold_candidates": thresholds,
        "selection_objective": "period balanced_accuracy then positive_recall",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"route": args.route, "variant": args.variant or "all", "selected_thresholds": selected, "dev": dev_overall, "confirm": confirm_overall}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--label-col", default="target_spread")
    parser.add_argument("--score-col", default="prob_positive")
    parser.add_argument("--dev-start", default="2026-04-01")
    parser.add_argument("--dev-end", default="2026-06-30")
    parser.add_argument("--confirm-start", default="2026-07-01")
    parser.add_argument("--confirm-end", default="2026-08-14")
    parser.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
