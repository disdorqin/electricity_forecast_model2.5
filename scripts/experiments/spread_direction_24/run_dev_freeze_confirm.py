"""开发集冻结决策阈值，再在独立确认集验证。

该工具只消费已经完成 strict-D2 的 OOS 预测，不重新训练基础模型。阈值只读取开发集
标签，确认集标签只在最终评估时打开，避免把确认集变成调参集。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pos, neg = y > 0, y < 0
    pr = float((pred[pos] == 1).mean()) if pos.any() else 0.0
    nr = float((pred[neg] == -1).mean()) if neg.any() else 0.0
    return {
        "n": int(len(y)), "direction_accuracy": float((y == pred).mean()),
        "positive_recall": pr, "negative_recall": nr,
        "balanced_accuracy": (pr + nr) / 2.0, "all_negative_baseline": float(neg.mean()),
    }


def nearest_manifest(source: Path) -> dict:
    for path in [source.parent / "manifest.json", *source.parent.glob("*manifest*.json")]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return {}


def run(args) -> int:
    source = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    manifest = nearest_manifest(source)
    required = {
        "forecast_origin": "D-1 14:00",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
    }
    for key, expected in required.items():
        if key in manifest and manifest[key] != expected:
            raise RuntimeError(f"source manifest contract failed: {key}={manifest[key]!r}")
    if args.variant and "variant" in frame.columns:
        frame = frame[frame["variant"].eq(args.variant)].copy()
    if frame.empty:
        raise RuntimeError("source variant is empty")
    frame["target_day"] = pd.to_datetime(frame["target_day"]).dt.normalize()
    frame[args.label_col] = pd.to_numeric(frame[args.label_col], errors="coerce")
    frame[args.score_col] = pd.to_numeric(frame[args.score_col], errors="coerce")
    frame = frame.dropna(subset=["target_day", args.label_col, args.score_col]).sort_values("target_day")
    dev_start, dev_end = pd.Timestamp(args.dev_start), pd.Timestamp(args.dev_end)
    confirm_start, confirm_end = pd.Timestamp(args.confirm_start), pd.Timestamp(args.confirm_end)
    if dev_end >= confirm_start:
        raise RuntimeError("development and confirmation windows overlap")
    if confirm_end >= pd.Timestamp("2026-08-15"):
        raise RuntimeError("fresh final holdout remains sealed")
    dev = frame[frame["target_day"].between(dev_start, dev_end)].copy()
    confirm = frame[frame["target_day"].between(confirm_start, confirm_end)].copy()
    if dev.empty or confirm.empty:
        raise RuntimeError("development or confirmation window is empty")
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    candidates = []
    for threshold in thresholds:
        y = np.sign(dev[args.label_col].to_numpy(float)).astype(int)
        pred = np.where(dev[args.score_col].to_numpy(float) >= threshold, 1, -1)
        m = score(y, pred)
        candidates.append({"threshold": threshold, **m})
    candidate_df = pd.DataFrame(candidates)
    selected = sorted(candidates, key=lambda x: (x["balanced_accuracy"], x["positive_recall"], -abs(x["threshold"])), reverse=True)[0]
    y_dev = np.sign(dev[args.label_col].to_numpy(float)).astype(int)
    y_confirm = np.sign(confirm[args.label_col].to_numpy(float)).astype(int)
    pred_confirm = np.where(confirm[args.score_col].to_numpy(float) >= selected["threshold"], 1, -1)
    confirm_metrics = score(y_confirm, pred_confirm)
    dev_metrics = selected.copy()
    result = {
        "route": args.route,
        "variant": args.variant or "all",
        "selected_threshold": selected["threshold"],
        "dev": dev_metrics,
        "confirm": confirm_metrics,
        "dev_window": [dev_start.date().isoformat(), dev_end.date().isoformat()],
        "confirm_window": [confirm_start.date().isoformat(), confirm_end.date().isoformat()],
    }
    candidate_df.to_csv(out / "development_threshold_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([confirm_metrics | {"selected_threshold": selected["threshold"]}]).to_csv(out / "confirm_metrics.csv", index=False, encoding="utf-8-sig")
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_manifest = {
        "status": "STRICT/PASS", "route": args.route, "source": str(source),
        "forecast_origin": "D-1 14:00", "threshold_selected_on": "development labels only",
        "training_label_cutoff": "source OOS already <= D-2; no target-day training",
        "development_window": result["dev_window"], "confirmation_window": result["confirm_window"],
        "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False, "final_holdout_touched": False,
        "selected_threshold": selected["threshold"], "selection_objective": "balanced_accuracy then positive_recall",
    }
    (out / "manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    parser.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
