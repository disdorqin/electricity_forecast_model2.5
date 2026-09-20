"""在开发集冻结 B 线正类概率校准，并在确认集做一次性验证。

校准器只读取开发期 OOS 标签，确认期标签仅用于验收；不读取 final holdout，
不把确认集结果反向用于阈值、模型或路线选择。该实验用于验证 Cycle 08
诊断出的 ``p_positive`` 先验偏低是否是可校准的概率偏差。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = (frame["y_true"] > 0).to_numpy(dtype=int)
    pred = (frame["p_calibrated"] >= threshold).astype(int)
    pos, neg = y == 1, y == 0
    pos_recall = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    neg_recall = float((pred[neg] == 0).mean()) if neg.any() else float("nan")
    return {
        "slots": int(len(frame)),
        "days": int(frame["target_day"].nunique()),
        "direction_accuracy": float((pred == y).mean()),
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "balanced_accuracy": float(np.nanmean([pos_recall, neg_recall])),
        "all_negative_baseline": float(neg.mean()),
        "actual_positive_rate": float(pos.mean()),
        "mean_p_calibrated": float(frame["p_calibrated"].mean()),
        "brier_positive": float(((frame["p_calibrated"] - y) ** 2).mean()),
    }


def choose_threshold(dev: pd.DataFrame, thresholds: np.ndarray) -> float:
    best = None
    for threshold in thresholds:
        row = metrics(dev, float(threshold))
        key = (row["balanced_accuracy"], row["positive_recall"], -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return best[1]


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source)
    required = {"target_day", "y_true", "p_positive_spike", "training_last_day"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"source 缺少列: {sorted(missing)}")
    frame["target_day"] = pd.to_datetime(frame["target_day"], errors="coerce").dt.normalize()
    frame["training_last_day"] = pd.to_datetime(frame["training_last_day"], errors="coerce").dt.normalize()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["p_raw"] = pd.to_numeric(frame["p_positive_spike"], errors="coerce")
    frame = frame.dropna(subset=["target_day", "training_last_day", "y_true", "p_raw"]).copy()
    if frame.empty or frame["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("空数据或触碰 fresh final holdout")
    if (frame["training_last_day"] > frame["target_day"] - pd.Timedelta(days=2)).any():
        raise RuntimeError("source training_last_day violates strict D-2 boundary")
    dev_end = pd.Timestamp(args.dev_end)
    confirm_start = dev_end + pd.Timedelta(days=1)
    dev = frame[frame["target_day"] <= dev_end].copy()
    confirm = frame[frame["target_day"] >= confirm_start].copy()
    if dev.empty or confirm.empty:
        raise RuntimeError("开发集或确认集为空")
    y_dev = (dev["y_true"] > 0).astype(int).to_numpy()
    if len(np.unique(y_dev)) < 2:
        raise RuntimeError("开发集缺少正负两类，不能拟合校准器")

    # Platt calibration on development OOS only. Use clipped logits to avoid inf.
    p_dev = dev["p_raw"].clip(1e-6, 1 - 1e-6)
    p_confirm = confirm["p_raw"].clip(1e-6, 1 - 1e-6)
    x_dev = np.log(p_dev / (1 - p_dev)).to_numpy().reshape(-1, 1)
    x_confirm = np.log(p_confirm / (1 - p_confirm)).to_numpy().reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(x_dev, y_dev)
    dev["p_calibrated"] = calibrator.predict_proba(x_dev)[:, 1]
    confirm["p_calibrated"] = calibrator.predict_proba(x_confirm)[:, 1]

    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    threshold = choose_threshold(dev, thresholds)
    dev_metrics = metrics(dev, threshold)
    confirm_metrics = metrics(confirm, threshold)

    rows = []
    for phase, group in (("development", dev), ("confirmation", confirm)):
        for month, month_group in group.groupby(group["target_day"].dt.strftime("%Y-%m"), sort=True):
            row = {"phase": phase, "month": month, "threshold": threshold}
            row.update(metrics(month_group, threshold))
            rows.append(row)
    pd.DataFrame(rows).to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"phase": "development", "threshold": threshold, **dev_metrics}, {"phase": "confirmation", "threshold": threshold, **confirm_metrics}]).to_csv(output / "phase_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([
        dev.assign(phase="development"),
        confirm.assign(phase="confirmation"),
    ], ignore_index=True)[["target_day", "phase", "y_true", "p_raw", "p_calibrated", "training_last_day"]].to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "status": "STRICT/PASS",
        "route": "B_distributional_states_calibrated",
        "source_ledger": str(source),
        "forecast_origin": "D-1 14:00",
        "training_last_day": "source asserted <= D-2 per target day",
        "calibration_training_range": [dev["target_day"].min().date().isoformat(), dev["target_day"].max().date().isoformat()],
        "confirmation_range": [confirm["target_day"].min().date().isoformat(), confirm["target_day"].max().date().isoformat()],
        "calibration_labels": "development OOS only",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "selected_threshold": threshold,
        "selection_objective": "development balanced accuracy, positive recall tie-break",
        "note": "confirmation labels are report-only and never used to refit calibration or threshold",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "STRICT/PASS",
        "route": "B_distributional_states_calibrated",
        "development": dev_metrics,
        "confirmation": confirm_metrics,
        "selected_threshold": threshold,
        "final_holdout_touched": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# B线开发集概率校准冻结",
        "",
        "- 状态：`STRICT/PASS`；校准器为开发集 OOS 上拟合的 Platt logistic calibration。",
        f"- 冻结阈值：`{threshold:.3f}`；确认集只做一次性验证。",
        "- final holdout `2026-08-15..2026-08-21` 未触碰。",
        "",
        "## 指标",
        "",
        pd.DataFrame([{"phase": "development", **dev_metrics}, {"phase": "confirmation", **confirm_metrics}]).to_markdown(index=False),
        "",
        "## 决策",
        "",
        "该结果必须与未校准B线在同一冻结协议下比较；确认集不再调参。只有同时改善 raw、balanced、"
        "positive/negative recall 且跨月超过全负基线，才可申请 fresh final holdout。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(pd.DataFrame([{"phase": "development", **dev_metrics}, {"phase": "confirmation", **confirm_metrics}]).to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-end", default="2026-06-30")
    parser.add_argument("--threshold-min", type=float, default=0.20)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
