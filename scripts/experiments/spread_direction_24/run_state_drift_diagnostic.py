"""诊断三状态概率路线的跨月漂移与可靠性，不训练、不校准、不触碰 final holdout。

该脚本只消费已通过 strict-D2 审计的 OOS 预测，回答两个问题：
1. 正/负事件率和三状态概率质量是否随月份发生漂移；
2. p_positive 的可靠性是否在开发集到确认集之间崩溃。

标签只用于事后诊断，绝不反向修改阈值、权重、特征或模型，因此产物是
``diagnostic_only``，不能作为路线晋级依据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y_positive"].to_numpy(dtype=int)
    pred = (frame["p_positive"] >= 0.5).astype(int)
    pos = y == 1
    neg = y == 0
    pos_recall = _safe_rate(float((pred[pos] == 1).sum()), float(pos.sum()))
    neg_recall = _safe_rate(float((pred[neg] == 0).sum()), float(neg.sum()))
    return {
        "direction_accuracy": float((pred == y).mean()),
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "balanced_accuracy": float(np.nanmean([pos_recall, neg_recall])),
        "all_negative_baseline": float(neg.mean()),
    }


def _entropy(frame: pd.DataFrame) -> pd.Series:
    p = frame[["p_negative", "p_regular", "p_positive"]].clip(lower=1e-12)
    return -(p * np.log(p)).sum(axis=1)


def run(args: argparse.Namespace) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    required = {
        "target_day", "y_true", "y_pred", "p_negative_spike", "p_regular", "p_positive_spike",
        "training_last_day",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"source 缺少列: {sorted(missing)}")

    frame["target_day"] = pd.to_datetime(frame["target_day"], errors="coerce").dt.normalize()
    frame["training_last_day"] = pd.to_datetime(frame["training_last_day"], errors="coerce").dt.normalize()
    for col in ["y_true", "y_pred", "p_negative_spike", "p_regular", "p_positive_spike"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=list(required)).copy()
    if frame.empty:
        raise RuntimeError("source 没有可诊断样本")
    if frame["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("diagnostic source touches fresh final holdout")
    if (frame["training_last_day"] > frame["target_day"] - pd.Timedelta(days=2)).any():
        raise RuntimeError("training_last_day violates strict D-2 boundary")

    frame = frame.rename(columns={
        "p_negative_spike": "p_negative",
        "p_positive_spike": "p_positive",
    })
    prob_sum = frame[["p_negative", "p_regular", "p_positive"]].sum(axis=1)
    if not np.allclose(prob_sum.to_numpy(), 1.0, atol=1e-6):
        raise RuntimeError("three-state probabilities do not sum to one")
    frame["y_positive"] = (frame["y_true"] > 0).astype(int)
    frame["month"] = frame["target_day"].dt.strftime("%Y-%m")
    frame["phase"] = np.where(frame["target_day"] <= pd.Timestamp("2026-06-30"), "development", "confirmation")
    frame["predicted_state"] = frame[["p_negative", "p_regular", "p_positive"]].idxmax(axis=1)
    frame["entropy"] = _entropy(frame)

    monthly_rows: list[dict[str, object]] = []
    for month, group in frame.groupby("month", sort=True):
        row: dict[str, object] = {
            "month": month,
            "phase": group["phase"].iloc[0],
            "slots": int(len(group)),
            "days": int(group["target_day"].nunique()),
            "actual_positive_rate": float(group["y_positive"].mean()),
            "mean_p_negative": float(group["p_negative"].mean()),
            "mean_p_regular": float(group["p_regular"].mean()),
            "mean_p_positive": float(group["p_positive"].mean()),
            "mean_entropy": float(group["entropy"].mean()),
            "brier_positive": float(((group["p_positive"] - group["y_positive"]) ** 2).mean()),
            "logloss_positive": float(-np.log(np.where(group["y_positive"] == 1, group["p_positive"], 1 - group["p_positive"]).clip(1e-12, 1.0)).mean()),
        }
        row.update(_binary_metrics(group))
        row["gain_vs_all_negative"] = row["direction_accuracy"] - row["all_negative_baseline"]
        monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)

    bins = np.linspace(0.0, 1.0, 11)
    frame["reliability_bin"] = pd.cut(frame["p_positive"], bins=bins, include_lowest=True, right=True)
    reliability = frame.groupby("reliability_bin", observed=False).agg(
        slots=("y_positive", "size"),
        mean_predicted_probability=("p_positive", "mean"),
        observed_positive_rate=("y_positive", "mean"),
    ).reset_index()
    reliability["calibration_gap"] = reliability["observed_positive_rate"] - reliability["mean_predicted_probability"]
    reliability["reliability_bin"] = reliability["reliability_bin"].astype(str)

    state_rows: list[dict[str, object]] = []
    for (month, state), group in frame.groupby(["month", "predicted_state"], sort=True):
        state_rows.append({
            "month": month,
            "predicted_state": state,
            "slots": int(len(group)),
            "share_of_month": float(len(group) / len(frame[frame["month"] == month])),
            "actual_positive_rate": float(group["y_positive"].mean()),
            "mean_p_positive": float(group["p_positive"].mean()),
            "direction_accuracy_at_0_5": _binary_metrics(group)["direction_accuracy"],
        })
    state_mix = pd.DataFrame(state_rows)

    phase_rows = []
    for phase, group in frame.groupby("phase", sort=True):
        row = {"phase": phase, "slots": int(len(group)), "days": int(group["target_day"].nunique())}
        row.update(_binary_metrics(group))
        row.update({
            "actual_positive_rate": float(group["y_positive"].mean()),
            "mean_p_positive": float(group["p_positive"].mean()),
            "mean_p_negative": float(group["p_negative"].mean()),
            "mean_p_regular": float(group["p_regular"].mean()),
            "brier_positive": float(((group["p_positive"] - group["y_positive"]) ** 2).mean()),
            "mean_entropy": float(group["entropy"].mean()),
        })
        phase_rows.append(row)
    phases = pd.DataFrame(phase_rows)

    monthly.to_csv(output / "monthly_drift.csv", index=False, encoding="utf-8-sig")
    reliability.to_csv(output / "reliability_bins.csv", index=False, encoding="utf-8-sig")
    state_mix.to_csv(output / "state_mix.csv", index=False, encoding="utf-8-sig")
    phases.to_csv(output / "phase_comparison.csv", index=False, encoding="utf-8-sig")

    overall = _binary_metrics(frame)
    manifest = {
        "status": "STRICT/PASS",
        "diagnostic_only": True,
        "route": "B_distributional_states",
        "source_ledger": str(source),
        "forecast_origin": "D-1 14:00",
        "training_last_day": "in source, asserted <= D-2 per target day",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "label_usage": "retrospective reliability and drift diagnostics only; no calibration or selection",
        "screen_range": [frame["target_day"].min().date().isoformat(), frame["target_day"].max().date().isoformat()],
        "fresh_final_holdout": "2026-08-15..2026-08-21 sealed",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "status": "STRICT/PASS",
        "diagnostic_only": True,
        "overall": overall,
        "monthly_mean_direction_accuracy": float(monthly["direction_accuracy"].mean()),
        "monthly_mean_balanced_accuracy": float(monthly["balanced_accuracy"].mean()),
        "monthly_mean_gain_vs_all_negative": float(monthly["gain_vs_all_negative"].mean()),
        "months": monthly["month"].tolist(),
        "final_holdout_touched": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Cycle 08 B线状态漂移与概率可靠性诊断",
        "",
        "- 状态：`STRICT/PASS`；性质：`diagnostic_only`。",
        f"- 样本区间：`{manifest['screen_range'][0]}..{manifest['screen_range'][1]}`；final holdout 未触碰。",
        "- 诊断标签仅用于事后统计，不用于阈值、权重、路由、特征选择或模型切换。",
        "",
        "## 阶段比较",
        "",
        phases.to_markdown(index=False),
        "",
        "## 月度漂移",
        "",
        monthly.to_markdown(index=False),
        "",
        "## 解释",
        "",
        "- `phase_comparison.csv` 用于观察开发集到确认集的事件率、概率均值和可靠性变化。",
        "- `reliability_bins.csv` 的 calibration_gap 为观测正类率减预测正类概率；不可据此直接调参。",
        "- `state_mix.csv` 用于识别三状态模型是否长期退化为 regular 或单一方向状态。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(phases.to_string(index=False))
    print("--- monthly drift ---")
    print(monthly.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
