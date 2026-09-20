"""对已有 strict-D2 OOS 概率做因果阈值路由。

该脚本不重新训练基础专家，只消费已经通过 strict-D2 审计的 OOS ledger。对目标日 D，
阈值选择池严格限制为历史 OOS 日 ``<= D-2``，用于验证“概率阈值/类别代价”是否能改善
正负召回和平衡准确率。它不能读取目标日标签来调阈值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = np.arange(0.30, 0.71, 0.025)


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pos, neg = y > 0, y < 0
    pos_recall = float((pred[pos] == 1).mean()) if pos.any() else float("nan")
    neg_recall = float((pred[neg] == -1).mean()) if neg.any() else float("nan")
    return {
        "direction_accuracy": float((pred == y).mean()),
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "balanced_accuracy": float(np.nanmean([pos_recall, neg_recall])),
        "all_negative_baseline": float(neg.mean()),
    }


def choose_threshold(history: pd.DataFrame, score_col: str, label_col: str, window_days: int, thresholds: list[float]) -> tuple[float, int]:
    days = sorted(pd.to_datetime(history["target_day"]).dt.normalize().unique())[-window_days:]
    h = history[pd.to_datetime(history["target_day"]).dt.normalize().isin(days)]
    y = np.sign(h[label_col].to_numpy(float))
    p = h[score_col].to_numpy(float)
    best = None
    for threshold in thresholds:
        pred = np.where(p >= threshold, 1, -1)
        metric = scores(y, pred)
        # Balanced accuracy first; tiny positive-recall tie-break discourages all-negative rules.
        key = (metric["balanced_accuracy"], metric["positive_recall"], -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    if best is None:
        return 0.5, len(days)
    return best[1], len(days)


def run(args) -> int:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    if args.score_col not in ledger.columns or args.label_col not in ledger.columns:
        raise RuntimeError(f"source 缺少 {args.score_col}/{args.label_col}")
    if args.variant and "variant" in ledger.columns:
        ledger = ledger[ledger["variant"].eq(args.variant)].copy()
    ledger["target_day"] = pd.to_datetime(ledger["target_day"]).dt.normalize()
    ledger[args.label_col] = pd.to_numeric(ledger[args.label_col], errors="coerce")
    ledger[args.score_col] = pd.to_numeric(ledger[args.score_col], errors="coerce")
    ledger = ledger.dropna(subset=["target_day", args.label_col, args.score_col])
    variants = sorted(ledger["variant"].dropna().unique()) if "variant" in ledger.columns else [args.variant or "all"]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if not thresholds:
        thresholds = [float(x) for x in THRESHOLDS]
    start = pd.Timestamp(args.start) if args.start else ledger["target_day"].min()
    end = pd.Timestamp(args.end) if args.end else ledger["target_day"].max()
    if end >= pd.Timestamp("2026-08-15"):
        raise RuntimeError("fresh final holdout 2026-08-15..2026-08-21 remains sealed")
    rows, audits = [], []
    days = sorted(d for d in ledger["target_day"].unique() if start <= d <= end)
    for day in days:
        cutoff = day - pd.Timedelta(days=2)
        for variant in variants:
            current = ledger[(ledger["target_day"] == day) & (ledger["variant"] == variant)].copy()
            history = ledger[(ledger["target_day"] <= cutoff) & (ledger["variant"] == variant)].copy()
            if current.empty or history["target_day"].nunique() < args.min_history_days:
                continue
            threshold, used_days = choose_threshold(history, args.score_col, args.label_col, args.window_days, thresholds)
            assert history["target_day"].max() <= cutoff
            y = np.sign(current[args.label_col].to_numpy(float)).astype(int)
            pred = np.where(current[args.score_col].to_numpy(float) >= threshold, 1, -1).astype(int)
            metric = scores(y, pred)
            rows.append({"target_day": day.date().isoformat(), "variant": variant, "threshold": threshold, "threshold_history_days": used_days, **metric})
            audits.append({
                "target_day": day.date().isoformat(), "variant": variant,
                "threshold_training_last_day": history["target_day"].max().date().isoformat(),
                "required_last_day": cutoff.date().isoformat(),
                "strict_ok": bool(history["target_day"].max() <= cutoff),
                "threshold_history_days": used_days,
            })
    if not rows:
        raise RuntimeError("没有满足历史 OOS 和目标日期范围的阈值路由结果")
    daily = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if not audit["strict_ok"].all():
        raise RuntimeError("threshold causal audit failed")
    monthly = daily.assign(month=daily["target_day"].str[:7]).groupby(["variant", "month"], as_index=False).agg(
        days=("target_day", "nunique"),
        direction_accuracy=("direction_accuracy", "mean"),
        positive_recall=("positive_recall", "mean"),
        negative_recall=("negative_recall", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        all_negative_baseline=("all_negative_baseline", "mean"),
    )
    robustness = monthly.groupby("variant", as_index=False).agg(
        months=("month", "nunique"), mean_month_acc=("direction_accuracy", "mean"),
        mean_month_bal=("balanced_accuracy", "mean"), mean_positive_recall=("positive_recall", "mean"),
        mean_negative_recall=("negative_recall", "mean"), mean_all_negative=("all_negative_baseline", "mean"),
    )
    robustness["mean_gain_vs_all_negative"] = robustness["mean_month_acc"] - robustness["mean_all_negative"]
    daily.to_csv(output / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(output / "robustness.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(output / "threshold_training_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS",
        "route": args.route,
        "source_ledger": str(source),
        "forecast_origin": "D-1 14:00",
        "threshold_training_labels": "historical OOS <= D-2",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "threshold_candidates": thresholds,
        "selection_objective": "balanced_accuracy then positive_recall then threshold proximity to 0.5",
        "screen_range": [start.date().isoformat(), end.date().isoformat()],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(robustness.to_string(index=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/cycle_02_a_dynamic_competence_5m/ledger.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/cycle_03_a_threshold_router"))
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--min-history-days", type=int, default=30)
    parser.add_argument("--score-col", default="prob_positive")
    parser.add_argument("--label-col", default="target_spread")
    parser.add_argument("--variant")
    parser.add_argument("--route", default="A_prequential_probability_threshold")
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
