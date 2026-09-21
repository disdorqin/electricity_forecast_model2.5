"""A 线 Cycle 04：因果 direct-meta sign router。

消费已审计的 P6/Similar-Day/context 专家 OOS 与合法状态描述符，逐目标日训练一个
class-balanced LightGBM sign router。所有 router 训练样本和阈值候选均限制为 OOS 日
``<= D-2``；这是独立于 expert competence 的新候选，不改生产链路。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_iteration7_recurring_emergent_router import (
    ROOT, build_meta_features, load_experts,
)


FEATURES = [
    "p6_p", "sd_p", "ctx_p", "p6_conf", "sd_conf", "ctx_conf",
    "sd_ctx_gap", "sd_p6_gap", "ctx_p6_gap", "p6_votes_sd", "p6_votes_ctx",
    "ctx_spread_mean14", "ctx_spread_std14", "ctx_spread_last", "ctx_spread_mean3",
    "ctx_spread_range14", "ctx_spread_absmean14", "ctx_spread_positive_rate14", "ctx_spread_slope14",
    "residual_load_renew", "renewable_share", "bidding_space_ratio", "interconnect_share",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "sd20_spread_mean", "sd20_spread_median", "sd20_positive_rate", "sd20_weighted_positive_rate",
    "sd20_spread_std", "sd20_mean_distance", "sd20_day_positive_rate",
    "sd20_1_8_positive_rate", "sd20_9_16_positive_rate", "sd20_17_24_positive_rate",
]


def metric(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pos, neg = y > 0, y < 0
    pr = float((pred[pos] == 1).mean()) if pos.any() else 0.0
    nr = float((pred[neg] == -1).mean()) if neg.any() else 0.0
    return {
        "direction_accuracy": float((y == pred).mean()), "positive_recall": pr,
        "negative_recall": nr, "balanced_accuracy": (pr + nr) / 2,
        "all_negative_baseline": float(neg.mean()),
    }


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", class_weight="balanced", n_estimators=120,
        learning_rate=0.035, num_leaves=15, max_depth=5, min_child_samples=35,
        subsample=0.9, colsample_bytree=0.85, reg_lambda=3.0,
        random_state=seed, n_jobs=4, verbosity=-1,
    )


def run(args) -> int:
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    expert = load_experts(root)
    x, sd_audit = build_meta_features(root, expert)
    missing = [c for c in FEATURES if c not in x.columns]
    if missing:
        raise RuntimeError(f"missing legal meta features: {missing}")
    x["target_day"] = pd.to_datetime(x["target_day"]).dt.normalize()
    x = x.sort_values(["target_day", "hour_business"])
    all_days = sorted(x["target_day"].unique())
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end >= pd.Timestamp("2026-08-15"):
        raise RuntimeError("fresh final holdout remains sealed")
    rows, audits = [], []
    for day in [d for d in all_days if start <= d <= end]:
        cutoff = pd.Timestamp(day) - pd.Timedelta(days=2)
        train_days = [d for d in all_days if d <= cutoff][-args.history_days:]
        if len(train_days) < args.min_history_days:
            continue
        assert max(train_days) <= cutoff
        hist = x[x["target_day"].isin(train_days)].copy()
        q = x[x["target_day"] == day].copy()
        if len(q) != 24:
            raise RuntimeError(f"{day.date()}: expected 24 rows, got {len(q)}")
        y_hist = np.where(hist["target_spread"].to_numpy(float) >= 0, 1, 0)
        y_true = np.where(q["target_spread"].to_numpy(float) >= 0, 1, -1)
        model = make_model(args.seed)
        model.fit(hist[FEATURES], y_hist)
        p_positive = model.predict_proba(q[FEATURES])[:, 1]
        pred = np.where(p_positive >= args.threshold, 1, -1)
        rows.append({"target_day": day.date().isoformat(), "threshold": args.threshold, **metric(y_true, pred), "training_last_day": max(train_days).date().isoformat()})
        audits.append({"target_day": day.date().isoformat(), "training_last_day": max(train_days).date().isoformat(), "required_last_day": cutoff.date().isoformat(), "strict_ok": bool(max(train_days) <= cutoff), "history_days": len(train_days)})
    if not rows:
        raise RuntimeError("no target days")
    daily = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    if not audit["strict_ok"].all():
        raise RuntimeError("direct meta training boundary failed")
    monthly = daily.assign(month=daily["target_day"].str[:7]).groupby("month", as_index=False).agg(
        days=("target_day", "nunique"), direction_accuracy=("direction_accuracy", "mean"),
        positive_recall=("positive_recall", "mean"), negative_recall=("negative_recall", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), all_negative_baseline=("all_negative_baseline", "mean"),
    )
    summary = {k: float(daily[k].mean()) for k in ("direction_accuracy", "positive_recall", "negative_recall", "balanced_accuracy", "all_negative_baseline")}
    summary.update({"days": int(daily["target_day"].nunique()), "route": "A_direct_meta_router", "threshold": args.threshold})
    daily.to_csv(out / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(out / "monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(out / "router_training_audit.csv", index=False, encoding="utf-8-sig")
    sd_audit.to_csv(out / "similar_day_causal_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "STRICT/PASS", "route": "A_direct_meta_router", "forecast_origin": "D-1 14:00",
        "router_training_labels": "historical OOS <= D-2", "training_last_day": "per target day <= D-2",
        "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False, "final_holdout_touched": False,
        "similar_day_latest_candidate": "<= D-2", "threshold": args.threshold,
        "screen_range": [start.date().isoformat(), end.date().isoformat()], "feature_count": len(FEATURES),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "summary": summary}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/cycle_04_direct_meta_router"))
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--history-days", type=int, default=120)
    parser.add_argument("--min-history-days", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260823)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
