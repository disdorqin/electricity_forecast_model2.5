"""B 线：strict-D2 三状态分布式价差方向 pilot。

这是研究区的 hourly-only pilot，不接入生产链路。它把训练日的价差按训练集分位点
划为 regular / positive-spike / negative-spike，再用 regular 条件幅度回归决定 regular
状态的方向。每个目标日都重新计算训练分位点并只使用 ``target_day <= D-2`` 的完整标签。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def load_features(cube: Path) -> list[str]:
    registry = json.loads((cube / "feature_registry.json").read_text(encoding="utf-8"))
    features = []
    for item in registry.get("features", []):
        name = item.get("feature")
        availability = str(item.get("availability", "")).lower()
        leakage = str(item.get("leakage_status", "")).lower()
        if not name or name not in COLUMNS:
            continue
        if any(token in availability for token in ("target day actual", "post14", "cutoff")):
            continue
        if "safe" not in leakage and leakage:
            continue
        features.append(name)
    return features


COLUMNS: set[str] = set()


def strict_train_days(target_day: pd.Timestamp, all_days: pd.Series, window: int) -> list[pd.Timestamp]:
    """唯一训练日 helper：完整标签最晚只能到 D-2。"""
    eligible = sorted(day for day in all_days.unique() if pd.Timestamp(day) <= target_day - pd.Timedelta(days=2))
    return eligible[-window:]


def state_labels(
    y: pd.Series,
    q_low: float = 0.10,
    q_high: float = 0.90,
    mode: str = "raw_quantile",
    sign_tail_quantile: float = 0.50,
) -> tuple[pd.Series, float, float]:
    """Build train-window-only state labels.

    ``raw_quantile`` preserves the original q_low/q_high definition. ``sign_aware``
    estimates separate absolute-magnitude cutoffs inside the negative and positive
    sides, so state labels retain economic direction rather than moving solely with
    the unconditional spread distribution.
    """
    if mode == "sign_aware":
        if not 0.0 < sign_tail_quantile < 1.0:
            raise ValueError(f"invalid sign tail quantile: {sign_tail_quantile}")
        neg = pd.to_numeric(y[y < 0], errors="coerce").abs().dropna()
        pos = pd.to_numeric(y[y > 0], errors="coerce").dropna()
        if neg.empty or pos.empty:
            raise ValueError("sign-aware state labels need both negative and positive training samples")
        neg_cut = -float(neg.quantile(sign_tail_quantile))
        pos_cut = float(pos.quantile(sign_tail_quantile))
        labels = pd.Series(np.where(y <= neg_cut, "negative_spike", np.where(y >= pos_cut, "positive_spike", "regular")), index=y.index)
        return labels, neg_cut, pos_cut
    if mode != "raw_quantile":
        raise ValueError(f"unknown state mode: {mode}")
    if not 0.0 < q_low < q_high < 1.0:
        raise ValueError(f"invalid state quantiles: {q_low}, {q_high}")
    low, high = float(y.quantile(q_low)), float(y.quantile(q_high))
    labels = pd.Series(np.where(y <= low, "negative_spike", np.where(y >= high, "positive_spike", "regular")), index=y.index)
    return labels, low, high


def fit_log_tail_regressor(x_train, target: pd.Series, mask: np.ndarray, x_test, args, seed: int):
    """估计尖峰严重程度；只在训练窗口内拟合，返回带符号的条件幅度。"""
    if mask.sum() < args.min_tail_rows:
        return None
    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed,
    )
    magnitude = np.log1p(np.abs(target.to_numpy(float)[mask]))
    model.fit(x_train.iloc[mask], magnitude)
    return model.predict(x_test)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, target_day: pd.Timestamp) -> dict:
    pos_mask, neg_mask = y_true == 1, y_true == -1
    pos = float((y_pred[pos_mask] == 1).mean()) if pos_mask.any() else 0.0
    neg = float((y_pred[neg_mask] == -1).mean()) if neg_mask.any() else 0.0
    return {
        "target_day": target_day.date().isoformat(),
        "n_slots": int(len(y_true)),
        "direction_accuracy": float((y_true == y_pred).mean()),
        "positive_recall": float(pos),
        "negative_recall": float(neg),
        "balanced_accuracy": float((pos + neg) / 2.0),
        "all_negative_baseline": float((y_true == -1).mean()),
    }


def run(args) -> int:
    global COLUMNS
    cube = args.cube.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    table_path = cube / "slot_table.parquet"
    df = pd.read_parquet(table_path)
    COLUMNS = set(df.columns)
    features = load_features(cube)
    if not features:
        raise RuntimeError("feature_registry 未提供可用且通过 safe contract 的特征")
    df["target_day"] = pd.to_datetime(df["target_day"]).dt.normalize()
    df["target_spread"] = pd.to_numeric(df["target_spread"], errors="coerce")
    df = df.dropna(subset=["target_day", "target_spread"]).sort_values(["target_day", "时刻"])
    days = sorted(df["target_day"].unique())
    start = pd.Timestamp(args.start) if args.start else pd.Timestamp(days[-30])
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(days[-1])
    eval_days = [pd.Timestamp(d) for d in days if start <= pd.Timestamp(d) <= end]
    rows, prediction_rows, feature_audits = [], [], []
    decision_thresholds = [float(x.strip()) for x in args.decision_thresholds.split(",") if x.strip()]
    if not decision_thresholds:
        raise RuntimeError("decision_thresholds 不能为空")
    for target_day in eval_days:
        train_days = strict_train_days(target_day, df["target_day"], args.train_window)
        if len(train_days) < args.min_train_days:
            continue
        assert max(train_days) <= target_day - pd.Timedelta(days=2), (
            f"strict training boundary violated: target={target_day.date()} "
            f"training_last_day={max(train_days).date()}"
        )
        train = df[df["target_day"].isin(train_days)].copy()
        test = df[df["target_day"] == target_day].copy()
        if test.empty:
            continue
        # 只在训练窗口拟合填充，避免把目标日分布带进训练过程。
        selected_features = list(features)
        dropped_features = []
        if args.drop_missing_rate is not None:
            miss = train[features].apply(pd.to_numeric, errors="coerce").isna().mean()
            selected_features = [c for c in features if float(miss[c]) <= args.drop_missing_rate]
            dropped_features = [c for c in features if c not in selected_features]
            if len(selected_features) < args.min_selected_features:
                raise RuntimeError(
                    f"{target_day.date()}: missing-rate filter leaves only {len(selected_features)} features"
                )
        medians = train[selected_features].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
        x_train = train[selected_features].apply(pd.to_numeric, errors="coerce").fillna(medians).astype(float)
        x_test = test[selected_features].apply(pd.to_numeric, errors="coerce").fillna(medians).astype(float)
        feature_audits.append({
            "target_day": target_day.date().isoformat(), "training_last_day": max(train_days).date().isoformat(),
            "selected_feature_count": len(selected_features), "dropped_feature_count": len(dropped_features),
            "dropped_features": ",".join(dropped_features), "max_allowed_missing_rate": args.drop_missing_rate,
        })
        labels, q10, q90 = state_labels(
            train["target_spread"], args.state_q_low, args.state_q_high,
            args.state_mode, args.sign_tail_quantile,
        )
        y_state = labels.map({"negative_spike": -1, "regular": 0, "positive_spike": 1}).to_numpy()
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=args.seed,
        )
        clf.fit(x_train, y_state)
        state_prob = clf.predict_proba(x_test)
        state_values = clf.classes_.tolist()
        p = {int(s): state_prob[:, i] for i, s in enumerate(state_values)}
        regular = y_state == 0
        if regular.sum() >= args.min_regular_rows:
            reg = HistGradientBoostingRegressor(
                max_iter=args.max_iter, learning_rate=0.05, max_leaf_nodes=15,
                l2_regularization=1.0, random_state=args.seed,
            )
            reg.fit(x_train.iloc[regular], train.loc[train.index[regular], "target_spread"])
            regular_value = reg.predict(x_test)
        else:
            regular_value = np.zeros(len(test))
        neg_tail = float(train.loc[y_state == -1, "target_spread"].mean()) if (y_state == -1).any() else -1.0
        pos_tail = float(train.loc[y_state == 1, "target_spread"].mean()) if (y_state == 1).any() else 1.0
        if args.tail_severity:
            neg_tail_pred = fit_log_tail_regressor(x_train, train["target_spread"], y_state == -1, x_test, args, args.seed + 101)
            pos_tail_pred = fit_log_tail_regressor(x_train, train["target_spread"], y_state == 1, x_test, args, args.seed + 102)
            if neg_tail_pred is not None:
                neg_tail_values = -np.expm1(np.maximum(neg_tail_pred, 0.0))
            else:
                neg_tail_values = np.full(len(test), neg_tail)
            if pos_tail_pred is not None:
                pos_tail_values = np.expm1(np.maximum(pos_tail_pred, 0.0))
            else:
                pos_tail_values = np.full(len(test), pos_tail)
        else:
            neg_tail_values = np.full(len(test), neg_tail)
            pos_tail_values = np.full(len(test), pos_tail)
        expected = p.get(1, np.zeros(len(test))) * pos_tail_values + p.get(-1, np.zeros(len(test))) * neg_tail_values + p.get(0, np.zeros(len(test))) * regular_value
        y_true = np.where(test["target_spread"].to_numpy() >= 0, 1, -1).astype(int)
        for threshold in decision_thresholds:
            pred = np.where(expected >= threshold, 1, -1).astype(int)
            variant = f"B_distributional_states_thr{threshold:g}"
            rows.append(metric_row(y_true, pred, target_day) | {
                "variant": variant, "decision_threshold": threshold,
                "training_last_day": max(train_days).date().isoformat(),
                "latest_candidate_day": max(train_days).date().isoformat(),
                "q10": q10, "q90": q90,
            })
            for i, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append({
                    "target_day": target_day.date().isoformat(),
                    "时刻": row["时刻"], "variant": variant,
                    "y_true": int(y_true[i]), "y_pred": int(pred[i]),
                    "expected_spread": float(expected[i]), "decision_threshold": threshold,
                    "p_negative_spike": float(p.get(-1, np.zeros(len(test)))[i]),
                    "p_regular": float(p.get(0, np.zeros(len(test)))[i]),
                    "p_positive_spike": float(p.get(1, np.zeros(len(test)))[i]),
                    "training_last_day": max(train_days).date().isoformat(),
                })
    if not rows:
        raise RuntimeError("没有满足 strict_train_days 和评估范围的目标日")
    metrics = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(out / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_audits).to_csv(out / "feature_selection_audit.csv", index=False, encoding="utf-8-sig")
    monthly = metrics.assign(month=metrics["target_day"].str[:7]).groupby(["variant", "month"], as_index=False).agg(
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
    monthly.to_csv(out / "monthly.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(out / "robustness.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for variant, group in predictions.groupby("variant", sort=True):
        y_true, y_pred = group["y_true"].to_numpy(), group["y_pred"].to_numpy()
        summary = metric_row(y_true, y_pred, pd.Timestamp("2000-01-01"))
        summary.pop("target_day", None)
        summary.update({"variant": variant, "days": int(group["target_day"].nunique()), "route": "B_distributional_states"})
        summary_rows.append(summary)
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    summary = summary_rows[0]
    manifest = {
        "status": "STRICT/PASS",
        "route": "B_distributional_states",
        "resolution": "hourly",
        "forecast_origin": "D-1 14:00",
        "training_last_day": "per target day <= D-2",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "latest_candidate_day": "per target day <= D-2",
        "state_model": ["regular", "positive_spike", "negative_spike"],
        "conditional_magnitude": "regular-state HistGradientBoostingRegressor",
        "tail_severity": "covariate-dependent log-magnitude regressors" if args.tail_severity else "global conditional tail means",
        "feature_count": len(features),
        "feature_selection": "per-target strict training-window missing-rate filter" if args.drop_missing_rate is not None else "none",
        "drop_missing_rate": args.drop_missing_rate,
        "min_selected_features": args.min_selected_features,
        "selected_feature_count_mean": float(pd.DataFrame(feature_audits)["selected_feature_count"].mean()),
        "feature_source": str((cube / "feature_registry.json").as_posix()),
        "screen_range": [start.date().isoformat(), end.date().isoformat()],
        "threshold_calibration": "training window quantiles only",
        "state_quantiles": {"low": args.state_q_low, "high": args.state_q_high},
        "state_definition": args.state_mode,
        "sign_tail_quantile": args.sign_tail_quantile,
        "decision_thresholds": decision_thresholds,
        "note": "pilot only; no production integration",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "target_days": int(metrics["target_day"].nunique()), "decision_variants": len(decision_thresholds), "metrics": summary}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/01_spread_24/alternative_distributional/cycle_01_pilot"))
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--train-window", type=int, default=90)
    parser.add_argument("--min-train-days", type=int, default=30)
    parser.add_argument("--min-regular-rows", type=int, default=50)
    parser.add_argument("--min-tail-rows", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tail-severity", action="store_true", help="fit positive/negative tail magnitude regressors")
    parser.add_argument("--state-q-low", type=float, default=0.10, help="lower state quantile, fit separately inside each strict train window")
    parser.add_argument("--state-q-high", type=float, default=0.90, help="upper state quantile, fit separately inside each strict train window")
    parser.add_argument("--state-mode", choices=["raw_quantile", "sign_aware"], default="raw_quantile")
    parser.add_argument("--sign-tail-quantile", type=float, default=0.50, help="within-sign absolute-magnitude quantile for sign_aware states")
    parser.add_argument("--drop-missing-rate", type=float, default=None, help="drop features whose strict train-window missing rate exceeds this value")
    parser.add_argument("--min-selected-features", type=int, default=30)
    parser.add_argument("--decision-thresholds", default="0", help="expected spread decision thresholds, comma-separated")
    parser.add_argument("--seed", type=int, default=20260823)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
