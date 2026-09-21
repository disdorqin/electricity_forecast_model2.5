from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_auc_score, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    DEFAULT_96,
    DEFAULT_OUT,
    GaussianHMM1D,
    atomic_csv,
    atomic_json,
    load_shandong_96,
    pinball,
    regression_metrics,
)

PAPER_TARGET = {
    "rows": 35136,
    "spread_mean": -7.50,
    "spread_median": -0.005,
    "q05": -196.84,
    "q95": 137.65,
    "quantile_mae": 24.015,
    "quantile_rmse": 49.673,
    "quantile_pinball": 7.021,
    "linear_state_rmse": 46.745,
    "upper_tail_recall": 0.872,
    "lower_tail_recall": 0.841,
}

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spread_lag1"] = out["spread_rt_minus_da"].shift(1)
    out["spread_lag96"] = out["spread_rt_minus_da"].shift(96)
    out["interval"] = out["slot"].astype(int)
    # paper uses month, intraday interval, weekend controls
    return out.dropna(subset=["spread_lag1", "spread_lag96", "日前出清价格", "直调负荷实际", "load_change"]).reset_index(drop=True)


def split_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = pd.to_datetime(df["timestamp"])
    train = (ts >= pd.Timestamp("2024-01-01 00:15")) & (ts < pd.Timestamp("2024-10-01 00:15"))
    val = (ts >= pd.Timestamp("2024-10-01 00:15")) & (ts < pd.Timestamp("2024-12-01 00:15"))
    test = (ts >= pd.Timestamp("2024-12-01 00:15")) & (ts <= pd.Timestamp("2025-01-01 00:00"))
    return train.to_numpy(), val.to_numpy(), test.to_numpy()


def preprocessor() -> ColumnTransformer:
    continuous = ["日前出清价格", "直调负荷实际", "load_change", "spread_lag1", "spread_lag96"]
    categorical = ["month", "interval", "is_weekend"]
    return ColumnTransformer(
        [
            ("cont", StandardScaler(), continuous),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    order = np.argsort(recall)
    return float(auc(recall[order], precision[order]))


def evaluate_tail(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, train_lo: float, train_hi: float) -> dict:
    upper = (y >= train_hi).astype(int)
    lower = (y <= train_lo).astype(int)
    upper_sig = q_hi >= train_hi
    lower_sig = q_lo <= train_lo
    upper_recall = float(upper_sig[upper == 1].mean()) if upper.sum() else np.nan
    lower_recall = float(lower_sig[lower == 1].mean()) if lower.sum() else np.nan
    return {
        "upper_tail_threshold": float(train_hi),
        "lower_tail_threshold": float(train_lo),
        "upper_tail_recall": upper_recall,
        "lower_tail_recall": lower_recall,
        "upper_warning_share": float(upper_sig.mean()),
        "lower_warning_share": float(lower_sig.mean()),
    }


def fit_hmm_state_features(full: pd.DataFrame, train_mask: np.ndarray) -> tuple[GaussianHMM1D, np.ndarray]:
    y = full["spread_rt_minus_da"].to_numpy(float)
    hmm = GaussianHMM1D(n_states=3, max_iter=80, tol=1e-5).fit(y[train_mask])
    probs = hmm.filtered_predictive_probs(y)
    return hmm, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_96.relative_to(Path(__file__).resolve().parents[4])))
    ap.add_argument("--output", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/r1_regime_quantile")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    data_path = root / args.data
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    raw = load_shandong_96("2024-01-01", "2024-12-31", data_path)
    data_stats = {
        "rows": int(len(raw)),
        "spread_mean": float(raw["spread_rt_minus_da"].mean()),
        "spread_median": float(raw["spread_rt_minus_da"].median()),
        "q01": float(raw["spread_rt_minus_da"].quantile(0.01)),
        "q05": float(raw["spread_rt_minus_da"].quantile(0.05)),
        "q95": float(raw["spread_rt_minus_da"].quantile(0.95)),
        "q99": float(raw["spread_rt_minus_da"].quantile(0.99)),
    }
    full = build_features(raw)
    train_mask, val_mask, test_mask = split_masks(full)
    train = full.loc[train_mask].copy()
    val = full.loc[val_mask].copy()
    test = full.loc[test_mask].copy()

    feature_cols = ["日前出清价格", "直调负荷实际", "load_change", "spread_lag1", "spread_lag96", "month", "interval", "is_weekend"]
    y_train = train["spread_rt_minus_da"].to_numpy(float)
    y_test = test["spread_rt_minus_da"].to_numpy(float)

    q_preds: dict[float, np.ndarray] = {}
    q_rows = []
    for tau in QUANTILES:
        model = Pipeline([
            ("prep", preprocessor()),
            ("qr", QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")),
        ])
        model.fit(train[feature_cols], y_train)
        pred = model.predict(test[feature_cols]).astype(float)
        q_preds[tau] = pred
        q_rows.append({"quantile": tau, "pinball": pinball(y_test, pred, tau)})

    median_pred = q_preds[0.50]
    quantile_metrics = regression_metrics(y_test, median_pred)
    quantile_metrics["average_pinball_7q"] = float(np.mean([r["pinball"] for r in q_rows]))
    quantile_metrics["median_pinball"] = pinball(y_test, median_pred, 0.50)
    tail = evaluate_tail(
        y_test,
        q_preds[0.05],
        q_preds[0.95],
        float(np.quantile(y_train, 0.05)),
        float(np.quantile(y_train, 0.95)),
    )

    # Paper linear baseline and linear + lagged regime probabilities.
    linear = Pipeline([("prep", preprocessor()), ("lr", LinearRegression())])
    linear.fit(train[feature_cols], y_train)
    linear_pred = linear.predict(test[feature_cols])
    linear_metrics = regression_metrics(y_test, linear_pred)

    hmm, state_probs = fit_hmm_state_features(full, train_mask)
    state_cols = [f"state_prob_{i}" for i in range(3)]
    full_state = full.copy()
    for i, c in enumerate(state_cols):
        full_state[c] = state_probs[:, i]
    train_s = full_state.loc[train_mask]
    test_s = full_state.loc[test_mask]
    state_features = feature_cols + state_cols
    prep_state = ColumnTransformer(
        [
            ("cont", StandardScaler(), ["日前出清价格", "直调负荷实际", "load_change", "spread_lag1", "spread_lag96", *state_cols]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), ["month", "interval", "is_weekend"]),
        ],
        sparse_threshold=0.0,
    )
    linear_state = Pipeline([("prep", prep_state), ("lr", LinearRegression())])
    linear_state.fit(train_s[state_features], train_s["spread_rt_minus_da"].to_numpy(float))
    linear_state_pred = linear_state.predict(test_s[state_features])
    linear_state_metrics = regression_metrics(y_test, linear_state_pred)

    pred_table = test[["timestamp", "market_date", "slot", "spread_rt_minus_da"]].copy()
    pred_table["linear_pred"] = linear_pred
    pred_table["linear_state_pred"] = linear_state_pred
    for tau in QUANTILES:
        pred_table[f"q{int(tau*100):02d}"] = q_preds[tau]
    atomic_parquet = None
    try:
        from common import atomic_parquet as _apq
        _apq(out / "predictions.parquet", pred_table)
    except Exception:
        pass
    atomic_csv(out / "quantile_losses.csv", pd.DataFrame(q_rows))
    atomic_csv(out / "hmm_transition.csv", pd.DataFrame(hmm.transmat_, columns=state_cols))
    atomic_csv(out / "hmm_states.csv", pd.DataFrame({
        "state": state_cols,
        "mean": hmm.means_,
        "std": np.sqrt(hmm.vars_),
        "expected_duration_intervals": hmm.expected_duration_,
        "expected_duration_hours": hmm.expected_duration_ * 0.25,
    }))

    # Dataset-level reproduction is close but not byte-identical to the paper's private file.
    paper_match = {
        "row_count_exact": data_stats["rows"] == PAPER_TARGET["rows"],
        "q05_abs_diff": abs(data_stats["q05"] - PAPER_TARGET["q05"]),
        "q95_abs_diff": abs(data_stats["q95"] - PAPER_TARGET["q95"]),
        "quantile_mae_abs_diff": abs(quantile_metrics["mae"] - PAPER_TARGET["quantile_mae"]),
        "quantile_rmse_abs_diff": abs(quantile_metrics["rmse"] - PAPER_TARGET["quantile_rmse"]),
        "upper_recall_abs_diff": abs(tail["upper_tail_recall"] - PAPER_TARGET["upper_tail_recall"]),
        "lower_recall_abs_diff": abs(tail["lower_tail_recall"] - PAPER_TARGET["lower_tail_recall"]),
    }
    manifest = {
        "paper": "Risk-Aware Trading Signals for Smart Aggregators in Multi-Time-Scale Electricity Markets Using Regime-Switching and Tail-Risk Analysis",
        "paper_protocol": {
            "market": "Shandong",
            "resolution": "15min",
            "target": "RT-DA spread",
            "train": "2024-01 through 2024-09",
            "validation": "2024-10 through 2024-11",
            "test": "2024-12",
            "quantiles": QUANTILES,
            "features": feature_cols,
            "hmm_states": 3,
        },
        "fidelity": "near-dataset strict-method reproduction; local 2024 file has exactly 35136 rows but descriptive statistics differ slightly from the non-public paper dataset",
        "data_stats": data_stats,
        "paper_targets": PAPER_TARGET,
        "paper_match": paper_match,
        "quantile_metrics": quantile_metrics,
        "tail_metrics": tail,
        "linear_metrics": linear_metrics,
        "linear_state_metrics": linear_state_metrics,
        "hmm": {
            "means": hmm.means_.tolist(),
            "std": np.sqrt(hmm.vars_).tolist(),
            "transition": hmm.transmat_.tolist(),
            "expected_duration_hours": (hmm.expected_duration_ * 0.25).tolist(),
        },
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
