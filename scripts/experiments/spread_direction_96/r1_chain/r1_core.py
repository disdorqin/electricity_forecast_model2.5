from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CONTINUOUS = ("日前出清价格", "直调负荷实际", "load_change", "spread_lag1", "spread_lag96")
CATEGORICAL = ("month", "interval", "is_weekend")
FEATURES = (*CONTINUOUS, *CATEGORICAL)


def load_shandong_96(path: Path, start: str, end: str) -> pd.DataFrame:
    usecols = ["market_date", "时段", "日前出清价格", "实时出清价格", "直调负荷实际"]
    frame = pd.read_csv(path, usecols=usecols)
    frame["market_date"] = frame["market_date"].astype(str)
    frame = frame[(frame["market_date"] >= start) & (frame["market_date"] <= end)].copy()
    hhmm = frame["时段"].astype(str).str.extract(r"^(\d{1,2}):(\d{2})$").astype(float)
    frame["_minute"] = (hhmm[0] * 60 + hhmm[1]).astype(int)
    frame = frame.sort_values(["market_date", "_minute"]).reset_index(drop=True)
    frame["slot"] = frame.groupby("market_date", sort=False).cumcount() + 1
    frame["timestamp"] = pd.to_datetime(frame["market_date"]) + pd.to_timedelta(frame["_minute"], unit="m")
    for c in ("日前出清价格", "实时出清价格", "直调负荷实际"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["spread"] = frame["实时出清价格"] - frame["日前出清价格"]
    frame["load_change"] = frame["直调负荷实际"].diff().fillna(0.0)
    frame["month"] = frame["timestamp"].dt.month.astype(int)
    frame["is_weekend"] = (frame["timestamp"].dt.weekday >= 5).astype(int)
    frame["interval"] = frame["slot"].astype(int)
    return frame


def build_paper_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["spread_lag1"] = out["spread"].shift(1)
    out["spread_lag96"] = out["spread"].shift(96)
    return out.dropna(subset=["spread", "spread_lag1", "spread_lag96", "日前出清价格", "直调负荷实际", "load_change"]).reset_index(drop=True)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cont", StandardScaler(), list(CONTINUOUS)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"), list(CATEGORICAL)),
        ],
        sparse_threshold=0.0,
    )


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    yt = np.sign(np.asarray(y_true, dtype=float))
    yp = np.sign(np.asarray(y_pred, dtype=float))
    mask = yt != 0
    pos = yt > 0
    neg = yt < 0
    correct = yt == yp
    pos_acc = float(correct[pos].mean()) if pos.any() else float("nan")
    neg_acc = float(correct[neg].mean()) if neg.any() else float("nan")
    return {
        "n": int(mask.sum()),
        "direction_accuracy": float(correct[mask].mean()) if mask.any() else float("nan"),
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    d = p - y
    denom = np.abs(p) + np.abs(y)
    smape = np.where(denom == 0, 0.0, 2.0 * np.abs(d) / denom)
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "spread_smape_pct": float(100.0 * np.mean(smape)),
        **direction_metrics(y, p),
    }


@dataclass
class R1QuantileModel:
    quantiles: tuple[float, ...] = QUANTILES

    def fit(self, train: pd.DataFrame) -> "R1QuantileModel":
        self.models_: dict[float, Pipeline] = {}
        y = train["spread"].to_numpy(float)
        for tau in self.quantiles:
            model = Pipeline([
                ("prep", preprocessor()),
                ("qr", QuantileRegressor(quantile=tau, alpha=0.0, solver="highs")),
            ])
            model.fit(train[list(FEATURES)], y)
            self.models_[tau] = model
        return self

    def predict_quantiles(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["timestamp", "market_date", "slot", "spread"]].copy()
        for tau, model in self.models_.items():
            out[f"q{int(tau * 100):02d}"] = model.predict(frame[list(FEATURES)]).astype(float)
        return out

    def predict_median(self, frame: pd.DataFrame) -> np.ndarray:
        return self.models_[0.50].predict(frame[list(FEATURES)]).astype(float)
