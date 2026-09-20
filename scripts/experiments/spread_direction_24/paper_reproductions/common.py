from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_96 = PROJECT_ROOT / "data/96/authoritative/pmos_96_全量.csv"
DEFAULT_CUBE = PROJECT_ROOT / "outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821"
DEFAULT_OUT = PROJECT_ROOT / "outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_shandong_96(start: str = "2024-01-01", end: str = "2024-12-31", path: Path = DEFAULT_96) -> pd.DataFrame:
    usecols = [
        "market_date", "时段", "日前出清价格", "实时出清价格", "直调负荷预测", "直调负荷实际",
        "地方电厂出力预测", "外电预测", "风电预测", "光伏预测", "核电预测", "自备电厂预测", "试验机组预测",
        "正备用预测", "负备用预测",
    ]
    raw = pd.read_csv(path, usecols=lambda c: c in usecols)
    raw["market_date"] = raw["market_date"].astype(str)
    raw = raw[(raw["market_date"] >= start) & (raw["market_date"] <= end)].copy()
    # 时段 is a string (00:15 ... 24:00); lexical sorting would put 10:00 before 2:00.
    # Parse it into minutes explicitly so lag1/load-change reproduce the physical 15-min sequence.
    hhmm = raw["时段"].astype(str).str.extract(r"^(\d{1,2}):(\d{2})$").astype(float)
    raw["_minute_of_business_day"] = (hhmm[0] * 60 + hhmm[1]).astype(int)
    raw = raw.sort_values(["market_date", "_minute_of_business_day"]).reset_index(drop=True)
    raw["slot"] = raw.groupby("market_date").cumcount() + 1
    raw["timestamp"] = pd.to_datetime(raw["market_date"]) + pd.to_timedelta(raw["_minute_of_business_day"], unit="m")
    for c in raw.columns:
        if c not in {"market_date", "时段", "timestamp"}:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["spread_rt_minus_da"] = raw["实时出清价格"] - raw["日前出清价格"]
    raw["dart_da_minus_rt"] = -raw["spread_rt_minus_da"]
    raw["load_change"] = raw["直调负荷实际"].diff().fillna(0.0)
    raw["month"] = raw["timestamp"].dt.month.astype(int)
    raw["weekday"] = raw["timestamp"].dt.weekday.astype(int)
    raw["is_weekend"] = (raw["weekday"] >= 5).astype(int)
    raw["hour"] = raw["timestamp"].dt.hour.astype(int)
    return raw


def aggregate_96_to_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    # Business-hour grouping: 4 consecutive 15-min records. We preserve market_date.
    work["hour_business"] = ((work["slot"] - 1) // 4 + 1).astype(int)
    numeric = work.select_dtypes(include=[np.number]).columns.tolist()
    agg = {c: "mean" for c in numeric if c not in {"slot", "hour_business", "month", "weekday", "is_weekend", "hour"}}
    agg.update({"slot": "max", "month": "first", "weekday": "first", "is_weekend": "first", "hour": "first"})
    out = work.groupby(["market_date", "hour_business"], as_index=False).agg(agg)
    out["timestamp"] = pd.to_datetime(out["market_date"]) + pd.to_timedelta(out["hour_business"], unit="h")
    out["spread_rt_minus_da"] = out["实时出清价格"] - out["日前出清价格"]
    out["dart_da_minus_rt"] = -out["spread_rt_minus_da"]
    return out.sort_values("timestamp").reset_index(drop=True)


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt = np.sign(np.asarray(y_true, float))
    yp = np.sign(np.asarray(y_pred, float))
    mask = yt != 0
    pos = yt > 0
    neg = yt < 0
    correct = yt == yp
    p = float(correct[pos].mean()) if pos.any() else math.nan
    n = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "n": int(mask.sum()),
        "direction_accuracy": float(correct[mask].mean()) if mask.any() else math.nan,
        "positive_accuracy": p,
        "negative_accuracy": n,
        "balanced_direction_accuracy": float(np.nanmean([p, n])),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    d = p - y
    denom = np.abs(p) + np.abs(y)
    smape = np.where(denom == 0, 0.0, 2.0 * np.abs(d) / denom)
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "spread_smape_pct": float(100.0 * np.mean(smape)),
        **direction_metrics(y, p),
    }


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    e = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


@dataclass
class GaussianHMM1D:
    n_states: int = 3
    max_iter: int = 100
    tol: float = 1e-5
    random_state: int = 42

    def fit(self, y: np.ndarray) -> "GaussianHMM1D":
        x = np.asarray(y, float)
        x = x[np.isfinite(x)]
        if len(x) < self.n_states * 10:
            raise ValueError("not enough data for HMM")
        qs = np.linspace(0.15, 0.85, self.n_states)
        means = np.quantile(x, qs)
        var0 = max(float(np.var(x)), 1e-3)
        vars_ = np.full(self.n_states, var0)
        trans = np.full((self.n_states, self.n_states), 0.03 / max(1, self.n_states - 1))
        np.fill_diagonal(trans, 0.97)
        trans /= trans.sum(axis=1, keepdims=True)
        pi = np.full(self.n_states, 1.0 / self.n_states)
        last_ll = -np.inf
        for _ in range(self.max_iter):
            log_emit = self._log_emission(x, means, vars_)
            log_alpha, ll = self._forward(log_emit, pi, trans)
            log_beta = self._backward(log_emit, trans)
            log_gamma = log_alpha + log_beta - ll
            gamma = np.exp(log_gamma)
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi_sum = np.zeros_like(trans)
            for t in range(len(x) - 1):
                z = (
                    log_alpha[t, :, None]
                    + np.log(np.clip(trans, 1e-12, None))
                    + log_emit[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                    - ll
                )
                xi = np.exp(z - logsumexp(z))
                xi_sum += xi
            pi = np.clip(gamma[0], 1e-8, None)
            pi /= pi.sum()
            trans = xi_sum + 1e-4
            trans /= trans.sum(axis=1, keepdims=True)
            weights = gamma.sum(axis=0) + 1e-8
            means = (gamma * x[:, None]).sum(axis=0) / weights
            vars_ = (gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
            vars_ = np.clip(vars_, 1e-3, None)
            if abs(ll - last_ll) < self.tol * (1 + abs(last_ll)):
                break
            last_ll = ll
        order = np.argsort(means)
        self.means_ = means[order]
        self.vars_ = vars_[order]
        self.transmat_ = trans[np.ix_(order, order)]
        self.startprob_ = pi[order]
        self.loglik_ = float(ll)
        self.expected_duration_ = 1.0 / np.clip(1.0 - np.diag(self.transmat_), 1e-9, None)
        return self

    @staticmethod
    def _log_emission(x: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
        return -0.5 * (np.log(2 * np.pi * vars_)[None, :] + (x[:, None] - means[None, :]) ** 2 / vars_[None, :])

    @staticmethod
    def _forward(log_emit: np.ndarray, pi: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, float]:
        T, K = log_emit.shape
        la = np.empty((T, K), float)
        la[0] = np.log(np.clip(pi, 1e-12, None)) + log_emit[0]
        lt = np.log(np.clip(trans, 1e-12, None))
        for t in range(1, T):
            la[t] = log_emit[t] + logsumexp(la[t - 1][:, None] + lt, axis=0)
        return la, float(logsumexp(la[-1]))

    @staticmethod
    def _backward(log_emit: np.ndarray, trans: np.ndarray) -> np.ndarray:
        T, K = log_emit.shape
        lb = np.zeros((T, K), float)
        lt = np.log(np.clip(trans, 1e-12, None))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(lt + log_emit[t + 1][None, :] + lb[t + 1][None, :], axis=1)
        return lb

    def filtered_predictive_probs(self, y: np.ndarray) -> np.ndarray:
        """Return P(z_t | y_<t), i.e. state probabilities available before observing y_t."""
        x = np.asarray(y, float)
        K = self.n_states
        out = np.empty((len(x), K), float)
        prev = self.startprob_.copy()
        for t, value in enumerate(x):
            pred = prev @ self.transmat_ if t > 0 else prev
            pred /= pred.sum()
            out[t] = pred
            if np.isfinite(value):
                emit = np.exp(self._log_emission(np.array([value]), self.means_, self.vars_)[0])
                post = pred * emit
                prev = post / max(post.sum(), 1e-12)
            else:
                prev = pred
        return out

    def filter_history(self, y: np.ndarray) -> np.ndarray:
        x = np.asarray(y, float)
        prev = self.startprob_.copy()
        for t, value in enumerate(x):
            pred = prev @ self.transmat_ if t > 0 else prev
            if np.isfinite(value):
                emit = np.exp(self._log_emission(np.array([value]), self.means_, self.vars_)[0])
                post = pred * emit
                prev = post / max(post.sum(), 1e-12)
            else:
                prev = pred
        return prev

    def propagate(self, prob: np.ndarray, steps: int) -> np.ndarray:
        p = np.asarray(prob, float)
        if steps <= 0:
            return p / p.sum()
        return p @ np.linalg.matrix_power(self.transmat_, int(steps))


def load_p6_features(cube_root: Path = DEFAULT_CUBE) -> tuple[pd.DataFrame, list[str]]:
    slot = pd.read_parquet(cube_root / "slot_table.parquet")
    groups = json.loads((cube_root / "feature_groups.json").read_text(encoding="utf-8"))
    key_tokens = ("风电", "光伏", "直调负荷", "竞价空间", "新能源")
    base = groups["F0"] + groups["F1"]
    raw = groups["F2"]
    phys = groups["F3"] + groups["F4"]
    err_core = [
        c for c in groups["F5"]
        if c.startswith("err_net_load_") or any(token in c for token in key_tokens)
    ]
    features = list(dict.fromkeys(base + raw + phys + err_core))
    return slot, features
