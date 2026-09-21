from __future__ import annotations

"""Method-faithful reproduction of the 2025 covariate-dependent DART mixture model.

Paper model (DART = DA - RT):
  f(y|x) = pi1(x) N(mu1(x), sigma1(x))
         + pi2(x) GPD(y; xi_pos, sigma_pos(x)) I[y >= 0]
         + pi3(x) GPD(-y; xi_neg, sigma_neg(x)) I[y <= 0]
where all mixture probabilities and all severity/location/scale parameters depend
on contemporaneously available covariates.  Scale uses the paper transform
h(a)=10 atan(a)+5*pi and 0<xi<0.5.

The original NYISO Long Island weather archive (HDD/CDD) and natural-gas
futures are not present in this repository.  We therefore reproduce the exact
statistical architecture and monthly expanding-window protocol on Shandong
using only non-fabricated analogous covariates: target-day load/wind/solar
forecasts plus daily/yearly Fourier terms.  This is a method reproduction, not
a claim of result-level replication on the unavailable original dataset.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from common import DEFAULT_96, atomic_csv, atomic_json, atomic_parquet, aggregate_96_to_hourly, load_shandong_96, regression_metrics  # noqa: E402

PAPER_FULL = {
    "market": "NYISO Long Island",
    "target": "DART = DA - RT",
    "covariates": ["LOAD", "HDD", "CDD", "WS", "UV", "CosD", "SinD", "CosY", "SinY", "NG"],
    "full_model_lambda": 100.0,
    "oos_mae": 20.89,
    "oos_crps": 0.61,
    "xi_positive_about": 0.18,
    "xi_negative_about": 0.42,
    "monthly_retrain": "train through second-last day of previous month",
}


def prepare(path: Path) -> tuple[pd.DataFrame, list[str]]:
    q = load_shandong_96("2022-01-01", "2026-08-17", path)
    h = aggregate_96_to_hourly(q)
    ts = pd.to_datetime(h["timestamp"])
    hour = (h["hour_business"].astype(float) - 1.0) % 24.0
    doy = ts.dt.dayofyear.astype(float) - 1.0
    h["LOAD"] = pd.to_numeric(h["直调负荷预测"], errors="coerce")
    h["WS"] = pd.to_numeric(h["风电预测"], errors="coerce")
    h["UV"] = pd.to_numeric(h["光伏预测"], errors="coerce")
    h["CosD"] = np.cos(2.0 * np.pi * hour / 24.0)
    h["SinD"] = np.sin(2.0 * np.pi * hour / 24.0)
    h["CosY"] = np.cos(2.0 * np.pi * doy / 365.25)
    h["SinY"] = np.sin(2.0 * np.pi * doy / 365.25)
    features = ["LOAD", "WS", "UV", "CosD", "SinD", "CosY", "SinY"]
    h = h.dropna(subset=features + ["dart_da_minus_rt"]).sort_values("timestamp").reset_index(drop=True)
    return h, features


def _inv_h(scale: float) -> float:
    s = float(np.clip(scale, 0.25, 10 * np.pi - 0.25))
    return float(np.tan((s - 5.0 * np.pi) / 10.0))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    return math.log(p / (1 - p))


def _unpack(theta: torch.Tensor, d: int):
    pos = 0
    vecs = []
    for _ in range(6):
        vecs.append(theta[pos:pos + d])
        pos += d
    eta, z1, z2, z3, b2, b3 = vecs
    xi_pos = 0.499 * torch.sigmoid(theta[pos])
    xi_neg = 0.499 * torch.sigmoid(theta[pos + 1])
    return eta, z1, z2, z3, b2, b3, xi_pos, xi_neg


def _h(a: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.atan(a) + 5.0 * math.pi


def _log_gpd(excess: torch.Tensor, scale: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    scale = torch.clamp(scale, min=1e-5)
    z = torch.clamp(1.0 + xi * excess / scale, min=1e-12)
    return -torch.log(scale) - (1.0 / xi + 1.0) * torch.log(z)


def initialize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    ridge = 1e-5 * np.eye(d)
    eta = np.linalg.solve(X.T @ X + ridge, X.T @ y)
    resid = y - X @ eta
    z1 = np.zeros(d); z2 = np.zeros(d); z3 = np.zeros(d)
    z1[0] = _inv_h(max(np.std(resid), 5.0))
    pos = y[y > 0]; neg = -y[y < 0]
    z2[0] = _inv_h(max(float(np.median(pos)) if len(pos) else 15.0, 5.0))
    z3[0] = _inv_h(max(float(np.median(neg)) if len(neg) else 15.0, 5.0))
    b2 = np.zeros(d); b3 = np.zeros(d)
    b2[0] = math.log(0.15 / 0.70)
    b3[0] = math.log(0.15 / 0.70)
    tail = np.array([_logit(0.18 / 0.499), _logit(0.42 / 0.499)])
    return np.concatenate([eta, z1, z2, z3, b2, b3, tail]).astype(float)


class ExactMixtureMLE:
    def __init__(self, l1_lambda: float = 100.0, maxiter: int = 100):
        self.l1_lambda = float(l1_lambda)
        self.maxiter = int(maxiter)

    def fit(self, Xraw: np.ndarray, y: np.ndarray, warm_start: np.ndarray | None = None) -> "ExactMixtureMLE":
        Xraw = np.asarray(Xraw, float)
        y = np.asarray(y, float)
        self.mean_ = Xraw.mean(axis=0)
        self.std_ = Xraw.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        Z = (Xraw - self.mean_) / self.std_
        X = np.c_[np.ones(len(Z)), Z]
        d = X.shape[1]
        theta0 = initialize(X, y) if warm_start is None or len(warm_start) != 6 * d + 2 else warm_start.copy()
        Xt = torch.tensor(X, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        invalid = torch.tensor(-1e30, dtype=torch.float64)

        def fun_and_grad(theta_np: np.ndarray):
            th = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
            eta, z1, z2, z3, b2, b3, xi_p, xi_n = _unpack(th, d)
            mu = Xt @ eta
            s1 = _h(Xt @ z1)
            sp = _h(Xt @ z2)
            sn = _h(Xt @ z3)
            logits = torch.stack([torch.zeros_like(yt), Xt @ b2, Xt @ b3], dim=1)
            logpi = torch.log_softmax(logits, dim=1)
            logn = -0.5 * math.log(2 * math.pi) - torch.log(s1) - 0.5 * ((yt - mu) / s1) ** 2
            logp = _log_gpd(torch.clamp(yt, min=0.0), sp, xi_p)
            logm = _log_gpd(torch.clamp(-yt, min=0.0), sn, xi_n)
            c1 = logpi[:, 0] + logn
            c2 = torch.where(yt >= 0, logpi[:, 1] + logp, invalid)
            c3 = torch.where(yt <= 0, logpi[:, 2] + logm, invalid)
            loglike = torch.logsumexp(torch.stack([c1, c2, c3], dim=1), dim=1).sum()
            # Paper L1 regularization: penalize covariate slopes, not intercepts or tail shapes.
            slope_pen = sum(torch.abs(v[1:]).sum() for v in (eta, z1, z2, z3, b2, b3))
            loss = -loglike + self.l1_lambda * slope_pen
            loss.backward()
            return float(loss.detach()), th.grad.detach().cpu().numpy()

        res = minimize(fun_and_grad, theta0, jac=True, method="L-BFGS-B", options={"maxiter": self.maxiter, "ftol": 1e-9, "gtol": 1e-5, "maxls": 30})
        self.theta_ = np.asarray(res.x, float)
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        self.nit_ = int(res.nit)
        self.objective_ = float(res.fun)
        self.d_ = d
        return self

    def components(self, Xraw: np.ndarray) -> dict[str, np.ndarray | float]:
        Z = (np.asarray(Xraw, float) - self.mean_) / self.std_
        X = np.c_[np.ones(len(Z)), Z]
        th = torch.tensor(self.theta_, dtype=torch.float64)
        Xt = torch.tensor(X, dtype=torch.float64)
        eta, z1, z2, z3, b2, b3, xi_p, xi_n = _unpack(th, self.d_)
        with torch.no_grad():
            mu = (Xt @ eta).numpy()
            s1 = _h(Xt @ z1).numpy()
            sp = _h(Xt @ z2).numpy()
            sn = _h(Xt @ z3).numpy()
            logits = torch.stack([torch.zeros(len(Xt), dtype=torch.float64), Xt @ b2, Xt @ b3], dim=1)
            probs = torch.softmax(logits, dim=1).numpy()
            xp = float(xi_p); xn = float(xi_n)
        expected = probs[:, 0] * mu + probs[:, 1] * sp / (1.0 - xp) - probs[:, 2] * sn / (1.0 - xn)
        return {"mu": mu, "sigma_regular": s1, "scale_pos": sp, "scale_neg": sn, "probs": probs, "xi_pos": xp, "xi_neg": xn, "expected": expected}


def approximate_quantiles(comp: dict, taus: np.ndarray, draws: int = 300, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = np.asarray(comp["probs"], float)
    n = len(probs)
    reg_u = rng.random((n, draws))
    val_u = np.clip(rng.random((n, draws)), 1e-7, 1 - 1e-7)
    samples = np.empty((n, draws), float)
    neg = reg_u < probs[:, 2, None]
    regular = (reg_u >= probs[:, 2, None]) & (reg_u < (probs[:, 2] + probs[:, 0])[:, None])
    pos = ~(neg | regular)
    # inverse GPD: scale/xi * ((1-u)^(-xi)-1)
    xp = float(comp["xi_pos"]); xn = float(comp["xi_neg"])
    ep = np.asarray(comp["scale_pos"])[:, None] / xp * ((1 - val_u) ** (-xp) - 1.0)
    en = np.asarray(comp["scale_neg"])[:, None] / xn * ((1 - val_u) ** (-xn) - 1.0)
    normal = np.asarray(comp["mu"])[:, None] + np.asarray(comp["sigma_regular"])[:, None] * rng.standard_normal((n, draws))
    samples[neg] = (-en)[neg]
    samples[regular] = normal[regular]
    samples[pos] = ep[pos]
    return np.quantile(samples, taus, axis=1).T


def pinball_mean(y: np.ndarray, q: np.ndarray, taus: np.ndarray) -> float:
    yy = y[:, None]
    e = yy - q
    loss = np.maximum(taus[None, :] * e, (taus[None, :] - 1.0) * e)
    return float(loss.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_96.relative_to(Path(__file__).resolve().parents[4])))
    ap.add_argument("--output", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24/paper_reproductions/r3_covariate_mixture_v2")
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--test-end", default="2025-12-31")
    ap.add_argument("--lambda-l1", type=float, default=100.0)
    ap.add_argument("--maxiter-first", type=int, default=100)
    ap.add_argument("--maxiter-warm", type=int, default=45)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[4]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data, features = prepare(root / args.data)
    test_start = pd.Timestamp(args.test_start)
    test_end = pd.Timestamp(args.test_end)
    months = pd.period_range(test_start.to_period("M"), test_end.to_period("M"), freq="M")
    rows = []
    pred_parts = []
    fit_rows = []
    warm = None
    taus = np.linspace(0.05, 0.95, 19)

    for i, month in enumerate(months):
        mstart = month.start_time
        mend = min(month.end_time, test_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
        if mend < test_start or mstart > test_end:
            continue
        # Paper rule: omit the final calendar day of the previous month from training.
        prev_month_end = mstart - pd.Timedelta(days=1)
        train_end = prev_month_end - pd.Timedelta(days=1)
        train = data[pd.to_datetime(data["timestamp"]) <= train_end + pd.Timedelta(hours=23, minutes=59)]
        test = data[(pd.to_datetime(data["timestamp"]) >= max(mstart, test_start)) & (pd.to_datetime(data["timestamp"]) <= mend)]
        if len(train) < 24 * 365 or len(test) < 24:
            continue
        model = ExactMixtureMLE(args.lambda_l1, args.maxiter_first if warm is None else args.maxiter_warm)
        model.fit(train[features].to_numpy(float), train["dart_da_minus_rt"].to_numpy(float), warm_start=warm)
        warm = model.theta_.copy()
        comp = model.components(test[features].to_numpy(float))
        y = test["dart_da_minus_rt"].to_numpy(float)
        exp = np.asarray(comp["expected"], float)
        m = regression_metrics(y, exp)
        q = approximate_quantiles(comp, taus, draws=250, seed=42 + i)
        pb = pinball_mean(y, q, taus)
        rows.append({"month": str(month), "n_train": len(train), "n_test": len(test), "train_end": str(train_end.date()), **m, "pinball_19q": pb})
        fit_rows.append({"month": str(month), "success": model.success_, "message": model.message_, "nit": model.nit_, "objective": model.objective_, "xi_positive": comp["xi_pos"], "xi_negative": comp["xi_neg"]})
        p = test[["market_date", "hour_business", "timestamp", "dart_da_minus_rt"]].copy()
        p["expected"] = exp
        p["p_regular"] = np.asarray(comp["probs"])[:, 0]
        p["p_positive"] = np.asarray(comp["probs"])[:, 1]
        p["p_negative"] = np.asarray(comp["probs"])[:, 2]
        p["test_month"] = str(month)
        pred_parts.append(p)
        print(f"{month}: n={len(test)} MAE={m['mae']:.3f} dir={m['direction_accuracy']:.3f} xi+={comp['xi_pos']:.3f} xi-={comp['xi_neg']:.3f} it={model.nit_}")

    metrics = pd.DataFrame(rows)
    fits = pd.DataFrame(fit_rows)
    pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    atomic_csv(out / "monthly_metrics.csv", metrics)
    atomic_csv(out / "fit_diagnostics.csv", fits)
    if not pred.empty:
        atomic_parquet(out / "predictions.parquet", pred)
        overall = regression_metrics(pred["dart_da_minus_rt"].to_numpy(float), pred["expected"].to_numpy(float))
    else:
        overall = {}
    manifest = {
        "paper": "Distributional forecasting of electricity DART spreads with a covariate-dependent mixture model",
        "method_fidelity": "exact paper distributional architecture and monthly expanding-window timing; reduced non-fabricated Shandong covariate set because paper-specific HDD/CDD and NG futures are unavailable",
        "paper_protocol": PAPER_FULL,
        "local_features": features,
        "missing_original_covariates": ["HDD", "CDD", "NG"],
        "scale_transform": "h(a)=10*atan(a)+5*pi",
        "shape_constraint": "0 < xi_positive, xi_negative < 0.499 via sigmoid",
        "mixture": ["regular Gaussian", "positive GPD", "negative reflected GPD"],
        "lambda_l1": args.lambda_l1,
        "test_range": [args.test_start, args.test_end],
        "overall": overall,
        "mean_monthly_pinball_19q": float(metrics["pinball_19q"].mean()) if len(metrics) else None,
        "fit_success_months": int(fits["success"].sum()) if len(fits) else 0,
        "fit_total_months": int(len(fits)),
        "runtime_seconds": time.perf_counter() - t0,
        "production_chain_touched": False,
    }
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
