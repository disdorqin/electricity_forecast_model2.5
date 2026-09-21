from __future__ import annotations

"""Shandong-adapted reproduction of Dinler (Applied Energy, 2021).

Paper target:
    before day-ahead market closure, classify whether day-ahead or balancing price
    will be higher at each hour of the next day.
Paper mechanism:
    LSTM autoencoder preprocessing + hybrid of five binary classifiers:
    RF, SVC, Logistic Regression, stochastic Gradient Boosting, XGBoost.

Shandong adaptation:
    target = sign(RT - DA), forecast origin = D-1 14:00, 24 outputs for D.
    Input = strict P6 regular features only. No D target actual/RT/spread/DA and no
    D-1 post-14 realized spread are introduced here; the P6 cube owns that contract.

This is an adapted reproduction, not a claim that Turkish raw feature definitions are
identical to Shandong. It tests whether the paper's AE + heterogeneous classifier hybrid
transfers under our business information boundary.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import atomic_csv, atomic_json, atomic_parquet, load_p6_features  # noqa: E402
from integrate_paper_modules_p6_strict import strict_train_days  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def require_preflight() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/tests/check_preflight_health.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-12:])
        raise RuntimeError("project preflight is not all green; refusing training/backtest\n" + tail)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def metrics(y_spread: np.ndarray, direction: np.ndarray) -> dict[str, float | int]:
    yt = np.sign(np.asarray(y_spread, float))
    yp = np.asarray(direction, int)
    eligible = yt != 0
    pos = yt > 0
    neg = yt < 0
    correct = yt == yp
    pa = float(correct[pos].mean()) if pos.any() else math.nan
    na = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "n_slots": int(eligible.sum()),
        "direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
        "all_negative_baseline": float(neg[eligible].mean()) if eligible.any() else math.nan,
    }


class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.reconstruct = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc_seq, (h, _) = self.encoder(x)
        repeated = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        dec_seq, _ = self.decoder(repeated)
        recon = self.reconstruct(dec_seq)
        return recon, enc_seq

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        enc_seq, _ = self.encoder(x)
        return enc_seq


def prepare_days(frame: pd.DataFrame, features: list[str], days: list[str], scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    part = frame[frame["target_day"].astype(str).isin(days)].copy().sort_values(["target_day", "hour_business"])
    counts = part.groupby("target_day").size()
    bad = counts[counts != 24]
    if len(bad):
        raise ValueError(f"incomplete 24-point days: {bad.head().to_dict()}")
    X = part[features].apply(pd.to_numeric, errors="coerce")
    if scaler is None:
        med = X.median(axis=0, skipna=True).fillna(0.0)
        scaler = StandardScaler().fit(X.fillna(med))
        scaler._efm3_medians = med.to_numpy(float)  # experiment-local convenience
    med_arr = np.asarray(getattr(scaler, "_efm3_medians"), float)
    arr = X.to_numpy(float)
    miss = ~np.isfinite(arr)
    if miss.any():
        arr[miss] = np.take(med_arr, np.where(miss)[1])
    arr = scaler.transform(arr)
    return arr.reshape(len(days), 24, len(features)).astype(np.float32), scaler


def fit_autoencoder(train_seq: np.ndarray, hidden: int, epochs: int, seed: int) -> tuple[LSTMAutoencoder, list[float]]:
    set_seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    x = torch.from_numpy(train_seq)
    model = LSTMAutoencoder(train_seq.shape[-1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x))
        epoch_loss = 0.0
        n = 0
        for start in range(0, len(x), 16):
            batch = x[order[start : start + 16]]
            opt.zero_grad(set_to_none=True)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach()) * len(batch)
            n += len(batch)
        losses.append(epoch_loss / max(n, 1))
    return model.eval(), losses


def encode(model: LSTMAutoencoder, seq: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = model.encode(torch.from_numpy(seq)).cpu().numpy()
    return z.reshape(-1, z.shape[-1])


def classifiers(seed: int) -> dict[str, object]:
    return {
        "rf": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=seed,
        ),
        "svc": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "logistic": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed),
        "sgb": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
            random_state=seed,
        ),
    }


def fit_hybrid(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    probs: dict[str, np.ndarray] = {}
    votes = []
    for name, model in classifiers(seed).items():
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X_test))[:, 1]
        else:
            score = np.asarray(model.decision_function(X_test), float)
            p = 1.0 / (1.0 + np.exp(-score))
        probs[name] = p
        votes.append(np.where(p >= 0.5, 1, -1))
    vote_mat = np.stack(votes, axis=1)
    majority = np.where(vote_mat.sum(axis=1) > 0, 1, -1)
    mean_prob = np.mean(np.stack(list(probs.values()), axis=1), axis=1)
    return probs, majority, mean_prob


def fit_one_day(frame: pd.DataFrame, features: list[str], all_days: list[str], day: str, training_days: int, ae_epochs: int, hidden: int, seed: int) -> tuple[pd.DataFrame, dict]:
    train_days = strict_train_days(all_days, day, training_days)
    train = frame[frame["target_day"].astype(str).isin(train_days)].copy().sort_values(["target_day", "hour_business"])
    test = frame[frame["target_day"].astype(str).eq(day)].copy().sort_values("hour_business")
    if len(test) != 24:
        raise ValueError(f"{day}: expected 24 test rows, got {len(test)}")
    y_train = (train["target_spread"].to_numpy(float) > 0).astype(int)

    train_seq, scaler = prepare_days(frame, features, train_days)
    test_seq, _ = prepare_days(frame, features, [day], scaler)
    ae, losses = fit_autoencoder(train_seq, hidden, ae_epochs, seed)
    Xae_train = encode(ae, train_seq)
    Xae_test = encode(ae, test_seq)

    # Paper-style: classifiers act on representation learned by LSTM autoencoder.
    ae_probs, ae_vote, ae_mean_prob = fit_hybrid(Xae_train, y_train, Xae_test, seed)

    # Control: same five-classifier hybrid on strictly legal raw P6 features without AE.
    Xraw_train = train_seq.reshape(-1, train_seq.shape[-1])
    Xraw_test = test_seq.reshape(-1, test_seq.shape[-1])
    raw_probs, raw_vote, raw_mean_prob = fit_hybrid(Xraw_train, y_train, Xraw_test, seed + 100)

    # Transfer check: retain raw features and append AE representation; this is an adapted diagnostic,
    # not the primary paper-style score.
    Xplus_train = np.concatenate([Xraw_train, Xae_train], axis=1)
    Xplus_test = np.concatenate([Xraw_test, Xae_test], axis=1)
    _, plus_vote, plus_mean_prob = fit_hybrid(Xplus_train, y_train, Xplus_test, seed + 200)

    out = test[["target_day", "hour_business", "period", "target_spread"]].copy().rename(columns={"target_spread": "y_true_spread"})
    out["dinler_ae_vote_direction"] = ae_vote
    out["dinler_ae_mean_prob"] = ae_mean_prob
    out["dinler_raw_vote_direction"] = raw_vote
    out["dinler_raw_mean_prob"] = raw_mean_prob
    out["dinler_raw_plus_ae_vote_direction"] = plus_vote
    out["dinler_raw_plus_ae_mean_prob"] = plus_mean_prob
    for name, p in ae_probs.items():
        out[f"ae_{name}_prob"] = p
    for name, p in raw_probs.items():
        out[f"raw_{name}_prob"] = p
    out["strict_train_last_day"] = train_days[-1]

    diag = {
        "target_day": day,
        "strict_train_last_day": train_days[-1],
        "ae_loss_first": float(losses[0]),
        "ae_loss_last": float(losses[-1]),
        "ae_hidden": hidden,
        "ae_epochs": ae_epochs,
    }
    return out, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--ae-epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    require_preflight()
    outdir = PROJECT_ROOT / args.output
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    slot, features = load_p6_features(PROJECT_ROOT / args.cube_root)
    slot["target_day"] = slot["target_day"].astype(str)
    all_days = sorted(slot["target_day"].unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    if not target_days:
        raise ValueError("no target days")

    t0 = time.perf_counter()
    parts = []
    diags = []
    for i, day in enumerate(target_days, 1):
        pred, diag = fit_one_day(slot, features, all_days, day, args.training_days, args.ae_epochs, args.hidden, args.seed)
        parts.append(pred)
        diags.append(diag)
        if i % 5 == 0 or i == len(target_days):
            print(f"dinler-adapted {i}/{len(target_days)}: {day}")

    ledger = pd.concat(parts, ignore_index=True)
    atomic_parquet(outdir / "ledger.parquet", ledger)
    atomic_csv(outdir / "ae_training_diagnostics.csv", pd.DataFrame(diags))

    summary_rows = []
    for name, col in [
        ("dinler_ae_hybrid_vote", "dinler_ae_vote_direction"),
        ("dinler_raw_hybrid_vote", "dinler_raw_vote_direction"),
        ("dinler_raw_plus_ae_hybrid_vote", "dinler_raw_plus_ae_vote_direction"),
    ]:
        summary_rows.append({"model": name, **metrics(ledger["y_true_spread"], ledger[col])})
    for clf in ["rf", "svc", "logistic", "sgb", "xgboost"]:
        summary_rows.append({"model": f"ae_{clf}", **metrics(ledger["y_true_spread"], np.where(ledger[f"ae_{clf}_prob"] >= 0.5, 1, -1))})
    summary = pd.DataFrame(summary_rows)
    atomic_csv(outdir / "summary.csv", summary)

    period_rows = []
    for period, g in ledger.groupby("period", sort=False):
        for name, col in [
            ("dinler_ae_hybrid_vote", "dinler_ae_vote_direction"),
            ("dinler_raw_hybrid_vote", "dinler_raw_vote_direction"),
            ("dinler_raw_plus_ae_hybrid_vote", "dinler_raw_plus_ae_vote_direction"),
        ]:
            period_rows.append({"period": period, "model": name, **metrics(g["y_true_spread"], g[col])})
    atomic_csv(outdir / "period_metrics.csv", pd.DataFrame(period_rows))

    manifest = {
        "status": "complete",
        "experiment": "dinler_2021_applied_energy_shandong_adapted_reproduction",
        "paper": "Reducing balancing cost of a wind power plant by deep learning in market data: A case study for Turkey",
        "doi": "10.1016/j.apenergy.2021.116728",
        "paper_reported_best_accuracy": 0.6108,
        "paper_mechanism": "LSTM autoencoder + RF/SVC/Logistic/SGB/XGBoost hybrid binary classifier",
        "reproduction_scope": "mechanism and next-day sign task adapted to Shandong strict P6 features; not identical raw Turkish feature schema",
        "dataset": "Shandong 24-point canonical/P6",
        "forecast_origin": "D-1 14:00",
        "target": "sign(RT-DA) for all 24 hours of D",
        "start": args.start,
        "end": args.end,
        "training_days": args.training_days,
        "ae_epochs": args.ae_epochs,
        "ae_hidden": args.hidden,
        "seed": args.seed,
        "feature_count": len(features),
        "target_day_actual_features": False,
        "target_day_da_input": False,
        "dminus1_post14_realized_input": False,
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(outdir / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
