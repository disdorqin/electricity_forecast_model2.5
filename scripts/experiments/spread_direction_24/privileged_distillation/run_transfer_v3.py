from __future__ import annotations

"""Deployable Teacher-transfer experiment for Shandong 24-point spread direction.

This runner targets the transfer bottleneck found in Phase-1/V2:
- Oracle teacher with intermediate spread states is strong (>80% on the matched window).
- Plain global-alpha soft-target distillation does not transfer that gain.

Variants here remain deployable at inference:
1. base_p6: regular P6 only.
2. tpd_corrected: reliability-aware corrected teacher target; inference still P6 only.
3. state_proxy: predict training-only intermediate spread states from P6, then use the
   predicted states (never the realized privileged states) for direction prediction.
4. state_proxy_tpd: combine the predicted-state cascade with corrected teacher targets.

The target-day test row never exposes a realized privileged feature to a deployable model.
All training outputs are written only under outputs/experiments/.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent / "paper_reproductions"
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

from common import atomic_csv, atomic_json, atomic_parquet  # noqa: E402
from integrate_paper_modules_p6 import lgb_classifier  # noqa: E402
from integrate_paper_modules_p6_strict import strict_train_days  # noqa: E402
from run_phase1_daily import build_privileged_table, cat_teacher, lgb_regressor  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
INTERMEDIATE = [
    "priv_intermediate_spread_lag1",
    "priv_intermediate_spread_lag2",
    "priv_intermediate_spread_mean3",
    "priv_intermediate_spread_std3",
    "priv_intermediate_spread_positive3",
    "priv_intermediate_spread_abs_lag1",
    "priv_intermediate_day_mean",
    "priv_intermediate_day_positive_rate",
]


def require_preflight() -> None:
    """Honor AGENTS/efm3-lessons: no training/backtest unless project preflight is all green."""
    py = Path(sys.executable)
    proc = subprocess.run(
        [str(py), "scripts/tests/check_preflight_health.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-12:])
        raise RuntimeError("project preflight is not all green; refusing training/backtest\n" + tail)


def metrics(y_spread: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y = np.sign(np.asarray(y_spread, float))
    pred = np.where(np.asarray(prob, float) >= threshold, 1, -1)
    eligible = y != 0
    pos = y > 0
    neg = y < 0
    correct = y == pred
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


def proxy_regressor(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=140,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        n_jobs=4,
        random_state=seed,
    )


def teacher_oof_probs(train: pd.DataFrame, regular: list[str], privileged: list[str], seed: int) -> np.ndarray:
    """Day-grouped OOF teacher probabilities to avoid in-sample teacher overconfidence."""
    days = train["target_day"].astype(str)
    unique_days = days.nunique()
    n_splits = min(5, unique_days)
    if n_splits < 2:
        raise ValueError("not enough training days for teacher OOF")
    out = np.full(len(train), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    X = train[regular + privileged]
    y = (train["target_spread"].to_numpy(float) > 0).astype(int)
    for fold, (tr, va) in enumerate(splitter.split(X, y, groups=days), 1):
        model = lgb_classifier(seed + 100 + fold).fit(X.iloc[tr], y[tr])
        out[va] = model.predict_proba(X.iloc[va])[:, 1]
    if not np.isfinite(out).all():
        raise RuntimeError("teacher OOF probabilities incomplete")
    return out


def proxy_oof_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    regular: list[str],
    proxy_targets: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Cross-fit privileged-state proxies on train; fit full history for test inference."""
    groups = train["target_day"].astype(str)
    n_splits = min(5, groups.nunique())
    if n_splits < 2:
        raise ValueError("not enough training days for proxy OOF")
    splitter = GroupKFold(n_splits=n_splits)
    tr_proxy = pd.DataFrame(index=train.index)
    te_proxy = pd.DataFrame(index=test.index)
    diagnostics: dict[str, float] = {}

    for j, target in enumerate(proxy_targets):
        oof = np.full(len(train), np.nan, dtype=float)
        y = pd.to_numeric(train[target], errors="coerce").to_numpy(float)
        valid_all = np.isfinite(y)
        # Fold by day, but train each proxy only where its privileged regression target is observed.
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(train[regular], groups=groups), 1):
            tr_valid = tr_idx[valid_all[tr_idx]]
            if len(tr_valid) < 100:
                continue
            m = proxy_regressor(seed + 1000 + 31 * j + fold).fit(train.iloc[tr_valid][regular], y[tr_valid])
            oof[va_idx] = m.predict(train.iloc[va_idx][regular])
        # Any rare uncovered row is filled by a model trained on other rows only when possible.
        missing = ~np.isfinite(oof)
        if missing.any():
            good = valid_all & ~missing
            if good.sum() < 100:
                good = valid_all
            fallback = proxy_regressor(seed + 2000 + j).fit(train.loc[good, regular], y[good])
            oof[missing] = fallback.predict(train.loc[missing, regular])
        full = proxy_regressor(seed + 3000 + j).fit(train.loc[valid_all, regular], y[valid_all])
        test_pred = full.predict(test[regular])

        name = "proxy_" + target.removeprefix("priv_")
        tr_proxy[name] = oof
        te_proxy[name] = test_pred
        observed = valid_all & np.isfinite(oof)
        if observed.any():
            diagnostics[f"{name}_mae_oof"] = float(np.mean(np.abs(oof[observed] - y[observed])))
            denom = float(np.nanstd(y[observed]))
            diagnostics[f"{name}_nmae_oof"] = float(diagnostics[f"{name}_mae_oof"] / denom) if denom > 1e-12 else math.nan
    return tr_proxy.reset_index(drop=True), te_proxy.reset_index(drop=True), diagnostics


def corrected_teacher_target(y: np.ndarray, teacher_prob: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """TPD-inspired target: imitate correct teacher, invert teacher evidence on its mistakes.

    Distillation strength rises with teacher confidence. Ground-truth remains present for every row.
    """
    y = np.asarray(y, int)
    p = np.clip(np.asarray(teacher_prob, float), 1e-5, 1 - 1e-5)
    teacher_pred = (p >= 0.5).astype(int)
    corrected = np.where(teacher_pred == y, p, 1.0 - p)
    confidence = 2.0 * np.abs(p - 0.5)
    strength = float(alpha) * confidence
    return np.clip((1.0 - strength) * y + strength * corrected, 0.0, 1.0)


def fit_day(
    frame: pd.DataFrame,
    regular: list[str],
    privileged: list[str],
    all_days: list[str],
    target_day: str,
    training_days: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    train_days = strict_train_days(all_days, target_day, training_days)
    train = frame[frame["target_day"].isin(train_days)].copy().reset_index(drop=True)
    test = frame[frame["target_day"].eq(target_day)].copy().sort_values("hour_business").reset_index(drop=True)
    if len(test) != 24:
        raise ValueError(f"{target_day}: expected 24 slots, got {len(test)}")

    y_train = (train["target_spread"].to_numpy(float) > 0).astype(int)
    base = lgb_classifier(seed).fit(train[regular], y_train)
    base_prob = base.predict_proba(test[regular])[:, 1]

    # OOF teacher is used only to create training supervision; the target-day oracle is diagnostics only.
    teacher_oof = teacher_oof_probs(train, regular, privileged, seed)
    oracle = lgb_classifier(seed + 7).fit(train[regular + privileged], y_train)
    oracle_prob = oracle.predict_proba(test[regular + privileged])[:, 1]

    tpd_y = corrected_teacher_target(y_train, teacher_oof, alpha=0.55)
    tpd = lgb_regressor(seed + 19).fit(train[regular], tpd_y)
    tpd_prob = np.clip(tpd.predict(test[regular]), 0.0, 1.0)

    tr_proxy, te_proxy, proxy_diag = proxy_oof_and_test(train, test, regular, INTERMEDIATE, seed)
    proxy_cols = tr_proxy.columns.tolist()
    Xtr_proxy = pd.concat([train[regular].reset_index(drop=True), tr_proxy], axis=1)
    Xte_proxy = pd.concat([test[regular].reset_index(drop=True), te_proxy], axis=1)

    state_model = lgb_classifier(seed + 29).fit(Xtr_proxy, y_train)
    state_prob = state_model.predict_proba(Xte_proxy)[:, 1]

    proxy_tpd = lgb_regressor(seed + 41).fit(Xtr_proxy, tpd_y)
    proxy_tpd_prob = np.clip(proxy_tpd.predict(Xte_proxy), 0.0, 1.0)

    out = test[["target_day", "hour_business", "period", "target_spread"]].copy()
    out = out.rename(columns={"target_spread": "y_true_spread"})
    out["base_p6_prob"] = base_prob
    out["tpd_corrected_prob"] = tpd_prob
    out["state_proxy_prob"] = state_prob
    out["state_proxy_tpd_prob"] = proxy_tpd_prob
    out["oracle_teacher_prob"] = oracle_prob
    out["strict_train_last_day"] = train_days[-1]
    out["deployable_proxy_feature_count"] = len(proxy_cols)

    teacher_pred = (teacher_oof >= 0.5).astype(int)
    teacher_conf = 2.0 * np.abs(teacher_oof - 0.5)
    diag = {
        **proxy_diag,
        "teacher_oof_accuracy": float(np.mean(teacher_pred == y_train)),
        "teacher_oof_high_conf_share": float(np.mean(teacher_conf >= 0.8)),
        "teacher_oof_high_conf_accuracy": float(np.mean((teacher_pred == y_train)[teacher_conf >= 0.8])) if np.any(teacher_conf >= 0.8) else math.nan,
    }
    return out, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--canonical", default="data/24/canonical/shandong_pmos_hourly.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    require_preflight()
    outdir = PROJECT_ROOT / args.output
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    frame, regular, groups = build_privileged_table(
        PROJECT_ROOT,
        PROJECT_ROOT / args.cube_root,
        PROJECT_ROOT / args.canonical,
    )
    all_days = sorted(frame["target_day"].astype(str).unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    if not target_days:
        raise ValueError("no target days")

    # The V2 gain was dominated by shifted/intermediate spread states. Keep the next experiment focused.
    privileged = (
        groups["actual_core"]
        + groups["realized_errors"]
        + groups["actual_physics"]
        + groups["da_state"]
        + groups["d1_evening"]
        + groups["intermediate_spread"]
    )

    parts = []
    diag_rows = []
    for i, day in enumerate(target_days, 1):
        daily, diag = fit_day(frame, regular, privileged, all_days, day, args.training_days, args.seed)
        parts.append(daily)
        diag_rows.append({"target_day": day, **diag})
        if i % 5 == 0 or i == len(target_days):
            print(f"transfer-v3 {i}/{len(target_days)}: {day}")

    ledger = pd.concat(parts, ignore_index=True)
    diagnostics = pd.DataFrame(diag_rows)
    atomic_parquet(outdir / "ledger.parquet", ledger)
    atomic_csv(outdir / "proxy_diagnostics.csv", diagnostics)

    rows = []
    for col in ["base_p6_prob", "tpd_corrected_prob", "state_proxy_prob", "state_proxy_tpd_prob", "oracle_teacher_prob"]:
        rows.append({"model": col, **metrics(ledger["y_true_spread"].to_numpy(float), ledger[col].to_numpy(float))})
    summary = pd.DataFrame(rows)
    atomic_csv(outdir / "summary.csv", summary)

    period_rows = []
    for period, g in ledger.groupby("period", sort=False):
        for col in ["base_p6_prob", "tpd_corrected_prob", "state_proxy_prob", "state_proxy_tpd_prob", "oracle_teacher_prob"]:
            period_rows.append({"period": period, "model": col, **metrics(g["y_true_spread"].to_numpy(float), g[col].to_numpy(float))})
    atomic_csv(outdir / "period_metrics.csv", pd.DataFrame(period_rows))

    manifest = {
        "status": "complete",
        "experiment": "privileged_transfer_v3_tpd_and_state_proxy",
        "dataset": "Shandong 24-point canonical only",
        "forecast_origin": "D-1 14:00",
        "start": args.start,
        "end": args.end,
        "training_days": args.training_days,
        "seed": args.seed,
        "regular_feature_count": len(regular),
        "predicted_privileged_targets": INTERMEDIATE,
        "deployable_models": ["base_p6_prob", "tpd_corrected_prob", "state_proxy_prob", "state_proxy_tpd_prob"],
        "oracle_models": ["oracle_teacher_prob"],
        "student_target_day_privileged_input": False,
        "proxy_training": "day-grouped OOF on historical training rows; full historical fit for target-day proxy inference",
        "tpd_training": "day-grouped OOF teacher; correctness-corrected confidence-weighted soft target",
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(outdir / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
