"""Goal-70 iteration 2: deepen causal similar-day learning and test DART regime mixtures.

Experiment only. The runner uses the strict hourly Feature Cube produced at D-1 14:00.
No target-day DA/RT/spread/actual values enter any feature. Complete historical spread
labels used by analog construction are restricted to <= D-2.

Two literature-inspired branches are tested on one fixed protocol:
A) Shandong DSA deepening: three causal analog notions (level, shape, same-weekday),
   locally weighted LightGBM, and compact top-feature variants.
B) DART regime mixture: a three-state negative-tail/regular/positive-tail model whose
   covariate-dependent probabilities are blended with the direct P6 sign classifier.

The fresh 2026-08-15..2026-08-21 holdout is NOT evaluated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    CORE_SD_TOKENS,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    dedupe,
    direction_metrics,
    p6_features,
    strict_train_days,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _day_maps(slot: pd.DataFrame):
    day_map = {}
    for d, g in slot.groupby("target_day", sort=True):
        gg = g.sort_values("hour_business")
        if len(gg) == 24 and gg["target_spread"].notna().all():
            day_map[str(d)] = gg
    return sorted(day_map), day_map


def _profile_cols(groups: dict[str, list[str]]) -> list[str]:
    cols = [c for c in groups["F2"] if any(t in c for t in CORE_SD_TOKENS)]
    cols += [c for c in groups["F3"] if c in {"residual_load_renew", "renewable_share", "bidding_space_ratio", "interconnect_share"}]
    return dedupe(cols)


def _shape_profile(g: pd.DataFrame, cols: list[str]) -> np.ndarray:
    a = g[cols].to_numpy(float)
    med = np.nanmedian(a, axis=0)
    a = np.where(np.isfinite(a), a, med[None, :])
    mu = np.mean(a, axis=0, keepdims=True)
    sd = np.std(a, axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((a - mu) / sd).reshape(-1)


def _distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    X2 = np.where(np.isfinite(X), X, med)
    q2 = np.where(np.isfinite(q), q, med)
    scale = np.nanstd(X2, axis=0)
    scale = np.where((~np.isfinite(scale)) | (scale < 1e-6), 1.0, scale)
    return np.sqrt(np.mean(((X2 - q2) / scale) ** 2, axis=1))


def _analog_stats(S: np.ndarray, dd: np.ndarray, prefix: str) -> dict[str, np.ndarray | float]:
    w = np.exp(-dd / max(float(np.nanmedian(dd)), 1e-6))
    if not np.isfinite(w).all() or float(w.sum()) <= 0:
        w = np.ones(len(dd), float)
    w = w / w.sum()
    rec: dict[str, np.ndarray | float] = {
        f"{prefix}_mean": np.nanmean(S, axis=0),
        f"{prefix}_median": np.nanmedian(S, axis=0),
        f"{prefix}_std": np.nanstd(S, axis=0),
        f"{prefix}_positive_rate": np.nanmean(S > 0, axis=0),
        f"{prefix}_weighted_positive_rate": np.sum((S > 0) * w[:, None], axis=0),
        f"{prefix}_weighted_spread": np.sum(S * w[:, None], axis=0),
        f"{prefix}_distance_mean": float(np.mean(dd)),
        f"{prefix}_distance_min": float(np.min(dd)),
        f"{prefix}_day_positive_rate": float(np.mean(S > 0)),
    }
    for p, sl in (("1_8", slice(0, 8)), ("9_16", slice(8, 16)), ("17_24", slice(16, 24))):
        rec[f"{prefix}_{p}_positive_rate"] = float(np.mean(S[:, sl] > 0))
    return rec


def add_multisource_analogs(slot: pd.DataFrame, groups: dict[str, list[str]], *, k: int = 20, lookback: int = 365) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict[str, np.ndarray]]:
    out = slot.copy()
    days, dm = _day_maps(out)
    pcols = _profile_cols(groups)
    level = {d: dm[d][pcols].to_numpy(float).reshape(-1) for d in days}
    shape = {d: _shape_profile(dm[d], pcols) for d in days}
    spread = {d: dm[d]["target_spread"].to_numpy(float) for d in days}

    day_rec: dict[str, dict[str, np.ndarray | float]] = {}
    audit = []
    target_distance_vectors: dict[str, np.ndarray] = {}
    for i, day in enumerate(days):
        cutoff = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        cand = [d for d in days[max(0, i - lookback - 2):i] if d <= cutoff]
        if len(cand) < max(k, 12):
            continue
        rec: dict[str, np.ndarray | float] = {}

        # A1: absolute forecast/fundamental level analogs.
        X = np.stack([level[d] for d in cand])
        dl = _distance(X, level[day])
        idx = np.argsort(dl)[:k]
        sel = [cand[j] for j in idx]
        rec.update(_analog_stats(np.stack([spread[d] for d in sel]), dl[idx], f"analog_level_k{k}"))

        # A2: normalized curve-shape analogs (removes level, keeps daily ramps/patterns).
        Xs = np.stack([shape[d] for d in cand])
        ds = _distance(Xs, shape[day])
        idxs = np.argsort(ds)[:k]
        sels = [cand[j] for j in idxs]
        rec.update(_analog_stats(np.stack([spread[d] for d in sels]), ds[idxs], f"analog_shape_k{k}"))

        # A3: weekday-consistent analogs, a calendar-constrained DSA branch.
        dow = pd.Timestamp(day).dayofweek
        wcand = [d for d in cand if pd.Timestamp(d).dayofweek == dow]
        if len(wcand) >= 12:
            Xw = np.stack([level[d] for d in wcand])
            dw = _distance(Xw, level[day])
            kw = min(k, len(wcand))
            idxw = np.argsort(dw)[:kw]
            selw = [wcand[j] for j in idxw]
            rec.update(_analog_stats(np.stack([spread[d] for d in selw]), dw[idxw], f"analog_weekday_k{kw}"))

        day_rec[day] = rec
        # Store distances to the complete causal candidate list for locally weighted training.
        target_distance_vectors[day] = pd.Series(dl, index=cand).to_dict()
        audit.append({
            "target_day": day,
            "latest_candidate_day": max(cand),
            "required_latest_candidate_le": cutoff,
            "causal_ok": max(cand) <= cutoff,
            "candidate_count": len(cand),
            "weekday_candidate_count": len(wcand),
        })

    new_cols: set[str] = set()
    pieces = []
    for day, g in out.groupby("target_day", sort=False):
        gg = g.copy()
        rec = day_rec.get(str(day))
        if rec:
            for name, val in rec.items():
                if isinstance(val, np.ndarray):
                    if len(gg) == 24:
                        gg[name] = val
                else:
                    gg[name] = val
                new_cols.add(name)
        pieces.append(gg)
    out = pd.concat(pieces, ignore_index=True)
    aud = pd.DataFrame(audit)
    if aud.empty or not aud["causal_ok"].all():
        raise RuntimeError("multisource analog causal audit failed")
    return out, sorted(new_cols), aud, target_distance_vectors


def _lgbm(seed: int = 42, class_weight="balanced", *, multiclass: bool = False):
    kw = dict(
        n_estimators=190,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    if multiclass:
        return lgb.LGBMClassifier(objective="multiclass", num_class=3, **kw)
    return lgb.LGBMClassifier(objective="binary", class_weight=class_weight, **kw)


def _fit_predict_binary(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int, sample_weight=None, top_n: int | None = None) -> tuple[np.ndarray, list[str]]:
    y = (train["target_spread"].to_numpy(float) > 0).astype(int)
    used = list(features)
    if top_n is not None and top_n < len(used):
        fs = _lgbm(seed=seed)
        fs.fit(train[used], y, sample_weight=sample_weight)
        gain = fs.booster_.feature_importance(importance_type="gain")
        order = np.argsort(gain)[::-1]
        used = [used[i] for i in order[:top_n]]
    m = _lgbm(seed=seed)
    m.fit(train[used], y, sample_weight=sample_weight)
    return m.predict_proba(test[used])[:, 1].astype(float), used


def predict_analog_variant(slot: pd.DataFrame, features: list[str], target_days: list[str], distance_map: dict, *, name: str, training_days: int = 90, seed: int = 42, local_tau: float | None = None, top_n: int | None = None) -> pd.DataFrame:
    all_days = sorted(slot["target_day"].dropna().astype(str).unique())
    rows = []
    selected_feature_rows = []
    for day in target_days:
        train_days = strict_train_days(all_days, day, training_days)
        tr = slot[slot["target_day"].isin(train_days)].copy()
        te = slot[slot["target_day"].eq(day)].sort_values("hour_business").copy()
        if len(te) != 24 or len(train_days) < min(60, training_days):
            raise RuntimeError(f"{name} {day}: incomplete")
        sw = None
        if local_tau is not None:
            ddict = distance_map.get(day, {})
            dvals = np.asarray([ddict.get(d, np.nan) for d in train_days], float)
            finite = np.isfinite(dvals)
            fill = float(np.nanmedian(dvals[finite])) if finite.any() else 1.0
            dvals = np.where(finite, dvals, fill)
            scale = max(float(np.nanmedian(dvals)), 1e-6)
            day_w = np.exp(-dvals / (local_tau * scale))
            # Keep a small floor so recent-but-nonanalog days are not completely discarded.
            day_w = 0.15 + 0.85 * (day_w / max(float(day_w.max()), 1e-9))
            # Slight recency shrinkage to cope with price-regime drift.
            age = np.arange(len(train_days) - 1, -1, -1, dtype=float)
            day_w *= np.exp(-age / 180.0)
            wmap = dict(zip(train_days, day_w))
            sw = tr["target_day"].map(wmap).to_numpy(float)
        prob, used = _fit_predict_binary(tr, te, features, seed, sample_weight=sw, top_n=top_n)
        pred = np.where(prob >= 0.5, 1, -1)
        o = te[["target_day", "hour_business", "period", "target_spread"]].copy()
        o["variant"] = name
        o["prob_positive"] = prob
        o["predicted_direction"] = pred
        o["training_last_day"] = train_days[-1]
        o["training_days"] = len(train_days)
        rows.append(o)
        selected_feature_rows.append({"target_day": day, "variant": name, "n_features": len(used), "features": "|".join(used)})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(selected_feature_rows)


def predict_regime_mixture(slot: pd.DataFrame, features: list[str], target_days: list[str], *, q: float, alpha: float, seed: int = 42, training_days: int = 90) -> pd.DataFrame:
    """Covariate-dependent 3-state DART mixture; state thresholds are train-only."""
    all_days = sorted(slot["target_day"].dropna().astype(str).unique())
    rows = []
    for day in target_days:
        train_days = strict_train_days(all_days, day, training_days)
        tr = slot[slot.target_day.isin(train_days)].copy()
        te = slot[slot.target_day.eq(day)].sort_values("hour_business").copy()
        y_sp = tr["target_spread"].to_numpy(float)
        y_bin = (y_sp > 0).astype(int)
        base = _lgbm(seed=seed)
        base.fit(tr[features], y_bin)
        pbase = base.predict_proba(te[features])[:, 1]

        mag = np.abs(y_sp[np.isfinite(y_sp)])
        threshold = float(np.quantile(mag, q))
        state = np.ones(len(y_sp), dtype=int)  # regular=1
        state[y_sp <= -threshold] = 0
        state[y_sp >= threshold] = 2
        mix = _lgbm(seed=seed, multiclass=True)
        mix.fit(tr[features], state)
        pm = mix.predict_proba(te[features])
        # P(positive) = P(pos-tail) + P(regular)*P_base(positive|regular proxy).
        pmix = pm[:, 2] + pm[:, 1] * pbase
        p = (1.0 - alpha) * pbase + alpha * pmix
        o = te[["target_day", "hour_business", "period", "target_spread"]].copy()
        o["variant"] = f"regime_q{int(q*100):02d}_a{alpha:.2f}"
        o["prob_positive"] = p
        o["predicted_direction"] = np.where(p >= 0.5, 1, -1)
        o["regime_abs_threshold"] = threshold
        o["training_last_day"] = train_days[-1]
        o["training_days"] = len(train_days)
        rows.append(o)
    return pd.concat(rows, ignore_index=True)


def metrics_by_split(ledger: pd.DataFrame, splits: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for split, (start, end) in splits.items():
        p = ledger[(ledger.target_day >= start) & (ledger.target_day <= end)]
        for name, g in p.groupby("variant", sort=False):
            rows.append({"split": split, "variant": name, "days": int(g.target_day.nunique()), **direction_metrics(g.target_spread.to_numpy(), g.predicted_direction.to_numpy())})
    return pd.DataFrame(rows)


def calibrate_thresholds(ledger: pd.DataFrame, *, selection=("2026-07-16", "2026-07-30"), validation=("2026-07-31", "2026-08-14")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune thresholds on selection15 only; apply unchanged to validation15."""
    rows = []
    calibrated = []
    for name, allg in ledger.groupby("variant", sort=False):
        if allg["prob_positive"].isna().all():
            continue
        sel = allg[(allg.target_day >= selection[0]) & (allg.target_day <= selection[1])]
        val = allg[(allg.target_day >= validation[0]) & (allg.target_day <= validation[1])]
        for mode in ("global", "period"):
            ths = {}
            groups = [("all", sel)] if mode == "global" else list(sel.groupby("period", sort=False))
            for key, g in groups:
                y = np.sign(g.target_spread.to_numpy(float))
                best = None
                for t in np.arange(0.34, 0.661, 0.01):
                    pr = np.where(g.prob_positive.to_numpy(float) >= t, 1, -1)
                    met = direction_metrics(y, pr)
                    # Avoid buying raw accuracy by collapsing the positive class.
                    score = 0.55 * met["direction_accuracy"] + 0.45 * met["balanced_direction_accuracy"]
                    cand = (score, met["balanced_direction_accuracy"], met["direction_accuracy"], -abs(t - 0.5), float(t))
                    if best is None or cand > best:
                        best = cand
                ths[str(key)] = best[-1]
            for split_name, g in (("selection15", sel), ("validation15", val)):
                pred = []
                for _, r in g.iterrows():
                    t = ths["all"] if mode == "global" else ths[str(r.period)]
                    pred.append(1 if float(r.prob_positive) >= t else -1)
                met = direction_metrics(g.target_spread.to_numpy(), np.asarray(pred))
                rows.append({"variant": name, "mode": mode, "split": split_name, "thresholds": json.dumps(ths, ensure_ascii=False), **met})
                gg = g.copy()
                gg["variant"] = name + f"__cal_{mode}"
                gg["predicted_direction"] = pred
                gg["calibration_thresholds"] = json.dumps(ths, ensure_ascii=False)
                calibrated.append(gg)
    return pd.DataFrame(rows), pd.concat(calibrated, ignore_index=True) if calibrated else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/feature_cube")
    ap.add_argument("--output-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_goal70_20260822/iteration2_analog_regime")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cube = Path(args.cube_root)
    out = Path(args.output_root)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    cube_manifest = json.loads((cube / "manifest.json").read_text(encoding="utf-8"))
    base = p6_features(groups)
    slot, analog_cols, audit, dmap = add_multisource_analogs(slot, groups, k=20, lookback=365)
    atomic_csv(out / "analog_causal_audit.csv", audit)

    target_days = [d for d in sorted(slot.target_day.unique()) if "2026-07-16" <= d <= "2026-08-14"]
    ledgers = []
    feature_records = []
    p6_plus_analog = dedupe(base + analog_cols)

    # Keep the screen deliberately small: one direct representation, one local-weighted
    # model, and two embedded feature-selection sizes from the Shandong DSA paper.
    variants = [
        ("A_multianalog_w90", 90, None, None),
        ("A_multianalog_local_tau10_w180", 180, 1.0, None),
        ("A_multianalog_top60_w90", 90, None, 60),
        ("A_multianalog_top80_w90", 90, None, 80),
    ]
    for i, (name, w, tau, topn) in enumerate(variants, 1):
        print(f"[A {i}/{len(variants)}] {name}", flush=True)
        le, fr = predict_analog_variant(slot, p6_plus_analog, target_days, dmap, name=name, training_days=w, seed=args.seed, local_tau=tau, top_n=topn)
        ledgers.append(le)
        feature_records.append(fr)

    # Current iteration-1 champion representation: P6 + level analog k20.
    level20 = [c for c in analog_cols if c.startswith("analog_level_k20_")]
    champion_feats = dedupe(base + level20)
    for q, alpha in ((0.67, 0.50), (0.67, 0.75), (0.67, 1.00), (0.75, 0.75)):
        print(f"[B] regime q={q} alpha={alpha}", flush=True)
        ledgers.append(predict_regime_mixture(slot, champion_feats, target_days, q=q, alpha=alpha, seed=args.seed, training_days=90))

    # Exact iteration-1-style champion rebuilt on the new analog implementation for fair calibration.
    champ, fr = predict_analog_variant(slot, champion_feats, target_days, dmap, name="A_level20_champion_rebuild", training_days=90, seed=args.seed)
    ledgers.append(champ); feature_records.append(fr)

    ledger = pd.concat(ledgers, ignore_index=True)
    atomic_parquet(out / "ledger.parquet", ledger)
    splits = {
        "selection15": ("2026-07-16", "2026-07-30"),
        "validation15": ("2026-07-31", "2026-08-14"),
        "combined30": ("2026-07-16", "2026-08-14"),
    }
    summary = metrics_by_split(ledger, splits)
    atomic_csv(out / "summary.csv", summary)
    cal_summary, cal_ledger = calibrate_thresholds(ledger)
    atomic_csv(out / "calibration_summary.csv", cal_summary)
    if not cal_ledger.empty:
        atomic_parquet(out / "calibrated_ledger.parquet", cal_ledger)
    if feature_records:
        atomic_csv(out / "selected_features_by_day.csv", pd.concat(feature_records, ignore_index=True))

    # Ranking uses untouched validation15 first, balanced metric second, and selection/validation stability third.
    rows = []
    for name in summary.variant.unique():
        s = summary[(summary.variant == name) & (summary.split == "selection15")].iloc[0]
        v = summary[(summary.variant == name) & (summary.split == "validation15")].iloc[0]
        rows.append({
            "variant": name,
            "selection_acc": s.direction_accuracy,
            "validation_acc": v.direction_accuracy,
            "selection_bal": s.balanced_direction_accuracy,
            "validation_bal": v.balanced_direction_accuracy,
            "score": 0.55 * v.direction_accuracy + 0.35 * v.balanced_direction_accuracy + 0.10 * min(s.direction_accuracy, v.direction_accuracy),
        })
    ranking = pd.DataFrame(rows).sort_values(["score", "validation_acc", "validation_bal"], ascending=False)
    atomic_csv(out / "ranking.csv", ranking)
    atomic_json(out / "manifest.json", {
        "status": "complete",
        "experiment": "goal70_iteration2_analog_regime",
        "forecast_origin": "D-1 14:00",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "analog_history_latest": "D-2",
        "screen_range": ["2026-07-16", "2026-08-14"],
        "fresh_final_holdout_reserved": ["2026-08-15", "2026-08-21"],
        "final_holdout_touched": False,
        "cube_sha256": _sha256(cube / "slot_table.parquet"),
        "cube_information_boundary": cube_manifest.get("information_boundary"),
        "branch_A": "three-kind causal day similarity + local weighting + per-target train-only feature selection",
        "branch_B": "covariate-dependent three-state DART regime mixture + selection-only probability threshold calibration",
        "literature_basis": [
            "Huang et al., Applied Energy 2024, Shandong DSA + embedded feature selection + DNN",
            "Forgetta et al., Energy Economics 2025, covariate-dependent regular/positive-spike/negative-spike DART mixture",
            "Galarneau-Vincent et al., Energy Economics 2023, tailored DART spread spike features and probability thresholds",
        ],
        "runtime_seconds": time.perf_counter() - t0,
    })
    print("\nRAW VALIDATION TOP", flush=True)
    print(summary[summary.split.eq("validation15")].sort_values(["direction_accuracy", "balanced_direction_accuracy"], ascending=False).head(12).to_string(index=False), flush=True)
    print("\nCALIBRATED VALIDATION TOP", flush=True)
    print(cal_summary[cal_summary.split.eq("validation15")].sort_values(["direction_accuracy", "balanced_direction_accuracy"], ascending=False).head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
