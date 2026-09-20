from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent / "paper_reproductions"
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

from common import atomic_csv, atomic_json, atomic_parquet, load_p6_features  # noqa: E402
from integrate_paper_modules_p6 import lgb_classifier  # noqa: E402
from integrate_paper_modules_p6_strict import strict_train_days  # noqa: E402

BASES = (
    "地方电厂总加",
    "联络线受电负荷",
    "风电总加",
    "光伏总加",
    "核电总加",
    "自备机组总加",
    "试验机组总加",
    "直调负荷",
    "竞价空间",
    "新能源总加",
)
CORE_BASES = ("风电总加", "光伏总加", "直调负荷", "竞价空间", "新能源总加", "联络线受电负荷")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    d = pd.to_numeric(b, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(a, errors="coerce") / d


def soften_probability(p: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)) / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))


def lgb_regressor(seed: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=160,
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


def cat_teacher(seed: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=180,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def build_privileged_table(project_root: Path, cube_root: Path, canonical_path: Path) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    slot, p6 = load_p6_features(cube_root)
    slot["target_day"] = slot["target_day"].astype(str)
    slot["时刻"] = pd.to_datetime(slot["时刻"], errors="raise")

    raw = pd.read_csv(canonical_path, encoding="gb18030")
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    required = ["时刻", "日前电价", "实时电价"]
    for base in BASES:
        required.extend([f"{base}预测值", f"{base}实际值"])
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"canonical missing columns: {missing}")
    raw = raw[required].copy()
    for c in required[1:]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    frame = slot.merge(raw, on="时刻", how="left", validate="one_to_one")
    implied = frame["实时电价"] - frame["日前电价"]
    spread_diff = np.nanmax(np.abs(implied.to_numpy(float) - frame["target_spread"].to_numpy(float)))
    if not np.isfinite(spread_diff) or spread_diff > 1e-6:
        raise RuntimeError(f"canonical/cube spread mismatch max={spread_diff}")

    groups: dict[str, list[str]] = {
        "actual_core": [],
        "realized_errors": [],
        "actual_physics": [],
        "da_state": [],
        "d1_evening": [],
        "intermediate_spread": [],
    }

    for base in BASES:
        acol = f"{base}实际值"
        fcol = f"{base}预测值"
        a = f"priv_actual_{base}"
        e = f"priv_error_{base}"
        frame[a] = frame[acol]
        frame[e] = frame[acol] - frame[fcol]
        groups["actual_core"].append(a)
        groups["realized_errors"].append(e)

    # Realized target-day physical state. These are teacher-only and never enter Student inference.
    load = frame["priv_actual_直调负荷"]
    wind = frame["priv_actual_风电总加"]
    solar = frame["priv_actual_光伏总加"]
    renew = frame["priv_actual_新能源总加"]
    space = frame["priv_actual_竞价空间"]
    inter = frame["priv_actual_联络线受电负荷"]
    phys = {
        "priv_actual_residual_load_ws": load - wind - solar,
        "priv_actual_residual_load_renew": load - renew,
        "priv_actual_renewable_share": safe_div(renew, load),
        "priv_actual_wind_share": safe_div(wind, load),
        "priv_actual_solar_share": safe_div(solar, load),
        "priv_actual_bidding_space_ratio": safe_div(space, load),
        "priv_actual_interconnect_share": safe_div(inter, load),
        "priv_actual_renewable_minus_space": renew - space,
    }
    for name, value in phys.items():
        frame[name] = value
        groups["actual_physics"].append(name)
    for short, source in {
        "load": "priv_actual_直调负荷",
        "wind": "priv_actual_风电总加",
        "solar": "priv_actual_光伏总加",
        "renewable": "priv_actual_新能源总加",
        "bidding_space": "priv_actual_竞价空间",
        "residual_load": "priv_actual_residual_load_renew",
    }.items():
        c = f"priv_actual_ramp_{short}"
        frame[c] = frame.groupby("target_day", sort=False)[source].diff().fillna(0.0)
        groups["actual_physics"].append(c)

    # Target-day DA market state is teacher-only in spread-v3 even though it may be observable in some market workflows.
    frame["priv_da_price"] = frame["日前电价"]
    frame["priv_da_ramp"] = frame.groupby("target_day", sort=False)["日前电价"].diff().fillna(0.0)
    frame["priv_da_daily_centered"] = frame["日前电价"] - frame.groupby("target_day", sort=False)["日前电价"].transform("mean")
    frame["priv_da_daily_rank"] = frame.groupby("target_day", sort=False)["日前电价"].rank(pct=True)
    groups["da_state"] = ["priv_da_price", "priv_da_ramp", "priv_da_daily_centered", "priv_da_daily_rank"]

    # D-1 p15-p24 realized spread is intermediate time-series privileged information.
    # It is never joined as Student input; only the historical Teacher sees it.
    day_spread = {
        str(day): g.sort_values("hour_business").set_index("hour_business")["target_spread"].to_dict()
        for day, g in slot.groupby("target_day", sort=False)
    }
    evening_rows = []
    for day in sorted(slot["target_day"].unique()):
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        vals = np.asarray([day_spread.get(prev, {}).get(h, np.nan) for h in range(15, 25)], float)
        rec: dict[str, float | str] = {"target_day": day}
        for j, h in enumerate(range(15, 25)):
            rec[f"priv_d1_spread_h{h:02d}"] = vals[j]
        finite = vals[np.isfinite(vals)]
        rec["priv_d1_evening_mean"] = float(np.mean(finite)) if len(finite) else np.nan
        rec["priv_d1_evening_std"] = float(np.std(finite)) if len(finite) else np.nan
        rec["priv_d1_evening_last"] = float(finite[-1]) if len(finite) else np.nan
        rec["priv_d1_evening_min"] = float(np.min(finite)) if len(finite) else np.nan
        rec["priv_d1_evening_max"] = float(np.max(finite)) if len(finite) else np.nan
        rec["priv_d1_evening_positive_rate"] = float(np.mean(finite > 0)) if len(finite) else np.nan
        if len(finite) >= 2:
            rec["priv_d1_evening_slope"] = float(np.polyfit(np.arange(len(finite)), finite, 1)[0])
        else:
            rec["priv_d1_evening_slope"] = np.nan
        evening_rows.append(rec)
    evening = pd.DataFrame(evening_rows)
    ev_cols = [c for c in evening.columns if c != "target_day"]
    groups["d1_evening"] = ev_cols
    frame = frame.merge(evening, on="target_day", how="left", validate="many_to_one")

    # R1/time-series-LUPI transfer: realized intermediate spread states after the forecast origin,
    # available only to the historical Teacher. Current target spread is NEVER used; every feature
    # is shifted so it contains only earlier physical slots than the current target slot.
    frame = frame.sort_values("时刻").reset_index(drop=True)
    frame["priv_intermediate_spread_lag1"] = frame["target_spread"].shift(1)
    frame["priv_intermediate_spread_lag2"] = frame["target_spread"].shift(2)
    prior = frame["target_spread"].shift(1)
    frame["priv_intermediate_spread_mean3"] = prior.rolling(3, min_periods=1).mean()
    frame["priv_intermediate_spread_std3"] = prior.rolling(3, min_periods=2).std(ddof=0)
    frame["priv_intermediate_spread_positive3"] = prior.gt(0).astype(float).rolling(3, min_periods=1).mean()
    frame["priv_intermediate_spread_abs_lag1"] = frame["priv_intermediate_spread_lag1"].abs()
    # Same-business-day expanding state: for h>1 this summarizes D h1..h-1; h1 remains unavailable.
    frame["priv_intermediate_day_mean"] = frame.groupby("target_day", sort=False)["target_spread"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    frame["priv_intermediate_day_positive_rate"] = frame.groupby("target_day", sort=False)["target_spread"].transform(
        lambda s: s.shift(1).gt(0).astype(float).expanding(min_periods=1).mean()
    )
    groups["intermediate_spread"] = [
        "priv_intermediate_spread_lag1", "priv_intermediate_spread_lag2",
        "priv_intermediate_spread_mean3", "priv_intermediate_spread_std3",
        "priv_intermediate_spread_positive3", "priv_intermediate_spread_abs_lag1",
        "priv_intermediate_day_mean", "priv_intermediate_day_positive_rate",
    ]

    # Explicitly remove answer-like raw columns from any candidate feature list.
    forbidden = {"实时电价", "target_spread", "target_direction"}
    all_priv = [c for v in groups.values() for c in v]
    if forbidden.intersection(all_priv):
        raise RuntimeError("direct answer column entered privileged feature set")
    if any(c.startswith("priv_") for c in p6):
        raise RuntimeError("Student regular feature list unexpectedly contains privileged feature")
    return frame, p6, groups


def fit_one_day(frame: pd.DataFrame, p6: list[str], groups: dict[str, list[str]], all_days: list[str], day: str, training_days: int, seed: int, include_intermediate_spread: bool = False) -> pd.DataFrame:
    train_days = strict_train_days(all_days, day, training_days)
    train = frame[frame["target_day"].isin(train_days)].copy()
    test = frame[frame["target_day"].eq(day)].sort_values("hour_business").copy()
    if len(test) != 24:
        raise ValueError(f"{day}: expected 24 slots, got {len(test)}")

    y = (train["target_spread"].to_numpy(float) > 0).astype(int)
    regular = p6
    priv_physical = groups["actual_core"] + groups["realized_errors"] + groups["actual_physics"]
    priv_full = priv_physical + groups["da_state"] + groups["d1_evening"]
    if include_intermediate_spread:
        priv_full = priv_full + groups["intermediate_spread"]

    base = lgb_classifier(seed).fit(train[regular], y)
    base_train = base.predict_proba(train[regular])[:, 1]
    base_test = base.predict_proba(test[regular])[:, 1]

    teacher_phys = lgb_classifier(seed + 11).fit(train[regular + priv_physical], y)
    teacher_phys_test = teacher_phys.predict_proba(test[regular + priv_physical])[:, 1]

    teacher = lgb_classifier(seed + 23).fit(train[regular + priv_full], y)
    teacher_train = teacher.predict_proba(train[regular + priv_full])[:, 1]
    teacher_test = teacher.predict_proba(test[regular + priv_full])[:, 1]

    cat = cat_teacher(seed + 37)
    cat.fit(train[regular + priv_full], y)
    cat_test = cat.predict_proba(test[regular + priv_full])[:, 1]

    teacher_soft = soften_probability(teacher_train, temperature=2.0)
    student_probs: dict[str, np.ndarray] = {}
    for alpha in (0.25, 0.50, 0.75):
        target = (1.0 - alpha) * y + alpha * teacher_soft
        m = lgb_regressor(seed + int(alpha * 100)).fit(train[regular], target)
        student_probs[f"student_soft_a{int(alpha*100):02d}_prob"] = np.clip(m.predict(test[regular]), 0.0, 1.0)

    mimic = lgb_regressor(seed + 91).fit(train[regular], teacher_soft)
    mimic_test = np.clip(mimic.predict(test[regular]), 0.0, 1.0)
    student_probs["student_mimic_t2_prob"] = mimic_test
    for lam in (0.25, 0.50, 0.75):
        student_probs[f"student_blend_mimic_l{int(lam*100):02d}_prob"] = (1.0 - lam) * base_test + lam * mimic_test

    # Teacher confidence weighting: privileged information changes training importance, not inference features.
    confidence = 2.0 * np.abs(teacher_train - 0.5)
    for beta in (1.0, 2.0):
        weights = 1.0 + beta * confidence
        m = lgb_classifier(seed + 101 + int(beta)).fit(train[regular], y, sample_weight=weights)
        student_probs[f"student_confweight_b{int(beta)}_prob"] = m.predict_proba(test[regular])[:, 1]

    out = test[["target_day", "hour_business", "period", "target_spread"]].copy()
    out = out.rename(columns={"target_spread": "y_true_spread"})
    out["base_p6_prob"] = base_test
    out["oracle_teacher_physical_lgb_prob"] = teacher_phys_test
    out["oracle_teacher_full_lgb_prob"] = teacher_test
    out["oracle_teacher_full_cat_prob"] = cat_test
    for name, prob in student_probs.items():
        out[name] = prob
    out["strict_train_last_day"] = train_days[-1]
    out["student_regular_feature_count"] = len(regular)
    out["teacher_physical_feature_count"] = len(regular) + len(priv_physical)
    out["teacher_full_feature_count"] = len(regular) + len(priv_full)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--canonical", default="data/24/canonical/shandong_pmos_hourly.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--include-intermediate-spread", action="store_true", help="Teacher-only shifted target/intermediate spread states; Student never receives them")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[4]
    cube_root = root / args.cube_root
    canonical = root / args.canonical
    outdir = root / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    frame, p6, groups = build_privileged_table(root, cube_root, canonical)
    all_days = sorted(frame["target_day"].unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    if not target_days:
        raise ValueError("no target days")

    rows = []
    audits = []
    for i, day in enumerate(target_days, 1):
        daily = fit_one_day(frame, p6, groups, all_days, day, args.training_days, args.seed, include_intermediate_spread=args.include_intermediate_spread)
        rows.append(daily)
        target_ts = pd.Timestamp(day)
        train_last = pd.Timestamp(daily["strict_train_last_day"].iloc[0])
        audits.append({
            "target_day": day,
            "strict_train_last_day": str(train_last.date()),
            "train_last_le_D_minus_2": bool(train_last <= target_ts - pd.Timedelta(days=2)),
            "student_uses_privileged_features": False,
            "oracle_teacher_uses_privileged_features": True,
            "n_student_features": len(p6),
            "n_privileged_physical": len(groups["actual_core"] + groups["realized_errors"] + groups["actual_physics"]),
            "n_privileged_full": len(groups["actual_core"] + groups["realized_errors"] + groups["actual_physics"] + groups["da_state"] + groups["d1_evening"] + (groups["intermediate_spread"] if args.include_intermediate_spread else [])),
        })
        if i % 10 == 0 or i == len(target_days):
            print(f"phase1 {i}/{len(target_days)}: {day}")

    ledger = pd.concat(rows, ignore_index=True)
    audit = pd.DataFrame(audits)
    if not audit["train_last_le_D_minus_2"].all():
        raise RuntimeError("training boundary audit failed")
    if audit["student_uses_privileged_features"].any():
        raise RuntimeError("Student privileged-feature audit failed")

    atomic_parquet(outdir / "ledger.parquet", ledger)
    atomic_csv(outdir / "information_boundary_audit.csv", audit)
    feature_manifest = {
        "student_regular_features": p6,
        "privileged_groups": groups,
        "forbidden_teacher_features": ["实时电价", "target_spread", "target_direction"],
    }
    atomic_json(outdir / "feature_manifest.json", feature_manifest)

    manifest = {
        "status": "complete",
        "experiment": "70pct_sprint_phase1_privileged_teacher_student",
        "dataset": "Shandong 24-point canonical only",
        "start": args.start,
        "end": args.end,
        "target_days": len(target_days),
        "training_days": args.training_days,
        "seed": args.seed,
        "student_inference_contract": "P6 regular features only; no priv_* columns",
        "oracle_contract": "teacher may read privileged historical/holdout columns solely to measure information ceiling; never deployable",
        "canonical_sha256": sha256(canonical),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "catboost": __import__("catboost").__version__,
        "include_intermediate_spread": bool(args.include_intermediate_spread),
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(outdir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
