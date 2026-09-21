from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_r1_ahead import build_features, load_data

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = ROOT / "data/96/authoritative/pmos_96_全量.csv"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def feature_block(frame: pd.DataFrame, feats: list[str], day: str) -> pd.DataFrame:
    out = frame.loc[frame["market_date"].eq(day), ["slot", *feats]].sort_values("slot").reset_index(drop=True)
    if len(out) != 96:
        raise RuntimeError(f"{day}: expected 96 rows, got {len(out)}")
    return out


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    aa = a.drop(columns=["slot"]).to_numpy(float)
    bb = b.drop(columns=["slot"]).to_numpy(float)
    d = np.abs(aa - bb)
    finite = np.isfinite(d)
    return float(np.nanmax(d[finite])) if finite.any() else 0.0


def mutate_prices(raw: pd.DataFrame, day: str, slots: np.ndarray, delta: float) -> pd.DataFrame:
    out = raw.copy()
    mask = out["market_date"].eq(day) & out["slot"].isin(slots.tolist())
    out.loc[mask, "实时出清价格"] = pd.to_numeric(out.loc[mask, "实时出清价格"], errors="coerce") + delta
    out.loc[mask, "日前出清价格"] = pd.to_numeric(out.loc[mask, "日前出清价格"], errors="coerce") - delta * 0.37
    out.loc[mask, "spread"] = out.loc[mask, "实时出清价格"] - out.loc[mask, "日前出清价格"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA.relative_to(ROOT)))
    ap.add_argument("--days", nargs="+", default=["2026-02-15", "2026-04-15", "2026-06-15", "2026-08-14"])
    ap.add_argument("--output", default="outputs/experiments/02_spread_96/spread_direction_96/r1_chain/information_boundary_audit_v2")
    args = ap.parse_args()

    raw = load_data(ROOT / args.data)
    base, feats = build_features(raw)
    rows = []
    for day in args.days:
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        b = feature_block(base, feats, day)

        # Forbidden source 1: target-day realized DA/RT/spread. Must not affect target-day model features.
        mut_target = mutate_prices(raw, day, np.arange(1, 97), 10000.0)
        f_target, _ = build_features(mut_target)
        target_diff = max_abs_diff(b, feature_block(f_target, feats, day))

        # Forbidden source 2: D-1 post-cutoff realized prices/spread (p57..p96). Must not affect D features.
        mut_post = mutate_prices(raw, prev, np.arange(57, 97), 12000.0)
        f_post, _ = build_features(mut_post)
        post_diff = max_abs_diff(b, feature_block(f_post, feats, day))

        # Positive control: D-1 p1..p56 is intentionally available context and must affect at least one feature.
        mut_pre = mutate_prices(raw, prev, np.arange(1, 57), 14000.0)
        f_pre, _ = build_features(mut_pre)
        pre_diff = max_abs_diff(b, feature_block(f_pre, feats, day))

        rows.append({
            "target_day": day,
            "forecast_origin": f"{prev} 14:00",
            "target_day_price_perturbation_max_feature_diff": target_diff,
            "dminus1_post14_perturbation_max_feature_diff": post_diff,
            "dminus1_pre14_positive_control_max_feature_diff": pre_diff,
            "target_day_forbidden_invariance_pass": bool(target_diff <= 1e-10),
            "dminus1_post14_forbidden_invariance_pass": bool(post_diff <= 1e-10),
            "dminus1_pre14_positive_control_pass": bool(pre_diff > 1e-8),
        })

    audit = pd.DataFrame(rows)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "counterfactual_feature_audit.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "pass" if bool(audit[["target_day_forbidden_invariance_pass", "dminus1_post14_forbidden_invariance_pass", "dminus1_pre14_positive_control_pass"]].all().all()) else "fail",
        "forecast_origin": "D-1 14:00 / p56",
        "tested_days": args.days,
        "n_features": len(feats),
        "feature_names": feats,
        "forbidden_sources": ["target-day DA/RT/spread", "D-1 p57-p96 DA/RT/spread"],
        "allowed_positive_control": "D-1 p1-p56 spread context",
        "production_chain_touched": False,
    }
    atomic_json(out / "manifest.json", payload)
    print(audit.to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
