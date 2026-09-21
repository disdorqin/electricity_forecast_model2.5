"""Cycle 19 A pilot: origin-visible quality filtering for Similar-Day candidates.

Candidate selection uses only target-day forecast-profile distance plus quality
signals that were already available at each candidate day's own D-1 14:00 origin:
missingness of forecast descriptors and historical forecast-error uncertainty.  No
candidate target-day spread label is used to filter or weight the training pool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.goal70_engineering.run_iteration9_regime_sample_selection import (
    build_selection_map,
    complete_days,
    fit_one,
    profile_columns,
    select_top,
)
from scripts.experiments.spread_direction_24.goal70_engineering.run_iteration6_visible_d1_context import add_visible_context
from scripts.experiments.spread_direction_24.goal70_engineering.run_model_screen import (
    add_similar_day_features,
    dedupe,
    direction_metrics,
    p6_features,
)


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def quality_map(slot: pd.DataFrame, groups: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Build candidate quality from origin-visible feature state only."""
    days, day_map = complete_days(slot)
    pcols = profile_columns(groups)
    uncertainty_cols = [
        c for c in groups.get("F5", [])
        if c.startswith("err_net_load_") and c.endswith(("_std", "_q90"))
    ]
    uncertainty_cols += [c for c in groups.get("F6", []) if c.endswith("_width")]
    uncertainty_cols = [c for c in dedupe(uncertainty_cols) if c in slot.columns]
    result = {}
    for day in days:
        g = day_map[day]
        profile_missing = float(g[pcols].apply(pd.to_numeric, errors="coerce").isna().mean().mean())
        if uncertainty_cols:
            u = g[uncertainty_cols].apply(pd.to_numeric, errors="coerce").abs().to_numpy(float)
            uncertainty = float(np.nanmedian(u)) if np.isfinite(u).any() else 0.0
        else:
            uncertainty = 0.0
        result[day] = {"profile_missing_rate": profile_missing, "uncertainty_level": uncertainty}
    return result


def filtered_candidates(cand: list[str], quality: dict[str, dict[str, float]], mode: str) -> tuple[list[str], dict[str, float]]:
    if mode == "none":
        return list(cand), {d: 0.0 for d in cand}
    missing = np.asarray([quality[d]["profile_missing_rate"] for d in cand], dtype=float)
    uncert = np.asarray([quality[d]["uncertainty_level"] for d in cand], dtype=float)
    # Thresholds are computed from the candidate pool visible at this target origin;
    # they are not learned from candidate target-day labels.
    miss_cut = float(np.nanquantile(missing, 0.75))
    unc_cut = float(np.nanquantile(uncert, 0.75))
    if mode == "missing_q75":
        keep = missing <= miss_cut
    elif mode == "uncertainty_q75":
        keep = uncert <= unc_cut
    elif mode == "composite_q75":
        m_scale = max(float(np.nanstd(missing)), 1e-9)
        u_scale = max(float(np.nanstd(uncert)), 1e-9)
        composite = (missing - np.nanmedian(missing)) / m_scale + (uncert - np.nanmedian(uncert)) / u_scale
        keep = composite <= float(np.nanquantile(composite, 0.75))
    else:
        raise ValueError(mode)
    selected = [d for d, flag in zip(cand, keep) if bool(flag)]
    score = {d: float(missing[i] + uncert[i] / max(float(np.nanmedian(uncert)), 1.0)) for i, d in enumerate(cand)}
    return selected, score


def summarize(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = ledger.assign(month=ledger["target_day"].str[:7])
    for (variant, month), g in x.groupby(["variant", "month"], sort=True):
        rows.append({"variant": variant, "month": month, "days": int(g["target_day"].nunique()), **direction_metrics(
            g["target_spread"].to_numpy(float), g["predicted_direction"].to_numpy(int)
        )})
    monthly = pd.DataFrame(rows)
    overall = []
    for variant, g in x.groupby("variant", sort=True):
        overall.append({"variant": variant, "month": "OVERALL", "days": int(g["target_day"].nunique()), **direction_metrics(
            g["target_spread"].to_numpy(float), g["predicted_direction"].to_numpy(int)
        )})
    return pd.concat([pd.DataFrame(overall), monthly], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube"))
    parser.add_argument("--raw-path", type=Path, default=Path("data/24/canonical/shandong_pmos_hourly.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--pool-days", type=int, default=365)
    parser.add_argument("--train-window", type=int, default=90)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--modes", default="none,missing_q75,uncertainty_q75,composite_q75")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end >= FINAL_HOLDOUT_START:
        raise RuntimeError("fresh final holdout remains sealed")
    cube, out = args.cube.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    # Keep the canonical string day key expected by the existing causal context
    # and Similar-Day helpers; timestamps are only used for boundary assertions.
    slot["target_day"] = pd.to_datetime(slot["target_day"]).dt.strftime("%Y-%m-%d")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    slot, context_cols, context_audit = add_visible_context(slot, str(args.raw_path))
    slot, sd = add_similar_day_features(slot, groups, k_values=(20,), lookback_days=args.pool_days)
    if sd["audit"].empty or not sd["audit"]["causal_ok"].all():
        raise RuntimeError("Similar-Day causal audit failed")
    all_days = sorted(slot["target_day"].dropna().unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    target_day_strings = target_days
    selection, selection_audit = build_selection_map(slot, groups, context_cols, target_day_strings, args.pool_days)
    quality = quality_map(slot, groups)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    base = dedupe(p6_features(groups) + [c for c in sd["features"] if c.startswith("sd20_")])
    ledgers, audits = [], []
    for day in target_day_strings:
        cand, dist = selection[(day, "profile")]
        for mode in modes:
            filtered, qscore = filtered_candidates(cand, quality, mode)
            if len(filtered) < args.top_k:
                raise RuntimeError(f"{day}/{mode}: quality filter leaves {len(filtered)} < top_k={args.top_k}")
            idx = np.asarray([cand.index(d) for d in filtered], dtype=int)
            filtered_dist = np.asarray(dist)[idx]
            selected = select_top(filtered, filtered_dist, args.top_k)
            latest = max(selected)
            cutoff = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            if latest > cutoff:
                raise RuntimeError(f"{day}/{mode}: selected latest {latest} > {cutoff}")
            frame = fit_one(slot, base, day, selected, f"quality_{mode}_top{args.top_k}", args.seed)
            ledgers.append(frame)
            audits.append({
                "target_day": day, "mode": mode, "candidate_count": len(cand),
                "filtered_candidate_count": len(filtered), "selected_count": len(selected),
                "latest_candidate_day": latest, "required_latest_candidate_le": cutoff,
                "causal_ok": latest <= cutoff,
                "selected_quality_mean": float(np.mean([qscore[d] for d in selected])),
            })
    ledger = pd.concat(ledgers, ignore_index=True)
    summary = summarize(ledger)
    out_manifest = {
        "status": "STRICT/PASS", "experiment_status": "CANDIDATE",
        "route": "A_strict_DSA", "forecast_origin": "D-1 14:00",
        "training_last_day": "selected complete days <= D-2",
        "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False, "similar_day_latest_candidate": "<= D-2",
        "final_holdout_touched": False, "screen_range": [args.start, args.end],
        "pool_days": args.pool_days, "train_window": args.train_window, "top_k": args.top_k,
        "quality_modes": modes, "quality_sources": ["forecast-profile missingness", "historical error uncertainty at candidate origin"],
        "note": "pilot only; candidate quality filter uses no candidate target-day labels",
    }
    ledger.to_parquet(out / "ledger.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(out / "quality_selection_audit.csv", index=False, encoding="utf-8-sig")
    selection_audit.to_csv(out / "similar_day_selection_audit.csv", index=False, encoding="utf-8-sig")
    context_audit.to_csv(out / "visible_context_causal_audit.csv", index=False, encoding="utf-8-sig")
    (out / "manifest.json").write_text(json.dumps(out_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[summary["month"].eq("OVERALL")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
