"""Cycle 20 A pilot: Similar-Day distance plus historical state-label consistency.

For target D, all candidate days are <= D-2.  Their complete historical spread
sign curves may therefore be used as candidate labels, as allowed by the strict
Similar-Day contract.  The target day's spread labels are never used to construct
the consensus, filter candidates, or fit a model.
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


def state_vectors(slot: pd.DataFrame) -> dict[str, np.ndarray]:
    days, day_map = complete_days(slot)
    out = {}
    for day in days:
        values = pd.to_numeric(day_map[day]["target_spread"], errors="coerce").to_numpy(float)
        if len(values) != 24 or not np.isfinite(values).all():
            continue
        out[day] = np.where(values >= 0.0, 1, -1).astype(int)
    return out


def consensus_filter(
    cand: list[str], dist: np.ndarray, vectors: dict[str, np.ndarray], consensus_k: int,
    mode: str, alpha: float,
) -> tuple[list[str], dict[str, float], np.ndarray]:
    order = np.argsort(np.asarray(dist, dtype=float))
    seed_idx = order[: min(consensus_k, len(order))]
    seed_days = [cand[i] for i in seed_idx if cand[i] in vectors]
    if len(seed_days) < 60:
        raise RuntimeError(f"consensus seed has only {len(seed_days)} complete labeled days")
    seed_dist = np.asarray([dist[cand.index(d)] for d in seed_days], dtype=float)
    weights = 1.0 / np.maximum(seed_dist, 1e-4)
    weights /= weights.sum()
    mat = np.stack([vectors[d] for d in seed_days])
    positive_prob = np.sum(weights[:, None] * (mat > 0), axis=0)
    consensus = np.where(positive_prob >= 0.5, 1, -1)
    agreement = {d: float(np.mean(vectors[d] == consensus)) for d in cand if d in vectors}
    if mode == "distance_top":
        selected_pool = list(cand)
        score = np.asarray(dist, dtype=float)
    elif mode.startswith("agreement_q"):
        q = float(mode.replace("agreement_q", ""))
        values = np.asarray(list(agreement.values()), dtype=float)
        cutoff = float(np.nanquantile(values, q))
        selected_pool = [d for d in cand if d in agreement and agreement[d] >= cutoff]
        score = np.asarray([dist[cand.index(d)] for d in selected_pool], dtype=float)
    elif mode.startswith("combined_a"):
        selected_pool = list(cand)
        d = np.asarray(dist, dtype=float)
        d_norm = (d - np.nanmedian(d)) / max(float(np.nanstd(d)), 1e-6)
        a = np.asarray([agreement.get(day, 0.0) for day in cand], dtype=float)
        score = d_norm + alpha * (1.0 - a)
    else:
        raise ValueError(mode)
    return selected_pool, agreement, score


def summarize(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = ledger.assign(month=ledger["target_day"].str[:7])
    for variant, g in x.groupby("variant", sort=True):
        rows.append({"variant": variant, "month": "OVERALL", "days": int(g["target_day"].nunique()), **direction_metrics(
            g["target_spread"].to_numpy(float), g["predicted_direction"].to_numpy(int)
        )})
        for month, mg in g.groupby("month", sort=True):
            rows.append({"variant": variant, "month": month, "days": int(mg["target_day"].nunique()), **direction_metrics(
                mg["target_spread"].to_numpy(float), mg["predicted_direction"].to_numpy(int)
            )})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube"))
    parser.add_argument("--raw-path", type=Path, default=Path("data/24/canonical/shandong_pmos_hourly.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-08-14")
    parser.add_argument("--pool-days", type=int, default=365)
    parser.add_argument("--consensus-k", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--modes", default="distance_top,agreement_q0.25,agreement_q0.50,combined_a0.25,combined_a0.50")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    if pd.Timestamp(args.end) >= FINAL_HOLDOUT_START:
        raise RuntimeError("fresh final holdout remains sealed")
    cube, out = args.cube.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    slot["target_day"] = pd.to_datetime(slot["target_day"]).dt.strftime("%Y-%m-%d")
    groups = json.loads((cube / "feature_groups.json").read_text(encoding="utf-8"))
    slot, context_cols, context_audit = add_visible_context(slot, str(args.raw_path))
    slot, sd = add_similar_day_features(slot, groups, k_values=(20,), lookback_days=args.pool_days)
    if sd["audit"].empty or not sd["audit"]["causal_ok"].all():
        raise RuntimeError("Similar-Day causal audit failed")
    all_days = sorted(slot["target_day"].dropna().astype(str).unique())
    target_days = [d for d in all_days if args.start <= d <= args.end]
    selection, selection_audit = build_selection_map(slot, groups, context_cols, target_days, args.pool_days)
    vectors = state_vectors(slot)
    base = dedupe(p6_features(groups) + [c for c in sd["features"] if c.startswith("sd20_")])
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ledgers, audits = [], []
    for day in target_days:
        cand, dist = selection[(day, "profile")]
        for mode in modes:
            alpha = float(mode.replace("combined_a", "")) if mode.startswith("combined_a") else 0.0
            filtered, agreement, score = consensus_filter(cand, dist, vectors, args.consensus_k, mode, alpha)
            if len(filtered) < args.top_k:
                raise RuntimeError(f"{day}/{mode}: only {len(filtered)} candidates remain")
            selected = select_top(filtered, score, args.top_k)
            cutoff = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            latest = max(selected)
            if latest > cutoff:
                raise RuntimeError(f"{day}/{mode}: selected latest {latest} > {cutoff}")
            variant = f"state_consistency_{mode}_top{args.top_k}"
            ledgers.append(fit_one(slot, base, day, selected, variant, args.seed))
            audits.append({
                "target_day": day, "mode": mode, "candidate_count": len(cand),
                "filtered_candidate_count": len(filtered), "selected_count": len(selected),
                "latest_candidate_day": latest, "required_latest_candidate_le": cutoff,
                "causal_ok": latest <= cutoff,
                "selected_agreement_mean": float(np.mean([agreement.get(d, 0.0) for d in selected])),
                "candidate_label_source": "complete historical spread labels <= D-2",
            })
    ledger = pd.concat(ledgers, ignore_index=True)
    summary = summarize(ledger)
    manifest = {
        "status": "STRICT/PASS", "experiment_status": "CANDIDATE", "route": "A_strict_DSA",
        "forecast_origin": "D-1 14:00", "training_last_day": "selected complete days <= D-2",
        "target_day_actual_as_feature": False, "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False, "similar_day_latest_candidate": "<= D-2",
        "candidate_label_usage": "historical candidate labels allowed only for candidates <= D-2; target labels excluded",
        "final_holdout_touched": False, "screen_range": [args.start, args.end],
        "pool_days": args.pool_days, "consensus_k": args.consensus_k, "top_k": args.top_k,
        "modes": modes, "note": "pilot only; no production integration",
    }
    ledger.to_parquet(out / "ledger.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(out / "state_consistency_audit.csv", index=False, encoding="utf-8-sig")
    selection_audit.to_csv(out / "similar_day_selection_audit.csv", index=False, encoding="utf-8-sig")
    context_audit.to_csv(out / "visible_context_causal_audit.csv", index=False, encoding="utf-8-sig")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary[summary["month"].eq("OVERALL")].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
