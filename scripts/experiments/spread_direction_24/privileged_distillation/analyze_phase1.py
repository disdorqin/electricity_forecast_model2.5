from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent / "paper_reproductions"
if str(PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_DIR))

from common import atomic_csv, atomic_json, atomic_parquet  # noqa: E402

THRESHOLDS = (0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60)
BLOCKS = {
    "A_early": ("2026-04-17", "2026-06-15"),
    "B_late": ("2026-06-16", "2026-08-14"),
}


def metrics(y_spread: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
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
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "direction_accuracy": float(correct[eligible].mean()),
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
    }


def best_rule(frame: pd.DataFrame, columns: list[str]) -> dict:
    y = frame["y_true_spread"].to_numpy(float)
    rows = []
    for c in columns:
        p = frame[c].to_numpy(float)
        for t in THRESHOLDS:
            m = metrics(y, p, t)
            rows.append({"probability_column": c, "threshold": t, **m})
    table = pd.DataFrame(rows)
    table["threshold_distance"] = (table["threshold"] - 0.5).abs()
    table = table.sort_values(
        ["direction_accuracy", "balanced_direction_accuracy", "threshold_distance"],
        ascending=[False, False, True],
    )
    return table.iloc[0].drop(labels=["threshold_distance"]).to_dict()


def add_split(frame: pd.DataFrame, block: str) -> pd.DataFrame:
    start, end = BLOCKS[block]
    part = frame[(frame["target_day"] >= start) & (frame["target_day"] <= end)].copy()
    days = sorted(part["target_day"].unique())
    if len(days) != 60:
        raise ValueError(f"{block}: expected 60 days, got {len(days)}")
    mapping = {d: ("design45" if i < 45 else "holdout15") for i, d in enumerate(days)}
    part["block"] = block
    part["split"] = part["target_day"].map(mapping)
    return part


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[4]
    parts_root = root / args.parts_root
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    part_files = sorted(parts_root.glob("part_*/ledger.parquet"))
    if not part_files:
        raise FileNotFoundError(f"no ledger parts under {parts_root}")
    frame = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
    frame["target_day"] = frame["target_day"].astype(str)
    frame = frame.drop_duplicates(["target_day", "hour_business"], keep="last").sort_values(["target_day", "hour_business"]).reset_index(drop=True)

    expected_days = pd.date_range("2026-04-17", "2026-08-14", freq="D").strftime("%Y-%m-%d").tolist()
    missing = sorted(set(expected_days) - set(frame["target_day"].unique()))
    if missing:
        raise ValueError(f"missing prediction days: {missing[:10]} ({len(missing)})")

    regular_cols = [
        "base_p6_prob",
        "student_soft_a25_prob",
        "student_soft_a50_prob",
        "student_soft_a75_prob",
        "student_mimic_t2_prob",
        "student_blend_mimic_l25_prob",
        "student_blend_mimic_l50_prob",
        "student_blend_mimic_l75_prob",
        "student_confweight_b1_prob",
        "student_confweight_b2_prob",
    ]
    oracle_cols = [
        "oracle_teacher_physical_lgb_prob",
        "oracle_teacher_full_lgb_prob",
        "oracle_teacher_full_cat_prob",
    ]
    for c in [*regular_cols, *oracle_cols]:
        if c not in frame.columns:
            raise ValueError(f"missing probability column {c}")

    block_frames = [add_split(frame, b) for b in BLOCKS]
    ledger = pd.concat(block_frames, ignore_index=True)
    atomic_parquet(output / "ledger.parquet", ledger)

    # Default threshold comparison keeps model effects separate from threshold tuning.
    default_rows = []
    for (block, split), g in ledger.groupby(["block", "split"], sort=False):
        for c in [*regular_cols, *oracle_cols]:
            default_rows.append({"block": block, "split": split, "probability_column": c, "threshold": 0.5, **metrics(g["y_true_spread"].to_numpy(float), g[c].to_numpy(float), 0.5)})
    default_summary = pd.DataFrame(default_rows)
    atomic_csv(output / "default_summary.csv", default_summary)

    selected_rows = []
    selection_manifest = {}
    selected_ledgers = []
    for block in BLOCKS:
        part = ledger[ledger["block"].eq(block)].copy()
        design = part[part["split"].eq("design45")]
        hold = part[part["split"].eq("holdout15")].copy()

        # Base gets its own design-only threshold selection for a fair production-oriented comparator.
        base_rule = best_rule(design, ["base_p6_prob"])
        student_rule = best_rule(design, regular_cols)
        oracle_rule = best_rule(design, oracle_cols)
        selection_manifest[block] = {"base": base_rule, "student": student_rule, "oracle": oracle_rule}

        out = hold[["target_day", "hour_business", "period", "y_true_spread"]].copy()
        for role, rule in (("selected_base", base_rule), ("selected_student", student_rule), ("selected_oracle", oracle_rule)):
            c = str(rule["probability_column"])
            t = float(rule["threshold"])
            p = hold[c].to_numpy(float)
            pred = np.where(p >= t, 1, -1)
            out[f"{role}_prob"] = p
            out[f"{role}_direction"] = pred
            m = metrics(hold["y_true_spread"].to_numpy(float), p, t)
            selected_rows.append({
                "block": block,
                "role": role,
                "probability_column": c,
                "threshold": t,
                **m,
            })
        selected_ledgers.append(out.assign(block=block))

    selected = pd.DataFrame(selected_rows)
    selected_ledger = pd.concat(selected_ledgers, ignore_index=True)
    atomic_csv(output / "selected_holdout_summary.csv", selected)
    atomic_parquet(output / "selected_holdout_ledger.parquet", selected_ledger)
    atomic_json(output / "selected_rules.json", selection_manifest)

    # Holdout period diagnostics for selected strategies.
    period_rows = []
    for (block, period), g in selected_ledger.groupby(["block", "period"], sort=False):
        for role in ("selected_base", "selected_student", "selected_oracle"):
            y = g["y_true_spread"].to_numpy(float)
            pred = g[f"{role}_direction"].to_numpy(int)
            prob = np.where(pred > 0, 1.0, 0.0)
            period_rows.append({"block": block, "period": period, "role": role, **metrics(y, prob, 0.5)})
    period = pd.DataFrame(period_rows)
    atomic_csv(output / "selected_period_metrics.csv", period)

    # Pooled two-holdout score. Each block's selection remains based only on its own preceding design45.
    pooled_rows = []
    for role in ("selected_base", "selected_student", "selected_oracle"):
        y = selected_ledger["y_true_spread"].to_numpy(float)
        pred = selected_ledger[f"{role}_direction"].to_numpy(int)
        prob = np.where(pred > 0, 1.0, 0.0)
        pooled_rows.append({"role": role, **metrics(y, prob, 0.5)})
    pooled = pd.DataFrame(pooled_rows)
    atomic_csv(output / "pooled_holdout_summary.csv", pooled)

    base_map = selected[selected["role"].eq("selected_base")].set_index("block")
    student_map = selected[selected["role"].eq("selected_student")].set_index("block")
    oracle_map = selected[selected["role"].eq("selected_oracle")].set_index("block")
    acceptance_rows = []
    for block in BLOCKS:
        acceptance_rows.append({
            "block": block,
            "base_direction": float(base_map.loc[block, "direction_accuracy"]),
            "student_direction": float(student_map.loc[block, "direction_accuracy"]),
            "oracle_direction": float(oracle_map.loc[block, "direction_accuracy"]),
            "student_delta_pp": 100 * float(student_map.loc[block, "direction_accuracy"] - base_map.loc[block, "direction_accuracy"]),
            "oracle_headroom_pp": 100 * float(oracle_map.loc[block, "direction_accuracy"] - student_map.loc[block, "direction_accuracy"]),
            "student_hits_70": bool(student_map.loc[block, "direction_accuracy"] >= 0.70),
            "oracle_hits_70": bool(oracle_map.loc[block, "direction_accuracy"] >= 0.70),
            "oracle_hits_75": bool(oracle_map.loc[block, "direction_accuracy"] >= 0.75),
            "oracle_hits_80": bool(oracle_map.loc[block, "direction_accuracy"] >= 0.80),
        })
    acceptance = pd.DataFrame(acceptance_rows)
    atomic_csv(output / "phase1_acceptance.csv", acceptance)

    manifest = {
        "status": "complete",
        "dataset": "Shandong only",
        "parts": [str(p.relative_to(root)) for p in part_files],
        "selection": "candidate and threshold selected on each block design45 only; evaluated frozen on its holdout15",
        "main_metric": "direction_accuracy",
        "target": 0.70,
        "production_chain_touched": False,
        "acceptance": acceptance.to_dict("records"),
    }
    atomic_json(output / "manifest.json", manifest)

    print("\nDEFAULT HOLDOUT")
    print(default_summary[default_summary["split"].eq("holdout15")].to_string(index=False))
    print("\nSELECTED HOLDOUT")
    print(selected.to_string(index=False))
    print("\nPOOLED")
    print(pooled.to_string(index=False))
    print("\nACCEPTANCE")
    print(acceptance.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
