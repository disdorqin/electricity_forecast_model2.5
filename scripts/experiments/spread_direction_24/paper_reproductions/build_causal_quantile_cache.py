from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import atomic_parquet, load_p6_features  # noqa: E402
from integrate_paper_modules_p6 import build_causal_quantiles  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube-root", default="outputs/experiments/01_spread_24/feature_model_screening/spread_feature_cube_v1_20260821")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--training-days", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[4]
    slot, p6 = load_p6_features(root / args.cube_root)
    slot["target_day"] = slot["target_day"].astype(str)
    all_days = sorted(slot["target_day"].dropna().unique())
    days = [d for d in all_days if args.start <= d <= args.end]
    if not days:
        raise ValueError("no days in requested range")
    q = build_causal_quantiles(slot, p6, all_days, days, args.training_days, args.seed)
    out = root / args.output
    atomic_parquet(out, q)
    print(f"wrote {out}: days={q.target_day.nunique()} rows={len(q)} range={q.target_day.min()}..{q.target_day.max()}")


if __name__ == "__main__":
    main()
