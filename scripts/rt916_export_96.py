"""
RT916 96-point joint backtest runner + export to output/prediction_96/rt916_96.csv.

Runs the production joint DA->RT daily backtest at 15-min resolution for a
single target day (exactly what ledger_predict/_predict_rt916 will invoke once
the resolution kwarg is forwarded), then exports the RT prediction column as
``时刻,prediction`` matching the other prediction_96 staging files.

Usage (epf-2 GPU env):
  python scripts/rt916_export_96.py --date 2026-07-16 \\
      --data-path data/shandong_pmos_96_full_v2.xlsx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from RT916_SpikeFusionNet.pipeline import ModelPipeline  # noqa: E402
from utils.io import ensure_prediction_frame  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-07-16")
    parser.add_argument("--data-path", default="data/shandong_pmos_96_full_v2.xlsx")
    parser.add_argument("--cutoff-hour", type=int, default=14)
    parser.add_argument("--training-months", type=int, default=12)
    parser.add_argument("--out", default="output/prediction_96/rt916_96.csv")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run backtest only, print head + metrics; skip CSV export.",
    )
    args = parser.parse_args()

    print(f"=== RT916 96-point joint backtest | date={args.date} | "
          f"data={args.data_path} ===", flush=True)

    pl = ModelPipeline()
    result = pl.predict_range(
        target="realtime",
        resolution="15min",
        data_path=args.data_path,
        predict_date=args.date,
        start=args.date,
        end=args.date,
        realtime_cutoff_hour=args.cutoff_hour,
        training_months=args.training_months,
        seed=42,
        deterministic=True,
    )
    if result is None or result.frame is None or len(result.frame) == 0:
        print("ERROR: RT916 backtest produced no predictions", flush=True)
        return 1

    df = result.frame
    print(f"backtest rows: {len(df)}", flush=True)
    print(df.head(3).to_string(), flush=True)

    if not args.no_export:
        norm = ensure_prediction_frame(df, "预测实时电价")
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        norm.to_csv(out_path, index=False)
        print(f"OK: exported {len(norm)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
