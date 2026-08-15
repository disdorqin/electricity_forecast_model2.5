"""
Generic 96-point model export to output/prediction_96/<model>_<task>_96.csv.

Runs one model's pipeline.predict_range with resolution="15min" for a single
target day, then exports ``时刻,prediction`` (matching the existing
prediction_96 staging files). Used for legs without a dedicated runner.

Usage (CPU models — default python is fine):
  python scripts/export_96_model.py --model lightgbm --target dayahead \
      --date 2026-07-16 --data-path data/shandong_pmos_96_full_v2.xlsx
  python scripts/export_96_model.py --model sgdfnet --target realtime \
      --date 2026-07-16 --data-path data/shandong_pmos_96_full_v2.xlsx
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

from runners.registry import get_model_pipeline  # noqa: E402
from utils.io import ensure_prediction_frame  # noqa: E402

TARGET_LABEL = {"dayahead": "日前电价", "realtime": "实时电价"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["lightgbm", "sgdfnet", "timemixer", "rt916"])
    parser.add_argument("--target", required=True, choices=["dayahead", "realtime"])
    parser.add_argument("--date", default="2026-07-16")
    parser.add_argument("--data-path", default="data/shandong_pmos_96_full_v2.xlsx")
    parser.add_argument("--cutoff-hour", type=int, default=14)
    parser.add_argument("--training-months", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # 命名沿用现有约定：<model>_<da|rt>_96.csv
    short_target = "da" if args.target == "dayahead" else "rt"
    out = args.out or f"outputs/crawl/prediction_96/{args.model}_{short_target}_96.csv"
    print(f"=== export_96: model={args.model} target={args.target} date={args.date} "
          f"data={args.data_path} -> {out} ===", flush=True)

    pl = get_model_pipeline(args.model)
    result = pl.predict_range(
        target=args.target,
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
        print(f"ERROR: {args.model}/{args.target} produced no predictions", flush=True)
        return 1

    df = result.frame
    print(f"rows: {len(df)}", flush=True)
    print(df.head(3).to_string(), flush=True)

    # Column name for the source prediction (may vary per pipeline)
    pred_col = "预测实时电价" if args.target == "realtime" else "预测日前电价"
    norm = ensure_prediction_frame(df, pred_col)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm.to_csv(out_path, index=False)
    print(f"OK: exported {len(norm)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
