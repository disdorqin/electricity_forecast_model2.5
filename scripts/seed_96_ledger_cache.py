"""
Seed runs_96/<date>/<task>/prediction/<model>_predictions.csv from
output/prediction_96/*.csv so ledger_predict runs with cache hits
(avoids re-running all 7 models for the ledger chain test).

Usage:
  python scripts/seed_96_ledger_cache.py --date 2026-07-16

Re-runnable: missing files (e.g. rt916_96.csv before RT916 finishes)
are skipped; re-run after they appear.
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

from utils.business_day import standardize_business_columns  # noqa: E402

# model -> (prediction_96 file, da_feature_source)
FILES = {
    "dayahead": {
        "timesfm": ("timesfm_da_96.csv", "none"),
        "timemixer": ("timemixer_da_96.csv", "none"),
        "lightgbm": ("lightgbm_da_96.csv", "none"),
    },
    "realtime": {
        "timesfm": ("timesfm_rt_96.csv", "timesfm_none"),
        "timemixer": ("timemixer_rt_96.csv", "timemixer_internal_dayahead_prediction"),
        # NOTE: realtime 模型组不含 lightgbm（REALTIME_MODELS = 4 模型），
        # lightgbm_rt_96.csv 只用于 prediction_96 暂存，不 seed 进 realtime 缓存。
        "sgdfnet": ("sgdfnet_rt_96.csv", "sgdfnet_config_da_fill"),
        "rt916": ("rt916_96.csv", "rt916_internal_joint_dayahead_prediction"),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-07-16")
    parser.add_argument("--pred-dir", default="outputs/crawl/prediction_96")
    parser.add_argument("--runs-root", default="outputs/runs_96")
    parser.add_argument(
        "--write-equal-weights",
        action="store_true",
        default=False,
        help="Also write equal-weight weights.csv for the cold-start run "
             "(ledger_weight has no 30-day history for the first 96 run).",
    )
    args = parser.parse_args()

    DATE = args.date
    PRED_DIR = Path(args.pred_dir)
    RUNS_ROOT = Path(args.runs_root)

    seeded, skipped = 0, []
    for task, models in FILES.items():
        for model, (fname, da_source) in models.items():
            src = PRED_DIR / fname
            if not src.exists():
                skipped.append(fname)
                print(f"SKIP missing: {fname}")
                continue

            df = pd.read_csv(src)
            y_col = "预测值" if "预测值" in df.columns else (
                "prediction" if "prediction" in df.columns else None
            )
            if y_col is None:
                print(f"ERROR: no prediction column in {src}")
                continue

            df = standardize_business_columns(
                df,
                ds_col="时刻",
                y_pred_col=y_col,
                task_label=task,
                model_name=model,
                forecast_date=DATE,
                target_day=DATE,
                data_cutoff=(
                    f"{pd.Timestamp(DATE) - pd.Timedelta(days=1)}"
                    if task == "dayahead"
                    else f"{DATE} 14:00:00"
                ),
                run_id=f"{model}_v2_{DATE}",
                model_version="v2.0",
                resolution="15min",
            )
            df["da_feature_source"] = da_source

            out_path = RUNS_ROOT / DATE / task / "prediction" / f"{model}_predictions.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            seeded += 1
            print(f"seeded {out_path} ({len(df)} rows)")

    if args.write_equal_weights:
        _write_equal_weights(RUNS_ROOT, DATE)

    print(f"done: seeded {seeded}, skipped {len(skipped)}: {skipped}")
    return 0


def _write_equal_weights(runs_root: Path, date: str) -> None:
    """Cold-start bootstrap: equal weights for all models per period.

    ledger_weight needs 30 complete days of 96 history to learn weights;
    the first 96 run has none, so this writes an explicit equal-weight
    baseline so ledger_fuse can complete. Replaced by real weights once
    the ledger accumulates history.
    """
    periods = ["1_32", "33_64", "65_96"]
    rows = []
    for task, models in FILES.items():
        n = len(models)
        for model in models:
            for period in periods:
                rows.append({
                    "task": task,
                    "model_name": model,
                    "period": period,
                    "weight": round(1.0 / n, 6),
                })
    wdf = pd.DataFrame(rows, columns=["task", "model_name", "period", "weight"])
    for task in FILES:
        out = runs_root / date / task / "weight" / "weights.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        wdf[wdf["task"] == task].to_csv(out, index=False)
        print(f"wrote equal weights -> {out} ({len(wdf[wdf['task'] == task])} rows)")


if __name__ == "__main__":
    raise SystemExit(main())
