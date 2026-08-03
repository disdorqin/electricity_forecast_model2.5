"""
重建 96 点 prediction ledger（不重跑模型）。

服务器上已跑出正确的 96 点预测 CSV（含 00:15 等 15min 档），但账本 append 时
因旧代码 keep_cols 漏 business_period，dedup key 只用 hour_business 把 96 点
压成 24 点。本脚本扫描 runs_96/<date>/{task}/prediction/*_predictions.csv，
用修复后的 append_predictions_to_ledger 重建整个 prediction ledger。

用法（服务器）:
  python scripts/rebuild_prediction_ledger_96.py --runs-root outputs/runs_96 \
      --ledger-root outputs/ledger_96
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.prediction_ledger import append_predictions_to_ledger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="outputs/runs_96")
    parser.add_argument("--ledger-root", default="outputs/ledger_96")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    ledger_root = Path(args.ledger_root)

    # 收集所有已生成的预测 CSV（按日期排序，保证 append 顺序稳定）
    files = sorted(runs_root.glob("*/*/prediction/*_predictions.csv"))
    files = [f for f in files if "all_model_predictions_long" not in f.name]
    if not files:
        print("未找到任何预测 CSV，请确认已跑过预热。")
        return 1

    print(f"找到 {len(files)} 个预测 CSV，开始重建 prediction ledger...")
    total_appended = 0
    for f in files:
        # 路径结构: runs_root/<date>/<task>/prediction/<model>_predictions.csv
        rel = f.relative_to(runs_root)
        date_str, task, _pred, fname = rel.parts
        model = fname.replace("_predictions.csv", "")

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  SKIP 读取失败 {f}: {e}")
            continue

        # 校验：96 点 CSV 必须有 96 行
        n = len(df)
        if n != 96:
            print(f"  WARN {f}: {n} 行（非 96，跳过）")
            continue

        df["task"] = task
        df["model_name"] = model
        df["forecast_date"] = date_str
        df["target_day"] = date_str

        result = append_predictions_to_ledger(
            df=df,
            ledger_root=ledger_root,
            task=task,
            source_file=str(f),
        )
        total_appended += result.get("new_rows", 0)
        if n != 96:
            print(f"  OK {date_str}/{task}/{model}: +{result.get('new_rows', 0)}")

    print(f"\n重建完成，共写入 {total_appended} 行。")

    # 验证
    for task in ["dayahead", "realtime"]:
        p = ledger_root / task / "prediction" / "prediction_ledger.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            expected = 288 if task == "dayahead" else 384
            bad = df.groupby("target_day").size()
            bad = bad[bad != expected]
            print(f"{task}: 总行数={len(df)}, 天数={df['target_day'].nunique()}, "
                  f"行数≠{expected} 的天数={len(bad)}")
            if len(bad):
                for d, c in sorted(bad.items())[:5]:
                    print(f"  {d}: {c} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
