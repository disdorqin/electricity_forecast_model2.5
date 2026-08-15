#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TimesFM 调参实验 — 实验专区（本机 CPU，零样本）

对比 segment_count（分段数）对预测精度的影响。
TimesFM 是零样本模型，可调参数主要是分段方式（segment_count/skip_style）。

注意：TimesFM 单日 CPU 约 40s，实验窗用小样本（默认 5 天）控制耗时。

用法：
  python scripts/experiments/timesfm_ab/run_ab.py --start 2026-05-25 --end 2026-05-31
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "timesfm_ab"
RESULTS = EXP_ROOT / "results"
DATA = PROJECT_ROOT / "data" / "shandong_pmos_96_full_v2.xlsx"


def capped_smape(y_true, y_pred, floor=50.0):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    yt = np.where(y_true < floor, floor, y_true)
    yp = np.where(y_pred < floor, floor, y_pred)
    denom = (np.abs(yp) + np.abs(yt)) / 2.0
    num = np.abs(yp - yt)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = num / denom
        terms[denom == 0] = 0.0
    return float(np.mean(terms))


def load_truth(start: str, end: str) -> pd.DataFrame:
    """加载真实电价（用于评估）。"""
    df = pd.read_excel(DATA)
    df["时刻"] = pd.to_datetime(df["时刻"], errors="coerce")
    df = df[(df["时刻"] >= start) & (df["时刻"] <= pd.to_datetime(end) + pd.Timedelta(days=1))]
    return df[["时刻", "实时电价"]].rename(columns={"实时电价": "y_true"})


def main() -> int:
    parser = argparse.ArgumentParser(description="TimesFM 分段数对比实验（CPU 零样本）")
    parser.add_argument("--start", default="2026-05-25")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--segments", default="1,2,3,4", help="逗号分隔的 segment_count 列表")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    os_env = __import__("os")
    os_env.environ["TIMESFM_DEVICE"] = "cpu"

    from TimesFMBackend.infer import predict_price_for_date

    segments = [int(x) for x in args.segments.split(",") if x.strip()]
    dates = pd.date_range(args.start, args.end).strftime("%Y-%m-%d").tolist()
    truth = load_truth(args.start, args.end)
    truth["时刻"] = pd.to_datetime(truth["时刻"])

    print(f"数据: {DATA.name}, 日期 {args.start}~{args.end}, {len(dates)}天")
    print(f"测试分段: {segments}\n")

    all_results = []
    for sc in segments:
        preds = []
        for d in dates:
            t0 = time.perf_counter()
            df = predict_price_for_date(str(DATA), d, target="realtime",
                                        resolution="15min", segment_count=sc)
            el = time.perf_counter() - t0
            preds.append((d, df, el))
            print(f"  seg={sc} {d}: {len(df)}行 {el:.0f}s", flush=True)

        # 合并预测
        pdf = pd.concat([df.assign(date=d) for d, df, _ in preds], ignore_index=True)
        pdf["时刻"] = pd.to_datetime(pdf["时刻"], errors="coerce")
        m = pdf.merge(truth, on="时刻", how="inner").dropna(subset=["预测值", "y_true"])
        mae = float(np.mean(np.abs(m["预测值"] - m["y_true"])))
        mse = float(np.mean((m["预测值"] - m["y_true"]) ** 2))
        smape = capped_smape(m["y_true"].values, m["预测值"].values)
        total_s = sum(el for _, _, el in preds)
        all_results.append({"segment_count": sc, "MAE": round(mae, 3), "MSE": round(mse, 3),
                            "SMAPE": round(smape, 5), "total_seconds": round(total_s, 1),
                            "rows": len(m)})
        print(f"  → seg={sc}: MAE={mae:.2f} SMAPE={smape:.4f} 总耗时={total_s:.0f}s")

    summary = pd.DataFrame(all_results)
    summary.to_csv(RESULTS / "timesfm_ab_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {RESULTS / 'timesfm_ab_metrics.csv'}")
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
