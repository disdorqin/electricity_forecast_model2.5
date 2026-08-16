#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极端价分类器 — 日滚动入口（run_daily.py）

被 fusion/classifier_bridge.run_extreme_price_classifier 调用：
  python run_daily.py <start> <end> --output <dir> --data <data_path>

读取市场数据（含「实时电价」列），按日滚动运行两阶段级联分类，
输出 <start>_<end>_clf.xlsx（含 时刻/p1_prob/p2_prob/final_pred/真实极值标签 等）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd  # noqa: E402

# 把 merge_model 加入路径（run_daily.py 位于 ExtremPriceClf/merge_model_scripts/）
_PKG_ROOT = Path(__file__).resolve().parent.parent  # ExtremPriceClf/
_CORE = _PKG_ROOT / "merge_model"
for _p in (str(_PKG_ROOT), str(_CORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from merge_model.core.cascade_daily import (  # noqa: E402
    Stage2Config,
    run_rolling_daily_cascade,
    prepare_dataset,
)

DEFAULT_TARGET = "实时电价"
DEFAULT_THRESHOLD = -50.0


def main() -> int:
    parser = argparse.ArgumentParser(description="极端价分类器日滚动入口")
    parser.add_argument("start", help="开始日期 YYYY-MM-DD")
    parser.add_argument("end", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--data", required=True, help="市场数据文件路径")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--resolution", default="hourly", choices=["hourly", "15min"],
                        help="输入数据分辨率：hourly=24点（默认）、15min=96点（自动按小时聚合后分类）")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(str(data_path))
    if df.empty:
        print("❌ 数据为空")
        return 1

    # 剔除 LightGBM 不支持的列（datetime/object 非数值列），
    # 只保留数值特征 + 时刻 + 目标价。96 点宽表含 market_date/开机状态 等非数值列。
    if "时刻" not in df.columns:
        print("❌ 数据缺 '时刻' 列")
        return 1
    keep = ["时刻"]
    if args.target in df.columns:
        keep.append(args.target)
    for c in df.columns:
        if c in keep:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            keep.append(c)
    df = df[keep].copy()

    # 96 点（15min）输入：按小时聚合为小时级（数值列取该小时 4 刻度的均值）。
    # 分类器 cascade 是小时级模型（tail(24*9)/iloc[-24:] 行数语义 = 24 点/天），
    # 直接喂 96 点会使推理窗只覆盖当天最后 6 小时。聚合后输出 24 行小时 final_pred，
    # 由 classifier_bridge.merge_clf_results 广播回 4 个 15min 刻度。
    if args.resolution == "15min":
        _num_cols = [c for c in df.columns if c != "时刻" and pd.api.types.is_numeric_dtype(df[c])]
        df["时刻"] = pd.to_datetime(df["时刻"])
        _agg = {c: "mean" for c in _num_cols}
        df = df.groupby(df["时刻"].dt.floor("h"), as_index=False).agg({"时刻": "first", **_agg})
        df["时刻"] = df["时刻"].dt.floor("h")
        df = df.sort_values("时刻").reset_index(drop=True)
        print(f"ℹ️ 15min 数据已按小时聚合: {len(df)} 小时")

    # 训练/推理时间窗
    train_start = "2022-01-01"
    stage2_train_start = "2024-01-01"
    oof_cutoff = "2024-12-31"
    test_time_range = [args.start + " 00:00:00", args.end + " 23:00:00"]

    cfg = Stage2Config()
    # 动态灰度阈值在 cascade 内部按历史窗自动寻优（dynamic_gray_enabled）
    cfg.dynamic_gray_enabled = True

    p1_cache_path = str(output_dir / ".p1_cache.xlsx")

    results = run_rolling_daily_cascade(
        df=df,
        target_name=args.target,
        price_threshold=args.threshold,
        test_time_range=test_time_range,
        train_start=train_start,
        stage2_train_start=stage2_train_start,
        stage2_config=cfg,
        p1_cache_path=p1_cache_path,
        oof_cutoff=oof_cutoff,
    )

    if results is None or len(results) == 0:
        print("❌ 分类器未生成结果")
        return 1

    out_xlsx = output_dir / f"{args.start}_{args.end}_clf.xlsx"
    results.to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"✅ 分类器结果已保存: {out_xlsx} ({len(results)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
