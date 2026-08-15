#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM 超参对比实验 — 实验专区（本机 CPU 可跑，不碰生产代码）

在 outputs/experiments/lightgbm_ab/ 下跑。对同一历史窗，用多组超参
分别训练实时价 LightGBM 模型，对比：
  - MAE / MSE / SMAPE（验证集）
  - 训练时间
  - loss 收敛（early stopping 时的 best iteration 与 val loss 轨迹）

用法：
  python scripts/experiments/lightgbm_ab/run_ab.py --data data/shandong_pmos_96_full_v2.xlsx \
    --start 2026-04-01 --end 2026-07-31 --resolution 15min
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
import lightgbm as lgb  # noqa: E402

from lightGBM.train_fix import LGBMPowerPredictor  # noqa: E402
from lightGBM.main_fix import _segment_masks, _slot_col, _split_history_train_val  # noqa: E402

EXP_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "lightgbm_ab"
RESULTS = EXP_ROOT / "results"


def capped_smape(y_true, y_pred, floor=50.0):
    """与生产口径一致的改良 SMAPE（train_fix.calculate_smape）。

    生产公式：值<50 先用 50 替代（clip 值），再算 |p-t| / ((|p|+|t|)/2)。
    注意：不是"分母 floor50"——是先裁剪值再算，两者对负价+尖峰双峰数据差异巨大。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_f = np.where(y_true < floor, floor, y_true)
    y_pred_f = np.where(y_pred < floor, floor, y_pred)
    denom = (np.abs(y_pred_f) + np.abs(y_true_f)) / 2.0
    num = np.abs(y_pred_f - y_true_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = num / denom
        terms[denom == 0] = 0.0
    return float(np.mean(terms))


def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "MAE": round(mae, 3),
        "MSE": round(mse, 3),
        "RMSE": round(np.sqrt(mse), 3),
        "SMAPE": round(capped_smape(y_true, y_pred), 5),
    }


def run_one_config(predictor, df, seg_name, seg_slots, slot_col, params, label):
    """对某时段段训练一个 LightGBM，返回指标+耗时。"""
    train_df = df[df[slot_col].isin(seg_slots)].copy()
    train_df, val_df = _split_history_train_val(train_df, val_ratio=0.2)
    # 目标裁剪（与生产一致）
    upper = train_df["y"].quantile(0.995)
    train_df["y_clipped"] = train_df["y"].clip(lower=-100, upper=upper)
    if len(val_df) == 0 or train_df.empty:
        return None

    t0 = time.perf_counter()
    model = lgb.LGBMRegressor(
        objective="regression",
        n_jobs=int(__import__("os").getenv("LGBM_N_JOBS", "4")),
        device_type="cpu",
        verbose=-1,
        random_state=42,
        **params,
    )
    model.fit(
        train_df[predictor.features_list],
        train_df["y_clipped"],
        eval_set=[(val_df[predictor.features_list], val_df["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    elapsed = time.perf_counter() - t0

    pred = model.predict(val_df[predictor.features_list])
    metrics = eval_metrics(val_df["y"].values, pred)
    metrics["best_iteration"] = getattr(model, "best_iteration_", None)
    metrics["n_estimators_final"] = len(model.booster_.dump_model()["tree_info"])
    metrics["train_seconds"] = round(elapsed, 1)
    metrics["segment"] = seg_name
    metrics["label"] = label
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="LightGBM 超参对比实验（CPU）")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "shandong_pmos_96_full_v2.xlsx"))
    parser.add_argument("--start", default="2026-04-01", help="历史窗起点")
    parser.add_argument("--end", default="2026-07-31", help="历史窗终点")
    parser.add_argument("--resolution", default="15min", choices=["15min", "hourly"])
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    from utils.resolution import resolve_resolution
    res = resolve_resolution(args.resolution)
    predictor = LGBMPowerPredictor(resolution=res)
    raw = predictor.load_and_process_data(args.data, target="实时电价", resolution=res)
    full = predictor.feature_engineering(raw, resolution=res)
    slot_col = _slot_col(res)
    valley_h, solar_h, peak_h = _segment_masks(res)
    segments = {"谷段": valley_h, "峰段": peak_h, "平段": solar_h}

    # 截取历史窗
    start_dt, end_dt = pd.to_datetime(args.start), pd.to_datetime(args.end)
    df = full[(full["ds"] >= start_dt) & (full["ds"] <= end_dt)].copy()
    print(f"历史窗: {args.start} ~ {args.end} | {len(df)} 行 | 特征 {len(predictor.features_list)} 个")

    # 超参网格（对照生产默认 vs 候选）
    configs = [
        {"label": "A_基线(lr05_leaf31)", "params": {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 31}},
        {"label": "B_小学习率(lr02_leaf63)", "params": {"n_estimators": 3000, "learning_rate": 0.02, "num_leaves": 63}},
        {"label": "C_浅树防过拟合(lr05_leaf15)", "params": {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 15}},
        {"label": "D_深树多迭代(lr03_leaf127)", "params": {"n_estimators": 4000, "learning_rate": 0.03, "num_leaves": 127}},
        {"label": "E_大学习率快收敛(lr10_leaf31)", "params": {"n_estimators": 1000, "learning_rate": 0.10, "num_leaves": 31}},
    ]

    all_results = []
    for cfg in configs:
        label = cfg["label"]
        print(f"\n{'='*60}\n[{label}] 训练中...")
        for seg_name, seg_slots in segments.items():
            m = run_one_config(predictor, df, seg_name, seg_slots, slot_col, cfg["params"], label)
            if m:
                all_results.append(m)
                print(f"  {seg_name}: MAE={m['MAE']} MSE={m['MSE']} SMAPE={m['SMAPE']} "
                      f"iter={m['best_iteration']} time={m['train_seconds']}s")

    # 汇总表
    summary = pd.DataFrame(all_results)
    summary.to_csv(RESULTS / "lightgbm_ab_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\n{'='*60}\n结果已保存: {RESULTS / 'lightgbm_ab_metrics.csv'}")

    # 按段打印对比
    for seg in summary["segment"].unique():
        sub = summary[summary["segment"] == seg]
        print(f"\n── {seg} 段对比 ──")
        best = sub.loc[sub["SMAPE"].idxmin()]
        print(f"  SMAPE 最优: {best['label']} → {best['SMAPE']:.5f}")
        for _, r in sub.iterrows():
            print(f"  {r['label']:28s} SMAPE={r['SMAPE']:.5f} MAE={r['MAE']:7.2f} "
                  f"iter={int(r['best_iteration'])} time={r['train_seconds']}s")

    with open(RESULTS / "lightgbm_ab_summary.json", "w", encoding="utf-8") as f:
        json.dump({"window": {"start": args.start, "end": args.end}, "results": all_results},
                  f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
