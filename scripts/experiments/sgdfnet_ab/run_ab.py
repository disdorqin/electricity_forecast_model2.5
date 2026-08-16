#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SGDFNet 调参实验 — 实验专区（本机 CPU 可跑）

先测训练耗时（>3min 则停调），再对比多组 HGBModelConfig 超参。

SGDFNet 用 HistGradientBoosting（sklearn），可调：
  loss / learning_rate / max_depth / max_iter / min_samples_leaf / l2_regularization

用法：
  python scripts/experiments/sgdfnet_ab/run_ab.py --data data/shandong_pmos_96_full_v2.xlsx \
    --start 2026-04-01 --end 2026-05-31 --resolution 15min
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

# SGDFNet 是 src 布局：包在 SGDFNet/src/sgdfnet
_SGDF_SRC = PROJECT_ROOT / "SGDFNet" / "src"
if str(_SGDF_SRC) not in sys.path:
    sys.path.insert(0, str(_SGDF_SRC))
from sgdfnet.models import HGBModelConfig, DeltaRegressor  # noqa: E402
from utils.resolution import resolve_resolution  # noqa: E402

EXP_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "sgdfnet_ab"
RESULTS = EXP_ROOT / "results"


def capped_smape(y_true, y_pred, floor=50.0):
    """生产口径 SMAPE：值<50 先 clip 到 50 再算（train_fix.calculate_smape）。"""
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


def load_frame(data_path: Path, start: str, end: str, resolution) -> pd.DataFrame:
    """加载宽表并构造 delta 训练帧（简化：直接用实际价序列做回归）。"""
    df = pd.read_excel(data_path)
    slot_col = "business_period" if resolution.slots_per_day > 24 else "hour_business"
    if slot_col not in df.columns:
        # 96点宽表用 period_no
        if "period_no" in df.columns:
            slot_col = "period_no"
        else:
            # 构造 business_period 从 时刻
            df["时刻"] = pd.to_datetime(df["时刻"])
            df["period_no"] = ((df["时刻"].dt.hour * 60 + df["时刻"].dt.minute) // 15).replace({0: 96})
            slot_col = "period_no"
    df["_ts"] = pd.to_datetime(df["时刻"] if "时刻" in df.columns else df["ds"], errors="coerce")
    df = df.dropna(subset=["_ts"])
    df = df[(df["_ts"] >= start) & (df["_ts"] <= pd.to_datetime(end) + pd.Timedelta(days=1))]
    # delta 目标
    y = pd.to_numeric(df.get("实时电价", pd.Series(dtype=float)), errors="coerce")
    df["delta_target"] = y - y.shift(resolution.slots_per_day)
    df["lag_da"] = pd.to_numeric(df.get("日前电价", pd.Series(dtype=float)), errors="coerce").shift(resolution.slots_per_day)
    # 特征：时段/滞后/负荷
    for c in ["直调负荷预测值", "风电总加预测值", "光伏总加预测值", "竞价空间预测值"]:
        if c not in df.columns:
            for alt in [c.replace("值", ""), c.replace("预测值", "预测")]:
                if alt in df.columns:
                    df[c] = df[alt]
                    break
    return df


def run_one(cfg_label: str, hgb_cfg: HGBModelConfig, frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    """训练一个 DeltaRegressor，返回指标+耗时。"""
    train = frame.dropna(subset=["delta_target"]).copy()
    if train.empty or len(train) < 100:
        return {"label": cfg_label, "error": "训练样本不足"}

    # 简单时间切分：后20%为验证
    n = len(train)
    split = int(n * 0.8)
    tr, va = train.iloc[:split], train.iloc[split:]

    t0 = time.perf_counter()
    model = DeltaRegressor(hgb_cfg)
    model.fit(tr, feature_cols, target_col="delta_target")
    elapsed = time.perf_counter() - t0

    # 预测（delta + 前一日常量 ≈ 绝对价）
    pred_delta = model.predict(va, feature_cols)
    pred_abs = pred_delta  # delta 近似（简化：不还原绝对价）

    y_true = va["delta_target"].values
    mae = float(np.mean(np.abs(y_true - pred_delta)))
    mse = float(np.mean((y_true - pred_delta) ** 2))
    smape = capped_smape(y_true, pred_delta)

    return {
        "label": cfg_label,
        "MAE": round(mae, 3), "MSE": round(mse, 3),
        "SMAPE": round(smape, 5),
        "train_seconds": round(elapsed, 1),
        "train_rows": len(tr),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SGDFNet 超参对比实验（CPU）")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "shandong_pmos_96_full_v2.xlsx"))
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--resolution", default="15min", choices=["15min", "hourly"])
    parser.add_argument("--smoke-only", action="store_true", help="只测1次耗时（>3min则停）")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    res = resolve_resolution(args.resolution)

    print(f"加载数据 {args.data} ...")
    frame = load_frame(Path(args.data), args.start, args.end, res)
    print(f"数据行数: {len(frame)}")

    # 特征列（从宽表可用列）
    feature_cols = [c for c in ["period_no", "lag_da", "直调负荷预测值", "风电总加预测值", "光伏总加预测值"]
                    if c in frame.columns]
    print(f"特征: {feature_cols}")

    # 超参网格
    configs = [
        {"label": "A_基线(lr05_iter300)", "cfg": HGBModelConfig(learning_rate=0.05, max_iter=300)},
        {"label": "B_小学习率(lr02_iter500)", "cfg": HGBModelConfig(learning_rate=0.02, max_iter=500)},
        {"label": "C_浅树(lr05_d3)", "cfg": HGBModelConfig(learning_rate=0.05, max_depth=3)},
        {"label": "D_深树(lr05_d10)", "cfg": HGBModelConfig(learning_rate=0.05, max_depth=10)},
        {"label": "E_大迭代(lr03_iter800)", "cfg": HGBModelConfig(learning_rate=0.03, max_iter=800)},
        # 深调（用户：时间无所谓，可给多时间）
        {"label": "F_深收敛(lr015_iter1200)", "cfg": HGBModelConfig(learning_rate=0.015, max_iter=1200)},
        {"label": "G_深收敛(lr01_iter2000)", "cfg": HGBModelConfig(learning_rate=0.01, max_iter=2000)},
        {"label": "H_深收敛+深树(lr02_iter800_d10)", "cfg": HGBModelConfig(learning_rate=0.02, max_iter=800, max_depth=10)},
        {"label": "I_小学习率+小叶(lr02_iter800_ms20)", "cfg": HGBModelConfig(learning_rate=0.02, max_iter=800, min_samples_leaf=20)},
    ]

    all_results = []
    for cfg in configs:
        print(f"\n[{cfg['label']}] 训练中...", flush=True)
        m = run_one(cfg["label"], cfg["cfg"], frame, feature_cols)
        if "error" in m:
            print(f"  {m['error']}")
            all_results.append(m)
            continue
        all_results.append(m)
        print(f"  MAE={m['MAE']} MSE={m['MSE']} SMAPE={m['SMAPE']} time={m['train_seconds']}s", flush=True)
        # 若首次训练>180s 且是 smoke-only，提前停
        if args.smoke_only and m["train_seconds"] > 180:
            print(f"⚠ 训练耗时 {m['train_seconds']}s > 3min，按规则停止调参")
            break

    summary = pd.DataFrame(all_results)
    summary.to_csv(RESULTS / "sgdfnet_ab_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {RESULTS / 'sgdfnet_ab_metrics.csv'}")
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"window": {"start": args.start, "end": args.end}, "results": all_results},
                  f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
