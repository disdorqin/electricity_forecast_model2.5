#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冠军锚定 + 预测形态门控的历史代理实验。

目标是验证一个极轻量的“冠军学习”器：只使用已生成预测账本中的模型输出，
通过过去 23 天的模型损失训练强正则 Ridge 门控器，再用最近 7 天做门控验证。
RT 的 65_96 段另行保留经验证的冠军锚定有符号校正；所有结果仍是历史账本代理，
不代表真实 96 点特征上的生产精度。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from fusion.learners.daily_ledger_gef import compute_daily_loss  # noqa: E402
from scripts.experiments.nnls_ab.run_champion_anchor_ab import (  # noqa: E402
    GATE_TOL,
    HALF_LIFE_DAYS,
    TRAIN_DAYS,
    WINDOW_DAYS,
    champion_index,
    day_weights,
    fit_signed_anchor,
    fit_weighted_nnls,
    load_task,
    one_hot,
)

META_ALPHA = 1000.0
META_GATE_TOL = 0.0
META_BLEND_RHO = 0.30


def feature_vector(mat: Any) -> np.ndarray:
    """将一个 period 的预测矩阵压成不依赖实际值的形态特征。"""
    x = mat.X
    values: list[float] = []
    for j in range(x.shape[1]):
        z = x[:, j]
        values.extend([float(z.mean()), float(z.std()), float(np.median(z)), float(z.min()), float(z.max())])
    for j in range(x.shape[1]):
        for k in range(j + 1, x.shape[1]):
            values.extend([float(np.mean(np.abs(x[:, j] - x[:, k]))), float(np.mean(x[:, j] - x[:, k]))])
    return np.asarray(values, dtype=float)


def fit_meta(mats: list[Any], q_days: np.ndarray, alpha: float = META_ALPHA) -> tuple[np.ndarray, ...]:
    features = np.vstack([feature_vector(mat) for mat in mats])
    losses = np.vstack([
        [compute_daily_loss(mat.y, mat.X[:, j], "composite") for j in range(mat.X.shape[1])]
        for mat in mats
    ])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (features - mean) / scale
    root_q = np.sqrt(q_days)
    design = z * root_q[:, None]
    response = losses * root_q[:, None]
    coef = np.linalg.solve(
        design.T @ design + alpha * np.eye(design.shape[1]),
        design.T @ response,
    )
    intercept = losses.mean(axis=0) - (mean / scale) @ coef
    return mean, scale, coef, intercept


def predict_meta(mats: list[Any], fitted: tuple[np.ndarray, ...]) -> np.ndarray:
    mean, scale, coef, intercept = fitted
    z = (np.vstack([feature_vector(mat) for mat in mats]) - mean) / scale
    return z @ coef + intercept


def dynamic_loss(mats: list[Any], predicted_losses: np.ndarray) -> float:
    values = []
    for mat, estimates in zip(mats, predicted_losses):
        model = int(np.argmin(estimates))
        values.append(compute_daily_loss(mat.y, mat.X[:, model], "composite"))
    return float(np.mean(values))


def dynamic_blend_loss(
    mats: list[Any], predicted_losses: np.ndarray, champion: int, rho: float
) -> float:
    values = []
    for mat, estimates in zip(mats, predicted_losses):
        model = int(np.argmin(estimates))
        fused = (1.0 - rho) * mat.X[:, champion] + rho * mat.X[:, model]
        values.append(compute_daily_loss(mat.y, fused, "composite"))
    return float(np.mean(values))


def champion_loss(mats: list[Any], champion: int) -> float:
    return float(np.mean([
        compute_daily_loss(mat.y, mat.X[:, champion], "composite")
        for mat in mats
    ]))


def weighted_loss(mats: list[Any], weights: np.ndarray) -> float:
    return float(np.mean([
        compute_daily_loss(mat.y, mat.X @ weights, "composite")
        for mat in mats
    ]))


def target_dynamic_loss(mat: Any, predicted_losses: np.ndarray) -> float:
    model = int(np.argmin(predicted_losses[0]))
    return float(compute_daily_loss(mat.y, mat.X[:, model], "composite"))


def target_dynamic_blend_loss(
    mat: Any, predicted_losses: np.ndarray, champion: int, rho: float
) -> float:
    model = int(np.argmin(predicted_losses[0]))
    fused = (1.0 - rho) * mat.X[:, champion] + rho * mat.X[:, model]
    return float(compute_daily_loss(mat.y, fused, "composite"))


def run_task(task: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    day_data, models, periods = load_task(task)
    days = sorted(day_data)
    rows: list[dict[str, Any]] = []
    for target_day in days[WINDOW_DAYS:]:
        idx = days.index(target_day)
        window_days = days[idx - WINDOW_DAYS:idx]
        train_days = window_days[:TRAIN_DAYS]
        validation_days = window_days[TRAIN_DAYS:]
        q_train = day_weights(len(train_days), HALF_LIFE_DAYS)
        q_all = day_weights(len(window_days), HALF_LIFE_DAYS)
        for period in periods:
            train = [day_data[d][period] for d in train_days]
            validation = [day_data[d][period] for d in validation_days]
            window = [day_data[d][period] for d in window_days]
            target = day_data[target_day][period]
            champion = champion_index(train, models, q_train)
            champion_val = champion_loss(validation, champion)
            champion_target = compute_daily_loss(target.y, target.X[:, champion], "composite")

            # DA 沿用上一轮已验证的冠军锚定候选协议，作为 DA 基线策略；
            # 它只在最近 7 日验证不劣于冠军时才可用于目标日。
            w_nnls_train = fit_weighted_nnls(train, q_train)
            w_signed_train = fit_signed_anchor(train, q_train, champion)
            anchor_validation = {
                "champion": champion_val,
                "weighted_nnls": weighted_loss(validation, w_nnls_train),
                "signed_anchor": weighted_loss(validation, w_signed_train),
            }
            eligible_anchor = {
                name: value
                for name, value in anchor_validation.items()
                if value <= champion_val * (1.0 + GATE_TOL)
            }
            anchor_strategy = min(eligible_anchor, key=eligible_anchor.get)
            w_nnls_all = fit_weighted_nnls(window, q_all)
            w_signed_all = fit_signed_anchor(window, q_all, champion)
            if anchor_strategy == "weighted_nnls":
                anchor_target = weighted_loss([target], w_nnls_all)
            elif anchor_strategy == "signed_anchor":
                anchor_target = weighted_loss([target], w_signed_all)
            else:
                anchor_target = champion_target

            meta_train = fit_meta(train, q_train)
            meta_val_pred = predict_meta(validation, meta_train)
            meta_val = dynamic_loss(validation, meta_val_pred)
            meta_allowed = meta_val <= champion_val * (1.0 + META_GATE_TOL)
            meta_blend_val = dynamic_blend_loss(
                validation, meta_val_pred, champion, META_BLEND_RHO
            )
            meta_blend_allowed = meta_blend_val <= champion_val * (1.0 + META_GATE_TOL)

            meta_all = fit_meta(window, q_all)
            meta_target = target_dynamic_loss(target, predict_meta([target], meta_all))
            meta_blend_target = target_dynamic_blend_loss(
                target, predict_meta([target], meta_all), champion, META_BLEND_RHO
            )

            signed_all = fit_signed_anchor(window, q_all, champion)
            signed_val = float(np.mean([
                # 验证集只能评估训练集拟合出的权重，不能用全30日权重回看验证集。
                compute_daily_loss(mat.y, mat.X @ w_signed_train, "composite")
                for mat in validation
            ]))
            signed_target = compute_daily_loss(target.y, target.X @ signed_all, "composite")
            signed_allowed = signed_val <= champion_val * (1.0 + GATE_TOL)

            # 研究策略：DA 保持前一轮冠军锚定候选；RT 前两段用门控 meta，
            # 65_96 使用验证通过的有符号冠军锚定，避免把弱门控器强行用于 RT。
            if task == "realtime" and period in {"1_32", "33_64"}:
                policy = "meta_blend_gate"
                selected_target = meta_blend_target if meta_blend_allowed else champion_target
            elif task == "realtime" and period == "65_96":
                policy = "signed_gate"
                selected_target = signed_target if signed_allowed else champion_target
            else:
                policy = "anchor_gate"
                selected_target = anchor_target

            rows.append({
                "task": task,
                "target_day": target_day,
                "period": period,
                "policy": policy,
                "champion_model": models[champion],
                "meta_allowed": bool(meta_allowed),
                "meta_blend_allowed": bool(meta_blend_allowed),
                "signed_allowed": bool(signed_allowed),
                "val_champion": champion_val,
                "val_meta": meta_val,
                "val_meta_blend": meta_blend_val,
                "val_signed": signed_val,
                "anchor_strategy": anchor_strategy,
                "val_anchor": anchor_validation[anchor_strategy],
                "target_anchor": anchor_target,
                "target_champion": champion_target,
                "target_meta": meta_target,
                "target_meta_blend": meta_blend_target,
                "target_signed": signed_target,
                "target_policy": selected_target,
                "meta_model_target": models[int(np.argmin(predict_meta([target], meta_all)[0]))],
            })
    return pd.DataFrame(rows), {
        "task": task,
        "models": models,
        "periods": periods,
        "complete_days": len(days),
        "target_days": len(days) - WINDOW_DAYS,
        "elapsed_seconds": time.perf_counter() - started,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output = PROJECT / "outputs/experiments/champion_weight_ab/meta_hybrid"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [
        PROJECT / "outputs/ledger_96/dayahead/prediction/prediction_ledger.parquet",
        PROJECT / "outputs/ledger_96/dayahead/actual/actual_ledger.parquet",
        PROJECT / "outputs/ledger_96/realtime/prediction/prediction_ledger.parquet",
        PROJECT / "outputs/ledger_96/realtime/actual/actual_ledger.parquet",
    ]
    manifest = {
        "experiment": "champion_meta_hybrid_ab",
        "status": "historical-invalid-features-proxy-only",
        "resolution": "15min",
        "window_days": WINDOW_DAYS,
        "train_days": TRAIN_DAYS,
        "validation_days": WINDOW_DAYS - TRAIN_DAYS,
        "half_life_days": HALF_LIFE_DAYS,
        "meta_alpha": META_ALPHA,
        "meta_gate_tolerance": META_GATE_TOL,
        "meta_blend_rho": META_BLEND_RHO,
        "inputs": {
            str(path.relative_to(PROJECT)): {
                "sha256": sha256_file(path) if path.exists() else None,
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for path in inputs
        },
        "forbidden_input": "data/96/model_input/shandong_pmos_96_model_input.xlsx",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results: list[pd.DataFrame] = []
    metadata = {}
    for task in ("dayahead", "realtime"):
        result, meta = run_task(task)
        result.to_csv(output / f"rolling_results_{task}.csv", index=False)
        results.append(result)
        metadata[task] = meta
        print(json.dumps(meta, ensure_ascii=False))
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output / "rolling_results_all.csv", index=False)

    summary_rows = []
    for (task, period), group in combined.groupby(["task", "period"]):
        base = group["target_champion"]
        for method in ["target_champion", "target_meta", "target_signed", "target_policy"]:
            summary_rows.append({
                "task": task,
                "period": period,
                "method": method,
                "n": len(group),
                "mean_composite": group[method].mean(),
                "median_composite": group[method].median(),
                "mean_delta_vs_champion": (group[method] - base).mean(),
                "win_rate_vs_champion": (group[method] < base).mean(),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary_all.csv", index=False)

    block_rows = []
    for (task, period), group in combined.groupby(["task", "period"]):
        group = group.sort_values("target_day").copy()
        group["block"] = np.where(np.arange(len(group)) < len(group) / 2, "first_half", "second_half")
        for block, part in group.groupby("block"):
            block_rows.append({
                "task": task,
                "period": period,
                "block": block,
                "n": len(part),
                "champion_mean": part.target_champion.mean(),
                "policy_mean": part.target_policy.mean(),
                "policy_delta_vs_champion": (part.target_policy - part.target_champion).mean(),
                "policy_win_rate": (part.target_policy < part.target_champion).mean(),
            })
    pd.DataFrame(block_rows).to_csv(output / "summary_halves.csv", index=False)
    (output / "runtime.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nHALVES")
    print(pd.DataFrame(block_rows).to_string(index=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
