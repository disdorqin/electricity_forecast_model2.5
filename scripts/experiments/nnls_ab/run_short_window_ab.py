#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""短窗口冠军学习器实验：14 日窗、7 日训练、7 日验证。

这是针对“减少学习耗时、近期样本权重大”的正式实验版本。输入仍严格限定为
历史 prediction/actual ledger，结果只写入实验目录，不替换生产学习器。
"""
from __future__ import annotations

import hashlib
import json
import argparse
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
    champion_index,
    day_weights,
    fit_signed_anchor,
    fit_weighted_nnls,
    load_task,
    one_hot,
)
from scripts.experiments.nnls_ab.run_meta_hybrid_ab import (  # noqa: E402
    META_ALPHA,
    META_BLEND_RHO,
    dynamic_blend_loss,
    fit_meta,
    predict_meta,
    target_dynamic_blend_loss,
)

WINDOW_DAYS = 14
TRAIN_DAYS = 7
VALIDATION_DAYS = 7
COMMON_START = 30
HALF_LIFE_DAYS = {"dayahead": 7.0, "realtime": 7.0}
META_GATE_TOL = 0.0


def weighted_loss(mats: list[Any], weights: np.ndarray) -> float:
    return float(np.mean([
        compute_daily_loss(mat.y, mat.X @ weights, "composite")
        for mat in mats
    ]))


def champion_loss(mats: list[Any], champion: int) -> float:
    return float(np.mean([
        compute_daily_loss(mat.y, mat.X[:, champion], "composite")
        for mat in mats
    ]))


def run_task(task: str, half_life: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    day_data, models, periods = load_task(task)
    days = sorted(day_data)
    target_days = days[COMMON_START:]
    rows: list[dict[str, Any]] = []

    for target_day in target_days:
        index = days.index(target_day)
        window_days = days[index - WINDOW_DAYS:index]
        train_days = window_days[:TRAIN_DAYS]
        validation_days = window_days[TRAIN_DAYS:]
        q_train = day_weights(TRAIN_DAYS, half_life)
        q_all = day_weights(WINDOW_DAYS, half_life)

        for period in periods:
            train = [day_data[d][period] for d in train_days]
            validation = [day_data[d][period] for d in validation_days]
            window = [day_data[d][period] for d in window_days]
            target = day_data[target_day][period]
            champion = champion_index(train, models, q_train)
            champion_val = champion_loss(validation, champion)
            champion_target = compute_daily_loss(target.y, target.X[:, champion], "composite")

            w_nnls_train = fit_weighted_nnls(train, q_train)
            w_signed_train = fit_signed_anchor(train, q_train, champion)
            w_nnls_all = fit_weighted_nnls(window, q_all)
            w_signed_all = fit_signed_anchor(window, q_all, champion)
            anchor_validation = {
                "champion": champion_val,
                "weighted_nnls": weighted_loss(validation, w_nnls_train),
                "signed_anchor": weighted_loss(validation, w_signed_train),
            }
            eligible = {
                name: value
                for name, value in anchor_validation.items()
                if value <= champion_val * (1.0 + GATE_TOL)
            }
            anchor_strategy = min(eligible, key=eligible.get)
            if anchor_strategy == "weighted_nnls":
                anchor_target = weighted_loss([target], w_nnls_all)
            elif anchor_strategy == "signed_anchor":
                anchor_target = weighted_loss([target], w_signed_all)
            else:
                anchor_target = champion_target

            meta_train = fit_meta(train, q_train, META_ALPHA)
            meta_val_pred = predict_meta(validation, meta_train)
            meta_val = dynamic_blend_loss(
                validation, meta_val_pred, champion, META_BLEND_RHO
            )
            meta_allowed = meta_val <= champion_val * (1.0 + META_GATE_TOL)
            meta_all = fit_meta(window, q_all, META_ALPHA)
            meta_target = target_dynamic_blend_loss(
                target,
                predict_meta([target], meta_all),
                champion,
                META_BLEND_RHO,
            )

            signed_val = weighted_loss(validation, w_signed_train)
            signed_allowed = signed_val <= champion_val * (1.0 + GATE_TOL)
            signed_target = weighted_loss([target], w_signed_all)

            if task == "realtime" and period in {"1_32", "33_64"}:
                policy = "meta_blend_gate"
                selected_target = meta_target if meta_allowed else champion_target
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
                "anchor_strategy": anchor_strategy,
                "meta_allowed": bool(meta_allowed),
                "signed_allowed": bool(signed_allowed),
                "val_champion": champion_val,
                "val_anchor": anchor_validation[anchor_strategy],
                "val_meta_blend": meta_val,
                "val_signed": signed_val,
                "target_champion": champion_target,
                "target_anchor": anchor_target,
                "target_meta_blend": meta_target,
                "target_signed": signed_target,
                "target_policy": selected_target,
            })

    return pd.DataFrame(rows), {
        "task": task,
        "models": models,
        "periods": periods,
        "complete_days": len(days),
        "target_days": len(target_days),
        "window_days": WINDOW_DAYS,
        "train_days": TRAIN_DAYS,
        "validation_days": VALIDATION_DAYS,
        "half_life_days": half_life,
        "elapsed_seconds": time.perf_counter() - started,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/experiments/champion_weight_ab/short14")
    parser.add_argument("--da-half-life", type=float, default=HALF_LIFE_DAYS["dayahead"])
    parser.add_argument("--rt-half-life", type=float, default=HALF_LIFE_DAYS["realtime"])
    args = parser.parse_args()
    half_life_days = {"dayahead": args.da_half_life, "realtime": args.rt_half_life}
    output = PROJECT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    input_paths = [
        PROJECT / "outputs/ledger_96/dayahead/prediction/prediction_ledger.parquet",
        PROJECT / "outputs/ledger_96/dayahead/actual/actual_ledger.parquet",
        PROJECT / "outputs/ledger_96/realtime/prediction/prediction_ledger.parquet",
        PROJECT / "outputs/ledger_96/realtime/actual/actual_ledger.parquet",
    ]
    manifest = {
        "experiment": "champion_short_window_ab",
        "status": "historical-invalid-features-proxy-only",
        "resolution": "15min",
        "window_days": WINDOW_DAYS,
        "train_days": TRAIN_DAYS,
        "validation_days": VALIDATION_DAYS,
        "common_target_start_index": COMMON_START,
        "half_life_days": half_life_days,
        "meta_alpha": META_ALPHA,
        "meta_blend_rho": META_BLEND_RHO,
        "meta_gate_tolerance": META_GATE_TOL,
        "inputs": {
            str(path.relative_to(PROJECT)): {
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for path in input_paths
        },
        "forbidden_input": "data/96/model_input/shandong_pmos_96_model_input.xlsx",
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for task in ("dayahead", "realtime"):
        result, meta = run_task(task, half_life_days[task])
        result.to_csv(output / f"rolling_results_{task}.csv", index=False)
        results.append(result)
        metadata[task] = meta
        print(json.dumps(meta, ensure_ascii=False))
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output / "rolling_results_all.csv", index=False)

    summary_rows = []
    for (task, period), group in combined.groupby(["task", "period"]):
        base = group["target_champion"]
        for method in ["target_champion", "target_policy", "target_anchor", "target_meta_blend", "target_signed"]:
            summary_rows.append({
                "task": task,
                "period": period,
                "method": method,
                "n": len(group),
                "mean_composite": group[method].mean(),
                "mean_delta_vs_champion": (group[method] - base).mean(),
                "win_rate_vs_champion": (group[method] < base).mean(),
            })
    pd.DataFrame(summary_rows).to_csv(output / "summary_all.csv", index=False)

    half_rows = []
    for (task, period), group in combined.groupby(["task", "period"]):
        ordered = group.sort_values("target_day").reset_index(drop=True)
        for label, part in [("first_half", ordered.iloc[:100]), ("second_half", ordered.iloc[100:])]:
            delta = part.target_policy - part.target_champion
            half_rows.append({
                "task": task,
                "period": period,
                "block": label,
                "n": len(part),
                "champion_mean": part.target_champion.mean(),
                "policy_mean": part.target_policy.mean(),
                "delta": delta.mean(),
                "win_rate": (delta < 0).mean(),
            })
    pd.DataFrame(half_rows).to_csv(output / "summary_halves.csv", index=False)
    (output / "runtime.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nHALVES")
    print(pd.DataFrame(half_rows).to_string(index=False))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
