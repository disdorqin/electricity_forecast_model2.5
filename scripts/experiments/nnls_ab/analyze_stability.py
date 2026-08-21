#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对历史代理实验做成对稳定性与不确定性审计。

只读取已生成的 rolling_results_all.csv，不参与权重拟合，也不修改生产链路。
delta = target_policy - target_champion，负值表示候选策略更好。
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

PROJECT = Path(__file__).resolve().parents[3]
INPUT = PROJECT / "outputs/experiments/champion_weight_ab/meta_hybrid/rolling_results_all.csv"
OUTPUT = INPUT.parent / "stability_audit.csv"
BOOTSTRAP_SEED = 20260817
BOOTSTRAP_REPS = 20_000


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_REPS, len(values)), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def audit_group(group: pd.DataFrame, label: str) -> dict[str, object]:
    delta = group["target_policy"].to_numpy(dtype=float) - group["target_champion"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(delta)
    try:
        t_p = float(ttest_rel(group["target_policy"], group["target_champion"]).pvalue)
    except Exception:
        t_p = float("nan")
    try:
        w_p = float(wilcoxon(delta, alternative="less", zero_method="wilcox").pvalue)
    except Exception:
        w_p = float("nan")
    return {
        "scope": label,
        "n": int(len(delta)),
        "champion_mean": float(group["target_champion"].mean()),
        "policy_mean": float(group["target_policy"].mean()),
        "mean_delta_policy_minus_champion": float(delta.mean()),
        "bootstrap95_low": float(ci_low),
        "bootstrap95_high": float(ci_high),
        "median_delta": float(np.median(delta)),
        "win_rate": float(np.mean(delta < 0)),
        "p95_delta": float(np.quantile(delta, 0.95)),
        "worst_delta": float(delta.max()),
        "paired_t_pvalue": t_p,
        "wilcoxon_less_pvalue": w_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else PROJECT / args.input
    output_path = args.output or input_path.with_name("stability_audit.csv")
    if not output_path.is_absolute():
        output_path = PROJECT / output_path
    result = pd.read_csv(input_path)
    result["target_day"] = result["target_day"].astype(str)
    rows: list[dict[str, object]] = []
    rows.append(audit_group(result, "all_tasks"))
    for task, task_group in result.groupby("task"):
        rows.append(audit_group(task_group, f"task={task}"))
        for period, period_group in task_group.groupby("period"):
            rows.append(audit_group(period_group, f"task={task},period={period}"))
            ordered = period_group.sort_values("target_day").reset_index(drop=True)
            midpoint = len(ordered) // 2
            rows.append(audit_group(ordered.iloc[:midpoint], f"task={task},period={period},first_half"))
            rows.append(audit_group(ordered.iloc[midpoint:], f"task={task},period={period},second_half"))
    audit = pd.DataFrame(rows)
    audit.to_csv(output_path, index=False)
    metadata = {
        "input": str(input_path.relative_to(PROJECT)),
        "output": str(output_path.relative_to(PROJECT)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "interpretation": "negative delta means policy beats rolling champion; CI excluding zero is stronger evidence",
    }
    (output_path.parent / "stability_audit_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit.to_string(index=False))
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
