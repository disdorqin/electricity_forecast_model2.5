"""Cycle 10 retrospective diagnostics for A expert errors and B state stability.

This script consumes strict OOS ledgers only. It does not fit a model, choose a
threshold, or open the final holdout. Its purpose is to identify where a future
pilot should spend capacity: hourly/period error concentration for A and the
stability of rolling state quantiles for B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")


def _metric_rows(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        y = group["y"].to_numpy(int)
        p = group["pred"].to_numpy(int)
        pos, neg = y > 0, y < 0
        pr = float((p[pos] == 1).mean()) if pos.any() else float("nan")
        nr = float((p[neg] == -1).mean()) if neg.any() else float("nan")
        row = dict(zip(keys, values))
        row.update({
            "slots": int(len(group)),
            "raw_accuracy": float((p == y).mean()),
            "positive_recall": pr,
            "negative_recall": nr,
            "balanced_accuracy": float(np.nanmean([pr, nr])),
            "all_negative_baseline": float(neg.mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose_a(path: Path, output: Path) -> dict[str, object]:
    ledger = pd.read_parquet(path)
    required = {"target_day", "hour_business", "target_spread", "predicted_direction", "variant"}
    missing = required.difference(ledger.columns)
    if missing:
        raise RuntimeError(f"A ledger missing columns: {sorted(missing)}")
    ledger["target_day"] = pd.to_datetime(ledger["target_day"]).dt.normalize()
    if ledger["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("A diagnostic touches final holdout")
    ledger["y"] = np.sign(pd.to_numeric(ledger["target_spread"], errors="coerce")).astype(int)
    ledger["pred"] = pd.to_numeric(ledger["predicted_direction"], errors="coerce").astype(int)
    ledger["month"] = ledger["target_day"].dt.strftime("%Y-%m")
    ledger["period"] = pd.cut(ledger["hour_business"], [0, 8, 16, 24], labels=["1_8", "9_16", "17_24"])
    ledger["correct"] = ledger["y"].eq(ledger["pred"])
    monthly = _metric_rows(ledger, ["variant", "month"])
    period = _metric_rows(ledger, ["variant", "period"])
    hourly = _metric_rows(ledger, ["variant", "hour_business"])
    miss = ledger.loc[~ledger["correct"]].copy()
    miss["error_type"] = np.select([
        (miss["y"] > 0) & (miss["pred"] < 0),
        (miss["y"] < 0) & (miss["pred"] > 0),
    ], ["positive_missed", "negative_false_positive"], default="zero_or_invalid")
    error_by_hour = miss.groupby(["variant", "hour_business", "error_type"], as_index=False).size()
    wide = ledger.pivot_table(index=["target_day", "hour_business"], columns="variant", values="pred", aggfunc="first")
    wide.columns.name = None
    variant_cols = list(wide.columns)
    wide["n_unique_predictions"] = wide[variant_cols].nunique(axis=1, dropna=True)
    wide["experts_disagree"] = wide["n_unique_predictions"] > 1
    truth = ledger.drop_duplicates(["target_day", "hour_business"])[["target_day", "hour_business", "y"]].set_index(["target_day", "hour_business"])
    wide = wide.join(truth)
    wide["any_variant_correct"] = (wide[variant_cols].to_numpy() == wide["y"].to_numpy()[:, None]).any(axis=1)
    agreement = wide.groupby("experts_disagree", as_index=False).agg(
        slots=("y", "size"), any_variant_correct=("any_variant_correct", "mean"),
        positive_rate=("y", lambda x: float((x > 0).mean())),
    )
    oracle = {
        "status": "ORACLE_DIAGNOSTIC_ONLY",
        "variants": variant_cols,
        "any_variant_correct": float(wide["any_variant_correct"].mean()),
        "not_a_production_score": True,
    }
    monthly.to_csv(output / "a_monthly_metrics.csv", index=False, encoding="utf-8-sig")
    period.to_csv(output / "a_period_metrics.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(output / "a_hourly_metrics.csv", index=False, encoding="utf-8-sig")
    error_by_hour.to_csv(output / "a_error_by_hour.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(output / "a_disagreement_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "a_oracle_diagnostic.json").write_text(json.dumps(oracle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_hours = hourly.sort_values(["variant", "raw_accuracy"], ascending=[True, False]).groupby("variant", as_index=False).head(3)
    return {"variants": variant_cols, "oracle_any_variant_correct": oracle["any_variant_correct"], "best_hours": best_hours.to_dict(orient="records")}


def diagnose_b(paths: list[Path], output: Path) -> dict[str, object]:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"target_day", "variant", "q10", "q90"}
        if required.difference(frame.columns):
            raise RuntimeError(f"B daily metrics missing columns in {path}: {sorted(required.difference(frame.columns))}")
        frame["source"] = path.parent.name
        frames.append(frame)
    b = pd.concat(frames, ignore_index=True)
    b["target_day"] = pd.to_datetime(b["target_day"]).dt.normalize()
    if b["target_day"].max() >= FINAL_HOLDOUT_START:
        raise RuntimeError("B diagnostic touches final holdout")
    b["month"] = b["target_day"].dt.strftime("%Y-%m")
    b["state_width"] = b["q90"] - b["q10"]
    b = b.sort_values(["source", "variant", "target_day"])
    b["q10_abs_step"] = b.groupby(["source", "variant"])["q10"].diff().abs()
    b["q90_abs_step"] = b.groupby(["source", "variant"])["q90"].diff().abs()
    monthly = b.groupby(["source", "variant", "month"], as_index=False).agg(
        days=("target_day", "nunique"), q10_mean=("q10", "mean"), q10_std=("q10", "std"),
        q10_min=("q10", "min"), q10_max=("q10", "max"), q90_mean=("q90", "mean"),
        q90_std=("q90", "std"), q90_min=("q90", "min"), q90_max=("q90", "max"),
        state_width_mean=("state_width", "mean"), q10_abs_step_mean=("q10_abs_step", "mean"),
        q90_abs_step_mean=("q90_abs_step", "mean"),
    )
    overall = b.groupby(["source", "variant"], as_index=False).agg(
        days=("target_day", "nunique"), q10_mean=("q10", "mean"), q10_std=("q10", "std"),
        q90_mean=("q90", "mean"), q90_std=("q90", "std"), state_width_mean=("state_width", "mean"),
        state_width_std=("state_width", "std"), q10_abs_step_mean=("q10_abs_step", "mean"),
        q90_abs_step_mean=("q90_abs_step", "mean"),
    )
    b.to_csv(output / "b_daily_quantile_audit.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "b_monthly_quantile_stability.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output / "b_quantile_stability_summary.csv", index=False, encoding="utf-8-sig")
    return {"sources": [str(p) for p in paths], "summary": overall.to_dict(orient="records")}


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    a_summary = diagnose_a(args.a_ledger.resolve(), output)
    b_paths = [Path(p).resolve() for p in args.b_daily_metrics]
    b_summary = diagnose_b(b_paths, output)
    manifest = {
        "status": "STRICT/PASS",
        "diagnostic_only": True,
        "forecast_origin": "D-1 14:00",
        "training_last_day": "source ledgers asserted <= D-2; no refit performed",
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False,
        "label_usage": "retrospective error/state diagnostics only",
        "a_summary": a_summary,
        "b_summary": b_summary,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    report = [
        "# Cycle 10：专家错误结构与状态稳定性诊断",
        "",
        "- 状态：`STRICT/PASS`；性质：`diagnostic_only`。",
        "- 只读取历史 strict OOS，未训练、未调阈值、未选择模型，final holdout 未触碰。",
        "- A 的 `a_oracle_diagnostic.json` 仅表示“已有专家中至少一个正确”的上限诊断，不是可交付成绩。",
        "",
        "## 诊断产物",
        "",
        "- A：`a_monthly_metrics.csv`、`a_period_metrics.csv`、`a_hourly_metrics.csv`、`a_error_by_hour.csv`、`a_disagreement_metrics.csv`。",
        "- B：`b_monthly_quantile_stability.csv`、`b_quantile_stability_summary.csv`、`b_daily_quantile_audit.csv`。",
        "",
        "下一轮候选只能根据这些诊断提出假设，仍需重新通过开发集→确认集→新鲜留出集流程。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": "STRICT/PASS"}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-ledger", type=Path, required=True)
    parser.add_argument("--b-daily-metrics", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
