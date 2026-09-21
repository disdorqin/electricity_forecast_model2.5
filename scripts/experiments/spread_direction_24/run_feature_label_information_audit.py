"""Audit feature availability and spread-direction label information.

This is a diagnostic, not a feature-selection or model-selection step. It uses
the feature registry as the contract source, reports missingness/variation and
retrospective label association, and explicitly flags target-like names. No
feature or label is modified and the final holdout remains sealed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_HOLDOUT_START = pd.Timestamp("2026-08-15")
FORBIDDEN_NAME_TOKENS = ("actual", "realized", "target_day_actual", "实时电价", "日前电价")
ALLOWED_CONTEXT_TOKENS = ("spread_lag", "ctx_spread", "sd")


def numeric_feature_stats(slot: pd.DataFrame, registry: dict) -> pd.DataFrame:
    rows = []
    for item in registry.get("features", []):
        name = str(item.get("feature", ""))
        if not name or name not in slot.columns:
            rows.append({
                "group": item.get("group", ""), "feature": name, "source": item.get("source", ""),
                "availability": item.get("availability", ""), "missing_in_slot_table": True,
            })
            continue
        s = pd.to_numeric(slot[name], errors="coerce")
        finite = s[np.isfinite(s)]
        lower = name.lower()
        forbidden = [token for token in FORBIDDEN_NAME_TOKENS if token.lower() in lower or token in name]
        allowed_context = any(token.lower() in lower for token in ALLOWED_CONTEXT_TOKENS)
        rows.append({
            "group": item.get("group", ""), "feature": name, "source": item.get("source", ""),
            "availability": item.get("availability", ""), "task": item.get("task", ""),
            "leakage_status": item.get("leakage_status", ""), "missing_rate": float(s.isna().mean()),
            "finite_rows": int(len(finite)), "n_unique": int(finite.nunique()),
            "std": float(finite.std(ddof=0)) if len(finite) else float("nan"),
            "q01": float(finite.quantile(0.01)) if len(finite) else float("nan"),
            "q50": float(finite.quantile(0.50)) if len(finite) else float("nan"),
            "q99": float(finite.quantile(0.99)) if len(finite) else float("nan"),
            "forbidden_name_tokens": ",".join(forbidden),
            "allowed_context_name": allowed_context,
            "missing_in_slot_table": False,
        })
    return pd.DataFrame(rows)


def label_stats(slot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = slot[["target_day", "hour_business", "target_spread"]].copy()
    work["target_day"] = pd.to_datetime(work["target_day"], errors="coerce").dt.normalize()
    work["target_spread"] = pd.to_numeric(work["target_spread"], errors="coerce")
    work = work.dropna(subset=["target_day", "target_spread"])
    work["sign"] = np.sign(work["target_spread"]).astype(int)
    work["month"] = work["target_day"].dt.strftime("%Y-%m")
    work["abs_spread"] = work["target_spread"].abs()
    work["near_1"] = work["abs_spread"] <= 1
    work["near_5"] = work["abs_spread"] <= 5
    work["near_10"] = work["abs_spread"] <= 10

    def aggregate(group: pd.DataFrame, key: str) -> pd.DataFrame:
        rows = []
        for value, g in group.groupby(key, sort=True):
            rows.append({
                key: value, "slots": int(len(g)), "positive_rate": float((g["sign"] > 0).mean()),
                "negative_rate": float((g["sign"] < 0).mean()), "zero_rate": float((g["sign"] == 0).mean()),
                "near_1_rate": float(g["near_1"].mean()), "near_5_rate": float(g["near_5"].mean()),
                "near_10_rate": float(g["near_10"].mean()), "median_abs_spread": float(g["abs_spread"].median()),
                "q90_abs_spread": float(g["abs_spread"].quantile(0.90)),
            })
        return pd.DataFrame(rows)

    monthly = aggregate(work, "month")
    hourly = aggregate(work, "hour_business")
    overall = pd.DataFrame([{
        "slots": int(len(work)), "days": int(work["target_day"].nunique()),
        "positive_rate": float((work["sign"] > 0).mean()), "negative_rate": float((work["sign"] < 0).mean()),
        "zero_rate": float((work["sign"] == 0).mean()), "near_1_rate": float(work["near_1"].mean()),
        "near_5_rate": float(work["near_5"].mean()), "near_10_rate": float(work["near_10"].mean()),
        "median_abs_spread": float(work["abs_spread"].median()), "q90_abs_spread": float(work["abs_spread"].quantile(0.90)),
    }])
    return monthly, hourly, overall


def retrospective_association(slot: pd.DataFrame, registry: dict) -> pd.DataFrame:
    work = slot.copy()
    y = np.sign(pd.to_numeric(work["target_spread"], errors="coerce")).astype(float)
    rows = []
    for item in registry.get("features", []):
        name = str(item.get("feature", ""))
        if name not in work.columns:
            continue
        x = pd.to_numeric(work[name], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 20 or x[valid].nunique() < 2:
            corr = float("nan")
        else:
            corr = float(x[valid].corr(y[valid], method="spearman"))
        rows.append({"group": item.get("group", ""), "feature": name, "spearman_sign_association": corr,
                     "abs_association": abs(corr) if np.isfinite(corr) else float("nan"),
                     "availability": item.get("availability", ""), "diagnostic_only": True})
    return pd.DataFrame(rows).sort_values("abs_association", ascending=False)


def run(args: argparse.Namespace) -> int:
    cube = args.cube.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    slot = pd.read_parquet(cube / "slot_table.parquet")
    registry = json.loads((cube / "feature_registry.json").read_text(encoding="utf-8"))
    slot["target_day"] = pd.to_datetime(slot["target_day"], errors="coerce").dt.normalize()
    audit_start = pd.Timestamp(args.start)
    audit_end = pd.Timestamp(args.end)
    if audit_end >= FINAL_HOLDOUT_START:
        raise RuntimeError("audit end touches final holdout")
    slot = slot[slot["target_day"].between(audit_start, audit_end)].copy()
    if slot.empty:
        raise RuntimeError("no rows in audited pre-holdout range")
    stats = numeric_feature_stats(slot, registry)
    monthly, hourly, overall = label_stats(slot)
    association = retrospective_association(slot, registry)
    stats.to_csv(output / "feature_stats.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "label_monthly.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(output / "label_hourly.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(output / "label_overall.csv", index=False, encoding="utf-8-sig")
    association.to_csv(output / "feature_sign_association_diagnostic.csv", index=False, encoding="utf-8-sig")
    forbidden = stats[stats["forbidden_name_tokens"].fillna("").ne("")]
    missing = stats[stats["missing_in_slot_table"].eq(True)]
    manifest = {
        "status": "STRICT/PASS" if forbidden.empty and missing.empty else "STRICT/REVIEW",
        "diagnostic_only": True, "cube": str(cube), "forecast_origin": "D-1 14:00",
        "training_last_day": "not applicable; no training performed", "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False, "d1_post14_spread_as_feature": False,
        "final_holdout_touched": False, "audit_range": [args.start, args.end], "feature_count": int(len(stats)),
        "forbidden_name_token_count": int(len(forbidden)), "missing_registry_feature_count": int(len(missing)),
        "label_usage": "retrospective audit only; no feature selection or threshold calibration",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Cycle 15：特征与标签信息量审计",
        "",
        f"- 状态：`{manifest['status']}`；性质：`diagnostic_only`。",
        "- 只读取 feature cube 和目标标签做审计，不修改特征、不选择模型、不打开 final holdout。",
        "",
        "## 标签总体",
        "",
        overall.to_markdown(index=False),
        "",
        "## 解释",
        "",
        "- `label_monthly.csv`/`label_hourly.csv` 用于识别正负比例、近零价差和跨月漂移。",
        "- `feature_stats.csv` 检查缺失率、常数列、来源和可得时间。",
        "- `feature_sign_association_diagnostic.csv` 的相关性只用于信息量诊断，不可据此直接做 feature selection。",
        "- 若出现 forbidden name token，必须回到 feature registry 做人工 contract 审查。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=Path("outputs/experiments/01_spread_24/main_strict_dsa/spread_direction_24_goal70_20260822/feature_cube"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-08-14")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
