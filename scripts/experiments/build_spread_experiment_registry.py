"""生成价差实验注册表、路线比较表和可审计的状态摘要。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import stat
from pathlib import Path
from statistics import mean


REQUIRED_ROUTE_COLUMNS = [
    "route", "variant", "raw_direction_accuracy", "positive_recall",
    "negative_recall", "balanced_accuracy", "all_negative_baseline",
    "gain_vs_all_negative", "leakage_status", "final_holdout_touched",
]
SKIP_DIRS = {"cache", "ledger", "runs", "models", "checkpoints", "__pycache__"}


def is_reparse_point(path: Path) -> bool:
    try:
        return bool(getattr(os.lstat(path), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return True


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def nearest_manifest(directory: Path) -> tuple[Path | None, dict]:
    candidates = sorted(directory.glob("*manifest*.json")) + sorted(directory.glob("manifest.json"))
    for path in candidates:
        data = read_json(path)
        if data:
            return path, data
    return None, {}


def invalid_marker(directory: Path) -> tuple[str, str]:
    for path in directory.glob("INVALID*.json"):
        data = read_json(path)
        reason = str(data.get("reason") or data.get("summary") or data.get("details") or path.name)
        return "INVALID-LEAKAGE", reason.replace("\n", " ")[:500]
    for path in directory.glob("*PRIVILEGED*.json"):
        return "PRIVILEGED", path.name
    return "", ""


def infer_leakage(directory: Path, manifest: dict) -> tuple[str, str]:
    status, reason = invalid_marker(directory)
    if status:
        return status, reason
    if not manifest:
        return "LEGACY-UNVERIFIED", "no manifest"
    strict = str(manifest.get("leakage_status", "")).upper()
    if strict in {"INVALID-LEAKAGE", "ORACLE", "PRIVILEGED", "STRICT/PASS", "STRICT"}:
        return strict, str(manifest.get("leakage_reason", ""))[:500]
    if manifest.get("final_holdout_touched") is True:
        return "INVALID-LEAKAGE", "manifest final_holdout_touched=true"
    cutoff = manifest.get("training_last_day") or manifest.get("training_label_cutoff") or manifest.get("training_cutoff")
    strict_audit = (
        manifest.get("strict_training_audit_pass") is True
        or bool(manifest.get("router_training_labels"))
        or bool(manifest.get("threshold_training_labels"))
    )
    if manifest.get("forecast_origin") and (cutoff or strict_audit) and manifest.get("final_holdout_touched") is not True:
        return "STRICT/PASS", "manifest contains forecast origin and training cutoff"
    return "LEGACY-UNVERIFIED", "missing strict audit fields"


def as_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def summary_metrics(directory: Path) -> dict[str, float | str]:
    """只读取小型汇总 CSV，不扫描 ledger/parquet。"""
    candidates = []
    for name in ("summary.csv", "robustness.csv", "monthly.csv", "confirm_metrics.csv", "route_comparison.csv"):
        candidates.extend(directory.glob(name))
    raw, bal, pos, neg, base = [], [], [], [], []
    for path in candidates:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    normalized = {str(k).lower().strip().replace(" ", "_"): v for k, v in row.items()}
                    def get(*keys):
                        for key in keys:
                            if key in normalized:
                                return as_float(normalized[key])
                        return None
                    for target, keys in [
                        (raw, ("raw_direction_accuracy", "direction_accuracy", "accuracy", "raw_acc", "mean_month_acc")),
                        (bal, ("balanced_accuracy", "balanced_acc", "bal_acc", "balanced_direction_accuracy", "mean_month_bal")),
                        (pos, ("positive_recall", "pos_recall", "recall_positive", "positive_accuracy")),
                        (neg, ("negative_recall", "neg_recall", "recall_negative", "negative_accuracy")),
                        (base, ("all_negative_baseline", "all_negative", "baseline", "all_negative_accuracy")),
                    ]:
                        value = get(*keys)
                        if value is not None:
                            target.append(value)
        except (OSError, csv.Error):
            continue
    out: dict[str, float | str] = {}
    for name, values in (("raw_direction_accuracy", raw), ("balanced_accuracy", bal), ("positive_recall", pos), ("negative_recall", neg), ("all_negative_baseline", base)):
        if values:
            out[f"{name}_max"] = max(values)
            out[f"{name}_mean"] = mean(values)
    if raw and base:
        out["gain_vs_all_negative_mean"] = mean(raw) - mean(base)
    return out


def monthly_variant_metrics(directory: Path) -> list[tuple[str, dict[str, float]]]:
    """按 variant 汇总 monthly.csv，避免把一个路由器目录里的多个专家平均成一个模型。"""
    path = directory / "monthly.csv"
    if not path.exists():
        return []
    buckets: dict[str, dict[str, list[float]]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                variant = str(row.get("variant") or directory.name)
                b = buckets.setdefault(variant, {})
                aliases = {
                    "raw": ("direction_accuracy", "raw_direction_accuracy", "mean_month_acc"),
                    "pos": ("positive_recall", "positive_accuracy"),
                    "neg": ("negative_recall", "negative_accuracy"),
                    "bal": ("balanced_accuracy", "balanced_direction_accuracy", "mean_month_bal"),
                    "base": ("all_negative_baseline", "all_negative_accuracy"),
                }
                for key, names in aliases.items():
                    for name in names:
                        value = as_float(row.get(name))
                        if value is not None:
                            b.setdefault(key, []).append(value)
                            break
    except (OSError, csv.Error):
        return []
    result = []
    for variant, values in buckets.items():
        metrics = {f"{key}_mean": mean(items) for key, items in values.items() if items}
        if "raw_mean" in metrics:
            metrics["raw_direction_accuracy_mean"] = metrics["raw_mean"]
        if "pos_mean" in metrics:
            metrics["positive_recall_mean"] = metrics["pos_mean"]
        if "neg_mean" in metrics:
            metrics["negative_recall_mean"] = metrics["neg_mean"]
        if "bal_mean" in metrics:
            metrics["balanced_accuracy_mean"] = metrics["bal_mean"]
        if "base_mean" in metrics:
            metrics["all_negative_baseline_mean"] = metrics["base_mean"]
        result.append((variant, metrics))
    return result


def experiment_candidates(root: Path):
    def ignore_error(_error):
        return None
    for current, dirs, _files in os.walk(root, topdown=True, onerror=ignore_error):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not is_reparse_point(Path(current) / d)]
        path = Path(current)
        if any(part in SKIP_DIRS for part in path.parts) or is_reparse_point(path):
            continue
        manifest_path, manifest = nearest_manifest(path)
        invalid = any(path.glob("INVALID*.json"))
        evidence = manifest_path or invalid or any(path.glob("summary.csv")) or any(path.glob("robustness.csv")) or any(path.glob("monthly.csv"))
        if evidence:
            yield path, manifest_path, manifest


def write_registry(root: Path) -> Path:
    registry_dir = root / "00_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    output = registry_dir / "experiment_index.csv"
    rows = []
    seen = set()
    for path, manifest_path, manifest in sorted(experiment_candidates(root), key=lambda x: str(x[0]).lower()):
        relative = path.relative_to(root).as_posix()
        experiment_id = str(manifest.get("experiment_id") or manifest.get("run_id") or path.name)
        unique_id = f"{experiment_id}@{relative}"
        if unique_id in seen:
            continue
        seen.add(unique_id)
        leakage, leakage_reason = infer_leakage(path, manifest)
        metrics = summary_metrics(path)
        requested_status = str(manifest.get("experiment_status", "")).upper()
        if requested_status not in {"ACTIVE", "CANDIDATE", "RETAINED", "REJECTED", "INVALID-LEAKAGE", "ORACLE", "PRIVILEGED"}:
            requested_status = ""
        inferred_status = "INVALID-LEAKAGE" if leakage == "INVALID-LEAKAGE" else ("ACTIVE" if leakage == "STRICT/PASS" else "RETAINED")
        row = {
            "experiment_id": experiment_id,
            "relative_path": relative,
            "status": requested_status or inferred_status,
            "leakage_status": leakage,
            "leakage_reason": leakage_reason,
            "forecast_origin": manifest.get("forecast_origin", ""),
            "training_last_day": manifest.get("training_last_day", manifest.get("training_label_cutoff", "")),
            "final_holdout_touched": manifest.get("final_holdout_touched", ""),
            "manifest_path": manifest_path.relative_to(root).as_posix() if manifest_path else "",
        }
        row.update(metrics)
        rows.append(row)
    columns = ["experiment_id", "relative_path", "status", "leakage_status", "leakage_reason", "forecast_origin", "training_last_day", "final_holdout_touched", "manifest_path", "raw_direction_accuracy_max", "raw_direction_accuracy_mean", "balanced_accuracy_max", "balanced_accuracy_mean", "positive_recall_mean", "negative_recall_mean", "all_negative_baseline_mean", "gain_vs_all_negative_mean"]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output


def route_rows(root: Path) -> list[dict]:
    """从 goal70 目录的 strict 结果中抽取路线候选；不把旧泄露结果自动晋级。"""
    rows = []
    goals = [
        root / "01_spread_24" / "main_strict_dsa" / "spread_direction_24_goal70_20260822",
        root / "01_spread_24" / "invalid_leakage" / "goal70_20260822",
        root / "01_spread_24" / "main_strict_dsa",
        root / "01_spread_24" / "alternative_distributional",
    ]
    for goal in goals:
        if not goal.is_dir():
            continue
        for directory in sorted(p for p in goal.iterdir() if p.is_dir()):
            _manifest_path, manifest = nearest_manifest(directory)
            if manifest.get("route_comparison_exclude") is True:
                continue
            leakage, _ = infer_leakage(directory, manifest)
            metric_variants = monthly_variant_metrics(directory)
            if not metric_variants:
                metric_variants = [(directory.name, summary_metrics(directory))]
            for variant, metrics in metric_variants:
                if not metrics:
                    continue
                raw = metrics.get("raw_direction_accuracy_mean", "")
                base = metrics.get("all_negative_baseline_mean", "")
                gain = (raw - base) if isinstance(raw, float) and isinstance(base, float) else ""
                if "alternative_distributional" in goal.as_posix():
                    route = "B_distributional_states"
                elif "main_strict_dsa" in goal.as_posix() or "invalid_leakage" in goal.as_posix():
                    route = "screening" if any(token in variant.lower() for token in ("debug", "smoke")) else "A_strict_DSA"
                else:
                    route = "screening"
                rows.append({
                    "route": route,
                    "variant": f"{directory.name}::{variant}" if variant != directory.name else variant,
                    "raw_direction_accuracy": raw,
                    "positive_recall": metrics.get("positive_recall_mean", ""),
                    "negative_recall": metrics.get("negative_recall_mean", ""),
                    "balanced_accuracy": metrics.get("balanced_accuracy_mean", ""),
                    "all_negative_baseline": base,
                    "gain_vs_all_negative": gain,
                    "leakage_status": leakage,
                    "final_holdout_touched": str(manifest.get("final_holdout_touched", False)).lower(),
                })
    return rows


def write_route_comparison(root: Path) -> tuple[Path, Path]:
    registry_dir = root / "00_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    rows = route_rows(root)
    csv_path = registry_dir / "route_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_ROUTE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    strict_all = [
        r for r in rows
        if r["leakage_status"] == "STRICT/PASS"
        and isinstance(r["raw_direction_accuracy"], float)
        and not any(token in r["variant"].lower() for token in ("debug", "smoke"))
    ]
    fresh = [r for r in strict_all if "_5m" in r["variant"].lower()]
    strict = fresh or strict_all
    best = max(strict, key=lambda r: r["raw_direction_accuracy"], default=None)
    md_path = registry_dir / "route_comparison.md"
    lines = [
        "# 价差双支线路线比较",
        "",
        "状态：仅纳入 `STRICT/PASS` 且非 debug/smoke 的数字参与路线排名；旧泄露结果保留但不得晋级。",
        "",
        "## 当前结论",
        "",
    ]
    if best:
        lines.append(f"- 严格候选当前最高总体 raw direction accuracy（优先采用最新 5 个月完整窗口）：`{best['variant']}` = `{best['raw_direction_accuracy']:.4f}`。")
    else:
        lines.append("- 当前没有足够的 STRICT/PASS 候选进入排名。")
    lines.extend([
        "- 最新shadow候选不参与主线晋级排名；其结果单独记录，避免局部开发窗口覆盖更长确认窗口。",
        "- 主线：A，strict-D2 + DSA/Similar-Day + P6 + D-1 p1-p14 上下文 + 预序列动态路由。",
        "- 备线：B，regular/positive-spike/negative-spike 分布式状态模型；首轮 pilot 已进入比较，但未达到主线切换门槛。",
        "- 切换门槛：同一新鲜留出集上 raw 与 balanced 均至少提升 1 个百分点，且正负召回、多月基线条件不恶化。",
        "",
        "## 字段",
        "",
        "`route_comparison.csv` 提供总体候选汇总；跨月明细仍以各实验目录中的 `monthly.csv`/`robustness.csv` 为准。",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"实验目录不存在：{root}")
    print(write_registry(root))
    csv_path, md_path = write_route_comparison(root)
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
