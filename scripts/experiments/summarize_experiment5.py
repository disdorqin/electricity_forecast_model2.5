"""快速扫描实验五 outputs/experiments，生成可追溯归类索引。

该脚本不移动产物，避免破坏历史 manifest、相对路径和兼容 junction；归类通过机器可读
索引完成，后续新链路只读取索引和明确标记的实验。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import stat
from pathlib import Path


SKIP = {"cache", "ledger", "runs", "models", "checkpoints", "__pycache__"}
CATEGORIES = {"00_registry", "01_spread_24", "02_spread_96", "03_fusion_weighting", "04_pipeline_audits", "99_legacy_unclassified"}


def reparse(path: Path) -> bool:
    try:
        return bool(getattr(os.lstat(path), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return True


def walk(root: Path):
    def onerror(_):
        return None
    for current, dirs, files in os.walk(root, topdown=True, onerror=onerror):
        dirs[:] = [d for d in dirs if d not in SKIP and not reparse(Path(current) / d)]
        path = Path(current)
        if reparse(path):
            continue
        yield path, files


def load_manifest(path: Path) -> dict:
    for candidate in (path / "manifest.json", path / "run_manifest.json"):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def route_for(relative: str) -> str:
    if relative.startswith("01_spread_24/spread_forecast_24_96_chain_v2"):
        return "research_chain_24_96"
    if "/04_diagnostics" in relative:
        return "research_diagnostics"
    if "/alternative_distributional/" in relative:
        return "B_distributional_states"
    if "/invalid_leakage/" in relative:
        return "INVALID_LEAKAGE"
    if "/main_strict_dsa/" in relative:
        return "A_strict_DSA"
    if relative.startswith("02_spread_96/"):
        return "spread_96"
    if relative.startswith("03_fusion_weighting/"):
        return "fusion_weighting"
    if relative.startswith("04_pipeline_audits/"):
        return "pipeline_audit"
    return "legacy_unclassified"


def method_family(name: str) -> str:
    n = name.lower()
    groups = [
        ("similar_day_dsa", ("similar", "dsa", "analog", "regime_selection")),
        ("context_features", ("context", "ctx", "f5", "f6", "f7", "f8", "feature")),
        ("router_and_gate", ("router", "gate", "activation", "disagreement", "hierarchical")),
        ("calibration_and_threshold", ("calibration", "threshold", "probability_ranking", "confidence")),
        ("positive_event", ("positive", "tail", "two_stage", "magnitude")),
        ("state_distributional", ("dart", "state", "three_state", "severity")),
        ("baseline_and_symmetric", ("baseline", "symmetric", "override")),
        ("pipeline_and_fusion", ("chain", "audit", "weight", "fusion", "re_prediction")),
    ]
    for label, tokens in groups:
        if any(token in n for token in tokens):
            return label
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()
    root = args.root.resolve()
    registry = root / "00_registry"
    registry.mkdir(parents=True, exist_ok=True)
    candidates = []
    for path, files in walk(root):
        relative = path.relative_to(root).as_posix()
        if not relative or relative.startswith("00_registry"):
            continue
        new_chain_root = "01_spread_24/spread_forecast_24_96_chain_v2"
        in_new_chain = (
            relative == new_chain_root
            or (
                relative.startswith(new_chain_root + "/cycles/")
                and ("manifest.json" in files or path.name.startswith("cycle_"))
            )
        )
        if not (in_new_chain or path.name.startswith("cycle_") or path.name.startswith("spread_") or path.name.startswith("feature_") or path.name.startswith("weight_") or path.name.startswith("chain_") or path.name.startswith("re_prediction_") or path.parent.name in CATEGORIES):
            continue
        manifest = load_manifest(path)
        invalid = any(path.glob("INVALID*.json"))
        status = "INVALID-LEAKAGE" if invalid else str(
            manifest.get("experiment_status") or manifest.get("status") or "LEGACY-UNVERIFIED"
        ).upper()
        candidates.append({
            "relative_path": relative,
            "route": route_for(relative),
            "method_family": method_family(path.name),
            "experiment_name": path.name,
            "status": status,
            "forecast_origin": manifest.get("forecast_origin", ""),
            "training_last_day": manifest.get("training_last_day", manifest.get("training_label_cutoff", "")),
            "final_holdout_touched": manifest.get("final_holdout_touched", ""),
            "manifest_experiment": manifest.get("experiment", manifest.get("route", "")),
            "file_count": len(files),
        })
    candidates.sort(key=lambda row: row["relative_path"].lower())
    output = registry / "experiment5_inventory.csv"
    columns = list(candidates[0].keys()) if candidates else ["relative_path"]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(candidates)
    counts = {}
    for row in candidates:
        key = (row["route"], row["method_family"], row["status"])
        counts[key] = counts.get(key, 0) + 1
    summary = registry / "method_outcome_index.csv"
    with summary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["route", "method_family", "status", "experiment_count"])
        for (route, family, status), count in sorted(counts.items()):
            writer.writerow([route, family, status, count])
    # 根目录保留的 junction 只是旧脚本兼容入口，不是第二份实验产物。
    # 单独登记，避免物理归档时误把兼容层当作未整理的实验目录。
    path_map = registry / "path_map.csv"
    compat_output = registry / "legacy_compat_manifest.csv"
    compat_rows = []
    if path_map.exists():
        with path_map.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if str(row.get("status", "")).upper() not in {
                    "COMPATIBILITY_LINK", "LEGACY_PATH_REMOVED"
                }:
                    continue
                old_path = Path(row.get("old_path", ""))
                new_path = Path(row.get("new_path", ""))
                if old_path.parent.resolve() != root:
                    continue
                compat_rows.append({
                    "legacy_name": old_path.name,
                    "legacy_path": old_path.as_posix(),
                    "organized_target": new_path.as_posix(),
                    "category": row.get("category", ""),
                    "reference_count": row.get("reference_count", "0"),
                    "status": (
                        "COMPAT_JUNCTION"
                        if str(row.get("status", "")).upper() == "COMPATIBILITY_LINK"
                        else "LEGACY_PATH_REMOVED"
                    ),
                    "note": (
                        "旧脚本路径仍保留；真实实验只认organized_target"
                        if str(row.get("status", "")).upper() == "COMPATIBILITY_LINK"
                        else "源码引用已迁移；旧路径已物理移除；真实实验只认organized_target"
                    ),
                })
    compat_rows.sort(key=lambda row: row["legacy_name"].lower())
    with compat_output.open("w", newline="", encoding="utf-8-sig") as f:
        columns = ["legacy_name", "legacy_path", "organized_target", "category", "reference_count", "status", "note"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(compat_rows)
    print(f"inventory={output} rows={len(candidates)}")
    print(f"method_summary={summary} groups={len(counts)}")
    print(f"legacy_compat_manifest={compat_output} rows={len(compat_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
