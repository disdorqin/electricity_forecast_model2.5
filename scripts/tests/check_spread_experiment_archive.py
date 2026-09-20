"""价差实验室归档与双支线路线注册表验收。"""

from __future__ import annotations

import csv
import json
import os
import stat
from pathlib import Path


ROOT = Path("outputs/experiments")
CATEGORIES = {"00_registry", "01_spread_24", "02_spread_96", "03_fusion_weighting", "04_pipeline_audits", "99_legacy_unclassified"}
REQUIRED_ROUTE = {"route", "variant", "raw_direction_accuracy", "positive_recall", "negative_recall", "balanced_accuracy", "all_negative_baseline", "gain_vs_all_negative", "leakage_status", "final_holdout_touched"}


def reparse(path: Path) -> bool:
    try:
        raw = os.path.abspath(os.fspath(path))
        # Windows legacy MAX_PATH can make a normal long file look missing;
        # use the extended-length namespace before declaring a reparse point.
        if os.name == "nt" and not raw.startswith("\\\\?\\"):
            raw = "\\\\?\\" + raw
        return bool(
            getattr(os.lstat(raw), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
    except OSError:
        return True


def main() -> int:
    failures = []
    if not ROOT.is_dir():
        failures.append(f"missing {ROOT}")
    for name in ("README.md", "00_registry/experiment_index.csv", "00_registry/path_map.csv", "00_registry/route_state.json", "00_registry/route_comparison.csv", "00_registry/route_comparison.md"):
        if not (ROOT / name).exists():
            failures.append(f"missing {ROOT / name}")
    top_entries = list(ROOT.iterdir())
    regular_top = [p.name for p in top_entries if p.is_dir() and not reparse(p) and p.name not in CATEGORIES]
    if regular_top:
        failures.append(f"unclassified regular top-level dirs: {regular_top}")
    top_reparse = [p.name for p in top_entries if reparse(p)]
    if top_reparse:
        failures.append(f"top-level reparse/junction paths are forbidden: {top_reparse}")

    # The archive must be a physical layout, not a collection of junctions.
    # Do not follow reparse points while walking so a stale link cannot hide
    # an unclassified tree outside the laboratory directory.
    reparse_paths = []
    for current, dirs, files in os.walk(ROOT, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_dirs = []
        for name in dirs:
            path = current_path / name
            if reparse(path):
                reparse_paths.append(str(path))
            else:
                safe_dirs.append(name)
        dirs[:] = safe_dirs
        for name in files:
            path = current_path / name
            if reparse(path):
                reparse_paths.append(str(path))
    if reparse_paths:
        failures.append(f"archive contains reparse/junction paths: {reparse_paths[:10]}")

    chain = ROOT / "01_spread_24/spread_forecast_24_96_chain_v2"
    if not chain.is_dir() or reparse(chain):
        failures.append(f"new research chain is not a physical directory inside laboratory: {chain}")
    project_root_chain = ROOT.parent.parent / "spread_forecast_24_96_chain_v2"
    if project_root_chain.exists():
        failures.append(f"research chain must not exist outside laboratory: {project_root_chain}")
    path_map = ROOT / "00_registry/path_map.csv"
    if path_map.exists():
        with path_map.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row.get("status") == "COLLISION":
                # 迁移后目标必然存在；只有源和目标同时存在才是真冲突。
                if (Path(row["old_path"]).exists() and not reparse(Path(row["old_path"]))) or not Path(row["new_path"]).exists():
                    failures.append(f"path collision: {row.get('new_path')}")
            if not Path(row["new_path"]).exists():
                failures.append(f"missing migrated destination: {row['new_path']}")
    index = ROOT / "00_registry/experiment_index.csv"
    if index.exists():
        with index.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        paths = [r["relative_path"] for r in rows]
        if len(paths) != len(set(paths)):
            failures.append("duplicate relative_path in experiment_index")
        allowed = {"ACTIVE", "CANDIDATE", "RETAINED", "REJECTED", "INVALID-LEAKAGE", "ORACLE", "PRIVILEGED"}
        for row in rows:
            if row.get("status") not in allowed:
                failures.append(f"unknown registry status: {row.get('status')}")
    route = ROOT / "00_registry/route_comparison.csv"
    if route.exists():
        with route.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not REQUIRED_ROUTE.issubset(rows[0].keys() if rows else set()):
            failures.append("route_comparison missing required columns")
        if not any(r.get("route") == "A_strict_DSA" for r in rows):
            failures.append("route comparison missing A")
        if not any(r.get("route") == "B_distributional_states" for r in rows):
            failures.append("route comparison missing B pilot")
        if any(r.get("leakage_status") == "INVALID-LEAKAGE" and r.get("variant", "").startswith("cycle_01") for r in rows):
            failures.append("cycle pilot unexpectedly marked invalid")
    state_path = ROOT / "00_registry/route_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("main_route", {}).get("id") != "A_strict_DSA":
            failures.append("main route state is not A_strict_DSA")
        if state.get("alternative_route", {}).get("id") != "B_distributional_states":
            failures.append("alternative route state is not B_distributional_states")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[PASS] spread experiment archive and route registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
