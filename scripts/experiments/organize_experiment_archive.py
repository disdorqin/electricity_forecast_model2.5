"""审计并整理 outputs/experiments。

默认只生成迁移计划；传入 ``--apply`` 后才执行目录迁移。该工具只迁移实验目录，
不触碰正式 outputs/ledger、outputs/runs、模型权重或数据源。
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path


CATEGORY_DIRS = {
    "00_registry",
    "01_spread_24",
    "02_spread_96",
    "03_fusion_weighting",
    "04_pipeline_audits",
    "99_legacy_unclassified",
}


@dataclass(frozen=True)
class MovePlan:
    source: Path
    destination: Path
    category: str
    reference_count: int = 0


def classify(name: str) -> str:
    """返回相对于实验根目录的稳定分类路径。"""
    n = name.lower()
    if n == "spread_direction_24":
        return "01_spread_24/legacy_root"
    if n.startswith("spread_direction_24_goal70"):
        return "01_spread_24/main_strict_dsa"
    if n.startswith(("spread_direction_24_cutoff14", "spread_direction_24_causal", "spread_direction_24_lear")):
        return "01_spread_24/main_strict_dsa"
    if n.startswith(("spread_direction_24_production_sim", "spread_direction_24_safe_mixed")):
        return "01_spread_24/main_strict_dsa"
    if n.startswith(("spread_direction_24_fast_", "spread_direction_24_xlinear_")):
        return "01_spread_24/feature_model_screening"
    if n.startswith(("spread_direction_24_fill_strategy", "spread_direction_24_feature")):
        return "01_spread_24/feature_model_screening"
    if n.startswith(("spread_direction_24_", "spread_feature_", "feature_cube", "feature_family", "feature_group", "feature_package")):
        return "01_spread_24/feature_model_screening"
    if n == "spread_direction_96" or n.startswith("spread_direction_96_"):
        return "02_spread_96"
    if n.startswith(("weight_", "formal_weight", "nnls_", "champion_weight")):
        return "03_fusion_weighting"
    if n.startswith(("chain_", "authority_", "full_chain_", "re_prediction_", "replay_full_", "dynamic_lgbm_96")):
        return "04_pipeline_audits"
    if n.startswith(("classifier_", "feature_store", "accel_cache", "debug_", "logs", "results", "lightgbm_", "sgdfnet_", "timesfm_", "timemixer_", "rt916_", "warmstart_")):
        return "04_pipeline_audits"
    return "99_legacy_unclassified"


def iter_text_files(repo: Path, experiments: Path):
    # 引用审计只需源码、配置和文档；跳过数据、模型和生成产物，避免一次扫描数 GB。
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache", "outputs", "data", "models", "checkpoints", "runs"}
    for root, dirs, files in os.walk(repo):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for filename in files:
            path = root_path / filename
            if experiments in path.parents:
                continue
            try:
                if path.stat().st_size > 8 * 1024 * 1024:
                    continue
            except OSError:
                continue
            if path.suffix.lower() in {".parquet", ".pkl", ".pt", ".pth", ".bin", ".zip", ".7z", ".dll", ".exe"}:
                continue
            yield path


def count_all_references(repo: Path, experiments: Path, needles: list[str]) -> dict[str, int]:
    """单次扫描仓库，避免对几十个目录重复读取大型源码文件。"""
    counts = {needle: 0 for needle in needles}
    for path in iter_text_files(repo, experiments):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for needle in needles:
            counts[needle] += text.count(needle)
    return counts


def build_plan(repo: Path, experiments: Path) -> list[MovePlan]:
    sources: list[Path] = []
    for source in sorted(experiments.iterdir(), key=lambda p: p.name.lower()):
        if not source.is_dir() or source.name in CATEGORY_DIRS:
            continue
        try:
            if getattr(os.lstat(source), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                continue
        except OSError:
            continue
        sources.append(source)
    counts = count_all_references(repo, experiments, [source.name for source in sources])
    return [
        MovePlan(source, experiments / classify(source.name) / source.name, classify(source.name), counts[source.name])
        for source in sources
    ]


def write_path_map(experiments: Path, plans: list[MovePlan]) -> Path:
    registry = experiments / "00_registry"
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / "path_map.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["old_path", "new_path", "category", "reference_count", "collision", "status"])
        for plan in plans:
            collision = plan.source.exists() and plan.destination.exists()
            migrated = (not plan.source.exists()) and plan.destination.exists()
            status = "COLLISION" if collision else ("MIGRATED" if migrated else "PLANNED")
            writer.writerow([
                str(plan.source),
                str(plan.destination),
                plan.category,
                plan.reference_count,
                str(collision).lower(),
                status,
            ])
    return path


def apply_plans(plans: list[MovePlan]) -> None:
    collisions = [p.destination for p in plans if p.destination.exists()]
    if collisions:
        joined = "\n".join(str(p) for p in collisions)
        raise RuntimeError(f"目标已存在，拒绝部分迁移：\n{joined}")
    for plan in plans:
        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(plan.source), str(plan.destination))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--apply", action="store_true", help="执行迁移；默认只生成计划")
    args = parser.parse_args()
    repo = args.repo.resolve()
    experiments = (repo / args.root).resolve() if not args.root.is_absolute() else args.root.resolve()
    if not experiments.is_dir():
        raise SystemExit(f"实验目录不存在：{experiments}")
    plans = build_plan(repo, experiments)
    existing_map = experiments / "00_registry" / "path_map.csv"
    map_path = write_path_map(experiments, plans) if plans or not existing_map.exists() else existing_map
    print(f"迁移计划：{map_path}")
    for plan in plans:
        print(f"[{plan.category}] {plan.source.name} -> {plan.destination.relative_to(experiments)} refs={plan.reference_count}")
    if args.apply:
        apply_plans(plans)
        print(f"已迁移 {len(plans)} 个实验目录。")
    else:
        print("未执行迁移；确认计划后追加 --apply。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
