"""迁移 outputs/experiments 根目录旧路径的源码引用。

实验产物已经物理归档到分类目录。根目录旧路径曾通过 junction 兼容，但这会让资源管理器
看起来像“没有整理”。本工具只改源码、配置和文档中的精确路径，不改 outputs/ 下的历史
产物；完成引用迁移后即可安全移除旧 junction。
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


SKIP_SUFFIXES = {
    ".parquet", ".csv", ".xlsx", ".xls", ".zip", ".pt", ".pth", ".bin",
    ".png", ".jpg", ".jpeg", ".pdf", ".pyc",
}


def load_replacements(root: Path) -> dict[str, str]:
    experiments = root / "outputs" / "experiments"
    path_map = experiments / "00_registry" / "path_map.csv"
    replacements: dict[str, str] = {}
    with path_map.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).upper() != "COMPATIBILITY_LINK":
                continue
            old_name = Path(row["old_path"]).name
            new_path = Path(row["new_path"])
            relative_target = new_path.relative_to(experiments).as_posix()
            replacements[f"outputs/experiments/{old_name}"] = (
                f"outputs/experiments/{relative_target}"
            )
    return replacements


def candidate_files(root: Path) -> list[Path]:
    try:
        raw = subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=root,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        paths = [root / line for line in raw.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        paths = [p for p in root.rglob("*") if p.is_file()]
    result = []
    output_dir = (root / "outputs").resolve()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not path.is_file() or resolved == output_dir or output_dir in resolved.parents:
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        result.append(path)
    return sorted(set(result))


def find_hits(root: Path, replacements: dict[str, str]) -> list[tuple[Path, list[str]]]:
    hits: list[tuple[Path, list[str]]] = []
    for path in candidate_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        matched = [old for old in replacements if old in text]
        if matched:
            hits.append((path, matched))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="写入精确路径迁移")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    replacements = load_replacements(root)
    hits = find_hits(root, replacements)
    if args.apply:
        changed = 0
        for path, _ in hits:
            text = path.read_text(encoding="utf-8")
            for old, new in replacements.items():
                text = text.replace(old, new)
            path.write_text(text, encoding="utf-8", newline="")
            changed += 1
        print(f"updated_files={changed}")
    else:
        print(f"files_with_legacy_refs={len(hits)}")
    for path, matched in hits:
        print(f"{path.relative_to(root)}: {', '.join(matched)}")
    return 0 if args.apply or not hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
