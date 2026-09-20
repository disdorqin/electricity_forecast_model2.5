from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_CONTRACT = "formal96_predictor_release_v1"

CORE_DIRS = (
    "cli",
    "pipelines",
    "runtime",
    "runners",
    "fusion",
    "utils",
    "optim",
    "lightGBM",
    "TimesFMBackend",
    "TimeMixer",
    "RT916_SpikeFusionNet",
    "SGDFNet",
)
CORE_FILES = (
    "main.py",
    "requirements.txt",
    ".env.example",
    "README.md",
    "docs/RUNBOOK.md",
    "docs/DATA_CONTRACT_96.md",
    "docs/LEAKAGE_AUDIT_96.md",
    "docs/PROJECT_LAYOUT.md",
    "docs/OUTPUT_CONVENTION.md",
    "scripts/__init__.py",
)
SCRIPT_DIRS = ("scripts/sync", "scripts/server")
MODEL_DIRS = ("models/LightGBM", "models/timesFM")

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".github",
    ".agents",
    ".opencode",
}
EXCLUDED_RELATIVE_PREFIXES = (
    "TimeMixer/outputs_v2",
    "SGDFNet/docs",
    "SGDFNet/scripts",
    "RT916_SpikeFusionNet/docs",
    "fusion/docs",
    "scripts/sync/archive",
)
EXCLUDED_SUFFIXES = (".pyc", ".pyo")
FORBIDDEN_RELEASE_PREFIXES = (
    "outputs",
    "data",
    "deliverables",
    "fixtures",
    "_archive",
    "dist",
    "build",
    "scripts/tests",
    "scripts/experiments",
    "scripts/crawler",
    "ExtremPriceClf",
)
FORBIDDEN_BASENAMES = {
    ".env",
    "config.json",
    "db_config.json",
}


def _norm(path: Path | str) -> str:
    # Release paths are already relative to project_root. Do not strip leading
    # dots: '.env.example' is a legitimate template file name.
    return Path(path).as_posix()


def _is_excluded(rel: Path) -> bool:
    rel_text = _norm(rel)
    if any(part in EXCLUDED_DIR_NAMES for part in rel.parts):
        return True
    if rel.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if any(
        rel_text == prefix or rel_text.startswith(prefix + "/")
        for prefix in EXCLUDED_RELATIVE_PREFIXES
    ):
        return True
    return False


def _is_forbidden_release_path(rel: Path) -> bool:
    rel_text = _norm(rel)
    if rel.name in FORBIDDEN_BASENAMES:
        # models/timesFM/config.json is a checkpoint config, not a secret.
        if rel_text == "models/timesFM/config.json":
            return False
        return True
    return any(
        rel_text == prefix or rel_text.startswith(prefix + "/")
        for prefix in FORBIDDEN_RELEASE_PREFIXES
    )


def _iter_tree(root: Path, relative_root: str) -> Iterable[Path]:
    base = root / relative_root
    if not base.exists():
        return
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if _is_excluded(rel) or _is_forbidden_release_path(rel):
            continue
        yield rel


def iter_release_files(project_root: Path = PROJECT_ROOT) -> list[Path]:
    project_root = Path(project_root)
    selected: set[Path] = set()

    for rel_text in CORE_FILES:
        rel = Path(rel_text)
        path = project_root / rel
        if path.is_file() and not _is_forbidden_release_path(rel):
            selected.add(rel)

    for rel_text in (*CORE_DIRS, *SCRIPT_DIRS, *MODEL_DIRS):
        selected.update(_iter_tree(project_root, rel_text) or ())

    return sorted(selected, key=lambda p: p.as_posix())


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(project_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    return value or None


def _role_for(rel: Path) -> str:
    text = _norm(rel)
    if text.startswith("models/"):
        return "static_model_asset"
    if text.startswith("docs/") or text == "README.md":
        return "operator_documentation"
    if text.startswith("scripts/server/"):
        return "production_server_tool"
    if text.startswith("scripts/sync/"):
        return "production_sync_tool"
    if text == ".env.example":
        return "config_template"
    return "application"


def build_release_manifest(
    project_root: Path = PROJECT_ROOT,
    *,
    hash_files: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    files = iter_release_files(project_root)
    entries: list[dict[str, Any]] = []

    for rel in files:
        if _is_forbidden_release_path(rel):
            raise RuntimeError(f"forbidden release path selected: {rel}")
        path = project_root / rel
        item = {
            "path": _norm(rel),
            "size_bytes": int(path.stat().st_size),
            "role": _role_for(rel),
        }
        if hash_files:
            item["sha256"] = _sha256(path)
        entries.append(item)

    required_model_files = {
        "models/LightGBM/best_model_日前电价.pkl",
        "models/timesFM/model.safetensors",
        "models/timesFM/config.json",
    }
    present = {item["path"] for item in entries}
    missing_models = sorted(required_model_files - present)
    if missing_models:
        raise FileNotFoundError(
            f"required model assets missing from release plan: {missing_models}"
        )

    return {
        "contract": RELEASE_CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(project_root),
        "source_git_commit": _git_head(project_root),
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
        "includes_models": True,
        "mutable_state_included": False,
        "crawler_included": False,
        "files": entries,
    }


def materialize_release(
    destination: Path,
    manifest: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    project_root = Path(project_root).resolve()
    destination = Path(destination).resolve()

    if destination == project_root:
        raise ValueError("release destination must not be the project root")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"release destination is not empty: {destination}. "
            "Use a new candidate directory; do not overwrite an existing release."
        )

    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        rel = Path(item["path"])
        src = project_root / rel
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    manifest_out = dict(manifest)
    manifest_out["materialized_root"] = str(destination)
    manifest_path = destination / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _print_summary(manifest: dict[str, Any], *, applied: bool, destination: Path | None) -> None:
    gib = manifest["total_bytes"] / (1024 ** 3)
    print(f"RELEASE_CONTRACT={manifest['contract']}")
    print(f"FILES={manifest['file_count']}")
    print(f"TOTAL_GIB={gib:.3f}")
    print(f"MODE={'APPLY' if applied else 'DRY_RUN'}")
    if destination is not None:
        print(f"DESTINATION={destination}")
    print("MUTABLE_STATE_INCLUDED=false")
    print("CRAWLER_INCLUDED=false")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a manifest-driven minimal formal96 predictor release."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Source checkout. Defaults to the current project root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Candidate release directory. Required with --apply.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Materialize the candidate release. Default is dry-run only.",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Skip SHA256 calculation for a fast local dry-run. Not allowed with --apply.",
    )
    args = parser.parse_args(argv)

    if args.apply and args.output_dir is None:
        parser.error("--output-dir is required with --apply")
    if args.apply and args.skip_hash:
        parser.error("--skip-hash is only allowed for dry-run")

    manifest = build_release_manifest(
        args.project_root,
        hash_files=not args.skip_hash,
    )

    manifest_path = None
    if args.apply:
        manifest_path = materialize_release(
            args.output_dir,
            manifest,
            project_root=args.project_root,
        )

    _print_summary(
        manifest,
        applied=args.apply,
        destination=args.output_dir,
    )
    if manifest_path is not None:
        print(f"MANIFEST={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
