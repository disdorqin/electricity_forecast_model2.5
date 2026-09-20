"""Reusable, resolution-isolated cache layout for the extreme-price classifier.

The classifier is hourly internally, but it can be fed by either the 24-point
or the 96-point chain. Cache identity therefore includes the *source chain*
resolution and every parameter that can change the rolling result. Production
cache files live under ``outputs/{24,96}/cache/classifier`` and are never mixed
with daily run artifacts or legacy roots.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from utils.resolution import resolve_resolution


CLASSIFIER_CACHE_SCHEMA = "classifier_cache_v5_incremental_prefix"


def _source_fingerprint(source: Path) -> dict[str, Any]:
    """Return a cheap but deterministic source identity for cache validation."""
    source = source.resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


@dataclass(frozen=True)
class ClassifierCacheSpec:
    """All semantic inputs that can affect a classifier replay."""

    resolution: str
    task: str
    target_name: str
    price_threshold: float
    train_start: str
    stage2_train_start: str
    oof_cutoff: str
    model_name: str = "lightgbm"
    feature_type: str = "预测值"
    dynamic_gray_enabled: bool = True
    min_precision: float = 0.7
    cache_schema: str = CLASSIFIER_CACHE_SCHEMA

    def canonical(self) -> dict[str, Any]:
        res = resolve_resolution(self.resolution)
        data = asdict(self)
        data["resolution"] = res.label
        data["slots_per_day"] = res.slots_per_day
        return data


@dataclass(frozen=True)
class ClassifierCacheLayout:
    """Stable paths for one classifier cache namespace."""

    root: Path
    key: str

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def normalized_input(self) -> Path:
        return self.root / "normalized_input.parquet"

    @property
    def p1_cache(self) -> Path:
        return self.root / "p1_cache.parquet"

    @property
    def stage1_features(self) -> Path:
        return self.root / "stage1_features.parquet"

    @property
    def stage2_features(self) -> Path:
        return self.root / "stage2_features.parquet"

    @property
    def result_ledger(self) -> Path:
        return self.root / "classifier_ledger.parquet"

    def ensure(self) -> "ClassifierCacheLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        return self


def classifier_cache_layout(
    *,
    project_root: Path,
    source: Path,
    spec: ClassifierCacheSpec,
    feature_store_root: Path | None = None,
) -> ClassifierCacheLayout:
    """Resolve a stable cache directory isolated by resolution/task/config.

    Source file size/mtime intentionally do *not* participate in the directory
    key. Daily as-of files are expected to change. Their semantic historical
    prefix is validated before p1 reuse by the range runner; cheap normalized
    and feature artifacts are refreshed whenever the concrete source changes.
    """
    canonical = {
        "spec": spec.canonical(),
    }
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    res = resolve_resolution(spec.resolution)
    base = feature_store_root or (project_root / "outputs" / str(res.slots_per_day))
    root = Path(base) / "cache" / "classifier" / spec.task / key
    return ClassifierCacheLayout(root=root, key=key)


def read_cache_manifest(layout: ClassifierCacheLayout) -> dict[str, Any] | None:
    if not layout.manifest.exists():
        return None
    try:
        return json.loads(layout.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_cache_manifest(layout: ClassifierCacheLayout, manifest: dict[str, Any]) -> None:
    """Atomically write a cache manifest so interrupted runs cannot validate."""
    layout.ensure()
    tmp = layout.manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, layout.manifest)


def cache_is_valid(
    layout: ClassifierCacheLayout,
    *,
    source: Path,
    spec: ClassifierCacheSpec,
    required_artifacts: tuple[str, ...] = ("normalized_input",),
) -> bool:
    """Validate semantic identity and required files before reuse."""
    manifest = read_cache_manifest(layout)
    if not manifest:
        return False
    if manifest.get("cache_schema") != CLASSIFIER_CACHE_SCHEMA:
        return False
    if manifest.get("spec") != spec.canonical():
        return False
    if manifest.get("source") != _source_fingerprint(source):
        return False
    for name in required_artifacts:
        path = getattr(layout, name, None)
        if path is None or not path.exists() or path.stat().st_size == 0:
            return False
    return True


def build_cache_manifest(
    *,
    source: Path,
    spec: ClassifierCacheSpec,
    layout: ClassifierCacheLayout,
    artifacts: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "cache_schema": CLASSIFIER_CACHE_SCHEMA,
        "cache_key": layout.key,
        "source": _source_fingerprint(source),
        "spec": spec.canonical(),
        "artifacts": artifacts or {},
    }
    if extra:
        manifest.update(extra)
    return manifest
