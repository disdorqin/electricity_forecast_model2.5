"""Resolution-generic range runner for the extreme-price classifier.

This is the first reusable execution layer around the existing cascade.  It
does not change the cascade model, feature definitions, thresholds, or
rolling semantics.  It only makes the input contract explicit, normalizes
24/96 inputs once, and gives every replay a stable shared cache namespace.

The current daily entry point remains the compatibility/reference path.  The
experiment entry point uses this module first; production will be switched
only after parity tests pass.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.classifier_cache import (
    ClassifierCacheSpec,
    build_cache_manifest,
    cache_is_valid,
    classifier_cache_layout,
    write_cache_manifest,
)
from utils.data_loader import load_table
from utils.resolution import HOURLY, resolve_resolution

from .cascade_daily import Stage2Config, build_stage2_features, run_rolling_daily_cascade
from .extreme_price_radar.features import FeatureEngineer


# The legacy classifier expects these internal names.  A caller for another
# market/link can supply additional source->canonical aliases without touching
# the cascade implementation.
DEFAULT_COLUMN_ALIASES: dict[str, str] = {
    "timestamp": "时刻",
    "datetime": "时刻",
    "ds": "时刻",
    "dayahead_price": "日前电价",
    "realtime_price": "实时电价",
    "da_price": "日前电价",
    "rt_price": "实时电价",
}


REQUIRED_CANONICAL_COLUMNS = (
    "时刻",
    "日前电价",
    "实时电价",
    "地方电厂总加实际值",
    "核电总加实际值",
    "自备机组总加实际值",
    "试验机组总加实际值",
    "直调负荷实际值",
    "联络线受电负荷实际值",
    "新能源总加实际值",
    "竞价空间实际值",
    "地方电厂总加预测值",
    "核电总加预测值",
    "自备机组总加预测值",
    "试验机组总加预测值",
    "直调负荷预测值",
    "联络线受电负荷预测值",
    "风电总加预测值",
    "光伏总加预测值",
    "新能源总加预测值",
    "竞价空间预测值",
)


@dataclass(frozen=True)
class ClassifierRangeSpec:
    """Public, reusable classifier execution contract."""

    start_date: str
    end_date: str
    resolution: str = "hourly"
    task: str = "realtime"
    target_name: str = "实时电价"
    source_time_col: str = "时刻"
    price_threshold: float = -50.0
    train_start: str = "2022-01-01"
    stage2_train_start: str = "2024-01-01"
    oof_cutoff: str = "2024-12-31"
    feature_type: str = "预测值"
    model_name: str = "lightgbm"
    min_precision: float = 0.7
    dynamic_gray_enabled: bool = True
    column_aliases: Mapping[str, str] = field(default_factory=dict)

    def cache_spec(self) -> ClassifierCacheSpec:
        return ClassifierCacheSpec(
            resolution=self.resolution,
            task=self.task,
            target_name=self.target_name,
            price_threshold=self.price_threshold,
            train_start=self.train_start,
            stage2_train_start=self.stage2_train_start,
            oof_cutoff=self.oof_cutoff,
            model_name=self.model_name,
            feature_type=self.feature_type,
            dynamic_gray_enabled=self.dynamic_gray_enabled,
            min_precision=self.min_precision,
        )


def _canonicalize_columns(df: pd.DataFrame, spec: ClassifierRangeSpec) -> pd.DataFrame:
    aliases = dict(DEFAULT_COLUMN_ALIASES)
    aliases.update(dict(spec.column_aliases))
    rename = {source: target for source, target in aliases.items() if source in df.columns and target not in df.columns}
    if spec.source_time_col != "时刻" and spec.source_time_col in df.columns and "时刻" not in df.columns:
        rename[spec.source_time_col] = "时刻"
    if spec.target_name not in {"日前电价", "实时电价"} and spec.target_name in df.columns:
        # The cascade uses the canonical target names for feature construction.
        canonical_target = "日前电价" if spec.task == "dayahead" else "实时电价"
        if canonical_target not in df.columns:
            rename[spec.target_name] = canonical_target
    return df.rename(columns=rename)


def _derive_contract_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive stable physical columns when a source omits redundant fields."""
    df = df.copy()
    pairs = (
        ("预测值", "风电总加预测值", "光伏总加预测值", "新能源总加预测值"),
        ("实际值", "风电总加实际值", "光伏总加实际值", "新能源总加实际值"),
    )
    for _, wind, solar, renewable in pairs:
        if renewable not in df.columns and wind in df.columns and solar in df.columns:
            df[renewable] = pd.to_numeric(df[wind], errors="coerce") + pd.to_numeric(df[solar], errors="coerce")

    space_pairs = (
        ("预测值", "直调负荷预测值", "风电总加预测值", "光伏总加预测值", "联络线受电负荷预测值", "竞价空间预测值"),
        ("实际值", "直调负荷实际值", "风电总加实际值", "光伏总加实际值", "联络线受电负荷实际值", "竞价空间实际值"),
    )
    for _, load, wind, solar, interconnect, space in space_pairs:
        if space not in df.columns and all(col in df.columns for col in (load, wind, solar, interconnect)):
            df[space] = (
                pd.to_numeric(df[load], errors="coerce")
                - pd.to_numeric(df[wind], errors="coerce")
                - pd.to_numeric(df[solar], errors="coerce")
                - pd.to_numeric(df[interconnect], errors="coerce")
            )
    return df


def normalize_classifier_input(path: Path, spec: ClassifierRangeSpec) -> pd.DataFrame:
    """Load and normalize one 24/96 source without changing legacy semantics."""
    df = _derive_contract_columns(_canonicalize_columns(load_table(path), spec))
    if "时刻" not in df.columns:
        raise ValueError("classifier input is missing time column '时刻'")
    df["时刻"] = pd.to_datetime(df["时刻"], errors="coerce")
    df = df.dropna(subset=["时刻"]).sort_values("时刻").reset_index(drop=True)

    target_name = "日前电价" if spec.task == "dayahead" else "实时电价"
    if target_name not in df.columns:
        raise ValueError(f"classifier input is missing target column: {target_name}")

    # Match run_daily.py: retain time, target, and numeric features only.
    keep = ["时刻"]
    for col in df.columns:
        if col != "时刻" and pd.api.types.is_numeric_dtype(df[col]):
            keep.append(col)
    df = df[keep].copy()

    res = resolve_resolution(spec.resolution)
    if res.label == "15min":
        numeric_cols = [c for c in df.columns if c != "时刻" and pd.api.types.is_numeric_dtype(df[c])]
        aggregations = {c: "mean" for c in numeric_cols}
        df = (
            df.groupby(df["时刻"].dt.floor("h"), as_index=False)
            .agg({"时刻": "first", **aggregations})
            .sort_values("时刻")
            .reset_index(drop=True)
        )
        df["时刻"] = df["时刻"].dt.floor("h")

    missing = [col for col in REQUIRED_CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "classifier input does not satisfy the canonical feature contract; "
            f"missing={missing}. Supply source->canonical aliases or a prepared table."
        )
    return df


def _cache_specs_compatible(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Compare classifier semantics while allowing cache-schema migration."""
    left = dict(candidate or {})
    right = dict(current or {})
    left.pop("cache_schema", None)
    right.pop("cache_schema", None)
    return left == right


def _semantic_p1_prefix_match(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    p1_df: pd.DataFrame,
    spec: ClassifierRangeSpec,
) -> tuple[bool, dict[str, Any]]:
    """Prove that an existing p1 cache can be reused with the current source.

    Stage-1 probabilities depend on forecast-side features at the predicted
    timestamps and on historical target labels used to train earlier daily
    models. The current decision-day target label may be masked by design, so
    label equality is required only through the day *before* the cached tail.
    """
    if p1_df.empty or "时刻" not in p1_df.columns or "p1_prob_OOF" not in p1_df.columns:
        return False, {"reason": "p1 cache missing required columns"}
    p1 = p1_df.copy()
    p1["时刻"] = pd.to_datetime(p1["时刻"], errors="coerce")
    if p1["时刻"].isna().any() or p1["时刻"].duplicated().any() or p1["p1_prob_OOF"].isna().any():
        return False, {"reason": "p1 cache contains invalid/duplicate timestamps or NaN probabilities"}
    cached_until = p1["时刻"].max()

    def prep(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["时刻"] = pd.to_datetime(out["时刻"], errors="coerce")
        return out.dropna(subset=["时刻"]).sort_values("时刻").set_index("时刻")

    ref = prep(reference_df)
    cur = prep(current_df)
    ref_prefix = ref.loc[ref.index <= cached_until]
    cur_prefix = cur.loc[cur.index <= cached_until]
    if not ref_prefix.index.equals(cur_prefix.index):
        return False, {
            "reason": "timestamp prefix changed",
            "cached_until": str(cached_until),
            "reference_rows": len(ref_prefix),
            "current_rows": len(cur_prefix),
        }

    forecast_cols = sorted(
        c for c in REQUIRED_CANONICAL_COLUMNS
        if c.endswith("预测值") and c in ref_prefix.columns and c in cur_prefix.columns
    )
    if not forecast_cols:
        return False, {"reason": "no forecast columns available for semantic validation"}
    for col in forecast_cols:
        a = pd.to_numeric(ref_prefix[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(cur_prefix[col], errors="coerce").to_numpy(dtype=float)
        mismatch = int((~np.isclose(a, b, equal_nan=True, rtol=1e-10, atol=1e-8)).sum())
        if mismatch:
            return False, {
                "reason": f"forecast prefix changed: {col}",
                "cached_until": str(cached_until),
                "mismatch_rows": mismatch,
            }

    target_col = "日前电价" if spec.task == "dayahead" else "实时电价"
    # For cached prediction day D, run_rolling_daily_cascade trains Stage1
    # through D-2 23:00 (current_infer_start - 25h). Labels after that boundary
    # never contributed to the cached p1 probabilities and may legitimately
    # differ between consecutive as-of views as the decision day advances.
    label_until = cached_until.normalize() - pd.Timedelta(hours=25)
    ref_labels = ref.loc[ref.index <= label_until]
    cur_labels = cur.loc[cur.index <= label_until]
    if not ref_labels.index.equals(cur_labels.index):
        return False, {"reason": "historical label timestamp prefix changed", "label_until": str(label_until)}
    a = pd.to_numeric(ref_labels[target_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(cur_labels[target_col], errors="coerce").to_numpy(dtype=float)
    mismatch = int((~np.isclose(a, b, equal_nan=True, rtol=1e-10, atol=1e-8)).sum())
    if mismatch:
        return False, {
            "reason": "historical target labels changed",
            "label_until": str(label_until),
            "mismatch_rows": mismatch,
        }

    return True, {
        "reason": "semantic historical prefix matches",
        "cached_until": str(cached_until),
        "label_validated_until": str(label_until),
        "forecast_columns": forecast_cols,
    }


def _iter_legacy_p1_candidates(project_root: Path, task: str):
    """Yield cache files from bounded classifier roots, skipping broken dirs."""
    roots = [
        project_root / "outputs" / "cache" / "classifier" / task,
        project_root / "outputs" / "96" / "feature_store" / "cache" / "classifier" / task,
        project_root / "outputs" / "experiments" / "04_pipeline_audits",
    ]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _exc: None):
            # Never follow symlink/junction directory trees during migration.
            dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
            if "p1_cache.parquet" not in filenames:
                continue
            candidate = Path(dirpath) / "p1_cache.parquet"
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def _adopt_legacy_p1_cache(
    *,
    project_root: Path,
    current_df: pd.DataFrame,
    spec: ClassifierRangeSpec,
    layout: Any,
) -> dict[str, Any] | None:
    """Find the newest compatible legacy p1 cache and migrate it safely."""
    current_spec = spec.cache_spec().canonical()
    best: tuple[pd.Timestamp, Path, dict[str, Any]] | None = None
    for candidate in _iter_legacy_p1_candidates(project_root, spec.task):
        if candidate.resolve() == layout.p1_cache.resolve():
            continue
        manifest_path = candidate.parent / "manifest.json"
        normalized_path = candidate.parent / "normalized_input.parquet"
        if not manifest_path.exists() or not normalized_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not _cache_specs_compatible(manifest.get("spec", {}), current_spec):
                continue
            p1 = pd.read_parquet(candidate)
            reference = pd.read_parquet(normalized_path)
            ok, detail = _semantic_p1_prefix_match(reference, current_df, p1, spec)
            if not ok:
                continue
            cached_until = pd.to_datetime(p1["时刻"], errors="coerce").max()
            if pd.isna(cached_until):
                continue
            if best is None or cached_until > best[0]:
                best = (cached_until, candidate, detail)
        except Exception:
            continue
    if best is None:
        return None
    layout.ensure()
    shutil.copy2(best[1], layout.p1_cache)
    return {
        "status": "adopted",
        "source_p1": str(best[1]),
        "cached_until": str(best[0]),
        **best[2],
    }


def prepare_classifier_cache(
    *,
    project_root: Path,
    source: Path,
    spec: ClassifierRangeSpec,
    feature_store_root: Path | None = None,
) -> tuple[pd.DataFrame, Any, bool]:
    """Build or reuse normalized input and its shared p1 cache namespace."""
    cache_spec = spec.cache_spec()
    layout = classifier_cache_layout(
        project_root=project_root,
        source=source,
        spec=cache_spec,
        feature_store_root=feature_store_root,
    )
    normalized_hit = cache_is_valid(
        layout, source=source, spec=cache_spec, required_artifacts=("normalized_input",)
    )
    p1_reuse: dict[str, Any] = {"status": "not_present"}
    if normalized_hit:
        df = pd.read_parquet(layout.normalized_input)
        if layout.p1_cache.exists():
            p1_reuse = {"status": "same_source_reuse", "p1_cache": str(layout.p1_cache)}
    else:
        # Normalize the current source first, then prove whether an existing p1
        # cache still has the same semantic historical prefix. Source mtime or
        # file size alone is deliberately not a reason to discard expensive p1.
        df = normalize_classifier_input(source, spec)
        previous_normalized = None
        if layout.normalized_input.exists() and layout.normalized_input.stat().st_size > 0:
            try:
                previous_normalized = pd.read_parquet(layout.normalized_input)
            except Exception:
                previous_normalized = None

        if layout.p1_cache.exists():
            if previous_normalized is None:
                layout.p1_cache.unlink()
                p1_reuse = {"status": "invalidated", "reason": "missing prior normalized input"}
            else:
                try:
                    p1_existing = pd.read_parquet(layout.p1_cache)
                    safe, detail = _semantic_p1_prefix_match(previous_normalized, df, p1_existing, spec)
                except Exception as exc:
                    safe, detail = False, {"reason": f"validation error: {exc}"}
                if safe:
                    p1_reuse = {"status": "semantic_prefix_reuse", **detail}
                else:
                    layout.p1_cache.unlink()
                    p1_reuse = {"status": "invalidated", **detail}

        layout.ensure()
        tmp = layout.normalized_input.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(layout.normalized_input)

    # One-time migration path for v4/source-fingerprint caches. Adoption is
    # allowed only after semantic prefix equality is proven against the current
    # normalized source.
    if not layout.p1_cache.exists():
        adopted = _adopt_legacy_p1_cache(
            project_root=project_root,
            current_df=df,
            spec=spec,
            layout=layout,
        )
        if adopted is not None:
            p1_reuse = adopted

    # Feature matrices are cheap compared with rolling p1. They may only be
    # reused when the concrete source fingerprint is unchanged; otherwise they
    # are rebuilt from the current as-of source.
    stage1_hit = normalized_hit and layout.stage1_features.exists() and layout.stage1_features.stat().st_size > 0
    stage2_hit = normalized_hit and layout.stage2_features.exists() and layout.stage2_features.stat().st_size > 0
    if not stage1_hit:
        stage1 = FeatureEngineer(time_col="时刻").process(df)
        tmp = layout.stage1_features.with_suffix(".parquet.tmp")
        stage1.to_parquet(tmp, index=False)
        tmp.replace(layout.stage1_features)
    else:
        stage1 = pd.read_parquet(layout.stage1_features)
    if not stage2_hit:
        stage2_start = pd.Timestamp(spec.stage2_train_start)
        # The legacy OOF builder first filters to stage2_train_start and then
        # FeatureEngineer drops its seven-day hourly warm-up.  Reproduce that
        # boundary before materializing stage-2 features; using the full-data
        # lag would otherwise give the first training rows a different value.
        warmup = pd.Timedelta(hours=HOURLY.slots_per_day * 7)
        effective_stage2_start = stage2_start + warmup
        stage2_source = df[df["时刻"] >= effective_stage2_start].copy()
        stage2 = build_stage2_features(stage2_source, spec.feature_type)
        tmp = layout.stage2_features.with_suffix(".parquet.tmp")
        stage2.to_parquet(tmp, index=False)
        tmp.replace(layout.stage2_features)

    stage1 = pd.read_parquet(layout.stage1_features)
    stage2 = pd.read_parquet(layout.stage2_features)
    manifest = build_cache_manifest(
        source=source,
        spec=cache_spec,
        layout=layout,
        artifacts={
            "normalized_input": str(layout.normalized_input),
            "stage1_features": str(layout.stage1_features),
            "stage2_features": str(layout.stage2_features),
        },
        extra={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
            "stage1_rows": len(stage1),
            "stage2_rows": len(stage2),
            "p1_cache_reuse": p1_reuse,
        },
    )
    write_cache_manifest(layout, manifest)
    return df, layout, bool(normalized_hit and stage1_hit and stage2_hit)


def run_classifier_range(
    *,
    project_root: Path,
    source: Path,
    spec: ClassifierRangeSpec,
    output_dir: Path,
    feature_store_root: Path | None = None,
    reuse_cache: bool = True,
) -> dict[str, Any]:
    """Run one range replay and write a reusable classifier ledger.

    This function is intentionally independent of model prediction and fusion.
    It consumes a frozen classifier input and can therefore be called by both
    the 24-point and 96-point chains.
    """
    if not source.exists():
        raise FileNotFoundError(source)
    if not reuse_cache:
        # A different source fingerprint is the safest invalidation mechanism;
        # callers can also pass a temporary feature_store_root for isolation.
        feature_store_root = (feature_store_root or project_root / "outputs" / "experiments") / "no_reuse"

    df, layout, cache_hit = prepare_classifier_cache(
        project_root=project_root,
        source=source,
        spec=spec,
        feature_store_root=feature_store_root,
    )
    stage1_features = pd.read_parquet(layout.stage1_features)
    stage2_features = pd.read_parquet(layout.stage2_features)

    cfg = Stage2Config()
    cfg.feature_type = spec.feature_type
    cfg.model_name = spec.model_name
    cfg.dynamic_gray_enabled = spec.dynamic_gray_enabled

    started = datetime.now(timezone.utc)
    results = run_rolling_daily_cascade(
        df=df,
        target_name="日前电价" if spec.task == "dayahead" else "实时电价",
        price_threshold=spec.price_threshold,
        test_time_range=[f"{spec.start_date} 00:00:00", f"{spec.end_date} 23:00:00"],
        train_start=spec.train_start,
        stage2_train_start=spec.stage2_train_start,
        stage2_config=cfg,
        p1_cache_path=str(layout.p1_cache),
        oof_cutoff=spec.oof_cutoff,
        min_precision=spec.min_precision,
        stage1_feature_cache=stage1_features,
        stage2_feature_cache=stage2_features,
    )
    if results is None or results.empty:
        raise RuntimeError("classifier range produced no rows")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "classifier_ledger.parquet"
    result_tmp = result_path.with_suffix(".parquet.tmp")
    results.to_parquet(result_tmp, index=False)
    result_tmp.replace(result_path)
    csv_path = output_dir / "classifier_ledger.csv"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    results.to_csv(csv_tmp, index=False, encoding="utf-8-sig")
    csv_tmp.replace(csv_path)

    manifest = build_cache_manifest(
        source=source,
        spec=spec.cache_spec(),
        layout=layout,
        artifacts={
            "normalized_input": str(layout.normalized_input),
            "stage1_features": str(layout.stage1_features),
            "stage2_features": str(layout.stage2_features),
            "p1_cache": str(layout.p1_cache),
            "result_ledger": str(result_path),
        },
        extra={
            "range": {"start_date": spec.start_date, "end_date": spec.end_date},
            "rows": len(results),
            "cache_hit": cache_hit,
            "started_at": started.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    (output_dir / "classifier_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {"status": "complete", "rows": len(results), "result_path": str(result_path), "cache_dir": str(layout.root), "cache_hit": cache_hit}
