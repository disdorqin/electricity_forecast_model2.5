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
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    if normalized_hit:
        df = pd.read_parquet(layout.normalized_input)
    else:
        df = normalize_classifier_input(source, spec)
        layout.ensure()
        tmp = layout.normalized_input.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(layout.normalized_input)

    stage1_hit = layout.stage1_features.exists() and layout.stage1_features.stat().st_size > 0
    stage2_hit = layout.stage2_features.exists() and layout.stage2_features.stat().st_size > 0
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
