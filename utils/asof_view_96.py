"""Build one shared leak-safe 96-point model view for a target day.

The source is the full canonical 96-point model store. The returned parquet is
created once per run and is shared by every model/stage in that run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS


RUNTIME_ROOT = Path("outputs") / "96" / "runtime"


def cleanup_stale_asof_96(max_age_hours: float = 6.0) -> int:
    """Remove orphaned scratch views old enough to be safely considered stale.

    Normal runs delete their scratch immediately. This TTL sweep covers hard
    kills / host restarts without deleting another concurrently active run.
    """
    if not RUNTIME_ROOT.exists():
        return 0
    cutoff = time.time() - max(0.0, float(max_age_hours)) * 3600.0
    removed = 0
    for candidate in RUNTIME_ROOT.glob("asof_96_*.parquet"):
        try:
            if candidate.stat().st_mtime < cutoff:
                candidate.unlink(missing_ok=True)
                candidate.with_suffix(candidate.suffix + ".tmp").unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


def transient_asof_path_96(
    target_day: str,
    runtime_root: str | Path | None = None,
) -> Path:
    """Return one scratch path; formal callers may inject an attempt-owned root.

    ``runtime_root=None`` preserves the legacy process-local layout. Formal96
    passes ``<attempt>/asof`` so the masked view is reclaimed with the whole
    invocation sandbox instead of becoming an orphan beside model scratch.
    """
    safe_day = str(target_day).replace("-", "")
    if runtime_root is None:
        cleanup_stale_asof_96()
        return RUNTIME_ROOT / f"asof_96_{safe_day}_{os.getpid()}.parquet"
    root = Path(runtime_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / "input.parquet"


def cleanup_transient_asof_96(path: str | Path | None) -> None:
    """Best-effort cleanup for the shared masked scratch view."""
    if not path:
        return
    candidate = Path(path)
    try:
        candidate.unlink(missing_ok=True)
        tmp = candidate.with_suffix(candidate.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
    except OSError:
        # Cleanup must not turn a completed forecast into a failed delivery.
        pass


FORECAST_COLUMNS = [
    "直调负荷预测值",
    "地方电厂总加预测值",
    "联络线受电负荷预测值",
    "风电总加预测值",
    "光伏总加预测值",
    "核电总加预测值",
    "自备机组总加预测值",
    "试验机组总加预测值",
    "竞价空间预测值",
    "新能源总加预测值",
]

# Dynamic-v1 serving contract.  These are deliberately kept next to the
# legacy as-of helper so there is one 96-point information-boundary module;
# ``build_asof_view_96`` remains available to legacy/replay callers.
DYNAMIC_PROTOCOL = "formal96_dynamic_snapshot_v1"
HISTORICAL_PROXY_PROTOCOL = "formal96_historical_proxy_v1"
STORED_LIVE_SNAPSHOT_REPLAY = "STORED_LIVE_SNAPSHOT_REPLAY"
HISTORICAL_PROXY_V1 = "HISTORICAL_PROXY_V1"
LIVE_DYNAMIC = "LIVE_DYNAMIC"
SNAPSHOT_PROTOCOLS = {DYNAMIC_PROTOCOL, HISTORICAL_PROXY_PROTOCOL}
PRIMITIVE_ACTUAL_COLUMNS = [
    "直调负荷实际值", "地方电厂总加实际值", "联络线受电负荷实际值",
    "风电总加实际值", "光伏总加实际值", "核电总加实际值",
    "自备机组总加实际值", "试验机组总加实际值",
]
PRIMITIVE_FORECAST_COLUMNS = [
    "直调负荷预测值", "地方电厂总加预测值", "联络线受电负荷预测值",
    "风电总加预测值", "光伏总加预测值", "核电总加预测值",
    "自备机组总加预测值", "试验机组总加预测值",
]
ACTUAL_TO_FORECAST = dict(zip(PRIMITIVE_ACTUAL_COLUMNS, PRIMITIVE_FORECAST_COLUMNS))
RAW_FORECAST_COLUMNS = {
    "直调负荷预测值": "直调负荷预测",
    "地方电厂总加预测值": "地方电厂出力预测",
    "联络线受电负荷预测值": "外电预测",
    "风电总加预测值": "风电预测",
    "光伏总加预测值": "光伏预测",
    "核电总加预测值": "核电预测",
    "自备机组总加预测值": "自备电厂预测",
    "试验机组总加预测值": "试验机组预测",
}
RAW_ACTUAL_COLUMNS = {
    "直调负荷实际值": "直调负荷实际",
    "地方电厂总加实际值": "地方电厂出力实际",
    "联络线受电负荷实际值": "外电实际",
    "风电总加实际值": "风电实际",
    "光伏总加实际值": "光伏实际",
    "核电总加实际值": "核电实际",
    "自备机组总加实际值": "自备电厂实际",
    "试验机组总加实际值": "试验机组实际",
}
RAW_TMP_COLUMNS = {
    "直调负荷实际值": "直调负荷临时实际",
    "地方电厂总加实际值": "地方电厂出力临时实际",
    "联络线受电负荷实际值": "外电临时实际",
    "风电总加实际值": "风电临时实际",
    "光伏总加实际值": "光伏临时实际",
    "核电总加实际值": "核电临时实际",
    "自备机组总加实际值": "自备电厂临时实际",
    "试验机组总加实际值": "试验机组临时实际",
}


def _read_96_table(path: str | Path) -> pd.DataFrame:
    """Read a 96-point table without changing the caller's source file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        out = pd.read_parquet(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        out = pd.read_excel(path)
    else:
        out = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "market_date" not in out.columns:
        raise ValueError(f"MISSING_CRITICAL_SOURCE source={path} field=market_date rows=0 expected=96")
    out["market_date"] = pd.to_datetime(out["market_date"], errors="raise").dt.normalize()
    if "period_no" not in out.columns:
        if "时段" in out.columns:
            def _period(v):
                text = str(v).strip()
                if text in {"24:00", "00:00"}:
                    return 96
                hh, mm = text.split(":", 1)
                return int(hh) * 4 + int(mm) // 15
            out["period_no"] = out["时段"].map(_period)
        elif "时刻" in out.columns:
            ts = pd.to_datetime(out["时刻"], errors="raise")
            out["period_no"] = ((ts.dt.hour * 60 + ts.dt.minute) // 15).replace(0, 96)
        else:
            raise ValueError(f"MISSING_CRITICAL_SOURCE source={path} field=period_no rows={len(out)} expected=96")
    out["period_no"] = pd.to_numeric(out["period_no"], errors="raise").astype(int)
    return out.sort_values(["market_date", "period_no"]).reset_index(drop=True)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_raw_view(authority: pd.DataFrame, model_store: pd.DataFrame, days: list[pd.Timestamp]) -> pd.DataFrame:
    """Convert authoritative/raw rows to the small stable snapshot schema."""
    frames = []
    for day in days:
        part = authority[authority["market_date"].eq(day)].copy()
        fallback = model_store[model_store["market_date"].eq(day)].copy()
        if part.empty:
            part = fallback.copy()
        if part.empty:
            continue
        if "period_no" not in part:
            raise ValueError("MISSING_CRITICAL_SOURCE field=period_no")
        if part["period_no"].duplicated().any():
            raise ValueError("SNAPSHOT_DUPLICATE_GRID_ROWS")
        out = part[["market_date", "period_no"]].copy()
        for canonical, raw in RAW_FORECAST_COLUMNS.items():
            if raw in part.columns:
                vals = part.set_index("period_no")[raw]
            elif canonical in part.columns:
                vals = part.set_index("period_no")[canonical]
            else:
                vals = pd.Series(dtype=float)
            if canonical not in out:
                out[canonical] = out["period_no"].map(vals)
            if canonical in fallback.columns:
                fb = fallback.set_index("period_no")[canonical]
                out[canonical] = pd.to_numeric(out[canonical], errors="coerce").fillna(out["period_no"].map(fb))
        for canonical, raw in RAW_ACTUAL_COLUMNS.items():
            if raw in part.columns:
                vals = part.set_index("period_no")[raw]
            elif canonical in part.columns:
                vals = part.set_index("period_no")[canonical]
            else:
                vals = pd.Series(dtype=float)
            out[canonical] = out["period_no"].map(vals)
        for canonical, raw in RAW_TMP_COLUMNS.items():
            if raw in part.columns:
                out[f"{canonical}__tmp"] = out["period_no"].map(part.set_index("period_no")[raw])
            else:
                out[f"{canonical}__tmp"] = np.nan
        for canonical, raw in (("日前电价", "日前出清价格"), ("实时电价", "实时出清价格")):
            if raw in part.columns:
                out[canonical] = out["period_no"].map(part.set_index("period_no")[raw])
            elif canonical in part.columns:
                out[canonical] = out["period_no"].map(part.set_index("period_no")[canonical])
            elif canonical in fallback.columns:
                out[canonical] = out["period_no"].map(fallback.set_index("period_no")[canonical])
            else:
                out[canonical] = np.nan
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Keep every period exactly once; duplicate authoritative rows are a hard
    # source error rather than a silent last-write-wins merge.
    if out.duplicated(["market_date", "period_no"]).any():
        raise ValueError("SNAPSHOT_DUPLICATE_GRID_ROWS")
    return out.sort_values(["market_date", "period_no"]).reset_index(drop=True)


class SnapshotBuilder:
    """Freeze the D/T serving facts after a successful formal DB sync."""

    protocol = DYNAMIC_PROTOCOL

    def __init__(self, *, model_store_path: str | Path, authoritative_path: str | Path | None = None,
                 decision_day: str | None = None, target_day: str, output_dir: str | Path):
        self.model_store_path = Path(model_store_path)
        self.authoritative_path = Path(authoritative_path) if authoritative_path else self.model_store_path
        self.decision_day = pd.Timestamp(decision_day or (pd.Timestamp(target_day) - pd.Timedelta(days=1))).normalize()
        self.target_day = pd.Timestamp(target_day).normalize()
        self.output_dir = Path(output_dir)

    def build(self) -> dict:
        model_store = _read_96_table(self.model_store_path)
        authority = _read_96_table(self.authoritative_path)

        # Critical-source readiness must be checked on the synchronized
        # authoritative facts before model-store fallback can hide a source
        # outage. Ordinary cell gaps are still allowed and routed later.
        decision_raw = authority[authority["market_date"].eq(self.decision_day)].copy()
        target_raw = authority[authority["market_date"].eq(self.target_day)].copy()

        da_source_col = (
            "日前出清价格" if "日前出清价格" in decision_raw.columns
            else "日前电价" if "日前电价" in decision_raw.columns
            else None
        )
        da_source_rows = (
            int(pd.to_numeric(decision_raw[da_source_col], errors="coerce").notna().sum())
            if da_source_col is not None else 0
        )
        if da_source_rows != 96:
            raise ValueError(
                "MISSING_CRITICAL_SOURCE source=decision_day_day_ahead "
                f"field={da_source_col or '日前电价'} rows={da_source_rows} expected=96"
            )

        raw_forecast_counts: dict[str, int] = {}
        for canonical, raw in RAW_FORECAST_COLUMNS.items():
            source_col = raw if raw in target_raw.columns else canonical if canonical in target_raw.columns else None
            raw_forecast_counts[canonical] = (
                int(pd.to_numeric(target_raw[source_col], errors="coerce").notna().sum())
                if source_col is not None else 0
            )
        if sum(raw_forecast_counts.values()) == 0:
            raise ValueError(
                "MISSING_CRITICAL_SOURCE source=target_day_forecast "
                f"field=ForecastData rows=0 expected>0 details={raw_forecast_counts}"
            )

        values = _canonical_raw_view(authority, model_store, [self.decision_day, self.target_day])
        expected = pd.MultiIndex.from_product([[self.decision_day, self.target_day], range(1, 97)], names=["market_date", "period_no"])
        actual = pd.MultiIndex.from_frame(values[["market_date", "period_no"]]) if not values.empty else pd.MultiIndex.from_arrays([[], []], names=expected.names)
        missing = expected.difference(actual)
        if len(missing):
            day = str(missing[0][0].date())
            raise ValueError(f"MISSING_CRITICAL_SOURCE source=decision_target_grid field=market_date/period_no rows={96-len(values[values['market_date'].eq(pd.Timestamp(day))]) if not values.empty else 0} expected=96 day={day}")
        da_rows = int(values.loc[values["market_date"].eq(self.decision_day), "日前电价"].notna().sum())
        target_forecast = {c: int(values.loc[values["market_date"].eq(self.target_day), c].notna().sum()) for c in PRIMITIVE_FORECAST_COLUMNS}
        if da_rows != 96:
            raise ValueError(f"MISSING_CRITICAL_SOURCE source=decision_day_day_ahead field=日前电价 rows={da_rows} expected=96")
        missing_forecast = {c: n for c, n in target_forecast.items() if n != 96}
        if missing_forecast:
            raise ValueError(f"MISSING_CRITICAL_SOURCE source=target_day_forecast field={next(iter(missing_forecast))} rows={next(iter(missing_forecast.values()))} expected=96")
        model_hash = _sha256_file(self.model_store_path)
        authority_hash = _sha256_file(self.authoritative_path)
        payload = values.copy()
        for col in payload.columns:
            if col not in {"market_date", "period_no"}:
                payload[col] = pd.to_numeric(payload[col], errors="coerce")
        fingerprint = {
            "protocol": self.protocol, "target_day": self.target_day.date().isoformat(),
            "decision_day": self.decision_day.date().isoformat(), "model_store_sha256": model_hash,
            "values": json.loads(payload.to_json(orient="records", date_format="iso")),
        }
        snapshot_id = hashlib.sha256(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        values_path = self.output_dir / "values.parquet"
        manifest_path = self.output_dir / "snapshot_manifest.json"
        tmp_values = values_path.with_suffix(".parquet.partial")
        values.to_parquet(tmp_values, index=False)
        os.replace(tmp_values, values_path)
        field_summary = {}
        for field in [*PRIMITIVE_FORECAST_COLUMNS, *PRIMITIVE_ACTUAL_COLUMNS, "日前电价", "实时电价"]:
            part = values[field] if field in values else pd.Series(dtype=float)
            nonnull = part.notna()
            field_summary[field] = {"nonnull_count": int(nonnull.sum()), "visible_until_period": int(values.loc[nonnull, "period_no"].max()) if nonnull.any() else None}
        manifest = {
            "protocol": self.protocol, "snapshot_id": snapshot_id,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "decision_day": self.decision_day.date().isoformat(), "target_day": self.target_day.date().isoformat(),
            "latest_closed_day": str(model_store.loc[model_store["market_date"] < self.decision_day, "market_date"].max().date()) if (model_store["market_date"] < self.decision_day).any() else None,
            "model_store_sha256": model_hash, "authoritative_sha256": authority_hash,
            "values_path": str(values_path), "values_sha256": _sha256_file(values_path), "field_summary": field_summary,
            "grid_rows": int(len(values)), "grid_expected": 192,
        }
        tmp_manifest = manifest_path.with_suffix(".json.partial")
        tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_manifest, manifest_path)
        manifest["manifest_path"] = str(manifest_path)
        # Keep the persisted JSON self-describing as well as the returned
        # in-memory manifest used by the prediction stage.
        tmp_manifest = manifest_path.with_suffix(".json.partial")
        tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_manifest, manifest_path)
        return {"status": "PASS", "snapshot_id": snapshot_id, "values_path": str(values_path), "manifest_path": str(manifest_path), "manifest": manifest}


def build_dynamic_snapshot_96(*, model_store_path: str | Path, authoritative_path: str | Path | None = None,
                              target_day: str, output_dir: str | Path) -> dict:
    return SnapshotBuilder(model_store_path=model_store_path, authoritative_path=authoritative_path, target_day=target_day, output_dir=output_dir).build()


def build_historical_proxy_snapshot_96(*, model_store_path: str | Path,
                                       authoritative_path: str | Path | None = None,
                                       target_day: str, output_dir: str | Path,
                                       proxy_cutoff_period: int = 56) -> dict:
    """Build the deterministic operational proxy for a closed historical day.

    The source is still frozen by :class:`SnapshotBuilder`; only the explicit
    historical policy masks the unavailable vintage afterwards.  This keeps
    FeatureViewBuilder as the sole serving visibility implementation.
    """
    base = SnapshotBuilder(
        model_store_path=model_store_path,
        authoritative_path=authoritative_path,
        target_day=target_day,
        output_dir=output_dir,
    ).build()
    values_path = Path(base["values_path"])
    manifest_path = Path(base["manifest_path"])
    values = pd.read_parquet(values_path)
    target = pd.Timestamp(target_day).normalize()
    decision = target - pd.Timedelta(days=1)
    cutoff = int(proxy_cutoff_period)
    decision_mask = values["market_date"].eq(decision)
    target_mask = values["market_date"].eq(target)
    # Historical final/RT are observable only through the legacy p56 proxy;
    # RealityTmp has no trustworthy historical vintage and is fully masked.
    for col in PRIMITIVE_ACTUAL_COLUMNS:
        values.loc[decision_mask & values["period_no"].gt(cutoff), col] = np.nan
        values.loc[decision_mask, f"{col}__tmp"] = np.nan
        values.loc[target_mask, col] = np.nan
        values.loc[target_mask, f"{col}__tmp"] = np.nan
    values.loc[decision_mask & values["period_no"].gt(cutoff), "实时电价"] = np.nan
    values.loc[target_mask, ["日前电价", "实时电价"]] = np.nan
    values = values.sort_values(["market_date", "period_no"]).reset_index(drop=True)
    payload = values.copy()
    for col in payload.columns:
        if col not in {"market_date", "period_no"}:
            payload[col] = pd.to_numeric(payload[col], errors="coerce")
    fingerprint = {
        "protocol": HISTORICAL_PROXY_PROTOCOL,
        "target_day": target.date().isoformat(),
        "decision_day": decision.date().isoformat(),
        "proxy_cutoff_period": cutoff,
        "values": json.loads(payload.to_json(orient="records", date_format="iso")),
    }
    snapshot_id = hashlib.sha256(
        json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    tmp_values = values_path.with_suffix(".parquet.partial")
    values.to_parquet(tmp_values, index=False)
    os.replace(tmp_values, values_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    field_summary = {}
    for field in [*PRIMITIVE_FORECAST_COLUMNS, *PRIMITIVE_ACTUAL_COLUMNS, "日前电价", "实时电价"]:
        part = values[field] if field in values else pd.Series(dtype=float)
        nonnull = part.notna()
        field_summary[field] = {
            "nonnull_count": int(nonnull.sum()),
            "visible_until_period": int(values.loc[nonnull, "period_no"].max()) if nonnull.any() else None,
        }
    manifest.update({
        "protocol": HISTORICAL_PROXY_PROTOCOL,
        "snapshot_id": snapshot_id,
        "snapshot_kind": "historical_proxy",
        "run_mode": HISTORICAL_PROXY_V1,
        "proxy_policy_version": "historical_proxy_v1_p56",
        "proxy_cutoff_period": cutoff,
        "proxy_cutoff": "14:00",
        "actual_prefix_source": "historical_final",
        "rt_prefix_source": "historical_final",
        "historical_vintage": "UNVERIFIED_LEGACY_VINTAGE",
        "strict_historical_vintage_proven": False,
        "field_summary": field_summary,
    })
    manifest["values_path"] = str(values_path)
    manifest["values_sha256"] = _sha256_file(values_path)
    manifest["manifest_path"] = str(manifest_path)
    tmp_manifest = manifest_path.with_suffix(".json.partial")
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    return {"status": "PASS", "snapshot_id": snapshot_id, "values_path": str(values_path),
            "manifest_path": str(manifest_path), "manifest": manifest}


def _manifest_prediction_source(payload: dict) -> dict | None:
    """Find a successful ledger_predict provenance without directory guessing."""
    if not isinstance(payload, dict):
        return None
    if payload.get("pipeline") == "ledger_predict" and (
        payload.get("dynamic_snapshot") or payload.get("snapshot_id")
    ):
        return payload if payload.get("status") in {"complete", "complete_with_warnings"} else None
    for nested_key in ("prediction_provenance", "previous_prediction_provenance"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            found = _manifest_prediction_source(nested)
            if found:
                return found
    stage = payload.get("stages", {}).get("ledger_predict")
    if isinstance(stage, dict):
        found = _manifest_prediction_source(stage)
        if found:
            return found
    return None


def _valid_stored_live_snapshot(run_dir: Path | None, target_day: str) -> dict | None:
    """Validate the exact snapshot bound by a successful manifest.

    We intentionally inspect only manifest-declared paths.  A sibling named
    ``attempt_*`` is never selected merely because it is newer.
    """
    if run_dir is None or not run_dir.exists():
        return None
    manifests = [run_dir / "run_manifest.json", run_dir / "runtime" / "stage_manifests" / "ledger_predict.json"]
    for path in manifests:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = _manifest_prediction_source(payload)
        if not source:
            continue
        snap_meta = source.get("dynamic_snapshot") or {}
        if (source.get("serving_protocol") or source.get("production_contract")) != DYNAMIC_PROTOCOL:
            continue
        if source.get("target_date") != str(target_day):
            continue
        if source.get("selected_model_pool"):
            if list(source["selected_model_pool"].get("dayahead", ())) != list(DAYAHEAD_MODELS):
                continue
            if list(source["selected_model_pool"].get("realtime", ())) != list(REALTIME_MODELS):
                continue
        results = source.get("results", {})
        if not isinstance(results, dict):
            continue
        if any(
            not isinstance(results.get(task), dict)
            or any(results[task].get(model, {}).get("status") not in {"ok", "cached"}
                   for model in models)
            for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS))
        ):
            continue
        values_path = snap_meta.get("values_path")
        manifest_path = snap_meta.get("manifest_path")
        if not values_path or not manifest_path or not Path(values_path).exists() or not Path(manifest_path).exists():
            continue
        try:
            persisted = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if persisted.get("protocol") != DYNAMIC_PROTOCOL:
                continue
            if persisted.get("snapshot_id") != source.get("snapshot_id") or persisted.get("snapshot_id") != snap_meta.get("snapshot_id"):
                continue
            if persisted.get("target_day") != str(target_day):
                continue
            decision = (pd.Timestamp(target_day) - pd.Timedelta(days=1)).date().isoformat()
            if persisted.get("decision_day") != decision:
                continue
            values = _read_96_table(values_path)
            if len(values) != 192 or values.duplicated(["market_date", "period_no"]).any():
                continue
            if persisted.get("values_sha256") and persisted["values_sha256"] != _sha256_file(values_path):
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        return {
            "route": STORED_LIVE_SNAPSHOT_REPLAY,
            "run_mode": STORED_LIVE_SNAPSHOT_REPLAY,
            "snapshot_kind": "live",
            "snapshot_id": source.get("snapshot_id"),
            "values_path": str(Path(values_path)),
            "manifest_path": str(Path(manifest_path)),
            "manifest": persisted,
            "source_manifest": source,
        }
    return None


def resolve_formal96_snapshot_route(*, target_day: str, model_store_path: str | Path,
                                    authoritative_path: str | Path | None = None,
                                    output_dir: str | Path | None = None,
                                    run_dir: str | Path | None = None,
                                    runs_root: str | Path | None = None,
                                    latest_closed_day: str | None = None,
                                    current_target_day: str | None = None) -> dict:
    """Resolve stored-live, historical-proxy, or live-dynamic in that order.

    Formal production callers must pass latest_closed_day from the DB sync
    manifest. current_target_day remains only as a compatibility fallback
    for isolated tests/legacy callers.
    """
    target = pd.Timestamp(target_day).normalize()
    candidate_dir = Path(run_dir) if run_dir is not None else (
        Path(runs_root) / str(target_day) if runs_root is not None else None
    )
    if latest_closed_day:
        closed = pd.Timestamp(latest_closed_day).normalize()
        historical_closed = target <= closed
    else:
        current = pd.Timestamp(current_target_day or datetime.now(timezone.utc).date()).normalize()
        historical_closed = target < current
    stored = _valid_stored_live_snapshot(candidate_dir, str(target_day)) if historical_closed else None
    if stored:
        return stored
    if historical_closed:
        if output_dir is None:
            raise ValueError("HISTORICAL_PROXY_OUTPUT_DIR_REQUIRED")
        built = build_historical_proxy_snapshot_96(
            model_store_path=model_store_path,
            authoritative_path=authoritative_path,
            target_day=str(target_day),
            output_dir=output_dir,
        )
        return {
            "route": HISTORICAL_PROXY_V1,
            "run_mode": HISTORICAL_PROXY_V1,
            "snapshot_kind": "historical_proxy",
            "snapshot_id": built["snapshot_id"],
            "values_path": built["values_path"],
            "manifest_path": built["manifest_path"],
            "manifest": built["manifest"],
            "source_manifest": None,
        }
    if output_dir is None:
        raise ValueError("LIVE_DYNAMIC_OUTPUT_DIR_REQUIRED")
    built = build_dynamic_snapshot_96(
        model_store_path=model_store_path,
        authoritative_path=authoritative_path,
        target_day=str(target_day),
        output_dir=output_dir,
    )
    manifest = dict(built["manifest"])
    manifest.update({"run_mode": LIVE_DYNAMIC, "snapshot_kind": "live"})
    # Persist the route metadata so later Route A validation has an immutable
    # source of truth even when this helper is called outside ledger_predict.
    manifest_path = Path(built["manifest_path"])
    tmp = manifest_path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, manifest_path)
    built["manifest"] = manifest
    return {
        "route": LIVE_DYNAMIC,
        "run_mode": LIVE_DYNAMIC,
        "snapshot_kind": "live",
        "snapshot_id": built["snapshot_id"],
        "values_path": built["values_path"],
        "manifest_path": built["manifest_path"],
        "manifest": manifest,
        "source_manifest": None,
    }


def _latest_same_period(history: pd.DataFrame, field: str, period: int, before_day: pd.Timestamp) -> float | None:
    series = history[(history["market_date"] < before_day) & (history["period_no"].eq(period))][["market_date", field]].dropna()
    if series.empty:
        return None
    return float(series.sort_values("market_date").iloc[-1][field])


def _median_same_period(history: pd.DataFrame, field: str, period: int, before_day: pd.Timestamp) -> float | None:
    series = history[(history["market_date"] < before_day) & (history["period_no"].eq(period))][field].dropna().tail(30)
    return float(series.median()) if not series.empty else None


def build_dynamic_feature_view_96(*, model_store_path: str | Path, snapshot_values: str | Path | pd.DataFrame,
                                 snapshot_manifest: dict | str | Path, target_day: str,
                                 output_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Route D/T facts into one canonical serving view; no model cutoff logic."""
    history = _read_96_table(model_store_path)
    if isinstance(snapshot_values, pd.DataFrame):
        snap = snapshot_values.copy()
    else:
        snap = _read_96_table(snapshot_values)
    if isinstance(snapshot_manifest, (str, Path)):
        manifest = json.loads(Path(snapshot_manifest).read_text(encoding="utf-8"))
    else:
        manifest = dict(snapshot_manifest)
    if manifest.get("protocol") not in SNAPSHOT_PROTOCOLS:
        raise ValueError(
            f"FEATURE_VIEW_PROTOCOL_MISMATCH expected={sorted(SNAPSHOT_PROTOCOLS)} "
            f"actual={manifest.get('protocol')!r}"
        )
    if not manifest.get("snapshot_id"):
        raise ValueError("FEATURE_VIEW_SNAPSHOT_ID_MISSING")
    if snap.empty or snap.duplicated(["market_date", "period_no"]).any():
        raise ValueError("FEATURE_VIEW_INVALID_SNAPSHOT_GRID")
    target = pd.Timestamp(target_day).normalize(); decision = target - pd.Timedelta(days=1)
    view = history.copy()
    for day in (decision, target):
        periods = view.loc[view["market_date"].eq(day), "period_no"]
        if len(periods) != 96 or set(periods.astype(int)) != set(range(1, 97)):
            raise ValueError(
                f"MISSING_CRITICAL_SOURCE source=model_store field=market_date/period_no "
                f"rows={len(periods)} expected=96 day={day.date()}"
            )
    # Ensure canonical derived columns exist before routing.
    for col in [*PRIMITIVE_FORECAST_COLUMNS, *PRIMITIVE_ACTUAL_COLUMNS, "日前电价", "实时电价", *FORECAST_COLUMNS]:
        if col not in view.columns:
            view[col] = np.nan
    routes = {c: {"final_cells": 0, "tmp_cells": 0, "forecast_fill_cells": 0, "da_fill_cells": 0, "latest_closed_fill_cells": 0, "median_fill_cells": 0, "remaining_nan_cells": 0} for c in [*PRIMITIVE_ACTUAL_COLUMNS, "实时电价"]}
    snap_idx = snap.set_index(["market_date", "period_no"]) if not snap.empty else pd.DataFrame()
    hist_idx = history.set_index(["market_date", "period_no"])
    for day in (decision, target):
        mask = view["market_date"].eq(day)
        for p in range(1, 97):
            row_mask = mask & view["period_no"].eq(p)
            if not row_mask.any():
                continue
            key = (day, p)
            srow = snap_idx.loc[key] if not snap_idx.empty and key in snap_idx.index else None

            # D/T serving facts come from the frozen snapshot, not from the
            # mutable persistent model store.  This makes snapshot_id the true
            # identity of the model input even if another sync happens later.
            if srow is not None:
                for forecast in PRIMITIVE_FORECAST_COLUMNS:
                    view.loc[row_mask, forecast] = srow.get(forecast, np.nan)
                if day == decision:
                    view.loc[row_mask, "日前电价"] = srow.get("日前电价", np.nan)

            for actual in PRIMITIVE_ACTUAL_COLUMNS:
                if day == target:
                    view.loc[row_mask, actual] = np.nan
                    continue
                value = srow.get(actual) if srow is not None else np.nan
                source = "final"
                if pd.notna(value):
                    routes[actual]["final_cells"] += 1
                else:
                    tmp_col = f"{actual}__tmp"
                    value = srow.get(tmp_col) if srow is not None else np.nan
                    source = "tmp"
                    if pd.notna(value): routes[actual]["tmp_cells"] += 1
                if pd.isna(value):
                    fcol = ACTUAL_TO_FORECAST[actual]
                    value = srow.get(fcol) if srow is not None else np.nan
                    source = "forecast"
                    if pd.notna(value): routes[actual]["forecast_fill_cells"] += 1
                if pd.isna(value):
                    value = _latest_same_period(history, actual, p, day); source = "latest_closed"
                    if value is not None: routes[actual]["latest_closed_fill_cells"] += 1
                if pd.isna(value):
                    value = _median_same_period(history, actual, p, day); source = "median"
                    if value is not None: routes[actual]["median_fill_cells"] += 1
                view.loc[row_mask, actual] = value
                if pd.isna(value): routes[actual]["remaining_nan_cells"] += 1
            # RT effective on D only.  T truth is always masked.
            if day == target:
                view.loc[row_mask, ["日前电价", "实时电价"]] = np.nan
            else:
                rt = srow.get("实时电价") if srow is not None else np.nan
                if pd.notna(rt): routes["实时电价"]["final_cells"] += 1
                else:
                    rt = srow.get("日前电价") if srow is not None else np.nan
                    if pd.notna(rt): routes["实时电价"]["da_fill_cells"] += 1
                if pd.isna(rt):
                    rt = _latest_same_period(history, "实时电价", p, day)
                    if rt is not None: routes["实时电价"]["latest_closed_fill_cells"] += 1
                if pd.isna(rt):
                    rt = _median_same_period(history, "实时电价", p, day)
                    if rt is not None: routes["实时电价"]["median_fill_cells"] += 1
                view.loc[row_mask, "实时电价"] = rt
                if pd.isna(rt): routes["实时电价"]["remaining_nan_cells"] += 1
    # DA is never target-day truth.  Keep D DA and recompute all derived fields.
    view.loc[view["market_date"].eq(target), "日前电价"] = np.nan
    view["新能源总加预测值"] = view[["风电总加预测值", "光伏总加预测值"]].sum(axis=1, min_count=2)
    view["竞价空间预测值"] = view["直调负荷预测值"] - view[[c for c in PRIMITIVE_FORECAST_COLUMNS if c != "直调负荷预测值"]].sum(axis=1, min_count=7)
    view["新能源总加实际值"] = view[["风电总加实际值", "光伏总加实际值"]].sum(axis=1, min_count=2)
    view["竞价空间实际值"] = view["直调负荷实际值"] - view[PRIMITIVE_ACTUAL_COLUMNS[1:]].sum(axis=1, min_count=7)
    realized = ["日前电价", "实时电价", *PRIMITIVE_ACTUAL_COLUMNS]
    view.loc[view["market_date"].eq(target), realized] = np.nan
    view = view.sort_values(["market_date", "period_no"]).reset_index(drop=True)
    required_nan = {c: int(view.loc[view["market_date"].eq(decision), c].isna().sum()) for c in [*PRIMITIVE_ACTUAL_COLUMNS, "实时电价"]}
    audit = {"status": "PASS" if not any(required_nan.values()) else "FAIL", "protocol": manifest.get("protocol"),
             "snapshot_id": manifest.get("snapshot_id"), "decision_day": decision.date().isoformat(), "target_day": target.date().isoformat(),
             "snapshot_kind": manifest.get("snapshot_kind", "live"),
             "run_mode": manifest.get("run_mode", LIVE_DYNAMIC),
             "routes": routes, "remaining_nan": required_nan, "target_truth_mask": int(view.loc[view["market_date"].eq(target), realized].notna().sum().sum()) == 0}
    if audit["status"] != "PASS":
        missing = next(c for c, n in required_nan.items() if n)
        raise ValueError(f"MISSING_CRITICAL_SOURCE source=feature_view field={missing} rows={96-required_nan[missing]} expected=96")
    if output_path is not None:
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".partial")
        view.to_parquet(tmp, index=False); os.replace(tmp, output_path)
        audit["output_path"] = str(output_path)
    return view, audit


def build_asof_view_96(
    *,
    source_path: str | Path,
    target_day: str,
    cutoff_hour: int = 15,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return a masked 96-point view matching production information timing."""
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if source_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(source_path)
    elif source_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(source_path)
    else:
        df = pd.read_csv(source_path, encoding="utf-8-sig", low_memory=False)

    required = {"时刻", "market_date", "period_no", "日前电价", "实时电价", *FORECAST_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"96 as-of source missing required columns: {missing}")

    out = df.copy()
    out["时刻"] = pd.to_datetime(out["时刻"], errors="raise")
    out["market_date"] = pd.to_datetime(out["market_date"], errors="raise").dt.normalize()
    out["period_no"] = pd.to_numeric(out["period_no"], errors="raise").astype(int)

    target = pd.Timestamp(target_day).normalize()
    decision = target - pd.Timedelta(days=1)
    cutoff = decision + pd.Timedelta(hours=int(cutoff_hour))

    # Never expose rows beyond the requested target day.
    out = out.loc[out["market_date"].le(target)].copy()

    decision_rows = out.loc[out["market_date"].eq(decision)]
    if len(decision_rows) != 96 or decision_rows["period_no"].nunique() != 96:
        raise RuntimeError(
            f"DECISION_DAY_GRID_NOT_READY day={decision.date()} rows={len(decision_rows)} "
            f"periods={decision_rows['period_no'].nunique()}"
        )

    target_rows = out.loc[out["market_date"].eq(target)]
    if len(target_rows) != 96 or target_rows["period_no"].nunique() != 96:
        raise RuntimeError(
            f"TARGET_FORECAST_NOT_READY target={target.date()} rows={len(target_rows)} "
            f"periods={target_rows['period_no'].nunique()}"
        )

    decision_da_rows = int(decision_rows["日前电价"].notna().sum())
    if decision_da_rows != 96:
        raise RuntimeError(
            f"DECISION_DAY_DA_NOT_READY day={decision.date()} rows={decision_da_rows}/96"
        )

    target_rows = out.loc[out["market_date"].eq(target)]
    missing_forecasts = {
        c: int(target_rows[c].notna().sum()) for c in FORECAST_COLUMNS if int(target_rows[c].notna().sum()) != 96
    }
    if missing_forecasts:
        raise RuntimeError(
            f"TARGET_FORECAST_NOT_READY target={target.date()} incomplete={missing_forecasts}"
        )

    actual_cols = [c for c in out.columns if c.endswith("实际值")]
    decision_mask = out["market_date"].eq(decision)
    visible_decision_mask = decision_mask & out["时刻"].le(cutoff)
    expected_visible = int(cutoff_hour) * 4
    visible_rt_rows = int(out.loc[visible_decision_mask, "实时电价"].notna().sum())
    if visible_rt_rows != expected_visible:
        raise RuntimeError(
            f"DECISION_DAY_RT_NOT_READY day={decision.date()} cutoff={cutoff_hour:02d}:00 "
            f"rows={visible_rt_rows}/{expected_visible}"
        )
    incomplete_actual = {
        c: int(out.loc[visible_decision_mask, c].notna().sum())
        for c in actual_cols
        if int(out.loc[visible_decision_mask, c].notna().sum()) != expected_visible
    }
    readiness_warnings: list[str] = []
    if incomplete_actual:
        readiness_warnings.append(
            f"DECISION_DAY_ACTUAL_PARTIAL day={decision.date()} cutoff={cutoff_hour:02d}:00 "
            f"incomplete={incomplete_actual}"
        )

    post_cutoff = decision_mask & out["时刻"].gt(cutoff)
    if actual_cols:
        out.loc[post_cutoff, actual_cols] = np.nan
    out.loc[post_cutoff, "实时电价"] = np.nan

    target_mask = out["market_date"].eq(target)
    realized = ["日前电价", "实时电价", *actual_cols]
    out.loc[target_mask, realized] = np.nan

    # Assertions are deliberately strict: this is the outermost leakage wall.
    masked_target = out.loc[target_mask]
    if int(masked_target[realized].notna().sum().sum()) != 0:
        raise AssertionError("target-day realized fields were not fully masked")
    masked_decision = out.loc[decision_mask & out["时刻"].gt(cutoff)]
    if int(masked_decision[["实时电价", *actual_cols]].notna().sum().sum()) != 0:
        raise AssertionError("decision-day post-cutoff realized fields were not fully masked")

    out = out.sort_values(["market_date", "period_no"]).reset_index(drop=True)
    decision_mask_final = out["market_date"].eq(decision)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        out.to_parquet(tmp, index=False)
        tmp.replace(output_path)

    audit = {
        "status": "PASS",
        "source": str(source_path),
        "output": str(output_path) if output_path is not None else None,
        "target_day": target.date().isoformat(),
        "decision_day": decision.date().isoformat(),
        "cutoff": str(cutoff),
        "rows": int(len(out)),
        "decision_day_da_visible": decision_da_rows,
        "decision_day_rt_visible": int(
            out.loc[decision_mask_final & out["时刻"].le(cutoff), "实时电价"].notna().sum()
        ),
        "decision_day_actual_visible": {
            c: int(out.loc[decision_mask_final & out["时刻"].le(cutoff), c].notna().sum())
            for c in actual_cols
        },
        "target_forecast_nonnull": {c: int(masked_target[c].notna().sum()) for c in FORECAST_COLUMNS},
        "target_realized_nonnull": 0,
        "warnings": readiness_warnings,
    }
    return out, audit
