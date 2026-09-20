"""Canonical epf_pmos_96_full -> local 96-point synchronization.

This is the production 96-point synchronization path used by
``main.py --pipeline sync_dataset --resolution 15min``.

Fresh-checkout contract
-----------------------
* Git does not need to contain ``data/``. Stable directories are created at
  runtime via :func:`utils.data_layout.ensure_data_directories`.
* The complete remote table is mirrored read-only to
  ``data/96/remote/parquet/epf_pmos_96_full.parquet``.
* The selected production unit is materialized to
  ``data/96/authoritative/pmos_96_全量.csv`` using the historical canonical
  filename expected by the rest of the 96-point chain.
* Remote access is SELECT-only. Credentials come from the same .env contract
  used by the 24-point synchronizer (DB_HOST, DB, DB_USER, DB_PWD, ...).

The authoritative CSV may contain a partial latest day. Training/model-input
builders must select closed historical days; live serving snapshots may use the
latest partial day according to the explicit as-of cutoff contract.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.data_layout import DATA, ensure_data_directories
from utils.database_operate import fetch_96_table, fetch_96_table_summary, get_db_server_version


TABLE = "epf_pmos_96_full"
UNIQUE_KEY = ["market_date", "时段", "unit_id"]
REMOTE_PARQUET = DATA.quarter_root / "remote" / "parquet" / f"{TABLE}.parquet"
REMOTE_RAW = DATA.quarter_root / "remote" / "raw" / f"{TABLE}.csv.gz"
AUTHORITATIVE_CSV = DATA.quarter_root / "authoritative" / "pmos_96_全量.csv"
MANIFEST_PATH = DATA.sync_96_root / "sync_manifest.json"

# Preserve the authoritative model/data contract while retaining QCTC disclosure
# extensions (temporary actual / boundary forecast) in the authoritative layer.
# model_input_full still decides separately which columns become model features.
AUTHORITATIVE_COLUMNS = [
    # 旧31列顺序保持不变，保证已有 CSV 可向后兼容；QCTC 扩展列统一追加在尾部。
    "market_date",
    "时段",
    "直调负荷预测",
    "地方电厂出力预测",
    "外电预测",
    "风电预测",
    "光伏预测",
    "核电预测",
    "自备电厂预测",
    "试验机组预测",
    "直调负荷实际",
    "地方电厂出力实际",
    "外电实际",
    "风电实际",
    "光伏实际",
    "核电实际",
    "自备电厂实际",
    "试验机组实际",
    "抽蓄实际",
    "日前出清价格",
    "日前出力",
    "日前电量",
    "日前开机状态",
    "日前电源类型",
    "实时出清价格",
    "实时出力",
    "实时电量",
    "实时开机状态",
    "实时电源类型",
    "正备用预测",
    "负备用预测",
    "全网负荷预测",
    "全网负荷实际",
    "直调负荷临时实际",
    "地方电厂出力临时实际",
    "外电临时实际",
    "风电临时实际",
    "光伏临时实际",
    "核电临时实际",
    "自备电厂临时实际",
    "试验机组临时实际",
    "抽蓄临时实际",
    "全网负荷临时实际",
    "边界全网负荷预测",
    "边界直调负荷预测",
    "边界外电预测",
    "边界风电预测",
    "边界光伏预测",
    "边界核电预测",
]

CLOSED_REQUIRED = [
    "日前出清价格",
    "实时出清价格",
    "直调负荷实际",
    "地方电厂出力实际",
    "外电实际",
    "风电实际",
    "光伏实际",
    "核电实际",
    "自备电厂实际",
    "试验机组实际",
]


def _period_number(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "24:00":
        return 96
    try:
        hh, mm = text.split(":", 1)
        minutes = int(hh) * 60 + int(mm)
    except (ValueError, AttributeError):
        return None
    if minutes == 0:
        return 96
    if minutes % 15 != 0 or minutes < 15 or minutes > 1440:
        return None
    return minutes // 15


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_csv_gz(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False, encoding="utf-8", compression="gzip")
    os.replace(tmp, path)


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _normalize_remote(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    missing = [c for c in UNIQUE_KEY if c not in df.columns]
    if missing:
        raise ValueError(f"{TABLE} missing key columns: {missing}")
    out = df.copy()
    out["market_date"] = pd.to_datetime(out["market_date"], errors="raise").dt.date
    out["_period_no"] = out["时段"].map(_period_number)
    if out["_period_no"].isna().any():
        bad = out.loc[out["_period_no"].isna(), "时段"].head(10).tolist()
        raise ValueError(f"Invalid 96-point period labels: {bad}")
    out = out.sort_values(["market_date", "_period_no", "unit_id"]).drop(columns=["_period_no"])
    out = out.drop_duplicates(UNIQUE_KEY, keep="last").reset_index(drop=True)
    return out


def _resolve_unit_id(df: pd.DataFrame, explicit: str | None) -> str:
    configured = (explicit or os.getenv("PMOS_96_UNIT_ID") or "").strip()
    available = sorted(str(v) for v in df.get("unit_id", pd.Series(dtype="object")).dropna().unique())
    if configured:
        if configured not in available:
            raise ValueError(f"Configured PMOS_96_UNIT_ID={configured!r} not found; available={available}")
        return configured
    if len(available) == 1:
        return available[0]
    if not available:
        raise ValueError(f"{TABLE} contains no unit_id values")
    raise ValueError(
        f"{TABLE} contains multiple units {available}; set --sync-unit-id or PMOS_96_UNIT_ID explicitly"
    )


def _latest_contiguous_rt_period(authority: pd.DataFrame) -> tuple[int, str | None]:
    if authority.empty:
        return 0, None
    latest_day = authority["market_date"].max()
    day = authority.loc[authority["market_date"].eq(latest_day)].copy()
    day["_period_no"] = day["时段"].map(_period_number)
    day = day.sort_values("_period_no")
    visible = set(day.loc[day["实时出清价格"].notna(), "_period_no"].astype(int))
    contiguous = 0
    for p in range(1, 97):
        if p not in visible:
            break
        contiguous = p
    if contiguous == 0:
        return 0, None
    label = day.loc[day["_period_no"].eq(contiguous), "时段"].iloc[0]
    return contiguous, str(label)


def _latest_closed_day(authority: pd.DataFrame) -> str | None:
    if authority.empty:
        return None
    complete: list[str] = []
    for day, group in authority.groupby("market_date", sort=True):
        periods = group["时段"].map(_period_number)
        if periods.nunique() != 96:
            continue
        if all(col in group.columns and group[col].notna().all() for col in CLOSED_REQUIRED):
            complete.append(str(day))
    return complete[-1] if complete else None


def _merge_incremental(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return _normalize_remote(incoming)
    return _normalize_remote(pd.concat([existing, incoming], ignore_index=True, sort=False))


def _ensure_hourly_fallback_source() -> dict[str, Any]:
    """Ensure the validated 24-point forecast fallback exists for 96 history."""
    if DATA.hourly_xlsx.exists() and DATA.hourly_xlsx.stat().st_size > 0:
        return {"status": "existing", "output_xlsx": str(DATA.hourly_xlsx)}
    from scripts.sync.sync_data import sync_dataset

    manifest = sync_dataset(source="auto", force=False)
    if manifest.get("status") not in {"ok", "skipped"}:
        raise RuntimeError(
            "96 model-input bootstrap requires the 24-point canonical forecast fallback, "
            f"but 24-point sync failed: {manifest.get('errors', [])}"
        )
    return manifest


def sync_pmos_96_full(args: Any) -> dict[str, Any]:
    """Synchronize ``epf_pmos_96_full`` and refresh all canonical 96 data layers."""
    ensure_data_directories()
    REMOTE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    REMOTE_RAW.parent.mkdir(parents=True, exist_ok=True)
    AUTHORITATIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    source = getattr(args, "sync_source", "db")
    if source not in {"db", "auto", "local"}:
        raise ValueError("96-point canonical sync supports --sync-source db|auto|local")
    mode = getattr(args, "sync_mode", "full")
    overlap_days = int(getattr(args, "sync_overlap_days", 7))
    explicit_unit = getattr(args, "sync_unit_id", None)

    started = datetime.now(timezone.utc)
    remote_summary: dict[str, Any] | None = None

    if source == "local":
        if not REMOTE_PARQUET.exists():
            raise FileNotFoundError(f"Local 96 mirror not found: {REMOTE_PARQUET}")
        merged = _normalize_remote(pd.read_parquet(REMOTE_PARQUET))
    else:
        if mode == "incremental" and REMOTE_PARQUET.exists():
            existing = _normalize_remote(pd.read_parquet(REMOTE_PARQUET))
            max_day = pd.to_datetime(existing["market_date"], errors="coerce").max()
            start_date = None if pd.isna(max_day) else (max_day - timedelta(days=overlap_days)).date().isoformat()
            incoming = fetch_96_table(
                TABLE,
                start_date=start_date,
                order_by="market_date ASC, 时段 ASC, unit_id ASC",
            )
            merged = _merge_incremental(existing, incoming)
        else:
            incoming = fetch_96_table(TABLE, order_by="market_date ASC, 时段 ASC, unit_id ASC")
            if incoming.empty:
                raise ValueError(f"Remote table {TABLE} returned 0 rows")
            merged = _normalize_remote(incoming)
        remote_summary = fetch_96_table_summary(TABLE)
        if mode == "full" and int(remote_summary.get("rows_total") or 0) != len(merged):
            raise RuntimeError(
                f"Full-sync row-count mismatch: remote={remote_summary.get('rows_total')} local={len(merged)}"
            )
        _atomic_parquet(merged, REMOTE_PARQUET)
        _atomic_csv_gz(merged, REMOTE_RAW)

    unit_id = _resolve_unit_id(merged, explicit_unit)
    authority = merged.loc[merged["unit_id"].astype(str).eq(unit_id)].copy()
    missing_cols = [c for c in AUTHORITATIVE_COLUMNS if c not in authority.columns]
    if missing_cols:
        raise ValueError(f"{TABLE} missing authoritative columns: {missing_cols}")
    authority["_period_no"] = authority["时段"].map(_period_number)
    authority = authority.sort_values(["market_date", "_period_no"]).drop(columns=["_period_no"])
    if authority.duplicated(["market_date", "时段"]).any():
        raise ValueError("Selected unit contains duplicate (market_date, 时段) rows")
    day_counts = authority.groupby("market_date")["时段"].nunique()
    bad_days = day_counts[day_counts != 96]
    if not bad_days.empty:
        raise ValueError(f"Selected unit contains incomplete 96-row dates: {bad_days.tail(10).to_dict()}")

    authority = authority[AUTHORITATIVE_COLUMNS].reset_index(drop=True)
    _atomic_csv(authority, AUTHORITATIVE_CSV)

    # Production model store: one persistent full parquet. Closed history
    # is a logical view selected by the runner; no second clean parquet is
    # rebuilt on every sync.
    hourly_bootstrap = _ensure_hourly_fallback_source()
    from scripts.sync.build_96_model_input_from_authoritative import (
        refresh_96_model_input_full,
    )

    hourly_fallback = DATA.hourly_csv if DATA.hourly_csv.exists() else DATA.hourly_xlsx
    model_inputs = refresh_96_model_input_full(
        authoritative_path=AUTHORITATIVE_CSV,
        hourly_path=hourly_fallback,
        overlap_days=overlap_days,
    )

    rt_period, rt_label = _latest_contiguous_rt_period(authority)
    latest_day = str(authority["market_date"].max()) if not authority.empty else None
    manifest = {
        "status": "ok",
        "pipeline": "sync_dataset",
        "resolution": "15min",
        "source_table": TABLE,
        "sync_mode": mode,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "db_server_version": get_db_server_version() if source != "local" else None,
        "unit_id": unit_id,
        "remote_rows": int(len(merged)),
        "authoritative_rows": int(len(authority)),
        "distinct_days": int(authority["market_date"].nunique()),
        "min_market_date": str(authority["market_date"].min()) if not authority.empty else None,
        "max_market_date": latest_day,
        "latest_closed_day": _latest_closed_day(authority),
        "latest_contiguous_rt_period": rt_period,
        "latest_contiguous_rt_label": rt_label,
        "remote_summary": remote_summary,
        "paths": {
            "remote_parquet": str(REMOTE_PARQUET),
            "remote_csv_gz": str(REMOTE_RAW),
            "authoritative_csv": str(AUTHORITATIVE_CSV),
            "model_input_full_parquet": model_inputs["full_parquet"],
        },
        "model_inputs": model_inputs,
        "hourly_fallback_bootstrap": hourly_bootstrap,
        "fresh_checkout_contract": "data/ is created automatically; DB credentials are read from .env",
        "leakage_contract": (
            "authoritative may contain a partial latest day; model_input_full is the single "
            "persistent model store; closed history is a logical view; the 96 runner builds an immutable "
            "D/T snapshot plus transient Dynamic-v1 FeatureView for one run and deletes the FeatureView "
            "after the full chain"
        ),
    }
    tmp = MANIFEST_PATH.with_suffix(".json.partial")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, MANIFEST_PATH)
    return manifest
