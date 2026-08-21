"""Build the clean 96-point model input from the authoritative rich CSV.

The crawler's rich table is intentionally kept separate from the model input.
This adapter is the only place that translates the rich 96-point schema into
the 24-compatible feature contract consumed by the existing models.

Rules
-----
* ``data/96/authoritative/pmos_96_全量.csv`` is read-only input.
* The clean source starts at 2022-07-12 by default.  Earlier rows contain
  genuine missing actual values for nuclear/self-owned/test units.
* Actual values are never filled from the 24-point table.
* Missing forecast values may be filled from the corresponding 24-point
  forecast.  The row-level provenance mask is written to the sync manifest.
* The derived bidding-space formula includes every supply component, matching
  the canonical 24-point table:
  direct_load - local_plant - tie_line - wind - solar - nuclear
  - self_owned - test_unit.

The generated CSV/XLSX/Parquet files are runtime data and are not source-code
artifacts.  The script is safe to rerun when the crawler refreshes the
authoritative table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_layout import DATA  # noqa: E402


DEFAULT_START_DATE = "2022-07-12"

RAW_FORECAST_TO_CANONICAL = {
    "直调负荷预测": "直调负荷预测值",
    "地方电厂出力预测": "地方电厂总加预测值",
    "外电预测": "联络线受电负荷预测值",
    "风电预测": "风电总加预测值",
    "光伏预测": "光伏总加预测值",
    "核电预测": "核电总加预测值",
    "自备电厂预测": "自备机组总加预测值",
    "试验机组预测": "试验机组总加预测值",
}
RAW_ACTUAL_TO_CANONICAL = {
    "直调负荷实际": "直调负荷实际值",
    "地方电厂出力实际": "地方电厂总加实际值",
    "外电实际": "联络线受电负荷实际值",
    "风电实际": "风电总加实际值",
    "光伏实际": "光伏总加实际值",
    "核电实际": "核电总加实际值",
    "自备电厂实际": "自备机组总加实际值",
    "试验机组实际": "试验机组总加实际值",
}

PRICE_MAP = {
    "日前出清价格": "日前电价",
    "实时出清价格": "实时电价",
}

CANONICAL_FORECAST = list(RAW_FORECAST_TO_CANONICAL.values())
CANONICAL_ACTUAL = list(RAW_ACTUAL_TO_CANONICAL.values())
DERIVED_COLUMNS = [
    "竞价空间预测值",
    "新能源总加预测值",
    "竞价空间实际值",
    "新能源总加实际值",
]
REQUIRED_CANONICAL = ["日前电价", "实时电价"] + CANONICAL_FORECAST + CANONICAL_ACTUAL + DERIVED_COLUMNS

SUPPLY_FORECAST = [
    "地方电厂总加预测值",
    "联络线受电负荷预测值",
    "风电总加预测值",
    "光伏总加预测值",
    "核电总加预测值",
    "自备机组总加预测值",
    "试验机组总加预测值",
]
SUPPLY_ACTUAL = [
    "地方电厂总加实际值",
    "联络线受电负荷实际值",
    "风电总加实际值",
    "光伏总加实际值",
    "核电总加实际值",
    "自备机组总加实际值",
    "试验机组总加实际值",
]

PAIR_AUDIT = [
    ("直调负荷预测", "直调负荷实际"),
    ("地方电厂出力预测", "地方电厂出力实际"),
    ("外电预测", "外电实际"),
    ("风电预测", "风电实际"),
    ("光伏预测", "光伏实际"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
    raise RuntimeError(f"Unable to decode CSV: {path}: {last_error}")


def _as_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _normalize_periods(values: pd.Series) -> pd.Series:
    """Normalize either 1..96 period numbers or HH:MM labels to 1..96."""
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    from_time = ((parsed.dt.hour * 60 + parsed.dt.minute) // 15).astype("Float64")
    from_time = from_time.where(from_time > 0, 96)
    from_time = from_time.where(parsed.notna(), text.eq("24:00").astype("Float64") * 96)
    numeric = numeric.where(numeric.between(1, 96), from_time)
    return numeric.round().astype("Int64")


def _business_day_and_hour(timestamps: pd.Series) -> tuple[pd.Series, pd.Series]:
    ts = pd.to_datetime(timestamps, errors="coerce")
    hour = ts.dt.hour.where(ts.dt.hour != 0, 24)
    business_day = ts.dt.normalize() - pd.to_timedelta((ts.dt.hour == 0).astype(int), unit="D")
    return business_day.dt.normalize(), hour.astype("Int64")


def _make_hourly_forecast_lookup(hourly: pd.DataFrame) -> pd.DataFrame:
    required = ["时刻"] + list(RAW_FORECAST_TO_CANONICAL.values())
    missing = [column for column in required if column not in hourly.columns]
    if missing:
        raise ValueError(f"24-point table is missing forecast columns: {missing}")

    out = hourly[required].copy()
    out["_business_day"], out["_hour"] = _business_day_and_hour(out["时刻"])
    _as_numeric(out, list(RAW_FORECAST_TO_CANONICAL.values()))
    out = out.drop(columns=["时刻"])
    if out.duplicated(["_business_day", "_hour"]).any():
        raise ValueError("24-point forecast lookup has duplicate business_day/hour keys")
    return out.set_index(["_business_day", "_hour"])


def _derive_space(frame: pd.DataFrame) -> None:
    frame["竞价空间预测值"] = frame["直调负荷预测值"] - frame[SUPPLY_FORECAST].sum(axis=1)
    frame["新能源总加预测值"] = frame["风电总加预测值"] + frame["光伏总加预测值"]
    frame["竞价空间实际值"] = frame["直调负荷实际值"] - frame[SUPPLY_ACTUAL].sum(axis=1)
    frame["新能源总加实际值"] = frame["风电总加实际值"] + frame["光伏总加实际值"]


def _validate_raw_shape(raw: pd.DataFrame, start_date: pd.Timestamp) -> None:
    required = {"market_date", "时段"} | set(RAW_FORECAST_TO_CANONICAL) | set(RAW_ACTUAL_TO_CANONICAL) | set(PRICE_MAP)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Authoritative 96 CSV is missing required columns: {missing}")

    if raw["market_date"].isna().any():
        raise ValueError("Authoritative 96 CSV contains invalid market_date values")
    if raw["时段"].isna().any():
        raise ValueError("Authoritative 96 CSV contains invalid period values")

    scoped = raw.loc[raw["market_date"] >= start_date].copy()
    if scoped.empty:
        raise ValueError(f"No authoritative rows from start date {start_date.date()}")
    scoped["时段"] = pd.to_numeric(scoped["时段"], errors="coerce")
    if not scoped["时段"].between(1, 96).all():
        raise ValueError("96-point period must be in 1..96")
    if scoped.duplicated(["market_date", "时段"]).any():
        raise ValueError("Duplicate (market_date, period) rows in authoritative CSV")
    counts = scoped.groupby("market_date")["时段"].nunique()
    bad = counts[counts != 96]
    if not bad.empty:
        raise ValueError(f"Incomplete 96-point dates after start: {bad.head(10).to_dict()}")


def _audit_prediction_actual_pairs(raw: pd.DataFrame, start_date: pd.Timestamp) -> dict:
    scoped = raw.loc[raw["market_date"] >= start_date]
    result: dict[str, dict[str, float | int]] = {}
    for forecast, actual in PAIR_AUDIT:
        f = pd.to_numeric(scoped[forecast], errors="coerce")
        a = pd.to_numeric(scoped[actual], errors="coerce")
        valid = f.notna() & a.notna()
        same = (f[valid] == a[valid]).sum()
        ratio = float(same / valid.sum()) if valid.sum() else 0.0
        result[forecast] = {"valid_rows": int(valid.sum()), "same_rows": int(same), "same_ratio": ratio}
        if ratio > 0.01:
            raise ValueError(f"Potential actual/forecast copy contamination: {forecast} ratio={ratio:.4%}")
    return result


def build_clean_input(
    authoritative_path: Path,
    hourly_path: Path,
    start_date: str = DEFAULT_START_DATE,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    start = pd.Timestamp(start_date).normalize()
    raw = _read_csv(authoritative_path)
    raw["market_date"] = pd.to_datetime(raw["market_date"], errors="coerce").dt.normalize()
    raw["时段"] = _normalize_periods(raw["时段"])
    _validate_raw_shape(raw, start)
    pair_audit = _audit_prediction_actual_pairs(raw, start)

    hourly = pd.read_excel(hourly_path)
    hourly_lookup = _make_hourly_forecast_lookup(hourly)

    source = raw.loc[raw["market_date"] >= start].copy()
    source = source.sort_values(["market_date", "时段"]).reset_index(drop=True)
    source["时刻"] = source["market_date"] + pd.to_timedelta((source["时段"].astype(int) * 15), unit="m")
    source.loc[source["时段"] == 96, "时刻"] = source.loc[source["时段"] == 96, "market_date"] + pd.Timedelta(days=1)

    clean = pd.DataFrame({
        "时刻": source["时刻"],
        "market_date": source["market_date"],
        "period_no": source["时段"].astype(int),
    })
    for raw_column, canonical in PRICE_MAP.items():
        clean[canonical] = pd.to_numeric(source[raw_column], errors="coerce")
    for raw_column, canonical in {**RAW_FORECAST_TO_CANONICAL, **RAW_ACTUAL_TO_CANONICAL}.items():
        clean[canonical] = pd.to_numeric(source[raw_column], errors="coerce")

    # Only forecast cells are eligible for the 24-point fallback.  The lookup
    # key is business_day + hourly hour, and each hourly value is repeated for
    # the four quarter-hour slots in that hour.
    clean["_business_day"] = source["market_date"]
    clean["_hour"] = ((source["时段"].astype(int) - 1) // 4 + 1).astype(int)
    lookup = hourly_lookup.reset_index()
    lookup = lookup.rename(columns={"_business_day": "_business_day", "_hour": "_hour"})
    clean = clean.merge(lookup, on=["_business_day", "_hour"], how="left", suffixes=("", "_hourly"), validate="many_to_one")

    fallback_counts: dict[str, int] = {}
    fallback_masks: dict[str, pd.Series] = {}
    for canonical in CANONICAL_FORECAST:
        hourly_column = f"{canonical}_hourly"
        if hourly_column not in clean.columns:
            raise AssertionError(f"Missing hourly fallback column: {hourly_column}")
        raw_missing = clean[canonical].isna()
        fallback_available = clean[hourly_column].notna()
        if (raw_missing & ~fallback_available).any():
            failed = clean.loc[raw_missing & ~fallback_available, ["market_date", "period_no"]].head(10)
            raise ValueError(f"24-point forecast fallback unavailable for {canonical}: {failed.to_dict('records')}")
        mask = raw_missing & fallback_available
        clean.loc[mask, canonical] = clean.loc[mask, hourly_column]
        fallback_masks[canonical] = mask
        fallback_counts[canonical] = int(mask.sum())

    clean.drop(columns=["_business_day", "_hour"] + [f"{c}_hourly" for c in CANONICAL_FORECAST], inplace=True)
    _derive_space(clean)

    # Actual columns after 2022-07-12 must be genuine authoritative values.
    actual_missing = {column: int(clean[column].isna().sum()) for column in CANONICAL_ACTUAL}
    if any(actual_missing.values()):
        raise ValueError(f"Missing actual values remain after {start.date()}: {actual_missing}")
    price_missing = {column: int(clean[column].isna().sum()) for column in ("日前电价", "实时电价")}
    if any(price_missing.values()):
        raise ValueError(f"Missing prices remain after {start.date()}: {price_missing}")

    derived_expected = {
        "竞价空间预测值": clean["直调负荷预测值"] - clean[SUPPLY_FORECAST].sum(axis=1),
        "新能源总加预测值": clean["风电总加预测值"] + clean["光伏总加预测值"],
        "竞价空间实际值": clean["直调负荷实际值"] - clean[SUPPLY_ACTUAL].sum(axis=1),
        "新能源总加实际值": clean["风电总加实际值"] + clean["光伏总加实际值"],
    }
    derived_max_diff = {}
    for column, expected in derived_expected.items():
        derived_max_diff[column] = float((clean[column] - expected).abs().max())
        if not np.allclose(clean[column], expected, atol=1e-8, rtol=1e-10):
            raise ValueError(f"Derived column mismatch: {column}")

    # Extra unit/status fields are retained in a separate extended table; the
    # clean model contract below deliberately contains only stable columns.
    extended = source.copy()
    extended["时刻"] = clean["时刻"].to_numpy()
    extended["period_no"] = clean["period_no"].to_numpy()
    extended["fallback_24_forecast_any"] = np.logical_or.reduce(
        [mask.to_numpy() for mask in fallback_masks.values()]
    )
    for canonical, mask in fallback_masks.items():
        extended[f"fallback_24_{canonical}"] = mask.to_numpy()

    clean = clean[["时刻", "market_date", "period_no"] + ["日前电价", "实时电价"] + CANONICAL_FORECAST + DERIVED_COLUMNS[:2] + CANONICAL_ACTUAL + DERIVED_COLUMNS[2:]]
    clean["时刻"] = pd.to_datetime(clean["时刻"])

    manifest = {
        "status": "ok",
        "adapter": "authoritative_96_to_24_compatible_v1",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "resolution": "15min",
        "start_date": start.date().isoformat(),
        "end_date": clean["market_date"].max().date().isoformat(),
        "rows": int(len(clean)),
        "days": int(clean["market_date"].nunique()),
        "rows_per_day": clean.groupby("market_date")["period_no"].nunique().value_counts().to_dict(),
        "source": str(authoritative_path),
        "source_sha256": _sha256(authoritative_path),
        "hourly_fallback_source": str(hourly_path),
        "hourly_source_sha256": _sha256(hourly_path),
        "forecast_fallback_policy": "missing_forecast_only_from_24_hourly_forecast",
        "actual_fallback_policy": "none",
        "forecast_fallback_counts": fallback_counts,
        "actual_missing_counts": actual_missing,
        "price_missing_counts": price_missing,
        "derived_max_abs_diff": derived_max_diff,
        "actual_forecast_same_value_audit": pair_audit,
        "required_columns": REQUIRED_CANONICAL,
        "model_training_start_policy": "source_start_plus_feature_warmup",
        "source_start_date": start.date().isoformat(),
        "feature_warmup_days": 7,
        "expected_unified_training_start": (start + pd.Timedelta(days=7)).date().isoformat(),
        "leakage_rule": "actual columns are labels/history only; target-day masking remains in model pipelines",
    }
    return clean, manifest, extended


def _write_outputs(clean: pd.DataFrame, extended: pd.DataFrame, manifest: dict, out_dir: Path, report_dir: Path, formats: set[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    if "parquet" in formats:
        path = out_dir / "shandong_pmos_96_model_input_clean.parquet"
        clean.to_parquet(path, index=False)
        output_paths["parquet"] = str(path)
    if "csv" in formats:
        path = out_dir / "shandong_pmos_96_model_input_clean.csv"
        clean.to_csv(path, index=False, encoding="utf-8-sig")
        output_paths["csv"] = str(path)
    if "xlsx" in formats:
        path = out_dir / "shandong_pmos_96_model_input_clean.xlsx"
        clean.to_excel(path, index=False)
        output_paths["xlsx"] = str(path)

    extended_path = report_dir / "pmos_96_extended_from_authoritative.parquet"
    extended.to_parquet(extended_path, index=False)
    manifest["outputs"] = {**output_paths, "extended_provenance_parquet": str(extended_path)}
    manifest_path = report_dir / "build_96_model_input_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapt authoritative rich 96-point data to the stable model input contract")
    parser.add_argument("--authority", type=Path, default=DATA.authoritative_96_actual_csv)
    parser.add_argument("--hourly", type=Path, default=DATA.hourly_xlsx)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--out-dir", type=Path, default=DATA.quarter_root / "model_input")
    parser.add_argument("--report-dir", type=Path, default=DATA.sync_96_root)
    parser.add_argument("--formats", default="parquet,csv,xlsx", help="comma-separated output formats")
    args = parser.parse_args()

    formats = {part.strip().lower() for part in args.formats.split(",") if part.strip()}
    allowed = {"parquet", "csv", "xlsx"}
    unknown = formats - allowed
    if unknown:
        parser.error(f"Unknown formats: {sorted(unknown)}")
    if not args.authority.exists():
        parser.error(f"Missing authoritative CSV: {args.authority}")
    if not args.hourly.exists():
        parser.error(f"Missing 24-point canonical table: {args.hourly}")

    clean, manifest, extended = build_clean_input(args.authority, args.hourly, args.start_date)
    _write_outputs(clean, extended, manifest, args.out_dir, args.report_dir, formats)
    print(json.dumps({
        "status": manifest["status"],
        "rows": manifest["rows"],
        "days": manifest["days"],
        "start_date": manifest["start_date"],
        "end_date": manifest["end_date"],
        "forecast_fallback_counts": manifest["forecast_fallback_counts"],
        "outputs": manifest["outputs"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
