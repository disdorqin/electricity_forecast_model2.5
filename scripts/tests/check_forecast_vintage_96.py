from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_layout import DATA  # noqa: E402
from utils.data_loader import load_table  # noqa: E402


FORECAST_COLUMNS = (
    "直调负荷预测",
    "地方电厂出力预测",
    "外电预测",
    "风电预测",
    "光伏预测",
    "核电预测",
    "自备电厂预测",
    "试验机组预测",
)


def _period_no(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    out = []
    for item in text:
        if item == "24:00":
            out.append(96)
            continue
        try:
            hh, mm = item.split(":", 1)
            out.append((int(hh) * 60 + int(mm)) // 15)
        except Exception:
            out.append(None)
    return pd.Series(out, index=values.index, dtype="Int64")


def _snapshot_status(
    snapshot_path: Path,
    target_day: str,
    cutoff: pd.Timestamp,
) -> tuple[str, dict[str, Any]]:
    if not snapshot_path.exists():
        return "UNVERIFIED_LEGACY_VINTAGE", {
            "snapshot": str(snapshot_path),
            "reason": "no independent D-1 forecast snapshot",
        }
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "SNAPSHOT_INVALID", {
            "snapshot": str(snapshot_path),
            "reason": f"unreadable snapshot: {exc}",
        }

    if str(payload.get("target_date")) != target_day:
        return "SNAPSHOT_INVALID", {
            "snapshot": str(snapshot_path),
            "reason": f"target_date={payload.get('target_date')!r}",
        }
    captured = pd.to_datetime(payload.get("captured_at"), errors="coerce", utc=True)
    cutoff_utc = cutoff.tz_localize("Asia/Shanghai").tz_convert("UTC")
    rows = payload.get("forecast") or []
    complete = bool(payload.get("complete"))
    if pd.isna(captured):
        return "SNAPSHOT_INVALID", {
            "snapshot": str(snapshot_path),
            "reason": "captured_at missing/unparseable",
        }
    if captured > cutoff_utc:
        return "SNAPSHOT_TOO_LATE", {
            "snapshot": str(snapshot_path),
            "captured_at": str(captured),
            "cutoff": str(cutoff_utc),
        }
    if not complete or len(rows) != 96:
        return "SNAPSHOT_INCOMPLETE", {
            "snapshot": str(snapshot_path),
            "captured_at": str(captured),
            "rows": len(rows),
            "complete": complete,
        }

    frame = pd.DataFrame(rows)
    missing = {}
    for column in FORECAST_COLUMNS:
        if column not in frame.columns:
            missing[column] = 0
            continue
        count = int(pd.to_numeric(frame[column], errors="coerce").notna().sum())
        if count != 96:
            missing[column] = count
    if missing:
        return "SNAPSHOT_INCOMPLETE", {
            "snapshot": str(snapshot_path),
            "captured_at": str(captured),
            "missing": missing,
        }

    return "STRICT_SNAPSHOT_AVAILABLE", {
        "snapshot": str(snapshot_path),
        "captured_at": str(captured),
        "cutoff": str(cutoff_utc),
        "rows": len(rows),
    }


def audit_vintage(
    *,
    source: Path,
    snapshot_root: Path,
    start: str,
    end: str,
    cutoff_hour: int = 15,
) -> dict[str, Any]:
    frame = load_table(source).copy()
    if "market_date" not in frame.columns or "时段" not in frame.columns:
        raise ValueError("source must contain market_date and 时段")
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce").dt.normalize()
    frame["period_no"] = _period_no(frame["时段"])

    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    daily = []
    status_counts: dict[str, int] = {}

    for target_day in dates:
        day = pd.Timestamp(target_day)
        cutoff = day - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour)
        rows = frame.loc[frame["market_date"].eq(day)].copy()
        latest_state = {
            "rows": int(len(rows)),
            "periods": int(rows["period_no"].nunique()) if not rows.empty else 0,
            "source_captured_at_min": None,
            "source_captured_at_max": None,
            "update_time_min": None,
            "update_time_max": None,
        }
        for column in ("source_captured_at", "update_time"):
            if column in rows.columns and not rows.empty:
                values = pd.to_datetime(rows[column], errors="coerce")
                latest_state[f"{column}_min"] = None if values.isna().all() else str(values.min())
                latest_state[f"{column}_max"] = None if values.isna().all() else str(values.max())

        status, evidence = _snapshot_status(
            snapshot_root / f"{target_day}.json",
            target_day,
            cutoff,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        daily.append({
            "target_day": target_day,
            "forecast_origin_cutoff": str(cutoff),
            "vintage_status": status,
            "strict_historical_vintage_proven": status == "STRICT_SNAPSHOT_AVAILABLE",
            "current_table": latest_state,
            "evidence": evidence,
        })

    strict_days = [row["target_day"] for row in daily if row["strict_historical_vintage_proven"]]
    return {
        "status": "PASS" if len(strict_days) == len(daily) else "UNVERIFIED",
        "source": str(source),
        "snapshot_root": str(snapshot_root),
        "range": {"start": start, "end": end},
        "cutoff_hour": int(cutoff_hour),
        "strict_days": strict_days,
        "strict_day_count": len(strict_days),
        "total_days": len(daily),
        "status_counts": status_counts,
        "interpretation": (
            "STRICT_SNAPSHOT_AVAILABLE proves an independent forecast snapshot existed by D-1 cutoff. "
            "UNVERIFIED_LEGACY_VINTAGE means the latest-state table cannot prove which forecast revision "
            "was available at the historical forecast origin."
        ),
        "daily": daily,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only historical forecast-vintage evidence audit for the 96-point chain."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DATA.remote_96_root / "parquet" / "epf_pmos_96_full.parquet",
    )
    parser.add_argument("--snapshot-root", type=Path, default=PROJECT_ROOT / "output_96" / "raw" / "next_forecast")
    parser.add_argument("--start", default="2026-08-15")
    parser.add_argument("--end", default="2026-09-15")
    parser.add_argument("--cutoff-hour", type=int, default=15)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--require-strict",
        action="store_true",
        help="Return non-zero unless every target day has an independent D-1 snapshot.",
    )
    args = parser.parse_args(argv)

    result = audit_vintage(
        source=args.source,
        snapshot_root=args.snapshot_root,
        start=args.start,
        end=args.end,
        cutoff_hour=args.cutoff_hour,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.report.with_suffix(args.report.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(args.report)
    print(payload)
    if args.require_strict and result["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
