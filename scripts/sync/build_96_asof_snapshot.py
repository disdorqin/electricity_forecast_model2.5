"""Build a leak-safe 96-point serving/backtest snapshot.

The snapshot reproduces the production information boundary:
* decision-day realtime/actual values after ``--rt-cutoff-hour`` are hidden;
* target-day realized prices and actual features are always hidden;
* target-day forecast fundamentals come from the read-only
  ``epf_pmos_96_full`` local mirror.

Use the same cutoff in historical backtests and live runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_MAP = {
    "直调负荷预测": "直调负荷预测值",
    "地方电厂出力预测": "地方电厂总加预测值",
    "外电预测": "联络线受电负荷预测值",
    "风电预测": "风电总加预测值",
    "光伏预测": "光伏总加预测值",
    "核电预测": "核电总加预测值",
    "自备电厂预测": "自备机组总加预测值",
    "试验机组预测": "试验机组总加预测值",
}
ACTUAL_MAP = {
    "直调负荷实际": "直调负荷实际值",
    "地方电厂出力实际": "地方电厂总加实际值",
    "外电实际": "联络线受电负荷实际值",
    "风电实际": "风电总加实际值",
    "光伏实际": "光伏总加实际值",
    "核电实际": "核电总加实际值",
    "自备电厂实际": "自备机组总加实际值",
    "试验机组实际": "试验机组总加实际值",
}


def _period_number(label: str) -> int:
    text = str(label).strip()
    if text == "24:00":
        return 96
    hh, mm = text.split(":", 1)
    return (int(hh) * 60 + int(mm)) // 15


def build_snapshot(
    *,
    base_path: Path,
    remote_path: Path,
    decision_day: str,
    target_day: str,
    rt_cutoff_hour: int,
    output_path: Path,
    unit_id: str | None = None,
) -> dict:
    decision = pd.Timestamp(decision_day).normalize()
    target_date = pd.Timestamp(target_day).normalize()
    cutoff = decision + pd.Timedelta(hours=rt_cutoff_hour)

    base = pd.read_parquet(base_path).copy()
    base["时刻"] = pd.to_datetime(base["时刻"], errors="raise")
    base["market_date"] = pd.to_datetime(base["market_date"], errors="raise").dt.normalize()
    if base.empty or base["market_date"].max() < decision - pd.Timedelta(days=1):
        raise RuntimeError(
            f"closed-history base does not reach D-2 for decision day {decision.date()}: "
            f"max={base['market_date'].max().date() if not base.empty else None}"
        )
    actual_cols = [c for c in base.columns if c.endswith("实际值")]
    # Decision-day rows are always reconstructed from the same remote mirror
    # used in live serving, even when the historical base already contains them.
    base = base.loc[base["market_date"].lt(decision)].copy()

    remote = pd.read_parquet(remote_path).copy()
    remote["market_date"] = pd.to_datetime(remote["market_date"], errors="raise").dt.normalize()
    if "unit_id" in remote.columns:
        units = sorted(str(v) for v in remote["unit_id"].dropna().unique())
        selected = unit_id or (units[0] if len(units) == 1 else None)
        if selected is None:
            raise RuntimeError(f"multiple unit_id values in remote mirror; pass --unit-id. available={units}")
        remote = remote.loc[remote["unit_id"].astype(str).eq(str(selected))].copy()

    decision_raw = remote.loc[remote["market_date"].eq(decision)].copy()
    target = remote.loc[remote["market_date"].eq(target_date)].copy()
    for name, frame in (("decision-day", decision_raw), ("target-day", target)):
        frame["period_no"] = frame["时段"].map(_period_number)
        if len(frame) != 96 or frame["period_no"].nunique() != 96:
            raise RuntimeError(
                f"{name} rows incomplete: rows={len(frame)} periods={frame['period_no'].nunique()}"
            )
    missing_forecast = {
        src: int(target[src].notna().sum())
        for src in FORECAST_MAP
        if src not in target.columns or int(target[src].notna().sum()) != 96
    }
    if missing_forecast:
        raise RuntimeError(f"target-day forecast columns incomplete: {missing_forecast}")

    def make_shell(raw_day: pd.DataFrame, market_day: pd.Timestamp, *, target_only: bool) -> pd.DataFrame:
        raw_day = raw_day.sort_values("period_no").reset_index(drop=True)
        shell = pd.DataFrame(columns=base.columns)
        shell["market_date"] = pd.Series([market_day] * 96, dtype="datetime64[ns]")
        shell["period_no"] = raw_day["period_no"].astype(int).to_numpy()
        shell["时刻"] = market_day + pd.to_timedelta(raw_day["period_no"].astype(int) * 15, unit="m")
        shell.loc[shell["period_no"].eq(96), "时刻"] = market_day + pd.Timedelta(days=1)
        for source, canonical in FORECAST_MAP.items():
            shell[canonical] = pd.to_numeric(raw_day[source], errors="coerce").to_numpy()
        shell["竞价空间预测值"] = (
            shell["直调负荷预测值"]
            - shell[[
                "地方电厂总加预测值",
                "联络线受电负荷预测值",
                "风电总加预测值",
                "光伏总加预测值",
                "核电总加预测值",
                "自备机组总加预测值",
                "试验机组总加预测值",
            ]].sum(axis=1)
        )
        shell["新能源总加预测值"] = shell["风电总加预测值"] + shell["光伏总加预测值"]
        if target_only:
            for col in ["日前电价", "实时电价", *actual_cols]:
                if col in shell.columns:
                    shell[col] = np.nan
        else:
            shell["日前电价"] = pd.to_numeric(raw_day["日前出清价格"], errors="coerce").to_numpy()
            shell["实时电价"] = pd.to_numeric(raw_day["实时出清价格"], errors="coerce").to_numpy()
            for source, canonical in ACTUAL_MAP.items():
                shell[canonical] = pd.to_numeric(raw_day[source], errors="coerce").to_numpy()
            shell["竞价空间实际值"] = (
                shell["直调负荷实际值"]
                - shell[[
                    "地方电厂总加实际值",
                    "联络线受电负荷实际值",
                    "风电总加实际值",
                    "光伏总加实际值",
                    "核电总加实际值",
                    "自备机组总加实际值",
                    "试验机组总加实际值",
                ]].sum(axis=1, min_count=7)
            )
            shell["新能源总加实际值"] = shell["风电总加实际值"] + shell["光伏总加实际值"]
            post_cutoff_mask = shell["时刻"].gt(cutoff)
            shell.loc[post_cutoff_mask, ["实时电价", *actual_cols]] = np.nan
        return shell

    decision_shell = make_shell(decision_raw, decision, target_only=False)
    target_shell = make_shell(target, target_date, target_only=True)
    combined = pd.concat([base, decision_shell, target_shell], ignore_index=True).sort_values("时刻").reset_index(drop=True)
    out_target = combined.loc[combined["market_date"].eq(target_date)]
    out_decision = combined.loc[combined["market_date"].eq(decision)]
    post_cutoff = out_decision.loc[out_decision["时刻"].gt(cutoff)]

    assert len(out_target) == 96 and out_target["period_no"].nunique() == 96
    assert out_target[["日前电价", "实时电价", *actual_cols]].notna().sum().sum() == 0
    assert post_cutoff[["实时电价", *actual_cols]].notna().sum().sum() == 0
    required_forecasts = [*FORECAST_MAP.values(), "竞价空间预测值", "新能源总加预测值"]
    assert all(int(out_target[c].notna().sum()) == 96 for c in required_forecasts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return {
        "status": "PASS",
        "output": str(output_path),
        "decision_day": decision_day,
        "target_day": target_day,
        "rt_cutoff": str(cutoff),
        "decision_day_rows": int(len(out_decision)),
        "decision_day_post_cutoff_rows": int(len(post_cutoff)),
        "target_day_rows": int(len(out_target)),
        "decision_day_da_nonnull": int(out_decision["日前电价"].notna().sum()),
        "decision_day_rt_visible_nonnull": int(out_decision.loc[out_decision["时刻"].le(cutoff), "实时电价"].notna().sum()),
        "target_forecast_nonnull": {c: int(out_target[c].notna().sum()) for c in required_forecasts},
        "target_price_actual_nonnull": int(
            out_target[["日前电价", "实时电价", *actual_cols]].notna().sum().sum()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build leak-safe 96-point as-of serving/backtest snapshot")
    parser.add_argument("--base", required=True)
    parser.add_argument("--remote-full", required=True, help="epf_pmos_96_full local mirror parquet")
    parser.add_argument("--unit-id", default=None)
    parser.add_argument("--decision-day", required=True)
    parser.add_argument("--target-day", required=True)
    parser.add_argument("--rt-cutoff-hour", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = build_snapshot(
        base_path=Path(args.base),
        remote_path=Path(args.remote_full),
        decision_day=args.decision_day,
        target_day=args.target_day,
        rt_cutoff_hour=args.rt_cutoff_hour,
        output_path=Path(args.output),
        unit_id=args.unit_id,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
