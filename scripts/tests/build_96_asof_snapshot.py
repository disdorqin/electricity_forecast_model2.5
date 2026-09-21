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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an isolated leak-safe 96-point as-of snapshot for chain validation.")
    parser.add_argument("--base", required=True, help="Clean model-input parquet through the decision day")
    parser.add_argument("--remote-market", required=True, help="Read-only epf_pmos_96_full mirror parquet containing target-day forecast rows")
    parser.add_argument("--unit-id", default=None, help="Optional epf_pmos_96_full unit_id; auto-selected when only one unit exists")
    parser.add_argument("--decision-day", required=True)
    parser.add_argument("--target-day", required=True)
    parser.add_argument("--rt-cutoff-hour", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base)
    remote_path = Path(args.remote_market)
    output_path = Path(args.output)
    decision_day = pd.Timestamp(args.decision_day).normalize()
    target_day = pd.Timestamp(args.target_day).normalize()
    cutoff = decision_day + pd.Timedelta(hours=args.rt_cutoff_hour)

    base = pd.read_parquet(base_path).copy()
    base["时刻"] = pd.to_datetime(base["时刻"], errors="raise")
    base["market_date"] = pd.to_datetime(base["market_date"], errors="raise").dt.normalize()
    if base["market_date"].max() < decision_day:
        raise RuntimeError(f"base does not reach decision day {decision_day.date()}: max={base['market_date'].max().date()}")

    # Physically mask realized realtime/actual information after the requested as-of cutoff.
    actual_cols = [c for c in base.columns if c.endswith("实际值")]
    decision_mask = base["market_date"].eq(decision_day) & base["时刻"].gt(cutoff)
    base.loc[decision_mask, ["实时电价", *actual_cols]] = np.nan

    # Never expose target-day prices or actuals in the inference snapshot.
    base = base.loc[base["market_date"].le(decision_day)].copy()

    remote = pd.read_parquet(remote_path).copy()
    remote["market_date"] = pd.to_datetime(remote["market_date"], errors="raise").dt.normalize()
    if "unit_id" in remote.columns:
        units = sorted(str(v) for v in remote["unit_id"].dropna().unique())
        unit_id = args.unit_id or (units[0] if len(units) == 1 else None)
        if unit_id is None:
            raise RuntimeError(f"multiple unit_id values in remote mirror; pass --unit-id. available={units}")
        remote = remote.loc[remote["unit_id"].astype(str).eq(str(unit_id))].copy()
    target = remote.loc[remote["market_date"].eq(target_day)].copy()
    target["period_no"] = target["时段"].astype(str).map(
        lambda s: 96 if s == "24:00" else (int(s[:2]) * 60 + int(s[3:5])) // 15
    )
    if len(target) != 96 or target["period_no"].nunique() != 96:
        raise RuntimeError(
            f"target-day forecast rows incomplete: rows={len(target)} periods={target['period_no'].nunique()}"
        )
    missing_forecast = {src: int(target[src].notna().sum()) for src in FORECAST_MAP if int(target[src].notna().sum()) != 96}
    if missing_forecast:
        raise RuntimeError(f"target-day forecast columns incomplete: {missing_forecast}")

    target = target.sort_values("period_no").reset_index(drop=True)
    shell = pd.DataFrame(columns=base.columns)
    shell["market_date"] = pd.Series([target_day] * 96, dtype="datetime64[ns]")
    shell["period_no"] = target["period_no"].astype(int).to_numpy()
    shell["时刻"] = target_day + pd.to_timedelta(target["period_no"].astype(int) * 15, unit="m")
    shell.loc[shell["period_no"].eq(96), "时刻"] = target_day + pd.Timedelta(days=1)
    for source, canonical in FORECAST_MAP.items():
        shell[canonical] = pd.to_numeric(target[source], errors="coerce").to_numpy()
    shell["竞价空间预测值"] = (
        shell["直调负荷预测值"]
        - shell[[
            "地方电厂总加预测值", "联络线受电负荷预测值", "风电总加预测值",
            "光伏总加预测值", "核电总加预测值", "自备机组总加预测值", "试验机组总加预测值",
        ]].sum(axis=1)
    )
    shell["新能源总加预测值"] = shell["风电总加预测值"] + shell["光伏总加预测值"]

    # Explicitly keep all target-day realized/label fields unavailable.
    for col in ["日前电价", "实时电价", *actual_cols]:
        if col in shell.columns:
            shell[col] = np.nan

    combined = pd.concat([base, shell], ignore_index=True)
    combined = combined.sort_values("时刻").reset_index(drop=True)

    out_target = combined.loc[combined["market_date"].eq(target_day)]
    out_decision = combined.loc[combined["market_date"].eq(decision_day)]
    assert len(out_target) == 96 and out_target["period_no"].nunique() == 96
    assert out_target[["日前电价", "实时电价", *actual_cols]].notna().sum().sum() == 0
    assert all(int(out_target[c].notna().sum()) == 96 for c in [*FORECAST_MAP.values(), "竞价空间预测值", "新能源总加预测值"])
    post_cutoff = out_decision.loc[out_decision["时刻"].gt(cutoff)]
    assert post_cutoff[["实时电价", *actual_cols]].notna().sum().sum() == 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print(
        {
            "status": "PASS",
            "output": str(output_path),
            "rows": int(len(combined)),
            "decision_day": args.decision_day,
            "target_day": args.target_day,
            "rt_cutoff": str(cutoff),
            "decision_day_rows": int(len(out_decision)),
            "decision_day_post_cutoff_rows": int(len(post_cutoff)),
            "target_day_rows": int(len(out_target)),
            "target_forecast_nonnull": {c: int(out_target[c].notna().sum()) for c in [*FORECAST_MAP.values(), "竞价空间预测值", "新能源总加预测值"]},
            "target_price_actual_nonnull": int(out_target[["日前电价", "实时电价", *actual_cols]].notna().sum().sum()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
