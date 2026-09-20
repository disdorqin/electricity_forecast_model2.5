"""
EFM3 preflight health checks.

Default profile:
    python scripts/tests/check_preflight_health.py

Historical Cycle88 profile:
    python scripts/tests/check_preflight_health.py --profile spread24-historical
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_layout import DATA  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


def check(results: list[tuple[str, str, str]], name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


def info(name: str, detail: str = "") -> None:
    print(f"[INFO] {name}" + (f" — {detail}" if detail else ""))


def finish(results: list[tuple[str, str, str]], title: str = "健康检查") -> int:
    print("\n" + "=" * 60)
    fails = [row for row in results if row[0] == FAIL]
    print(f"{title}: {len(results) - len(fails)}/{len(results)} PASS" + (f", {len(fails)} FAIL" if fails else ""))
    for row in fails:
        print(f"  [FAIL] {row[1]} — {row[2]}")
    return_code = 1 if fails else 0
    print("结论:", "✅ 全绿，可安全开跑" if not fails else "❌ 有红项，必须先修复")
    return return_code


def run_production_profile(
    *,
    require_96_full_chain: bool = False,
    target_date_96: str | None = None,
) -> int:
    """Run production health checks.

    The default gate validates data/model readiness for prediction/backfill.
    Full-chain readiness additionally requires >=30 complete production-ledger
    days and is enabled with ``--require-96-full-chain``.
    """
    results: list[tuple[str, str, str]] = []
    remote_96 = DATA.remote_96_root / "parquet"

    # ── 1. 24 点数据完整性 ────────────────────────────────────────────
    h24_path = DATA.hourly_xlsx
    if h24_path.exists():
        h24 = pd.read_excel(h24_path)
        h24["时刻"] = pd.to_datetime(h24["时刻"], errors="coerce")
        h24 = h24.dropna(subset=["时刻"])
        check(results, "24点表存在且非空", not h24.empty, f"rows={len(h24)}")
        if not h24.empty:
            check(results, "24点覆盖2022至今", h24["时刻"].min().date() <= pd.Timestamp("2022-01-01").date(), f"min={h24['时刻'].min()}")
            check(results, "24点数据新鲜(近3天内)", h24["时刻"].max() >= pd.Timestamp.now().normalize() - pd.Timedelta(days=3), f"max={h24['时刻'].max()}")
            for col in ["日前电价", "实时电价"]:
                if col in h24.columns:
                    nan_pct = h24[col].isna().mean() * 100
                    check(results, f"24点{col} NaN率<10%", nan_pct < 10, f"{nan_pct:.2f}%")
    else:
        check(results, "24点表存在且非空", False, "shandong_pmos_hourly.xlsx 缺失")

    # ── 2. 96 点 canonical 本地镜像真实性 ────────────────────────────
    mkt_path = remote_96 / "epf_pmos_96_full.parquet"
    if mkt_path.exists():
        mkt = pd.read_parquet(mkt_path)
        mkt["md"] = pd.to_datetime(mkt["market_date"], errors="coerce").dt.date
        cutoff = pd.Timestamp.now().normalize().date() - pd.Timedelta(days=30)
        recent = mkt[mkt["md"] >= cutoff]
        pairs = [("直调负荷实际", "直调负荷预测"), ("风电实际", "风电预测"), ("光伏实际", "光伏预测")]
        bad = []
        if len(recent):
            for actual, forecast in pairs:
                if actual in recent.columns and forecast in recent.columns:
                    valid = recent[actual].notna() & recent[forecast].notna()
                    if valid.any():
                        eq = (
                            pd.to_numeric(recent.loc[valid, actual], errors="coerce")
                            == pd.to_numeric(recent.loc[valid, forecast], errors="coerce")
                        ).mean() * 100
                        if eq > 20:
                            bad.append(f"{actual}=={forecast}:{eq:.1f}%")
            check(results, "96点新数据actual≠fcast(真实性)", not bad, "; ".join(bad) if bad else f"近30天actual/fcast独立({len(recent)}行)")
        else:
            check(results, "96点新数据actual≠fcast(真实性)", False, "近30天无数据")
        check(results, "96点canonical远程镜像存在", True, f"epf_pmos_96_full rows={len(mkt)}")
    else:
        check(results, "96点canonical远程镜像存在", False, "epf_pmos_96_full.parquet 缺失")

    # ── 3. 防泄漏规则检查（生产静态契约） ─────────────────────────────
    from utils.resolution import Resolution  # noqa: E402

    q = Resolution("15min", 96, 32, ("1_32", "33_64", "65_96"), "business_period", "15min", 15)
    check(results, "96点Resolution契约定义", q.slots_per_day == 96, f"slots={q.slots_per_day}")
    from utils.asof_view_96 import (  # noqa: E402
        DYNAMIC_PROTOCOL,
        SnapshotBuilder,
        build_dynamic_feature_view_96,
    )
    from pipelines.ledger_predict import FORMAL96_PREDICTION_CONTRACT  # noqa: E402

    check(
        results,
        "96点Dynamic serving协议一致",
        DYNAMIC_PROTOCOL == "formal96_dynamic_snapshot_v1"
        and FORMAL96_PREDICTION_CONTRACT == DYNAMIC_PROTOCOL,
        f"protocol={DYNAMIC_PROTOCOL}",
    )
    check(
        results,
        "96点信息边界由Snapshot/FeatureView统一",
        callable(SnapshotBuilder) and callable(build_dynamic_feature_view_96),
        "formal serving不再以固定15:00/p60作为可见性真源",
    )

    # ── 4. 账本可用性 ─────────────────────────────────────────────────
    def _prediction_days(root: Path) -> set[str]:
        days: set[str] = set()
        for task in ["dayahead", "realtime"]:
            pred = root / task / "prediction" / "prediction_ledger.parquet"
            if not pred.exists():
                continue
            try:
                pdf = pd.read_parquet(pred)
                day_col = next(
                    (c for c in ("target_day", "business_day", "market_date") if c in pdf.columns),
                    None,
                )
                if day_col:
                    days.update(pdf[day_col].astype(str).unique())
            except Exception:
                pass
        return days

    root24 = PROJECT_ROOT / "outputs" / "ledger"
    days24 = _prediction_days(root24)
    check(results, "24点production账本>=30天", len(days24) >= 30, f"distinct_days={len(days24)}")

    root96 = PROJECT_ROOT / "outputs" / "96" / "ledger"
    days96 = _prediction_days(root96)
    info(
        "96点production账本状态",
        f"distinct_days={len(days96)} root={root96}; prediction/backfill允许从0开始",
    )
    if require_96_full_chain:
        from fusion.model_pool import DAYAHEAD_MODELS, REALTIME_MODELS
        from pipelines.ledger_weight import select_complete_training_days
        from utils.resolution import QUARTER

        audit_target = target_date_96 or (max(days96) if days96 else None)
        if audit_target is None:
            check(results, "96点production账本>=30天(full-chain门禁)", False, "target_date unavailable")
        else:
            selections = {}
            for task, models in (("dayahead", DAYAHEAD_MODELS), ("realtime", REALTIME_MODELS)):
                selections[task] = select_complete_training_days(
                    task=task,
                    target_date=audit_target,
                    ledger_root=root96,
                    expected_models=list(models),
                    required_days=30,
                    max_lookback_days=90,
                    resolution=QUARTER,
                    history_lag_days=2,
                )
            ready = all(item.get("status") == "PASS" for item in selections.values())
            detail = "; ".join(
                f"{task}={item.get('selected_count', 0)}/30 anchor={item.get('anchor_start')}"
                for task, item in selections.items()
            )
            check(results, "96点production账本>=30天(full-chain门禁)", ready, f"target={audit_target}; {detail}")

    legacy96 = PROJECT_ROOT / "outputs" / "ledger_96"
    legacy_days96 = _prediction_days(legacy96)
    if legacy_days96:
        info("96点legacy账本(历史参考,不作production门禁)", f"distinct_days={len(legacy_days96)} root={legacy96}")

    # ── 5. 甲方96点全量数据 ───────────────────────────────────────────
    crawled96 = DATA.authoritative_96_actual_csv
    if crawled96.exists():
        c96 = pd.read_csv(crawled96, encoding="utf-8-sig")
        bad = []
        for col in ["直调负荷", "风电", "光伏", "外电", "地方电厂出力"]:
            forecast, actual = col + "预测", col + "实际"
            if forecast in c96.columns and actual in c96.columns:
                eq = (pd.to_numeric(c96[forecast], errors="coerce") == pd.to_numeric(c96[actual], errors="coerce")).mean() * 100
                if eq > 1:
                    bad.append(f"{col}:{eq:.1f}%")
        check(results, "甲方96点预测≠实际", not bad, "; ".join(bad) if bad else "预测/实际独立(0%)")
        days96, rows96 = c96["market_date"].nunique(), len(c96)
        check(results, "甲方96点网格完整", abs(rows96 / max(days96, 1) - 96) < 1, f"{rows96}行/{days96}天")
        closed_required = [
            "日前出清价格", "实时出清价格", "直调负荷实际", "地方电厂出力实际",
            "外电实际", "风电实际", "光伏实际", "核电实际", "自备电厂实际", "试验机组实际",
        ]
        closed_days = []
        for md, group in c96.groupby("market_date", sort=True):
            if len(group) == 96 and all(col in group.columns and group[col].notna().all() for col in closed_required):
                closed_days.append(str(md))
        check(results, "甲方96点存在闭合历史", bool(closed_days), f"latest_closed_day={closed_days[-1] if closed_days else None}")
        has_p96, has_p1 = (c96["时段"] == "24:00").any(), (c96["时段"] == "00:15").any()
        check(results, "甲方96点含p1/p96(业务时间完整)", has_p1 and has_p96, f"p1={has_p1} p96={has_p96}")
    else:
        check(results, "甲方96点全量数据存在", False, "data/pmos_96_全量.csv 缺失")

    return finish(results)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_spread24_historical_profile() -> int:
    """Check only the frozen Cycle88 hourly spread experiment contract."""
    results: list[tuple[str, str, str]] = []
    cycle = PROJECT_ROOT / "outputs" / "experiments" / "01_spread_24" / "spread_forecast_24_96_chain_v2" / "cycles" / "cycle_88_numeric_spread_da_minus_rt"
    cube = cycle / "feature_cubes" / "numeric_v2"
    slot_path = cube / "slot_table.parquet"
    groups_path = cube / "feature_groups.json"
    registry_path = cube / "feature_registry.json"
    manifest_path = cube / "manifest.json"

    check(results, "Cycle88 numeric_v2 frozen feature cube存在", cube.is_dir(), str(cube))
    check(results, "slot_table.parquet存在且非空", slot_path.exists() and slot_path.stat().st_size > 0, str(slot_path))
    check(results, "feature_groups.json存在", groups_path.exists(), str(groups_path))
    check(results, "feature_registry.json存在", registry_path.exists(), str(registry_path))
    if not (slot_path.exists() and groups_path.exists() and registry_path.exists()):
        return finish(results, "spread24-historical preflight")

    try:
        slot = pd.read_parquet(slot_path)
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
        registry = registry_payload["features"]
    except Exception as exc:
        check(results, "frozen cube文件可解析", False, repr(exc))
        return finish(results, "spread24-historical preflight")

    check(results, "slot_table.parquet非空", not slot.empty, f"rows={len(slot)}")
    slot["target_day"] = slot["target_day"].astype(str)
    counts = slot.groupby("target_day").size()
    complete_days = sorted(counts[counts.eq(24)].index.astype(str))
    m1_days = ["2026-06-01", "2026-06-02"]
    m2_days = [day for day in complete_days if day.startswith("2026-03-") or day.startswith("2026-05-")]
    check(results, "M1日期全部存在且每天24行", all(day in complete_days and int(counts.get(day, 0)) == 24 for day in m1_days), f"days={m1_days}")
    check(results, "M2日期覆盖2026-03和2026-05且完整", bool(m2_days) and all(int(counts.get(day, 0)) == 24 for day in m2_days), f"complete_days={len(m2_days)}")

    target = pd.to_numeric(slot.get("target_spread"), errors="coerce")
    check(results, "target_spread全部finite", bool(np_isfinite(target).all()), f"rows={len(target)}")
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    target_name = str(manifest.get("target_name", ""))
    check(results, "target定义为Cycle88 DA-RT", target_name == "spread_DA_minus_RT", f"target_name={target_name or 'missing'}")

    direction = pd.to_numeric(slot.get("target_direction"), errors="coerce")
    direction_ok = bool(np_isfinite(direction).all() and (direction.astype(int).to_numpy() == (target.to_numpy() > 0).astype(int)).all())
    check(results, "target_direction == (target_spread > 0)", direction_ok, "逐点一致" if direction_ok else "存在不一致")

    registry_names = [str(item.get("feature", "")) for item in registry]
    group_names = [str(name) for values in groups.values() for name in values]
    slot_columns = set(slot.columns)
    registry_missing = sorted(set(registry_names) - slot_columns)
    group_missing = sorted(set(group_names) - slot_columns)
    check(results, "feature registry里的feature存在于slot_table", not registry_missing, f"missing={registry_missing[:5]}")
    check(results, "feature_groups引用的feature存在于slot_table", not group_missing, f"missing={group_missing[:5]}")
    model_feature_names = sorted(set(registry_names) | set(group_names))
    forbidden = [name for name in model_feature_names if any(token in name.lower() for token in ("actual", "target_spread", "target_direction", "y_true"))]
    check(results, "模型输入feature名无直接泄漏字段", not forbidden, f"forbidden={forbidden[:5]}")

    if "context_source_max_ds" in slot.columns:
        source_ts = pd.to_datetime(slot["context_source_max_ds"], errors="coerce")
        target_ts = pd.to_datetime(slot["target_day"], errors="coerce")
        cutoff = target_ts - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        context_ok = bool((source_ts.isna() | (source_ts <= cutoff)).all())
        detail = f"max_violation={source_ts[source_ts > cutoff].max()}" if not context_ok else "all source timestamps <= D-1 14:00"
        check(results, "context_source_max_ds <= target_day D-1 14:00", context_ok, detail)
    else:
        check(results, "context_source_max_ds <= target_day D-1 14:00", False, "column missing")

    holdout = [day for day in m1_days + m2_days if "2026-08-15" <= day <= "2026-08-21"]
    check(results, "final holdout 2026-08-15..21未作为evaluation target", not holdout, f"blocked_targets={holdout}")

    try:
        focus_common = load_module(cycle / "focus_common.py", "cycle88_focus_common_scoped_preflight")
        strict_train_days = focus_common.strict_train_days
        check(results, "strict_train_days helper可导入", callable(strict_train_days), str(cycle / "focus_common.py"))
        train_audit = []
        for day in m1_days + m2_days:
            train_days = strict_train_days(complete_days, day, 180)
            latest = (pd.Timestamp(day) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            train_audit.append((day, train_days[-1], latest, train_days[-1] <= latest))
        bad_train = [row for row in train_audit if not row[3]]
        check(results, "M1/M2 strict_train_days训练边界 <= D-2", not bad_train, f"bad={bad_train[:3]}")
    except Exception as exc:
        check(results, "strict_train_days helper可导入", False, repr(exc))
        check(results, "M1/M2 strict_train_days训练边界 <= D-2", False, "helper调用失败")

    return finish(results, "spread24-historical preflight")


def np_isfinite(series: pd.Series) -> pd.Series:
    import numpy as np

    return pd.Series(np.isfinite(series.to_numpy(float)), index=series.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("production", "spread24-historical"), default="production")
    parser.add_argument(
        "--require-96-full-chain",
        action="store_true",
        help="Require 30 causally complete formal96 history days before weight/fuse/final.",
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="Formal96 forecast target D for the full-chain readiness selector.",
    )
    args = parser.parse_args()
    if args.profile == "spread24-historical":
        raise SystemExit(run_spread24_historical_profile())
    raise SystemExit(
        run_production_profile(
            require_96_full_chain=args.require_96_full_chain,
            target_date_96=args.target_date,
        )
    )


if __name__ == "__main__":
    main()
