from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from rt916_spikefusionnet import core
from rt916_spikefusionnet.dataprocess import (
    enrich_selected_features,
    feature_engineer_solar_terms,
    process_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "RT916_SpikeFusionNet"
PACKAGE_OUT_ROOT = PROJECT_ROOT / "outputs" / "RT916_SpikeMarketLab" / "model_packages" / "RT916_SpikeFusionNet"
LAB_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "RT916_SpikeMarketLab"
PROFILE = json.loads((PACKAGE_ROOT / "configs" / "release_safe_profile.json").read_text(encoding="utf-8"))
HISTORICAL_VALIDATION_ROOT = PROJECT_ROOT / PROFILE["release_safe"]["historical_validation_root"]
DATA_PATH = PROJECT_ROOT / "data" / "24" / "canonical" / "shandong_pmos_hourly.xlsx"

TIME_COL = "时刻"
DAYAHEAD_COL = "日前电价"
DAYAHEAD_PRED_COL = "预测日前电价"
REALTIME_COL = "实时电价"
REALTIME_PRED_COL = "预测实时电价"
TARGET_COLUMNS = {
    "dayahead": {"actual": DAYAHEAD_COL, "pred": DAYAHEAD_PRED_COL},
    "realtime": {"actual": REALTIME_COL, "pred": REALTIME_PRED_COL},
}
ALLOWED_RELEASE_MONTHS = {
    "dayahead": [str(m) for m in pd.period_range("2025-01", "2025-12", freq="M")]
    + [str(m) for m in pd.period_range("2026-01", "2026-05", freq="M")],
    "realtime": [str(m) for m in pd.period_range("2025-01", "2025-12", freq="M")],
}


def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{context} missing required columns: {missing}")


def capped_smape_series(actual: pd.Series, pred: pd.Series, floor: float = 50.0) -> pd.Series:
    actual_clipped = pd.to_numeric(actual, errors="coerce").clip(lower=floor)
    pred_clipped = pd.to_numeric(pred, errors="coerce").clip(lower=floor)
    denom = (actual_clipped.abs() + pred_clipped.abs()) / 2.0
    return (pred_clipped.sub(actual_clipped).abs() / denom) * 100.0


def month_bounds(month_label: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month_label}-01 01:00:00")
    end = ((start - pd.Timedelta(hours=1)) + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=24)
    return start, end


def month_labels_covering(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[str]:
    months = pd.period_range(start=start_ts.to_period("M"), end=end_ts.to_period("M"), freq="M")
    return [str(month) for month in months]


def validate_release_bundle_window(target: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[str]:
    months = month_labels_covering(start_ts, end_ts)
    allowed = set(ALLOWED_RELEASE_MONTHS[target])
    unsupported = [month for month in months if month not in allowed]
    if unsupported:
        raise ValueError(
            f"release_bundle target={target} only supports months {sorted(allowed)}; "
            f"unsupported months requested: {unsupported}"
        )
    return months


def read_month_predictions(experiment_id: str, month_label: str) -> pd.DataFrame:
    annual_path = LAB_OUTPUT_ROOT / "experiments" / experiment_id / "all_predictions_9_16.csv"
    if annual_path.exists():
        df = pd.read_csv(annual_path)
        if "month" in df.columns:
            df = df[df["month"] == month_label].copy()
    else:
        pred_path = LAB_OUTPUT_ROOT / "experiments" / experiment_id / "predictions" / f"{month_label}_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        df = pd.read_csv(pred_path)
    require_columns(df, [TIME_COL], f"prediction source {experiment_id} month={month_label}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def load_fixed_stage_predictions(target: str, month_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target == "dayahead":
        stage1_exp = PROFILE["dayahead"]["stage1_experiment"]
        stage2_exp = PROFILE["dayahead"]["stage2_experiment"]
        if month_label.startswith("2025-"):
            return read_month_predictions(stage1_exp, month_label), read_month_predictions(stage2_exp, month_label)
        if month_label.startswith("2026-"):
            eval_root = HISTORICAL_VALIDATION_ROOT / "eval_2026_01_05"
            stage1 = pd.read_csv(eval_root / "dayahead_stage1_2026_01_05.csv")
            stage2 = pd.read_csv(eval_root / "dayahead_stage2_2026_01_05.csv")
            for df in (stage1, stage2):
                require_columns(df, [TIME_COL], f"historical validation month={month_label}")
                df[TIME_COL] = pd.to_datetime(df[TIME_COL])
                df["month"] = df[TIME_COL].dt.strftime("%Y-%m")
            return (
                stage1[stage1["month"] == month_label].copy(),
                stage2[stage2["month"] == month_label].copy(),
            )
        raise ValueError(f"Unsupported day-ahead release bundle month: {month_label}")

    if target == "realtime":
        stage1_exp = PROFILE["realtime"]["stage1_experiment"]
        stage2_exp = PROFILE["realtime"]["stage2_experiment"]
        if month_label.startswith("2025-"):
            return read_month_predictions(stage1_exp, month_label), read_month_predictions(stage2_exp, month_label)
        raise ValueError(f"Unsupported realtime release bundle month: {month_label}")

    raise ValueError(f"Unsupported target: {target}")


def load_fixed_stage3_predictions(target: str, month_label: str) -> pd.DataFrame:
    if target == "realtime":
        return read_month_predictions(PROFILE["realtime"]["stage3_experiment"], month_label)
    if target == "dayahead":
        if month_label.startswith("2025-"):
            return read_month_predictions(PROFILE["dayahead"]["stage3_base_experiment"], month_label)
        if month_label.startswith("2026-"):
            eval_root = HISTORICAL_VALIDATION_ROOT / "eval_2026_01_05"
            df = pd.read_csv(eval_root / "dayahead_stage3_2026_01_05.csv")
            require_columns(df, [TIME_COL], f"historical stage3 validation month={month_label}")
            df[TIME_COL] = pd.to_datetime(df[TIME_COL])
            df["month"] = df[TIME_COL].dt.strftime("%Y-%m")
            return df[df["month"] == month_label].copy()
    raise ValueError(f"Unsupported target/month combination: target={target}, month={month_label}")


def prepare_processed_df(target_col: str = DAYAHEAD_COL) -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    df = process_features(df)
    df = feature_engineer_solar_terms(df)
    df = enrich_selected_features(df, target_col=target_col)
    require_columns(df, [TIME_COL, target_col], f"processed dataset target={target_col}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df.sort_values(TIME_COL).reset_index(drop=True)


@contextmanager
def temp_env(env_map: dict[str, str | None]):
    old_env = {key: os.environ.get(key) for key in env_map}
    try:
        for key, value in env_map.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def feature_env_for_variant(variant: str) -> dict[str, str | None]:
    if variant == "base":
        return {
            "SPIKE_DA_INPUT_VARIANT": "base",
            "SPIKE_DA_STAGE1_BACKBONE": None,
            "SPIKE_DA_PERIOD_BACKBONE": "base",
        }
    if variant == "timemixer":
        return {
            "SPIKE_DA_INPUT_VARIANT": "base",
            "SPIKE_DA_STAGE1_BACKBONE": None,
            "SPIKE_DA_PERIOD_BACKBONE": "timemixer",
        }
    raise ValueError(f"Unknown variant: {variant}")


def configure_core_window(
    target: str,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    run_id: str,
    artifact_root: Path,
    output_root: Path,
) -> None:
    core.CONFIG["RUN_ID"] = run_id
    core._update_config(target, [str(pred_start), str(pred_end)])
    core.CONFIG["SAVE_ROOT_DIR"] = artifact_root / run_id
    core.CONFIG["PREDICT_RESULT_DIR"] = output_root / run_id
    Path(core.CONFIG["SAVE_ROOT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(core.CONFIG["PREDICT_RESULT_DIR"]).mkdir(parents=True, exist_ok=True)


def run_train_predict_window(
    processed_df: pd.DataFrame,
    target: str,
    mod: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    env_map: dict[str, str | None],
    run_id: str,
    artifact_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    with temp_env(env_map):
        configure_core_window(target, pred_start, pred_end, run_id, artifact_root, output_root)
        train_df = processed_df[(processed_df[TIME_COL] >= train_start) & (processed_df[TIME_COL] <= train_end)].copy()
        if train_df.empty:
            raise RuntimeError(f"Empty train window: {train_start} -> {train_end}")
        core.train(train_df, mod=mod)

        pred_days = pd.date_range(pred_start.normalize(), pred_end.normalize(), freq="D")
        all_results: list[pd.DataFrame] = []
        for pred_day in pred_days:
            day_start = pred_day + pd.Timedelta(hours=1)
            day_end = pred_day + pd.Timedelta(days=1)
            if day_start < pred_start or day_end > pred_end:
                continue
            asof_ts = pred_day - pd.Timedelta(days=1) + pd.Timedelta(hours=15)
            window_start = day_start - pd.Timedelta(days=core.CONFIG["INPUT_LEN_LIST"])
            test_data = processed_df[(processed_df[TIME_COL] >= window_start) & (processed_df[TIME_COL] <= day_end)].copy()
            core.CONFIG["TEST_TOTAL_START_END_LIST"] = [str(day_start), str(day_end)]
            one_day = core.inference(test_data, mod=mod, asof_ts=asof_ts, external_da_pred_df=None)
            one_day = one_day[(one_day[TIME_COL] >= day_start) & (one_day[TIME_COL] <= day_end)].copy()
            all_results.append(one_day)

        if not all_results:
            raise RuntimeError(f"No predictions for window {pred_start} -> {pred_end}")
        result = pd.concat(all_results, ignore_index=True)
        return result.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last").reset_index(drop=True)


def cache_key(variant: str, month_label: str, tag: str, train_start: pd.Timestamp, train_end: pd.Timestamp) -> str:
    return (
        f"{variant}_{month_label}_{tag}_train_"
        f"{train_start.strftime('%Y%m%d%H')}_{train_end.strftime('%Y%m%d%H')}"
    )


def get_cached_or_run_stage3(
    processed_df: pd.DataFrame,
    variant: str,
    month_label: str,
    tag: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    pred_start: pd.Timestamp,
    pred_end: pd.Timestamp,
    force: bool = False,
) -> pd.DataFrame:
    cache_root = PACKAGE_OUT_ROOT / "_cache" / "release_policy"
    cache_root.mkdir(parents=True, exist_ok=True)
    key = cache_key(variant, month_label, tag, train_start, train_end)
    cache_file = cache_root / f"{key}.csv"
    if cache_file.exists() and not force:
        df = pd.read_csv(cache_file)
        require_columns(df, [TIME_COL], f"stage3 cache {key}")
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
        return df

    df = run_train_predict_window(
        processed_df=processed_df,
        target=DAYAHEAD_COL,
        mod="stage3",
        train_start=train_start,
        train_end=train_end,
        pred_start=pred_start,
        pred_end=pred_end,
        env_map=feature_env_for_variant(variant),
        run_id=key,
        artifact_root=PACKAGE_OUT_ROOT / "artifacts" / "release_policy_stage3",
        output_root=PACKAGE_OUT_ROOT / "_predict_cache" / "release_policy_stage3",
    )
    df.to_csv(cache_file, index=False, encoding="utf-8-sig")
    return df


def stage3_hour_mask(ts: pd.Series) -> pd.Series:
    return ts.dt.hour.isin([17, 18, 19, 20, 21, 22, 23, 0])


def assemble_month_full_day(target: str, month_label: str, stage3_df: pd.DataFrame) -> pd.DataFrame:
    stage1_df, stage2_df = load_fixed_stage_predictions(target, month_label)
    target_col = TARGET_COLUMNS[target]["actual"]
    pred_col = TARGET_COLUMNS[target]["pred"]
    for label, frame in [("stage1", stage1_df), ("stage2", stage2_df), ("stage3", stage3_df)]:
        require_columns(frame, [TIME_COL, target_col, pred_col], f"{target} {label} month={month_label}")

    stage1_df = stage1_df[[TIME_COL, target_col, pred_col]].copy()
    stage2_df = stage2_df[[TIME_COL, target_col, pred_col]].copy()
    stage3_df = stage3_df[[TIME_COL, target_col, pred_col]].copy()
    stage1_df["period"] = "stage1"
    stage2_df["period"] = "stage2"
    stage3_df["period"] = "stage3"

    combo = pd.concat([stage1_df, stage2_df, stage3_df], ignore_index=True)
    combo["month"] = month_label
    combo[TIME_COL] = pd.to_datetime(combo[TIME_COL])
    combo = combo.sort_values(TIME_COL).reset_index(drop=True)
    combo["smape"] = capped_smape_series(combo[target_col], combo[pred_col])
    return combo


def month_stage3_metrics(stage3_df: pd.DataFrame) -> dict[str, float]:
    require_columns(stage3_df, [TIME_COL, DAYAHEAD_COL, DAYAHEAD_PRED_COL], "stage3 evaluation frame")
    out = stage3_df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL])
    out = out[stage3_hour_mask(out[TIME_COL])].copy()
    out["smape"] = capped_smape_series(out[DAYAHEAD_COL], out[DAYAHEAD_PRED_COL])
    by_day = out.groupby(out[TIME_COL].dt.normalize())["smape"].mean().reset_index(name="day_smape")
    return {"stage3_smape": float(out["smape"].mean()), "stage3_days": int(len(by_day))}


def full_day_metrics(full_day_df: pd.DataFrame) -> dict[str, float]:
    return {
        "full_day_smape": float(full_day_df["smape"].mean()),
        "stage1_smape": float(full_day_df.loc[full_day_df["period"] == "stage1", "smape"].mean()),
        "stage2_smape": float(full_day_df.loc[full_day_df["period"] == "stage2", "smape"].mean()),
        "stage3_full_smape": float(full_day_df.loc[full_day_df["period"] == "stage3", "smape"].mean()),
    }


def daily_smape_table(full_day_df: pd.DataFrame) -> pd.DataFrame:
    out = full_day_df.copy()
    out["date"] = out[TIME_COL].dt.normalize()
    return (
        out.groupby("date")
        .agg(
            full_day_smape=("smape", "mean"),
            stage3_smape=("smape", lambda s: float(s[out.loc[s.index, "period"] == "stage3"].mean())),
        )
        .reset_index()
    )


def make_val_summary(
    month_label: str,
    val_month_label: str | None,
    base_eval_df: pd.DataFrame,
    timemixer_eval_df: pd.DataFrame,
    base_val_df: pd.DataFrame | None,
    timemixer_val_df: pd.DataFrame | None,
) -> dict[str, object]:
    base_eval_full = assemble_month_full_day("dayahead", month_label, base_eval_df)
    timemixer_eval_full = assemble_month_full_day("dayahead", month_label, timemixer_eval_df)
    base_eval_metrics = {**full_day_metrics(base_eval_full), **month_stage3_metrics(base_eval_df)}
    timemixer_eval_metrics = {**full_day_metrics(timemixer_eval_full), **month_stage3_metrics(timemixer_eval_df)}

    row: dict[str, object] = {
        "eval_month": month_label,
        "val_month": val_month_label,
        "eval_base_full_day_smape": base_eval_metrics["full_day_smape"],
        "eval_tm_full_day_smape": timemixer_eval_metrics["full_day_smape"],
        "eval_base_stage3_smape": base_eval_metrics["stage3_smape"],
        "eval_tm_stage3_smape": timemixer_eval_metrics["stage3_smape"],
    }

    if val_month_label is None or base_val_df is None or timemixer_val_df is None:
        row.update(
            {
                "val_base_full_day_smape": float("nan"),
                "val_tm_full_day_smape": float("nan"),
                "val_base_stage3_smape": float("nan"),
                "val_tm_stage3_smape": float("nan"),
                "val_delta": float("nan"),
                "val_full_day_delta": float("nan"),
                "improved_days": 0,
                "harmed_days": 0,
                "worst_val_period_delta": float("nan"),
                "rule_selected_policy": "base",
                "rule_reason": "cold_start_no_prior_val_month",
            }
        )
        return row

    base_val_full = assemble_month_full_day("dayahead", val_month_label, base_val_df)
    timemixer_val_full = assemble_month_full_day("dayahead", val_month_label, timemixer_val_df)
    base_val_metrics = {**full_day_metrics(base_val_full), **month_stage3_metrics(base_val_df)}
    timemixer_val_metrics = {**full_day_metrics(timemixer_val_full), **month_stage3_metrics(timemixer_val_df)}
    base_days = daily_smape_table(base_val_full).rename(columns={"stage3_smape": "base_day_stage3"})
    timemixer_days = daily_smape_table(timemixer_val_full).rename(columns={"stage3_smape": "tm_day_stage3"})
    day_cmp = base_days.merge(timemixer_days, on="date", how="inner")

    improved_days = int((day_cmp["tm_day_stage3"] < day_cmp["base_day_stage3"]).sum())
    harmed_days = int((day_cmp["tm_day_stage3"] > day_cmp["base_day_stage3"]).sum())
    stage3_delta = base_val_metrics["stage3_smape"] - timemixer_val_metrics["stage3_smape"]
    full_day_delta = base_val_metrics["full_day_smape"] - timemixer_val_metrics["full_day_smape"]
    worst_val_period_delta = min(
        base_val_metrics["stage1_smape"] - timemixer_val_metrics["stage1_smape"],
        base_val_metrics["stage2_smape"] - timemixer_val_metrics["stage2_smape"],
        base_val_metrics["stage3_full_smape"] - timemixer_val_metrics["stage3_full_smape"],
    )
    selected_policy = (
        "timemixer"
        if stage3_delta >= 0.3 and harmed_days <= improved_days and worst_val_period_delta >= -1.0 and full_day_delta >= -0.3
        else "base"
    )

    row.update(
        {
            "val_base_full_day_smape": base_val_metrics["full_day_smape"],
            "val_tm_full_day_smape": timemixer_val_metrics["full_day_smape"],
            "val_base_stage3_smape": base_val_metrics["stage3_smape"],
            "val_tm_stage3_smape": timemixer_val_metrics["stage3_smape"],
            "val_delta": stage3_delta,
            "val_full_day_delta": full_day_delta,
            "improved_days": improved_days,
            "harmed_days": harmed_days,
            "worst_val_period_delta": worst_val_period_delta,
            "rule_selected_policy": selected_policy,
            "rule_reason": "timemixer_passed_val_guardrails" if selected_policy == "timemixer" else "base_kept_due_to_guardrail",
        }
    )
    return row


def build_release_safe_month_tables(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    processed_df = prepare_processed_df(target_col=DAYAHEAD_COL)
    eval_months = validate_release_bundle_window("dayahead", start_ts, end_ts)
    stage3_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for month_label in eval_months:
        eval_start, eval_end = month_bounds(month_label)
        train_start = eval_start - pd.DateOffset(months=12)
        train_end_eval = eval_start - pd.Timedelta(seconds=1)

        base_eval_df = get_cached_or_run_stage3(
            processed_df,
            "base",
            month_label,
            "eval",
            train_start,
            train_end_eval,
            eval_start,
            eval_end,
            force=force_recompute,
        )
        timemixer_eval_df = get_cached_or_run_stage3(
            processed_df,
            "timemixer",
            month_label,
            "eval",
            train_start,
            train_end_eval,
            eval_start,
            eval_end,
            force=force_recompute,
        )
        stage3_cache[f"base_eval_{month_label}"] = base_eval_df
        stage3_cache[f"timemixer_eval_{month_label}"] = timemixer_eval_df

        if month_label == "2025-01":
            rows.append(make_val_summary(month_label, None, base_eval_df, timemixer_eval_df, None, None))
            continue

        val_month_label = (eval_start - pd.offsets.MonthBegin(1)).strftime("%Y-%m")
        val_start, val_end = month_bounds(val_month_label)
        train_end_val = val_start - pd.Timedelta(seconds=1)
        base_val_df = get_cached_or_run_stage3(
            processed_df,
            "base",
            month_label,
            "val",
            train_start,
            train_end_val,
            val_start,
            val_end,
            force=force_recompute,
        )
        timemixer_val_df = get_cached_or_run_stage3(
            processed_df,
            "timemixer",
            month_label,
            "val",
            train_start,
            train_end_val,
            val_start,
            val_end,
            force=force_recompute,
        )
        stage3_cache[f"base_val_{month_label}"] = base_val_df
        stage3_cache[f"timemixer_val_{month_label}"] = timemixer_val_df
        rows.append(make_val_summary(month_label, val_month_label, base_eval_df, timemixer_eval_df, base_val_df, timemixer_val_df))

    return pd.DataFrame(rows), stage3_cache


def summarize_policy_months(summary_df: pd.DataFrame, stage3_cache: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_frames: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, object]] = []
    for row in summary_df.itertuples(index=False):
        selected_policy = row.rule_selected_policy
        stage3_key = f"{selected_policy}_eval_{row.eval_month}" if selected_policy == "timemixer" else f"base_eval_{row.eval_month}"
        full_df = assemble_month_full_day("dayahead", row.eval_month, stage3_cache[stage3_key])
        full_df["selected_policy"] = selected_policy
        annual_frames.append(full_df)
        monthly_rows.append(
            {
                "month": row.eval_month,
                "selected_policy": selected_policy,
                "full_day_smape": float(full_df["smape"].mean()),
                "stage3_smape": float(full_df.loc[full_df["period"] == "stage3", "smape"].mean()),
            }
        )
    return pd.concat(annual_frames, ignore_index=True), pd.DataFrame(monthly_rows)


def assemble_realtime_release_bundle(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_frames: list[pd.DataFrame] = []
    monthly_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    for month_label in validate_release_bundle_window("realtime", start_ts, end_ts):
        stage3_df = load_fixed_stage3_predictions("realtime", month_label)
        full_df = assemble_month_full_day("realtime", month_label, stage3_df)
        full_df = full_df[(full_df[TIME_COL] >= start_ts) & (full_df[TIME_COL] <= end_ts)].copy()
        annual_frames.append(full_df)
        monthly_rows.append(
            {
                "month": month_label,
                "full_day_smape": float(full_df["smape"].mean()),
                "stage1_smape": float(full_df.loc[full_df["period"] == "stage1", "smape"].mean()),
                "stage2_smape": float(full_df.loc[full_df["period"] == "stage2", "smape"].mean()),
                "stage3_smape": float(full_df.loc[full_df["period"] == "stage3", "smape"].mean()),
            }
        )
        source_rows.extend(
            [
                {"month": month_label, "target": "realtime", "period": "stage1", "source": PROFILE["realtime"]["stage1_experiment"], "policy": "fixed"},
                {"month": month_label, "target": "realtime", "period": "stage2", "source": PROFILE["realtime"]["stage2_experiment"], "policy": "fixed"},
                {"month": month_label, "target": "realtime", "period": "stage3", "source": PROFILE["realtime"]["stage3_experiment"], "policy": "fixed"},
            ]
        )

    return pd.concat(annual_frames, ignore_index=True), pd.DataFrame(monthly_rows), pd.DataFrame(source_rows)


def assemble_dayahead_release_bundle(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    requested_months = validate_release_bundle_window("dayahead", start_ts, end_ts)
    summary_df, stage3_cache = build_release_safe_month_tables(start_ts, end_ts, force_recompute=force_recompute)
    annual_df, monthly_df = summarize_policy_months(summary_df, stage3_cache)
    annual_df = annual_df[(annual_df[TIME_COL] >= start_ts) & (annual_df[TIME_COL] <= end_ts)].copy()
    monthly_df = monthly_df[monthly_df["month"].isin(requested_months)].copy()

    source_rows: list[dict[str, object]] = []
    for row in summary_df.itertuples(index=False):
        source_rows.extend(
            [
                {"month": row.eval_month, "target": "dayahead", "period": "stage1", "source": PROFILE["dayahead"]["stage1_experiment"], "policy": "fixed"},
                {"month": row.eval_month, "target": "dayahead", "period": "stage2", "source": PROFILE["dayahead"]["stage2_experiment"], "policy": "fixed"},
                {
                    "month": row.eval_month,
                    "target": "dayahead",
                    "period": "stage3",
                    "source": PROFILE["dayahead"]["stage3_release_policy_experiment"],
                    "policy": row.rule_selected_policy,
                },
            ]
        )

    return annual_df, monthly_df, summary_df, pd.DataFrame(source_rows)


def write_release_bundle(
    target: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    annual_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    source_df: pd.DataFrame,
    policy_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    tag = f"{start_ts.strftime('%Y%m%d%H')}_{end_ts.strftime('%Y%m%d%H')}"
    out_dir = PACKAGE_OUT_ROOT / "predictions" / "release_bundle" / target / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    annual_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(out_dir / "monthly_summary.csv", index=False, encoding="utf-8-sig")
    source_df.to_csv(out_dir / "source_manifest.csv", index=False, encoding="utf-8-sig")
    if policy_df is not None:
        policy_df.to_csv(out_dir / "policy_selection.csv", index=False, encoding="utf-8-sig")

    summary = {
        "target": target,
        "start": str(start_ts),
        "end": str(end_ts),
        "rows": int(len(annual_df)),
        "avg_smape": float(annual_df["smape"].mean()) if len(annual_df) else None,
        "output_dir": str(out_dir),
        "best_policy": PROFILE["release_safe"]["best_policy"],
    }
    (out_dir / "bundle_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "predictions": str(out_dir / "predictions.csv"),
        "summary": str(out_dir / "bundle_summary.json"),
    }
