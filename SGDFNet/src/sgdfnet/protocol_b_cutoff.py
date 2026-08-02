from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data_contract import (
    ACTUAL_COLS,
    ACTUAL_TO_FORECAST_MAP,
    DA_COL,
    FeatureConfig,
    RT_COL,
    TIMESTAMP_COL,
    add_business_time_columns,
    build_feature_manifest,
    load_dataset,
    preprocess_dataframe,
)
from .metrics import build_metrics_frame, build_segment_metrics, build_tail_metrics
from .models import DeltaRegressor, HGBModelConfig, SegmentConditionedDeltaRegressor
from .protocol_b import (
    _apply_calibration_map,
    _build_error_gate_bias_map,
    _build_hour_bias_map,
    _build_month_bucket_hour_bias_map,
    _build_segment_bias_map,
    _build_segment_hour_bias_map,
    _build_segment_hour_sign_bias_map,
    _build_train_sample_weight,
    _restrict_calibration_window,
    ProtocolBConfig,
    load_protocol_b_config,
)
from .runtime_helpers import build_regressor, score_prediction_frame


@dataclass
class ProtocolBCutoffConfig:
    experiment_name: str
    data_path: str
    output_root: str
    start_day: str
    end_day: str
    decision_hour: int = 15
    val_days: int = 30
    resolution: int = 24  # 24=hourly, 96=15min
    train_min_rows: int = 24 * 90
    da_fill_mode: str = "raw_da"
    da_fill_bias_source: str = "val"
    apply_segment_bias_calibration: bool = False
    calibration_segments: list[str] | None = None
    calibration_mode: str = "segment_bias"
    calibration_threshold_quantile: float | None = None
    calibration_recent_days: int | None = None
    calibration_reducer: str = "median"
    calibration_residual_quantile: float | None = None
    calibration_hour_scope: str = "all"
    calibration_month_bucket_mode: str = "season"
    calibration_error_quantile: float = 0.8
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_config: HGBModelConfig = field(default_factory=HGBModelConfig)


def load_protocol_b_cutoff_config(path: str | Path) -> ProtocolBCutoffConfig:
    base = load_protocol_b_config(path)
    raw = json.loads(json.dumps(asdict(base), ensure_ascii=False))
    with open(path, "r", encoding="utf-8") as f:
        import yaml

        cfg = yaml.safe_load(f)
    return ProtocolBCutoffConfig(
        experiment_name=cfg.get("experiment_name", raw["experiment_name"]),
        data_path=cfg.get("data_path", raw["data_path"]),
        output_root=cfg.get("output_root", raw["output_root"]),
        start_day=cfg["start_day"],
        end_day=cfg["end_day"],
        decision_hour=cfg.get("decision_hour", 15),
        val_days=cfg.get("val_days", raw["val_days"]),
        resolution=cfg.get("resolution", 24),
        train_min_rows=cfg.get("train_min_rows", raw["train_min_rows"]),
        da_fill_mode=cfg.get("da_fill_mode", "raw_da"),
        da_fill_bias_source=cfg.get("da_fill_bias_source", "val"),
        apply_segment_bias_calibration=cfg.get("apply_segment_bias_calibration", raw["apply_segment_bias_calibration"]),
        calibration_segments=cfg.get("calibration_segments", raw.get("calibration_segments", [])),
        calibration_mode=cfg.get("calibration_mode", raw["calibration_mode"]),
        calibration_threshold_quantile=cfg.get("calibration_threshold_quantile", raw.get("calibration_threshold_quantile")),
        calibration_recent_days=cfg.get("calibration_recent_days", raw.get("calibration_recent_days")),
        calibration_reducer=cfg.get("calibration_reducer", raw.get("calibration_reducer", "median")),
        calibration_residual_quantile=cfg.get("calibration_residual_quantile", raw.get("calibration_residual_quantile")),
        calibration_hour_scope=cfg.get("calibration_hour_scope", raw.get("calibration_hour_scope", "all")),
        calibration_month_bucket_mode=cfg.get("calibration_month_bucket_mode", raw.get("calibration_month_bucket_mode", "season")),
        calibration_error_quantile=cfg.get("calibration_error_quantile", raw.get("calibration_error_quantile", 0.8)),
        feature_config=FeatureConfig(**cfg.get("feature_config", raw["feature_config"])),
        model_config=HGBModelConfig(**cfg.get("model_config", raw["model_config"])),
    )


def _decision_timestamp(decision_day: pd.Timestamp, decision_hour: int) -> pd.Timestamp:
    return pd.Timestamp(decision_day).normalize() + pd.Timedelta(hours=decision_hour)


def _build_da_fill_bias_map(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    decision_day: pd.Timestamp,
    config: ProtocolBCutoffConfig,
) -> dict[str, float]:
    if config.da_fill_mode == "raw_da":
        return {}

    if config.da_fill_bias_source == "train":
        source = train_df.copy()
    elif config.da_fill_bias_source == "train_val":
        source = pd.concat([train_df, val_df], ignore_index=False)
    else:
        source = val_df.copy()

    if source.empty:
        return {}

    source = source.copy()
    source["hour"] = source["hour"].astype(int)
    # decision 后的槽：24 点 decision_hour=15 → blocked 16..24；96 点 → 决策槽后到 96
    _decision_period = config.decision_hour * (config.resolution // 24)
    blocked_hours = list(range(_decision_period + 1, config.resolution + 1))
    source = source[source["hour"].isin(blocked_hours)]
    if source.empty:
        return {}

    source["_fill_delta"] = source["delta_target"]
    if config.da_fill_mode == "global_delta_bias":
        return {"global": float(source["_fill_delta"].median())}
    if config.da_fill_mode == "segment_delta_bias":
        return {
            str(segment): float(seg_df["_fill_delta"].median())
            for segment, seg_df in source.groupby("segment")
            if not seg_df.empty
        }
    return {}


def _apply_da_fill_bias(
    visible: pd.DataFrame,
    blocked_same_day_mask: pd.Series,
    bias_map: dict[str, float],
    da_fill_mode: str,
) -> None:
    if da_fill_mode == "global_delta_bias":
        bias = float(bias_map.get("global", 0.0))
        visible.loc[blocked_same_day_mask, "visible_rt_anchor"] = (
            pd.to_numeric(visible.loc[blocked_same_day_mask, "visible_rt_anchor"], errors="coerce") + bias
        )
    elif da_fill_mode == "segment_delta_bias":
        segment_bias = visible.loc[blocked_same_day_mask, "segment"].map(bias_map).fillna(0.0)
        visible.loc[blocked_same_day_mask, "visible_rt_anchor"] = (
            pd.to_numeric(visible.loc[blocked_same_day_mask, "visible_rt_anchor"], errors="coerce")
            + segment_bias.to_numpy(dtype=float)
        )


def _build_protocol_b_visible_frame(
    raw_df: pd.DataFrame,
    decision_day: pd.Timestamp,
    decision_hour: int,
    da_fill_mode: str = "raw_da",
    da_fill_bias_map: dict[str, float] | None = None,
    resolution: int = 24,
) -> pd.DataFrame:
    visible = raw_df.copy()
    visible = add_business_time_columns(visible, TIMESTAMP_COL, resolution=resolution)
    visible["hour"] = visible["target_hour"].astype(int)
    pp = resolution // 3
    _bins = [0, 8, 16, 24] if resolution == 24 else [0, pp, 2 * pp, resolution]
    _labels = ["1_8", "9_16", "17_24"] if resolution == 24 else [
        "1_%d" % pp, "%d_%d" % (pp + 1, 2 * pp), "%d_%d" % (2 * pp + 1, resolution)
    ]
    visible["segment"] = pd.cut(
        visible["hour"],
        bins=_bins,
        labels=_labels,
        include_lowest=True,
        right=True,
    ).astype(str)
    decision_ts = _decision_timestamp(decision_day, decision_hour)

    visible["visible_rt_anchor"] = pd.to_numeric(visible[RT_COL], errors="coerce")
    current_day_mask = visible["business_day"] == pd.Timestamp(decision_day).normalize()
    blocked_same_day_mask = current_day_mask & (visible[TIMESTAMP_COL] > decision_ts)
    visible.loc[blocked_same_day_mask, "visible_rt_anchor"] = pd.to_numeric(
        visible.loc[blocked_same_day_mask, DA_COL], errors="coerce"
    )
    _apply_da_fill_bias(visible, blocked_same_day_mask, da_fill_bias_map or {}, da_fill_mode)

    for actual_col, forecast_col in ACTUAL_TO_FORECAST_MAP.items():
        visible_col = f"visible_{actual_col}"
        visible[visible_col] = pd.to_numeric(visible[actual_col], errors="coerce")
        visible.loc[blocked_same_day_mask, visible_col] = pd.to_numeric(visible.loc[blocked_same_day_mask, forecast_col], errors="coerce")

    return visible


def _build_training_frame(raw_df: pd.DataFrame, feature_config: FeatureConfig, resolution: int = 24) -> tuple[pd.DataFrame, list[str]]:
    return preprocess_dataframe(raw_df, feature_config, resolution=resolution)


def _build_inference_frame(
    visible_df: pd.DataFrame,
    feature_config: FeatureConfig,
    resolution: int = 24,
) -> tuple[pd.DataFrame, list[str]]:
    if (feature_config.include_actual_history_columns or feature_config.include_forecast_residual_history_features) and not feature_config.use_visible_actual_history:
        raise ValueError(
            "Cutoff-safe inference requires use_visible_actual_history=true when actual-side or residual-history features are enabled."
        )
    actual_history_map = None
    if feature_config.use_visible_actual_history:
        actual_history_map = {col: f"visible_{col}" for col in ACTUAL_COLS}
    return preprocess_dataframe(
        visible_df,
        feature_config,
        rt_history_col="visible_rt_anchor",
        actual_history_source_map=actual_history_map,
        resolution=resolution,
    )


def _build_train_val_by_decision_day(frame: pd.DataFrame, decision_day: pd.Timestamp, val_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_day = pd.Timestamp(decision_day).normalize()
    val_start = decision_day - pd.Timedelta(days=val_days)
    train_df = frame[frame["business_day"] < val_start].copy()
    val_df = frame[(frame["business_day"] >= val_start) & (frame["business_day"] < decision_day)].copy()
    return train_df, val_df


def _build_cutoff_calibration_map(
    val_scored: pd.DataFrame,
    config: ProtocolBCutoffConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    calibration_source = _restrict_calibration_window(val_scored, config.calibration_recent_days)
    if not config.apply_segment_bias_calibration:
        return {}, calibration_source

    if config.calibration_mode == "hour_bias":
        calibration_map = _build_hour_bias_map(
            calibration_source,
            config.calibration_reducer,
            config.calibration_residual_quantile,
            config.calibration_hour_scope,
        )
    elif config.calibration_mode == "segment_hour_bias":
        calibration_map = _build_segment_hour_bias_map(
            calibration_source,
            config.calibration_segments or [],
            config.calibration_reducer,
            config.calibration_residual_quantile,
            config.calibration_hour_scope,
        )
    elif config.calibration_mode == "month_hour_bias":
        calibration_map = _build_month_bucket_hour_bias_map(
            calibration_source,
            config.calibration_reducer,
            config.calibration_residual_quantile,
            config.calibration_month_bucket_mode,
            config.calibration_hour_scope,
        )
    elif config.calibration_mode == "segment_hour_sign_bias":
        calibration_map = _build_segment_hour_sign_bias_map(
            calibration_source,
            config.calibration_segments or [],
            config.calibration_reducer,
            config.calibration_residual_quantile,
            config.calibration_hour_scope,
        )
    elif config.calibration_mode in {"error_gate_bias", "combo_error_sign_gate_bias", "segment_error_gate_bias"}:
        calibration_map = _build_error_gate_bias_map(
            calibration_source,
            config.calibration_reducer,
            config.calibration_residual_quantile,
            config.calibration_error_quantile,
            config.calibration_mode,
        )
    else:
        calibration_map = _build_segment_bias_map(
            calibration_source,
            config.calibration_segments or [],
            config.calibration_reducer,
            config.calibration_residual_quantile,
        )
    return calibration_map, calibration_source


def run_protocol_b_cutoff_experiment(config_path: str | Path) -> Path:
    config = load_protocol_b_cutoff_config(config_path)
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(config.data_path)
    train_frame, feature_cols = _build_training_frame(raw_df, config.feature_config, config.resolution)
    build_feature_manifest(feature_cols).to_csv(run_dir / "feature_manifest.csv", index=False, encoding="utf-8-sig")

    start_day = pd.Timestamp(config.start_day).normalize()
    end_day = pd.Timestamp(config.end_day).normalize()
    decision_days = pd.date_range(start=start_day - pd.Timedelta(days=1), end=end_day - pd.Timedelta(days=1), freq="D")

    prediction_rows: list[pd.DataFrame] = []
    leakage_audits: list[dict[str, object]] = []
    monthly_summary_rows: list[dict[str, object]] = []

    for decision_day in decision_days:
        target_day = decision_day + pd.Timedelta(days=1)
        train_df, val_df = _build_train_val_by_decision_day(train_frame, decision_day, config.val_days)
        if len(train_df) < config.train_min_rows or val_df.empty:
            continue

        train_weight, train_tail_threshold = _build_train_sample_weight(train_df, config.model_config)
        model = build_regressor(config.model_config)
        model.fit(train_df, feature_cols, sample_weight=None if train_weight is None else train_weight.to_numpy(dtype=float))

        calibration_map: dict[str, float] = {}
        calibration_source = val_df
        da_fill_bias_map = _build_da_fill_bias_map(train_df, val_df, decision_day, config)
        if config.apply_segment_bias_calibration:
            val_scored = val_df.copy()
            val_scored["delta_hat"] = model.predict(val_scored, feature_cols)
            val_scored["rt_hat"] = val_scored["da_anchor"] + val_scored["delta_hat"]
            calibration_map, calibration_source = _build_cutoff_calibration_map(val_scored, config)

        visible_df = _build_protocol_b_visible_frame(
            raw_df,
            decision_day,
            config.decision_hour,
            config.da_fill_mode,
            da_fill_bias_map,
            config.resolution,
        )
        inference_frame, inference_feature_cols = _build_inference_frame(visible_df, config.feature_config, config.resolution)
        target_rows = inference_frame[inference_frame["business_day"] == target_day].copy()
        if target_rows.empty:
            continue
        target_rows["delta_hat"] = model.predict(target_rows, inference_feature_cols)

        # FIX (2026-07-02): guard against NaN da_anchor after fallback
        if target_rows["da_anchor"].isna().any():
            n_nan = target_rows["da_anchor"].isna().sum()
            raise RuntimeError(
                f"SGDFNet da_anchor still has {n_nan} NaN values for target_day={target_day} "
                "after fallback — cannot compute rt_hat safely."
            )

        target_rows["rt_hat"] = target_rows["da_anchor"] + target_rows["delta_hat"]

        # FIX (2026-07-02): guard against NaN rt_hat
        if target_rows["rt_hat"].isna().any():
            n_nan = target_rows["rt_hat"].isna().sum()
            raise RuntimeError(
                f"SGDFNet rt_hat has {n_nan} NaN values for target_day={target_day} "
                "after da_anchor fallback — check delta_hat or da_anchor."
            )

        target_rows = _apply_calibration_map(target_rows, calibration_map, config.calibration_mode)
        target_rows["decision_day"] = decision_day.normalize()
        target_rows["target_day"] = target_day.normalize()
        target_rows["target_month"] = target_day.strftime("%Y-%m")
        target_rows["split"] = "test_walk_forward"
        target_rows["protocol_tag"] = "B_D15_cutoff_walk_forward"
        target_rows["calibration_mode"] = config.calibration_mode if config.apply_segment_bias_calibration else "none"
        target_rows["use_visible_actual_history"] = bool(config.feature_config.use_visible_actual_history)
        target_rows["da_fill_mode"] = config.da_fill_mode
        target_rows["train_tail_threshold"] = train_tail_threshold
        prediction_rows.append(
            target_rows[
                [
                    "timestamp",
                    "business_day",
                    "decision_day",
                    "target_day",
                    "hour",
                    "segment",
                    "segment_id",
                    "da_anchor",
                    "rt_actual",
                    "delta_target",
                    "delta_hat",
                    "rt_hat",
                    "direction_label",
                    "split",
                    "train_tail_threshold",
                    "target_month",
                    "protocol_tag",
                    "calibration_mode",
                    "use_visible_actual_history",
                    "da_fill_mode",
                ]
            ].copy()
        )

        monthly_row = {
            "month": target_day.strftime("%Y-%m"),
            "decision_day": str(decision_day.date()),
            "target_day": str(target_day.date()),
            "rows": int(len(target_rows)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_tail_threshold": train_tail_threshold,
            "calibration_mode": config.calibration_mode if config.apply_segment_bias_calibration else "none",
            "use_visible_actual_history": bool(config.feature_config.use_visible_actual_history),
            "da_fill_mode": config.da_fill_mode,
        }
        monthly_row.update(build_metrics_frame(target_rows))
        monthly_summary_rows.append(monthly_row)

        decision_ts = _decision_timestamp(decision_day, config.decision_hour)
        leakage_audits.append(
            {
                "decision_day": str(decision_day.date()),
                "target_day": str(target_day.date()),
                "decision_timestamp": str(decision_ts),
                "max_visible_realtime_timestamp": str(decision_ts),
                "blocked_same_day_window": f"{decision_day.strftime('%Y-%m-%d')} {config.decision_hour + 1:02d}:00:00 -> {target_day.strftime('%Y-%m-%d')} 00:00:00",
                "same_day_post_cutoff_filled_with_da": True,
                "predicted_rows": int(len(target_rows)),
                "feature_recomputed_after_cutoff": True,
                "protocol_tag": "B_D15_cutoff_walk_forward",
                "calibration_mode": config.calibration_mode if config.apply_segment_bias_calibration else "none",
                "calibration_source_rows": int(len(calibration_source)),
                "use_visible_actual_history": bool(config.feature_config.use_visible_actual_history),
                "da_fill_mode": config.da_fill_mode,
                "da_fill_bias_map": json.dumps(da_fill_bias_map, ensure_ascii=False),
            }
        )

    if not prediction_rows:
        raise RuntimeError("No walk-forward protocol B cutoff predictions were generated.")

    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly_summary_rows).to_csv(run_dir / "monthly_summary.csv", index=False, encoding="utf-8-sig")

    metrics_payload = score_prediction_frame(predictions)
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    with open(run_dir / "split_audit.json", "w", encoding="utf-8") as f:
        json.dump(leakage_audits, f, ensure_ascii=False, indent=2)
    pd.DataFrame(leakage_audits).to_csv(run_dir / "split_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics_payload["segment_metrics"]).to_csv(run_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics_payload["tail_metrics"]).to_csv(run_dir / "tail_metrics.csv", index=False, encoding="utf-8-sig")

    with open(run_dir / "run_config_snapshot.json", "w", encoding="utf-8") as f:
        config_payload = asdict(config)
        config_payload["start_day"] = str(config.start_day)
        config_payload["end_day"] = str(config.end_day)
        json.dump(
            {
                "protocol": "B_realtime_cutoff_D15_walk_forward",
                "config": config_payload,
                "feature_count": len(feature_cols),
                "coverage_start": str(predictions["timestamp"].min()),
                "coverage_end": str(predictions["timestamp"].max()),
                "predicted_days": int(predictions["target_day"].nunique()),
                "predicted_rows": int(len(predictions)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return run_dir
