from __future__ import annotations

import logging
import tempfile
from pathlib import Path
import sys

import pandas as pd
import yaml

from pipelines.base import BaseModelPipeline, PredictionResult

logger = logging.getLogger(__name__)
from utils.io import ensure_prediction_frame, ensure_runtime_dirs


SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sgdfnet.protocol_b_cutoff import run_protocol_b_cutoff_experiment  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "cutoff_recovery_2026_diag_a_prune_actualside.yaml"


class ModelPipeline(BaseModelPipeline):
    model_name = "sgdfnet"
    device_type = "cpu"

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path or DEFAULT_CONFIG)

    def train(self, **kwargs):
        return run_protocol_b_cutoff_experiment(self.config_path)

    def predict(self, **kwargs) -> PredictionResult:
        return self.predict_range(**kwargs)

    def predict_range(self, target: str, **kwargs) -> PredictionResult:
        # Reproducibility
        from utils.reproducibility import set_global_seed

        set_global_seed(int(kwargs.get("seed", 42)), bool(kwargs.get("deterministic", False)))

        from utils.resolution import resolve_resolution
        _res = resolve_resolution(kwargs.get("resolution", "hourly"))
        domain = "96" if _res.label == "15min" else "24"
        base_runtime = Path(
            kwargs.get("output_root") or f"outputs/{domain}/runtime/manual_models"
        )
        output_root = ensure_runtime_dirs(base_runtime / self.model_name / target)

        # Build a temporary config YAML with overrides from kwargs
        predict_date = pd.Timestamp(kwargs.get("predict_date", "2026-05-15"))
        data_path = kwargs.get("data_path")
        start = kwargs.get("start") or predict_date.strftime("%Y-%m-%d")
        end = kwargs.get("end") or predict_date.strftime("%Y-%m-%d")
        # SGDFNet core uses decision_days = [start_day-1 .. end_day-1],
        # so predictions cover [start_day .. end_day] inclusive.
        end_day = end

        decision_hour = int(kwargs.get("realtime_cutoff_hour", 15))
        resolution = kwargs.get("resolution", "hourly")
        # "hourly"→24, "15min"→96（protocol_b_cutoff 用整数分辨率）
        if isinstance(resolution, str):
            res_code = 96 if resolution in ("15min", "quarter") else 24
        else:
            res_code = getattr(resolution, "slots_per_day", 24)
        logger.info(f"SGDFNet decision_hour={decision_hour} resolution={res_code}")

        dynamic_serving = bool(kwargs.get("dynamic_serving", False))
        # Dynamic formal96 already runs inside a deeply nested attempt sandbox.
        # Keep the SGDFNet core path intentionally short on Windows; otherwise
        # feature_manifest.csv can exceed legacy MAX_PATH and fail before the
        # model even starts. Legacy runs retain the historical subdirectory.
        core_output_root = output_root if dynamic_serving else (output_root / "sgdfnet_runs")
        tmp_config = self._build_temp_config(
            data_path=data_path,
            start_day=start,
            end_day=end_day,
            output_root=str(core_output_root),
            decision_hour=decision_hour,
            resolution=res_code,
            seed=int(kwargs.get("seed", 42)),
            deterministic=bool(kwargs.get("deterministic", False)),
            dynamic_serving=dynamic_serving,
        )

        run_dir = Path(run_protocol_b_cutoff_experiment(tmp_config))
        predictions = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig")

        # Detect timestamp column
        if "timestamp" in predictions.columns:
            ts_col = "timestamp"
        elif "ds" in predictions.columns:
            ts_col = "ds"
        else:
            ts_col = predictions.columns[0]

        predictions[ts_col] = pd.to_datetime(predictions[ts_col], errors="coerce")
        start_date = pd.Timestamp(start).normalize().date()
        end_date = pd.Timestamp(end).normalize().date()
        # Always compute pred_dates for error reporting
        pred_dates = predictions[ts_col].dt.date

        # SGDFNet uses business_day for the 24-hour prediction window (01:00-24:00),
        # where the 24th point is timestamped as 00:00 of the next calendar day.
        # Filter by business_day when available so the last hour is not dropped.
        if "business_day" in predictions.columns:
            biz_dates = pd.to_datetime(predictions["business_day"], errors="coerce").dt.date
            mask = (biz_dates >= start_date) & (biz_dates <= end_date)
        else:
            mask = (pred_dates >= start_date) & (pred_dates <= end_date)
        filtered = predictions[mask].copy()

        if filtered.empty:
            available_min = pred_dates.min()
            available_max = pred_dates.max()
            raise ValueError(
                f"SGDFNet produced no predictions for [{start} .. {end}]. "
                f"Core returned {len(predictions)} rows covering "
                f"[{available_min} .. {available_max}]. "
                f"Possible causes: insufficient training data (train_min_rows=2160), "
                f"data gaps, or all decision_days skipped."
            )

        # Preserve anchor metadata before normalizing the timestamp name.
        # ensure_prediction_frame uses 时刻 while the SGDFNet core emits
        # timestamp; keeping metadata first avoids a stale-column lookup.
        anchor_cols = [
            "anchor_source_day", "anchor_source_type", "anchor_rows", "fallback_used",
        ]
        metadata = None
        if all(col in filtered.columns for col in anchor_cols):
            metadata = filtered[[ts_col, *anchor_cols]].copy()
            metadata[ts_col] = pd.to_datetime(metadata[ts_col], errors="coerce")
            if ts_col != "时刻":
                metadata = metadata.rename(columns={ts_col: "时刻"})

        # Rename timestamp column to '时刻' for ensure_prediction_frame.
        if ts_col != "时刻":
            filtered = filtered.rename(columns={ts_col: "时刻"})

        normalized = ensure_prediction_frame(filtered, "rt_hat")
        # Keep the formal anchor audit fields through the generic prediction
        # normalizer (which intentionally strips model-specific columns).
        if metadata is not None:
            normalized = normalized.merge(metadata, on="时刻", how="left")
        output_path = output_root / "predictions.csv"
        normalized.to_csv(output_path, index=False, encoding="utf-8-sig")
        return PredictionResult(
            model_name=self.model_name, target=target, output_path=output_path, frame=normalized
        )

    def _build_temp_config(
        self,
        data_path: str | None,
        start_day: str,
        end_day: str,
        output_root: str,
        decision_hour: int = 14,
        resolution: int = 24,
        seed: int = 42,
        deterministic: bool = False,
        dynamic_serving: bool = False,
    ) -> Path:
        """Create a temporary YAML config overriding key fields from the base config."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)

        # Override paths, date range, and cutoff
        if data_path:
            base_cfg["data_path"] = str(data_path)
        base_cfg["start_day"] = start_day
        base_cfg["end_day"] = end_day
        base_cfg["output_root"] = output_root
        base_cfg["decision_hour"] = int(decision_hour)
        base_cfg["resolution"] = int(resolution)
        base_cfg["seed"] = int(seed)
        base_cfg["deterministic"] = bool(deterministic)
        base_cfg["dynamic_serving"] = bool(dynamic_serving)
        if dynamic_serving:
            # Avoid a long experiment-name suffix under the already long
            # production attempt path. This changes only scratch naming.
            base_cfg["experiment_name"] = "dyn"

        # Write to temp file
        tmp_dir = Path(tempfile.gettempdir()) / "sgdfnet_staged_configs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"staged_{start_day}_{end_day}.yaml"
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.dump(base_cfg, f, allow_unicode=True, default_flow_style=False)

        return tmp_path
