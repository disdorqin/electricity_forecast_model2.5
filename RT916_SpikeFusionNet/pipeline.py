from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

from pipelines.base import BaseModelPipeline, PredictionResult
from utils.io import ensure_prediction_frame, ensure_runtime_dirs


SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rt916_spikefusionnet import core  # noqa: E402


TARGET_MAP = {
    "dayahead": "日前电价",
    "realtime": "实时电价",
}


class ModelPipeline(BaseModelPipeline):
    model_name = "rt916"
    device_type = "gpu"

    @staticmethod
    def _apply_seed(kwargs: dict) -> None:
        """Apply seed/deterministic to RT916 core config and global state."""
        from utils.reproducibility import set_global_seed

        seed = int(kwargs.get("seed", 42))
        deterministic = bool(kwargs.get("deterministic", False))
        set_global_seed(seed, deterministic)
        core.CONFIG["SEED"] = seed

    def train(self, target: str = "realtime", **kwargs):
        from utils.resolution import resolve_resolution
        _res = resolve_resolution(kwargs.get("resolution", "hourly"))
        core.set_resolution(_res.slots_per_day)
        self._apply_seed(kwargs)
        _dp = kwargs.get("data_path")
        if _dp:
            core.RAW_DF_PATH = os.path.abspath(_dp)
        start_end = self._resolve_start_end(kwargs)
        return core.train_interface(target=TARGET_MAP[target], start_end_list=start_end, mod="all")

    def predict(self, **kwargs) -> PredictionResult:
        return self.predict_range(**kwargs)

    def predict_range(self, target: str, **kwargs) -> PredictionResult:
        from utils.resolution import resolve_resolution
        _res = resolve_resolution(kwargs.get("resolution", "hourly"))
        core.set_resolution(_res.slots_per_day)
        # Reproducibility
        self._apply_seed(kwargs)
        # Disable AMP during RT916 inference — model weights saved in BFloat16
        # cause "Unsupported dtype BFloat16" when converted to numpy.
        # Training (separate path) still benefits from AMP.
        os.environ["OPTIM_AMP"] = "0"
        os.environ["SPIKE_TRAIN_MONTHS"] = str(int(kwargs.get("training_months", 12)))
        # Override frozen RAW_DF_PATH so the model works on other machines / paths
        _dp = kwargs.get("data_path")
        if _dp:
            core.RAW_DF_PATH = os.path.abspath(_dp)
        start_end = self._resolve_start_end(kwargs)

        # Read cutoff hour from kwargs (default 14 for realtime, 24 for dayahead)
        asof_hour = int(kwargs.get("realtime_cutoff_hour", 14))
        if target == "dayahead":
            # Dayahead uses full D-1 data; pass 24 to indicate end-of-day
            asof_hour = 24

        if target == "realtime":
            # RT916 realtime must first produce DA predictions, then inject them into RT.
            result = core.run_joint_da_rt_daily_backtest(
                start_end_list=start_end,
                mod="all",
                asof_hour=asof_hour,
            )
        else:
            result = core.run_daily_asof_backtest(
                target=TARGET_MAP[target],
                start_end_list=start_end,
                mod="all",
                asof_hour=asof_hour,
                retrain_daily=False,
            )
        prediction_col = "预测日前电价" if target == "dayahead" else "预测实时电价"
        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            raise ValueError(
                f"RT916 produced no predictions for target={target} "
                f"[{start_end[0]} to {start_end[1]}]. "
                f"Possible causes: insufficient training data, core returned empty DataFrame."
            )
        normalized = ensure_prediction_frame(result, prediction_col)
        output_root = ensure_runtime_dirs(Path(kwargs.get("output_root", "outputs/unified_runs")) / self.model_name / target)
        output_path = output_root / "predictions.csv"
        normalized.to_csv(output_path, index=False, encoding="utf-8-sig")
        return PredictionResult(model_name=self.model_name, target=target, output_path=output_path, frame=normalized)

    @staticmethod
    def _resolve_start_end(kwargs: dict) -> list[str]:
        from utils.resolution import resolve_resolution

        _res = resolve_resolution(kwargs.get("resolution", "hourly"))
        _slot_minutes = _res.minutes_per_slot  # 60=hourly, 15=quarter
        start = kwargs.get("start")
        end = kwargs.get("end")
        if start and end:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            if start_ts.hour == 0 and start_ts.minute == 0 and start_ts.second == 0:
                start_ts = start_ts.normalize() + pd.Timedelta(minutes=_slot_minutes)
            if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
                end_ts = end_ts.normalize() + pd.Timedelta(days=1)
            return [start_ts.strftime("%Y-%m-%d %H:%M:%S"), end_ts.strftime("%Y-%m-%d %H:%M:%S")]
        predict_date = pd.Timestamp(kwargs.get("predict_date"))
        start_ts = predict_date.normalize() + pd.Timedelta(minutes=_slot_minutes)
        end_ts = predict_date.normalize() + pd.Timedelta(days=1)
        return [start_ts.strftime("%Y-%m-%d %H:%M:%S"), end_ts.strftime("%Y-%m-%d %H:%M:%S")]
