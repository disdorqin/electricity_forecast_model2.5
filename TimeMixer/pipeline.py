from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from pipelines.base import BaseModelPipeline, PredictionResult
from utils.data_layout import data_path as _project_data_path
from utils.io import ensure_prediction_frame, ensure_runtime_dirs

from .repro_pipeline import RunConfig, run_monthly_reproduction


# Relative paths computed from project layout — works on any machine
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_XLSX = str(_project_data_path("hourly"))
DEFAULT_TIMEMIXER_CSV = str(_PROJECT_ROOT.parent / "epf" / "data" / "shandong_pmos_hourly.csv")


class ModelPipeline(BaseModelPipeline):
    model_name = "timemixer"
    device_type = "gpu"

    def train(self, **kwargs):
        raise NotImplementedError("TimeMixer unified train() is not wired yet; use predict_range or legacy pipeline.")

    def predict(self, **kwargs) -> PredictionResult:
        return self.predict_range(**kwargs)

    def predict_range(self, target: str, **kwargs) -> PredictionResult:
        import torch as _torch
        _deterministic = bool(kwargs.get("deterministic", False))
        if _torch.cuda.is_available() and _deterministic:
            raise RuntimeError(
                "TimeMixer CUDA does not support strict deterministic training "
                "because upsample backward has no deterministic implementation; "
                "use deterministic=False for GPU or force CPU."
            )
        # GPU 训练需要非确定性（upsample 等无确定性 CUDA 实现），显式关闭
        # deterministic_algorithms，避免链路残留 True 标志导致 CUDA 崩溃。
        if _torch.cuda.is_available():
            _torch.use_deterministic_algorithms(False, warn_only=False)
            if hasattr(_torch, "set_deterministic_debug_mode"):
                _torch.set_deterministic_debug_mode("default")
            _torch.backends.cudnn.deterministic = False

        from utils.resolution import resolve_resolution
        _res = resolve_resolution(kwargs.get("resolution", "hourly"))
        res_n = _res.slots_per_day
        domain = "96" if _res.label == "15min" else "24"
        base_runtime = Path(
            kwargs.get("output_root") or f"outputs/{domain}/runtime/manual_models"
        )
        output_root = ensure_runtime_dirs(base_runtime / self.model_name / target)
        predict_date = pd.Timestamp(kwargs.get("predict_date"))
        month = predict_date.strftime("%Y-%m")

        start_date = kwargs.get("start") or predict_date.strftime("%Y-%m-%d")
        end_date = kwargs.get("end") or predict_date.strftime("%Y-%m-%d")
        # TimeMixer 的 run_monthly_reproduction 使用半开区间 [test_start, test_end_exclusive)。
        # 当调用方传入 start == end 时（单日预测），应将 end 视为包含该日，
        # 因此 test_end_exclusive = end + 1 天，否则 test_days 为空。
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        # test_end_exclusive is a half-open bound: always add 1 day to end
        # so the range [start, end_exclusive) includes the full end date
        end_exclusive = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        run_cfg = RunConfig(
            data_path=self._prepare_data_path(kwargs.get("data_path")),
            output_dir=str(output_root),
            month=month,
            test_start=start_date,
            test_end_exclusive=end_exclusive,
            append_leaderboard=False,
            train_months=int(kwargs.get("training_months", 12)),
            val_ratio=float(kwargs.get("val_ratio", 0.2)),
            cutoff_hour_rt=int(kwargs.get("realtime_cutoff_hour", 15)),
            dynamic_serving=bool(kwargs.get("dynamic_serving", False)),
            epochs=int(kwargs.get("timemixer_epochs", 80)),
            patience=int(kwargs.get("timemixer_patience", 15)),
            batch_size=int(kwargs.get("timemixer_batch_size", 16)),
            seed=int(kwargs.get("seed", kwargs.get("timemixer_seeds", 42))),
            deterministic=bool(kwargs.get("deterministic", False)),
            resolution=res_n,
            # 96 点默认 4 天窗口（384 点）而非 7 天（672），降训练开销且精度影响小
            seq_len=int(kwargs.get("seq_len", 4 * res_n)),
        )
        result = run_monthly_reproduction(run_cfg)
        raw = pd.read_csv(Path(result["output_dir"]) / "predictions_raw.csv", encoding="utf-8-sig")
        if raw.empty:
            raise ValueError(f"TimeMixer produced empty predictions_raw for {predict_date.date()}")
        # predictions_raw uses 'y_pred' for both DA and RT tasks
        # 'pred_day_ahead_price' is only populated in RT rows (as DA injection reference)
        prediction_col = "y_pred"
        task_filter = "da" if target == "dayahead" else "rt"
        if "task" in raw.columns:
            filtered = raw[raw["task"] == task_filter].copy()
        else:
            filtered = raw.copy()
        if filtered.empty:
            raise ValueError(f"TimeMixer task={task_filter} filter yielded 0 rows for {predict_date.date()}")
        normalized = ensure_prediction_frame(filtered.rename(columns={"ds": "时刻"}), prediction_col)

        # ---- TimeMixer output validation: N rows, hours D 01:00 ~ D+1 00:00 ----
        _validate_timemixer_output(normalized, predict_date, target, resolution=res_n)

        output_path = output_root / "predictions.csv"
        normalized.to_csv(output_path, index=False, encoding="utf-8-sig")
        return PredictionResult(model_name=self.model_name, target=target, output_path=output_path, frame=normalized)

    @staticmethod
    def _prepare_data_path(data_path: str | None) -> str:
        if not data_path:
            # Prefer the xlsx in the project data/ folder
            if Path(DEFAULT_DATA_XLSX).exists():
                data_path = DEFAULT_DATA_XLSX
            elif Path(DEFAULT_TIMEMIXER_CSV).exists():
                return DEFAULT_TIMEMIXER_CSV
            else:
                return DEFAULT_DATA_XLSX  # let it error clearly if missing
        path = Path(data_path)
        if path.suffix.lower() != ".xlsx":
            return str(path)
        # Convert xlsx -> csv with explicit engine and encoding
        df = pd.read_excel(path, engine="openpyxl")
        tmp_dir = Path(tempfile.gettempdir()) / "timemixer_unified_cache"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / f"{path.stem}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return str(csv_path)


def _validate_timemixer_output(df: pd.DataFrame, predict_date: pd.Timestamp, target: str, resolution: int = 24) -> None:
    """Validate TimeMixer output using raw timestamp column.

    Business day D must be N 个点（24 小时 / 96 个 15 分钟）：
      hourly: D 01:00, ..., D+1 00:00
      15min:  D 00:15, ..., D+1 00:00
    """
    issues: list[str] = []
    N = resolution

    if len(df) != N:
        issues.append(f"expected {N} rows, got {len(df)}")

    ts_col = "时刻" if "时刻" in df.columns else ("ds" if "ds" in df.columns else None)
    if ts_col is None:
        issues.append(f"missing timestamp column, columns={list(df.columns)}")
    else:
        ts = pd.to_datetime(df[ts_col], errors="coerce").sort_values().reset_index(drop=True)
        if ts.isna().any():
            issues.append("timestamp contains NaT")

        target_dt = pd.Timestamp(predict_date).normalize()
        if N == 24:
            expected_start = target_dt + pd.Timedelta(hours=1)
            freq = "h"
        else:
            expected_start = target_dt + pd.Timedelta(minutes=15)
            freq = "15min"
        expected_end = target_dt + pd.Timedelta(days=1)
        expected_ts = pd.date_range(expected_start, expected_end, freq=freq)

        if len(ts) == N:
            if ts.iloc[0] != expected_start:
                issues.append(f"first timestamp {ts.iloc[0]} != expected {expected_start}")
            if ts.iloc[-1] != expected_end:
                issues.append(f"last timestamp {ts.iloc[-1]} != expected {expected_end}")
            if not ts.equals(pd.Series(expected_ts)):
                missing = sorted(set(expected_ts) - set(ts))
                extra = sorted(set(ts) - set(expected_ts))
                if missing:
                    issues.append(f"missing timestamps: {missing[:5]}")
                if extra:
                    issues.append(f"extra timestamps: {extra[:5]}")

        if (ts == target_dt).any():
            issues.append(f"TimeMixer incorrectly includes D 00:00: {target_dt}")

    slot_col = "hour_business" if "hour_business" in df.columns else ("business_period" if "business_period" in df.columns else None)
    if slot_col:
        slots = sorted(df[slot_col].dropna().astype(int).unique())
        if slots != list(range(1, N + 1)):
            issues.append(f"{slot_col} must be 1..{N}, got {slots}")
        if 0 in set(slots):
            issues.append(f"{slot_col}=0 found")

    if issues:
        raise ValueError(
            f"TimeMixer output validation FAILED for {target} {pd.Timestamp(predict_date).date()}: "
            + " | ".join(str(x) for x in issues)
        )
