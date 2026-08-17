"""
EPF v1.0 TimesFM Adapter

Wraps the best-performing TimesFM implementation from the EPF v1.0
repository. This is the SINGLE entry point for TimesFM in the ledger
pipeline — no more circular calls between TF/ and TimesFM/ directories.

Supports cutoff-safe prediction by constructing temporary truncated
data files when the original script doesn't support data_cutoff.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.business_day import standardize_business_columns

logger = logging.getLogger(__name__)


class TimesFMV1Adapter:
    """Adapter for TimesFM predictions.

    Uses the bundled TimesFMBackend/ implementation by default.
    Optionally supports external EPF v1.0 root for legacy compatibility.

    This is the canonical TimesFM wrapper for the ledger pipeline.
    No other wrapper should call TimesFM directly.

    Mode "exact": faithful v1 behavior, no data truncation.
    Mode "cutoff_safe": truncates data to enforce cutoff safety
                        (only enabled when explicitly requested).
    """

    def __init__(self, epf_root: str | None = None, mode: str = "exact"):
        self.epf_root = Path(epf_root) if epf_root else None
        self.mode = mode
        if self.epf_root and self.epf_root.exists():
            self._ensure_epf_on_path()
        elif self.epf_root:
            logger.warning("Ignoring missing legacy EPF root: %s", self.epf_root)
            self.epf_root = None

    def _ensure_epf_on_path(self):
        """Add EPF v1.0 root to sys.path for imports."""
        epf_str = str(self.epf_root.resolve())
        if epf_str not in sys.path:
            sys.path.insert(0, epf_str)

    def predict(
        self,
        target_date: str,
        target: str = "dayahead",
        data_path: Optional[str] = None,
        cutoff_date: Optional[str] = None,
        seed: int = 42,
        deterministic: bool = False,
        resolution: str = "hourly",
    ) -> pd.DataFrame:
        """
        Run TimesFM prediction for a single target day.

        Parameters
        ----------
        target_date : str
            Target business day (YYYY-MM-DD).
        target : str
            "dayahead" or "realtime".
        data_path : str, optional
            Path to data file.
        cutoff_date : str, optional
            Latest date allowed in input data. Default: target_date - 1 day.
        seed : int
            Global random seed for reproducibility.
        deterministic : bool
            Enable deterministic algorithms (may be slower).
        resolution : str
            "hourly" (24 点, default) or "15min" (96 点).

        Returns
        -------
        pd.DataFrame with standardized prediction columns.
        """
        from utils.reproducibility import set_global_seed


        set_global_seed(seed, deterministic)

        if cutoff_date is None:
            cutoff_date = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Map target
        tf_target = "realtime" if target == "realtime" else "dayahead"

        # Resolve data path
        if data_path is None:
            data_path = self._find_data_file()

        logger.info(
            f"TimesFM v1 [{self.mode}]: predicting {target_date} ({target}), "
            f"cutoff={cutoff_date}"
        )

        # Enforce cutoff safety: only when mode=cutoff_safe
        if self.mode == "cutoff_safe":
            safe_data_path = self._ensure_cutoff_safe_data(
                data_path, cutoff_date, target_date
            )
        else:
            # exact mode: pass data as-is, faithful to v1 behavior
            safe_data_path = data_path

        try:
            from TimesFMBackend.infer import predict_price_for_date

            df = predict_price_for_date(
                data_path=safe_data_path,
                forecast_date=target_date,
                target=tf_target,
                resolution=resolution,
            )
        except Exception as e:
            logger.error(f"TimesFM v1 prediction failed: {e}")
            raise

        # Standardize output — TimesFM returns columns like ["时刻", "预测值"]
        task_label = "dayahead" if target == "dayahead" else "realtime"
        df = standardize_business_columns(
            df,
            ds_col="时刻",
            y_pred_col="预测值",
            task_label=task_label,
            model_name="timesfm",
            forecast_date=target_date,
            target_day=target_date,
            data_cutoff=cutoff_date,
            run_id=f"timesfm_v1_{target_date}",
            model_version="epf_v1",
            resolution=resolution,
        )

        # Keep only required columns
        keep_cols = [
            "task", "model_name", "forecast_date", "target_day",
            "business_day", "ds", "business_period", "hour_business", "period", "y_pred",
            "data_cutoff", "run_id", "model_version",
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

        return df

    def _ensure_cutoff_safe_data(
        self, data_path: str, cutoff_date: str, target_date: str
    ) -> str:
        """
        If the data file contains data beyond cutoff_date, create a
        temporary truncated copy to enforce cutoff safety.

        Returns the path to use (original or temp copy).
        """
        ext = os.path.splitext(data_path)[1].lower()

        try:
            from utils.data_loader import load_table

            df = load_table(data_path)
        except Exception:
            return data_path

        # Find timestamp column
        ts_col = None
        for c in ["时刻", "ds", "timestamp", "time", "datetime"]:
            if c in df.columns:
                ts_col = c
                break

        if ts_col is None:
            return data_path

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        cutoff_dt = pd.Timestamp(cutoff_date) + pd.Timedelta(days=1)  # end of cutoff day

        # Check if any data is beyond cutoff
        future_mask = df[ts_col] > cutoff_dt
        if not future_mask.any():
            return data_path  # already safe

        logger.info(
            f"Truncating data: {future_mask.sum()} rows beyond cutoff {cutoff_date}"
        )

        # Truncate
        safe_df = df[~future_mask].copy()

        # Write temp file
        suffix = os.path.splitext(data_path)[1]
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="timesfm_safe_"
        )
        safe_path = tmp.name
        tmp.close()

        if suffix in (".xlsx", ".xls"):
            safe_df.to_excel(safe_path, index=False)
        elif suffix == ".parquet":
            safe_df.to_parquet(safe_path, index=False)
        else:
            safe_df.to_csv(safe_path, index=False)

        return safe_path

    def _find_data_file(self) -> str:
        """Auto-locate canonical 24-point data, then legacy/EPF paths."""
        from utils.data_layout import DATA
        candidates = [
            DATA.hourly_xlsx,
            DATA.hourly_csv,
            Path("data/shandong_pmos_hourly.xlsx"),
            Path("data/shandong_pmos_hourly.csv"),
        ]
        if self.epf_root:
            candidates.extend([
                self.epf_root / "data" / "shandong_pmos_hourly.xlsx",
                self.epf_root / "data" / "shandong_pmos_hourly.csv",
            ])
        for c in candidates:
            if c.exists():
                return str(c)
        raise FileNotFoundError(
            f"No data file found. Checked: {[str(c) for c in candidates]}"
        )
