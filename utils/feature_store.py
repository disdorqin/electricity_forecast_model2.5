"""
FeatureStore — 特征预计算（S2 最小版，先验证 LightGBM DA 96 点零损失）。

设计（docs/FeatureStore_特征预计算_设计.md）：
  - 全历史特征一次物化存 parquet，模型只按日期切片（消灭每次 read_excel + 重算）。
  - 特征注册表 = 唯一事实源，shift 常量 resolution 化（N = slots_per_day）。
  - 零精度损失：物化切片 vs 现状重算必须逐位相等（S2 验收）。

S2 范围：LightGBM DA 96 点（特征最简单，适合作首个零损失验证对象）。
后续：SGDFNet / TimeMixer / RT916 接入 + warm-start。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_ROOT = Path("outputs/feature_store")

# 特征注册表：唯一事实源。key=(model, task, resolution), value=特征列清单
# 物化时用本文件的 FEATURE_REGISTRY 版本号作缓存键一部分。
FEATURE_REGISTRY_VERSION = "s3_shared_views_96_v1"
SPREAD_FEATURE_STORE_SCHEMA = "spread_hourly_v1"

# LightGBM DA 需要的原始输入列（映射自宽表中文列名 → 内部名）
LGBM_DA_96_COLS = [
    "hour_sin", "hour_cos", "lag_price_target", "price_rolling_mean_24h",
    "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
    "ramp_load", "ramp_solar", "prev_day_avg", "prev_day_max", "prev_day_min",
    "month", "day_of_week", "is_weekend", "business_period",
]


def _source_fingerprint(path: Path) -> str:
    """Content fingerprint used to prevent stale feature reuse."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:24]


def _feature_def_version() -> str:
    return FEATURE_REGISTRY_VERSION


class FeatureStore:
    """轻量特征存储：原始数据 parquet 缓存 + 特征矩阵物化。

    两层：
      1. raw 缓存：把 30MB xlsx 转 parquet（16MB），消灭每次 read_excel(~30-76s)。
         load_raw() 替代各模型 load_and_process_data 里的 read_excel，零精度损失。
      2. 特征矩阵：全历史特征一次物化（S2 已验证 LightGBM DA 96 点零损失）。
    """

    def __init__(
        self,
        resolution: str = "15min",
        source: str | Path | None = None,
        root: str | Path | None = None,
    ):
        from utils.resolution import resolve_resolution

        self.res = resolve_resolution(resolution)
        self.source = Path(source) if source else None
        self.feature_root = Path(root) if root else FEATURE_ROOT
        self.version = f"res{self.res.slots_per_day}_v{_feature_def_version()}"
        if self.source:
            self.version += f"_{_source_fingerprint(self.source)}"
        self.dir = self.feature_root / self.version
        self.raw_path = self.dir / "raw.parquet"
        self.base_path = self.dir / f"base_{self.res.slots_per_day}.parquet"
        self.views_root = self.dir
        self.matrix_path = self.dir / f"da_matrix.parquet"  # 先做 DA
        self.spread_version = None
        self.spread_dir = None
        self.spread_base_path = None
        self.spread_cache_hit = False
        self._da = None
        self._raw = None

    def load_raw(self, force: bool = False) -> pd.DataFrame:
        """读取原始宽表（xlsx → parquet 缓存）。替代各模型 read_excel。"""
        if self._raw is not None:
            return self._raw
        if self.raw_path.exists() and not force:
            self._raw = pd.read_parquet(self.raw_path)
            logger.info(f"FeatureStore raw 缓存命中: {self.raw_path}")
            return self._raw
        if self.source is None:
            raise ValueError("source 数据文件未指定")
        self.spread_cache_hit = False
        # The raw cache is the only place allowed to read the source table.
        # Use the unified loader so the first build supports xlsx/csv/parquet
        # and every subsequent build is served from raw.parquet.
        from utils.data_loader import load_table

        df = load_table(self.source)
        self.dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.raw_path, index=False)
        self._raw = df
        logger.info(f"FeatureStore raw 已物化: {self.raw_path}")
        return df

    def ensure(self) -> "FeatureStore":
        """物化（若缓存存在且指纹匹配则复用），读入内存。"""
        if self.matrix_path.exists():
            manifest = self._load_manifest()
            if manifest and manifest.get("version") == self.version:
                self._da = pd.read_parquet(self.matrix_path)
                logger.info(f"FeatureStore 缓存命中: {self.matrix_path}")
                return self
        self._build_da()
        self._write_manifest()
        logger.info(f"FeatureStore 已物化: {self.matrix_path}")
        return self

    def ensure_base(self) -> Path:
        """Materialize the shared, resolution-isolated base table once."""
        if self.base_path.exists():
            return self.base_path
        raw = self.load_raw().copy()
        if "时刻" in raw.columns:
            raw["时刻"] = pd.to_datetime(raw["时刻"], errors="coerce")
            raw = raw.dropna(subset=["时刻"]).sort_values("时刻").reset_index(drop=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.base_path.with_name(self.base_path.name + f".tmp-{os.getpid()}")
        raw.to_parquet(tmp, index=False)
        os.replace(tmp, self.base_path)
        return self.base_path

    def ensure_spread_base(
        self,
        dayahead_col: str,
        realtime_col: str,
        spread_col: str = "价差",
        *,
        force: bool = False,
    ) -> Path:
        """Materialize a causal hourly spread base shared by all models.

        This is deliberately a *base* view only: it contains source prices,
        spread, and business-time keys.  Cutoff masking and fill strategies
        remain derived as-of views in the spread experiment layer so that
        ``masked_direct`` and ``safe_mixed_lag`` cannot share mutable state.
        """
        if self.source is None:
            raise ValueError("source 数据文件未指定")
        source_hash = _source_fingerprint(self.source)
        self.spread_version = f"res{self.res.slots_per_day}_{SPREAD_FEATURE_STORE_SCHEMA}_{source_hash}"
        self.spread_dir = self.feature_root / self.spread_version
        self.spread_base_path = self.spread_dir / f"base_{self.res.slots_per_day}.parquet"
        manifest_path = self.spread_dir / "manifest.json"
        if not force and self.spread_base_path.exists() and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("schema_version") == SPREAD_FEATURE_STORE_SCHEMA
                    and manifest.get("source_sha256") == source_hash
                    and manifest.get("dayahead_column") == dayahead_col
                    and manifest.get("realtime_column") == realtime_col
                    and manifest.get("spread_column") == spread_col
                ):
                    self.spread_cache_hit = True
                    return self.spread_base_path
            except (OSError, json.JSONDecodeError):
                pass

        work = self.load_raw(force=force).copy()
        if "时刻" not in work.columns:
            raise ValueError("spread source must contain 时刻")
        for col in (dayahead_col, realtime_col):
            if col not in work.columns:
                raise ValueError(f"spread source missing {col}")
        work["时刻"] = pd.to_datetime(work["时刻"], errors="coerce")
        work = work.dropna(subset=["时刻"]).sort_values("时刻").reset_index(drop=True)
        if work["时刻"].duplicated().any():
            raise ValueError("spread source contains duplicate timestamps")
        work["_business_day"] = work["时刻"].map(self.res.business_day_from_timestamp)
        work["_business_period"] = work["时刻"].map(self.res.business_period_from_timestamp).astype(int)
        work[spread_col] = pd.to_numeric(work[realtime_col], errors="coerce") - pd.to_numeric(
            work[dayahead_col], errors="coerce"
        )
        self.spread_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.spread_base_path.with_name(self.spread_base_path.name + f".tmp-{os.getpid()}")
        work.to_parquet(tmp, index=False)
        os.replace(tmp, self.spread_base_path)
        manifest = {
            "schema_version": SPREAD_FEATURE_STORE_SCHEMA,
            "version": self.spread_version,
            "source": str(self.source.resolve()),
            "source_sha256": source_hash,
            "resolution": self.res.label,
            "dayahead_column": dayahead_col,
            "realtime_column": realtime_col,
            "spread_column": spread_col,
            "rows": int(len(work)),
            "columns": list(work.columns),
            "materialized_at": datetime.now().isoformat(),
        }
        manifest_tmp = manifest_path.with_name(manifest_path.name + f".tmp-{os.getpid()}")
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
        return self.spread_base_path

    def ensure_view(self, model: str, task: str) -> Path:
        """Create a model/task namespaced immutable view.

        The first rollout uses the shared base schema for every adapter. This
        removes repeated source parsing without changing any adapter feature
        formula. Model-specific column projections can be added to the
        registry later and will invalidate this versioned namespace.
        """
        if task not in {"dayahead", "realtime"}:
            raise ValueError(f"Unsupported FeatureStore task: {task}")
        base = self.ensure_base()
        view_dir = self.views_root / task
        view_dir.mkdir(parents=True, exist_ok=True)
        view_path = view_dir / f"{model}.parquet"
        if not view_path.exists():
            tmp = view_path.with_name(view_path.name + f".tmp-{os.getpid()}")
            shutil.copyfile(base, tmp)
            os.replace(tmp, view_path)
        return view_path

    def _load_manifest(self) -> dict | None:
        p = self.dir / "manifest.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_manifest(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": self.version,
            "source": str(self.source),
            "resolution": self.res.label,
            "materialized_at": datetime.now().isoformat(),
            "rows": len(self._da),
            "cols": list(self._da.columns),
            "nan_counts": {c: int(self._da[c].isna().sum()) for c in self._da.columns if self._da[c].isna().any()},
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_da(self) -> None:
        """物化 LightGBM DA 96 点特征矩阵（全历史一次算）。"""
        if self.source is None:
            raise ValueError("source 数据文件未指定")
        # Matrix construction must reuse the raw cache; it must never reopen
        # the source workbook after load_raw() has materialized or hit it.
        df = self.load_raw().copy()
        N = self.res.slots_per_day  # 96

        df["ds"] = pd.to_datetime(df["时刻"], errors="coerce")
        df["y"] = pd.to_numeric(df["日前电价"], errors="coerce")
        df["load"] = pd.to_numeric(df["直调负荷预测值"], errors="coerce").ffill()
        df["wind"] = pd.to_numeric(df["风电总加预测值"], errors="coerce").ffill()
        df["solar"] = pd.to_numeric(df["光伏总加预测值"], errors="coerce").ffill()
        df["interconnect"] = pd.to_numeric(df["联络线受电负荷预测值"], errors="coerce").ffill()
        df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)

        # 1. 时间与周期（-1s 对齐，与 infer_da_fix 一致）
        adj = df["ds"] - pd.Timedelta(seconds=1)
        df["business_period"] = [self.res.business_period_from_timestamp(ts) for ts in df["ds"]]
        df["month"] = adj.dt.month
        df["day_of_week"] = adj.dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        hour_biz = adj.dt.hour + 1
        df["hour_sin"] = np.sin(2 * np.pi * (hour_biz - 1) / 23)
        df["hour_cos"] = np.cos(2 * np.pi * (hour_biz - 1) / 23)

        # 2. 日前滞后（resolution 化：lag_1day=N=96 行）
        lag_1day, lag_7day = N, 7 * N
        df["lag_24h"] = df["y"].shift(lag_1day)
        df["lag_168h"] = df["y"].shift(lag_7day)
        df["lag_price_target"] = np.where(df["day_of_week"] == 0, df["lag_168h"], df["lag_24h"])
        df["price_rolling_mean_24h"] = df["y"].shift(lag_1day).rolling(window=lag_1day).mean()

        # 3. 物理特征
        safe_load = df["load"].replace(0, 1)
        df["net_load"] = df["load"] - df["wind"] - df["solar"]
        df["solar_ratio"] = df["solar"] / safe_load
        df["net_load_sq"] = (df["net_load"] / 1000) ** 2
        df["bidding_space"] = df["net_load"] - df["interconnect"]
        df["space_ratio"] = df["bidding_space"] / safe_load
        df["wind_ratio"] = df["wind"] / safe_load
        df["renew_penetration"] = (df["wind"] + df["solar"]) / safe_load
        df["ramp_load"] = df["load"].diff().fillna(0)
        df["ramp_solar"] = df["solar"].diff().fillna(0)

        # 4. 昨日统计量（groupby 业务日 → shift(1天)）
        df["date_only"] = adj.dt.date
        daily_stats = df.groupby("date_only")["y"].agg(
            prev_day_avg="mean", prev_day_max="max", prev_day_min="min"
        ).shift(1).reset_index()
        df = df.merge(daily_stats, on="date_only", how="left")
        df = df.drop(columns=["date_only", "lag_24h", "lag_168h"])
        df = df.ffill().fillna(0)

        out = (
            df[["ds", "business_day", *LGBM_DA_96_COLS]].copy()
            if "business_day" in df.columns
            else df[["ds", *LGBM_DA_96_COLS]].copy()
        )
        # 补 business_day 派生
        out["business_day"] = [self.res.business_day_from_timestamp(ts) for ts in df["ds"]]
        out = out[["ds", "business_day", *LGBM_DA_96_COLS]]

        self.dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(self.matrix_path, index=False)
        self._da = out

    def slice_da(self, target_date: str, start_date: str | None = None) -> pd.DataFrame:
        """切片：返回 <= target_date 的特征（训练窗由外部切）。只读副本。"""
        if self._da is None:
            self.ensure()
        df = self._da[self._da["business_day"] <= target_date].copy()
        if start_date:
            df = df[df["business_day"] >= start_date]
        return df

    def get_feature_columns(self) -> list[str]:
        return LGBM_DA_96_COLS
