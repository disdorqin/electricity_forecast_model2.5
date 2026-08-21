"""统一数据加载：parquet / csv / xlsx 自适应。

FeatureStore 把 30MB xlsx 转 parquet（16MB），各模型 load 改为读 parquet
（read_parquet ~137ms vs read_excel ~30s，提速 226x，零精度损失）。
本函数按扩展名分派，保留 xlsx/csv 兼容。
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_table(path: str | Path) -> pd.DataFrame:
    """读表：.parquet / .csv / .xlsx 自动分派。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in (".csv", ".txt"):
        # Prefer UTF-8 because pandas.to_csv() and the synthetic/regression
        # fixtures use it.  GBK/GB18030 remains supported for PMOS exports.
        # Never silently skip malformed rows in the canonical loader.
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
            try:
                return pd.read_csv(p, encoding=encoding, on_bad_lines="error")
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError(f"Unable to read table: {p}")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p, engine="openpyxl")
    # 无扩展名回退：尝试 parquet → csv → xlsx
    for candidate in (p.with_suffix(".parquet"), p, p.with_suffix(".xlsx")):
        if candidate.exists():
            return load_table(candidate)
    raise FileNotFoundError(f"数据文件不存在: {path}")
