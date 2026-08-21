"""Canonical data locations for the 24-point and 96-point domains.

The repository used to put both resolutions directly under ``data/``.  This
module is the single resolver for production defaults during the migration to
the explicit ``data/24`` and ``data/96`` domains.  The old paths remain as a
read-only compatibility fallback so an existing checkout can be upgraded
without losing a runnable chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATA_24_ROOT = DATA_ROOT / "24"
DATA_96_ROOT = DATA_ROOT / "96"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


@dataclass(frozen=True)
class DataLayout:
    """Authoritative and model-input paths for one project checkout."""

    data_root: Path = DATA_ROOT
    hourly_root: Path = DATA_24_ROOT
    quarter_root: Path = DATA_96_ROOT

    @property
    def hourly_xlsx(self) -> Path:
        return _first_existing(
            self.hourly_root / "canonical" / "shandong_pmos_hourly.xlsx",
            self.data_root / "shandong_pmos_hourly.xlsx",
        )

    @property
    def hourly_csv(self) -> Path:
        return _first_existing(
            self.hourly_root / "canonical" / "shandong_pmos_hourly.csv",
            self.data_root / "shandong_pmos_hourly.csv",
        )

    @property
    def authoritative_96_actual_csv(self) -> Path:
        """甲方 96 点实际数据；不得当作含价格的模型宽表使用。"""
        return _first_existing(
            self.quarter_root / "authoritative" / "pmos_96_全量.csv",
            self.data_root / "pmos_96_全量.csv",
        )

    @property
    def model_96_xlsx(self) -> Path:
        """Clean 96-point model input; never silently fall back to quarantined history."""
        return self.quarter_root / "model_input" / "shandong_pmos_96_model_input_clean.xlsx"

    @property
    def remote_96_root(self) -> Path:
        return _first_existing(
            self.quarter_root / "remote",
            self.data_root / "remote_96",
        )

    @property
    def sync_24_root(self) -> Path:
        return PROJECT_ROOT / "outputs" / "24" / "sync"

    @property
    def sync_96_root(self) -> Path:
        return PROJECT_ROOT / "outputs" / "96" / "sync"


DATA = DataLayout()


def data_path(resolution: str, kind: str = "model") -> Path:
    """Resolve a production data path without duplicating root literals."""
    if resolution == "15min":
        if kind in {"actual", "authoritative"}:
            return DATA.authoritative_96_actual_csv
        if kind in {"remote", "remote_root"}:
            return DATA.remote_96_root
        return DATA.model_96_xlsx
    if kind == "csv":
        return DATA.hourly_csv
    return DATA.hourly_xlsx


def ensure_data_directories() -> None:
    """Create only the stable domain directories; never create data files."""
    for path in (
        DATA_24_ROOT / "canonical",
        DATA_24_ROOT / "reference",
        DATA_24_ROOT / "quarantine",
        DATA_96_ROOT / "authoritative",
        DATA_96_ROOT / "model_input",
        DATA_96_ROOT / "remote",
        DATA_96_ROOT / "derived",
        DATA_96_ROOT / "quarantine",
    ):
        path.mkdir(parents=True, exist_ok=True)
