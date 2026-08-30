from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import AuditResult, result


def load_holdout_registry(path: str | Path) -> dict:
    """Load the independently frozen holdout registry."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"holdout registry missing: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data.get("lockboxes"), list):
        raise ValueError("holdout registry must contain lockboxes")
    return data


def audit_holdout_registry(target_days: list[str], registry: dict) -> AuditResult:
    """Fail closed when a requested target day intersects a registered lockbox."""
    requested = set(target_days)
    protected: set[str] = set()
    for box in registry["lockboxes"]:
        protected.update(d.strftime("%Y-%m-%d") for d in pd.date_range(box["start_date"], box["end_date"], freq="D"))
    overlap = sorted(requested & protected)
    ok = registry.get("status") == "ACTIVE" and not overlap
    return result("final_holdout_registry", ok, f"status={registry.get('status')}; overlap={overlap}")


def audit_holdout(final_holdout_touched: bool, target_days: list[str] | None = None, final_holdout_days: list[str] | None = None) -> AuditResult:
    overlap = set(target_days or []).intersection(final_holdout_days or [])
    ok = not final_holdout_touched and not overlap
    return result("final_holdout_untouched", ok, f"touched={final_holdout_touched}; overlap={sorted(overlap)}")
