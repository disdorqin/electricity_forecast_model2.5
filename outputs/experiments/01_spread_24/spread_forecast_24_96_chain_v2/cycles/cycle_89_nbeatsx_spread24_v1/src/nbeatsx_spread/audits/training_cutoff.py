from __future__ import annotations

import pandas as pd

from ..contracts import latest_complete_label_day
from .common import AuditResult, result


def audit_training_cutoff(target_day: str, training_days: list[str] | tuple[str, ...]) -> AuditResult:
    latest = pd.Timestamp(latest_complete_label_day(target_day))
    bad = [d for d in training_days if pd.Timestamp(d) > latest]
    return result("training_label_cutoff", not bad, f"latest_allowed={latest.date()} observed_latest={max(training_days) if training_days else None}; bad={bad[:3]}")
