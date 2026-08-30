from __future__ import annotations

from ..contracts import STRICT34
from ..data.origin_index import OriginWindow
from .common import AuditResult, result


def audit_horizon(window: OriginWindow) -> AuditResult:
    ok = len(window.forecast_timestamps) == STRICT34.horizon and len(window.bridge_timestamps) == STRICT34.bridge_hours and len(window.scored_timestamps) == STRICT34.scored_hours and window.forecast_timestamps[0] > window.origin_timestamp and window.forecast_timestamps[-1].date() >= window.scored_timestamps[-1].date()
    return result("horizon34_alignment", ok, f"bridge={len(window.bridge_timestamps)} scored={len(window.scored_timestamps)} total={len(window.forecast_timestamps)}")
