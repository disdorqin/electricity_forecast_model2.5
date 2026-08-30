from __future__ import annotations

from ..contracts import STRICT34
from ..data.origin_index import OriginWindow, assert_origin_window, build_origin_window
from .common import AuditResult, result


def audit_origin(target_day: str, window: OriginWindow | None = None) -> AuditResult:
    try:
        w = window or build_origin_window(target_day, STRICT34)
        assert_origin_window(w)
        return result("origin_alignment", True, f"origin={w.origin_timestamp.isoformat()} target_day={target_day}")
    except Exception as exc:
        return result("origin_alignment", False, str(exc))
