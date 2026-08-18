"""Pickle-safe helpers for scheduler contract tests."""

from __future__ import annotations

import json
import time
from pathlib import Path


def write_marker_task(output_path: str, marker_path: str, sleep_seconds: float = 0.05) -> None:
    """Write a timestamped marker; intentionally has no model dependencies."""
    time.sleep(float(sleep_seconds))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")
    marker = Path(marker_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": path.name, "time": time.time()}) + "\n")
