"""Pickle-safe helpers for scheduler contract tests."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def write_marker_task(output_path: str, marker_path: str, sleep_seconds: float = 0.05) -> None:
    """Write a timestamped marker; intentionally has no model dependencies."""
    logger.info("scheduler marker task starting: %s", Path(output_path).name)
    marker = Path(marker_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": Path(output_path).name, "event": "start", "time": time.time()}) + "\n")
    time.sleep(float(sleep_seconds))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")
    with marker.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": path.name, "event": "end", "time": time.time()}) + "\n")
    logger.info("scheduler marker task done: %s", path.name)


def fail_task(**_: object) -> None:
    """Pickle-safe intentional failure for scheduler isolation tests."""
    raise RuntimeError("intentional scheduler test failure")
