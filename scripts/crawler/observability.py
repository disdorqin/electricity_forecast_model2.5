from __future__ import annotations

"""轻量级运行报告。

``crawler.log`` 负责详细人工排查；本模块只保存可检索的阶段/日期摘要，
并在每次重要状态变化后原子更新一个累计 ``report.json``。
"""

import hashlib
import json
import os
import re
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SECRET_KEY_RE = re.compile(
    r"(cookie|authorization|token|password|passwd|ukey|pin|secret|ticket)", re.I
)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if not SECRET_KEY_RE.search(str(k))}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def cookie_summary(cookie: str) -> dict[str, Any]:
    names = []
    for part in str(cookie or "").split(";"):
        if "=" in part:
            name = part.split("=", 1)[0].strip()
            if name:
                names.append(name)
    raw = str(cookie or "").encode("utf-8", errors="replace")
    return {
        "present": bool(cookie),
        "length": len(cookie or ""),
        "names": sorted(set(names)),
        "sha256": hashlib.sha256(raw).hexdigest()[:12] if cookie else "",
    }


class RunReport:
    """累计运行报告；不保存密钥和完整接口响应。"""

    def __init__(self, path: str | Path, *, build_version: str = "", args: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.data = self._load()
        run = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": None,
            "build_version": build_version,
            "args": _safe(args or {}),
            "status": "START",
            "stages": {},
            "dates": {},
            "events": [],
            "summary": {},
        }
        self.data.setdefault("schema_version", 1)
        self.data.setdefault("runs", []).append(run)
        self.data["last_run_id"] = self.run_id
        self._flush()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "last_run_id": None, "runs": []}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and isinstance(value.get("runs"), list):
                return value
        except Exception:
            # 不覆盖旧报告；创建一个恢复事件，详细错误仍由 crawler.log 保存。
            pass
        return {"schema_version": 1, "last_run_id": None, "runs": [], "recovered": True}

    @property
    def current(self) -> dict[str, Any]:
        return self.data["runs"][-1]

    def _flush(self) -> None:
        fd, name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
            os.replace(name, self.path)
        finally:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass

    def stage(self, name: str, status: str, **details: Any) -> None:
        self.current["stages"][name] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **_safe(details),
        }
        self._flush()

    def date(self, date_str: str, status: str, **details: Any) -> None:
        self.current["dates"][date_str] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **_safe(details),
        }
        self._flush()

    def event(self, level: str, code: str, message: str, **details: Any) -> None:
        self.current["events"].append({
            "time": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "code": code,
            "message": message,
            **_safe(details),
        })
        self._flush()

    def finish(self, status: str, **summary: Any) -> None:
        self.current["status"] = status
        self.current["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.current["summary"] = _safe(summary)
        self._flush()

    def exception(self, stage: str, exc: BaseException, **details: Any) -> None:
        self.stage(
            stage,
            "FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(limit=12),
            **details,
        )
