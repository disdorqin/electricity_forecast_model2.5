"""断点续跑与失败清单（本地 JSON，无需外部服务）。

解决两个日常痛点：
    1. **自动断点续跑**：中断后不必人工查库算 ``--date/--lookback``，
       直接根据「已完成到哪天」算出续跑参数。
    2. **失败清单（DLQ）**：不可重试的失败（404 / 解析错误）落到本地清单，
       下次运行可优先补，避免"静默丢数据"。

刻意不耦合数据库：调用方只需把「已知最早完成日期」传进来。
存储位置由环境变量 ``PMOS_RESILIENCE_HOME`` 或调用方指定，默认 ``.resilience/``。

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

_DEFAULT_HOME = ".resilience"


def default_home(base_dir: str | Path | None = None) -> Path:
    """防御机制的工作目录：优先环境变量，其次调用方给定目录，最后当前目录。"""
    env = os.environ.get("PMOS_RESILIENCE_HOME")
    if env:
        return Path(env)
    root = Path(base_dir) if base_dir else Path.cwd()
    return root / _DEFAULT_HOME


def _atomic_write(path: Path, payload: str) -> None:
    """原子写：先写临时文件再 replace，避免崩溃留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ─────────────────────────────────────────────────────────────────────────
# 一、断点续跑
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResumePlan:
    date: str          # 传给爬虫的 --date（已完成边界的前一天）
    lookback: int      # 传给爬虫的 --lookback
    target_start: str
    earliest_done: str
    note: str = ""

    def as_args(self) -> list[str]:
        return ["--date", self.date, "--lookback", str(self.lookback)]


def plan_resume(
    *,
    earliest_done: date | str | None,
    target_start: date | str,
    padding: int = 3,
) -> ResumePlan:
    """根据「已补齐到的最早日期」算出续跑参数。

    :param earliest_done: 库里已有的最早业务日期；None 表示尚未开始
    :param target_start: 希望回补到的目标起点（如 2022-01-01）
    :param padding: 余量天数，多给几天避免边界漏日（幂等 upsert，无害）
    """
    start = target_start if isinstance(target_start, date) else date.fromisoformat(str(target_start))

    if earliest_done is None:
        anchor = date.today()
        note = "首次运行：从今天往前回补"
    else:
        done = earliest_done if isinstance(earliest_done, date) else date.fromisoformat(str(earliest_done))
        anchor = done - timedelta(days=1)   # 从已完成边界的前一天继续
        note = f"续跑：已完成边界 {done.isoformat()}"

    lookback = max(0, (anchor - start).days) + max(0, int(padding))
    return ResumePlan(
        date=anchor.isoformat(),
        lookback=lookback,
        target_start=start.isoformat(),
        earliest_done=(earliest_done.isoformat() if isinstance(earliest_done, date) else str(earliest_done or "")),
        note=note,
    )


# ─────────────────────────────────────────────────────────────────────────
# 二、失败清单（DLQ）
# ─────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PendingItem:
    kind: str          # 业务类型，如 "aux_record" / "96_day"
    key: str           # 业务键，如 "2024-08-16:dcst_tmp_load"
    code: str          # 根因码
    reason: str = ""
    attempts: int = 0

    def as_dict(self) -> dict:
        return {"kind": self.kind, "key": self.key, "code": self.code,
                "reason": self.reason, "attempts": self.attempts}


class PendingQueue:
    """本地 JSON 待补清单。同一 ``(kind, key)`` 重复入队只累加尝试次数。"""

    def __init__(self, path: str | Path | None = None, *, base_dir: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_home(base_dir) / "pending.json"
        self._items: dict[tuple[str, str], PendingItem] = {}
        self.load()

    # ── 读写 ────────────────────────────────────────────────────────────
    def load(self) -> None:
        self._items.clear()
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 清单损坏不应中断主流程
            return
        for entry in (raw.get("items") if isinstance(raw, dict) else raw) or []:
            try:
                item = PendingItem(
                    kind=str(entry["kind"]), key=str(entry["key"]),
                    code=str(entry.get("code") or "UNKNOWN"),
                    reason=str(entry.get("reason") or ""),
                    attempts=int(entry.get("attempts") or 0),
                )
            except Exception:  # noqa: BLE001
                continue
            self._items[(item.kind, item.key)] = item

    def save(self) -> None:
        payload = json.dumps(
            {"version": 1, "items": [i.as_dict() for i in self._items.values()]},
            ensure_ascii=False, indent=2,
        )
        _atomic_write(self.path, payload)

    # ── 操作 ────────────────────────────────────────────────────────────
    def add(self, *, kind: str, key: str, code: str, reason: str = "") -> PendingItem:
        existing = self._items.get((kind, key))
        attempts = (existing.attempts + 1) if existing else 1
        item = PendingItem(kind=kind, key=key, code=code, reason=reason[:500], attempts=attempts)
        self._items[(kind, key)] = item
        self.save()
        return item

    def remove(self, *, kind: str, key: str) -> bool:
        removed = self._items.pop((kind, key), None) is not None
        if removed:
            self.save()
        return removed

    def entries(self) -> list[PendingItem]:
        return list(self._items.values())

    def keys(self) -> set[str]:
        return {key for (_, key) in self._items}

    def __len__(self) -> int:
        return len(self._items)

    def summary(self) -> dict:
        by_code: dict[str, int] = {}
        for item in self._items.values():
            by_code[item.code] = by_code.get(item.code, 0) + 1
        return {"total": len(self._items), "by_code": by_code, "path": str(self.path)}
