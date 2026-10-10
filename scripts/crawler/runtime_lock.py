"""Small cross-platform process lock for the 96-point crawler runtime.

The lock is deliberately an OS file lock rather than a pid/sentinel check:
an interrupted process releases it automatically and a stale lock file does
not block the next run.

[V10-r1] 新增：防止多个爬虫进程同时操作同一浏览器/CDP、report、raw 和数据库。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class RuntimeLockError(RuntimeError):
    """Raised when another crawler process owns the runtime lock."""


class RuntimeLock:
    """Hold an exclusive, non-blocking lock on one byte of ``path``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None
        self._held = False

    def acquire(self) -> "RuntimeLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            # msvcrt.locking requires an existing byte; fcntl does not, but a
            # byte also makes the lock file useful for diagnostics.
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - exercised only on non-Windows CI
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            handle.close()
            raise RuntimeLockError(f"runtime lock is already held: {self.path}") from exc
        self._handle = handle
        self._held = True
        return self

    def release(self) -> None:
        if not self._held or self._handle is None:
            return
        try:
            self._handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised only on non-Windows CI
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None
            self._held = False

    def __enter__(self) -> "RuntimeLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def acquire_runtime_lock(path: str | Path) -> RuntimeLock:
    """Acquire ``path`` and return the context manager to be released."""

    return RuntimeLock(path).acquire()
