"""浏览器运行环境治理：profile 锁检测、残留进程识别、安全清理。

背景（2026-09-27 复盘 + 业界调研）：
    Chrome/Edge 用 ``SingletonLock`` / ``SingletonSocket`` / ``SingletonCookie``
    保证同一 ``--user-data-dir`` 单实例。**正常退出会自动清理；强杀/崩溃不会**，
    残留锁会让下次启动的新进程"以为目录被占用"而立即退出（code=0），
    表现为「浏览器启动了但调试端口不可用」——极易被误诊为网络故障。

安全原则（用户拍板 2026-09-27）：
    1. **优先复用**：不主动杀任何浏览器进程；
    2. 只有「存在残留锁文件 **且** 无存活进程持有」时才清理锁文件（这是无主垃圾）；
    3. 若确有存活进程持有 profile → 只报告，交给上层换浏览器/等待，绝不 kill。

只用标准库 + 系统自带命令（wmic / PowerShell / tasklist），不引入 psutil 依赖。

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .diagnose import diagnose_environment
from .codes import Diagnosis

#: Chrome/Edge profile 单实例锁文件名（相对 profile 根目录）
SINGLETON_FILES: tuple[str, ...] = ("SingletonLock", "SingletonCookie", "SingletonSocket")

#: 受监控的浏览器可执行文件名
BROWSER_EXECUTABLES: tuple[str, ...] = ("chrome.exe", "msedge.exe")

_SUBPROCESS_TIMEOUT = 12


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    name: str
    command_line: str = ""

    def holds_profile(self, profile_dir: str) -> bool:
        """该进程是否正在使用指定 profile 目录（大小写不敏感、路径分隔符归一）。"""
        if not self.command_line:
            return False
        norm_cmd = self.command_line.replace("/", "\\").lower()
        norm_profile = str(profile_dir).replace("/", "\\").lower().rstrip("\\")
        return norm_profile and norm_profile in norm_cmd


@dataclass
class ProfileLockReport:
    profile_dir: str
    lock_files: list[str] = field(default_factory=list)
    live_holders: list[int] = field(default_factory=list)
    port_reachable: bool = False
    port: int | None = None
    error: str = ""

    @property
    def has_orphan_locks(self) -> bool:
        """有锁文件但没人持有 → 属于可安全清理的垃圾。"""
        return bool(self.lock_files) and not self.live_holders

    @property
    def busy(self) -> bool:
        return bool(self.live_holders)

    def diagnose(self) -> Diagnosis:
        return diagnose_environment(
            lock_files=self.lock_files,
            live_holders=self.live_holders,
            port_reachable=self.port_reachable,
            port=self.port,
        )


# ─────────────────────────────────────────────────────────────────────────
# 进程枚举（多后端 fallback，全部只读）
# ─────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT,
            encoding="utf-8", errors="replace",
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        return proc.stdout or ""
    except Exception:  # noqa: BLE001 —— 探测失败不应中断主流程
        return ""


def _processes_via_powershell() -> list[ProcessInfo]:
    script = (
        "Get-CimInstance Win32_Process "
        "| Where-Object { $_.Name -match 'chrome|msedge' } "
        "| Select-Object ProcessId,Name,CommandLine "
        "| ConvertTo-Csv -NoTypeInformation"
    )
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    result: list[ProcessInfo] = []
    reader = csv.reader(io.StringIO(out))
    header: list[str] | None = None
    for row in reader:
        if not row:
            continue
        if header is None:
            header = [c.strip().strip('"') for c in row]
            continue
        cells = [c.strip().strip('"') for c in row]
        record = dict(zip(header, cells))
        try:
            pid = int(record.get("ProcessId") or 0)
        except ValueError:
            continue
        if pid:
            result.append(ProcessInfo(pid, record.get("Name", ""), record.get("CommandLine", "")))
    return result


def _processes_via_wmic() -> list[ProcessInfo]:
    out = _run(["wmic", "process", "get", "ProcessId,Name,CommandLine", "/format:csv"])
    result: list[ProcessInfo] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "ProcessId" in line and "Name" in line:
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        # CSV 依次为 Node,CommandLine,Name,ProcessId
        command_line = ",".join(parts[1:-2])
        name = parts[-2].strip()
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if name.lower() in BROWSER_EXECUTABLES:
            result.append(ProcessInfo(pid, name, command_line))
    return result


def list_browser_processes() -> list[ProcessInfo]:
    """列出所有 chrome/msedge 进程（含命令行）。只读，失败返回空列表。"""
    if os.name != "nt":
        return []
    procs = _processes_via_powershell()
    if not procs:
        procs = _processes_via_wmic()
    return procs


def find_profile_holders(profile_dir: str | Path) -> list[int]:
    """找出命令行里引用了该 profile 目录的浏览器进程 PID。

    注意：Chrome/Edge 会派生子进程（渲染/GPU），它们的命令行通常**也带**
    ``--user-data-dir``，因此这里会把同族进程都列出来，供上层判断"是否真的有人在用"。
    """
    profile_dir = str(profile_dir)
    return [p.pid for p in list_browser_processes() if p.holds_profile(profile_dir)]


# ─────────────────────────────────────────────────────────────────────────
# 锁文件与端口
# ─────────────────────────────────────────────────────────────────────────

def find_lock_files(profile_dir: str | Path) -> list[str]:
    """列出 profile 目录下存在的 Singleton* 锁文件（只读）。"""
    root = Path(profile_dir)
    if not root.exists():
        return []
    return [name for name in SINGLETON_FILES if (root / name).exists()]


def probe_port(port: int, *, timeout: float = 1.5) -> bool:
    """TCP 探测本地调试端口是否在监听（只读）。"""
    import socket

    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def inspect_profile(profile_dir: str | Path, *, port: int | None = None) -> ProfileLockReport:
    """一站式探测：锁文件 / 存活持有者 / 端口。只读，不修改任何东西。"""
    root = Path(profile_dir)
    report = ProfileLockReport(profile_dir=str(root), port=port)
    try:
        report.lock_files = find_lock_files(root)
        report.live_holders = find_profile_holders(root)
        report.port_reachable = probe_port(port) if port else False
    except Exception as exc:  # noqa: BLE001
        report.error = f"{type(exc).__name__}: {exc}"
    return report


def clean_orphan_locks(profile_dir: str | Path, *, dry_run: bool = True) -> dict:
    """清理**无主**锁文件（无存活进程持有时）。

    :param dry_run: True 只报告不删除（默认）；False 才实际删除。
    :returns: ``{"removed": [...], "skipped": [...], "reason": "..."}``

    绝不在存在存活持有者时删除——那会破坏正在运行的浏览器。
    """
    root = Path(profile_dir)
    locks = find_lock_files(root)
    holders = find_profile_holders(root)
    result: dict = {"profile_dir": str(root), "removed": [], "skipped": list(locks), "reason": ""}

    if not locks:
        result["reason"] = "no_lock_files"
        return result
    if holders:
        result["reason"] = f"live_processes_hold_profile:{holders[:5]}"
        return result
    if dry_run:
        result["reason"] = "dry_run"
        return result

    for name in locks:
        path = root / name
        try:
            if path.is_dir():
                continue  # 目录形态的锁不动（少见，保守跳过）
            path.unlink()
            result["removed"].append(name)
            result["skipped"].remove(name)
        except Exception as exc:  # noqa: BLE001
            result["reason"] = f"unlink_failed:{name}:{type(exc).__name__}"
    result.setdefault("reason", "")
    if not result["reason"]:
        result["reason"] = "cleaned"
    return result


# ─────────────────────────────────────────────────────────────────────────
# 毒瘤孤儿回收：端口活着、但活动页是死页面(favicon/404/错误页)的自动化浏览器
#
# 为什么需要（2026-09-28 血泪复盘）：
#   残留的自动化 Chrome 会带着 ``--user-data-dir=<我们的 profile>`` 长期监听调试端口，
#   但活动页可能停在 ``favicon.ico → 404`` 这类死页面。此时：
#     ① 认证层会把它误判成「健康可复用」→ 死等 600s（表现为整轮卡死）；
#     ② 它持有 profile 单实例锁 → 新浏览器起不来(bootstrap_profile_maybe_locked)。
#   唯一出路是把它回收掉，释放端口与 profile 锁。**只在「配置里的调试端口」上操作**，
#   且优先走 CDP ``Browser.close`` 优雅关闭（浏览器会自行清理 Singleton* 锁）。
# ─────────────────────────────────────────────────────────────────────────

def pids_listening_on_port(port: int) -> list[int]:
    """返回正在 LISTEN 指定本地端口的进程 PID（只读，走 netstat）。"""
    out = _run(["netstat", "-ano"])
    pids: set[int] = set()
    suffix = f":{int(port)}"
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].endswith(suffix):
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    continue
    return sorted(pids)


def _cdp_browser_close(port: int, *, timeout: float = 3.0) -> bool:
    """通过 CDP 的 ``Browser.close`` 优雅关闭该端口的浏览器；成功返回 True。"""
    try:
        import json as _json
        import urllib.request as _req

        with _req.urlopen(f"http://127.0.0.1:{int(port)}/json/version", timeout=timeout) as resp:
            version = _json.loads(resp.read().decode("utf-8", errors="replace"))
        ws_url = str((version or {}).get("webSocketDebuggerUrl") or "")
        if not ws_url:
            return False
        import websocket  # 打包时已在 hiddenimports（websocket-client）

        ws = websocket.create_connection(ws_url, timeout=timeout)
        try:
            ws.send(_json.dumps({"id": 1, "method": "Browser.close"}))
        finally:
            try:
                ws.close()
            except Exception:  # noqa: BLE001
                pass
        return True
    except Exception:  # noqa: BLE001 —— 优雅关闭失败不应抛错，交 taskkill 兜底
        return False


def terminate_debug_browser(port: int, *, profile_dir: str | Path = "",
                            dry_run: bool = False) -> dict:
    """回收指定调试端口上的浏览器（优雅关闭优先，taskkill 兜底）。返回动作报告。

    :returns: ``{"port","graceful","killed","remaining","reason"}``
    """
    import time

    result: dict = {"port": int(port), "graceful": False, "killed": [],
                    "remaining": [], "reason": ""}
    if not probe_port(port):
        result["reason"] = "port_closed"
        return result
    if dry_run:
        result["reason"] = "dry_run"
        return result

    if _cdp_browser_close(port):
        result["graceful"] = True
        for _ in range(6):
            time.sleep(0.5)
            if not probe_port(port):
                break
    if not probe_port(port):
        result["reason"] = "closed_by_cdp" if result["graceful"] else "closed"
        return result

    pids: set[int] = set(pids_listening_on_port(port))
    if profile_dir:
        pids.update(find_profile_holders(profile_dir))
    for pid in sorted(pids):
        _run(["taskkill", "/PID", str(pid), "/T", "/F"])
        result["killed"].append(pid)
    time.sleep(1.0)
    result["remaining"] = pids_listening_on_port(port)
    result["reason"] = "killed" if not result["remaining"] else "still_listening"
    return result


def is_windows() -> bool:
    return sys.platform == "win32"
