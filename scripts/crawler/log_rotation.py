# -*- coding: utf-8 -*-
"""运行日志的「按天归档 + 只保留最近 N 天」策略（96 主爬虫 / AUX 辅助爬虫共用）。

背景
----
96 与 AUX 都以**批处理**方式运行（每天跑一次就退出），日志此前用
``FileHandler(mode="a")`` 无限追加到固定文件名（``crawler.log`` /
``aux_crawler.log``），长期把单文件撑到几十 MB（AUX 一度 65 MB），人工排查时
打开极慢。这里提供统一策略：

- **归档**：启动时若现有日志文件的「最后写入日期」早于今天，就把它重命名为
  ``<name>.<YYYY-MM-DD>``；``<name>`` 因此始终只承载「最近一次运行」的日志。
  文件名本身不变，README / report / 提示文本里对 ``crawler.log`` 的引用全部继续有效。
- **清理**：删除 ``<name>.<YYYY-MM-DD>`` 中日期早于保留窗口的旧备份。

为什么不直接用 ``TimedRotatingFileHandler``
-----------------------------------------
它只在进程**跨过午夜仍在写入**时才触发滚动。本场景是「每天跑一次就退出」，
几乎永远到不了午夜，日志会一直追加到同一个文件——达不到「按天保留」的目的。
因此在**每次启动**时做一次归档/清理，语义正好对应「过一天运行自动更新」。

该模块被 96（``collect/crawl_96_local.py``）与 AUX
（``collect/crawl_disclosure_aux.py``）各自顶层导入，互不影响。
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path

__all__ = ["rotate_and_prune_log", "DEFAULT_KEEP_DAYS"]

#: 默认保留天数（保留今天及往前共 7 天的日志）
DEFAULT_KEEP_DAYS = 7


def rotate_and_prune_log(log_path: Path, *, keep_days: int = DEFAULT_KEEP_DAYS) -> list[Path]:
    """按天归档 ``log_path``，并清理保留窗口之外的旧日志备份。

    参数：
        log_path: 当前日志文件（如 ``crawler.log``）。
        keep_days: 保留最近多少天（含今天）。

    返回：
        本次被删除的旧日志路径列表（便于日志/测试观测；正常路径通常为空）。

    说明：
        - 该函数**尽力而为、绝不抛异常**：任何 IO 失败都退化为「继续追加原文件」，
          不阻断爬虫主流程。
        - ``log_path`` 的父目录必须已存在（调用方通常已 mkdir）。
    """
    log_path = Path(log_path)
    parent = log_path.parent
    name = log_path.name
    today = date.today()

    # ── 1) 归档：现有日志的最后写入日期不是今天 → 归档为 <name>.<该日期> ──
    try:
        if log_path.exists() and log_path.stat().st_size > 0:
            last_day = datetime.fromtimestamp(log_path.stat().st_mtime).date()
            if last_day < today:
                archive = parent / f"{name}.{last_day.isoformat()}"
                seq = 1
                while archive.exists():  # 极端情况下同名（同日多次归档）加序号
                    archive = parent / f"{name}.{last_day.isoformat()}.{seq}"
                    seq += 1
                log_path.replace(archive)
    except OSError:
        pass  # 归档失败 → 退化为继续追加原文件，不影响运行

    # ── 2) 清理：删除日期早于保留窗口的旧备份 ──
    cutoff = today - timedelta(days=max(1, int(keep_days)) - 1)  # 含今天共 keep_days 天
    pattern = re.compile(re.escape(name) + r"\.(\d{4}-\d{2}-\d{2})(?:\.\d+)?$")
    removed: list[Path] = []
    try:
        candidates = list(parent.glob(f"{name}.*"))
    except OSError:
        candidates = []
    for candidate in candidates:
        match = pattern.search(candidate.name)
        if not match:
            continue
        try:
            if date.fromisoformat(match.group(1)) < cutoff:
                candidate.unlink()
                removed.append(candidate)
        except (ValueError, OSError):
            continue
    return removed
