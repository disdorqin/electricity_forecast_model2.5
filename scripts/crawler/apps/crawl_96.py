#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""96 点市场数据爬虫 —— **应用层主入口**（含防御编排）。

架构约定：
    - 实现在 ``scripts/crawler/collect/crawl_96_local.py``；
    - 本文件是**唯一入口**，负责：修路径 → 防御预检 → 转发实现 → 失败诊断；
    - ``collect/`` 的实现模块**不被修改**，这是"改入口不改实现"的安全边界。

为什么防御放在这一层：
    96 定时任务最常见的失败是「残留进程 / profile 锁 → 浏览器启动后连不上」。
    ``preflight()`` 会在**启动浏览器之前**做环境治理（清理无主锁、健康检查、
    复用判定），把这一大类故障在进入实现模块前就化解掉。

用法与原来完全一致::

    python scripts/crawler/apps/crawl_96.py                  # 补最近 14 天
    python scripts/crawler/apps/crawl_96.py --date 2026-09-24
    python scripts/crawler/apps/crawl_96.py --db-check

[2026-09-27] 由「纯薄封装」升级为主入口（内嵌 Guardian 防御编排）。
"""

from __future__ import annotations

import logging
import sys
import json
import os
from pathlib import Path

_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    _BASE_DIR = Path(sys.executable).parent.resolve()
else:
    # apps/crawl_96.py → [0]apps [1]crawler [2]scripts [3]仓库根
    # 与 collect/crawl_96_local.py 的推导结果**完全相同**，这是安全转发的前提。
    _BASE_DIR = Path(__file__).resolve().parents[3]

for _p in (str(_BASE_DIR), str(_BASE_DIR / "scripts" / "crawler")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 实现模块（必须在 sys.path 修好后导入）
from scripts.crawler.collect.crawl_96_local import (  # noqa: E402
    BUILD_VERSION as _IMPL_BUILD,
    main as _impl_main,
)

logger = logging.getLogger("apps.crawl_96")

#: 防御开关：可用环境变量关闭（PMOS_DISABLE_RESILIENCE=1），保证随时能退回原始行为
_RESILIENCE_ENABLED = str(os.environ.get("PMOS_DISABLE_RESILIENCE", "")).strip() not in ("1", "true", "True")

#: 认证类失败的判据（用于决定是否值得「丢弃失效会话后重试」）
_AUTH_FAILURE_TOKENS = ("401", "403", "auth", "unauthorized", "forbidden", "登录", "会话")


def _env_hints() -> tuple[str, list[int]]:
    """从配置里取出 profile 目录与候选调试端口（尽力而为，失败不阻断）。

    配置位置随运行形态不同：打包后与 exe 同目录；源码模式下在 ``scripts/crawler/``。
    两个候选都试一遍。
    """
    profile = ""
    ports: list[int] = []
    try:
        from scripts.crawler.auth.auto_crawler.config import AuthConfig

        candidates = [
            _BASE_DIR / "config.json",                        # frozen：与 exe 同目录
            _BASE_DIR / "scripts" / "crawler" / "config.json",  # 源码模式
        ]
        cfg = None
        for path in candidates:
            if path.exists():
                cfg = AuthConfig.from_file(path)
                break
        if cfg is not None:
            profile = str(getattr(cfg, "browser_profile_dir", "") or "")
            base = int(getattr(cfg, "debug_port", 0) or 0)
            if base:
                ports = [base, base + 1, base + 2]
    except Exception as exc:  # noqa: BLE001 —— 配置读取失败不应阻断爬取
        logger.debug("resilience: 读取配置失败，跳过环境预检: %s", exc)
    return profile, ports


def _ensure_logging() -> None:
    """保证防御日志可见，且**不影响**实现模块的日志配置。

    `preflight()` 在转发给实现模块之前执行，此时实现模块可能尚未调用
    ``logging.basicConfig``，root logger 仍是默认 WARNING → INFO 级防御日志会被丢弃。
    这里只给本模块 logger 挂独立 handler 并关闭向上传播，不抢占 basicConfig。
    """
    lg = logging.getLogger("apps.crawl_96")
    if lg.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False


def _preflight() -> None:
    """开跑前的防御预检（清理无主锁 + 健康检查 + 复用判定）。"""
    if not _RESILIENCE_ENABLED:
        return
    _ensure_logging()
    try:
        from scripts.crawler.resilience import Guardian, STRATEGY_REUSE_FIRST

        profile, ports = _env_hints()
        if not profile and not ports:
            # 既无 profile 也无端口可探，说明本环境无预检目标（如本机仅做 --help）。
            # 此时不做预检，避免输出"无锁无持有者→启动超时"这类误导性结论。
            logger.debug("resilience: 无预检目标，跳过环境预检")
            return
        guardian = Guardian(
            profile_dir=profile or None,
            port_candidates=ports,
            strategy=STRATEGY_REUSE_FIRST,
            auto_clean_locks=True,   # 只清「无存活进程持有」的锁文件，绝不杀进程
        )
        plan = guardian.preflight()
        terminated = getattr(plan, "terminated", []) or []
        logger.info(
            "resilience preflight: decision=%s root_cause=%s lock_files=%s live_holders=%s reuse_port=%s terminated=%s",
            plan.decision, plan.diagnosis.code,
            plan.lock_files, plan.live_holders, plan.reuse_port,
            [t.get("port") for t in terminated],
        )
        for act in terminated:
            logger.warning(
                "resilience: 已回收毒瘤孤儿浏览器 port=%s reason=%s first_url=%s "
                "（它会让复用死等 600s 且占住 profile 锁）",
                act.get("port"), act.get("reason"), act.get("first_url", "")[:120],
            )
    except Exception as exc:  # noqa: BLE001 —— 防御失败绝不能影响主流程
        logger.warning("resilience preflight 失败（已忽略，继续正常流程）: %s", exc)


def _run_impl(argv: list[str] | None = None) -> int:
    """调用实现模块并归一化退出码（实现失败时会 sys.exit）。"""
    try:
        return int(_impl_main() or 0)
    except SystemExit as exc:  # 实现内部以 sys.exit 终止（含 --help / 失败）
        return int(exc.code or 0)


def _output_dir() -> Path:
    """实现模块的输出目录（report.json 所在），随运行形态不同。"""
    for cand in (_BASE_DIR / "output_96", _BASE_DIR / "outputs" / "crawl" / "runtime_96"):
        if cand.exists():
            return cand
    return _BASE_DIR / "output_96"


def _read_runs() -> list:
    try:
        data = json.loads((_output_dir() / "report.json").read_text(encoding="utf-8"))
        return data.get("runs") or []
    except Exception:  # noqa: BLE001
        return []


def _last_run_is_auth_failure() -> bool:
    """判断最近一次运行**是否属于认证类失败**。

    只有认证类失败才值得「丢弃失效会话 + 重登」；其余失败重试无意义
    （与用户拍板的"找不到核心问题，试一百次也没用"一致）。
    """
    runs = _read_runs()
    if not runs:
        return False
    run = runs[-1]
    if str(run.get("status") or "").upper() != "FAIL":
        return False
    summary = run.get("summary") or {}
    # 实现层实际产出 `auth`（布尔），不是 `qctc_auth`；两者都认，避免判定永远落空
    if summary.get("qctc_auth") is False or summary.get("auth") is False:
        return True
    reason = str(summary.get("reason") or "").lower()
    if any(tok in reason for tok in _AUTH_FAILURE_TOKENS):
        return True
    for event in (run.get("events") or []):
        if str(event.get("code") or "") in ("QCTC_AUTH_REJECTED", "QCTC_HTTP_FAIL",
                                            "AUTH_STALE_SESSION_DETECTED",
                                            "AUTH_ATTEMPT_FAILED", "AUTH_FAILED"):
            return True
    return False


def _config_path() -> Path | None:
    for cand in (_BASE_DIR / "config.json",                          # frozen：与 exe 同目录
                 _BASE_DIR / "scripts" / "crawler" / "config.json",  # 源码模式
                 _BASE_DIR / "dist" / "crawler" / "config.json"):    # 源码模式下的部署目录
        if cand.exists():
            return cand
    return None


def _retry_after_discarding_session(argv: list[str] | None) -> int:
    """丢弃失效会话后重跑一次（走完整重新登录）。

    为什么必须禁用复用：复用到的常开浏览器**门户会话可能早已过期**（历史与本次
    故障都是如此——1 秒内判 logged_in，后端其实 401）。若不丢弃它，重试仍会复用
    同一个失效浏览器，永远失败。
    """
    path = _config_path()
    restore: tuple[bool, object, Path] | None = None
    if path is not None:
        try:
            original = json.loads(path.read_text(encoding="utf-8"))
            mutated = dict(original)
            mutated["browser_reuse"] = False
            path.write_text(json.dumps(mutated, ensure_ascii=False, indent=2), encoding="utf-8")
            restore = ("browser_reuse" in original, original.get("browser_reuse"), path)
            logger.info("resilience: 已临时设置 %s 的 browser_reuse=false（丢弃失效会话）", path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("resilience: 无法临时禁用会话复用（%s），重试可能仍复用旧会话", exc)
    try:
        _preflight()
        return _run_impl(argv)
    finally:
        if restore is not None:
            had_key, old_value, cfg = restore
            try:
                # 重新读取：实现可能已把 Cookie 写回该文件，只还原 browser_reuse 这一个键
                current = json.loads(cfg.read_text(encoding="utf-8"))
                if had_key:
                    current["browser_reuse"] = old_value
                else:
                    current.pop("browser_reuse", None)
                cfg.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("resilience: 已还原 %s 的 browser_reuse 设置", cfg.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("resilience: 还原 browser_reuse 失败（不影响数据正确性）: %s", exc)


def _db_config_path() -> Path | None:
    for cand in (_BASE_DIR / "db_config.json",
                 _BASE_DIR / "scripts" / "crawler" / "db_config.json"):
        if cand.exists():
            return cand
    return None


def _doctor() -> int:
    """``--doctor``：环境自检，一条命令回答「现在能不能跑通」。"""
    _ensure_logging()
    try:
        from scripts.crawler.resilience import run_doctor
    except Exception as exc:  # noqa: BLE001
        print(f"resilience 不可用，无法执行自检：{exc}")
        return 2
    profile, ports = _env_hints()
    return run_doctor(
        app_name="96 点市场数据爬虫",
        build_version=_IMPL_BUILD,
        config_path=_config_path(),
        db_config_path=_db_config_path(),
        profile_dir=profile,
        port_candidates=ports,
        output_dir=_output_dir(),
        printer=lambda text: print(text, flush=True),
    )


def _banner() -> None:
    """启动横幅：一眼看清版本、配置、输出去向。"""
    logger.info("=" * 64)
    logger.info("96 点市场数据爬虫 | 版本 %s", _IMPL_BUILD)
    logger.info("  配置 : %s", _config_path() or "(未找到)")
    logger.info("  输出 : %s", _output_dir())
    logger.info("  提示 : 想先做环境体检请运行  crawl_96_auto_v10.exe --doctor")
    logger.info("=" * 64)


def _conclusion(code: int, retried: bool) -> None:
    """结束摘要：一眼看清结果与是否触发过自动重试。"""
    logger.info("=" * 64)
    logger.info("运行结束 | 退出码=%s | %s", code, "成功" if code == 0 else "失败")
    if retried:
        logger.info("  已自动重试一次（丢弃失效会话后重新登录）")
    logger.info("  详细过程: crawler.log | 结构化结果: report.json")
    logger.info("=" * 64)


def main(argv: list[str] | None = None) -> int:
    """入口：防御预检 → 调用实现 → 认证失败则丢弃失效会话重试一次 → 失败诊断。"""
    _ensure_logging()

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--doctor" in raw_args:
        return _doctor()

    _banner()
    _preflight()

    runs_before = len(_read_runs())
    try:
        code = _run_impl(argv)
    except BaseException as exc:  # noqa: BLE001 —— _run_impl 已消化 SystemExit
        if _RESILIENCE_ENABLED:
            try:
                from scripts.crawler.resilience import diagnose_exception
                from scripts.crawler.resilience.codes import info

                d = diagnose_exception(exc)
                logger.error(
                    "resilience: 根因=%s 层=%s 可重试=%s 建议动作=%s | %s",
                    d.code, d.layer, d.retryable, info(d.code).action.value, d.reason,
                )
            except Exception:  # noqa: BLE001
                pass
        raise

    # 仅当「本轮新增了一次认证类失败」时才重试，避免误判历史失败或无意义重试
    retried = False
    if code != 0 and _RESILIENCE_ENABLED and len(_read_runs()) > runs_before and _last_run_is_auth_failure():
        logger.warning(
            "resilience: 本轮认证失败（复用的浏览器会话已失效）→ 丢弃该会话后重新登录，重试一次"
        )
        retried = True
        code = _retry_after_discarding_session(argv)

    _conclusion(code, retried)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
