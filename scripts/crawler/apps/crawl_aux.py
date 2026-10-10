#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""辅助信息披露（AUX）爬虫 —— **应用层主入口**（含防御编排）。

架构约定（与 96 对称）：
    - 实现在 ``scripts/crawler/collect/crawl_disclosure_aux.py``；
    - 本文件是**唯一入口**：修路径 → 防御预检 → 转发实现 → 失败诊断；
    - ``collect/`` 的实现模块**不被修改**，这是"改入口不改实现"的安全边界。

为什么防御放在这一层：
    AUX 长跑（1500+ 天）最常见的失败是「残留进程 / profile 锁 → 浏览器连不上」以及
    「浏览器中途死亡 → 后续每个源白跑」。``preflight()`` 在启动浏览器之前做环境治理
    （清理无主锁、健康检查、复用优先），把前者在进入实现模块前化解。

用法与原来完全一致::

    python scripts/crawler/apps/crawl_aux.py --date 2024-08-17 --lookback 960 --source all-designed
    python scripts/crawler/apps/crawl_aux.py --db-check

[2026-10-08 AUX-V1-r12] 新增探索模式入口 ``--explore``：登录后被动录制网站请求，
替代「人工 F12 导 HAR 再搬运」。实现在 ``collect/crawl_disclosure_aux_explore.py``，
采集链零改动。

[2026-09-27] 由「纯薄封装」升级为主入口（内嵌 Guardian 防御编排）。
"""

from __future__ import annotations

import logging
import os
import sys
import json
from pathlib import Path

_FROZEN = getattr(sys, "frozen", False)

if _FROZEN:
    _BASE_DIR = Path(sys.executable).parent.resolve()
else:
    # apps/crawl_aux.py → [0]apps [1]crawler [2]scripts [3]仓库根
    # 与 collect/crawl_disclosure_aux.py 的推导结果**完全相同**，这是安全转发的前提。
    _BASE_DIR = Path(__file__).resolve().parents[3]

for _p in (str(_BASE_DIR), str(_BASE_DIR / "scripts" / "crawler")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 实现模块（必须在 sys.path 修好后导入）
from scripts.crawler.collect.crawl_disclosure_aux import (  # noqa: E402
    BUILD_VERSION as _IMPL_BUILD,
    main as _impl_main,
)

# [AUX-V1-r12] 探索模式：同样必须**顶层**导入（PyInstaller AST 静态收集，见下方防御块）。
# 它只在入口层被 --explore 分走，采集实现 crawl_disclosure_aux.py 零改动。
try:  # pragma: no cover
    from scripts.crawler.collect.crawl_disclosure_aux_explore import (  # noqa: E402
        EXPLORE_BUILD as _EXPLORE_BUILD,
        main as _explore_main,
    )
    _EXPLORE_AVAILABLE = True
    _explore_exc: Exception | None = None
except Exception as _exc:  # noqa: BLE001
    _EXPLORE_AVAILABLE = False
    _explore_exc = _exc

# ── 防御机制：顶层导入（**不要挪进函数**）──────────────────────────────
# PyInstaller 用 AST 静态分析收集模块；若放在函数内做延迟导入，AUX 的 spec
# 没有 collect_submodules，会把 resilience 漏掉，打包后防御静默失效。
# 放在顶层（哪怕包在 try 里）即可被静态发现。
try:  # pragma: no cover - 依赖可用性由部署环境保证
    from scripts.crawler.resilience import (  # noqa: E402
        Guardian,
        STRATEGY_REUSE_FIRST,
        diagnose_exception,
    )
    from scripts.crawler.resilience.codes import info as _code_info  # noqa: E402
    _RESILIENCE_AVAILABLE = True
except Exception as _res_exc:  # noqa: BLE001
    _RESILIENCE_AVAILABLE = False
    _res_exc = _res_exc

logger = logging.getLogger("apps.crawl_aux")

#: 防御开关：可用环境变量关闭（PMOS_DISABLE_RESILIENCE=1），保证随时能退回原始行为
_RESILIENCE_ENABLED = _RESILIENCE_AVAILABLE and str(
    os.environ.get("PMOS_DISABLE_RESILIENCE", "")
).strip() not in ("1", "true", "True")

#: 可恢复失败的判据（决定是否值得「丢弃会话 + 重新认证」重试一次）
_RETRYABLE_TOKENS = (
    "401", "403", "auth", "unauthorized", "forbidden", "会话", "登录",
    "浏览器启动后", "未打开可控", "bootstrap",
)
_RETRYABLE_ERROR_TYPES = (
    "AuxAuthRejected", "SessionExpiredError", "TimeoutError",
    "BrowserControlError", "BrowserLost",
)
_RETRYABLE_EVENT_CODES = (
    "AUX_BROWSER_RECOVERY_FAILED", "AUTH_LAYERS_EXHAUSTED",
    "BROWSER_BOOTSTRAP_FAILED", "AUTH_STALE_SESSION_DETECTED",
    "AUTH_ATTEMPT_FAILED", "AUTH_FAILED",
)


def _env_hints() -> tuple[str, list[int]]:
    """从配置里取出 profile 目录与候选调试端口（尽力而为，失败不阻断）。

    AUX 的配置文件位置随运行形态不同：
      - frozen：与 exe 同目录的 ``config_disclosure_aux.json``；
      - 源码模式：``dist/crawler/辅助信息披露爬虫/config_disclosure_aux.json``。
    """
    profile = ""
    ports: list[int] = []
    try:
        from scripts.crawler.auth.auto_crawler.config import AuthConfig
        from scripts.crawler.auth.auto_crawler.state_machine import AuthenticationStateMachine

        candidates = [
            _BASE_DIR / "config_disclosure_aux.json",
            _BASE_DIR / "config.json",
            _BASE_DIR / "dist" / "crawler" / "辅助信息披露爬虫" / "config_disclosure_aux.json",
        ]
        cfg = None
        for path in candidates:
            if path.exists():
                cfg = AuthConfig.from_file(path)
                break
        if cfg is not None:
            # 复用共享状态机的既有解析（相对路径 / frozen 基目录 / home 兜底全一致）
            profile = str(AuthenticationStateMachine(cfg)._profile_dir())
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
    这里只给本模块 logger 挂一个独立 handler 并关闭向上传播，
    既不抢占 basicConfig，也不会让日志重复输出。
    """
    lg = logging.getLogger("apps.crawl_aux")
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
        profile, ports = _env_hints()
        if not profile and not ports:
            logger.debug("resilience: 无预检目标，跳过环境预检")
            return
        guardian = Guardian(
            profile_dir=profile or None,
            port_candidates=ports,
            strategy=STRATEGY_REUSE_FIRST,
            auto_clean_locks=True,   # 只清「无存活进程持有」的锁文件（另：毒瘤孤儿才会被回收）
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
    """调用实现模块并归一化退出码（实现失败时会 sys.exit / return 1）。"""
    try:
        return int(_impl_main(argv) or 0)
    except SystemExit as exc:  # 实现内部以 sys.exit 终止（含 --help）
        return int(exc.code or 0)


def _output_dir() -> Path:
    """AUX 输出目录（``aux_report.json`` 所在），随运行形态不同。"""
    for cand in (_BASE_DIR / "output_aux",
                 _BASE_DIR / "dist" / "crawler" / "辅助信息披露爬虫" / "output_aux"):
        if cand.exists():
            return cand
    return _BASE_DIR / "output_aux"


def _read_runs() -> list:
    try:
        data = json.loads((_output_dir() / "aux_report.json").read_text(encoding="utf-8"))
        return data.get("runs") or []
    except Exception:  # noqa: BLE001
        return []


def _last_run_is_retryable_failure() -> bool:
    """判断最近一次运行是否属于「丢掉会话重来就可能恢复」的失败。

    覆盖两类（都是 AUX 长跑最常见的失败）：
      1. **认证类**：AuxAuthRejected / HTTP 401 / 会话失效 —— 参考 96 的口径；
      2. **浏览器启动类**：bootstrap 超时 / 未打开可控 PMOS 页面 —— preflight 清理
         无主锁之后重试通常能成功（历史上的 profile 锁问题正是这一类）。
    其余失败（网络、解析、用户中断）**不重试**，遵循"找不到核心问题，试一百次也没用"。
    """
    runs = _read_runs()
    if not runs:
        return False
    run = runs[-1]
    status = str(run.get("status") or "").upper()
    if status not in ("FAIL", "ERROR"):
        return False

    summary = run.get("summary") or {}
    if summary.get("auth") is False:   # 实现层实际产出 `auth`（布尔），不是 `reason`
        return True
    reason = str(summary.get("reason") or "").lower()
    if any(tok in reason for tok in _RETRYABLE_TOKENS):
        return True

    for stage in (run.get("stages") or {}).values():
        if not isinstance(stage, dict) or str(stage.get("status")) != "FAIL":
            continue
        etype = str(stage.get("error_type") or "")
        if etype in _RETRYABLE_ERROR_TYPES:
            return True

    for event in (run.get("events") or []):
        if str(event.get("code") or "") in _RETRYABLE_EVENT_CODES:
            return True
    return False


def _config_path() -> Path | None:
    for cand in (_BASE_DIR / "config_disclosure_aux.json",
                 _BASE_DIR / "config.json",
                 _BASE_DIR / "dist" / "crawler" / "辅助信息披露爬虫" / "config_disclosure_aux.json"):
        if cand.exists():
            return cand
    return None


def _retry_after_discarding_session(argv: list[str] | None) -> int:
    """丢弃失效会话后重跑一次（走完整重新认证）。

    与 96 同理：复用到的常开浏览器**门户会话可能早已过期**，若不禁用复用，
    重试仍会挂回同一个失效会话。这里临时把 ``browser_reuse`` 置 false，
    跑完再还原（只还原这一个键，不动实现写回的 Cookie）。
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
                 _BASE_DIR / "dist" / "crawler" / "辅助信息披露爬虫" / "db_config.json"):
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
        app_name="AUX 辅助信息披露爬虫",
        build_version=_IMPL_BUILD,
        config_path=_config_path(),
        db_config_path=_db_config_path(),
        profile_dir=profile,
        port_candidates=ports,
        output_dir=_output_dir(),
        printer=lambda text: print(text, flush=True),
    )


def _banner() -> None:
    logger.info("=" * 64)
    logger.info("AUX 辅助信息披露爬虫 | 版本 %s", _IMPL_BUILD)
    logger.info("  配置 : %s", _config_path() or "(未找到)")
    logger.info("  输出 : %s", _output_dir())
    logger.info("  提示 : 想先做环境体检请运行  crawl_disclosure_aux_v1.exe --doctor")
    logger.info("=" * 64)


def _conclusion(code: int, retried: bool) -> None:
    logger.info("=" * 64)
    logger.info("运行结束 | 退出码=%s | %s", code, "成功" if code == 0 else "失败")
    if retried:
        logger.info("  已自动重试一次（丢弃失效会话后重新认证）")
    logger.info("  详细过程: aux_crawler.log | 结构化结果: aux_report.json")
    logger.info("=" * 64)


def main(argv: list[str] | None = None) -> int:
    """入口：防御预检 → 调用实现 → 可恢复失败则丢弃会话重试一次 → 失败诊断。"""
    _ensure_logging()

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--doctor" in raw_args:
        return _doctor()

    # [AUX-V1-r12] 探索模式在入口层分走：登录复用 AUX 三层认证，之后只被动录制，
    # 不进采集实现、不碰数据库。放在 _preflight 之后，浏览器环境治理同样生效。
    if "--explore" in raw_args:
        if not _EXPLORE_AVAILABLE:
            logger.error("探索模式模块导入失败，无法进入 --explore：%s", _explore_exc)
            return 2
        _banner()
        logger.info("  探索模式 : %s（只录制，不采集、不入库）", _EXPLORE_BUILD)
        _preflight()
        return int(_explore_main(raw_args) or 0)

    _banner()
    _preflight()

    runs_before = len(_read_runs())
    try:
        code = _run_impl(argv)
    except BaseException as exc:  # noqa: BLE001 —— _run_impl 已消化 SystemExit
        if _RESILIENCE_ENABLED:
            try:
                d = diagnose_exception(exc)
                logger.error(
                    "resilience: 根因=%s 层=%s 可重试=%s 建议动作=%s | %s",
                    d.code, d.layer, d.retryable, _code_info(d.code).action.value, d.reason,
                )
            except Exception:  # noqa: BLE001
                pass
        raise

    # 仅当「本轮新增了一次可恢复失败」时才重试
    retried = False
    if code != 0 and _RESILIENCE_ENABLED and len(_read_runs()) > runs_before and _last_run_is_retryable_failure():
        logger.warning(
            "resilience: 本轮属可恢复失败（认证失效 / 浏览器启动失败）→ 丢弃会话后重新认证，重试一次"
        )
        retried = True
        code = _retry_after_discarding_session(argv)

    _conclusion(code, retried)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
