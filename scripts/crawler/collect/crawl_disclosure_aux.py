#!/usr/bin/env python
"""AUX-V1 独立 PMOS 辅助信息披露爬虫入口。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Source-mode direct execution (`python scripts/...`) does not automatically
# put the repository root on sys.path; frozen builds keep their own import
# bootstrap. [AUX-V1]
_FROZEN = getattr(sys, "frozen", False)
REPO_ROOT = Path(sys.executable).resolve().parent if _FROZEN else Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.crawler.auth.auto_crawler.browser import (
    BrowserControlError,
    browser_executable_candidates,
)
from scripts.crawler.auth.auto_crawler.config import AuthConfig
from scripts.crawler.auth.auto_crawler.state_machine import (
    AuthenticationStateMachine,
    SessionExpiredError,
)
from scripts.crawler.observability import RunReport
from scripts.crawler.runtime_lock import RuntimeLock, RuntimeLockError
from scripts.crawler.collect.disclosure_aux import (
    BUILD_VERSION,
    AuxAuthRejected,
    AuxBrowserLost,
    PmosDisclosureAuxCrawler,
    SOURCE_REGISTRY,
    STATUS_COMPLETE,
    STATUS_EMPTY_VALID,
    STATUS_FAILED_SOURCE,
    STATUS_PARTIAL,
    STATUS_SKIPPED_NOT_READY,
)
from scripts.crawler.sync_db import disclosure_aux as aux_db
# [日志轮转 2026-09-28] 按天归档 + 只保留最近 7 天（顶层导入，PyInstaller 静态可收集）
from scripts.crawler.log_rotation import rotate_and_prune_log

logger = logging.getLogger("crawl_disclosure_aux")
AUX_DIR = REPO_ROOT if _FROZEN else REPO_ROOT / "dist" / "crawler" / "辅助信息披露爬虫"
OUTPUT_DIR = AUX_DIR / "output_aux"

# [AUX-V1-r11d] Navigation-race tolerance for the shared auth state machine.
# Shared module stays untouched (96 keeps using it as-is); the retry lives only
# in the AUX entry point.
_AUTH_MAX_ATTEMPTS = 3
_AUTH_RETRY_DELAY_SEC = 6.0

# [AUX-V1-r11e] 仅 AUX 注入：放行 CFCA UKey 插件对本机 localhost 的访问。
_AUTH_LNA_FLAG = "--disable-features=LocalNetworkAccessChecks"
# [AUX-V1-r11f] 可降级异常：均属「浏览器/会话层面失效」，换一层重试可能恢复。
# requests/urllib3 的 ConnectionError 继承 OSError，覆盖 CDP 端口连不上的场景。
_AUTH_RECOVERABLE_ERRORS = (
    BrowserControlError, SessionExpiredError, TimeoutError, RuntimeError, OSError,
)


@contextmanager
def _diagnostic_phase(name: str, *, reporter=None, interval_sec: float = 20.0):
    """[AUX-V1-r10-diag] Log bounded phase heartbeats without touching shared auth."""
    started = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(interval_sec):
            elapsed = round(time.monotonic() - started, 1)
            logger.warning("AUX phase still running phase=%s elapsed_sec=%.1f", name, elapsed)

    logger.info("AUX phase start phase=%s", name)
    if reporter is not None:
        reporter.stage(name, "RUNNING")
    worker = threading.Thread(target=heartbeat, name=f"aux-phase-{name}", daemon=True)
    worker.start()
    try:
        yield
    except BaseException as exc:
        logger.error("AUX phase failed phase=%s elapsed_sec=%.1f error_type=%s error=%s",
                     name, time.monotonic() - started, type(exc).__name__, str(exc)[:500])
        if reporter is not None:
            reporter.stage(name, "FAIL", error_type=type(exc).__name__, error=str(exc)[:500],
                           elapsed_sec=round(time.monotonic() - started, 1))
        raise
    else:
        logger.info("AUX phase complete phase=%s elapsed_sec=%.1f", name, time.monotonic() - started)
        if reporter is not None:
            reporter.stage(name, "PASS", elapsed_sec=round(time.monotonic() - started, 1))
    finally:
        stop.set()
        worker.join(timeout=1.0)


def configure_aux_logging(output_dir: Path) -> Path:
    """[AUX-V1-r1] Console + UTF-8 AUX-only log, never main crawler.log.

    [日志轮转 2026-09-28] 每次启动先按天归档 + 只保留最近 7 天：把「最后写入日期
    不是今天」的旧日志归档为 ``aux_crawler.log.<YYYY-MM-DD>``，删除旧于保留窗口的
    备份；``aux_crawler.log`` 因此始终只承载最近一次运行的日志。归档必须在
    FileHandler 打开文件**之前**执行（Windows 下重命名已打开文件会失败）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "aux_crawler.log"
    _removed = rotate_and_prune_log(log_path, keep_days=7)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(console)
    resolved = str(log_path.resolve()).lower()
    if not any(isinstance(h, logging.FileHandler) and str(Path(h.baseFilename).resolve()).lower() == resolved for h in root.handlers):
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
    if _removed:
        logger.info("日志轮转：已清理 %d 个过期日志备份 (%s)", len(_removed), ", ".join(p.name for p in _removed))
    return log_path


def _resolve(base: Path, value: str | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else (base / path).resolve()


def _default_aux_config(base_dir: Path) -> dict[str, Any]:
    """[AUX-V1-r2] Safe first-run defaults for a standalone EXE folder."""
    auth_name = "config.json" if (base_dir / "config.json").exists() else "../config.json"
    db_name = "db_config.json" if (base_dir / "db_config.json").exists() else "../db_config.json"
    lock_name = "output_96/.crawler.lock" if (base_dir / "output_96").exists() else "../output_96/.crawler.lock"
    return {
        "auth_config_path": auth_name,
        "db_config_path": db_name,
        "shared_lock_path": lock_name,
        "output_dir": "output_aux",
        "lookback_days": 1,
        # [AUX-V1-r2] Missing config must never silently enable DB writes.
        "db_upload": False,
    }


def load_aux_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        # [AUX-V1-r2] EXE-only transfer should fail safe but remain runnable:
        # materialize a usable no-upload config beside the EXE and explain it.
        value = _default_aux_config(path.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"AUX_CONFIG_CREATED path={path} db_upload=false; review config and credentials before DB upload")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("AUX config must be a JSON object")
    return value


def _dates(target: str | None, lookback: int) -> list[str]:
    end = date.fromisoformat(target) if target else date.today()
    return [(end - timedelta(days=i)).isoformat() for i in range(max(0, int(lookback)) + 1)]


def _monthly_sources_to_skip(source: str, target_date: str, seen_months: set[str]) -> set[str]:
    """[AUX-V1-r10] Schedule each monthly registry source once per run/month."""
    if source != "all-designed":
        return set()
    month_key = target_date[:7]
    if month_key in seen_months:
        return {spec.name for spec in SOURCE_REGISTRY.values() if spec.resolution == "monthly"}
    seen_months.add(month_key)
    return set()


def _aux_auth_config(base: AuthConfig) -> AuthConfig:
    """[AUX-V1-r11e/r11f] 仅 AUX 生效的开关。

    96 主爬虫不调用本函数，其 ``AuthConfig.extra`` 也不含这些键，因此共享状态机
    对 96 的行为保持零变化。
    """
    extra = dict(base.extra)
    extra.setdefault("extra_browser_args", [])
    if _AUTH_LNA_FLAG not in extra["extra_browser_args"]:
        extra["extra_browser_args"].append(_AUTH_LNA_FLAG)
    # [AUX-V1-r11f] 页面提示「已失效」时自动刷新（等价人工 F5）。
    extra["session_expired_refresh"] = True
    extra.setdefault("session_expired_refresh_max", 2)
    return replace(base, extra=extra)


def _auth(auth_path: Path, reporter: RunReport, *, force_new: bool = False):
    """[AUX-V1-r11f] 三层防护认证调度器（仅 AUX；共享模块与 96 行为不变）。

    认证是整条采集链路的核心前置——拿不到会话，后面 33 个源 × N 天全部归零。
    因此按「复用 → 重开 → 换浏览器」依次执行，**每层完整跑完才允许降级**，
    三层用尽才自主退出（不死循环、不静默继续）。

    * L1 复用层：优先复用已有浏览器进程。
      - Cookie 有效 → 直接返回会话；
      - 页面提示失效 → 状态机内部自动刷新恢复（等价人工 F5）；
      - 刷新仍无效 → 抛 SessionExpiredError → 降级。
    * L1' 重开层：复用不可行时启动新浏览器（profile 目录不变，绝不用临时
      profile——临时 profile 下 CFCA/UKey 原生弹窗不出现，会卡在 CERTIFICATE）。
    * L2 切换层：按 Chrome→Edge 候选顺序，每个浏览器独立走完整登录流程；
      打不开网站 / 登录超时 / 证书或凭据被拒 → 切换下一个浏览器。
    * L3 兜底：三层用尽 → 结构化诊断 + 抛出最后异常。
    """
    base_config = _aux_auth_config(AuthConfig.from_file(auth_path))
    if force_new:
        reporter.stage(
            "browser_cdp", "RETRY", mode="force_new",
            reason="existing_qctc_session_rejected",
        )
    # [AUX-V1-r11f] 候选**顺序完全交给程序原有逻辑**（browser_executable_candidates：
    # 显式 browser_path 优先，否则 Windows 固定 Chrome→Edge）。本调度器不自行排序，
    # 只按「可执行文件名」去重：实测同一浏览器会因 Program Files / (x86) 两个安装
    # 目录被列两次（chrome.exe、msedge.exe、msedge.exe），按路径去不掉，而重复尝试
    # 同一个浏览器只会白白多等一整轮登录超时。
    seen: set[str] = set()
    candidates: list[str] = []
    for item in browser_executable_candidates(base_config.browser_path):
        key = Path(str(item)).name.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(str(item))

    # profile 基准目录直接复用共享状态机的既有解析逻辑（相对路径、frozen 基目录、
    # home 兜底全一致），保证覆盖后的目录与程序默认行为同源。
    profile_dir = str(AuthenticationStateMachine(base_config)._profile_dir())
    # plan 元素：(层名, 说明, 浏览器可执行文件, profile 目录覆盖)
    plan: list[tuple[str, str, str | None, str | None]] = []
    if not force_new:
        plan.append(("L1-REUSE", "复用已有浏览器进程；页面失效时自动刷新恢复", None, None))
    for index, executable in enumerate(candidates, start=1):
        # 与共享状态机 _launch_initial_browser 的既有约定保持一致：首个候选沿用原
        # profile，后续候选改用 <profile>_<浏览器名>（如 pmos_auto_profile_msedge）。
        # 否则 Edge 会去打开 Chrome 的 profile，既可能被单实例锁挡住，也违背程序
        # 既有顺序。注意这些是**固定**目录，不是 *_fallback_<ts> 临时 profile。
        override_profile = None
        if index > 1 and profile_dir:
            override_profile = f"{profile_dir}_{Path(executable).stem.lower()}"
        plan.append((f"L2-BROWSER{index}", f"启动 {Path(executable).name} 走完整登录流程",
                     executable, override_profile))

    last_exc: Exception | None = None
    for layer, label, executable, override_profile in plan:
        # L1 复用；L2 强制开新浏览器（browser_reuse=False 会跳过现存 CDP 探测）。
        # 统一关闭状态机内部 fallback：它一旦介入就会新建 *_fallback_<ts> 临时
        # profile，而临时 profile 下 UKey 原生弹窗不出现，必然卡死（历史教训）。
        overrides: dict[str, object] = {
            "browser_reuse": executable is None,
            "browser_fallback": False,
        }
        if executable:
            overrides["browser_path"] = executable
        if override_profile:
            overrides["browser_profile_dir"] = override_profile
        config = replace(base_config, **overrides)
        logger.info(
            "AUX auth layer begin layer=%s reuse=%s browser=%s profile=%s mode=%s login_timeout_sec=%s",
            layer, executable is None, executable or "existing-cdp",
            override_profile or profile_dir or "-",
            "force_new" if force_new else "configured", config.login_timeout_sec,
        )
        reporter.stage("auth_layer", "TRY", layer=layer, detail=label,
                       browser=executable or "existing-cdp",
                       profile=override_profile or profile_dir or "-")
        try:
            with _diagnostic_phase("auth", reporter=reporter):
                result = AuthenticationStateMachine(config, reporter=reporter).run()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("AUX auth layer failed layer=%s error_type=%s error=%s",
                           layer, type(exc).__name__, str(exc)[:300])
            reporter.event(
                "WARN", "AUTH_LAYER_FAILED", f"{layer} 失败，降级到下一层",
                layer=layer, detail=label, error_type=type(exc).__name__,
                error=str(exc)[:300],
            )
            if not isinstance(exc, _AUTH_RECOVERABLE_ERRORS):
                # 非浏览器/会话层面的错误（如凭据配置缺失）继续重试只会空转。
                raise
            time.sleep(_AUTH_RETRY_DELAY_SEC)
            continue
        logger.info("AUX auth accepted layer=%s debug_port=%s cookie_present=%s elapsed_sec=%.1f",
                    layer, int(getattr(result, "debug_port", 0) or 0), bool(result.cookie),
                    float(getattr(result, "elapsed_sec", 0.0) or 0.0))
        reporter.stage("auth_layer", "PASS", layer=layer,
                       debug_port=int(getattr(result, "debug_port", 0) or 0),
                       cookie_present=bool(result.cookie))
        return result
    assert last_exc is not None
    logger.error("AUX auth exhausted all %d layers; giving up", len(plan))
    reporter.event(
        "ERROR", "AUTH_LAYERS_EXHAUSTED", "三层防护全部用尽，自主退出",
        layers=[entry[0] for entry in plan],
        error_type=type(last_exc).__name__, error=str(last_exc)[:300],
    )
    raise last_exc


def _new_crawler(auth_result, reporter: RunReport, output_dir: Path) -> PmosDisclosureAuxCrawler:
    return PmosDisclosureAuxCrawler(
        base_url="https://pmos.sd.sgcc.com.cn:18080/trade",
        cookie=str(auth_result.cookie or ""),
        browser_debug_port=int(getattr(auth_result, "debug_port", 0) or 0),
        data_api_mode="qctc", reporter=reporter, output_dir=output_dir,
    )


def _collect_with_auth_recovery(
    crawler: PmosDisclosureAuxCrawler,
    *, auth_path: Path, reporter: RunReport, output_dir: Path,
    source: str, business_date: str, unitid: str | None,
    skip_sources: set[str] | None = None,
) -> tuple[PmosDisclosureAuxCrawler, list]:
    """[AUX-V1-r4] On one real 401/403 (or a dead browser), re-auth and retry once.

    [AUX-V1-r11f] 触发条件从「仅 401/403」扩展到「浏览器进程死亡」：
    长跑中浏览器中途退出会让后续每个源都以 CDP 端口不可达失败，此前只记一条
    warning 继续扫，导致整轮静默白跑；现在同样进入三层防护重认证。
    """
    collect_kwargs = {"source": source, "business_date": business_date, "unitid": unitid}
    if skip_sources:
        collect_kwargs["skip_sources"] = skip_sources
    try:
        with _diagnostic_phase(f"collect:{source}:{business_date}", reporter=reporter):
            return crawler, crawler.collect(**collect_kwargs)
    except (AuxAuthRejected, AuxBrowserLost) as exc:
        browser_lost = isinstance(exc, AuxBrowserLost)
        reporter.event(
            "WARN", "AUX_BROWSER_RECOVERY_START",
            "承载采集的浏览器进程已死亡，启动三层防护重新认证" if browser_lost
            else "现有浏览器上下文被业务接口拒绝，自动启动独立浏览器重试一次",
            source=source, error_type=type(exc).__name__, reason=str(exc),
        )
        # 浏览器死亡时复用层仍可能有可用残留进程，故走完整三层防护；
        # 401/403 说明会话确实被服务端拒绝，直接重开新浏览器。
        auth_result = _auth(auth_path, reporter, force_new=not browser_lost)
        reporter.stage(
            "auth", "PASS", mode="force_new",
            debug_port=int(getattr(auth_result, "debug_port", 0) or 0),
            cookie_present=bool(auth_result.cookie),
        )
        recovered = _new_crawler(auth_result, reporter, output_dir)
        recovered.fetch_csrf_token()
        reporter.event(
            "INFO", "AUX_BROWSER_RECOVERY_READY",
            "独立浏览器认证完成，重试被拒绝的数据请求",
            debug_port=int(getattr(auth_result, "debug_port", 0) or 0),
        )
        try:
            with _diagnostic_phase(f"collect_retry:{source}:{business_date}", reporter=reporter):
                results = recovered.collect(**collect_kwargs)
        except (AuxAuthRejected, AuxBrowserLost) as retry_exc:
            reporter.event(
                "ERROR", "AUX_BROWSER_RECOVERY_FAILED",
                "三层防护重认证后仍失败，停止本轮",
                source=source, error_type=type(retry_exc).__name__, reason=str(retry_exc),
            )
            raise
        reporter.event(
            "INFO", "AUX_BROWSER_RECOVERY_PASS",
            "独立浏览器重试成功", source=source,
        )
        return recovered, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PMOS AUX-V1 辅助信息披露爬虫")
    parser.add_argument("--date", help="业务日期 YYYY-MM-DD")
    parser.add_argument("--lookback", type=int, default=None, help="向前采集天数")
    # [AUX-V1-r2] exact source names are explicit debug opt-ins; group/all keep
    # the enabled_by_default allowlist. [AUX-V1-r10] all-designed is an explicit
    # registry sweep and does not change the legacy meaning of `all`.
    source_choices = ("unit", "constraint", "event", "curve", "stat", "contract", "all", "all-designed", *sorted(SOURCE_REGISTRY))
    parser.add_argument(
        "--source", choices=source_choices, default="all",
        help="all=仅默认启用来源；all-designed=一次遍历全部已登记来源，未验证/缺依赖/参数不全的来源会报告跳过",
    )
    parser.add_argument("--unitid", default=None, help="显式依赖型 source 的真实 unitid；不会自动 fan-out")
    parser.add_argument("--capture-only", action="store_true", help="只保存 raw，不写结构化表")
    parser.add_argument("--dry-run", action="store_true", help="认证并请求但不上传数据库")
    parser.add_argument("--auth-only", action="store_true", help="仅运行共享认证，不采集数据")
    parser.add_argument("--db-check", action="store_true", help="检查 AUX DDL；提供 DB 配置时同时初始化表")
    parser.add_argument("--no-db-upload", action="store_true", help="禁用数据库写入")
    parser.add_argument("--config", default=None, help="AUX 配置 JSON，默认同目录 config_disclosure_aux.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = _resolve(AUX_DIR, args.config, "config_disclosure_aux.json")
    logger.info("AUX startup build=%s frozen=%s executable=%s config_path=%s", BUILD_VERSION,
                bool(getattr(sys, "frozen", False)), sys.executable, config_path)
    cfg = load_aux_config(config_path)
    output_dir = _resolve(AUX_DIR, str(cfg.get("output_dir") or "output_aux"), "output_aux")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_aux_logging(output_dir)
    report = RunReport(output_dir / "aux_report.json", build_version=BUILD_VERSION, args=vars(args))
    db_cfg_path = _resolve(AUX_DIR, str(cfg.get("db_config_path") or "../db_config.json"), "../db_config.json")
    db_enabled = bool(cfg.get("db_upload", True)) and not args.dry_run and not args.capture_only and not args.no_db_upload
    logger.info("AUX-V1 start build=%s run_id=%s source=%s date=%s lookback=%s config_path=%s output_dir=%s db_enabled=%s db_config_exists=%s",
                BUILD_VERSION, report.run_id, args.source, args.date or "today", args.lookback,
                config_path, output_dir, db_enabled, db_cfg_path.exists())
    lock_path = _resolve(AUX_DIR, str(cfg.get("shared_lock_path") or "../output_96/.crawler.lock"), "../output_96/.crawler.lock")
    lock = RuntimeLock(lock_path)
    try:
        logger.info("AUX lock acquire begin path=%s", lock_path)
        with lock:
            logger.info("AUX lock acquired path=%s", lock_path)
            if args.db_check:
                with _diagnostic_phase("db_ddl_validate", reporter=report):
                    ok, errors = aux_db.validate_ddl()
                report.stage("db_schema", "PASS" if ok else "FAIL", errors=errors, tables=sorted(aux_db.ALLOWED_TABLES))
                if not ok:
                    report.finish("FAIL", reason="DDL_CONTRACT")
                    return 1
                db_cfg_path = _resolve(AUX_DIR, str(cfg.get("db_config_path") or "../db_config.json"), "../db_config.json")
                if db_cfg_path.exists() and cfg.get("db_upload", True) and not args.no_db_upload:
                    with _diagnostic_phase("db_schema_init", reporter=report):
                        aux_db.init_aux_tables(json.loads(db_cfg_path.read_text(encoding="utf-8")))
                report.finish("PASS", ddl_only=True)
                return 0

            auth_path = _resolve(AUX_DIR, str(cfg.get("auth_config_path") or "../config.json"), "../config.json")
            result = _auth(auth_path, report)
            report.stage("auth", "PASS", debug_port=int(getattr(result, "debug_port", 0) or 0), cookie_present=bool(result.cookie))
            if args.auth_only:
                report.finish("PASS", auth_only=True)
                return 0

            crawler = _new_crawler(result, report, output_dir)
            with _diagnostic_phase("qctc_context", reporter=report):
                crawler.fetch_csrf_token()  # QCTC context is a soft gate; real source decides.
            db = None
            if db_enabled:
                if not db_cfg_path.exists():
                    raise FileNotFoundError(f"DB config not found: {db_cfg_path}; use --no-db-upload for capture-only")
                db_cfg = json.loads(db_cfg_path.read_text(encoding="utf-8"))
                with _diagnostic_phase("db_schema_init", reporter=report):
                    aux_db.init_aux_tables(db_cfg)
                with _diagnostic_phase("db_connect", reporter=report):
                    db = aux_db.get_db(db_cfg)
            try:
                all_results = []
                lookback = args.lookback if args.lookback is not None else int(cfg.get("lookback_days", 1))
                seen_months: set[str] = set()
                for target_date in _dates(args.date, lookback):
                    logger.info("AUX date begin date=%s source=%s", target_date, args.source)
                    skip_sources = _monthly_sources_to_skip(args.source, target_date, seen_months)
                    crawler, results = _collect_with_auth_recovery(
                        crawler, auth_path=auth_path, reporter=report, output_dir=output_dir,
                        source=args.source, business_date=target_date, unitid=args.unitid,
                        skip_sources=skip_sources,
                    )
                    all_results.extend(results)
                    date_sources = []
                    for source_result in results:
                        logger.info("AUX source result date=%s source=%s status=%s http_status=%s rows=%s error=%s raw_path=%s",
                                    target_date, source_result.name, source_result.status,
                                    source_result.http_status, len(source_result.rows),
                                    str(source_result.error or "")[:400],
                                    source_result.raw.get("raw_path") if isinstance(source_result.raw, dict) else None)
                        report.stage(f"source:{source_result.name}", source_result.status, rows=len(source_result.rows), error=source_result.error)
                        date_sources.append({
                            "source": source_result.name,
                            "status": source_result.status,
                            "rows": len(source_result.rows),
                            "error": source_result.error,
                        })
                        # A not-ready result is a scheduling audit, not a
                        # response; never store it as a raw platform payload.
                        if db is not None and source_result.status != STATUS_SKIPPED_NOT_READY:
                            with _diagnostic_phase(f"db_upsert:{source_result.name}:{target_date}", reporter=report):
                                written = aux_db.upsert_source_result(db, source_result)
                            logger.info("AUX DB upsert complete date=%s source=%s rows_written=%s", target_date, source_result.name, written)
                    if args.source == "all-designed":
                        date_status = "PARTIAL" if any(
                            item["status"] in {STATUS_FAILED_SOURCE, STATUS_PARTIAL, STATUS_SKIPPED_NOT_READY}
                            for item in date_sources
                        ) else "PASS"
                        report.date(
                            target_date,
                            date_status,
                            sources=date_sources,
                            monthly_sources_scheduled=sorted(
                                spec.name for spec in SOURCE_REGISTRY.values()
                                if spec.resolution == "monthly" and spec.name not in skip_sources
                            ),
                        )
                    logger.info("AUX date complete date=%s source_results=%s", target_date, len(date_sources))
                statuses = [r.status for r in all_results]
                failed = sum(status == STATUS_FAILED_SOURCE for status in statuses)
                partial = sum(status == STATUS_PARTIAL for status in statuses)
                skipped = sum(status == STATUS_SKIPPED_NOT_READY for status in statuses)
                report.finish("PARTIAL" if failed or partial or skipped else "PASS", sources=len(all_results), failed=failed, partial=partial, skipped_not_ready=skipped,
                              complete=sum(status == STATUS_COMPLETE for status in statuses), empty_valid=sum(status == STATUS_EMPTY_VALID for status in statuses))
                return 0 if not failed and not (args.source == "all-designed" and (partial or skipped)) else 2
            finally:
                if db is not None:
                    db.close()
    except KeyboardInterrupt:
        logger.warning("AUX interrupted by user run_id=%s; inspect last phase/source log before retry", report.run_id)
        report.finish("INTERRUPTED", reason="KeyboardInterrupt")
        return 130
    except RuntimeLockError as exc:
        logger.error("已有主/辅助爬虫运行，AUX 不启动浏览器或数据库: %s", exc)
        report.finish("FAIL", reason="CRAWLER_ALREADY_RUNNING")
        return 2
    except AuxAuthRejected as exc:
        logger.error("认证被拒绝，终止 AUX: %s", exc)
        report.finish("FAIL", reason="AUTH_REJECTED")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("AUX-V1 failed")
        report.exception("fatal", exc)
        report.finish("FAIL", reason=str(exc))
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
