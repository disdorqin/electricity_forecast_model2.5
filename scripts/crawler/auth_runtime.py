# -*- coding: utf-8 -*-
"""96 点爬虫运行时认证桥接。

把“已有 Cookie、账号密码自动登录、浏览器人工兜底”接到原有爬取链路前面，
不改动后续 PMOS 数据接口。认证过程只向调用方返回 Cookie；日志只记录阶段、
状态和脱敏摘要，不记录密码、Cookie 值或完整响应体。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def cookie_summary(cookie: str) -> str:
    """返回不含 Cookie 值的字段摘要。"""
    names: list[str] = []
    for item in str(cookie or "").split(";"):
        item = item.strip()
        if "=" in item:
            name = item.split("=", 1)[0].strip()
            if name and name not in names:
                names.append(name)
    digest = hashlib.sha256(str(cookie or "").encode("utf-8")).hexdigest()[:12]
    return f"names={','.join(names[:30]) or '-'} count={len(names)} len={len(str(cookie or ''))} sha256={digest}"


def atomic_update_cookie(config_path: str | Path, cookie: str) -> None:
    """原子更新 config.json 的 cookie，保留其它配置字段。"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data["cookie"] = cookie

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    logger.info("[auth] config.json 已原子写回 Cookie: path=%s %s", path, cookie_summary(cookie))


def _has_credentials(cfg: dict[str, Any]) -> bool:
    username = str(cfg.get("username") or "").strip()
    password = str(cfg.get("password") or "").strip()
    # 只拦截模板占位值，不按“password/username”子串判断，避免真实密码恰好包含这些字符。
    placeholder_values = {
        "你的pmos登录账号", "你的pmos登录密码", "your username", "your password",
        "username", "password", "账号", "密码",
    }
    return bool(username and password) and username.lower() not in placeholder_values and password.lower() not in placeholder_values


def _validate_cookie(cookie: str, cfg: dict[str, Any]) -> bool:
    if not cookie or len(cookie) < 20:
        return False
    try:
        from scripts.crawler.browser_session import validate_cookie

        ok = bool(validate_cookie(cookie, cfg, verbose=True))
        logger.info("[auth] existing_cookie_validate=%s %s", "PASS" if ok else "FAIL", cookie_summary(cookie))
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("[auth] existing_cookie_validate=ERROR type=%s msg=%s", type(exc).__name__, exc)
        return False


def ensure_authenticated_config(
    cfg: dict[str, Any],
    config_path: str | Path,
    *,
    base_dir: str | Path,
    mode: str | None = None,
    timeout_sec: int | None = None,
    max_retries: int | None = None,
) -> str:
    """获取可供原有 PmosCrawler 使用的 Cookie，并写回 config.json。

    mode:
      auto    现有 Cookie → 账号密码自动滑块 → 浏览器登录态兜底
      account 现有 Cookie → 账号密码自动滑块
      browser 现有 Cookie → 浏览器 DevTools 登录态
      static 只接受已有 Cookie，不主动认证
    """
    auth_mode = str(mode or cfg.get("auth_mode") or "auto").strip().lower()
    if auth_mode not in {"auto", "account", "browser", "static"}:
        raise ValueError(f"不支持的 auth_mode={auth_mode!r}，可选 auto/account/browser/static")
    timeout = int(timeout_sec or cfg.get("auth_timeout_sec") or cfg.get("browser_login_timeout_sec") or 300)
    retries = int(max_retries or cfg.get("auth_max_retries") or 3)
    cookie = str(cfg.get("cookie") or "").strip()

    logger.info(
        "[auth] begin mode=%s config=%s has_cookie=%s has_account=%s timeout=%ss retries=%s",
        auth_mode, config_path, bool(cookie), _has_credentials(cfg), timeout, retries,
    )

    if cookie and _validate_cookie(cookie, cfg):
        cfg["cookie"] = cookie
        logger.info("[auth] 使用现有有效 Cookie")
        return cookie

    if auth_mode == "static":
        if cookie:
            logger.warning("[auth] static 模式跳过有效性校验，沿用现有 Cookie")
            return cookie
        raise RuntimeError("static 模式需要 config.json 中存在 cookie")

    errors: list[str] = []
    if auth_mode in {"auto", "account"} and _has_credentials(cfg):
        logger.info("[auth] phase=account_login start")
        try:
            from scripts.crawler.auth import PmosAuth

            auth = PmosAuth(
                username=str(cfg["username"]).strip(),
                password=str(cfg["password"]),
                auth_host=str(cfg.get("auth_host") or "https://pmos.sd.sgcc.com.cn"),
                trade_base=str(cfg.get("base_url") or "https://pmos.sd.sgcc.com.cn:18080/trade"),
                timeout=min(max(timeout, 15), 120),
                verify_ssl=bool(cfg.get("verify_ssl", False)),
            )
            cookie = auth.login(max_captcha_retry=max(1, retries))
            if not cookie:
                raise RuntimeError("账号登录返回空 Cookie")
            atomic_update_cookie(config_path, cookie)
            cfg["cookie"] = cookie
            logger.info("[auth] phase=account_login PASS %s", cookie_summary(cookie))
            return cookie
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            errors.append(msg)
            logger.exception("[auth] phase=account_login FAIL: %s", msg)
            if auth_mode == "account":
                raise RuntimeError("账号密码自动登录失败；请查看 output_96/crawler.log") from exc
    elif auth_mode in {"auto", "account"}:
        logger.warning("[auth] phase=account_login SKIP：config.json 未提供可用账号密码")

    if auth_mode in {"auto", "browser"}:
        logger.info("[auth] phase=browser_login start：等待浏览器完成登录/滑块")
        try:
            from scripts.crawler.browser_session import ensure_browser_cookie

            cookie = ensure_browser_cookie(
                cfg,
                config_path,
                base_dir=base_dir,
                timeout_sec=timeout,
                headless=False,
            )
            if not cookie:
                raise RuntimeError("浏览器登录态返回空 Cookie")
            atomic_update_cookie(config_path, cookie)
            cfg["cookie"] = cookie
            logger.info("[auth] phase=browser_login PASS %s", cookie_summary(cookie))
            return cookie
        except Exception as exc:  # noqa: BLE001
            msg = f"{type(exc).__name__}: {exc}"
            errors.append(msg)
            logger.exception("[auth] phase=browser_login FAIL: %s", msg)

    detail = "；".join(errors[-3:]) if errors else "无可用认证方式"
    raise RuntimeError(f"认证失败：{detail}；请查看 output_96/crawler.log")
