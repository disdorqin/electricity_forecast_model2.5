from __future__ import annotations

import argparse
import hashlib
import logging
import socket
import ssl
import sys
from pathlib import Path
from urllib.parse import urlparse

# 允许在公司电脑直接进入目录后执行：python main.py
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from scripts.auto_crawler.config import AuthConfig
from scripts.auto_crawler.state_machine import AuthenticationStateMachine


# 每次改变认证状态机行为时更新。它会写入运行日志，用于确认公司电脑没有在运行旧 EXE。
BUILD_MARKER = "pmos-auto-auth-2026-09-02-template-slider-ukey-pin-portal-502-recovery"


def default_config_path() -> Path:
    """源码目录或便携包 EXE 同目录中的唯一默认配置。"""
    if getattr(sys, "frozen", False):
        path = Path(sys.executable).resolve().with_name("config.json")
    else:
        path = Path(__file__).with_name("config.json")
    if not path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{path}；请在程序同目录创建 config.json")
    return path


EXPECTED_OPENSSL_PREFIX = "OpenSSL 3.0.13"


def ssl_check(auth_host: str, *, probe_network: bool = True) -> int:
    logging.info("ssl.openssl=%s", ssl.OPENSSL_VERSION)
    if not ssl.OPENSSL_VERSION.startswith(EXPECTED_OPENSSL_PREFIX):
        logging.error("ssl.version_mismatch expected=%s", EXPECTED_OPENSSL_PREFIX)
        return 3
    if not probe_network:
        logging.info("ssl.version_check=PASS")
        return 0
    host = urlparse(auth_host).hostname
    if not host:
        logging.error("ssl.probe invalid_host=%s", auth_host)
        return 2
    try:
        context = ssl._create_unverified_context()
        with socket.create_connection((host, 443), timeout=15) as tcp:
            with context.wrap_socket(tcp, server_hostname=host) as tls:
                logging.info("ssl.probe handshake=PASS protocol=%s cipher=%s", tls.version(), tls.cipher()[0])
        return 0
    except OSError as exc:
        logging.error("ssl.probe failed=%s: %s", type(exc).__name__, exc)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="PMOS 浏览器认证状态机")
    parser.add_argument("--config", default=None, help="可选：指定配置文件；默认读取同目录 config.json")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--ssl-check", action="store_true", help="打印 OpenSSL 版本并验证 PMOS TLS 连通性")
    parser.add_argument("--ssl-version-check", action="store_true", help="仅验证内置 OpenSSL 版本，不访问网络")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    config_path = Path(args.config) if args.config else default_config_path()
    logging.info("authentication.build marker=%s frozen=%s", BUILD_MARKER, bool(getattr(sys, "frozen", False)))
    config = AuthConfig.from_file(config_path)
    logging.info("authentication.config path=%s", config_path)
    if args.ssl_check or args.ssl_version_check:
        return ssl_check(config.auth_host, probe_network=args.ssl_check)
    result = AuthenticationStateMachine(config).run()
    digest = hashlib.sha256(result.cookie.encode("utf-8")).hexdigest()[:12]
    logging.info(
        "authentication.complete browser=%s elapsed=%.1fs cookie_len=%s cookie_sha256=%s",
        result.browser_path,
        result.elapsed_sec,
        len(result.cookie),
        digest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
