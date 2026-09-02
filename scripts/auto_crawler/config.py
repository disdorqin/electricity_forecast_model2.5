from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class AuthConfig:
    """认证配置。默认读取同目录 config.json，环境变量可覆盖敏感字段。"""

    auth_host: str = "https://pmos.sd.sgcc.com.cn"
    trade_base: str = "https://pmos.sd.sgcc.com.cn:18080/trade"
    trade_entry: str = "DaJyjgfbPlantQuery.do?appkey=187"
    browser_path: str = ""
    browser_profile_dir: str = ""
    browser_profile_name: str = ""
    debug_port: int = 9222
    browser_ready_timeout_sec: int = 60
    login_timeout_sec: int = 600
    poll_interval_sec: float = 0.75
    login_retry_interval_sec: float = 2.0
    cfca_retry_interval_sec: float = 2.0
    login_check_interval_sec: float = 3.0
    transient_error_timeout_sec: int = 60
    username_env: str = "PMOS_USERNAME"
    password_env: str = "PMOS_PASSWORD"
    pin_env: str = "PMOS_UKEY_PIN"
    # 公司电脑部署时可直接在 config.json 填写；环境变量优先级更高。
    username: str = ""
    password: str = ""
    ukey_pin: str = ""
    slider_handler: str = "manual"
    pin_handler: str = "manual"
    slider_plugin: str = ""
    slider_max_attempts: int = 3
    slider_drag_duration_ms: int = 900
    slider_artifact_dir: str = "auth_debug/slider_samples"
    pin_plugin: str = ""
    ukey_window_title: str = "验证UKey用户口令"
    cfca_port: int = 7693
    success_probe_paths: tuple[str, ...] = (
        "/main/index.do",
        "/DaJyxxPlDa.do?appkey=112",
    )
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def login_url(self) -> str:
        service = self.trade_base.rstrip("/") + "/" + self.trade_entry.lstrip("/")
        return self.auth_host.rstrip("/") + "/?service=" + quote(service, safe="")

    @property
    def resolved_username(self) -> str:
        return os.environ.get(self.username_env, self.username).strip()

    @property
    def resolved_password(self) -> str:
        return os.environ.get(self.password_env, self.password)

    @property
    def resolved_pin(self) -> str:
        return os.environ.get(self.pin_env, self.ukey_pin).strip()

    @classmethod
    def from_file(cls, path: str | Path) -> "AuthConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("认证配置必须是 JSON 对象")
        known = set(cls.__dataclass_fields__) - {"extra"}
        values = {key: value for key, value in raw.items() if key in known}
        if "success_probe_paths" in values:
            values["success_probe_paths"] = tuple(values["success_probe_paths"])
        values["extra"] = {key: value for key, value in raw.items() if key not in known}
        return cls(**values)
