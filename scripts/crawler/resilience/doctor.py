"""环境自检（doctor）：一条命令回答「现在这台机器能不能跑通」。

设计动机（用户 2026-09-27 需求）：
    "每一次看日志是否可以直接看出问题？程序里能否加入一些设置帮助你更好辅助判断？"
    → 与其让人在几千行日志里翻找，不如给一个**结构化自检报告**：
      每项检查给出 OK / WARN / FAIL / SKIP + 具体证据 + 建议动作。

覆盖项（全部为只读探测，不做任何修改）：
    1. 版本 / 构建标识
    2. 配置文件是否存在、关键字段是否就位
    3. 浏览器 profile：残留锁文件、是否有存活进程持有
    4. CDP 调试端口：可达性、是否挂着 PMOS 页面
    5. 输出目录可写性
    6. 数据库配置是否存在（不实际连库，连通性交给 --db-check）

[RESILIENCE-v1] 新增文件，不改动任何既有模块。
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import browser_env, health

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
_STATUS_ICON = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}


@dataclass
class CheckItem:
    name: str
    status: str
    detail: str = ""
    advice: str = ""

    def line(self) -> str:
        text = f"{_STATUS_ICON.get(self.status, '[????]')} {self.name}"
        if self.detail:
            text += f"：{self.detail}"
        return text

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "advice": self.advice}


@dataclass
class DoctorReport:
    title: str
    items: list[CheckItem] = field(default_factory=list)

    @property
    def worst(self) -> str:
        for level in (FAIL, WARN):
            if any(i.status == level for i in self.items):
                return level
        return OK

    @property
    def has_fail(self) -> bool:
        return any(i.status == FAIL for i in self.items)

    def render(self) -> str:
        width = 68
        out = ["=" * width, f" {self.title}", "=" * width]
        for item in self.items:
            out.append(" " + item.line())
            if item.advice:
                out.append(f"        └ 建议：{item.advice}")
        verdict = {
            OK: "环境检查通过，具备运行条件。",
            WARN: "环境基本可用，但有告警项，建议按上面提示处理。",
            FAIL: "存在阻塞性问题，请先修复再运行。",
        }[self.worst]
        out.append("-" * width)
        out.append(f" 结论：{verdict}")
        out.append("=" * width)
        return "\n".join(out)

    def as_dict(self) -> dict:
        return {"title": self.title, "verdict": self.worst,
                "items": [i.as_dict() for i in self.items]}


# ─────────────────────────────────────────────────────────────────────────

def _check_version(build_version: str) -> CheckItem:
    if not build_version:
        return CheckItem("构建版本", WARN, "未取到版本号", "检查实现模块的 BUILD_VERSION")
    return CheckItem("构建版本", OK, build_version)


def _check_config(config_path: Path | None) -> CheckItem:
    if config_path is None or not config_path.exists():
        return CheckItem("配置文件", FAIL, "未找到配置文件",
                         "复制 config.example.json 为 config.json 并填入账号/Cookie")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return CheckItem("配置文件", FAIL, f"解析失败：{type(exc).__name__}",
                         "检查 JSON 格式是否合法")
    # 只做「存在 + 可解析 + 关键开关可视化」，不校验业务字段：
    # 各爬虫的配置结构不同，硬校验字段极易误报（如用 Cookie 登录时本就没有 login_url）。
    reuse = data.get("browser_reuse", True)
    reuse_text = "复用优先" if reuse else "每次新开"
    detail = f"{config_path.name}（{len(data)} 项；browser_reuse={reuse} → {reuse_text}）"
    return CheckItem("配置文件", OK, detail)


def _check_profile(profile_dir: str) -> list[CheckItem]:
    if not profile_dir:
        return [CheckItem("浏览器 profile", SKIP, "未配置 browser_profile_dir",
                          "未配置时程序使用默认目录，通常无需处理")]

    root = Path(profile_dir)
    items: list[CheckItem] = []
    if not root.exists():
        items.append(CheckItem("浏览器 profile", OK,
                               f"{profile_dir}（尚未创建，首次启动会生成）"))
    else:
        locks = browser_env.find_lock_files(root)
        holders = browser_env.find_profile_holders(root)
        if holders:
            items.append(CheckItem(
                "profile 占用", WARN,
                f"有 {len(holders)} 个存活进程持有该 profile（pid={holders[:5]}）",
                "程序会优先换浏览器/等待，不会强杀；如需复用它请先手动关闭这些浏览器",
            ))
        elif locks:
            items.append(CheckItem(
                "profile 锁文件", WARN, f"发现无主锁文件 {locks}",
                "程序启动预检会自动清理（无存活进程持有时才清）",
            ))
        else:
            items.append(CheckItem("profile 占用", OK, "无残留锁、无存活进程持有"))
    return items


def _check_cdp(ports: list[int]) -> CheckItem:
    if not ports:
        return CheckItem("CDP 调试端口", SKIP, "未配置候选端口",
                         "未配置 debug_port 时程序会自行分配")
    probes = [health.probe_cdp(int(p)) for p in ports]
    reachable = [p for p in probes if p.reachable]
    healthy = [p for p in probes if p.healthy]
    if healthy:
        probe = healthy[0]
        return CheckItem("CDP 调试端口", OK,
                         f"端口 {probe.port} 可直接复用（{probe.browser or 'unknown'}，页面含 PMOS）")
    if reachable:
        detail = "；".join(
            f"{p.port} 可达但无 PMOS 页面（{p.error or '页面不符'}）" for p in reachable
        )
        return CheckItem("CDP 调试端口", WARN, detail,
                         "已存在的端口不可复用，程序将启动新浏览器")
    return CheckItem("CDP 调试端口", OK,
                     f"候选端口 {ports} 均未监听（正常：程序会启动新浏览器）")


def _check_output_dir(output_dir: Path) -> CheckItem:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=str(output_dir), prefix=".doctor_", delete=True):
            pass
        return CheckItem("输出目录", OK, f"{output_dir} 可写")
    except Exception as exc:  # noqa: BLE001
        return CheckItem("输出目录", FAIL, f"{output_dir} 不可写：{type(exc).__name__}",
                         "检查磁盘空间与目录权限")


def _check_db_config(db_config: Path | None) -> CheckItem:
    if db_config is None or not db_config.exists():
        return CheckItem("数据库配置", WARN, "未找到 db_config.json",
                         "需要写库时请补齐；仅本地抓取可忽略")
    return CheckItem("数据库配置", OK, db_config.name)


def run_doctor(
    *,
    app_name: str,
    build_version: str = "",
    config_path: Path | None = None,
    db_config_path: Path | None = None,
    profile_dir: str = "",
    port_candidates: list[int] | None = None,
    output_dir: Path | None = None,
    printer=print,
) -> int:
    """执行环境自检并打印报告。返回退出码（0=可用，1=有告警，2=有阻塞）。"""
    report = DoctorReport(title=f"{app_name} 环境自检（doctor）")
    report.items.append(_check_version(build_version))
    report.items.append(_check_config(config_path))
    report.items.extend(_check_profile(profile_dir))
    report.items.append(_check_cdp(list(port_candidates or [])))
    if output_dir is not None:
        report.items.append(_check_output_dir(output_dir))
    report.items.append(_check_db_config(db_config_path))

    printer(report.render())
    if report.has_fail:
        return 2
    return 1 if report.worst == WARN else 0
