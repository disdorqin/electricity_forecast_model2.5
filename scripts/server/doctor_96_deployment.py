from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RELEASE_CONTRACT = "formal96_predictor_release_v1"
REQUIRED_FILES = (
    "main.py",
    "requirements.txt",
    "models/LightGBM/best_model_日前电价.pkl",
    "models/timesFM/model.safetensors",
    "models/timesFM/config.json",
)
REQUIRED_DIRS = (
    "cli",
    "pipelines",
    "runtime",
    "runners",
    "fusion",
    "utils",
    "optim",
    "lightGBM",
    "TimesFMBackend",
    "TimeMixer",
    "RT916_SpikeFusionNet",
    "SGDFNet",
    "scripts/sync",
    "scripts/server",
)
STRICT_RELEASE_FORBIDDEN = (
    "outputs/experiments",
    "deliverables",
    "fixtures",
    "_archive",
    "dist",
    "build",
    "scripts/tests",
    "scripts/experiments",
    "scripts/crawler",
    "ExtremPriceClf",
    ".agents",
    ".opencode",
    ".env",
)
IMPORT_PROBES = (
    "cli.parser",
    "pipelines.ledger_full",
    "pipelines.ledger_predict",
    "runners.adapters.timesfm_v1",
    "lightGBM.pipeline",
    "TimeMixer.pipeline",
    "RT916_SpikeFusionNet.pipeline",
    "SGDFNet.pipeline",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _ok(name: str, detail: str) -> Check:
    return Check(name, "PASS", detail)


def _warn(name: str, detail: str) -> Check:
    return Check(name, "WARN", detail)


def _fail(name: str, detail: str) -> Check:
    return Check(name, "FAIL", detail)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_python() -> Check:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) != (3, 11):
        return _fail("python", f"Python {text}; production baseline requires Python 3.11.x")
    return _ok("python", f"Python {text}")


def check_layout(root: Path, *, strict_release: bool) -> list[Check]:
    checks: list[Check] = []
    missing_files = [rel for rel in REQUIRED_FILES if not (root / rel).is_file()]
    missing_dirs = [rel for rel in REQUIRED_DIRS if not (root / rel).is_dir()]
    if missing_files or missing_dirs:
        checks.append(
            _fail(
                "release_layout",
                f"missing_files={missing_files} missing_dirs={missing_dirs}",
            )
        )
    else:
        checks.append(_ok("release_layout", "all required formal96 application/model paths present"))

    if strict_release:
        present = [rel for rel in STRICT_RELEASE_FORBIDDEN if (root / rel).exists()]
        if present:
            checks.append(
                _fail(
                    "release_hygiene",
                    f"forbidden dev/research/runtime assets present: {present}",
                )
            )
        else:
            checks.append(_ok("release_hygiene", "no forbidden dev/research/runtime assets"))

        manifest_path = root / "release_manifest.json"
        if not manifest_path.exists():
            checks.append(_fail("release_manifest", "release_manifest.json missing"))
        else:
            checks.extend(check_release_manifest(root, manifest_path))
    return checks


def check_release_manifest(root: Path, manifest_path: Path) -> list[Check]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [_fail("release_manifest", f"cannot read manifest: {exc}")]

    if payload.get("contract") != RELEASE_CONTRACT:
        return [
            _fail(
                "release_manifest",
                f"contract={payload.get('contract')!r} expected={RELEASE_CONTRACT!r}",
            )
        ]

    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        return [_fail("release_manifest", "manifest files list missing/empty")]

    errors: list[str] = []
    checked = 0
    for item in entries:
        rel = item.get("path")
        expected_hash = item.get("sha256")
        expected_size = item.get("size_bytes")
        if not rel:
            errors.append("entry without path")
            continue
        path = root / rel
        if not path.is_file():
            errors.append(f"missing:{rel}")
            continue
        actual_size = int(path.stat().st_size)
        if expected_size is not None and actual_size != int(expected_size):
            errors.append(f"size:{rel}")
            continue
        if expected_hash:
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                errors.append(f"sha256:{rel}")
                continue
        checked += 1

    if errors:
        return [
            _fail(
                "release_manifest",
                f"verified={checked}/{len(entries)} errors={errors[:10]}",
            )
        ]
    return [_ok("release_manifest", f"verified {checked} files against manifest")]


def check_imports(root: Path) -> Check:
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    failures: list[str] = []
    origins: list[str] = []
    for name in IMPORT_PROBES:
        try:
            module = importlib.import_module(name)
            origin = getattr(module, "__file__", None)
            if origin:
                origin_path = Path(origin).resolve()
                try:
                    origin_path.relative_to(root.resolve())
                except ValueError:
                    failures.append(f"{name}: imported outside deployment root ({origin_path})")
                    continue
                origins.append(f"{name}:{origin_path.name}")
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    if failures:
        return _fail("model_imports", "; ".join(failures[:10]))
    return _ok("model_imports", f"{len(IMPORT_PROBES)} production modules import from deployment root")


def check_torch(*, require_cuda: bool) -> list[Check]:
    try:
        import torch
    except Exception as exc:
        return [_fail("torch", f"import failed: {exc}")]

    checks = []
    version = str(torch.__version__)
    if not version.startswith("2.6.0"):
        checks.append(_warn("torch", f"torch={version}; validated baseline is 2.6.0+cu124"))
    else:
        checks.append(_ok("torch", f"torch={version}"))

    cuda = bool(torch.cuda.is_available())
    if require_cuda and not cuda:
        checks.append(_fail("cuda", "CUDA unavailable; formal TimeMixer/RT916 production requires GPU"))
    elif cuda:
        device = torch.cuda.get_device_name(0)
        checks.append(_ok("cuda", f"available device={device}"))
    else:
        checks.append(_warn("cuda", "CUDA unavailable; imports/CPU checks only"))
    return checks


def check_model_assets(root: Path) -> Check:
    paths = [
        root / "models/LightGBM/best_model_日前电价.pkl",
        root / "models/timesFM/model.safetensors",
        root / "models/timesFM/config.json",
    ]
    bad = [str(path.relative_to(root)) for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if bad:
        return _fail("model_assets", f"missing/empty required assets: {bad}")
    total = sum(path.stat().st_size for path in paths)
    return _ok("model_assets", f"required static assets present ({total} bytes)")


def check_timesfm_model_resolution(root: Path, *, strict_release: bool) -> Check:
    try:
        backend = importlib.import_module(
            "TimesFMBackend.price_forecast_copy_分时段预测"
        )
        resolver = getattr(backend, "_resolve_timesfm_model_dir", None)
        if resolver is None:
            return _fail(
                "timesfm_model_resolution",
                "backend lacks deployment-stable _resolve_timesfm_model_dir",
            )
        resolved = Path(resolver()).expanduser().resolve()
    except Exception as exc:
        return _fail(
            "timesfm_model_resolution",
            f"{type(exc).__name__}: {exc}",
        )

    expected = (root / "models" / "timesFM").resolve()
    weights = resolved / "model.safetensors"
    if not weights.is_file():
        return _fail(
            "timesfm_model_resolution",
            f"resolved checkpoint missing model.safetensors: {resolved}",
        )
    if strict_release and resolved != expected:
        return _fail(
            "timesfm_model_resolution",
            f"strict release resolved outside bundled asset: {resolved} != {expected}",
        )
    return _ok(
        "timesfm_model_resolution",
        f"checkpoint={resolved}",
    )


def check_runtime_writable(root: Path) -> Check:
    runtime_root = root / "outputs" / "96" / "runtime"
    probe = runtime_root / f".deployment_doctor_{os.getpid()}"
    try:
        probe.mkdir(parents=True, exist_ok=False)
        marker = probe / "write_test.txt"
        marker.write_text("ok", encoding="utf-8")
        marker.unlink()
        probe.rmdir()
    except Exception as exc:
        try:
            if probe.exists():
                shutil.rmtree(probe, ignore_errors=True)
        except Exception:
            pass
        return _fail("runtime_writable", f"{runtime_root}: {exc}")
    return _ok("runtime_writable", f"{runtime_root} create/write/delete PASS")


def check_ledger_readiness(root: Path, target_date: str) -> Check:
    ledger_root = root / "outputs" / "96" / "ledger"
    if not ledger_root.exists():
        return _fail("ledger_readiness", f"missing production ledger root: {ledger_root}")

    try:
        from pipelines.ledger_full import _strict_history_readiness

        result = _strict_history_readiness(
            ledger_root,
            target_date,
            required_days=30,
            max_lookback_days=90,
        )
    except Exception as exc:
        return _fail("ledger_readiness", f"selector failed: {type(exc).__name__}: {exc}")

    counts = {
        task: item.get("selected_count", 0)
        for task, item in result.get("tasks", {}).items()
    }
    if result.get("status") != "PASS":
        return _fail("ledger_readiness", f"target={target_date} selected={counts}")
    return _ok("ledger_readiness", f"target={target_date} selected={counts}")


def check_final_delivery(root: Path, target_date: str) -> Check:
    runs_root = root / "outputs" / "96" / "runs"
    try:
        from pipelines.delivery_quality import validate_daily_submission
        from utils.resolution import QUARTER

        result = validate_daily_submission(
            runs_root,
            target_date,
            resolution=QUARTER,
        )
    except Exception as exc:
        return _fail("final_delivery", f"validator failed: {type(exc).__name__}: {exc}")

    if result.get("status") != "PASS":
        return _fail("final_delivery", f"target={target_date} errors={result.get('errors', [])[:5]}")
    return _ok("final_delivery", f"target={target_date} submission/postflight contract PASS")


def check_database() -> Check:
    try:
        from utils.database_operate import fetch_96_table_summary, get_db_server_version

        version = get_db_server_version()
        summary = fetch_96_table_summary("epf_pmos_96_full")
    except Exception as exc:
        return _fail("database", f"{type(exc).__name__}: {exc}")

    rows = int(summary.get("rows_total") or 0)
    if rows <= 0:
        return _fail("database", f"server={version}; epf_pmos_96_full is empty")
    return _ok(
        "database",
        f"server={version}; epf_pmos_96_full rows={rows} "
        f"range={summary.get('d_min')}..{summary.get('d_max')}",
    )


def run_doctor(
    root: Path,
    *,
    strict_release: bool = False,
    require_cuda: bool = False,
    check_db_flag: bool = False,
    check_writable_flag: bool = False,
    target_date: str | None = None,
    check_final_flag: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    checks: list[Check] = []
    checks.append(check_python())
    checks.extend(check_layout(root, strict_release=strict_release))
    checks.append(check_model_assets(root))
    checks.append(check_imports(root))
    checks.append(
        check_timesfm_model_resolution(
            root,
            strict_release=strict_release,
        )
    )
    checks.extend(check_torch(require_cuda=require_cuda))

    if check_db_flag:
        checks.append(check_database())
    if check_writable_flag:
        checks.append(check_runtime_writable(root))
    if target_date:
        checks.append(check_ledger_readiness(root, target_date))
        if check_final_flag:
            checks.append(check_final_delivery(root, target_date))

    failed = [item for item in checks if item.status == "FAIL"]
    warnings = [item for item in checks if item.status == "WARN"]
    return {
        "status": "PASS" if not failed else "FAIL",
        "root": str(root),
        "strict_release": strict_release,
        "checks": [item.as_dict() for item in checks],
        "failures": len(failed),
        "warnings": len(warnings),
    }


def _print_report(result: dict[str, Any]) -> None:
    print(f"FORMAL96 DEPLOYMENT DOCTOR: {result['status']}")
    for item in result["checks"]:
        print(f"[{item['status']}] {item['name']}: {item['detail']}")
    print(f"failures={result['failures']} warnings={result['warnings']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only/formal96-safe deployment preflight for predictor releases."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--strict-release",
        action="store_true",
        help="Require a materialized minimal release and reject dev/research trees.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is unavailable.",
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Connect read-only and verify epf_pmos_96_full is reachable.",
    )
    parser.add_argument(
        "--check-writable",
        action="store_true",
        help="Create and remove a tiny runtime probe under outputs/96/runtime.",
    )
    parser.add_argument(
        "--target-date",
        help="Also run formal lag=2 / 30-of-90 production ledger readiness for this target.",
    )
    parser.add_argument(
        "--check-final",
        action="store_true",
        help="With --target-date, validate the existing final submission/postflight.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the human report.",
    )
    args = parser.parse_args(argv)

    if args.check_final and not args.target_date:
        parser.error("--check-final requires --target-date")

    result = run_doctor(
        args.root,
        strict_release=args.strict_release,
        require_cuda=args.require_cuda,
        check_db_flag=args.check_db,
        check_writable_flag=args.check_writable,
        target_date=args.target_date,
        check_final_flag=args.check_final,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_report(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
