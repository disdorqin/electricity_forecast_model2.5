#!/usr/bin/env python
"""Environment dependency checker for electricity_forecast_model2.1."""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "pyarrow",
    "openpyxl",
    "sklearn",
    "torch",
    "lightgbm",
    "yaml",
    "matplotlib",
    "joblib",
    "dotenv",
]

OPTIONAL_PACKAGES = [
    "xgboost",
    "catboost",
    "jax",
    "huggingface_hub",
    "safetensors",
    "vmdpy",
    "chinese_calendar",
    "borax",
]


def check_module(name: str) -> bool:
    spec = importlib.util.find_spec(name)
    return spec is not None


def main() -> int:
    print("ENV_CHECK")
    errors = []

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    print(f"  python: {py_version} {'OK' if py_ok else 'FAIL (need >= 3.10)'}")
    if not py_ok:
        errors.append(f"Python {py_version} < 3.10")

    # Platform
    print(f"  platform: {platform.system()} {platform.release()}")
    print(f"  arch: {platform.machine()}")

    # CUDA
    if importlib.util.find_spec("torch"):
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_avail else 0
        cuda_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        print(f"  cuda_available: {'yes' if cuda_avail else 'no'} ({cuda_count} devices, {cuda_name})")
    else:
        print("  cuda_available: unknown (torch not found)")
        errors.append("torch not installed")

    # Dependencies
    print("  dependencies:")
    all_pkgs_ok = True
    for pkg in REQUIRED_PACKAGES:
        ok = check_module(pkg)
        if not ok:
            errors.append(f"Missing required package: {pkg}")
            all_pkgs_ok = False
        print(f"    {pkg}: {'OK' if ok else 'MISSING'}")
    for pkg in OPTIONAL_PACKAGES:
        ok = check_module(pkg)
        label = "optional" if ok else "optional (missing)"
        print(f"    {pkg}: {label}")

    # Local paths
    print("  local_paths:")
    project_root = Path(__file__).resolve().parent.parent

    # Bundled model directories
    print("  bundled_models:")
    bundled_dirs = [
        ("lightGBM", "lightGBM"),
        ("TimesFMBackend", "TimesFMBackend"),
        ("TimeMixer", "TimeMixer"),
        ("SGDFNet", "SGDFNet"),
        ("RT916_SpikeFusionNet", "RT916_SpikeFusionNet"),
    ]
    all_bundled_ok = True
    for label, dirname in bundled_dirs:
        ok = (project_root / dirname).is_dir()
        if not ok:
            errors.append(f"Bundled model directory missing: {dirname}/")
            all_bundled_ok = False
        print(f"    {label}: {'OK' if ok else 'MISSING'}")

    print(f"    external_epf_root_required: false")

    from utils.data_layout import DATA
    data_file = DATA.hourly_xlsx
    data_ok = data_file.is_file()
    print(f"    {data_file}: {'OK' if data_ok else 'MISSING'}")
    if not data_ok:
        errors.append(f"Input data file missing: {data_file}")

    outputs_dir = project_root / "outputs"
    outputs_writable = outputs_dir.is_dir() or (outputs_dir.parent.is_dir())
    print(f"    outputs/: {'writable' if outputs_writable else 'MISSING'}")

    # Status
    if errors:
        print(f"  status: FAIL")
        for e in errors:
            print(f"    - {e}")
        return 1
    else:
        print(f"  status: PASS")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
