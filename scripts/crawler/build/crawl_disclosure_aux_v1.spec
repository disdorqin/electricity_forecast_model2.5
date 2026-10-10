# -*- mode: python ; coding: utf-8 -*-
"""Canonical PyInstaller spec for PMOS AUX-V1.

Tracked source-of-truth for the current r14-capable application entry. Build
scratch/output remains under dist/build_artifacts/.
"""
import os

from PyInstaller.utils.hooks import collect_submodules

# <repo>/scripts/crawler/build/
_ROOT = os.path.abspath(os.path.join(SPECPATH, "..", "..", ".."))

a = Analysis(
    [os.path.join(_ROOT, "scripts", "crawler", "apps", "crawl_aux.py")],
    pathex=[_ROOT],
    binaries=[],
    datas=[],
    hiddenimports=[
        "pymysql",
        "websocket",
        "scripts.crawler.log_rotation",
        "scripts.crawler.collect.crawl_disclosure_aux_explore",
    ] + collect_submodules("scripts.crawler.resilience"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "torchaudio", "tensorflow", "keras",
        "jax", "jaxlib", "scipy", "sklearn", "sklearn.*", "xgboost",
        "catboost", "lightgbm", "numba", "llvmlite", "matplotlib",
        "IPython", "jedi", "pytest", "sphinx", "pyarrow", "h5py", "onnxruntime",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="crawl_disclosure_aux_v1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
