# -*- mode: python ; coding: utf-8 -*-
"""Windows 免安装目录包；配置文件保持 EXE 外置。"""
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("scripts.auto_crawler") + [
    "requests", "urllib3", "urllib3.util", "certifi", "charset_normalizer", "idna",
    "websocket", "websocket._core", "websocket._exceptions",
]

a = Analysis(
    ["frozen_entry.py"],
    pathex=["../.."],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "torchaudio", "tensorflow", "keras", "jax", "jaxlib",
        "scipy", "sklearn", "xgboost", "catboost", "lightgbm", "pandas",
        "matplotlib", "pytest", "IPython",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="pmos_auto_auth",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=False,
    name="pmos_auto_auth",
)
