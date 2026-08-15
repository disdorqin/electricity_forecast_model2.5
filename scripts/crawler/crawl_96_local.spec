# -*- mode: python ; coding: utf-8 -*-
# crawl_96_local.exe 打包规格
# 入口: scripts/crawler/crawl_96_local.py
# 目标: 甲方电脑（无 Python 环境）爬预测+实际 96 点存本地 CSV
#
# 注意：必须用 venv_build（OpenSSL 3.0.13）打包！
#   用 epf-2 (OpenSSL 3.6.1) 打包会报 [ASN1: NOT_ENOUGH_DATA]，
#   国网 PMOS 的 TLS 证书与 OpenSSL 3.6 不兼容，与 run_full.exe 保持一致。
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('scripts.crawler')

# 第三方依赖（与 run_full.exe.spec 一致，确保 requests/ssl 依赖链完整）
hiddenimports += [
    'requests',
    'urllib3', 'urllib3.util', 'urllib3.packages', 'urllib3.packages.ssl_match_hostname',
    'websocket', 'websocket._core', 'websocket._exceptions',
    'certifi',
    'charset_normalizer', 'idna',
]


a = Analysis(
    ['scripts\\crawler\\crawl_96_local.py'],
    pathex=['.'],
    binaries=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'keras', 'jax', 'jaxlib',
        'scipy', 'sklearn', 'sklearn.*',
        'xgboost', 'catboost', 'lightgbm',
        'numba', 'llvmlite', 'matplotlib',
        'IPython', 'jedi', 'pytest', 'sphinx',
        'pyarrow', 'h5py', 'onnxruntime',
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
    name='crawl_96_local',
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
