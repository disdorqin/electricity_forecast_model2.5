param(
    [switch]$Clean,
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BuildPython = Join-Path $ProjectRoot "dist\build_artifacts\venv_build\Scripts\python.exe"
$PackageDir = Join-Path $ProjectRoot "dist\crawler\pmos_auto_auth"

if (-not (Test-Path $BuildPython)) {
    throw "Build Python not found: $BuildPython. Run prepare_windows_build_env.ps1 first; do not use epf-2."
}

$BuildOpenSsl = & $BuildPython -c "import ssl; print(ssl.OPENSSL_VERSION)"
if ($BuildOpenSsl -notmatch "^OpenSSL 3\.0\.13\b") {
    throw "venv_build OpenSSL is incompatible: $BuildOpenSsl. PMOS packaging requires OpenSSL 3.0.13."
}
$BasePrefix = & $BuildPython -c "import sys; print(sys.base_prefix)"
$OpenSslBin = Join-Path $BasePrefix "Library\bin"
if (-not (Test-Path $OpenSslBin)) {
    throw "Conda OpenSSL DLL directory not found: $OpenSslBin"
}
$SslDll = Get-ChildItem -Path $OpenSslBin -Filter "libssl-3*.dll" | Select-Object -First 1
$CryptoDll = Get-ChildItem -Path $OpenSslBin -Filter "libcrypto-3*.dll" | Select-Object -First 1
if (-not $SslDll -or -not $CryptoDll) {
    throw "OpenSSL 3 DLLs not found in: $OpenSslBin"
}
Write-Host "Bundling OpenSSL DLLs from: $OpenSslBin"
$env:PATH = "$OpenSslBin;$env:PATH"

Push-Location $ProjectRoot
try {
    if ($Clean) {
        Remove-Item -Recurse -Force $PackageDir -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force "dist\build_artifacts\pyi_tmp\pmos_auto_auth" -ErrorAction SilentlyContinue
    }
    & $BuildPython -m PyInstaller --clean --noconfirm `
        --distpath "dist\crawler" `
        --workpath "dist\build_artifacts\pyi_tmp\pmos_auto_auth" `
        "scripts\auto_crawler\portable_onedir.spec"
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed, exit=$LASTEXITCODE" }

    if ($ConfigPath) {
        Copy-Item (Resolve-Path $ConfigPath) (Join-Path $PackageDir "config.json") -Force
    }
    else {
        Copy-Item "scripts\auto_crawler\config.template.json" (Join-Path $PackageDir "config.json") -Force
    }
    Copy-Item "scripts\auto_crawler\run_auth.cmd" (Join-Path $PackageDir "run_auth.cmd") -Force
    Set-Content -Path (Join-Path $PackageDir "build_marker.txt") -Value "pmos-auto-auth-2026-09-02-template-slider-ukey-pin" -Encoding ascii

    Push-Location $PackageDir
    try {
        & .\pmos_auto_auth.exe --ssl-version-check
        if ($LASTEXITCODE -ne 0) { throw "EXE TLS self-check failed, exit=$LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
    Write-Host "Portable package built: $PackageDir"
}
finally {
    Pop-Location
}
