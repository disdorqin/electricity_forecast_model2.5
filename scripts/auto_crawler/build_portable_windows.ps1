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

    Push-Location $PackageDir
    try {
        & .\pmos_auto_auth.exe --ssl-check
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
