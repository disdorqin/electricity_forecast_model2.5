param(
    [switch]$Clean,
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BuildPython = Join-Path $ProjectRoot "dist\build_artifacts\venv_build\Scripts\python.exe"
$PackageDir = Join-Path $ProjectRoot "dist\crawler\pmos_auto_auth"

if (-not (Test-Path $BuildPython)) {
    throw "未找到专用打包 Python：$BuildPython。必须使用 venv_build，不能使用 epf-2。"
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
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败，exit=$LASTEXITCODE" }

    if ($ConfigPath) {
        Copy-Item (Resolve-Path $ConfigPath) (Join-Path $PackageDir "config.json") -Force
    }
    else {
        Copy-Item "scripts\auto_crawler\config.template.json" (Join-Path $PackageDir "config.json") -Force
    }
    Copy-Item "scripts\auto_crawler\run_auth.cmd" (Join-Path $PackageDir "运行认证.cmd") -Force

    Push-Location $PackageDir
    try {
        & .\pmos_auto_auth.exe --ssl-check
        if ($LASTEXITCODE -ne 0) { throw "EXE TLS 自检失败，exit=$LASTEXITCODE" }
    }
    finally {
        Pop-Location
    }
    Write-Host "便携包构建完成：$PackageDir"
}
finally {
    Pop-Location
}
