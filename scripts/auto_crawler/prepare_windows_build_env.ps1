param(
    [string]$PythonExe = "python",
    [string]$PipIndexUrl = "https://pypi.org/simple"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VenvDir = Join-Path $ProjectRoot "dist\build_artifacts\venv_build"

if ($PythonExe -eq "python") {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$Version = & $PythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
$OpenSsl = & $PythonExe -c "import ssl; print(ssl.OPENSSL_VERSION)"
Write-Host "Build Python: $PythonExe"
Write-Host "Python: $Version"
Write-Host "OpenSSL: $OpenSsl"

if ($OpenSsl -notmatch "^OpenSSL 3\.0\.13\b") {
    throw (
        "This Python does not use the validated OpenSSL 3.0.13." + [Environment]::NewLine +
        "Provide a Python with OpenSSL 3.0.13 and run:" + [Environment]::NewLine +
        ".\prepare_windows_build_env.ps1 -PythonExe C:\path\to\python.exe" + [Environment]::NewLine +
        "Do not package the PMOS tool with OpenSSL 3.6.x."
    )
}

if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv_build, exit=$LASTEXITCODE" }
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip --version
if ($LASTEXITCODE -ne 0) { throw "pip self-check failed, exit=$LASTEXITCODE" }
& $VenvPython -m pip install `
    --isolated `
    --index-url $PipIndexUrl `
    --timeout 120 `
    --retries 10 `
    --prefer-binary `
    --no-cache-dir `
    "pyinstaller>=6,<7" "requests==2.32.5" "websocket-client>=1.8,<2"
if ($LASTEXITCODE -ne 0) { throw "Failed to install build dependencies, exit=$LASTEXITCODE" }

$VenvOpenSsl = & $VenvPython -c "import ssl; print(ssl.OPENSSL_VERSION)"
if ($VenvOpenSsl -notmatch "^OpenSSL 3\.0\.13\b") {
    throw "venv_build OpenSSL verification failed: $VenvOpenSsl"
}
& $VenvPython -m PyInstaller --version
if ($LASTEXITCODE -ne 0) { throw "PyInstaller self-check failed, exit=$LASTEXITCODE" }
Write-Host "Build environment ready: $VenvDir"
